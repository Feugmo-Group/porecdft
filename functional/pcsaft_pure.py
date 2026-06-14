"""Pure-component PC-SAFT cDFT functional (dispersion + hard-chain).

Ports the ``DispersionPure`` + ``HardChainPure`` contributions of the
reference Stierle & Gross 2024 ``dft_ad_jax`` package
(Chem. Eng. Sci. 298, 120380, 2024) into porecdft.

The total excess Helmholtz energy is

    F^exc = F^HS  +  F^chain  +  F^disp

where F^HS is the FMT-aWBII hard-sphere contribution (already supplied by
:mod:`porecdft.functional.fmt`), and F^chain + F^disp are added here for
chain molecules with m > 1.

This module exposes a single class :class:`PurePCSAFTFunctional` that
returns the c¹ functional derivative of the chain + dispersion piece on a
3-D real-space grid, computed via ``jax.grad`` of the Helmholtz density
integral.  The FMT-aWBII c¹ should be added to it.

Hard-sphere diameter d(T) = σ · (1 − 0.12 · exp(−3 ε/k / T))  (Gross 2001).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import spherical_jn

# Gross & Sadowski 2001 dispersion-integral coefficients
_A0 = jnp.array([0.91056314451539, 0.63612814494991, 2.68613478913903,
                 -26.5473624914884, 97.7592087835073, -159.591540865600,
                 91.2977740839123])
_A1 = jnp.array([-0.30840169182720, 0.18605311591713, -2.50300472586548,
                 21.4197936296668, -65.2558853303492, 83.3186804808856,
                 -33.7469229297323])
_A2 = jnp.array([-0.09061483509767, 0.45278428063920, 0.59627007280101,
                 -1.72418291311787, -4.13021125311661, 13.7766318697211,
                 -8.67284703679646])
_B0 = jnp.array([0.72409469413165, 2.23827918609380, -4.00258494846342,
                 -21.00357681484648, 26.85564136266150, 206.55133840661881,
                 -355.60235612207947])
_B1 = jnp.array([-0.57554980753450, 0.69950955214436, 3.89256733895307,
                 -17.21547164777212, 192.67226446524950, -161.82646164876479,
                 -165.20769345556070])
_B2 = jnp.array([0.09768831158356, -0.25575749816100, -9.15585615297321,
                 20.64207597439724, -38.80443005206285, 93.62677407701460,
                 -29.66690558514725])
_PSI = 1.3862


def hsd_pcsaft(sigma: float, eps_k: float, T: float) -> float:
    """PC-SAFT Barker–Henderson hard-sphere diameter (Å)."""
    return float(sigma * (1.0 - 0.12 * np.exp(-3.0 * eps_k / T)))


def _spherical_jn_np(n: int, x):
    """spherical_jn returning a jnp array (computed once on host CPU)."""
    return jnp.asarray(spherical_jn(n, np.asarray(x)))


@dataclass
class PurePCSAFTFunctional:
    """Pure-component PC-SAFT dispersion + hard-chain c¹ on a 3-D grid.

    Parameters
    ----------
    m : float
        Chain length (segments per molecule). For ``m == 1`` only the
        dispersion contribution is non-zero (no chain term).
    sigma : float
        Segment diameter σ in Å.
    eps_k : float
        Dispersion energy ε/k_B in K.
    T : float
        Temperature in K (fixed at construction; rebuild for a new T).

    Notes
    -----
    The c¹ returned by :meth:`c1` is the dispersion + chain part **only**.
    Add the FMT-aWBII c¹ (from :func:`porecdft.functional.fmt.compute_c1`)
    to it before passing to the solver.
    """
    m: float
    sigma: float
    eps_k: float
    T: float

    def _weight_disp_hat(self, K) -> jnp.ndarray:
        """Dispersion weight ω_disp(k) = m · (j0 + j2)(2 ψ R k) — scalar."""
        R = 0.5 * hsd_pcsaft(self.sigma, self.eps_k, self.T)
        arg = 2.0 * _PSI * R * np.asarray(K)
        j0 = spherical_jn(0, arg)
        j2 = spherical_jn(2, arg)
        return jnp.asarray(self.m * (j0 + j2))

    def _weight_chain_hat(self, K) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Hard-chain (pure) weights: ω_λ(k) = j0(2 R k);
        ω_ζ3(k) = (π/6) m (2R)³ (j0 + j2)(2 R k)."""
        R = 0.5 * hsd_pcsaft(self.sigma, self.eps_k, self.T)
        arg = 2.0 * R * np.asarray(K)
        j0 = spherical_jn(0, arg)
        j2 = spherical_jn(2, arg)
        omega_lambd = jnp.asarray(j0)
        omega_zeta3 = jnp.asarray((np.pi / 6.0) * self.m
                                  * (2.0 * R) ** 3 * (j0 + j2))
        return omega_lambd, omega_zeta3

    def _phi_disp(self, n_disp: jnp.ndarray) -> jnp.ndarray:
        """Dispersion Helmholtz density φ_disp(r)/k_BT  given n_disp = ω*ρ."""
        R = 0.5 * hsd_pcsaft(self.sigma, self.eps_k, self.T)
        n = jnp.clip(n_disp, 0.0, None)
        eta = (4.0 / 3.0) * jnp.pi * n * R ** 3
        m = self.m
        m1 = (m - 1.0) / m
        m2 = (m - 2.0) / m
        i1 = jnp.zeros_like(eta)
        c1i2 = jnp.zeros_like(eta)
        for i in range(7):
            i1 = i1 + (_A0[i] + m1 * _A1[i] + m1 * m2 * _A2[i]) * eta ** i
            c1i2 = c1i2 + (_B0[i] + m1 * _B1[i] + m1 * m2 * _B2[i]) * eta ** i
        C1_denom = (1.0
                    + m * (8.0 * eta - 2.0 * eta ** 2) / (1.0 - eta) ** 4
                    + (1.0 - m) * (20.0 * eta - 27.0 * eta ** 2
                                    + 12.0 * eta ** 3 - 2.0 * eta ** 4)
                    / ((1.0 - eta) * (2.0 - eta)) ** 2)
        c1i2 = c1i2 / C1_denom
        eps_T = self.eps_k / self.T
        return ((-2.0 * jnp.pi * i1
                 - jnp.pi * m * c1i2 * eps_T)
                * (n ** 2 * eps_T * self.sigma ** 3))

    def _phi_chain(self, n_lambd: jnp.ndarray, n_zeta3: jnp.ndarray,
                   rho: jnp.ndarray) -> jnp.ndarray:
        """Hard-chain Helmholtz density φ_chain(r)/k_BT (pure).

        Both ``n_lambd`` and ``rho`` are clipped to a floor of 1e-8 /Å³
        so the ``log(y_dd · n_λ)`` term cannot drive a runaway attractive
        c¹ in repulsive cores where the weighted density vanishes.
        Bulk fluid densities of interest live at 1e-5 ... 1e-2 /Å³ — the
        1e-8 floor is well below any physical bulk and high enough to
        keep |log| ≲ 18 (vs ~70 with the previous 1e-30 floor).
        """
        floor = 1e-8
        n_l = jnp.clip(n_lambd, floor, None)
        rho_safe = jnp.clip(rho, floor, None)
        z3 = jnp.clip(n_zeta3, 0.0, 0.95)
        inv = 1.0 / (1.0 - z3)
        y_dd = inv + 0.5 * z3 * inv ** 2 * (3.0 + z3 * inv)
        return -(self.m - 1.0) * rho_safe * (jnp.log(y_dd * n_l) - 1.0)

    def helmholtz_density(self, rho: jnp.ndarray, w_disp_hat,
                          w_lambd_hat, w_zeta3_hat) -> jnp.ndarray:
        """Total (chain + dispersion) Helmholtz density on the real grid."""
        rho_hat = jnp.fft.rfftn(rho)
        n_disp  = jnp.fft.irfftn(rho_hat * w_disp_hat,  s=rho.shape)
        phi = self._phi_disp(n_disp)
        if abs(self.m - 1.0) > 1e-12:
            n_lambd = jnp.fft.irfftn(rho_hat * w_lambd_hat, s=rho.shape)
            n_zeta3 = jnp.fft.irfftn(rho_hat * w_zeta3_hat, s=rho.shape)
            phi = phi + self._phi_chain(n_lambd, n_zeta3, rho)
        return phi

    def c1(self, rho: jnp.ndarray, dV: float, w_disp_hat,
           w_lambd_hat, w_zeta3_hat) -> jnp.ndarray:
        """c¹(r) = −β δF/δρ via jax.grad of the integrated Helmholtz."""
        def F_of_rho(r):
            phi = self.helmholtz_density(r, w_disp_hat,
                                          w_lambd_hat, w_zeta3_hat)
            return jnp.sum(phi) * dV
        # δF/δρ ≈ dF/dρ_i divided by dV (discrete functional derivative)
        grad = jax.grad(F_of_rho)(rho)
        return -grad / dV

    def bulk_c1(self, rho_b: float) -> float:
        """Bulk reference c¹ at uniform density ρ_b.

        Uses **the same jax.grad path** as :meth:`c1`, on a tiny uniform grid,
        so the bulk reference is consistent with the spatial c¹ by
        construction.  This is essential — any difference between the two
        routes appears as a fictitious force at the pore--bulk boundary in
        the Euler--Lagrange iteration and drives runaway condensation.
        """
        # Small periodic box; FFT modes other than k=0 vanish for uniform ρ.
        n = 4
        d = 1.0  # arbitrary lattice spacing — drops out for uniform ρ.
        rho = jnp.full((n, n, n), float(rho_b), dtype=jnp.float32)
        # Build matching k-grid + weights on the rfft layout (n//2+1 last axis).
        from porecdft.functional.fmt import make_k_grid
        _, _, _, K_rfft = make_k_grid((n, n, n), d, d, d, real_fft=True)
        w_disp = self._weight_disp_hat(K_rfft)
        w_lambd, w_zeta3 = self._weight_chain_hat(K_rfft)
        c1_field = self.c1(rho, d ** 3, w_disp, w_lambd, w_zeta3)
        return float(c1_field.mean())


# ═══════════════════════════════════════════════════════════════════════════
# Multi-component PC-SAFT cDFT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MultiPCSAFTFunctional:
    """Multi-component PC-SAFT (hard-chain + dispersion) cDFT functional.

    Implements Stierle 2024 Appendix A in full:

    * Hard-chain Eq. (16) with cavity correlation y^dd(n_ζ2, n_ζ3) of
      Eq. (17) (multi-component form — depends on **both** ζ2 and ζ3
      with explicit R_i factors, unlike the :class:`PurePCSAFTFunctional`
      simplification).
    * Dispersion Eq. (A.10 of Gross & Sadowski 2001) with the standard
      Lorentz--Berthelot binary combining rules
      σ_ij = (σ_i+σ_j)/2 and  ε_ij = √(ε_i ε_j)(1 − k_ij).

    Parameters are arrays of length ``C`` (the number of components).
    Density inputs/outputs are shape ``(C, Nx, Ny, Nz)``.

    The c¹ on the grid is obtained via ``jax.grad`` on the integrated
    Helmholtz density; the bulk reference uses the same path on a tiny
    uniform 4×4×4 box for consistency.

    Notes
    -----
    Only the **dispersion + hard-chain** part of F^exc is returned here.
    Combine with the m-scaled FMT-aWBII c¹ (one HS reference per
    component, weights multiplied by m_c) for the full PC-SAFT cDFT.
    """
    m:      "jnp.ndarray"   # (C,)
    sigma:  "jnp.ndarray"   # (C,) Å
    eps_k:  "jnp.ndarray"   # (C,) K
    T:      float
    k_ij:   "jnp.ndarray | None" = None    # (C, C) binary interaction

    def __post_init__(self):
        self.m     = jnp.asarray(self.m,     dtype=jnp.float32)
        self.sigma = jnp.asarray(self.sigma, dtype=jnp.float32)
        self.eps_k = jnp.asarray(self.eps_k, dtype=jnp.float32)
        if self.k_ij is None:
            self.k_ij = jnp.zeros((self.m.size, self.m.size), dtype=jnp.float32)
        else:
            self.k_ij = jnp.asarray(self.k_ij, dtype=jnp.float32)

    # ------------------------- HSD per component ---------------------
    def _d(self):  # (C,)
        return self.sigma * (1.0 - 0.12 * jnp.exp(-3.0 * self.eps_k / self.T))

    def _R(self):  # (C,)
        return 0.5 * self._d()

    # ------------------------- weight functions ----------------------
    def _weights(self, K):
        """Return ``(w_disp, w_lambd, w_zeta2, w_zeta3)`` each shape (C, *K)."""
        R = self._R()                                           # (C,)
        # broadcast: each row uses its own R, K is full grid
        K_np = np.asarray(K)
        ws_disp, ws_l, ws_z2, ws_z3 = [], [], [], []
        for c in range(R.size):
            arg = 2.0 * float(R[c]) * K_np
            j0  = spherical_jn(0, arg)
            j2  = spherical_jn(2, arg)
            d_c = 2.0 * float(R[c])
            m_c = float(self.m[c])
            # dispersion ω_disp = m · (j0 + j2)  at 2 ψ R k
            arg_d = 2.0 * _PSI * float(R[c]) * K_np
            j0d   = spherical_jn(0, arg_d)
            j2d   = spherical_jn(2, arg_d)
            ws_disp.append(jnp.asarray(m_c * (j0d + j2d)))
            ws_l.append(jnp.asarray(j0))
            ws_z2.append(jnp.asarray((np.pi / 6.0) * m_c * d_c ** 2 * (j0 + j2)))
            ws_z3.append(jnp.asarray((np.pi / 6.0) * m_c * d_c ** 3 * (j0 + j2)))
        return (jnp.stack(ws_disp, axis=0),
                jnp.stack(ws_l,    axis=0),
                jnp.stack(ws_z2,   axis=0),
                jnp.stack(ws_z3,   axis=0))

    # ------------------------ Helmholtz density ----------------------
    def _phi_disp(self, n: jnp.ndarray) -> jnp.ndarray:
        """Multi-component dispersion Φ_disp/k_BT (Stierle Eq. via Gross 2001)."""
        eps_T  = self.eps_k / self.T                                # (C,)
        # binary parameters σ_ij, ε_ij with Lorentz--Berthelot
        sigma_ij = 0.5 * (self.sigma[:, None] + self.sigma[None, :])
        eps_ij   = jnp.sqrt(self.eps_k[:, None] * self.eps_k[None, :]) * (1.0 - self.k_ij)
        eps_ij_T = eps_ij / self.T
        e1sig3 = eps_ij_T * sigma_ij ** 3
        e2sig3 = (eps_ij_T) ** 2 * sigma_ij ** 3
        # local packing fraction η(r) = Σ_c (4π/3) n_c R_c³
        R = self._R()
        n_pos = jnp.clip(n, 0.0, None)
        eta = jnp.sum((4.0 / 3.0) * jnp.pi * n_pos
                      * (R[(slice(None),) + (None,) * (n.ndim - 1)] ** 3),
                      axis=0)
        # mean chain length m_hat — Gross & Sadowski Eq. (A.5)
        m = self.m[(slice(None),) + (None,) * (n.ndim - 1)]
        denom = jnp.sum(n_pos / m, axis=0)
        # Safe-divide: avoid 0/0 NaN in jax.grad when denom=0 at near-zero density.
        # JAX evaluates gradients through both branches of jnp.where, so we must
        # ensure the "true" branch is numerically safe even where the condition is False.
        safe_denom = jnp.where(denom > 1e-30, denom, 1.0)
        m_hat = jnp.where(denom > 1e-30, jnp.sum(n_pos, axis=0) / safe_denom, 1.0)
        m1 = (m_hat - 1.0) / m_hat
        m2 = (m_hat - 2.0) / m_hat
        # I1, I2 polynomials in η
        i1   = jnp.zeros_like(eta)
        c1i2 = jnp.zeros_like(eta)
        for i in range(7):
            i1   = i1   + (_A0[i] + m1 * _A1[i] + m1 * m2 * _A2[i]) * eta ** i
            c1i2 = c1i2 + (_B0[i] + m1 * _B1[i] + m1 * m2 * _B2[i]) * eta ** i
        C1_denom = (1.0
                    + m_hat * (8.0 * eta - 2.0 * eta ** 2) / (1.0 - eta) ** 4
                    + (1.0 - m_hat) * (20.0 * eta - 27.0 * eta ** 2
                                        + 12.0 * eta ** 3 - 2.0 * eta ** 4)
                    / ((1.0 - eta) * (2.0 - eta)) ** 2)
        c1i2 = c1i2 / C1_denom
        # quadratic forms Σ_ij n_i n_j (ε σ³)_ij
        nn_e1 = jnp.einsum("i...,j...,ij->...", n_pos, n_pos, e1sig3)
        nn_e2 = jnp.einsum("i...,j...,ij->...", n_pos, n_pos, e2sig3)
        return -2.0 * jnp.pi * i1 * nn_e1 - jnp.pi * m_hat * c1i2 * nn_e2

    def _phi_chain(self, rho, n_lambd, n_zeta2, n_zeta3) -> jnp.ndarray:
        """Multi-component hard-chain Φ_hc/k_BT  (Stierle Eq. 16 + 17)."""
        floor = 1e-8
        d_arr = self._d()                                   # (C,)
        z3    = jnp.clip(n_zeta3, 0.0, 0.95)
        inv   = 1.0 / (1.0 - z3)
        # Per-component cavity correlation y^dd_i(n_ζ2, n_ζ3) — Stierle (17).
        per_comp = []
        for c in range(d_arr.size):
            R_c   = 0.5 * d_arr[c]
            term  = (inv
                     + 3.0 * R_c * n_zeta2 * inv ** 2
                     + 2.0 * (R_c * n_zeta2) ** 2 * inv ** 3)
            n_l_c = jnp.clip(n_lambd[c], floor, None)
            rho_c = jnp.clip(rho[c],     floor, None)
            per_comp.append(-(self.m[c] - 1.0) * rho_c
                             * (jnp.log(term * n_l_c) - 1.0))
        return jnp.stack(per_comp, axis=0).sum(axis=0)

    def helmholtz_density(self, rho, w_disp, w_lambd, w_zeta2, w_zeta3):
        """Total dispersion + hard-chain φ(r)/k_BT, shape (Nx,Ny,Nz)."""
        rho_hat = jnp.fft.rfftn(rho, axes=(-3, -2, -1))         # (C, ..., k)
        inv = lambda H: jnp.fft.irfftn(H, s=rho.shape[-3:], axes=(-3, -2, -1))
        # per-component weighted densities (dispersion, lambda)
        n_disp  = inv(rho_hat * w_disp)                          # (C, Nx, Ny, Nz)
        n_lambd = inv(rho_hat * w_lambd)
        # summed weighted densities ζ2, ζ3
        n_zeta2 = inv(jnp.sum(rho_hat * w_zeta2, axis=0))         # (Nx, Ny, Nz)
        n_zeta3 = inv(jnp.sum(rho_hat * w_zeta3, axis=0))
        phi = self._phi_disp(n_disp)
        # Only add chain term if any m > 1.
        if bool(jnp.any(self.m > 1.0 + 1e-9)):
            phi = phi + self._phi_chain(rho, n_lambd, n_zeta2, n_zeta3)
        return phi

    def c1(self, rho, dV, w_disp, w_lambd, w_zeta2, w_zeta3):
        """Per-component c¹_c(r) via reverse-mode AD; shape (C, *grid)."""
        def F(r):
            return jnp.sum(self.helmholtz_density(r, w_disp, w_lambd,
                                                  w_zeta2, w_zeta3)) * dV
        grad = jax.grad(F)(rho)
        return -grad / dV

    def bulk_c1(self, rho_b):
        """Per-component bulk c¹ at uniform densities ``rho_b`` (shape (C,))."""
        from porecdft.functional.fmt import make_k_grid
        n = 4
        d = 1.0
        rho_b = jnp.asarray(rho_b, dtype=jnp.float32)
        rho = jnp.broadcast_to(rho_b[:, None, None, None],
                                (rho_b.size, n, n, n))
        _, _, _, K_rfft = make_k_grid((n, n, n), d, d, d, real_fft=True)
        wD, wL, wZ2, wZ3 = self._weights(K_rfft)
        c1_field = self.c1(rho, d ** 3, wD, wL, wZ2, wZ3)
        return jnp.asarray([float(c1_field[c].mean()) for c in range(rho_b.size)])
