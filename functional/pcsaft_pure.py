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
