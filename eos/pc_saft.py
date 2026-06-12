"""Perturbed-Chain SAFT (PC-SAFT) equation of state — bulk density route (v0.2).

Implements the canonical PC-SAFT EOS of Gross & Sadowski for a single,
non-associating pure component. Hard-chain reference (Chapman 1990,
Carnahan-Starling hard-sphere term) plus a Barker-Henderson perturbation
dispersion contribution truncated at second order — full Eqs. (A.6)-(A.11)
from the original paper with the universal coefficients given in
Eqs. (A.18)-(A.19).

Scope (v0.2)
------------
* Pure-component, no association sites.
* No mixtures (single segment type, single ε, single σ).
* Provides :meth:`pressure` and :meth:`bulk_density`.  Chemical-potential
  and VLE work belong to v0.3.

GPU compatibility
-----------------
Implementation uses ``jax.numpy`` throughout and computes the
compressibility factor via :func:`jax.grad` of the residual Helmholtz
energy.  All scalar quantities are JAX-traceable and the EOS can be
wrapped in ``jax.jit`` for GPU/TPU evaluation.

Reference
---------
Gross, J.; Sadowski, G.  *Ind. Eng. Chem. Res.* **2001**, *40*, 1244.
Chapman, W. G. et al.   *Ind. Eng. Chem. Res.* **1990**, *29*, 1709.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from porecdft.eos.base import EOSBase

# ─── Physical constants ──────────────────────────────────────────────────────
_K_B_J_PER_K = 1.380649e-23
_BAR_PA = 1.0e5
_ANGSTROM3_M3 = 1.0e-30


# ─── Universal PC-SAFT constants (Gross & Sadowski 2001, Eqs. A.18-A.19) ────
_A0 = jnp.array([0.9105631445, 0.6361281449, 2.6861347891, -26.547362491,
                 97.759208784, -159.59154087, 91.297774084])
_A1 = jnp.array([-0.3084016918, 0.1860531159, -2.5030047259, 21.419793629,
                 -65.255885330, 83.318680481, -33.746922930])
_A2 = jnp.array([-0.0906148351, 0.4527842806, 0.5962700728, -1.7241829131,
                 -4.1302112531, 13.776631870, -8.6728470368])

_B0 = jnp.array([0.7240946941, 2.2382791861, -4.0025849485, -21.003576815,
                 26.855641363, 206.55133841, -355.60235612])
_B1 = jnp.array([-0.5755498075, 0.6995095521, 3.8925673390, -17.215471648,
                 192.67226447, -161.82646165, -165.20769346])
_B2 = jnp.array([0.0976883116, -0.2557574982, -9.1558561530, 20.642075974,
                 -38.804430052, 93.626774077, -29.666905585])

_I_POW = jnp.arange(7)


def _a_coeffs(m):
    """``a_i(m)`` per Eq. A.18 — length-7 jnp.array."""
    f1 = (m - 1.0) / m
    f2 = f1 * (m - 2.0) / m
    return _A0 + f1 * _A1 + f2 * _A2


def _b_coeffs(m):
    """``b_i(m)`` per Eq. A.19 — length-7 jnp.array."""
    f1 = (m - 1.0) / m
    f2 = f1 * (m - 2.0) / m
    return _B0 + f1 * _B1 + f2 * _B2


def _I1(eta, m):
    """``I1(η, m)`` — power series of order 6 in η."""
    return jnp.sum(_a_coeffs(m) * eta ** _I_POW)


def _I2(eta, m):
    """``I2(η, m)`` — power series of order 6 in η."""
    return jnp.sum(_b_coeffs(m) * eta ** _I_POW)


def _C1(eta, m):
    """``C1(η, m)`` per Eq. A.11 (inverse compressibility-like factor)."""
    t1 = m * (8.0 * eta - 2.0 * eta ** 2) / (1.0 - eta) ** 4
    t2 = (1.0 - m) * (
        20.0 * eta - 27.0 * eta ** 2 + 12.0 * eta ** 3 - 2.0 * eta ** 4
    ) / ((1.0 - eta) * (2.0 - eta)) ** 2
    return 1.0 / (1.0 + t1 + t2)


class PCSAFTEOS(EOSBase):
    """Pure-fluid PC-SAFT equation of state (Gross & Sadowski 2001).

    Parameters
    ----------
    m : float
        Number of segments per chain.
    sigma_A : float
        Segment diameter σ in Å.
    epsilon_K : float
        Segment energy ε/k_B in K.
    molar_mass : float, optional
        Molar mass (g/mol), stored for convenience.
    name : str, optional
        Identifier; defaults to ``"PC-SAFT"``.
    """

    name = "PC-SAFT"

    def __init__(
        self,
        m: float,
        sigma_A: float,
        epsilon_K: float,
        molar_mass: float = 1.0,
        name: str | None = None,
    ):
        self.m = float(m)
        self.sigma_A = float(sigma_A)
        self.epsilon_K = float(epsilon_K)
        self.molar_mass = float(molar_mass)
        if name is not None:
            self.name = name

    # ------------------------------------------------------------ structure
    def _d(self, T_K):
        """Temperature-dependent Barker-Henderson diameter ``d(T)``.

        ``d = σ · (1 − 0.12·exp(−3·ε/(k_B T)))``  (Gross & Sadowski Eq. A.9).
        """
        return self.sigma_A * (1.0 - 0.12 * jnp.exp(-3.0 * self.epsilon_K / T_K))

    def _eta(self, rho, T_K):
        """Packing fraction ``η = (π/6)·ρ·m·d³`` with ρ in molecules/Å³."""
        d = self._d(T_K)
        return (jnp.pi / 6.0) * rho * self.m * d ** 3

    # ----------------------------------------------------------- a_res / NkT
    def _a_hs(self, eta):
        """Carnahan-Starling reduced hard-sphere Helmholtz energy per segment."""
        return (4.0 * eta - 3.0 * eta ** 2) / (1.0 - eta) ** 2

    def _g_hs(self, eta):
        """Hard-sphere contact RDF (single-component form)."""
        return (1.0 - 0.5 * eta) / (1.0 - eta) ** 3

    def _a_res_per_NkT(self, rho, T_K):
        """Residual Helmholtz energy ``ã^res = a_res / (N k_B T)``.

        Sum of hard-chain (Chapman 1990) and dispersion (Gross & Sadowski 2001)
        contributions.  All quantities are scalar; can be auto-diffed.
        """
        eta = self._eta(rho, T_K)
        d = self._d(T_K)
        # Hard-chain: m·ã^hs − (m−1)·ln g^hs   (single species)
        a_hc = self.m * self._a_hs(eta) - (self.m - 1.0) * jnp.log(self._g_hs(eta))
        # Dispersion (Eq. A.10)
        eps_over_T = self.epsilon_K / T_K
        I1 = _I1(eta, self.m)
        I2 = _I2(eta, self.m)
        C1 = _C1(eta, self.m)
        a_disp = (
            -2.0 * jnp.pi * rho * I1 * (self.m ** 2) * eps_over_T * self.sigma_A ** 3
            - jnp.pi * rho * self.m * C1 * I2
            * (self.m ** 2) * eps_over_T ** 2 * self.sigma_A ** 3
        )
        return a_hc + a_disp

    # ----------------------------------------------------------- Z, P, ρ
    def _Z(self, rho, T_K):
        """Compressibility factor ``Z = 1 + ρ · ∂ã^res/∂ρ``.

        Uses :func:`jax.grad` to obtain the density derivative — analytic in
        η is also possible but auto-diff keeps the code short and trustworthy.
        """
        da_drho = jax.grad(self._a_res_per_NkT, argnums=0)(rho, T_K)
        return 1.0 + rho * da_drho

    def pressure(self, rho, T_K):
        """Pressure in **bar** at density ``rho`` (molecules/Å³) and *T*."""
        Z = self._Z(jnp.asarray(rho, dtype=jnp.float64),
                    jnp.asarray(T_K, dtype=jnp.float64))
        rho_per_m3 = float(rho) / _ANGSTROM3_M3
        P_Pa = rho_per_m3 * _K_B_J_PER_K * float(T_K) * float(Z)
        return P_Pa / _BAR_PA

    def bulk_density(self, P_bar, T_K):
        """Solve ``P(ρ, T) = P_bar`` on the gas branch via safeguarded Newton.

        Ideal-gas seed; Newton step capped at ±50 % of current ρ to avoid
        overshoot into the liquid root.  Jacobian via ``jax.grad`` of the
        pressure with respect to ρ.
        """
        P_target_Pa = float(P_bar) * _BAR_PA
        T = float(T_K)
        # Ideal-gas seed (molecules/Å³)
        rho = (P_target_Pa / (_K_B_J_PER_K * T)) * _ANGSTROM3_M3

        # Hard-sphere cap so we never cross into the liquid root.
        d = float(self._d(T))
        eta_cap = 0.5
        rho_cap = eta_cap / ((jnp.pi / 6.0) * self.m * d ** 3)

        # Local closure returning pressure in Pa as a JAX-traceable scalar.
        def _P_Pa(rho_val):
            Z = self._Z(rho_val, jnp.asarray(T, dtype=jnp.float64))
            return rho_val / _ANGSTROM3_M3 * _K_B_J_PER_K * T * Z

        dP_drho = jax.grad(_P_Pa)

        for _ in range(100):
            rho_j = jnp.asarray(rho, dtype=jnp.float64)
            P_calc = float(_P_Pa(rho_j))
            f = P_calc - P_target_Pa
            if abs(f) < 1e-8 * max(P_target_Pa, 1.0):
                return float(rho)
            J = float(dP_drho(rho_j))
            if J <= 0.0:
                rho *= 0.5
                continue
            step = f / J
            # Cap step at ±50 % of current ρ
            max_step = 0.5 * rho
            if step > max_step:
                step = max_step
            elif step < -max_step:
                step = -max_step
            rho_new = rho - step
            if rho_new <= 0.0:
                rho_new = 0.5 * rho
            if rho_new > rho_cap:
                rho_new = 0.5 * (rho + rho_cap)
            if abs(rho_new - rho) < 1e-16:
                return float(rho_new)
            rho = float(rho_new)
        return float(rho)


# ─── Pre-built singletons (Gross & Sadowski 2001, Table 1) ──────────────────

#: CO2 — m=2.0729, σ=2.7852 Å, ε/k_B=169.21 K
CO2_PCSAFT = PCSAFTEOS(
    m=2.0729, sigma_A=2.7852, epsilon_K=169.21,
    molar_mass=44.01, name="CO2_PCSAFT",
)

#: N2 — m=1.2053, σ=3.3130 Å, ε/k_B=90.96 K
N2_PCSAFT = PCSAFTEOS(
    m=1.2053, sigma_A=3.3130, epsilon_K=90.96,
    molar_mass=28.014, name="N2_PCSAFT",
)

#: CH4 — m=1.0000, σ=3.7039 Å, ε/k_B=150.03 K
CH4_PCSAFT = PCSAFTEOS(
    m=1.0000, sigma_A=3.7039, epsilon_K=150.03,
    molar_mass=16.043, name="CH4_PCSAFT",
)
