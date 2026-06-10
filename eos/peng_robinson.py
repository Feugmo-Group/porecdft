"""Peng-Robinson equation of state for real-gas bulk density.

Used to convert pressure → number density for gases that deviate from
ideal-gas behaviour at high pressures (H2 above ~10 bar, CH4, etc.).

The cubic Z³ + c2·Z² + c1·Z + c0 = 0 is solved via the companion-matrix
eigenvalue method so the code runs under JAX on any backend (CPU/GPU/TPU)
without falling back to ``np.roots``.
"""
from __future__ import annotations

import jax.numpy as jnp

_R_GAS = 8.314        # J/(mol·K)
_NA    = 6.022e23     # mol⁻¹


class PengRobinsonEOS:
    """Peng-Robinson EOS for a single-component real gas.

    Parameters
    ----------
    Tc : float
        Critical temperature (K).
    Pc : float
        Critical pressure (Pa).
    omega : float
        Acentric factor.
    molar_mass : float
        Molar mass (g/mol). Not used internally but stored for convenience.
    """

    def __init__(self, Tc: float, Pc: float, omega: float, molar_mass: float = 1.0):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.molar_mass = molar_mass
        self._kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2

    def bulk_density(self, P_bar: float, T_K: float) -> float:
        """Number density in molecules·Å⁻³ at pressure P (bar) and T (K).

        Selects the largest real positive root of the PR cubic (gas branch).
        """
        P = P_bar * 1e5  # Pa
        alpha = (1.0 + self._kappa * (1.0 - jnp.sqrt(T_K / self.Tc)))**2
        a = 0.45724 * _R_GAS**2 * self.Tc**2 / self.Pc * alpha
        b = 0.07780 * _R_GAS * self.Tc / self.Pc

        A = a * P / (_R_GAS * T_K)**2
        B = b * P / (_R_GAS * T_K)

        # Z³ − (1−B)·Z² + (A−3B²−2B)·Z − (AB−B²−B³) = 0
        c2 = -(1.0 - B)
        c1 = A - 3.0*B**2 - 2.0*B
        c0 = -(A*B - B**2 - B**3)

        companion = jnp.array([
            [0.0, 0.0, -c0],
            [1.0, 0.0, -c1],
            [0.0, 1.0, -c2],
        ])
        roots = jnp.linalg.eigvals(companion)
        real_parts = jnp.real(roots)
        is_real_pos = (jnp.abs(jnp.imag(roots)) < 1e-10) & (real_parts > 0)
        Z = float(jnp.max(jnp.where(is_real_pos, real_parts, -jnp.inf)))

        Vm = Z * _R_GAS * T_K / P   # m³/mol
        return _NA / (Vm * 1e30)    # molecules/Å³


# ─── Pre-built instances for common gases ─────────────────────────────────────

#: H2 (single-site TraPPE) — Tc=33.145 K, Pc=12.964 bar, ω=−0.219
H2_PR = PengRobinsonEOS(Tc=33.145, Pc=12.964e5, omega=-0.219, molar_mass=2.016)

#: N2 — Tc=126.2 K, Pc=33.9 bar, ω=0.039
N2_PR = PengRobinsonEOS(Tc=126.2, Pc=33.9e5, omega=0.039, molar_mass=28.014)

#: CH4 — Tc=190.6 K, Pc=46.1 bar, ω=0.011
CH4_PR = PengRobinsonEOS(Tc=190.6, Pc=46.1e5, omega=0.011, molar_mass=16.043)
