"""Soave-Redlich-Kwong equation of state for real-gas bulk density.

A two-parameter cubic EOS — same algebraic form as Peng-Robinson but with a
different temperature-dependent α(T) and a different κ(ω) correlation
(Soave 1972). At ambient conditions for small molecules (CO2, CH4, N2) it
reproduces ideal-gas densities to within ~1 %; deviations grow with pressure.

The cubic Z³ + c2·Z² + c1·Z + c0 = 0 is solved via the companion-matrix
eigenvalue method (see :mod:`porecdft.eos.cubic_utils`) so the code runs
under JAX on any backend (CPU/GPU/TPU) without falling back to ``np.roots``.

Reference
---------
Soave, G. *Chem. Eng. Sci.* **1972**, 27, 1197.
"""
from __future__ import annotations

import jax.numpy as jnp

from porecdft.eos.base import EOSBase
from porecdft.eos.cubic_utils import (
    R_GAS_J_MOL_K,
    bar_to_Pa,
    number_density_from_Z,
    solve_cubic_gas_root,
)


class SRKEOS(EOSBase):
    """Soave-Redlich-Kwong EOS for a single-component real gas.

    Parameters
    ----------
    Tc : float
        Critical temperature (K).
    Pc : float
        Critical pressure (Pa).
    omega : float
        Acentric factor.
    molar_mass : float, optional
        Molar mass (g/mol). Stored for convenience.
    name : str, optional
        Short identifier (default ``"SRK"``).
    """

    name = "SRK"
    #: Same JAX-internal / float-cast story as Peng-Robinson — runs on the
    #: JAX backend but not currently wrappable in ``jax.jit``.
    JIT_SAFE = False
    GPU_READY = True

    def __init__(
        self,
        Tc: float,
        Pc: float,
        omega: float,
        molar_mass: float = 1.0,
        name: str | None = None,
    ):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.molar_mass = molar_mass
        if name is not None:
            self.name = name
        # Soave (1972) κ(ω) — differs from PR's 0.37464 + 1.54226 ω − 0.26992 ω²
        self._kappa = 0.480 + 1.574 * omega - 0.176 * omega**2

    # ------------------------------------------------------------------ a, b
    def _a(self, T_K: float) -> float:
        """Temperature-dependent attractive parameter ``a(T)``."""
        alpha = (1.0 + self._kappa * (1.0 - jnp.sqrt(T_K / self.Tc))) ** 2
        return 0.42748 * R_GAS_J_MOL_K**2 * self.Tc**2 / self.Pc * alpha

    def _b(self) -> float:
        """Co-volume parameter ``b``."""
        return 0.08664 * R_GAS_J_MOL_K * self.Tc / self.Pc

    # ------------------------------------------------------------ EOSBase API
    def bulk_density(self, P_bar: float, T_K: float) -> float:
        """Number density in molecules·Å⁻³ at pressure *P* (bar) and *T* (K)."""
        P_Pa = bar_to_Pa(P_bar)
        a = self._a(T_K)
        b = self._b()

        A = a * P_Pa / (R_GAS_J_MOL_K * T_K) ** 2
        B = b * P_Pa / (R_GAS_J_MOL_K * T_K)

        # Z³ − Z² + (A − B − B²)·Z − A·B = 0
        c2 = -1.0
        c1 = A - B - B**2
        c0 = -A * B

        Z = solve_cubic_gas_root(c2, c1, c0)
        return number_density_from_Z(Z, P_Pa, T_K)

    def pressure(self, rho: float, T_K: float) -> float:
        """Pressure in bar at density ``rho`` (molecules / Å³) and *T* (K).

        Uses the explicit SRK form ``P = R T / (V_m − b) − a / (V_m·(V_m + b))``.
        """
        # rho [molecules/Å³] -> V_m [m³/mol]
        N_A = 6.022e23
        Vm = N_A / (rho * 1e30)
        a = self._a(T_K)
        b = self._b()
        P_Pa = R_GAS_J_MOL_K * T_K / (Vm - b) - a / (Vm * (Vm + b))
        return float(P_Pa / 1e5)


# ─── Pre-built instances for common gases ─────────────────────────────────────

#: CO2 — Tc=304.13 K, Pc=73.77 bar, ω=0.225
CO2_SRK = SRKEOS(
    Tc=304.13, Pc=73.77e5, omega=0.225, molar_mass=44.01, name="CO2_SRK"
)

#: CH4 — Tc=190.6 K, Pc=46.1 bar, ω=0.011
CH4_SRK = SRKEOS(
    Tc=190.6, Pc=46.1e5, omega=0.011, molar_mass=16.043, name="CH4_SRK"
)

#: N2 — Tc=126.2 K, Pc=33.9 bar, ω=0.039
N2_SRK = SRKEOS(
    Tc=126.2, Pc=33.9e5, omega=0.039, molar_mass=28.014, name="N2_SRK"
)
