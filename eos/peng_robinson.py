"""Peng-Robinson equation of state for real-gas bulk density.

Used to convert pressure → number density for gases that deviate from
ideal-gas behaviour at high pressures (H2 above ~10 bar, CH4, etc.).

The cubic Z³ + c2·Z² + c1·Z + c0 = 0 is solved via the companion-matrix
eigenvalue method (see :mod:`porecdft.eos.cubic_utils`) so the code runs
under JAX on any backend (CPU/GPU/TPU) without falling back to ``np.roots``.

Reference
---------
Peng, D.-Y.; Robinson, D. B. *Ind. Eng. Chem. Fundam.* **1976**, 15, 59.
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


class PengRobinsonEOS(EOSBase):
    """Peng-Robinson EOS for a single-component real gas.

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
        Short identifier (default ``"PR"``).
    """

    name = "PR"
    #: ``bulk_density`` uses ``jnp`` internally so the heavy linear algebra
    #: runs on the JAX backend (CPU/GPU/TPU). The companion-matrix cubic-root
    #: solver currently includes ``float()`` casts that prevent wrapping the
    #: whole function in ``jax.jit``; if a JIT-able variant is needed for
    #: batched bulk-density evaluation, refactor ``cubic_utils.solve_cubic_gas_root``
    #: to return a ``jnp`` array and delete the casts.
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
        self._kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2

    # ------------------------------------------------------------------ a, b
    def _a(self, T_K: float) -> float:
        """Temperature-dependent attractive parameter ``a(T)``."""
        alpha = (1.0 + self._kappa * (1.0 - jnp.sqrt(T_K / self.Tc))) ** 2
        return 0.45724 * R_GAS_J_MOL_K**2 * self.Tc**2 / self.Pc * alpha

    def _b(self) -> float:
        """Co-volume parameter ``b``."""
        return 0.07780 * R_GAS_J_MOL_K * self.Tc / self.Pc

    # ------------------------------------------------------------ EOSBase API
    def bulk_density(self, P_bar: float, T_K: float) -> float:
        """Number density in molecules·Å⁻³ at pressure *P* (bar) and *T* (K)."""
        P_Pa = bar_to_Pa(P_bar)
        a = self._a(T_K)
        b = self._b()

        A = a * P_Pa / (R_GAS_J_MOL_K * T_K) ** 2
        B = b * P_Pa / (R_GAS_J_MOL_K * T_K)

        # Z³ − (1−B)·Z² + (A−3B²−2B)·Z − (AB−B²−B³) = 0
        c2 = -(1.0 - B)
        c1 = A - 3.0 * B**2 - 2.0 * B
        c0 = -(A * B - B**2 - B**3)

        Z = solve_cubic_gas_root(c2, c1, c0)
        return number_density_from_Z(Z, P_Pa, T_K)

    def pressure(self, rho: float, T_K: float) -> float:
        """Pressure in bar at density ``rho`` (molecules / Å³) and *T* (K).

        Uses the explicit PR form ``P = R T / (V_m − b) − a / (V_m² + 2 b V_m − b²)``.
        """
        # rho [molecules/Å³] -> V_m [m³/mol]
        # rho * 1e30 = molecules/m³ ; divide by N_A → mol/m³ ; invert → m³/mol
        N_A = 6.022e23
        Vm = N_A / (rho * 1e30)
        a = self._a(T_K)
        b = self._b()
        P_Pa = R_GAS_J_MOL_K * T_K / (Vm - b) - a / (Vm * Vm + 2.0 * b * Vm - b * b)
        return float(P_Pa / 1e5)


# ─── Pre-built instances for common gases ─────────────────────────────────────

#: H2 (single-site TraPPE) — Tc=33.145 K, Pc=12.964 bar, ω=−0.219
H2_PR = PengRobinsonEOS(
    Tc=33.145, Pc=12.964e5, omega=-0.219, molar_mass=2.016, name="H2_PR"
)

#: N2 — Tc=126.2 K, Pc=33.9 bar, ω=0.039
N2_PR = PengRobinsonEOS(
    Tc=126.2, Pc=33.9e5, omega=0.039, molar_mass=28.014, name="N2_PR"
)

#: CH4 — Tc=190.6 K, Pc=46.1 bar, ω=0.011
CH4_PR = PengRobinsonEOS(
    Tc=190.6, Pc=46.1e5, omega=0.011, molar_mass=16.043, name="CH4_PR"
)
