"""Equation-of-state abstract base class for porecdft.

All bulk EOS classes in ``porecdft.eos`` should subclass :class:`EOSBase`.

Required interface
------------------
``bulk_density(P_bar, T_K) -> float``
    Number density in molecules / Å³ for a single-phase fluid at pressure
    *P* (bar) and temperature *T* (K). Must select the gas/fluid branch when
    multiple roots exist.

Optional but recommended
------------------------
``pressure(rho, T_K) -> float``
    Inverse mapping, used by the round-trip test
    ``pressure(bulk_density(P, T), T) ≈ P``.
``chemical_potential_excess(rho, T_K) -> float``
    μ_ex(ρ, T) in units of K (i.e. divided by k_B). Needed when the EOS is
    used as the bulk reference for a cDFT functional.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class EOSBase(ABC):
    """Abstract base for single-component bulk equations of state.

    Subclasses must implement :meth:`bulk_density` and **should** implement
    :meth:`pressure` so the round-trip identity
    ``pressure(bulk_density(P, T), T) ≈ P`` can be exercised by the test suite.

    Attributes
    ----------
    name : str
        Short identifier used in registries and log messages.
    molar_mass : float
        Molar mass in g/mol. Stored for convenience; not used by the engine.
    """

    name: str = "EOSBase"
    molar_mass: float = 1.0

    # ------------------------------------------------------------------ core
    @abstractmethod
    def bulk_density(self, P_bar: float, T_K: float) -> float:
        """Bulk number density in molecules / Å³ at *(P, T)*.

        Parameters
        ----------
        P_bar : float
            Pressure in bar.
        T_K : float
            Temperature in K.
        """

    def pressure(self, rho: float, T_K: float) -> float:
        """Pressure in bar at given bulk density *rho* (molecules / Å³) and *T*.

        Default raises :class:`NotImplementedError`; override in subclasses
        whose closed-form expression is available.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement pressure(rho, T)"
        )

    def chemical_potential_excess(self, rho: float, T_K: float) -> float:
        """Excess chemical potential μ_ex(ρ, T) / k_B  (units: K).

        Default raises :class:`NotImplementedError`; override when needed.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement chemical_potential_excess"
        )

    # ------------------------------------------------------------ helpers
    @staticmethod
    def reduced_T(T_K: float, Tc: float) -> float:
        """Reduced temperature ``T / Tc``."""
        return T_K / Tc

    @staticmethod
    def reduced_P(P_bar: float, Pc_bar: float) -> float:
        """Reduced pressure ``P / Pc`` (both in bar)."""
        return P_bar / Pc_bar

    # ------------------------------------------------------------ admin
    def __repr__(self) -> str:  # pragma: no cover — cosmetic only
        return f"<{type(self).__name__} name={self.name!r} M={self.molar_mass}>"
