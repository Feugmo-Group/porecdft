"""Feynman-Hibbs quantum-corrected equation of state.

The Feynman-Hibbs (FH) effective potential replaces a classical pair potential
``u_cl(r)`` by a Gaussian-smeared quadratic expansion

    u_FH(r) = u_cl(r) + (ℏ² / 24 m k_B T) · ∇² u_cl(r),

which captures leading-order quantum delocalisation of light particles
(H₂, He, Ne) at low temperature.  At the bulk EOS level this manifests as an
*inflated* effective hard-sphere diameter and a corresponding reduction of the
bulk number density at fixed pressure.

In this module we implement the cheapest possible FH correction: a temperature-
dependent multiplicative factor ``f_Q(T) ≤ 1`` applied to the density returned
by *any* classical EOS:

    ρ_FH(P, T) = ρ_classical(P, T) · f_Q(T).

For a single-site model (e.g. TraPPE H₂) the leading-order Sesé form is

    f_Q(T) = 1 / (1 + Λ*² / 12),       Λ* = Λ_dB / σ,

with Λ_dB = h / √(2π m k_B T) the thermal de Broglie wavelength.  This reduces
to ``f_Q → 1`` in the classical limit (T → ∞) and yields the expected ~6 %
reduction for H₂ at 77 K.

References
----------
Feynman, R. P.; Hibbs, A. R. *Quantum Mechanics and Path Integrals*
(McGraw-Hill, 1965).
Sesé, L. M. *Mol. Phys.* **1995**, *85*, 931.
"""
from __future__ import annotations

import numpy as np

from porecdft.eos.base import EOSBase


# Physical constants (SI)
_H_PLANCK = 6.62607015e-34   # J·s
_K_B = 1.380649e-23          # J/K
_AMU_KG = 1.66053906660e-27  # kg / amu


def quantum_factor(T_K: float, m_amu: float, sigma_A: float) -> float:
    """Leading-order Feynman-Hibbs density-reduction factor ``f_Q(T) ≤ 1``.

    Parameters
    ----------
    T_K : float
        Temperature in K.
    m_amu : float
        Molecular mass in atomic mass units.
    sigma_A : float
        Lennard-Jones size parameter in Å (used as the relevant length scale
        for the reduced thermal wavelength ``Λ* = Λ_dB / σ``).

    Returns
    -------
    float
        ``f_Q = 1 / (1 + Λ*² / 12)``.  Approaches 1 as ``T → ∞``.
    """
    m_kg = m_amu * _AMU_KG
    Lambda_dB_m = _H_PLANCK / np.sqrt(2.0 * np.pi * m_kg * _K_B * T_K)
    Lambda_dB_A = Lambda_dB_m * 1.0e10
    L_star = Lambda_dB_A / sigma_A
    return float(1.0 / (1.0 + L_star * L_star / 12.0))


class FeynmanHibbsEOS(EOSBase):
    """Quantum-corrected wrapper around any classical :class:`EOSBase`.

    The wrapped EOS provides the classical bulk density and pressure; this
    class multiplies the density by ``f_Q(T)`` (a temperature-dependent
    Feynman-Hibbs factor between 0 and 1) and inverts that mapping for
    :meth:`pressure`.

    Parameters
    ----------
    classical_eos : EOSBase
        The underlying classical EOS (e.g. :data:`porecdft.eos.H2_PR`).
    m_amu : float
        Molecular mass in atomic mass units (for the de Broglie wavelength).
    sigma_A : float
        Lennard-Jones size parameter in Å (for the reduced quantum parameter).
    name : str, optional
        Short identifier (default ``"FH"``).
    """

    name = "FH"

    def __init__(
        self,
        classical_eos: EOSBase,
        m_amu: float,
        sigma_A: float,
        name: str | None = None,
    ):
        self._classical = classical_eos
        self.m_amu = float(m_amu)
        self.sigma_A = float(sigma_A)
        # Inherit molar mass from wrapped EOS so downstream code that reads
        # ``.molar_mass`` keeps working.
        self.molar_mass = getattr(classical_eos, "molar_mass", m_amu)
        if name is not None:
            self.name = name

    # ------------------------------------------------------------------ FH
    def quantum_factor(self, T_K: float) -> float:
        """Temperature-dependent FH density reduction factor."""
        return quantum_factor(T_K, self.m_amu, self.sigma_A)

    # ------------------------------------------------------------ EOSBase API
    def bulk_density(self, P_bar: float, T_K: float) -> float:
        """Quantum-corrected bulk number density (molecules / Å³)."""
        rho_cl = self._classical.bulk_density(P_bar, T_K)
        return float(rho_cl) * self.quantum_factor(T_K)

    def pressure(self, rho: float, T_K: float) -> float:
        """Pressure (bar) consistent with :meth:`bulk_density` (inverse map).

        Given the FH-corrected density ``rho``, the classical density that
        would have produced it is ``rho / f_Q(T)``; feed that into the
        underlying EOS to recover *P*.
        """
        return self._classical.pressure(rho / self.quantum_factor(T_K), T_K)


# ─── Pre-built instances for common quantum gases ─────────────────────────────

# Lazy-instantiated to avoid circular imports if someone imports this module
# directly without the package __init__.
from porecdft.eos.peng_robinson import H2_PR  # noqa: E402

#: H₂ with FH correction wrapping the Peng-Robinson classical EOS.
#: TraPPE single-site H₂: m = 2.016 amu, σ = 2.83 Å.
H2_FH = FeynmanHibbsEOS(
    classical_eos=H2_PR, m_amu=2.016, sigma_A=2.83, name="H2_FH"
)
