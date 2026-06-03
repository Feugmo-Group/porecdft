"""Ideal-gas bulk EOS.

Provides ρ_bulk(p, T) and μ(p, T) for an ideal gas. Adequate for CO2 at
T ≥ 273 K and p ≤ 1 bar (compressibility factor Z ≈ 0.995 in that range).
For higher pressure or near the critical point, swap in `LJ-MBWR` or `PC-SAFT`.
"""
from __future__ import annotations

import numpy as np

# k_B in joule per kelvin
K_B_J_PER_K = 1.380649e-23

# 1 bar in Pa
BAR_PA = 1.0e5

# 1 Å³ in m³
ANGSTROM3_M3 = 1.0e-30


def density_from_pressure(p_bar: float, T_K: float) -> float:
    """Bulk number density in molecules per Å³ at pressure p (bar) and T (K)."""
    p_Pa = p_bar * BAR_PA
    rho_per_m3 = p_Pa / (K_B_J_PER_K * T_K)
    return float(rho_per_m3 * ANGSTROM3_M3)


def chemical_potential_excess(rho: float, T_K: float, reference_rho: float = 1.0) -> float:
    """Excess chemical potential μ_id(ρ) − μ_id(ρ_ref) for an ideal gas, in units of K.

    Includes only the configurational ln(ρ/ρ_ref) term — translational and
    rotational kinetic-energy parts cancel when comparing bulk to confined.
    """
    return float(T_K * np.log(rho / reference_rho))
