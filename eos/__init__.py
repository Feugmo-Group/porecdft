"""Bulk equations of state for the fluid reservoir.

- Ideal gas:       adequate for CO2 at T ≥ 273 K, p ≤ 1 bar (Z > 0.99).
- LJ-MBWR:         needed for LJ WDA functional (bh_diameter, muexc, fexc).
- Peng-Robinson:   real-gas bulk density for H2 at high pressure (>10 bar).
"""
from porecdft.eos.ideal_gas import (
    density_from_pressure,
    chemical_potential_excess,
    K_B_J_PER_K,
    BAR_PA,
    ANGSTROM3_M3,
)
from porecdft.eos.lj_mbwr import LJEOS, bh_diameter
from porecdft.eos.peng_robinson import PengRobinsonEOS, H2_PR, N2_PR, CH4_PR

__all__ = [
    "density_from_pressure",
    "chemical_potential_excess",
    "K_B_J_PER_K",
    "BAR_PA",
    "ANGSTROM3_M3",
    "LJEOS",
    "bh_diameter",
    "PengRobinsonEOS",
    "H2_PR",
    "N2_PR",
    "CH4_PR",
]
