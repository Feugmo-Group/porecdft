"""Bulk equations of state for the fluid reservoir.

Phase-2 starting point — ideal gas only. LJ-MBWR (migrating from `cdft/eos.py`)
and PC-SAFT (from `cdft_pcsaft/eos.py`) will be added when we need accurate
high-pressure behaviour. For CO2 at T ≥ 273 K and p ≤ 1 bar, the ideal-gas
approximation is excellent (Z > 0.99).
"""
from porecdft.eos.ideal_gas import (
    density_from_pressure,
    chemical_potential_excess,
    K_B_J_PER_K,
    BAR_PA,
    ANGSTROM3_M3,
)

__all__ = [
    "density_from_pressure",
    "chemical_potential_excess",
    "K_B_J_PER_K",
    "BAR_PA",
    "ANGSTROM3_M3",
]
