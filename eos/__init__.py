"""Bulk equations of state for the fluid reservoir.

- Ideal gas:       adequate for CO2 at T ≥ 273 K, p ≤ 1 bar (Z > 0.99).
- LJ-MBWR:         needed for LJ WDA functional (bh_diameter, muexc, fexc).
- Peng-Robinson:   real-gas bulk density for H2 at high pressure (>10 bar).

All cubic / non-ideal EOS subclass :class:`porecdft.eos.base.EOSBase`.
Cubic-root utilities live in :mod:`porecdft.eos.cubic_utils`.
"""
from porecdft.eos.base import EOSBase
from porecdft.eos.cubic_utils import (
    R_GAS_J_MOL_K,
    N_A,
    solve_cubic_gas_root,
    solve_cubic_liquid_root,
    solve_cubic_real_roots,
    number_density_from_Z,
    bar_to_Pa,
    cubic_to_gas_density,
)
from porecdft.eos.ideal_gas import (
    density_from_pressure,
    chemical_potential_excess,
    K_B_J_PER_K,
    BAR_PA,
    ANGSTROM3_M3,
)
from porecdft.eos.lj_mbwr import LJEOS, bh_diameter
from porecdft.eos.peng_robinson import PengRobinsonEOS, H2_PR, N2_PR, CH4_PR
from porecdft.eos.span_wagner_co2 import SpanWagnerCO2EOS, CO2_SW
from porecdft.eos.srk import SRKEOS, CO2_SRK, CH4_SRK, N2_SRK
from porecdft.eos.cpa import CPAEOS, H2O_CPA
from porecdft.eos.saft_vr_mie import SAFTVRMieEOS, CO2_SAFT_VR_Mie
from porecdft.eos.pc_saft import PCSAFTEOS, CO2_PCSAFT, N2_PCSAFT, CH4_PCSAFT
from porecdft.eos.feynman_hibbs import FeynmanHibbsEOS, H2_FH

__all__ = [
    # base + utilities
    "EOSBase",
    "R_GAS_J_MOL_K",
    "N_A",
    "solve_cubic_gas_root",
    "solve_cubic_liquid_root",
    "solve_cubic_real_roots",
    "number_density_from_Z",
    "bar_to_Pa",
    "cubic_to_gas_density",
    # ideal gas
    "density_from_pressure",
    "chemical_potential_excess",
    "K_B_J_PER_K",
    "BAR_PA",
    "ANGSTROM3_M3",
    # LJ-MBWR
    "LJEOS",
    "bh_diameter",
    # Peng-Robinson
    "PengRobinsonEOS",
    "H2_PR",
    "N2_PR",
    "CH4_PR",
    # Span-Wagner CO2
    "SpanWagnerCO2EOS",
    "CO2_SW",
    # SRK
    "SRKEOS",
    "CO2_SRK",
    "CH4_SRK",
    "N2_SRK",
    # CPA
    "CPAEOS",
    "H2O_CPA",
    # SAFT-VR-Mie
    "SAFTVRMieEOS",
    "CO2_SAFT_VR_Mie",
    # PC-SAFT
    "PCSAFTEOS",
    "CO2_PCSAFT",
    "N2_PCSAFT",
    "CH4_PCSAFT",
    # Feynman-Hibbs quantum-corrected wrapper
    "FeynmanHibbsEOS",
    "H2_FH",
]
