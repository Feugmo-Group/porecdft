"""Fluid (adsorbate) models — molecular geometry, site labels, LJ params, charges.

A Fluid is a small dataclass that describes a single molecule in its body frame.
The cDFT engine reads `body_sites`, `site_labels`, plus per-label LJ/charge
information; it does not bake in any specific adsorbate.
"""

from porecdft.fluid.base import Fluid
from porecdft.fluid.co2 import EPM2_CO2, TraPPE_CO2, SingleSiteLJ_CO2
from porecdft.fluid.n2 import TraPPE_N2

__all__ = ["Fluid", "EPM2_CO2", "TraPPE_CO2", "SingleSiteLJ_CO2", "TraPPE_N2"]
