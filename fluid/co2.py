"""Carbon dioxide fluid models.

EPM2 (Harris & Yung 1995): 3 LJ sites + 3 point charges, rigid linear.
  d_CO = 1.149 Å; q_C = +0.6512 e, q_O = -0.3256 e.
  σ_C = 2.757 Å, ε_C/k_B = 28.129 K;  σ_O = 3.033 Å, ε_O/k_B = 80.507 K.

TraPPE (Potoff & Siepmann 2001): 3 LJ sites + 3 charges, rigid linear.
  d_CO = 1.160 Å; q_C = +0.700 e, q_O = -0.350 e.
  σ_C = 2.800 Å, ε_C/k_B = 27.0 K;  σ_O = 3.050 Å, ε_O/k_B = 79.0 K.

SingleSiteLJ_CO2 (mTraFF in the legacy LJ-cDFT code): one LJ centre, no charges,
no quadrupole. σ = 3.017 Å, ε/k_B = 85.671 K. Useful as a cheap baseline.
"""

from __future__ import annotations

import numpy as np

from porecdft.io.forcefield import FFEntry
from porecdft.fluid.base import Fluid

# Geometry helpers
def _linear_three_site(d: float) -> np.ndarray:
    """Linear molecule along +z: O at -d, C at 0, O at +d."""
    return np.array([[0.0, 0.0, -d], [0.0, 0.0, 0.0], [0.0, 0.0, +d]])


# ---- EPM2 ---------------------------------------------------------------
_EPM2_FF = {
    "C": FFEntry("C", 2.757, 28.129, "EPM2"),
    "O": FFEntry("O", 3.033, 80.507, "EPM2"),
}
_EPM2_Q = {"C": +0.6512, "O": -0.3256}
_EPM2_D = 1.149
_EPM2_THETA_ZZ = 2.0 * (-0.3256) * _EPM2_D * _EPM2_D  # ≈ -0.8597 e·Å² (oblate, Θ_zz<0)

EPM2_CO2 = Fluid(
    name="CO2-EPM2",
    body_sites=_linear_three_site(_EPM2_D),
    site_labels=["O", "C", "O"],
    ff=_EPM2_FF,
    charges=_EPM2_Q,
    theta_zz=_EPM2_THETA_ZZ,
    molar_mass=44.01,
)

# ---- TraPPE -------------------------------------------------------------
_TRAPPE_FF = {
    "C": FFEntry("C", 2.800, 27.0, "TraPPE"),
    "O": FFEntry("O", 3.050, 79.0, "TraPPE"),
}
_TRAPPE_Q = {"C": +0.700, "O": -0.350}
_TRAPPE_D = 1.160
_TRAPPE_THETA_ZZ = 2.0 * (-0.350) * _TRAPPE_D * _TRAPPE_D

TraPPE_CO2 = Fluid(
    name="CO2-TraPPE",
    body_sites=_linear_three_site(_TRAPPE_D),
    site_labels=["O", "C", "O"],
    ff=_TRAPPE_FF,
    charges=_TRAPPE_Q,
    theta_zz=_TRAPPE_THETA_ZZ,
    molar_mass=44.01,
)

# ---- Single-site LJ ------------------------------------------------------
_SS_FF = {"CO2": FFEntry("CO2", 3.017, 85.671, "mTraFF/Garcia-Sanchez2009")}
SingleSiteLJ_CO2 = Fluid(
    name="CO2-SingleSiteLJ",
    body_sites=np.array([[0.0, 0.0, 0.0]]),
    site_labels=["CO2"],
    ff=_SS_FF,
    charges={"CO2": 0.0},
    theta_zz=0.0,
    molar_mass=44.01,
)
