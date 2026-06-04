"""Hydrogen fluid models.

SingleSite_H2: one LJ centre, no charges. Used with a Morse external potential
for H₂ adsorption at open metal sites in COFs (Pramudya & Mendoza-Cortes 2016).
σ = 2.96 Å, ε/k_B = 34.2 K  (Buch 1994 / commonly cited for physisorption).

For electrostatic models of H₂ (quadrupole + point charges) a three-site
representation analogous to TraPPE-N2 should be used; that model is not
included here because the Morse-only benchmark does not require it.
"""

from __future__ import annotations

import numpy as np

from porecdft.io.forcefield import FFEntry
from porecdft.fluid.base import Fluid

# Buch 1994 single-site H₂  (used for physisorption / MOF benchmarks)
SingleSite_H2 = Fluid(
    name="H2-SingleSite",
    body_sites=np.array([[0.0, 0.0, 0.0]]),
    site_labels=["H2"],
    ff={"H2": FFEntry("H2", 2.96, 34.2, "Buch1994")},
    charges={"H2": 0.0},
    theta_zz=0.0,
    molar_mass=2.016,
)
