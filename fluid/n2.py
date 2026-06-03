"""Nitrogen fluid models.

TraPPE-N2 (Potoff & Siepmann 2001): 3 sites — two LJ N atoms + central charge site.
  d_NN = 1.10 Å; q_N = -0.482 e at each N, q_COM = +0.964 e at the centre of mass.
  σ_N = 3.31 Å, ε_N/k_B = 36.0 K. LJ only on N sites (centre has none).
"""

from __future__ import annotations

import numpy as np

from porecdft.io.forcefield import FFEntry
from porecdft.fluid.base import Fluid

_TRAPPE_N2_FF = {"N": FFEntry("N", 3.31, 36.0, "TraPPE")}
_TRAPPE_N2_Q = {"N": -0.482, "M": +0.964}
_d = 1.10
TraPPE_N2 = Fluid(
    name="N2-TraPPE",
    body_sites=np.array([[0, 0, -_d / 2], [0, 0, 0], [0, 0, +_d / 2]]),
    site_labels=["N", "M", "N"],
    ff=_TRAPPE_N2_FF,           # only N has LJ; M is charge-only
    charges=_TRAPPE_N2_Q,
    theta_zz=2.0 * (-0.482) * (_d / 2) ** 2,
    molar_mass=28.014,
)
