"""Henry constant from an external-potential grid.

For an ideal fluid at infinite dilution in a rigid framework,

    K_H = (1 / V_pore) ∫_pore exp(-β ⟨V_ext(r)⟩_orient) dV,

with V_pore the He-probe accessible volume. Both functionals (LJ-cDFT and
PC-SAFT cDFT) must reproduce this analytical value at low pressure — see
Phase 1.5 of the plan.
"""

from __future__ import annotations

import numpy as np


def henry_constant_from_vext(
    vext_grid: np.ndarray,
    dV: float,
    temperature_K: float,
    pore_volume: float | None = None,
) -> float:
    """Compute K_H = (1/V_pore) ∫ exp(-βV) dV.

    Parameters
    ----------
    vext_grid : ndarray
        Orientation-averaged Vext on a 3D grid in units of K.
    dV : float
        Volume per voxel in Å³.
    temperature_K : float
        Temperature in K.
    pore_volume : float, optional
        He-probe pore volume in Å³. If None, the integral is returned without
        normalisation (i.e. integrated Boltzmann weight in Å³).
    """
    beta = 1.0 / temperature_K
    boltz = np.exp(-beta * vext_grid)
    boltz = np.where(np.isfinite(boltz), boltz, 0.0)
    integral = float(boltz.sum() * dV)
    if pore_volume is None:
        return integral
    return integral / pore_volume
