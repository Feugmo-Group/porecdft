"""Henry constant from an external-potential grid.

For an ideal fluid at infinite dilution in a rigid framework the
analytical Henry constant per unit cell is (see SI Eq. S25)

    K_H = (β / V_cell) ∫_cell exp(-β ⟨V_ext(r)⟩_orient) dV
        = 1/(k_B T V_cell) ∫ exp(-β V_ext) dV.

The explicit β = 1/(k_B T) prefactor comes from the ideal-gas relation
ρ_bulk = P/(k_B T) and is required for K_H to have units of inverse
pressure (so that downstream conversion to mmol·g⁻¹·bar⁻¹ is correct).

Both functionals (LJ-cDFT and PC-SAFT cDFT) must reproduce this
analytical value at low pressure — see Phase 1.5 of the plan.

If ``pore_volume`` is given as a plain geometric value the *ratio*
form K_H^ratio = ∫ exp(-βV) dV / V_pore is returned instead (no β
prefactor); this is the dimensionless adsorbed/bulk equilibrium
constant used in some texts.  The default with ``pore_volume=None``
returns the bare Boltzmann integral so the caller can apply the
prefactor it needs.
"""

from __future__ import annotations

import numpy as np


def henry_constant_from_vext(
    vext_grid: np.ndarray,
    dV: float,
    temperature_K: float,
    pore_volume: float | None = None,
) -> float:
    """Compute the Boltzmann pore integral I = ∫ exp(-βV) dV (Å³).

    See module docstring — physical K_H = β I / V_cell.

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
