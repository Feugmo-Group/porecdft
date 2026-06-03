"""Pore-volume probe and generic binding-site finder.

Two utilities:
- `probe_pore_volume`: given an external potential on a 3D grid and a probe energy,
  return the integrated accessible volume V_pore = ∫ exp(-β V_He) dV.
- `find_local_minima`: scan a Vext grid for local minima (candidate binding sites)
  using a 3×3×3 maximum-comparison filter.
"""

from __future__ import annotations

import numpy as np


def probe_pore_volume(vext_grid: np.ndarray, dV: float, beta: float) -> float:
    """Boltzmann-weighted accessible volume.

    V_pore = ∫ exp(-β V(r)) dV ≈ Σ exp(-β V_i) dV_i

    Parameters
    ----------
    vext_grid : ndarray
        External potential evaluated on a regular 3D grid, energy units consistent
        with `beta`. NaNs / +inf are treated as inaccessible (e^{-β·inf} = 0).
    dV : float
        Volume per grid cell (Å³).
    beta : float
        1/(k_B T) in inverse-energy units matching `vext_grid`.

    Returns
    -------
    float
        Accessible volume in the same Å³ units as `dV`.
    """
    boltz = np.exp(-beta * vext_grid)
    boltz = np.where(np.isfinite(boltz), boltz, 0.0)
    return float(boltz.sum() * dV)


def find_local_minima(vext_grid: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Return integer indices of local minima in `vext_grid`.

    A local minimum is a voxel strictly lower than all 26 neighbours. If
    `threshold` is given, only minima below that value are returned.

    Returns an (M, 3) array of (i, j, k) indices. Useful as an automated way to
    locate candidate binding sites (small cavity, large cavity) without hand-coding
    fractional coordinates.
    """
    g = vext_grid
    nx, ny, nz = g.shape
    mask = np.ones_like(g, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == dj == dk == 0:
                    continue
                shifted = np.roll(g, shift=(di, dj, dk), axis=(0, 1, 2))
                mask &= g < shifted
    if threshold is not None:
        mask &= g < threshold
    return np.argwhere(mask)
