"""Potential abstract base class.

A Potential computes the energy of a single fluid molecule placed at a given
position (and orientation) in the field of a fixed host. Concrete subclasses cover
analytic forms (LJ, Morse, Coulomb, quadrupole) and ML interatomic potentials.

Energy units are Kelvin throughout (Boltzmann units, ε/k_B convention) — convert
to kJ/mol only at the reporting boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from porecdft.structure.host import HostAtoms


@dataclass(frozen=True)
class PotentialEnergy:
    """Container returned by ``Potential.energy``.

    The decomposition `parts` is optional but recommended — it lets the diagnostic
    layer break down a binding-site energy into LJ / Coulomb / quadrupole / ML
    contributions for the validation bar charts in Phase 1.3.
    """
    total: float                       # K
    parts: dict[str, float] | None = None  # {"LJ": ..., "Coul": ..., ...}


class Potential(ABC):
    """Energy of a fluid molecule in the field of a fixed host.

    The fluid is represented by an iterable of (site_offset, site_label) pairs
    that the potential is free to interpret. For a single-site LJ fluid, this is
    a single (0, 0, 0) offset with a generic label; for EPM2 CO₂ it is three
    offsets and labels in the body frame.

    Subclasses implement either `energy_at` (one position + one orientation) or
    `energy_grid` (vectorized over an Å grid). The default `energy_grid` is a
    vmap over `energy_at`; analytic subclasses should override it for speed.
    """

    name: str = "Potential"

    @abstractmethod
    def energy_at(
        self,
        r_center: np.ndarray,         # (3,) fluid centre in Å (Cartesian)
        rot: np.ndarray,              # (3, 3) rotation matrix, body→lab frame
        host: HostAtoms,
        fluid_sites: np.ndarray,      # (S, 3) body-frame site offsets in Å
        fluid_site_labels: list[str], # length-S labels (e.g. "C", "O" for EPM2)
    ) -> PotentialEnergy:
        ...

    def energy_grid(
        self,
        grid_xyz: np.ndarray,         # (Ng, 3) lab-frame grid points
        rot: np.ndarray,
        host: HostAtoms,
        fluid_sites: np.ndarray,
        fluid_site_labels: list[str],
    ) -> np.ndarray:
        """Vectorized energy at each grid point for a single orientation.

        Default implementation is a Python loop over `energy_at` (correct but slow).
        Subclasses should override for vectorized speed.
        """
        out = np.empty(len(grid_xyz), dtype=float)
        for i, r in enumerate(grid_xyz):
            out[i] = self.energy_at(r, rot, host, fluid_sites, fluid_site_labels).total
        return out
