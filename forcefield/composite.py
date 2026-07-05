"""Sum of multiple Potential contributions with per-component breakdown.

`CompositePotential([LJ, Coulomb, Quad])` returns a PotentialEnergy whose
`parts` dict contains every named sub-contribution, which is exactly what the
diagnostic bar charts in Phase 1.3 need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from porecdft.forcefield.base import Potential, PotentialEnergy


@dataclass(frozen=True)
class CompositePotential(Potential):
    components: tuple[Potential, ...]
    name: str = "Composite"

    def __init__(self, components: Sequence[Potential]):
        object.__setattr__(self, "components", tuple(components))
        object.__setattr__(self, "name", "+".join(c.name for c in components))

    def energy_at(self, r_center, rot, host, fluid_sites, fluid_site_labels) -> PotentialEnergy:
        parts: dict[str, float] = {}
        total = 0.0
        for comp in self.components:
            e = comp.energy_at(r_center, rot, host, fluid_sites, fluid_site_labels)
            total += e.total
            # Merge parts; if a name collides (two LJ terms), suffix with index.
            for k, v in (e.parts or {}).items():
                key = k
                idx = 2
                while key in parts:
                    key = f"{k}#{idx}"
                    idx += 1
                parts[key] = v
        return PotentialEnergy(total=total, parts=parts)

    def energy_grid(self, grid_xyz, rot, host, fluid_sites, fluid_site_labels, use_warp=False):
        out = np.zeros(len(grid_xyz), dtype=float)
        for comp in self.components:
            if comp.name == "QuadrupoleEFG":
                # No Warp implementation of QuadrupoleEFG yet -- fall back to NumPy path
                out = out + comp.energy_grid(grid_xyz, rot, host, fluid_sites, fluid_site_labels)
            else:
                out = out + comp.energy_grid(grid_xyz, rot, host, fluid_sites, fluid_site_labels, use_warp)
        return out
