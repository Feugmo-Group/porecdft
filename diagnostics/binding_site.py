"""Binding-site probe.

Place a fluid molecule at a given lab-frame position, sweep over orientations,
return:
- the minimum energy and the orientation that achieved it,
- the Boltzmann-averaged energy at a reference temperature,
- the per-component decomposition (LJ / Coulomb / Quad / ...) at the minimum,
- the full per-orientation energy array (for rose plots and histograms).

This is the workhorse for validating Vext against the DFT binding energies
−18.4 kJ/mol (SC) and −8.1 kJ/mol (LC) from Evans et al. — Phase 1.3 of the plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from porecdft.fluid.base import Fluid
from porecdft.forcefield.base import Potential
from porecdft.structure.host import HostAtoms

K_TO_KJ_PER_MOL = 8.314462618e-3   # 1 K * k_B in kJ/mol


@dataclass(frozen=True)
class BindingSiteResult:
    site_label: str
    r_center: np.ndarray
    energies_K: np.ndarray            # (Norient,) per-orientation total energy in K
    parts_at_min: dict[str, float]    # decomposition at the lowest-energy orientation, K
    argmin: int
    directions: np.ndarray            # (Norient, 3) molecular axis direction in lab frame
    temperature_K: float | None

    @property
    def E_min_K(self) -> float:
        return float(self.energies_K.min())

    @property
    def E_min_kJ_per_mol(self) -> float:
        return float(self.E_min_K * K_TO_KJ_PER_MOL)

    @property
    def E_mean_K(self) -> float:
        return float(self.energies_K.mean())

    @property
    def E_max_K(self) -> float:
        return float(self.energies_K.max())

    def boltzmann_average_K(self, T: float) -> float:
        beta = 1.0 / T
        E = self.energies_K
        E0 = E.min()
        w = np.exp(-beta * (E - E0))
        return float(E0 - T * np.log(w.mean()))


def probe_binding_site(
    host: HostAtoms,
    fluid: Fluid,
    potential: Potential,
    r_center: np.ndarray,
    rotations: np.ndarray,            # (Norient, 3, 3)
    site_label: str = "site",
    temperature_K: float | None = 298.15,
) -> BindingSiteResult:
    """Sweep orientations at a single position; return energy + decomposition."""
    r_center = np.asarray(r_center, dtype=float)
    Norient = len(rotations)
    energies = np.empty(Norient, dtype=float)
    parts_per_orient: list[dict[str, float]] = []
    for k, R in enumerate(rotations):
        e = potential.energy_at(r_center, R, host, fluid.body_sites, fluid.site_labels)
        energies[k] = e.total
        parts_per_orient.append(dict(e.parts or {"Total": e.total}))
    argmin = int(np.argmin(energies))
    directions = np.array([R @ np.array([0.0, 0.0, 1.0]) for R in rotations])
    return BindingSiteResult(
        site_label=site_label,
        r_center=r_center,
        energies_K=energies,
        parts_at_min=parts_per_orient[argmin],
        argmin=argmin,
        directions=directions,
        temperature_K=temperature_K,
    )
