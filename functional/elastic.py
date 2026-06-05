"""Elastic penalty for framework deformation (gate-opening / breathing MOFs).

Physical picture
----------------
When a flexible framework expands or contracts, the lattice constant changes
from its equilibrium value L_0 by a strain ε = (L - L_0) / L_0. The elastic
restoring free energy per unit cell is:

    F_elastic(ε) = ½ K_bulk V_0 ε²

where K_bulk is the bulk modulus (in Pa, converted internally to K/Å³) and
V_0 = L_0³ is the equilibrium cell volume.

cDFT workflow
-------------
For each trial strain ε in a discrete set {ε_i}, a separate Vext grid is
pre-computed (the framework coordinates are scaled by (1+ε)). The cDFT grand
potential Ω(ε) is minimised over ε at each pressure:

    Ω_total(ε) = Ω_fluid(ρ*(ε), ε) + F_elastic(ε)

where ρ*(ε) is the converged density field at strain ε.

This module provides:
  - ElasticPenalty: evaluates F_elastic(ε) in K (Boltzmann units)
  - scale_host: returns a new HostAtoms with lattice and positions scaled by (1+ε)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Conversion: 1 Pa·Å³ = 1e-30 J / k_B = 1e-30 / 1.380649e-23 K·Å³
# → 1 GPa·Å³ = 72.43 K
PA_A3_TO_K = 1e-30 / 1.380649e-23       # K per Pa·Å³
GPA_A3_TO_K = PA_A3_TO_K * 1e9          # K per GPa·Å³


@dataclass(frozen=True)
class ElasticPenalty:
    """Harmonic elastic penalty for isotropic cell deformation.

    Parameters
    ----------
    K_bulk_GPa : float
        Bulk modulus in GPa. For ALF: ~14 GPa (Evans et al. Sci. Adv. 2022).
    V0_A3 : float
        Equilibrium cell volume in Å³ (from the undeformed CIF).

    Usage
    -----
    >>> penalty = ElasticPenalty(K_bulk_GPa=14.0, V0_A3=host.cell_volume)
    >>> F_el = penalty.energy_K(strain=0.01)   # strain = (L-L0)/L0
    """
    K_bulk_GPa: float
    V0_A3: float

    def energy_K(self, strain: float) -> float:
        """F_elastic = ½ K_bulk V_0 ε²  in Kelvin."""
        return 0.5 * self.K_bulk_GPa * GPA_A3_TO_K * self.V0_A3 * strain ** 2

    def strain_grid(self, strains: np.ndarray) -> np.ndarray:
        """Evaluate F_elastic over an array of strains. Shape matches input."""
        return 0.5 * self.K_bulk_GPa * GPA_A3_TO_K * self.V0_A3 * np.asarray(strains) ** 2


def scale_host(host, strain: float):
    """Return a new HostAtoms with lattice and positions uniformly scaled by (1+ε).

    Parameters
    ----------
    host : HostAtoms
        Original (equilibrium) framework.
    strain : float
        Linear strain ε = (L - L₀) / L₀. Positive = expansion, negative = contraction.
        For ALF, the framework anomalously contracts under CO₂ loading: use negative ε.

    Returns
    -------
    HostAtoms
        Scaled copy. Fractional coordinates are preserved; Cartesian positions and
        lattice are multiplied by (1 + ε).
    """
    from porecdft.structure.host import HostAtoms
    from dataclasses import replace

    s = 1.0 + strain
    return replace(
        host,
        positions=host.positions * s,
        lattice=host.lattice * s,
    )
