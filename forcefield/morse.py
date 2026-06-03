"""Morse potential.

V(r) = D_e [ (1 - exp(-a(r - r_e)))^2 - 1 ]
     = D_e [exp(-2a(r-r_e)) - 2 exp(-a(r-r_e))]

Per-element parameters: D_e (well depth, K), a (width parameter, 1/Å), r_e
(equilibrium distance, Å). Combining rules for cross-pair Morse are not
universally agreed upon — we default to geometric mean for D_e and
arithmetic mean for a and r_e, which is the most common convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from porecdft.forcefield.base import Potential, PotentialEnergy


@dataclass(frozen=True)
class MorseParam:
    element: str
    D_e: float       # K
    a: float         # 1/Å
    r_e: float       # Å


@dataclass(frozen=True)
class MorsePotential(Potential):
    host_params: dict[str, MorseParam]
    fluid_params: dict[str, MorseParam]
    cutoff: float = 15.0
    name: str = "Morse"

    def _pair(self, host_el: str, fluid_label: str) -> tuple[float, float, float]:
        a = self.host_params[host_el]
        b = self.fluid_params[fluid_label]
        D_e = float(np.sqrt(a.D_e * b.D_e))
        alpha = 0.5 * (a.a + b.a)
        r_e = 0.5 * (a.r_e + b.r_e)
        return D_e, alpha, r_e

    def energy_at(self, r_center, rot, host, fluid_sites, fluid_site_labels) -> PotentialEnergy:
        r_center = np.asarray(r_center)
        sites_lab = r_center + fluid_sites @ rot.T
        total = 0.0
        for s_idx, label in enumerate(fluid_site_labels):
            site_pos = sites_lab[s_idx]
            for h_idx, h_el in enumerate(host.species):
                dr = host.positions[h_idx] - site_pos
                r2 = float(dr @ dr)
                if r2 > self.cutoff * self.cutoff or r2 == 0.0:
                    continue
                r = float(np.sqrt(r2))
                D_e, alpha, r_e = self._pair(h_el, label)
                x = np.exp(-alpha * (r - r_e))
                total += D_e * (x * x - 2.0 * x)
        return PotentialEnergy(total=total, parts={"Morse": total})


@dataclass(frozen=True)
class MorseScalarPotential:
    """Simple scalar Morse potential V(r) for benchmarking and 1-D profiles.

    V(r) = D_e_K * [(1 - exp(-a_invA * (r - r_e_A)))^2 - 1]

    So V(r_e) = -D_e_K and V -> 0 as r -> infinity.

    Parameters
    ----------
    D_e_K : float
        Well depth in Kelvin.
    r_e_A : float
        Equilibrium distance in Angstroms.
    a_invA : float
        Width (stiffness) parameter in Angstrom^-1.
    """

    D_e_K: float    # well depth in K
    r_e_A: float    # equilibrium distance in Å
    a_invA: float   # width in Å^-1

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        x = np.exp(-self.a_invA * (r - self.r_e_A))
        return self.D_e_K * (1.0 - x) ** 2 - self.D_e_K
