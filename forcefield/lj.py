"""Lennard-Jones (12-6) potential.

Energy units: Kelvin (ε is the well depth in K, σ in Å).

Mixing rules supported:
- "lorentz-berthelot": σ_ij = (σ_i+σ_j)/2,  ε_ij = √(ε_i·ε_j)   (default)
- "waldman-hagler":     σ_ij = ((σ_i^6 + σ_j^6)/2)^(1/6),  ε_ij = √(ε_i·ε_j)·(σ_i^3·σ_j^3)/((σ_i^6+σ_j^6)/2)
- "kong":               more complex; supported for future MLIP comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from porecdft.forcefield.base import Potential, PotentialEnergy
from porecdft.io.forcefield import FFEntry
from porecdft.structure.host import HostAtoms


@dataclass(frozen=True)
class LJPotential(Potential):
    """LJ between fluid sites and host atoms.

    Parameters
    ----------
    host_ff : dict[str, FFEntry]
        Per-host-element LJ parameters.
    fluid_ff : dict[str, FFEntry]
        Per-fluid-site-label LJ parameters.
    cutoff : float
        Hard cutoff in Å beyond which the interaction is zero. Default 15 Å.
    mixing : str
        Combining rule. Default "lorentz-berthelot".
    exclude_species : frozenset[str] or None
        Host species to skip entirely — useful when a MorsePotential already
        handles those atoms (e.g. open metal sites in COFs) so that
        ``CompositePotential([morse, lj])`` does not double-count them.
    """
    host_ff: dict[str, FFEntry]
    fluid_ff: dict[str, FFEntry]
    cutoff: float = 15.0
    mixing: str = "lorentz-berthelot"
    exclude_species: frozenset | None = None
    epsilon_scale: float = 1.0   # multiply every ε_ij by this factor (e.g. 1.41 from original fit)
    name: str = "LJ"

    def _pair_params(self, host_el: str, fluid_label: str) -> tuple[float, float]:
        a = self.host_ff[host_el]
        b = self.fluid_ff[fluid_label]
        if self.mixing == "lorentz-berthelot":
            sigma = 0.5 * (a.sigma + b.sigma)
            epsilon = float(np.sqrt(a.epsilon * b.epsilon))
        elif self.mixing == "waldman-hagler":
            sigma = ((a.sigma**6 + b.sigma**6) / 2.0) ** (1.0 / 6.0)
            epsilon = float(
                np.sqrt(a.epsilon * b.epsilon)
                * (a.sigma**3 * b.sigma**3)
                / ((a.sigma**6 + b.sigma**6) / 2.0)
            )
        else:
            raise ValueError(f"Unknown mixing rule: {self.mixing}")
        return sigma, epsilon * self.epsilon_scale

    def energy_at(self, r_center, rot, host, fluid_sites, fluid_site_labels) -> PotentialEnergy:
        r_center = np.asarray(r_center)
        sites_lab = r_center + fluid_sites @ rot.T  # (S, 3)
        total = 0.0
        exc = self.exclude_species or frozenset()
        for s_idx, label in enumerate(fluid_site_labels):
            if label not in self.fluid_ff:
                continue          # charge-only site (e.g. TraPPE N₂ central 'M')
            site_pos = sites_lab[s_idx]
            for h_idx, h_el in enumerate(host.species):
                if h_el in exc:
                    continue
                dr = host.positions[h_idx] - site_pos
                # NOTE: caller is responsible for replicating host across PBC if needed.
                # See `vext/builder.build_vext_on_grid` which handles PBC by supercell.
                r2 = float(dr @ dr)
                if r2 > self.cutoff * self.cutoff or r2 == 0.0:
                    continue
                sigma, epsilon = self._pair_params(h_el, label)
                sr6 = (sigma * sigma / r2) ** 3
                total += 4.0 * epsilon * (sr6 * sr6 - sr6)
        return PotentialEnergy(total=total, parts={"LJ": total})

    def energy_grid(self, grid_xyz, rot, host, fluid_sites, fluid_site_labels, use_warp):
        if use_warp: # cal warp subroutine
            from porecdft.warp_backend import lj_vext_grid_warp
            # prepare tensor feed inte cuda kernel
            host_elements = host.species
            S, Na = len(fluid_site_labels), len(host_elements)
            sigma_ij = np.zeros((S, Na))
            epsilon_ij = np.zeros((S, Na))
            active = np.zeros((S, Na))            
            for s_idx, label in enumerate(fluid_site_labels):
                for h_idx, h_el in enumerate(host_elements):
                    active[s_idx, h_idx] = label in self.fluid_ff
                    if label not in self.fluid_ff:
                        continue          # charge-only site (e.g. TraPPE N₂ central 'M')
                    sigma, epsilon = self._pair_params(h_el, label)
                    sigma_ij[s_idx, h_idx] = sigma
                    epsilon_ij[s_idx, h_idx] = epsilon
            site_offset = fluid_sites @ rot.T

            return lj_vext_grid_warp(grid_xyz, 
                                     site_offset, 
                                     host.positions, 
                                     sigma_ij,
                                     epsilon_ij,
                                     active,
                                     self.cutoff
                                     )
        else: # numpy implementation
            # Vectorized over grid + sites + host atoms in one shot.
            grid = np.asarray(grid_xyz)               # (Ng, 3)
            sites_lab = grid[:, None, :] + (fluid_sites @ rot.T)[None, :, :]  # (Ng, S, 3)
            host_pos = host.positions                  # (Na, 3)
            host_elements = host.species
            total = np.zeros(len(grid), dtype=float)
            cutoff2 = self.cutoff * self.cutoff
            exc = self.exclude_species or frozenset()
            for s_idx, label in enumerate(fluid_site_labels):
                if label not in self.fluid_ff:
                    continue          # charge-only site (e.g. TraPPE N₂ central 'M')
                site = sites_lab[:, s_idx, :]          # (Ng, 3)
                dr = host_pos[None, :, :] - site[:, None, :]   # (Ng, Na, 3)
                r2 = np.einsum("gad,gad->ga", dr, dr)          # (Ng, Na)
                mask = (r2 < cutoff2) & (r2 > 0.0)
                for h_idx, h_el in enumerate(host_elements):
                    if h_el in exc:
                        continue
                    pair_mask = mask[:, h_idx]
                    if not np.any(pair_mask):
                        continue
                    sigma, epsilon = self._pair_params(h_el, label)
                    r2_i = r2[pair_mask, h_idx]
                    sr6 = (sigma * sigma / r2_i) ** 3
                    total[pair_mask] += 4.0 * epsilon * (sr6 * sr6 - sr6)
        return total
