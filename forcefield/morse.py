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
    cutoff: float    # Å


@dataclass(frozen=True)
class MorsePotential(Potential):
    """Morse between fluid sites and host atoms.

    Parameters
    ----------
    include_species : frozenset[str] or None
        If given, only host atoms of these species are evaluated.
        Use this to restrict Morse to open metal sites when pairing
        with an LJPotential that handles the organic backbone:
        ``LJPotential(..., exclude_species=frozenset({"Zn", "Ni"}))``
    """
    host_params: dict[str, MorseParam]
    fluid_params: dict[str, MorseParam] | None = None
    include_species: frozenset | None = None
    name: str = "Morse"
    cutoff: float = 15.0

    def _pair(self, host_el: str, fluid_label: str) -> tuple[float, float, float]:
        a = self.host_params[host_el]
        if self.fluid_params is not None: 
            b = self.fluid_params[fluid_label]
            D_e = float(np.sqrt(a["D_e"] * b["D_e"]))
            alpha = 0.5 * (a["a"] + b["a"])
            r_e = 0.5 * (a.r_e + b["r_e"])
        else: # in the case that not gas morse parameters, just use metal site parameters
            D_e, alpha, r_e = a["D_e"], a["a"], a["r_e"]
        return D_e, alpha, r_e

    def energy_at(self, r_center, rot, host, fluid_sites, fluid_site_labels) -> PotentialEnergy:
        r_center = np.asarray(r_center)
        sites_lab = r_center + fluid_sites @ rot.T
        total = 0.0
        inc = self.include_species
        cutoff = self.cutoff
        for s_idx, label in enumerate(fluid_site_labels):
            site_pos = sites_lab[s_idx]
            for h_idx, h_el in enumerate(host.species):
                if inc is not None and h_el not in inc:
                    continue
                if h_el not in self.host_params:
                    continue
                dr = host.positions[h_idx] - site_pos
                r2 = float(dr @ dr)
                D_e, alpha, r_e = self._pair(h_el, label)
                if r2 > cutoff * cutoff or r2 == 0.0:
                    continue
                r = float(np.sqrt(r2))

                x = np.exp(-alpha * (r - r_e))
                v_pair = D_e * (x * x - 2.0 * x)
                # Clamp to physical range
                if v_pair < -D_e:
                    v_pair = -D_e
                total += v_pair
        return PotentialEnergy(total=total, parts={"Morse": total})
    
    def energy_grid(self, grid_xyz, rot, host, fluid_sites, fluid_site_labels, use_warp):
        """overwrite based object implementation, include options that calls warp kernel"""
        if use_warp:
            # call warp kernel to build morse potential
            from porecdft.warp_backend import morse_vext_grid_warp

            # prepare tensors feed into cuda kernel
            host_elements = host.species
            exc = not self.include_species or frozenset()
            S, Na = len(fluid_site_labels), len(host_elements)

            site_offset = fluid_sites @ rot.T
            host_pos = host.positions
            cutoff = self.cutoff            
            # loop over each site of the fluid
            energy = np.zeros(grid_xyz.shape[0], dtype=float)
            for s_idx, label in enumerate(fluid_site_labels):
                D_e, alpha_w, r_e = np.zeros(Na), np.zeros(Na), np.zeros(Na)
                active = np.ones(Na)
                for h_idx, h_el in enumerate(host_elements):
                    if exc is not None or hel not in self.host_params:
                        active[h_idx] = 0 
                        continue 
                    D_e[h_idx], alpha_w[h_idx], r_e[h_idx] = self._pair(h_el, label)

                energy += morse_vext_grid_warp(grid_xyz, site_offset[s_idx], host_pos, D_e, alpha_w, r_e, active, cutoff)
            
            return energy 
        
        else: # numpy implementation (same as base)
            out = np.empty(len(grid_xyz), dtype=float)
            for i, r in enumerate(grid_xyz):
                out[i] = self.energy_at(r, rot, host, fluid_sites, fluid_site_labels).total
            return out

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
