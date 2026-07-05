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
    """Morse between fluid sites and host atoms.

    Parameters
    ----------
    include_species : frozenset[str] or None
        If given, only host atoms of these species are evaluated.
        Use this to restrict Morse to open metal sites when pairing
        with an LJPotential that handles the organic backbone:
        ``LJPotential(..., exclude_species=frozenset({"Zn", "Ni"}))``
    """
    host_params: dict[str, "MorseParam | dict"]
    fluid_params: dict[str, "MorseParam | dict"] | None = None
    cutoff: float = 15.0
    include_species: frozenset | None = None
    name: str = "Morse"

    @staticmethod
    def _get(p, key: str) -> float:
        """Read a param field from either a MorseParam dataclass or a dict."""
        return float(getattr(p, key)) if hasattr(p, key) else float(p[key])

    def _pair(self, host_el: str, fluid_label: str) -> tuple[float, float, float]:
        a = self.host_params[host_el]
        # If no fluid params are given, treat host entries as direct pair parameters
        # (e.g. Sun-style tabulated metal–H2 Morse from the literature).
        if self.fluid_params is None or fluid_label not in self.fluid_params:
            return self._get(a, "D_e"), self._get(a, "a"), self._get(a, "r_e")
        b = self.fluid_params[fluid_label]
        D_e   = float(np.sqrt(self._get(a, "D_e") * self._get(b, "D_e")))
        alpha = 0.5 * (self._get(a, "a")   + self._get(b, "a"))
        r_e   = 0.5 * (self._get(a, "r_e") + self._get(b, "r_e"))
        return D_e, alpha, r_e

    def energy_at(self, r_center, rot, host, fluid_sites, fluid_site_labels) -> PotentialEnergy:
        r_center = np.asarray(r_center)
        sites_lab = r_center + fluid_sites @ rot.T
        total = 0.0
        inc = self.include_species
        for s_idx, label in enumerate(fluid_site_labels):
            site_pos = sites_lab[s_idx]
            for h_idx, h_el in enumerate(host.species):
                if inc is not None and h_el not in inc:
                    continue
                if h_el not in self.host_params:
                    continue
                dr = host.positions[h_idx] - site_pos
                r2 = float(dr @ dr)
                if r2 > self.cutoff * self.cutoff or r2 == 0.0:
                    continue
                r = float(np.sqrt(r2))
                D_e, alpha, r_e = self._pair(h_el, label)
                x = np.exp(-alpha * (r - r_e))
                total += D_e * (x * x - 2.0 * x)
        return PotentialEnergy(total=total, parts={"Morse": total})

    def energy_grid(self, grid_xyz, rot, host, fluid_sites, fluid_site_labels, use_warp=False):
        """Vectorized Morse V_ext over a 3D grid.

        Mirrors the CPU logic of ``energy_at`` exactly. When ``use_warp=True``
        and a Warp kernel is available, the per-atom evaluation is dispatched
        to the GPU (falls back to Warp's CPU device when CUDA is missing).
        """
        host_elements = host.species
        inc = self.include_species
        Ng = grid_xyz.shape[0]

        if use_warp:
            from porecdft.warp_backend import morse_vext_grid_warp

            Na = len(host_elements)
            site_offsets = fluid_sites @ rot.T   # (S, 3) — one row per fluid site
            out = np.zeros(Ng, dtype=np.float32)

            for s_idx, label in enumerate(fluid_site_labels):
                # Per-atom mixed pair parameters and active mask.
                # Same filter as CPU path: skip atoms outside include_species
                # OR without an entry in host_params (e.g. LJ-only elements).
                D_e_arr   = np.zeros(Na, dtype=np.float32)
                alpha_arr = np.zeros(Na, dtype=np.float32)
                r_e_arr   = np.zeros(Na, dtype=np.float32)
                active    = np.zeros(Na, dtype=np.int32)
                for h_idx, h_el in enumerate(host_elements):
                    if inc is not None and h_el not in inc:
                        continue
                    if h_el not in self.host_params:
                        continue
                    D_e_arr[h_idx], alpha_arr[h_idx], r_e_arr[h_idx] = self._pair(h_el, label)
                    active[h_idx] = 1

                out = morse_vext_grid_warp(
                    grid_xyz, site_offsets[s_idx], host.positions,
                    D_e_arr, alpha_arr, r_e_arr, active,
                    self.cutoff, out=out,
                )
            return out.astype(float)

        # numpy path — vectorized version of energy_at
        grid = np.asarray(grid_xyz)
        sites_lab = grid[:, None, :] + (fluid_sites @ rot.T)[None, :, :]   # (Ng, S, 3)
        host_pos = host.positions
        total = np.zeros(Ng, dtype=float)
        cutoff2 = self.cutoff * self.cutoff
        for s_idx, label in enumerate(fluid_site_labels):
            site = sites_lab[:, s_idx, :]                              # (Ng, 3)
            dr = host_pos[None, :, :] - site[:, None, :]               # (Ng, Na, 3)
            r2 = np.einsum("gad,gad->ga", dr, dr)                      # (Ng, Na)
            for h_idx, h_el in enumerate(host_elements):
                if inc is not None and h_el not in inc:
                    continue
                if h_el not in self.host_params:
                    continue
                pair_mask = (r2[:, h_idx] < cutoff2) & (r2[:, h_idx] > 0.0)
                if not np.any(pair_mask):
                    continue
                D_e, alpha, r_e = self._pair(h_el, label)
                r = np.sqrt(r2[pair_mask, h_idx])
                x = np.exp(-alpha * (r - r_e))
                total[pair_mask] += D_e * (x * x - 2.0 * x)
        return total


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
