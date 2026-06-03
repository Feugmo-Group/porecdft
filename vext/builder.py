"""Build the orientation-averaged external potential on a 3D grid.

The workflow is:

    1. `build_grid(host, spacing)` returns an (Nx, Ny, Nz, 3) array of lab-frame
       Cartesian grid points covering the host unit cell.
    2. The host atom positions are replicated by a small supercell (typically
       2×2×2 with appropriate centring) so that LJ + Coulomb cutoffs are
       satisfied without explicit Ewald.
    3. For each rotation in `orientations`, evaluate Vext_total(r) and accumulate
       both the Boltzmann-averaged Vext and the per-orientation min.
    4. Cache to `.npy` next to the application's data directory.

This module is potential-agnostic — it receives a `Potential` instance, which
can be any analytic or ML adapter. The grid build is exactly the same.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from porecdft.fluid.base import Fluid
from porecdft.forcefield.base import Potential
from porecdft.structure.host import HostAtoms
from porecdft.structure.supercell import build_supercell


def build_grid(host: HostAtoms, spacing: float) -> tuple[np.ndarray, tuple[int, int, int], float]:
    """Build a regular 3D grid over the host unit cell.

    Returns
    -------
    grid_xyz : ndarray
        (Nx*Ny*Nz, 3) Cartesian Å positions, C-order along (i, j, k).
    shape : (Nx, Ny, Nz)
        Number of points along each lattice direction.
    dV : float
        Volume per voxel in Å³.

    The grid covers fractional coordinates [0, 1) along each axis.
    """
    L = host.lattice
    # Approximate edge lengths along a, b, c (for orthorhombic this is exact)
    lengths = np.linalg.norm(L, axis=1)
    n = tuple(max(2, int(np.ceil(l / spacing))) for l in lengths)
    fx = np.linspace(0.0, 1.0, n[0], endpoint=False)
    fy = np.linspace(0.0, 1.0, n[1], endpoint=False)
    fz = np.linspace(0.0, 1.0, n[2], endpoint=False)
    Fx, Fy, Fz = np.meshgrid(fx, fy, fz, indexing="ij")
    frac = np.stack([Fx, Fy, Fz], axis=-1).reshape(-1, 3)
    cart = frac @ L
    dV = host.cell_volume / np.prod(n)
    return cart, n, dV


def build_vext_on_grid(
    host: HostAtoms,
    fluid: Fluid,
    potential: Potential,
    orientations: np.ndarray,                # (Norient, 3, 3) rotation matrices
    spacing: float = 0.5,
    pbc_supercell: tuple[int, int, int] = (3, 3, 3),
    centre_supercell: bool = True,
    temperature_K: float | None = None,      # for Boltzmann averaging; if None, returns min
    cache_path: str | Path | None = None,
    v_clip_per_orient_K: float | None = None,        # legacy cap, kept for back-compat
    v_reject_below_K: float | None = -10000.0,        # reject (orient, voxel) pairs with V < this
) -> dict:
    """Compute Vext on a 3D grid, averaged or minimised over orientations.

    Parameters
    ----------
    host : HostAtoms
        The framework. Cartesian Å, with charges already assigned.
    fluid : Fluid
        Adsorbate (used for body_sites and site_labels — the `potential` already
        contains the FF parameters and charges that match `fluid.site_labels`).
    potential : Potential
        Pluggable: any subclass (LJ, Coulomb, Quadrupole, Composite, MLIP).
    orientations : ndarray
        (Norient, 3, 3) rotation matrices, e.g. from `fibonacci_rotations`.
    spacing : float
        Grid spacing in Å. Default 0.5 Å.
    pbc_supercell : tuple[int, int, int]
        Replicate the host this many times along (a, b, c) so the LJ/Coulomb
        cutoff fits inside without Ewald. Default 3×3×3.
    centre_supercell : bool
        If True, shift the supercell so the original cell is in the middle.
        Required so that grid points in the original cell see periodic images
        on all sides within the cutoff.
    temperature_K : float, optional
        If given, return the Boltzmann-averaged Vext at each grid point:
        V_avg(r) = -k_B T ln <exp(-βV(r,Ω))>_Ω.
        If None, return the minimum over orientations (useful for binding-site
        characterisation).
    cache_path : str or Path, optional
        If given and the file exists, load it instead of recomputing. After
        computation, save the result there.

    Returns
    -------
    dict
        Keys: 'vext_avg' (or 'vext_min'), 'grid_shape', 'dV', 'orient_min',
        'orient_argmin'. The arrays have shape `grid_shape`.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True).item()
            return data

    # Build replicated host once
    n_a, n_b, n_c = pbc_supercell
    host_super = build_supercell(host, n_a, n_b, n_c)
    if centre_supercell:
        shift = -(n_a // 2) * host.lattice[0] - (n_b // 2) * host.lattice[1] - (n_c // 2) * host.lattice[2]
        host_super = type(host_super)(
            positions=host_super.positions + shift,
            species=host_super.species,
            charges=host_super.charges,
            lattice=host_super.lattice,
            source=host_super.source,
            charge_source=host_super.charge_source,
        )

    grid_xyz, shape, dV = build_grid(host, spacing)
    Norient = len(orientations)
    Ng = grid_xyz.shape[0]

    # Compute Vext(r, Ω) — memory-friendly: loop over orientations, vector over grid.
    import sys, time
    all_v = np.empty((Norient, Ng), dtype=float)
    print(f"    Vext build: grid {shape} ({Ng} pts), {host_super.n_atoms} host atoms, "
          f"{Norient} orientations", flush=True)
    t0 = time.time()
    for k, rot in enumerate(orientations):
        v_k = potential.energy_grid(
            grid_xyz, rot, host_super, fluid.body_sites, fluid.site_labels
        )
        # Reject (orient, voxel) pairs with unphysically deep V — they come from
        # rigid EPM2 sites accidentally landing on top of framework atoms in
        # specific orientations (point-charge Coulomb singularity, LJ tail). The
        # threshold ~ -83 kJ/mol is safely below any physical CO2-MOF binding
        # but well above the artefactual −10²–10⁴ kJ/mol from contact orientations.
        if v_reject_below_K is not None:
            v_k = np.where(v_k < v_reject_below_K, +np.inf, v_k)
        # Optional legacy cap
        if v_clip_per_orient_K is not None:
            v_k = np.maximum(v_k, v_clip_per_orient_K)
        all_v[k] = v_k
        if (k + 1) == 1 or (k + 1) % max(1, Norient // 5) == 0 or (k + 1) == Norient:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (Norient - k - 1)
            print(f"      orient {k+1}/{Norient}  "
                  f"({elapsed:.1f}s elapsed, ETA {eta:.1f}s)", flush=True)

    orient_min = all_v.min(axis=0).reshape(shape)
    orient_argmin = all_v.argmin(axis=0).reshape(shape)

    out: dict = {
        "grid_shape": shape,
        "dV": dV,
        "orient_min": orient_min,
        "orient_argmin": orient_argmin,
    }
    if temperature_K is not None:
        beta = 1.0 / temperature_K           # K → 1/K (energy already in K)
        # If every orientation at a voxel was rejected (set to +inf), the voxel
        # is inaccessible — record V_avg = +inf and short-circuit the formula.
        v_min_grid = all_v.min(axis=0)
        all_inf = ~np.isfinite(v_min_grid)
        # Stable Boltzmann average over the FINITE orientations only.
        finite_mask = np.isfinite(all_v)
        shifted = np.where(finite_mask, all_v - np.where(all_inf, 0.0, v_min_grid)[None, :], +1e30)
        boltz = np.exp(-beta * shifted)                  # 0 where rejected
        n_valid = finite_mask.sum(axis=0).astype(float)
        mean_boltz = np.where(n_valid > 0, boltz.sum(axis=0) / np.maximum(n_valid, 1.0), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            v_avg = np.where(
                all_inf,
                +np.inf,
                v_min_grid - temperature_K * np.log(np.maximum(mean_boltz, 1e-300)),
            )
        out["vext_avg"] = v_avg.reshape(shape)
        out["temperature_K"] = temperature_K
    else:
        out["vext_min"] = orient_min

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out, allow_pickle=True)
    return out
