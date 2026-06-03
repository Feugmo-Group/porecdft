"""Supercell construction and minimum-image distance utilities."""

from __future__ import annotations

import numpy as np

from porecdft.structure.host import HostAtoms


def build_supercell(host: HostAtoms, nx: int, ny: int, nz: int) -> HostAtoms:
    """Return a supercell replicated nx × ny × nz times along the lattice vectors.

    Charges and species are tiled accordingly; positions are translated by
    integer combinations of lattice vectors.
    """
    shifts = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                shifts.append(ix * host.lattice[0] + iy * host.lattice[1] + iz * host.lattice[2])
    shifts = np.array(shifts)  # (M, 3)
    positions = (host.positions[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    species = host.species * (nx * ny * nz)
    charges = np.tile(host.charges, nx * ny * nz)
    lattice = host.lattice * np.array([[nx], [ny], [nz]])
    return HostAtoms(
        positions=positions,
        species=species,
        charges=charges,
        lattice=lattice,
        source=f"{host.source} [supercell {nx}x{ny}x{nz}]",
        charge_source=host.charge_source,
    )


def minimum_image(dr: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Apply minimum-image convention to displacement vectors `dr` (..., 3) under
    an orthorhombic or general lattice. Returns the wrapped displacement.

    For a general lattice we go through fractional coordinates: dr_frac = dr @ L^-T,
    wrap each component to [-0.5, 0.5), then map back.
    """
    inv_lat_t = np.linalg.inv(lattice).T
    frac = dr @ inv_lat_t
    frac -= np.round(frac)
    return frac @ lattice.T
