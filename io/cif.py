"""CIF reader via pymatgen.

Produces a HostAtoms object: positions in Å (Cartesian), element symbols, lattice.
Partial charges are NOT read here — they come from a separate `charges` file because
in practice you almost always want to override CIF charges with DDEC6 / REPEAT values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pymatgen.io.cif import CifParser

from porecdft.structure.host import HostAtoms


def read_cif(path: str | Path) -> HostAtoms:
    """Load a CIF and return a HostAtoms (Cartesian Å, no charges yet).

    Parameters
    ----------
    path : str or Path
        Path to the CIF file.

    Returns
    -------
    HostAtoms
        positions: (N, 3) ndarray in Å, species: list[str], lattice: (3, 3) ndarray.
        charges are zero-initialized — assign them separately via `read_charges_csv`
        or by hand.
    """
    structure = CifParser(str(path)).parse_structures(primitive=False)[0]
    positions = np.array(structure.cart_coords, dtype=float)
    species = [str(site.specie) for site in structure.sites]
    lattice = np.array(structure.lattice.matrix, dtype=float)
    charges = np.zeros(len(species), dtype=float)
    return HostAtoms(
        positions=positions,
        species=species,
        charges=charges,
        lattice=lattice,
        source=str(path),
    )
