"""CIF reader — pymatgen backend (default) or ASE backend.

Produces a HostAtoms: positions in Å (Cartesian), element symbols, lattice.
Partial charges are NOT read here; they come from a separate charges file
because in practice you almost always want DDEC6/REPEAT values.

Backend selection
-----------------
* ``backend="pymatgen"`` (default) — requires pymatgen.
* ``backend="ase"``               — requires ase; falls back gracefully
  when pymatgen is absent (e.g. lightweight deployments).
* ``backend="auto"``              — tries pymatgen first, then ase.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from porecdft.structure.host import HostAtoms


def _read_pymatgen(path: Path) -> HostAtoms:
    from pymatgen.io.cif import CifParser  # type: ignore[import]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        structure = CifParser(str(path)).parse_structures(primitive=False)[0]
    if w:
        warnings.warn(
            f"Issues encountered while parsing CIF: {w[-1].message}",
            stacklevel=4,
        )

    positions = np.array(structure.cart_coords, dtype=float)
    species = [str(site.specie) for site in structure.sites]
    lattice = np.array(structure.lattice.matrix, dtype=float)
    charges = np.zeros(len(species), dtype=float)
    return HostAtoms(positions=positions, species=species,
                     charges=charges, lattice=lattice, source=str(path))


def _read_ase(path: Path) -> HostAtoms:
    from ase.io import read as ase_read  # type: ignore[import]

    atoms = ase_read(str(path))
    positions = np.array(atoms.get_positions(), dtype=float)
    species = [s.symbol for s in atoms]
    # ASE cell rows are lattice vectors — same convention as pymatgen
    lattice = np.array(atoms.get_cell(), dtype=float)
    charges = np.zeros(len(species), dtype=float)
    return HostAtoms(positions=positions, species=species,
                     charges=charges, lattice=lattice, source=str(path))


def read_cif(
    path: str | Path,
    backend: str = "pymatgen",
) -> HostAtoms:
    """Load a CIF and return a HostAtoms (Cartesian Å, no charges yet).

    Parameters
    ----------
    path : str or Path
        Path to the CIF file.
    backend : {"pymatgen", "ase", "auto"}
        Library used to parse the CIF.  ``"auto"`` tries pymatgen first
        and falls back to ase if pymatgen is not installed.

    Returns
    -------
    HostAtoms
        positions: (N, 3) ndarray in Å, species: list[str],
        lattice: (3, 3) ndarray.  charges are zero-initialized.
    """
    path = Path(path)
    if backend == "pymatgen":
        return _read_pymatgen(path)
    elif backend == "ase":
        return _read_ase(path)
    elif backend == "auto":
        try:
            return _read_pymatgen(path)
        except Exception:
            return _read_ase(path)
    else:
        raise ValueError(f"Unknown backend {backend!r}; choose 'pymatgen', 'ase', or 'auto'.")
