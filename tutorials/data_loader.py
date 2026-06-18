"""Helpers to load the Stierle & Gross 2024 SI data into porecdft objects.

The SI ships three custom formats that porecdft's standard I/O does not
read natively:

* ``.dat`` and ``.nml`` files — plain-text dumps of (x, y, z, element)
  rows in Å.  Cell dimensions are stored separately in
  ``solid_database.json``.
* ``DREIDING.dat`` / ``PASCUAL.dat`` — LJ parameter tables
  ``element σ_Å ε_K mass_amu``.
* ``noble_gases.json`` / ``pcsaft_gross2001.json`` — PC-SAFT parameter
  databases keyed by IUPAC name.

These helpers turn those into :class:`porecdft.structure.HostAtoms`,
``dict[str, FFEntry]``, and a PC-SAFT EOS instance respectively.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from porecdft.structure.host import HostAtoms
from porecdft.io.forcefield import FFEntry


# ── data directory ──────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent / "data"
STRUCT_DIR = _DATA_DIR / "structures"
FF_DIR     = _DATA_DIR / "forcefields"
FLUID_DIR  = _DATA_DIR / "fluids"


# ── solid database lookup ───────────────────────────────────────────────────

def _solid_metadata(name: str) -> dict:
    """Look up cell dimensions / atom count / forcefield for a structure name.

    Reads from ``data/structures/solid_database.json`` (provided by
    Stierle 2024 SI).
    """
    db_path = STRUCT_DIR / "solid_database.json"
    with db_path.open() as f:
        db = json.load(f)
    for entry in db:
        if entry["Name"].lower() == name.lower():
            return entry
    raise KeyError(f"{name!r} not in solid_database.json. "
                   f"Known: {[e['Name'] for e in db]}")


# ── plain-text structure reader (.dat / .nml) ──────────────────────────────

def load_dat_structure(name_or_path) -> HostAtoms:
    """Read a Stierle-style ``.dat`` / ``.nml`` atomic-position file.

    The format is whitespace-separated rows of ``x y z element`` with all
    coordinates in Å.  The orthorhombic cell dimensions come from
    ``solid_database.json`` (looked up by file stem).

    Parameters
    ----------
    name_or_path : str or pathlib.Path
        Either a short framework name (``"MFI"``, ``"Dha"``, ``"LTA"``,
        ``"TpPA"``) or an explicit path to a ``.dat`` / ``.nml`` file.

    Returns
    -------
    HostAtoms
        With positions in Å, integer-zero charges (DREIDING/PASCUAL are
        Lennard-Jones-only), and a diagonal lattice matrix
        ``diag(Lx, Ly, Lz)``.
    """
    # Resolve path + name
    p = Path(name_or_path)
    if not p.exists():
        # try as a short name in the structures directory
        for ext in (".dat", ".nml"):
            cand = STRUCT_DIR / f"{name_or_path}{ext}"
            if cand.exists():
                p = cand
                break
        else:
            raise FileNotFoundError(name_or_path)
    name = p.stem

    # Cell from solid_database.json
    meta = _solid_metadata(name)
    Lx, Ly, Lz = (meta["Dimensions"]["Lx"],
                  meta["Dimensions"]["Ly"],
                  meta["Dimensions"]["Lz"])
    lattice = np.diag([Lx, Ly, Lz])

    # Atoms
    positions: list[list[float]] = []
    species:   list[str]         = []
    with p.open() as f:
        for line in f:
            toks = line.split()
            if len(toks) < 4:
                continue
            try:
                x, y, z = float(toks[0]), float(toks[1]), float(toks[2])
            except ValueError:
                continue
            el = toks[3]
            positions.append([x, y, z])
            species.append(el)

    pos_arr = np.asarray(positions, dtype=float)
    charges = np.zeros(len(species))
    return HostAtoms(
        positions=pos_arr,
        species=species,
        charges=charges,
        lattice=lattice,
        source=str(p),
    )


# ── force-field reader ──────────────────────────────────────────────────────

def load_dreiding_ff(path_or_name: str | Path = "DREIDING.dat") -> dict[str, FFEntry]:
    """Read a ``DREIDING.dat`` / ``PASCUAL.dat`` LJ parameter file.

    Each row is ``element σ_Å ε_K mass_amu``.  Returns the
    ``{element: FFEntry}`` dict porecdft's ``LJPotential`` expects.
    """
    p = Path(path_or_name)
    if not p.exists():
        p = FF_DIR / path_or_name
    ff: dict[str, FFEntry] = {}
    with p.open() as f:
        for line in f:
            toks = line.split()
            if len(toks) < 4:
                continue
            el = toks[0]
            try:
                sigma   = float(toks[1])
                epsilon = float(toks[2])
            except ValueError:
                continue
            ff[el] = FFEntry(el, sigma, epsilon, source=p.name)
    if not ff:
        raise ValueError(f"No FF entries parsed from {p}")
    return ff


# ── PC-SAFT fluid lookup ────────────────────────────────────────────────────

def load_pcsaft_fluid(name: str,
                      *paths: str | Path,
                      ) -> Tuple[float, float, float, float]:
    """Look up a fluid's PC-SAFT triplet ``(m, σ, ε/k_B, M)`` by IUPAC name.

    Searches every JSON file in ``data/fluids/`` (or the given ``paths``).
    Names are compared case-insensitively.

    Returns
    -------
    (m, sigma, epsilon_k, molar_mass)
    """
    if not paths:
        paths = (FLUID_DIR / "noble_gases.json",
                 FLUID_DIR / "pcsaft_gross2001.json")
    target = name.strip().lower()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with path.open() as f:
            entries = json.load(f)
        for entry in entries:
            ids = entry.get("identifier", {})
            candidates = [ids.get("name"), ids.get("iupac_name"),
                          ids.get("formula")]
            if any(c and c.lower() == target for c in candidates):
                mr = entry["model_record"]
                M  = entry.get("molarweight", 1.0)
                return float(mr["m"]), float(mr["sigma"]), float(mr["epsilon_k"]), float(M)
    raise KeyError(f"{name!r} not found in PC-SAFT databases under {FLUID_DIR}")


def make_pcsaft_eos(name: str):
    """Convenience: look up + build a ``PCSAFTEOS`` instance."""
    from porecdft.eos import PCSAFTEOS
    m, sigma, eps, M = load_pcsaft_fluid(name)
    return PCSAFTEOS(m=m, sigma_A=sigma, epsilon_K=eps,
                     molar_mass=M, name=f"{name}_PCSAFT")
