"""Partial-charge file readers.

Convention: a CSV with header `element,charge[,source]` or `label,element,charge`.
The reader supports both the by-element mapping (one charge per species) and the
by-site mapping (one charge per atom in the CIF, matching atom order).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from porecdft.structure.host import HostAtoms


def read_charges_csv(path: str | Path) -> dict[str, float]:
    """Read per-element partial charges from a CSV.

    Expected header: ``element,charge`` (extra columns ignored). Returns
    ``{element: charge_in_e}``.

    Example file::
        element,charge,source
        Al,1.92,DDEC6
        C,0.48,DDEC6
        O,-0.65,DDEC6
        H,0.18,DDEC6
    """
    charges: dict[str, float] = {}
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            element = row.get("element") or row.get("Element") or row.get("symbol")
            charge = row.get("charge") or row.get("Charge") or row.get("q")
            if element is None or charge is None:
                continue
            charges[element.strip()] = float(charge)
    return charges


def assign_hirshfeld_charges(
    host: "HostAtoms",
    charge_file: str | Path,
    source: str = "",
) -> "HostAtoms":
    """Read partial charges from ``charge_file`` and return a new HostAtoms.

    This is a convenience wrapper around :func:`read_charges_csv` and
    :meth:`HostAtoms.assign_charges`.  The name "Hirshfeld" is conventional —
    the file can contain charges from any partitioning scheme (DDEC6, REPEAT,
    Hirshfeld, EQeq); the ``source`` argument is stored for provenance.

    Parameters
    ----------
    host : HostAtoms
        The structure returned by :func:`porecdft.io.cif.read_cif`.
    charge_file : str or Path
        CSV with ``element,charge[,source]`` columns.
    source : str
        Provenance tag stored in ``host.charge_source`` (e.g. ``"DDEC6"``).
        If empty, the value is inferred from the ``source`` column of the CSV
        (first row that has one).

    Returns
    -------
    HostAtoms
        A new HostAtoms with charges assigned.
    """
    from porecdft.structure.host import HostAtoms  # avoid circular import at module level

    charges = read_charges_csv(charge_file)
    if not source:
        # try to infer source from CSV 'source' column
        try:
            with Path(charge_file).open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    s = row.get("source") or row.get("Source") or ""
                    if s:
                        source = s.strip()
                        break
        except Exception:
            pass
    return host.assign_charges(charges, source=source)
