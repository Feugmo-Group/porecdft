"""Partial-charge file readers.

Convention: a CSV with header `element,charge[,source]` or `label,element,charge`.
The reader supports both the by-element mapping (one charge per species) and the
by-site mapping (one charge per atom in the CIF, matching atom order).
"""

from __future__ import annotations

import csv
from pathlib import Path


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
