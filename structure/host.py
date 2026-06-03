"""HostAtoms — the canonical representation of a static host (MOF, zeolite, slit pore).

Holds Cartesian positions (Å), element symbols, partial charges (e), and a 3×3 lattice.
Designed to be immutable and JAX-friendly: positions/charges/lattice are plain ndarrays,
and `assign_charges` returns a new instance rather than mutating.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class HostAtoms:
    positions: np.ndarray              # (N, 3) Cartesian Å
    species: list[str]                 # length N element symbols
    charges: np.ndarray                # (N,) partial charges in e
    lattice: np.ndarray                # (3, 3) Å, rows are lattice vectors
    source: str = ""                   # provenance string (e.g., CIF path)
    charge_source: str = ""            # provenance for charges (e.g., "DDEC6")

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    @property
    def cell_volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    def assign_charges(self, charges_by_element: dict[str, float], source: str = "") -> "HostAtoms":
        """Return a copy with per-element charges applied.

        Raises ValueError if any element in `species` is missing from the dict.
        """
        missing = {s for s in self.species if s not in charges_by_element}
        if missing:
            raise ValueError(f"Missing charge for element(s): {sorted(missing)}")
        new_charges = np.array([charges_by_element[s] for s in self.species], dtype=float)
        return replace(self, charges=new_charges, charge_source=source)

    def neutrality_residual(self) -> float:
        """Return the absolute net charge of the unit cell; should be ≪ 1e-6."""
        return float(abs(self.charges.sum()))

    def select(self, element: str) -> np.ndarray:
        """Boolean mask for atoms of a given element."""
        return np.array([s == element for s in self.species], dtype=bool)

    def summary(self) -> str:
        """Human-readable one-line summary."""
        from collections import Counter
        counts = Counter(self.species)
        formula = " ".join(f"{el}{n}" for el, n in sorted(counts.items()))
        return (
            f"HostAtoms({self.n_atoms} atoms; {formula}; "
            f"V={self.cell_volume:.2f} Å³; net q={self.charges.sum():+.3e} e)"
        )
