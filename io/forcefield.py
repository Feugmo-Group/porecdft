"""Force-field parameter readers.

Supports the simple whitespace-delimited `.dat` format used by the legacy code:
    element sigma_A epsilon_K [optional notes...]

Returns a dict of per-element FFEntry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FFEntry:
    element: str
    sigma: float       # Å
    epsilon: float     # K (Boltzmann units)
    source: str = ""   # e.g. "UFF", "DREIDING"


def read_forcefield_dat(path: str | Path) -> dict[str, FFEntry]:
    """Read a Lennard-Jones force-field file.

    Format: one entry per non-comment line, whitespace-delimited::
        element sigma_A epsilon_K [source_tag]

    Lines starting with ``#`` or ``//`` are ignored. Returns ``{element: FFEntry}``.
    """
    entries: dict[str, FFEntry] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        element = parts[0]
        sigma = float(parts[1])
        epsilon = float(parts[2])
        source = parts[3] if len(parts) > 3 else ""
        entries[element] = FFEntry(element, sigma, epsilon, source)
    return entries
