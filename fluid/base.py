"""Base Fluid class.

Body frame convention: linear molecules have the molecular axis along +z.
`body_sites[i]` gives the position of site i in Å in the body frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from porecdft.io.forcefield import FFEntry


@dataclass(frozen=True)
class Fluid:
    name: str
    body_sites: np.ndarray            # (S, 3) Å, body frame
    site_labels: list[str]            # length S
    ff: dict[str, FFEntry]            # LJ params per label
    charges: dict[str, float]         # partial charges per label (e)
    theta_zz: float = 0.0             # body-frame quadrupole Θ_zz (e·Å²); 0 if non-linear or unused
    molar_mass: float = 0.0           # g/mol — used to convert N to mmol/g

    @property
    def n_sites(self) -> int:
        return len(self.site_labels)
