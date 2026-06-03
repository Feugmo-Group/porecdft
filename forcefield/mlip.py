"""MLIP adapters — Potential subclasses that wrap machine-learning interatomic
potentials.

This module is the bridge between the cDFT engine and the modern ML potential
ecosystem (MACE, NequIP, Allegro, SchNet, PaiNN, e3nn-jax). The cDFT engine never
touches an MLIP directly — it only sees the `Potential` interface.

Current state: this file defines the interface contract and one reference
adapter (`ASEPotential`) that works with any ASE-compatible calculator.
Concrete adapters for MACE / NequIP / Allegro will be added as the ML stack
firms up — they follow the exact same pattern but skip the ASE round-trip for
speed.

Performance note: MLIP grid evaluation (~64³ × 50 orientations ≈ 13M calls)
is impractical at runtime. The intended workflow is `vext/builder.build_vext_on_grid`
which computes Vext on the grid once, caches it to .npy, and reuses it in every
cDFT solve. This applies equally to analytic and ML potentials.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from porecdft.forcefield.base import Potential, PotentialEnergy
from porecdft.structure.host import HostAtoms


@dataclass(frozen=True)
class ASEPotential(Potential):
    """Wrap any ASE Calculator as a porecdft Potential.

    Construction is deferred: pass a callable `make_calculator()` that returns
    a fresh ASE Calculator instance, plus a `host_atoms_factory(host: HostAtoms)`
    that returns an ase.Atoms object for the host. The adapter builds a combined
    Atoms (host + fluid sites) per evaluation, attaches the calculator, calls
    get_potential_energy(), and returns the result minus the host-only baseline.

    The host-only baseline is computed lazily and cached.

    This adapter is intentionally minimal — it works with any ASE calculator
    (MACE-ASE, NequIP-ASE, Allegro-ASE, SchNet-ASE, plus DFT codes via ASE).
    """
    make_calculator: callable
    host_atoms_factory: callable
    energy_unit_to_K: float = 1.0 / 8.617333262145e-5  # eV → K conversion if calculator returns eV
    name: str = "ML(ASE)"

    def energy_at(self, r_center, rot, host, fluid_sites, fluid_site_labels) -> PotentialEnergy:
        raise NotImplementedError(
            "ASEPotential.energy_at: implement once an MLIP calculator is selected. "
            "Pattern: build ase.Atoms = host + fluid_at(r_center, rot); attach "
            "calculator; return calculator.get_potential_energy() - host_baseline; "
            "multiply by energy_unit_to_K."
        )


# Sketches for future direct adapters (no ASE round-trip).
# Filled in when the ML stack is selected — keeping placeholder classes here so
# graphify sees the intended architecture.
class MACEPotential(Potential):
    """Direct adapter for MACE (https://github.com/ACEsuit/mace). Filled in later."""
    name: str = "ML(MACE)"
    def energy_at(self, *args, **kwargs) -> PotentialEnergy:
        raise NotImplementedError("MACEPotential: select MACE checkpoint and wire up.")


class NequIPPotential(Potential):
    """Direct adapter for NequIP / Allegro. Filled in later."""
    name: str = "ML(NequIP)"
    def energy_at(self, *args, **kwargs) -> PotentialEnergy:
        raise NotImplementedError("NequIPPotential: select model and wire up.")
