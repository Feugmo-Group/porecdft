"""Pluggable potential interface and concrete implementations.

The `Potential` abstract base class is the single most important interface in
the package. Today it has analytic implementations (LJ, Morse, Coulomb, quadrupole).
Tomorrow it must accept any machine-learning interatomic potential (MACE, NequIP,
Allegro, SchNet) without changing the cDFT engine — see `mlip.py` for adapter stubs.
"""

from porecdft.forcefield.base import Potential, PotentialEnergy
from porecdft.forcefield.lj import LJPotential
from porecdft.forcefield.morse import MorsePotential, MorseScalarPotential, MorseParam
from porecdft.forcefield.coulomb import CoulombPotential
from porecdft.forcefield.quadrupole import QuadrupoleEFGPotential
from porecdft.forcefield.composite import CompositePotential
from porecdft.forcefield import mlip  # noqa: F401  (adapter module)
from porecdft.forcefield.mlip import TabulatedPotential

__all__ = [
    "Potential",
    "PotentialEnergy",
    "LJPotential",
    "MorsePotential",
    "MorseScalarPotential",
    "MorseParam",
    "CoulombPotential",
    "QuadrupoleEFGPotential",
    "CompositePotential",
    "TabulatedPotential",
    "mlip",
]
