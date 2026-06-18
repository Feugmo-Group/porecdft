"""porecdft — general-purpose classical DFT for fluids in arbitrary external fields.

The package is host-agnostic and fluid-agnostic. ALF/CO2 is the first application,
in `applications/alf_co2/`. The `Potential` interface is designed to accept any
external-field source: analytic forms (LJ, Morse, Buckingham, Coulomb, quadrupole)
today, machine-learning interatomic potentials (MACE, NequIP, Allegro) tomorrow.

Layout:
    io/          CIF, force-field, partial-charge readers
    structure/   HostAtoms, supercells, pore-volume probes, site finders
    forcefield/  Potential ABC + concrete implementations (LJ, Morse, Coulomb,
                 quadrupole-EFG, composite, MLIP adapters)
    fluid/       Fluid ABC + CO2 (EPM2 / TraPPE), N2, CH4, generic
    vext/        Orientation sampling + 3D Vext grid builder with caching
    eos/         Bulk equations of state (ideal gas, LJ-MBWR, PC-SAFT)
    functional/  Free-energy functionals (FMT, WDA-LJ, PC-SAFT, Wertheim)
    solver/      Picard, Anderson mixing, FIRE
    diagnostics/ Binding-site probe, Henry constant, isosteric heat
    plotting/    Standardized diagnostic plots
"""

__version__ = "0.0.1"

from porecdft.compute_config import ComputeConfig  # noqa: F401  — public API
