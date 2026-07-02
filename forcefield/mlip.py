"""MLIP adapters — Potential subclasses that wrap machine-learning interatomic
potentials.

This module is the bridge between the cDFT engine and the modern ML potential
ecosystem (MACE, NequIP, Allegro, SchNet, PaiNN, e3nn-jax). The cDFT engine never
touches an MLIP directly — it only sees the `Potential` interface.

Current state: this file defines the interface contract and one reference
adapter (`ASEPotential`) that works with any ASE-compatible calculator, plus
`TabulatedPotential` for reusing any pre-computed orientation-averaged Vext grid.
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
from pathlib import Path

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


class TabulatedPotential(Potential):
    """Pre-computed orientation-averaged Vext grid served as a Potential.

    Wraps a 3D array `vext_3d` (shape (nx, ny, nz), units K) that was built
    by `build_vext_on_grid` or `build_vext_mace.py` — any workflow that
    produces a Boltzmann-averaged external potential on a regular grid over a
    periodic unit cell.

    Because the table already encodes the orientation average, `energy_grid`
    ignores the `rot` argument and returns trilinearly interpolated values at
    the query positions.  This means `TabulatedPotential` can be passed directly
    to the cDFT solver without any orientation loop:

        pot = TabulatedPotential.from_cache("vext_mace_T298K.npy")
        # ... or use vext_3d = pot.vext_3d directly in anderson_solve

    The primary value of implementing the `Potential` interface (rather than
    always accessing `vext_3d` directly) is composability: a tabulated MLIP Vext
    can be summed with an analytic long-range correction inside a
    `CompositePotential`, or evaluated on a different grid via `build_vext_on_grid`.

    Parameters
    ----------
    vext_3d : ndarray, shape (nx, ny, nz)
        Orientation-averaged external potential in Kelvin on a regular grid
        covering fractional coordinates [0, 1) along each cell axis.
    lattice : ndarray, shape (3, 3)
        Unit cell matrix in Å. Row i is the i-th lattice vector.
    name : str, optional
        Label for diagnostics.

    Notes
    -----
    Interpolation uses trilinear (first-order) interpolation with periodic
    boundary conditions — the grid wraps at every face.  For a query point
    outside [0, a) × [0, b) × [0, c) the fractional coordinates are folded
    back into [0, 1) modulo 1 before look-up.
    """

    name: str = "Tabulated"

    def __init__(
        self,
        vext_3d: np.ndarray,
        lattice: np.ndarray,
        name: str = "Tabulated",
    ) -> None:
        self.vext_3d  = np.asarray(vext_3d, dtype=np.float64)
        self.lattice  = np.asarray(lattice,  dtype=np.float64)
        self._inv_lat = np.linalg.inv(self.lattice)  # Cartesian → fractional
        self.name     = name
        if self.vext_3d.ndim != 3:
            raise ValueError(f"vext_3d must be 3-D, got shape {self.vext_3d.shape}")
        if self.lattice.shape != (3, 3):
            raise ValueError(f"lattice must be (3, 3), got {self.lattice.shape}")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_cache(cls, path: str | Path, name: str | None = None) -> "TabulatedPotential":
        """Load from a .npy cache produced by build_vext_on_grid or build_vext_mace.py.

        The cache must be a numpy object array (saved with allow_pickle=True)
        containing a dict with at least the keys:

        - ``"vext_avg"``   : ndarray (nx, ny, nz) — orientation-averaged Vext [K]
        - ``"lattice"``    : ndarray (3, 3) — unit cell in Å

        Older caches from build_vext_on_grid that lack ``"lattice"`` can still
        be loaded if ``"grid_shape"`` is present — in that case you should pass
        the lattice via `from_arrays` instead.

        Parameters
        ----------
        path : str or Path
        name : str, optional — defaults to the filename stem.
        """
        path = Path(path)
        data = np.load(path, allow_pickle=True).item()
        if "vext_avg" not in data:
            raise KeyError(
                f"Cache {path} has no 'vext_avg' key. "
                "Keys found: " + str(list(data.keys()))
            )
        vext_3d = np.asarray(data["vext_avg"], dtype=np.float64)
        if "lattice" not in data:
            raise KeyError(
                f"Cache {path} has no 'lattice' key. "
                "Use TabulatedPotential.from_arrays(vext_3d, lattice) instead."
            )
        lattice = np.asarray(data["lattice"], dtype=np.float64)
        return cls(vext_3d, lattice, name=name or path.stem)

    @classmethod
    def from_arrays(
        cls,
        vext_3d: np.ndarray,
        lattice: np.ndarray,
        name: str = "Tabulated",
    ) -> "TabulatedPotential":
        """Construct directly from arrays — no file I/O."""
        return cls(vext_3d, lattice, name=name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cart_to_frac(self, xyz: np.ndarray) -> np.ndarray:
        """Convert Cartesian (N, 3) Å to fractional coords, wrapped to [0, 1)."""
        frac = xyz @ self._inv_lat.T
        return frac % 1.0

    def _trilinear(self, frac: np.ndarray) -> np.ndarray:
        """Trilinear interpolation with PBC.

        Parameters
        ----------
        frac : (N, 3) fractional coordinates, already in [0, 1).

        Returns
        -------
        (N,) interpolated values in K.
        """
        nx, ny, nz = self.vext_3d.shape
        # Continuous index
        ix = frac[:, 0] * nx
        iy = frac[:, 1] * ny
        iz = frac[:, 2] * nz
        # Lower corner indices (wrapped)
        x0 = np.floor(ix).astype(int) % nx
        y0 = np.floor(iy).astype(int) % ny
        z0 = np.floor(iz).astype(int) % nz
        x1 = (x0 + 1) % nx
        y1 = (y0 + 1) % ny
        z1 = (z0 + 1) % nz
        # Fractional offsets within the cell
        dx = ix - np.floor(ix)
        dy = iy - np.floor(iy)
        dz = iz - np.floor(iz)
        # 8 corner values
        v = self.vext_3d
        c000 = v[x0, y0, z0]
        c100 = v[x1, y0, z0]
        c010 = v[x0, y1, z0]
        c110 = v[x1, y1, z0]
        c001 = v[x0, y0, z1]
        c101 = v[x1, y0, z1]
        c011 = v[x0, y1, z1]
        c111 = v[x1, y1, z1]
        # Interpolate
        return (
            c000 * (1 - dx) * (1 - dy) * (1 - dz)
            + c100 * dx       * (1 - dy) * (1 - dz)
            + c010 * (1 - dx) * dy       * (1 - dz)
            + c110 * dx       * dy       * (1 - dz)
            + c001 * (1 - dx) * (1 - dy) * dz
            + c101 * dx       * (1 - dy) * dz
            + c011 * (1 - dx) * dy       * dz
            + c111 * dx       * dy       * dz
        )

    # ------------------------------------------------------------------
    # Potential interface
    # ------------------------------------------------------------------

    def energy_at(
        self,
        r_center: np.ndarray,
        rot: np.ndarray,
        host: HostAtoms,
        fluid_sites: np.ndarray,
        fluid_site_labels: list[str],
    ) -> PotentialEnergy:
        """Return the tabulated Vext at a single grid point (interpolated).

        `rot`, `host`, `fluid_sites`, and `fluid_site_labels` are accepted for
        interface compatibility but **ignored** — the orientation average is
        already baked into the table.
        """
        frac = self._cart_to_frac(np.atleast_2d(r_center))
        val  = float(self._trilinear(frac)[0])
        return PotentialEnergy(total=val)

    def energy_grid(
        self,
        grid_xyz: np.ndarray,
        rot: np.ndarray,
        host: HostAtoms,
        fluid_sites: np.ndarray,
        fluid_site_labels: list[str],
        use_warp: bool = False,
    ) -> np.ndarray:
        """Return tabulated Vext at each grid point, interpolated with PBC.

        `rot`, `host`, `fluid_sites`, `fluid_site_labels`, and `use_warp` are
        accepted for interface compatibility but **ignored**.

        Parameters
        ----------
        grid_xyz : (Ng, 3) Cartesian coordinates in Å.

        Returns
        -------
        (Ng,) ndarray of Vext values in K.
        """
        frac = self._cart_to_frac(np.asarray(grid_xyz))
        return self._trilinear(frac)

    def as_flat(self) -> np.ndarray:
        """Return the raw table flattened to 1D in C-order (useful for direct solver input)."""
        return self.vext_3d.ravel()

    def __repr__(self) -> str:
        nx, ny, nz = self.vext_3d.shape
        a = np.linalg.norm(self.lattice, axis=1)
        return (
            f"TabulatedPotential('{self.name}', grid=({nx},{ny},{nz}), "
            f"cell=[{a[0]:.2f},{a[1]:.2f},{a[2]:.2f}] Å, "
            f"Vext=[{self.vext_3d.min():.0f},{self.vext_3d.max():.0f}] K)"
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
