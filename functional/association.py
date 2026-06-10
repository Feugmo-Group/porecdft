"""Wertheim TPT-1 association for localized host-fluid binding sites.

Physical picture
----------------
Host framework atoms (e.g. formate-H in ALF) act as discrete H-bond donors.
A fluid molecule (CO₂) near site s can form an H-bond with additional energy
ε_s beyond the mean-field LJ+Coulomb+quadrupole Vext.  The orientation-averaged
Vext captures the *average* binding; Wertheim adds the *directed* H-bond bonus
that is missed by the orientational average.

Site-balance equation (TPT-1, one bond per molecule per site):

    X_s = 1 / (1 + ρ̄_s · κ_s · Δ_s(T))

where
    ρ̄_s  = mean fluid density inside the association sphere of site s (molecules/Å³)
    κ_s   = association volume in Å³  [≈ (4π/3) r_κ³]
    Δ_s   = exp(ε_s / T) - 1          [dimensionless; positive = attractive]
    ε_s   = association energy (K, positive value = attractive)
    X_s   = fraction of site s *unoccupied*

Correction to the one-body direct correlation function (for the Picard loop):

    c¹_assoc(r) = Σ_s  indicator_s(r) · Δ_s(T) / (1 + ρ̄_s · κ_s · Δ_s(T))

where indicator_s(r) = 1 if |r − r_s| < r_κ,s  else 0.

Additive loading contribution (molecules per unit cell):

    N_assoc = Σ_s (1 − X_s)

Usage
-----
>>> assoc = WertheimAssociation.from_host_element(host, "H", energy_K=800.0, kappa_A3=30.0)
>>> c1 = assoc.c1_correction(rho_grid, grid_xyz, T_K=298.0)
>>> N_extra = assoc.loading_contribution(rho_grid, grid_xyz, dV_A3, T_K=298.0)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AssociationSite:
    """A single host-fluid H-bond donor site."""
    position: np.ndarray   # (3,) Cartesian Å
    energy_K: float        # association energy in K (positive = attractive)
    kappa_A3: float        # association volume in Å³

    @property
    def radius_A(self) -> float:
        """Association radius r_κ such that (4π/3) r_κ³ = κ."""
        return (3.0 * self.kappa_A3 / (4.0 * np.pi)) ** (1.0 / 3.0)

    def delta_factor(self, T_K: float) -> float:
        """Δ(T) = exp(ε / T) - 1."""
        return float(np.exp(self.energy_K / T_K) - 1.0)


@dataclass
class WertheimAssociation:
    """Collection of host-fluid association sites (Wertheim TPT-1).

    Parameters
    ----------
    sites : sequence of AssociationSite
        All binding sites in the unit cell (Cartesian Å).

    Notes
    -----
    All grid-based methods expect ``grid_xyz`` with shape ``(*shape, 3)``
    (Å, Cartesian) matching the Vext grid layout from ``build_grid``.
    ``rho_grid`` must have shape ``shape``.
    """
    sites: list[AssociationSite]

    # --------------------------------------------------------------------------
    # Constructors
    # --------------------------------------------------------------------------

    @classmethod
    def from_positions(
        cls,
        positions: np.ndarray,    # (M, 3) Å
        energy_K: float | Sequence[float],
        kappa_A3: float | Sequence[float],
    ) -> "WertheimAssociation":
        """Build from an array of site positions with shared or per-site parameters."""
        M = len(positions)
        energies = np.broadcast_to(np.asarray(energy_K, dtype=float), (M,))
        kappas = np.broadcast_to(np.asarray(kappa_A3, dtype=float), (M,))
        sites = [
            AssociationSite(position=positions[i], energy_K=float(energies[i]),
                            kappa_A3=float(kappas[i]))
            for i in range(M)
        ]
        return cls(sites=sites)

    @classmethod
    def from_host_element(
        cls,
        host,                      # porecdft.structure.host.HostAtoms
        element: str,
        energy_K: float | Sequence[float],
        kappa_A3: float | Sequence[float],
    ) -> "WertheimAssociation":
        """Extract all atoms of ``element`` from ``host`` as association sites."""
        mask = host.select(element)
        positions = host.positions[mask]
        return cls.from_positions(positions, energy_K=energy_K, kappa_A3=kappa_A3)

    # --------------------------------------------------------------------------
    # Core per-site calculations
    # --------------------------------------------------------------------------

    def _rho_bar_all(
        self,
        rho_grid: np.ndarray,     # (*shape) molecules/Å³
        grid_xyz: np.ndarray,     # (*shape, 3) Å
        dV_A3: float,
        use_warp: bool = False,
    ) -> np.ndarray:
        """Mean fluid density inside each site's association sphere. Shape (M,).

        Parameters
        ----------
        use_warp : bool
            If True and warp-lang is installed, dispatch to the GPU kernel via
            ``porecdft.warp_backend.rho_bar_sphere_warp``.  Falls back to the
            NumPy path automatically when Warp is unavailable.
        """
        if use_warp:
            try:
                from porecdft.warp_backend import rho_bar_sphere_warp, WARP_AVAILABLE
                if WARP_AVAILABLE:
                    site_pos   = np.array([s.position  for s in self.sites], dtype=np.float32)
                    site_r2    = np.array([s.radius_A**2 for s in self.sites], dtype=np.float32)
                    site_kappa = np.array([s.kappa_A3  for s in self.sites], dtype=np.float32)
                    return np.asarray(rho_bar_sphere_warp(
                        grid_xyz.reshape(-1, 3).astype(np.float32),
                        rho_grid.ravel().astype(np.float32),
                        site_pos, site_r2, site_kappa, float(dV_A3),
                    ), dtype=float)
            except Exception:
                pass  # fall through to NumPy path

        # NumPy reference path
        flat_rho = rho_grid.ravel()
        flat_xyz = grid_xyz.reshape(-1, 3)
        rho_bar = np.empty(len(self.sites))
        for i, s in enumerate(self.sites):
            dr = flat_xyz - s.position
            r2 = np.einsum("nd,nd->n", dr, dr)
            inside = r2 <= s.radius_A ** 2
            n_inside = inside.sum()
            if n_inside == 0:
                rho_bar[i] = 0.0
            else:
                # ρ̄_s * κ_s = Σ_{r∈Ω_s} ρ(r) dV  (no /κ_s)
                rho_bar[i] = float(flat_rho[inside].sum() * dV_A3) / s.kappa_A3
        return rho_bar

    def fraction_unbound(
        self,
        rho_grid: np.ndarray,
        grid_xyz: np.ndarray,
        dV_A3: float,
        T_K: float,
        use_warp: bool = False,
    ) -> np.ndarray:
        """X_s = 1/(1 + ρ̄_s · κ_s · Δ_s(T)) for each site.  Shape (M,)."""
        rho_bar = self._rho_bar_all(rho_grid, grid_xyz, dV_A3, use_warp=use_warp)
        X = np.empty(len(self.sites))
        for i, s in enumerate(self.sites):
            delta = s.delta_factor(T_K)
            # ρ̄_s is already in units such that ρ̄_s * κ_s = ∫ dV ρ (dimensionless count)
            # rho_bar[i] = ρ̄_s as returned: units of 1 (occupancy numerator term)
            denom = 1.0 + rho_bar[i] * delta
            X[i] = 1.0 / denom
        return X

    # --------------------------------------------------------------------------
    # Additive loading
    # --------------------------------------------------------------------------

    def loading_contribution(
        self,
        rho_grid: np.ndarray,
        grid_xyz: np.ndarray,
        dV_A3: float,
        T_K: float,
        use_warp: bool = False,
    ) -> float:
        """N_assoc = Σ_s (1 − X_s) — extra molecules per unit cell from H-bonds."""
        X = self.fraction_unbound(rho_grid, grid_xyz, dV_A3, T_K, use_warp=use_warp)
        return float(np.sum(1.0 - X))

    # --------------------------------------------------------------------------
    # c¹ correction for the Picard / Anderson FMT loop
    # --------------------------------------------------------------------------

    def c1_correction(
        self,
        rho_grid: np.ndarray,
        grid_xyz: np.ndarray,
        dV_A3: float,
        T_K: float,
        use_warp: bool = False,
    ) -> np.ndarray:
        """Δc¹_assoc(r) — correction to the one-body DCF.  Same shape as rho_grid.

        Δc¹_assoc(r) = Σ_s indicator_s(r) · Δ_s(T) / (1 + ρ̄_s · κ_s · Δ_s(T))

        Enters the density equation as:
            ρ(r) = ρ_bulk · exp(−β V_ext(r) + c¹_HS(r) − c¹_HS_bulk + Δc¹_assoc(r))
        The bulk reference is zero (no association sites in bulk).
        """
        flat_xyz = grid_xyz.reshape(-1, 3)
        rho_bar = self._rho_bar_all(rho_grid, grid_xyz, dV_A3, use_warp=use_warp)
        c1_flat = np.zeros(len(flat_xyz))
        for i, s in enumerate(self.sites):
            delta = s.delta_factor(T_K)
            denom = 1.0 + rho_bar[i] * delta
            c1_contribution = delta / denom              # X_s · Δ_s (dimensionless)
            dr = flat_xyz - s.position
            r2 = np.einsum("nd,nd->n", dr, dr)
            inside = r2 <= s.radius_A ** 2
            c1_flat[inside] += c1_contribution
        return c1_flat.reshape(rho_grid.shape)

    def effective_vext(
        self,
        vext_grid: np.ndarray,
        rho_grid: np.ndarray,
        grid_xyz: np.ndarray,
        dV_A3: float,
        T_K: float,
    ) -> np.ndarray:
        """Return V_eff(r) = V_ext(r) − T · Δc¹_assoc(r).

        Deepens Vext near association sites by the Wertheim chemical-potential
        bonus.  Use iteratively inside a Langmuir Picard loop to capture the
        self-consistent density shift without double-counting.
        """
        c1 = self.c1_correction(rho_grid, grid_xyz, dV_A3, T_K)
        return vext_grid - T_K * c1


# Alias for compatibility with revision scripts
WertheimiAssociation = WertheimAssociation
