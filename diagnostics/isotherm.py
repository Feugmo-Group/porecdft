"""Adsorption-isotherm utilities — Henry baseline + Langmuir saturation cap.

Two models in increasing sophistication:

1. **Henry** (`compute_isotherm_henry`): ρ(r) = ρ_bulk · exp(-β V_ext_avg(r)).
   Exact in the low-pressure limit, but rises linearly without bound at high p.

2. **Langmuir-on-grid** (`compute_isotherm_langmuir`): ρ(r) = ρ_henry / (1 + ρ_henry · v_excl).
   Each voxel saturates at ρ_max = 1/v_excl, capping total loading at the pore
   filling capacity. The Langmuir form is the simplest correction beyond Henry
   that gives a physical plateau. Full FMT (Phase 2.2 proper) replaces the
   simple v_excl with weighted-density excluded volume — better physics but
   not needed for the qualitative shape.

3. **Langmuir + Wertheim association** (`compute_isotherm_langmuir_assoc`): adds
   N_assoc = Σ_s (1 − X_s) on top of the Langmuir baseline.  Each host site s
   contributes a discrete H-bond loading via the Wertheim TPT-1 site-balance.
   See `porecdft.functional.association` for the formulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from porecdft.eos.ideal_gas import (
    density_from_pressure,
)

K_TO_KJ_PER_MOL = 8.314462618e-3
AVOGADRO = 6.02214076e23


@dataclass(frozen=True)
class IsothermResult:
    """A single isotherm at fixed temperature.

    All densities are in molecules / Å³, energies in K. Loading is reported in
    both molecules-per-unit-cell (raw) and mmol per gram of framework.
    """
    temperature_K: float
    pressures_bar: np.ndarray
    rho_bulk: np.ndarray                # (P,) molecules/Å³
    loading_N_per_cell_abs: np.ndarray  # (P,) absolute CO2 per unit cell
    loading_N_per_cell_exc: np.ndarray  # (P,) excess
    loading_mmol_per_g_abs: np.ndarray  # (P,) absolute mmol/g
    loading_mmol_per_g_exc: np.ndarray  # (P,) excess mmol/g
    cell_volume_A3: float
    framework_mass_amu: float           # mass of one unit cell, in amu


def compute_isotherm_henry(
    vext_avg_grid_K: np.ndarray,
    dV_A3: float,
    pressures_bar: np.ndarray,
    temperature_K: float,
    framework_mass_amu: float,
    accessibility_mask: np.ndarray | None = None,
    v_min_clip_K: float = -4000.0,      # ~-33 kJ/mol — physically max binding for CO2
    boltz_cap: float = 50.0,
) -> IsothermResult:
    """Henry / ideal-gas-in-external-field isotherm.

    Parameters
    ----------
    vext_avg_grid_K : (Nx, Ny, Nz) ndarray
        Orientation-averaged Vext on a real-space grid, in units of K.
    dV_A3 : float
        Volume per voxel in Å³.
    pressures_bar : (P,) ndarray
        Pressures at which to evaluate.
    temperature_K : float
        Reservoir temperature.
    framework_mass_amu : float
        Mass of the unit cell (amu) — used to convert N → mmol/g.
    accessibility_mask : (Nx, Ny, Nz) bool ndarray, optional
        True where CO2 can be placed (grid points beyond the close-contact
        radius from every host atom). False voxels contribute zero to V_eff
        regardless of their Vext value. Recommended: ``nearest-atom distance
        ≥ 2.0 Å``. Without this mask, voxels sitting on top of host atoms
        produce huge negative Vext (e.g. -200000 kJ/mol from Coulomb at < 0.5 Å)
        that swamp the integral.
    v_min_clip_K : float
        Physical lower bound on Vext (in K). Defaults to ≈ -33 kJ/mol — well
        below the strongest physisorption binding for CO2/MOF, but well above
        the artefactual -200000 kJ/mol that appear at near-contact grid points
        when the framework is not accessibility-masked. Set to -np.inf to disable.
    boltz_cap : float
        Final |β V| clip for numerical safety.
    """
    beta = 1.0 / temperature_K
    V = np.where(np.isfinite(vext_avg_grid_K), vext_avg_grid_K, +1e6)
    # Physical floor: avoid artefactual deep wells from near-atom grid points
    V = np.maximum(V, v_min_clip_K)
    bV = np.clip(beta * V, -boltz_cap, +boltz_cap)
    boltz = np.exp(-bV)
    if accessibility_mask is not None:
        boltz = boltz * accessibility_mask
    V_cell = float(np.prod(vext_avg_grid_K.shape) * dV_A3)
    V_eff = float(boltz.sum() * dV_A3)
    # stash for downstream Langmuir use
    _stash_boltz = boltz
    _stash_dV = dV_A3

    rho_bulk = np.array([density_from_pressure(p, temperature_K) for p in pressures_bar])
    N_abs = rho_bulk * V_eff
    N_exc = rho_bulk * (V_eff - V_cell)

    framework_mass_g = framework_mass_amu / AVOGADRO
    to_mmol_per_g = 1000.0 / (AVOGADRO * framework_mass_g)
    out = IsothermResult(
        temperature_K=temperature_K,
        pressures_bar=np.asarray(pressures_bar),
        rho_bulk=rho_bulk,
        loading_N_per_cell_abs=N_abs,
        loading_N_per_cell_exc=N_exc,
        loading_mmol_per_g_abs=N_abs * to_mmol_per_g,
        loading_mmol_per_g_exc=N_exc * to_mmol_per_g,
        cell_volume_A3=V_cell,
        framework_mass_amu=framework_mass_amu,
    )
    # attach intermediates for the Langmuir cap to reuse without redoing exp(-βV)
    out.__dict__["_boltz_grid"] = _stash_boltz   # type: ignore[attr-defined]
    out.__dict__["_dV_A3"] = _stash_dV
    return out


def compute_isotherm_langmuir(
    vext_avg_grid_K: np.ndarray,
    dV_A3: float,
    pressures_bar: np.ndarray,
    temperature_K: float,
    framework_mass_amu: float,
    v_excl_A3: float = 57.0,                 # excluded volume per CO2 molecule (≈ 4·(4π/3)(σ/2)³)
    accessibility_mask: np.ndarray | None = None,
    v_min_clip_K: float = -4000.0,
    boltz_cap: float = 50.0,
) -> IsothermResult:
    """Langmuir-on-grid isotherm — each voxel saturates at ρ_max = 1/v_excl.

    Per-voxel Langmuir form:

        ρ_henry(r) = ρ_bulk · exp(-β V_avg(r))
        ρ_capped(r) = ρ_henry(r) / (1 + ρ_henry(r) · v_excl)

    This caps the loading at the pore filling capacity (≈ V_pore / v_excl
    molecules). It's the simplest correction beyond Henry that gives a physical
    plateau. Replace `v_excl` with FMT-derived weighted density for the
    rigorous version.
    """
    beta = 1.0 / temperature_K
    V = np.where(np.isfinite(vext_avg_grid_K), vext_avg_grid_K, +1e6)
    V = np.maximum(V, v_min_clip_K)
    bV = np.clip(beta * V, -boltz_cap, +boltz_cap)
    boltz = np.exp(-bV)
    if accessibility_mask is not None:
        boltz = boltz * accessibility_mask
    V_cell = float(np.prod(vext_avg_grid_K.shape) * dV_A3)

    rho_bulk = np.array([density_from_pressure(p, temperature_K) for p in pressures_bar])
    # Vectorise over pressures: shape (P, *grid)
    N_abs = np.empty(len(rho_bulk))
    N_exc = np.empty(len(rho_bulk))
    for i, rb in enumerate(rho_bulk):
        rho_henry = rb * boltz
        rho_capped = rho_henry / (1.0 + rho_henry * v_excl_A3)
        N_abs[i] = float(rho_capped.sum() * dV_A3)
        N_exc[i] = float((rho_capped - rb * accessibility_mask if accessibility_mask is not None
                           else rho_capped - rb).sum() * dV_A3)

    framework_mass_g = framework_mass_amu / AVOGADRO
    to_mmol_per_g = 1000.0 / (AVOGADRO * framework_mass_g)
    return IsothermResult(
        temperature_K=temperature_K,
        pressures_bar=np.asarray(pressures_bar),
        rho_bulk=rho_bulk,
        loading_N_per_cell_abs=N_abs,
        loading_N_per_cell_exc=N_exc,
        loading_mmol_per_g_abs=N_abs * to_mmol_per_g,
        loading_mmol_per_g_exc=N_exc * to_mmol_per_g,
        cell_volume_A3=V_cell,
        framework_mass_amu=framework_mass_amu,
    )


def compute_isotherm_langmuir_assoc(
    vext_avg_grid_K: np.ndarray,
    dV_A3: float,
    grid_xyz: np.ndarray,
    pressures_bar: np.ndarray,
    temperature_K: float,
    framework_mass_amu: float,
    assoc,                              # WertheimiAssociation
    v_excl_A3: float = 57.0,
    accessibility_mask: np.ndarray | None = None,
    v_min_clip_K: float = -4000.0,
    boltz_cap: float = 50.0,
) -> IsothermResult:
    """Langmuir + Wertheim TPT-1 association isotherm.

    Combines the standard Langmuir-on-grid loading with an additive association
    contribution from discrete host H-bond sites:

        N_total(p) = N_langmuir(p) + N_assoc(p)

    where N_assoc = Σ_s (1 − X_s) and X_s satisfies the Wertheim site-balance
    using the Langmuir-capped density ρ(r) as input.

    Parameters
    ----------
    grid_xyz : (*shape, 3) ndarray
        Cartesian positions (Å) matching the Vext grid, as returned by
        ``build_grid`` reshaped to (*shape, 3).
    assoc : WertheimiAssociation
        Pre-built collection of association sites.
    All other parameters are the same as ``compute_isotherm_langmuir``.
    """
    beta = 1.0 / temperature_K
    V = np.where(np.isfinite(vext_avg_grid_K), vext_avg_grid_K, +1e6)
    V = np.maximum(V, v_min_clip_K)
    bV = np.clip(beta * V, -boltz_cap, +boltz_cap)
    boltz = np.exp(-bV)
    if accessibility_mask is not None:
        boltz = boltz * accessibility_mask
    V_cell = float(np.prod(vext_avg_grid_K.shape) * dV_A3)

    rho_bulk_arr = np.array([density_from_pressure(p, temperature_K) for p in pressures_bar])
    N_abs = np.empty(len(rho_bulk_arr))
    N_exc = np.empty(len(rho_bulk_arr))
    for i, rb in enumerate(rho_bulk_arr):
        rho_henry = rb * boltz
        rho_capped = rho_henry / (1.0 + rho_henry * v_excl_A3)
        N_lang = float(rho_capped.sum() * dV_A3)
        N_assoc = assoc.loading_contribution(rho_capped, grid_xyz, dV_A3, temperature_K)
        N_abs[i] = N_lang + N_assoc
        bulk_term = rb * (accessibility_mask if accessibility_mask is not None
                          else np.ones_like(rho_capped))
        N_exc[i] = float((rho_capped - bulk_term).sum() * dV_A3) + N_assoc

    framework_mass_g = framework_mass_amu / AVOGADRO
    to_mmol_per_g = 1000.0 / (AVOGADRO * framework_mass_g)
    return IsothermResult(
        temperature_K=temperature_K,
        pressures_bar=np.asarray(pressures_bar),
        rho_bulk=rho_bulk_arr,
        loading_N_per_cell_abs=N_abs,
        loading_N_per_cell_exc=N_exc,
        loading_mmol_per_g_abs=N_abs * to_mmol_per_g,
        loading_mmol_per_g_exc=N_exc * to_mmol_per_g,
        cell_volume_A3=V_cell,
        framework_mass_amu=framework_mass_amu,
    )


def compute_isotherm_langmuir_assoc_sc(
    vext_avg_grid_K: np.ndarray,
    dV_A3: float,
    grid_xyz: np.ndarray,
    pressures_bar: np.ndarray,
    temperature_K: float,
    framework_mass_amu: float,
    assoc,                              # WertheimiAssociation
    v_excl_A3: float = 57.0,
    accessibility_mask: np.ndarray | None = None,
    v_min_clip_K: float = -4000.0,
    boltz_cap: float = 50.0,
    n_picard: int = 4,
) -> IsothermResult:
    """Langmuir with self-consistent Wertheim association (effective-Vext approach).

    At each pressure, iterates:

        V_eff(r) = V_ext(r) − T · Δc¹_assoc(r; ρ_capped)
        ρ_henry(r) = ρ_bulk · exp(−β V_eff(r))
        ρ_capped(r) = ρ_henry / (1 + ρ_henry · v_excl)

    ``n_picard`` times before recording loading.  This avoids double-counting:
    Wertheim deepens Vext near association sites so the Langmuir density
    redistributes self-consistently.  The bulk reference has no sites, so the
    correction vanishes there.

    Parameters
    ----------
    grid_xyz : (*shape, 3) Å ndarray
    assoc : WertheimiAssociation — pore-center sites with ε_assoc > 0
    n_picard : number of self-consistency iterations (3–5 is sufficient)
    """
    T = temperature_K
    beta = 1.0 / T
    V_clip = np.where(np.isfinite(vext_avg_grid_K), vext_avg_grid_K, +1e6)
    V_clip = np.maximum(V_clip, v_min_clip_K)
    V_cell = float(np.prod(vext_avg_grid_K.shape) * dV_A3)

    rho_bulk_arr = np.array([density_from_pressure(p, T) for p in pressures_bar])
    N_abs = np.empty(len(rho_bulk_arr))
    N_exc = np.empty(len(rho_bulk_arr))

    for i, rb in enumerate(rho_bulk_arr):
        # Start with unmodified Langmuir
        bV = np.clip(beta * V_clip, -boltz_cap, +boltz_cap)
        boltz = np.exp(-bV)
        if accessibility_mask is not None:
            boltz = boltz * accessibility_mask
        rho_capped = rb * boltz / (1.0 + rb * boltz * v_excl_A3)

        # Picard: update V_eff using current density, recompute density
        for _ in range(n_picard):
            V_eff = assoc.effective_vext(V_clip, rho_capped, grid_xyz, dV_A3, T)
            bV_eff = np.clip(beta * V_eff, -boltz_cap, +boltz_cap)
            boltz_eff = np.exp(-bV_eff)
            if accessibility_mask is not None:
                boltz_eff = boltz_eff * accessibility_mask
            rho_capped = rb * boltz_eff / (1.0 + rb * boltz_eff * v_excl_A3)

        N_abs[i] = float(rho_capped.sum() * dV_A3)
        bulk_term = rb * (accessibility_mask if accessibility_mask is not None
                          else np.ones_like(rho_capped))
        N_exc[i] = float((rho_capped - bulk_term).sum() * dV_A3)

    framework_mass_g = framework_mass_amu / AVOGADRO
    to_mmol_per_g = 1000.0 / (AVOGADRO * framework_mass_g)
    return IsothermResult(
        temperature_K=T,
        pressures_bar=np.asarray(pressures_bar),
        rho_bulk=rho_bulk_arr,
        loading_N_per_cell_abs=N_abs,
        loading_N_per_cell_exc=N_exc,
        loading_mmol_per_g_abs=N_abs * to_mmol_per_g,
        loading_mmol_per_g_exc=N_exc * to_mmol_per_g,
        cell_volume_A3=V_cell,
        framework_mass_amu=framework_mass_amu,
    )
