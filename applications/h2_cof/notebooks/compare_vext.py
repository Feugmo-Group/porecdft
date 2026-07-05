"""
Compare result and performance external potential with composite LJ + Morse potential between 
numpy and warp kernel

results cached to applications/h2_cof/results/:
vext_cache_h2_cof333_cpu.npy   — 3D Vext grid from numpy
vext_cache_h2_cof333_gpu.npy   — 3D Vext grid from warp kernel
"""
from __future__ import annotations

import sys
import os
import warnings
from pathlib import Path
import time

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARENT    = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try: sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

ROOT = str(_REPO_ROOT)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure

# ── porecdft imports ────────────────────────────────────────────────────────
from porecdft.io import read_cif
from porecdft.fluid.base import Fluid
from porecdft.io.forcefield import FFEntry
from porecdft.structure.host import HostAtoms
from porecdft.structure.supercell import build_supercell
from porecdft.vext import build_vext_on_grid, build_grid, fibonacci_rotations
from porecdft.compute_config import ComputeConfig
from porecdft.forcefield import (
    CompositePotential,  LJPotential, MorsePotential,
)

# ════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
KCAL_TO_K  = 503.228
SIGMA_H2   = 2.83       # Å
EPSILON_H2 = 59.7       # K (ε/kB)
MASS_H2    = 2.016      # g/mol
RCUT_H2    = 5.0 * SIGMA_H2
NA         = 6.022e23

MORSE_METALS = {"Co", "Fe", "Ni", "Cu", "Mn"}

# Direct Morse params for host-metal / H2 pairs (no combining rule)
MORSE_PARAMS = {
    "Co": dict(D_e=2*0.879*KCAL_TO_K, a=0.850, r_e=2.985, cutoff=12.0),
    "Fe": dict(D_e=2*1.092*KCAL_TO_K, a=1.180, r_e=3.015, cutoff=12.0),
    "Ni": dict(D_e=2*1.154*KCAL_TO_K, a=1.210, r_e=3.207, cutoff=12.0),
    "Cu": dict(D_e=2*0.818*KCAL_TO_K, a=1.462, r_e=2.931, cutoff=12.0),
    "Mn": dict(D_e=2*0.994*KCAL_TO_K, a=0.990, r_e=3.015, cutoff=12.0),
}

# DREIDING LJ params for organic elements
DREIDING_LJ = {   # organic DREIDING parameters for H₂ host interactions
    "H": FFEntry("H",  2.846,  7.649), "C": FFEntry("C",  3.473, 47.856),
    "N": FFEntry("N",  3.263, 38.949), "O": FFEntry("O",  3.033, 48.158),
    "Cl": FFEntry("Cl", 3.520, 114.23), "Co": FFEntry("Co", 2.558, 7.050),
}

MASS_MAP = {
    "H": 1.00784, "C": 12.0107, "N": 14.0067, "O": 15.999,
    "Co": 58.933,  "Cl": 35.45, "F": 18.998,  "Al": 26.9815,
    "Si": 28.0855, "Br": 79.904,"Cu": 63.546,  "Zn": 65.38,
    "Fe": 55.845,  "Ni": 58.693,"Mn": 54.938,
}

STRUCTURES_DIR = os.path.join(ROOT, "applications/h2_cof/structures")
RESULTS_DIR    = os.path.join(ROOT, "applications/h2_cof/results")
FIGURES_DIR    = os.path.join(ROOT, "applications/h2_cof/figures")
OUT_CACHE = os.path.join(ROOT, "applications/h2_cof/results/vext_cache")
OUT_RES   = RESULTS_DIR
OUT_FIG   = Path(FIGURES_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUT_CACHE,   exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 2. STRUCTURE LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_host(name: str) -> HostAtoms:
    cif = os.path.join(STRUCTURES_DIR, name + ".cif")
    pmg = Structure.from_file(cif)
    return HostAtoms(
        positions=pmg.cart_coords.copy(),
        species=[str(s) for s in pmg.species],
        charges=np.zeros(pmg.num_sites),
        lattice=pmg.lattice.matrix.copy(),
        source=cif,
    )


# ════════════════════════════════════════════════════════════════════════════
# 3. VEXT GRID  (3D)
# ════════════════════════════════════════════════════════════════════════════

def build_vext_3d(
    host: HostAtoms,
    grid_spacing: float = 0.25 * SIGMA_H2,
    supercell: tuple = (3, 3, 3),
    cache_path: str | None = None,
    use_warp: bool = True,
) -> tuple[np.ndarray, tuple, np.ndarray, float]:
    """Build Vext on a 3D grid (Nx, Ny, Nz) using Morse+LJ mixed FF.

    Returns
    -------
    vext_3d  : (Nx, Ny, Nz) ndarray in K
    n_pts    : (Nx, Ny, Nz) int tuple
    spacings : (dx, dy, dz) float ndarray in Å
    dV       : float voxel volume in Å³
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True).item()
        print(f"  Loaded Vext from cache: {cache_path}", flush=True)
        return data["vext_3d"], tuple(data["n_pts"]), data["spacings"], float(data["dV"])

    nx, ny, nz = supercell
    host_sc = build_supercell(host, nx, ny, nz)
    shift = (
        -(nx // 2) * host.lattice[0]
        - (ny // 2) * host.lattice[1]
        - (nz // 2) * host.lattice[2]
    )
    pos_sc  = host_sc.positions + shift
    spec_sc = host_sc.species

    # Grid over ONE unit cell
    lengths = np.linalg.norm(host.lattice, axis=1)
    n_pts   = tuple(max(2, int(np.ceil(l / grid_spacing))) for l in lengths)
    spacings = np.array([lengths[i] / n_pts[i] for i in range(3)])
    dV       = float(spacings.prod())

    fx = np.linspace(0, 1, n_pts[0], endpoint=False)
    fy = np.linspace(0, 1, n_pts[1], endpoint=False)
    fz = np.linspace(0, 1, n_pts[2], endpoint=False)
    Fx, Fy, Fz = np.meshgrid(fx, fy, fz, indexing="ij")
    grid_xyz = np.stack([Fx, Fy, Fz], axis=-1).reshape(-1, 3) @ host.lattice

    Ng   = grid_xyz.shape[0]
    vext = np.zeros(Ng, dtype=float)

    lj_params = {}
    for el in set(spec_sc):
        if el in MORSE_METALS or el not in DREIDING:
            continue
        sigma_s, epsilon_s = DREIDING[el]
        lj_params[el] = (
            0.5 * (SIGMA_H2 + sigma_s),
            float(np.sqrt(EPSILON_H2 * epsilon_s)),
        )

    print(f"  Computing Vext on {n_pts[0]}×{n_pts[1]}×{n_pts[2]} grid "
          f"({len(spec_sc)} atoms in {nx}×{ny}×{nz} supercell)...", flush=True)

    for i, (el, pos_i) in enumerate(zip(spec_sc, pos_sc)):
        dr = grid_xyz - pos_i[None, :]
        r  = np.sqrt(np.einsum("gi,gi->g", dr, dr).clip(1e-8))

        if el in MORSE_METALS:
            mp = MORSE_PARAMS[el]
            mask = r < mp["cutoff"]
            if mask.any():
                x = np.exp(-mp["a"] * (r[mask] - mp["r_e"]))
                v = mp["D_e"] * ((1.0 - x)**2 - 1.0)
                vext[mask] += np.clip(v, -mp["D_e"], 1e5)
        elif el in lj_params:
            sigma_sf, epsilon_sf = lj_params[el]
            mask = r < RCUT_H2
            if mask.any():
                sr6 = (sigma_sf / r[mask])**6
                vext[mask] += 4.0 * epsilon_sf * (sr6**2 - sr6)

    vext_3d = vext.reshape(n_pts)

    if cache_path:
        np.save(cache_path, {
            "vext_3d": vext_3d, "n_pts": np.array(n_pts),
            "spacings": spacings, "dV": dV,
        })
        print(f"  Cached Vext → {cache_path}", flush=True)

    return vext_3d, n_pts, spacings, dV


def main():
    print("=" * 65)
    print("Compare external potential for H₂ / COF-333-CoCl2")
    print("=" * 65)

    # ── 1. Host + fluid ───────────────────────────────────────────────────────────
    print("\nLoading structure...", flush=True)
    host  = load_host("COF-333-CoCl2")
    host  = host.assign_charges({s: 0.0 for s in set(host.species)})
    fluid = Fluid(
        name="H2", body_sites=np.zeros((1, 3)),
        site_labels=["H2"],
        ff={"H2": FFEntry("H2", SIGMA_H2, EPSILON_H2, "mTraFF")},
        charges={"H2": 0.0}, molar_mass=2.016,
    )
    mss    = sum(MASS_MAP.get(el, 0.0) for el in host.species)
    V_cell = host.cell_volume
    print(f"  V_cell = {V_cell:.1f} Å³   mss = {mss:.1f} u", flush=True)

    # ── 2. LJ + Morse composite potential ─────────────────────────────────────────
    # Morse handles Co open-metal sites; LJ handles everything else.
    morse_atoms = {s: MORSE_PARAMS["Co"] for s in host.species if s == "Co"}
    potential = CompositePotential([
        MorsePotential(host_params = morse_atoms,
                       fluid_params = None,
                       cutoff = 12.0
                       ),
        LJPotential(host_ff=DREIDING_LJ, fluid_ff=fluid.ff,
                    cutoff=5*SIGMA_H2, exclude_species=frozenset(["Co"])),
    ])
    
    # ── 3. Vext grid (cached) ─────────────────────────────────────────────────────
    T_list = [298.0, 273.0, 323]
    vext_runtime = {}
    for T_K in T_list:
        print(f"\n=== T = {T_K:.0f} K ===")

        runtime = time.time()
        print(f"  Building Vext on grid from cpu...")
        cache = os.path.join(OUT_CACHE,  f"vext_avg_T{T_K:.0f}K_cpu.npy")
        compute_config = ComputeConfig(use_warp    = False,
                                       warp_device = "cpu",
                                       jax_device  = "cpu",
                                       dtype       = "float64")
        cache = os.path.join(OUT_CACHE,  f"vext_avg_T{T_K:.0f}K_cpu.npy")
        data = build_vext_on_grid(
        host, fluid, potential,
        orientations=fibonacci_rotations(1),    # monatomic — one orientation
        spacing=0.5, pbc_supercell=(1, 1, 1),
        temperature_K=T_K,
        cache_path=cache,
        v_reject_below_K=-10000.0, v_cap_above_K=5000.0,
        compute = compute_config,
        averaging = "boltzmann"
        )
        elapsed = time.time() - runtime
        vext_runtime[f"cpu_T{T_K:.0f}K"] = [elapsed]
        vext_avg_cpu = np.asarray(data["vext_avg"])

        runtime = time.time()
        print(f"  Building Vext on grid from gpu...")
        cache = os.path.join(OUT_CACHE,  f"vext_avg_T{T_K:.0f}K_gpu.npy")
        compute_config = ComputeConfig(use_warp    = True,
                                       warp_device = "gpu",
                                       jax_device  = "gpu",
                                       dtype       = "float64")
        cache = os.path.join(OUT_CACHE,  f"vext_avg_T{T_K:.0f}K_gpu.npy")
        data = build_vext_on_grid(
        host, fluid, potential,
        orientations=fibonacci_rotations(1),    # monatomic — one orientation
        spacing=0.5, pbc_supercell=(1, 1, 1),
        temperature_K=T_K,
        cache_path=cache,
        v_reject_below_K=-10000.0, v_cap_above_K=5000.0,
        compute = compute_config,
        averaging = "boltzmann"
        )
        elapsed = time.time() - runtime 
        vext_runtime[f"gpu_T{T_K:.0f}K"] = [elapsed]
        vext_avg_gpu = np.asarray(data["vext_avg"])

        # Diagnostics — float32 (warp) vs float64 (numpy) precision difference
        # is expected to be O(0.01–1 K) for Vext values of O(100–5000 K).
        abs_diff = np.abs(vext_avg_cpu - vext_avg_gpu)
        print(f"  CPU vs GPU Vext comparison at T={T_K:.0f} K:")
        print(f"    max |diff|  = {abs_diff.max():.4f} K")
        print(f"    mean |diff| = {abs_diff.mean():.4f} K")
        print(f"    max |cpu|   = {np.abs(vext_avg_cpu).max():.1f} K")
        # Physically meaningful tolerance: 1 K absolute OR 0.1% relative.
        # Float32 introduces ~0.001–0.1 K errors; larger discrepancies indicate a bug.
        ok = np.allclose(vext_avg_cpu, vext_avg_gpu, atol=1.0, rtol=1e-3)
        print(f"    PASS (atol=1 K, rtol=0.1%): {ok}")
        assert ok, (
            f"CPU vs GPU Vext mismatch at T={T_K:.0f} K: "
            f"max |diff| = {abs_diff.max():.3f} K"
        )
    
    # store data 
    import pandas as pd
    print(vext_runtime)
    df = pd.DataFrame(vext_runtime)
    path = os.path.join(OUT_RES,  f"vext_runtime.csv")
    df.to_csv(path, index=False)
    
    # plot data
    df = pd.read_csv(path)
    COLORS = {"cpu": "blue", "gpu": "orange"}
    plt.figure(figsize=(8, 4))
    plt.bar(df.columns, df.iloc[0], color=[COLORS["cpu"] if "cpu" in c else COLORS["gpu"] for c in df.columns])
    plt.ylabel("Vext build time (s)")
    plt.title("Vext build time comparison (CPU vs GPU)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "vext_runtime_comparison.png", dpi=300)



if __name__=="__main__":
    main()