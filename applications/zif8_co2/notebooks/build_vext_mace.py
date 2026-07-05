"""Build V_ext(r; T) for CO2 in ZIF-8 using MACE-MP-0.

Pipeline
--------
1. Load ZIF-8 unit cell from structure/mofs/cif/ZIF-8.cif via pymatgen.
2. Assemble the EPM2 CO2 geometry (3-site rigid, C at origin, O at ±1.149 Å).
3. Load MACE-MP-0 universal potential (mace-torch, auto-downloaded).
4. Compute E_host and E_CO2_vac once each.
5. Build a regular 3D grid over the unit cell (spacing ~0.7 Å → ~25³ points).
6. For each of N_ORIENT=20 Fibonacci orientations (parallelised over CPU cores):
       For each accessible grid point:
           E_int(r, Ω) = E(host + CO2_at(r, Ω)) − E_host − E_CO2_vac  [eV→K]
7. Boltzmann-average over orientations:
       V_ext(r; T) = −k_B T  ln ⟨ exp(−V / k_B T) ⟩_Ω
8. Save cache to results/vext_cache/vext_mace_T{T}K.npy.

Parallelisation strategy
------------------------
One worker process per Fibonacci orientation.  Each worker loads the MACE model
once, then loops over all accessible grid points sequentially.  The hard-core
exclusion mask (grid points within HC_CUTOFF Å of any framework atom) is
precomputed and shared via read-only numpy arrays.

Result for each orientation is checkpointed to a .npy shard so the run can
be resumed if interrupted.

Usage::

    cd porecdft
    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \\
        applications/zif8_co2/notebooks/build_vext_mace.py

Speed estimate (CPU only, ~1 s/point, 13 824 pts × 20 orients):
    - 1 worker  : ~83 h
    - N_WORKERS : ~83 / N_WORKERS h   (set N_WORKERS to cpu_count)
    - Typical  : ~4 h on 20-core Mac
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── sys.path so `porecdft` and `applications` are importable ──────────────────
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARENT    = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try:    sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

import warnings
warnings.filterwarnings("ignore")

from applications.zif8_co2 import ZIF8_CIF, CACHE_DIR

# ── physical constants ─────────────────────────────────────────────────────────
EV_TO_K = 11604.518   # 1 eV / k_B  (K)

# ── simulation parameters ──────────────────────────────────────────────────────
T_K          = 298.0   # temperature (K)
SPACING_ANG  = 1.4     # grid spacing (Å) → 12³ grid; interpolated to FMT grid via TabulatedPotential
N_ORIENT     = 20      # Fibonacci orientations
N_WORKERS    = 6   # keep RAM < ~6 GB; 18 workers caused 57 GB RSS + swap thrashing
HC_CUTOFF    = 2.0     # hard-core: skip grid pts within this Å of any ZIF atom
                       # (orientation-independent pre-filter, conservative)

# Known reference energies (computed previously on the same system/model)
E_HOST_EV    = -1663.8010   # eV   (276-atom ZIF-8, MACE-MP-0 medium, float32)
E_CO2_VAC_EV = -22.7101     # eV   (EPM2 CO2 in vacuum)

# EPM2 CO2 body-frame geometry (Å): C at origin, O at ±1.149 Å along z
CO2_BODY = np.array([
    [0.0, 0.0,  1.149],   # O
    [0.0, 0.0,  0.0  ],   # C
    [0.0, 0.0, -1.149],   # O
])
CO2_SYMBOLS = ["O", "C", "O"]


# ── helpers ────────────────────────────────────────────────────────────────────

def fibonacci_directions(n: int) -> np.ndarray:
    """n quasi-uniform unit vectors on S², shape (n, 3)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    i   = np.arange(n)
    z   = 1.0 - 2.0 * (i + 0.5) / n
    r   = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
    th  = 2.0 * np.pi * i / phi
    return np.stack([r * np.cos(th), r * np.sin(th), z], axis=-1)


def rotation_z_to(u: np.ndarray) -> np.ndarray:
    """3×3 rotation R such that R @ ẑ = u (Rodrigues)."""
    z = np.array([0., 0., 1.])
    v = np.cross(z, u)
    s = np.linalg.norm(v)
    c = float(np.dot(z, u))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + V + V @ V * ((1 - c) / s ** 2)


def build_grid(lattice: np.ndarray, spacing: float):
    """Return grid_xyz (N, 3) in Å and integer shape (nx, ny, nz)."""
    a, b, c = [np.linalg.norm(v) for v in lattice]
    nx = max(1, round(a / spacing))
    ny = max(1, round(b / spacing))
    nz = max(1, round(c / spacing))
    frac = np.stack(np.meshgrid(
        np.linspace(0, 1, nx, endpoint=False),
        np.linspace(0, 1, ny, endpoint=False),
        np.linspace(0, 1, nz, endpoint=False),
        indexing="ij",
    ), axis=-1).reshape(-1, 3)
    return frac @ lattice, (nx, ny, nz)


def hard_core_mask(grid_xyz: np.ndarray, host_pos: np.ndarray,
                   lattice: np.ndarray, cutoff: float) -> np.ndarray:
    """Boolean mask: True for grid pts that are NOT too close to any host atom.

    Uses minimum-image convention for PBC distance.
    Returns shape (N_grid,) with True = accessible.
    """
    inv_lat = np.linalg.inv(lattice)
    accessible = np.ones(len(grid_xyz), dtype=bool)
    for r in host_pos:
        diff = grid_xyz - r         # (N, 3)
        frac = diff @ inv_lat       # (N, 3) fractional
        frac -= np.round(frac)      # minimum image
        d2   = np.sum((frac @ lattice) ** 2, axis=-1)
        accessible &= (d2 >= cutoff ** 2)
    return accessible


# ── per-orientation worker ─────────────────────────────────────────────────────

def _worker_init(model_path: str, device: str, dtype: str):
    """Initialiser: load MACE once per worker process."""
    import torch
    torch.set_num_threads(1)   # prevent thread oversubscription across workers

    import warnings
    warnings.filterwarnings("ignore")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from mace.calculators import MACECalculator
    global _calc
    _calc = MACECalculator(model_paths=model_path, device=device,
                           default_dtype=dtype)
    _calc.use_forces = False   # skip backward pass — we only need energy (~2× speedup)


def _worker_orient(args):
    """Compute E_int for all accessible grid points in one Fibonacci orientation.

    Returns (orient_idx, E_int_K: np.ndarray shape (N_grid,)) where inaccessible
    points are set to +inf.
    """
    (orient_idx, u, grid_xyz, accessible, host_sym, host_pos,
     cell, N_grid) = args

    import gc
    import torch
    from ase import Atoms as AseAtoms

    R          = rotation_z_to(u)
    co2_body_R = CO2_BODY @ R.T     # (3, 3): rotated site offsets

    E_int = np.full(N_grid, np.inf, dtype=np.float32)
    n_acc = int(accessible.sum())
    acc_idx = np.where(accessible)[0]

    # Pre-fetch host data as contiguous arrays to avoid repeated copies
    host_pos_c  = np.ascontiguousarray(host_pos)
    co2_pos_buf = np.empty((3, 3), dtype=np.float64)

    t0 = time.time()
    for k, j in enumerate(acc_idx):
        r0      = grid_xyz[j]
        np.add(r0, co2_body_R, out=co2_pos_buf)   # in-place, no alloc
        combined = AseAtoms(
            symbols   = host_sym + CO2_SYMBOLS,
            positions = np.vstack([host_pos_c, co2_pos_buf]),
            cell      = cell,
            pbc       = True,
        )
        combined.calc = _calc
        E_comb   = combined.get_potential_energy()   # eV
        E_int[j] = float(E_comb - E_HOST_EV - E_CO2_VAC_EV)

        # Explicitly break ASE circular reference and run GC periodically
        combined.calc = None
        _calc.atoms   = None
        _calc.results = {}
        if k % 50 == 0 and k > 0:
            gc.collect()

        if k % 100 == 0 and k > 0:
            elapsed = time.time() - t0
            eta     = elapsed / k * (n_acc - k)
            print(f"  orient {orient_idx+1:02d}/{N_ORIENT}  "
                  f"pt {k}/{n_acc}  ETA {eta/60:.1f} min",
                  flush=True)

    V_orient_K = E_int * EV_TO_K   # K
    print(f"  orient {orient_idx+1:02d}/{N_ORIENT}  done in "
          f"{(time.time()-t0)/60:.1f} min  "
          f"E_int: [{np.nanmin(E_int[accessible])*EV_TO_K:.0f}, "
          f"{np.nanmax(E_int[accessible])*EV_TO_K:.0f}] K",
          flush=True)
    return orient_idx, V_orient_K


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import torch
    from pymatgen.io.cif import CifParser
    from pymatgen.io.ase import AseAtomsAdaptor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}   workers: {N_WORKERS}")

    # 1. Load ZIF-8
    print(f"Loading ZIF-8 from {ZIF8_CIF}")
    parser   = CifParser(ZIF8_CIF)
    zif8_pmg = parser.parse_structures(primitive=False)[0]
    adaptor  = AseAtomsAdaptor()
    host_ase = adaptor.get_atoms(zif8_pmg)
    lattice  = np.array(host_ase.get_cell())
    host_pos = host_ase.get_positions()
    host_sym = list(host_ase.get_chemical_symbols())
    print(f"ZIF-8: {len(host_ase)} atoms, cell = {np.diag(lattice)}")

    # 2. Grid
    grid_xyz, grid_shape = build_grid(lattice, SPACING_ANG)
    N_grid = len(grid_xyz)
    dV     = np.prod([np.linalg.norm(lattice[i]) for i in range(3)]) / N_grid
    print(f"Grid: {grid_shape} = {N_grid} points, dV = {dV:.3f} Å³")

    # 3. Hard-core mask
    print(f"Computing hard-core mask (cutoff = {HC_CUTOFF} Å) …", flush=True)
    accessible = hard_core_mask(grid_xyz, host_pos, lattice, HC_CUTOFF)
    n_acc = int(accessible.sum())
    print(f"  {n_acc}/{N_grid} accessible ({100*n_acc/N_grid:.1f} %)")

    # 4. Fibonacci orientations
    dirs = fibonacci_directions(N_ORIENT)
    print(f"Orientations: {N_ORIENT} Fibonacci directions")

    # 5. Locate cached MACE model
    mace_cache_dir = Path.home() / ".cache" / "mace"
    model_files    = list(mace_cache_dir.glob("*mace*"))
    if not model_files:
        raise FileNotFoundError(
            "MACE-MP model not found in ~/.cache/mace/.  "
            "Run once without multiprocessing to auto-download."
        )
    model_path = str(model_files[0])
    print(f"Using cached MACE model: {model_path}")

    # 6. Orientation shards
    shard_dir = CACHE_DIR / "orient_shards"
    shard_dir.mkdir(exist_ok=True)
    V_orient  = np.full((N_ORIENT, N_grid), np.inf, dtype=np.float32)

    # Check which orientations already have shards
    pending = []
    for i, u in enumerate(dirs):
        shard = shard_dir / f"orient_{i:02d}.npy"
        if shard.exists():
            print(f"  orient {i+1:02d}/{N_ORIENT}  loading shard …")
            V_orient[i] = np.load(str(shard))
        else:
            pending.append((i, u, grid_xyz, accessible, host_sym, host_pos,
                            host_ase.get_cell(), N_grid))

    if pending:
        print(f"Computing {len(pending)} orientations with {N_WORKERS} workers …")
        t_all = time.time()
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=N_WORKERS,
            initializer=_worker_init,
            initargs=(model_path, device, "float32"),
        ) as pool:
            for orient_idx, V_k in pool.imap_unordered(_worker_orient, pending):
                V_orient[orient_idx] = V_k
                shard = shard_dir / f"orient_{orient_idx:02d}.npy"
                np.save(str(shard), V_k)
                n_done = N_ORIENT - len(pending) + int(shard.exists())
                print(f"  shard {orient_idx:02d} saved  "
                      f"({time.time()-t_all:.0f}s elapsed)", flush=True)
        print(f"All orientations done in {(time.time()-t_all)/60:.1f} min")
    else:
        print("All shards already present — skipping MACE computation.")

    # 7. Boltzmann average over orientations
    beta    = 1.0 / T_K                              # K⁻¹
    V_f64   = V_orient.astype(np.float64)
    # Set inaccessible (inf) points to a large but finite cap so exp underflows
    # gracefully rather than producing NaN
    V_f64   = np.where(np.isfinite(V_f64), V_f64, 1e6)
    log_sum = np.log(np.mean(np.exp(-beta * V_f64), axis=0))
    vext_avg = (-T_K * log_sum).astype(np.float32)   # (N_grid,) in K
    vext_3d  = vext_avg.reshape(grid_shape)

    # 8. Save
    out_path = CACHE_DIR / f"vext_mace_T{T_K:.0f}K.npy"
    np.save(str(out_path), {
        "vext_avg" : vext_3d,
        "shape"    : grid_shape,
        "T_K"      : T_K,
        "spacing"  : SPACING_ANG,
        "n_orient" : N_ORIENT,
        "lattice"  : lattice,
        "method"   : "MACE-MP-0 medium, EPM2 CO2, Fibonacci N_Ω=20",
    })
    finite_mask = np.isfinite(V_orient).all(axis=0)
    print(f"\nSaved Vext to {out_path}")
    print(f"  shape {grid_shape}")
    print(f"  min  = {vext_3d[np.isfinite(vext_3d)].min():.0f} K")
    print(f"  max  = {vext_3d[np.isfinite(vext_3d)].max():.0f} K")
    print(f"  mean = {vext_3d[np.isfinite(vext_3d)].mean():.1f} K")


if __name__ == "__main__":
    # macOS requires spawn context; set it before any other mp usage
    mp.set_start_method("spawn", force=True)
    main()
