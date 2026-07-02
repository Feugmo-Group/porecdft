"""Build V_ext(r; T) for CO2 in ZIF-8 using MACE-MP-0.

Pipeline
--------
1. Load ZIF-8 unit cell from structure/mofs/cif/ZIF-8.cif via pymatgen.
2. Assemble the EPM2 CO2 geometry (3-site rigid, C at origin, O at ±1.149 Å).
3. Load MACE-MP-0 universal potential (mace-torch, auto-downloaded).
4. Compute E_host and E_CO2_vac once each.
5. Build a regular 3D grid over the unit cell (spacing ~0.7 Å → ~25³ points).
6. For each of N_ORIENT=20 Fibonacci orientations:
       For each grid point (batched by row):
           E_int(r, Ω) = E(host + CO2_at(r, Ω)) − E_host − E_CO2_vac  [eV→K]
7. Boltzmann-average over orientations:
       V_ext(r; T) = −k_B T  ln ⟨ exp(−V / k_B T) ⟩_Ω
8. Save cache to results/vext_cache/vext_mace_T{T}K.npy.

Runtime: ~20 min on CPU for a 25³ grid × 20 orientations × 276+3 atoms.
Speed-up: use GPU via device="cuda" if available.

Usage::

    cd porecdft
    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \\
        applications/zif8_co2/notebooks/build_vext_mace.py

Dependencies::

    pip install mace-torch    # or conda install -c conda-forge mace-torch
"""
from __future__ import annotations

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

# ── silence pymatgen CIF symmetry warning ─────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", message=".*symmetry_equiv_pos_as_xyz.*")

from applications.zif8_co2 import ZIF8_CIF, CACHE_DIR

# ── physical constants ─────────────────────────────────────────────────────────
EV_TO_K = 11604.518  # 1 eV = 11604.518 K  (eV / k_B)

# ── simulation parameters ──────────────────────────────────────────────────────
T_K        = 298.0    # temperature (K)
SPACING_ANG = 0.7     # grid spacing (Å)
N_ORIENT   = 20       # Fibonacci orientations
BATCH_SIZE = 64       # grid points evaluated per MACE call (reduce if OOM)

# EPM2 CO2 body-frame geometry (Å): C at origin, O at ±1.149 Å along z
CO2_BODY = np.array([
    [0.0, 0.0,  1.149],   # O
    [0.0, 0.0,  0.0  ],   # C
    [0.0, 0.0, -1.149],   # O
])
CO2_SYMBOLS = ["O", "C", "O"]


# ── helpers ────────────────────────────────────────────────────────────────────

def _fibonacci_directions(n: int) -> np.ndarray:
    """n quasi-uniform unit vectors on S² (Fibonacci sphere), shape (n, 3)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    i   = np.arange(n)
    z   = 1.0 - 2.0 * (i + 0.5) / n
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
    theta = 2.0 * np.pi * i / phi
    return np.stack([r_xy * np.cos(theta), r_xy * np.sin(theta), z], axis=-1)


def _rotation_z_to(u: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix R such that R @ ẑ = u (Rodrigues, stable for u ≈ ẑ)."""
    z_hat = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_hat, u)
    s = np.linalg.norm(v)
    c = float(np.dot(z_hat, u))
    if s < 1e-12:
        return np.eye(3) if c > 0.0 else np.diag([1.0, -1.0, -1.0])
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + V + V @ V * ((1.0 - c) / s ** 2)


def _build_grid(lattice: np.ndarray, spacing: float):
    """Return grid_xyz (N,3) and shape (3,) for a regular grid over the unit cell."""
    a, b, c = [np.linalg.norm(v) for v in lattice]
    nx, ny, nz = max(1, int(round(a / spacing))), max(1, int(round(b / spacing))), max(1, int(round(c / spacing)))
    xs = np.linspace(0.0, 1.0, nx, endpoint=False)
    ys = np.linspace(0.0, 1.0, ny, endpoint=False)
    zs = np.linspace(0.0, 1.0, nz, endpoint=False)
    fx, fy, fz = np.meshgrid(xs, ys, zs, indexing="ij")
    frac = np.stack([fx.ravel(), fy.ravel(), fz.ravel()], axis=-1)  # (N, 3)
    xyz  = frac @ lattice                                            # (N, 3) Å
    return xyz, (nx, ny, nz)


def _energy_ase(calc, atoms):
    """Return potential energy (eV) of an ASE Atoms object."""
    atoms.calc = calc
    return atoms.get_potential_energy()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import torch
    from pymatgen.io.cif import CifParser
    from pymatgen.io.ase import AseAtomsAdaptor

    try:
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise SystemExit(
            "mace-torch is not installed.\n"
            "Install with:  pip install mace-torch\n"
            f"Original error: {exc}"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading ZIF-8 from {ZIF8_CIF}")

    # 1. Load ZIF-8
    parser   = CifParser(ZIF8_CIF)
    zif8_pmg = parser.parse_structures(primitive=False)[0]
    adaptor  = AseAtomsAdaptor()
    host_ase = adaptor.get_atoms(zif8_pmg)
    lattice  = np.array(host_ase.get_cell())   # (3, 3) Å
    print(f"ZIF-8: {len(host_ase)} atoms, cell = {np.diag(lattice)}")

    # 2. MACE calculator
    print("Loading MACE-MP-0 …")
    t0 = time.time()
    calc = mace_mp(model="medium", device=device, default_dtype="float32")
    print(f"  loaded in {time.time()-t0:.1f}s")

    # 3. Host energy (once)
    print("Computing E_host …")
    host_ase_copy = host_ase.copy()
    E_host = _energy_ase(calc, host_ase_copy)
    print(f"  E_host = {E_host:.4f} eV")

    # 4. Isolated CO2 energy (once)
    from ase import Atoms as AseAtoms
    co2_vac = AseAtoms(symbols=CO2_SYMBOLS, positions=CO2_BODY, pbc=False)
    co2_vac.calc = calc
    E_co2_vac = _energy_ase(calc, co2_vac)
    print(f"  E_CO2_vac = {E_co2_vac:.4f} eV")

    # 5. Grid
    grid_xyz, shape = _build_grid(lattice, SPACING_ANG)
    N_grid = len(grid_xyz)
    dV = np.prod([np.linalg.norm(lattice[i]) for i in range(3)]) / N_grid  # Å³
    print(f"Grid: {shape} = {N_grid} points, dV = {dV:.3f} Å³")

    # 6. Orientation sampling
    dirs = _fibonacci_directions(N_ORIENT)
    print(f"Orientations: {N_ORIENT} Fibonacci directions")

    V_orient = np.zeros((N_ORIENT, N_grid), dtype=np.float32)

    for i_orient, u in enumerate(dirs):
        R = _rotation_z_to(u)
        co2_rotated = CO2_BODY @ R.T  # shape (3, 3), each row is one site

        t_orient = time.time()
        E_int = np.empty(N_grid, dtype=np.float32)

        for batch_start in range(0, N_grid, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, N_grid)
            for j in range(batch_start, batch_end):
                r0 = grid_xyz[j]
                co2_pos = r0 + co2_rotated            # (3, 3) absolute positions
                combined_pos = np.vstack([host_ase.get_positions(), co2_pos])
                combined_sym = list(host_ase.get_chemical_symbols()) + CO2_SYMBOLS

                combined = AseAtoms(
                    symbols=combined_sym,
                    positions=combined_pos,
                    cell=host_ase.get_cell(),
                    pbc=True,
                )
                combined.calc = calc
                E_comb = combined.get_potential_energy()
                E_int[j] = float(E_comb - E_host - E_co2_vac)

            if (batch_start // BATCH_SIZE) % 20 == 0:
                frac = batch_end / N_grid
                elapsed = time.time() - t_orient
                eta = elapsed / max(frac, 1e-6) * (1 - frac)
                print(f"  orient {i_orient+1}/{N_ORIENT}  "
                      f"pt {batch_end}/{N_grid}  "
                      f"ETA {eta/60:.1f} min", end="\r", flush=True)

        print(f"  orient {i_orient+1}/{N_ORIENT}  done in {(time.time()-t_orient)/60:.1f} min  "
              f"E_int: [{E_int.min()*EV_TO_K:.0f}, {E_int.max()*EV_TO_K:.0f}] K")
        V_orient[i_orient] = E_int * EV_TO_K   # eV → K

    # 7. Boltzmann average over orientations
    beta = 1.0 / T_K
    log_sum = np.log(np.mean(np.exp(-beta * V_orient.astype(np.float64)), axis=0))
    vext_avg = -T_K * log_sum                  # shape (N_grid,) in K
    vext_3d  = vext_avg.reshape(shape)

    # 8. Save
    out_path = CACHE_DIR / f"vext_mace_T{T_K:.0f}K.npy"
    np.save(out_path, {
        "vext_avg": vext_3d,
        "shape":    shape,
        "T_K":      T_K,
        "spacing":  SPACING_ANG,
        "n_orient": N_ORIENT,
        "lattice":  lattice,
        "method":   "MACE-MP-0 medium, EPM2 CO2, Fibonacci N_Ω=20",
    })
    print(f"\nSaved Vext to {out_path}")
    print(f"  shape {shape}, min={vext_3d.min():.0f} K, max={vext_3d.max():.0f} K, "
          f"mean={vext_3d.mean():.1f} K")


if __name__ == "__main__":
    main()
