"""Re-Boltzmann-average existing per-orientation shards at additional temperatures.

The MACE evaluations only depend on geometry, not T; so we can generate
V_ext(r; T) at any temperature from the same 20 orientation shards written
by build_vext_mace.py. This script skips MACE entirely and just recomputes
the -kT log <exp(-V/kT)>_Ω average.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARENT    = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try:    sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

from applications.zif8_co2 import CACHE_DIR, ZIF8_CIF


TEMPERATURES = [273.0, 323.0]     # extra temperatures for SI T-dependence figure
GRID_SHAPE   = (12, 12, 12)
SPACING_ANG  = 1.4
N_ORIENT     = 20
METHOD_STR   = "MACE-MP-0 medium, EPM2 CO2, Fibonacci N_Ω=20"


def load_lattice() -> np.ndarray:
    """Read ZIF-8 lattice matrix from the same source used by build_vext_mace.py."""
    from pymatgen.io.cif import CifParser
    parser = CifParser(str(ZIF8_CIF))
    struct = parser.parse_structures(primitive=False)[0]
    return np.asarray(struct.lattice.matrix)


def load_shards() -> np.ndarray:
    """Stack 20 shards into (N_ORIENT, N_grid)."""
    shard_dir = CACHE_DIR / "orient_shards"
    shards = sorted(shard_dir.glob("orient_*.npy"))
    if len(shards) != N_ORIENT:
        raise RuntimeError(f"Expected {N_ORIENT} shards; found {len(shards)}")
    return np.stack([np.load(str(s)) for s in shards], axis=0)


def boltzmann_average(V_orient: np.ndarray, T_K: float) -> np.ndarray:
    """V_ext(r; T) = -T ln <exp(-V/T)>_Ω, with inf-capped V for numerical safety."""
    V64 = V_orient.astype(np.float64)
    V64 = np.where(np.isfinite(V64), V64, 1e6)
    return (-T_K * np.log(np.mean(np.exp(-V64 / T_K), axis=0))).astype(np.float32)


def main() -> None:
    V_orient = load_shards()               # (20, 1728)
    lattice  = load_lattice()
    print(f"Loaded {V_orient.shape[0]} shards, {V_orient.shape[1]} points each")

    for T in TEMPERATURES:
        vext_flat = boltzmann_average(V_orient, T)
        vext_3d   = vext_flat.reshape(GRID_SHAPE)

        finite = vext_3d[np.isfinite(vext_3d)]
        print(f"\nT = {T:.0f} K")
        print(f"  min  = {finite.min():.0f} K")
        print(f"  max  = {finite.max():.0f} K")
        print(f"  mean = {finite.mean():.1f} K")

        out = CACHE_DIR / f"vext_mace_T{T:.0f}K.npy"
        np.save(str(out), {
            "vext_avg": vext_3d,
            "shape":    GRID_SHAPE,
            "T_K":      T,
            "spacing":  SPACING_ANG,
            "n_orient": N_ORIENT,
            "lattice":  lattice,
            "method":   METHOD_STR,
        })
        print(f"  saved → {out.name}")


if __name__ == "__main__":
    main()
