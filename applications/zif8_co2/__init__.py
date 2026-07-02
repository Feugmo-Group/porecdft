"""CO₂ adsorption in ZIF-8 using MACE-MP external potential + porecdft cDFT engine.

This application demonstrates the MLIP → cDFT pipeline:

    1. build_vext_mace.py  : evaluate the MACE-MP-0 universal potential on a
                             3D grid over the ZIF-8 unit cell for 20 Fibonacci
                             orientations of CO₂, Boltzmann-average to V_ext(r;T),
                             and cache to results/vext_cache/.

    2. make_isotherm_zif8.py : load the cached V_ext grid via numpy, set up
                               the FMT-aWBII functional, and run the Anderson
                               solver to compute the CO₂ adsorption isotherm.

Key parameters
--------------
ZIF-8 CIF  : structure/mofs/cif/ZIF-8.cif  (cubic, a = 16.991 Å, 276 atoms)
CO₂ model  : EPM2 3-site (C at origin, O at ±1.149 Å); MACE evaluates the full
             3-atom molecule inserted into the periodic ZIF-8 supercell.
MACE model : MACE-MP-0  (mace-torch or mace-jax, auto-downloaded from HuggingFace)
T          : 298 K (room temperature)
Pressures  : 0.05 – 50 bar
"""
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]

ZIF8_CIF  = _REPO_ROOT / "structure" / "mofs" / "cif" / "ZIF-8.cif"
DATA_DIR  = _HERE
CACHE_DIR = _HERE / "results" / "vext_cache"
FIG_DIR   = _HERE / "figures"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
