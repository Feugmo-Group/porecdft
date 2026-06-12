"""Tutorial 3 — CO2 / Dha-Tph COF isotherm at 298 K.

Stierle & Gross 2024 SI benchmark (cf. Fig. 9).
This is the first tutorial that exercises the **smeared-Coulomb** module,
since the Dha-Tph CIF ships with QEq partial charges on every atom and
CO2 carries the standard EPM2 charge model.

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \
        tutorials/03_co2_in_dha_cof/run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO.parent), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, str(_REPO / "tutorials"))

from data_loader import STRUCT_DIR, FF_DIR, load_dreiding_ff
from porecdft.fluid import EPM2_CO2
from porecdft.io import read_cif
from porecdft.forcefield import (
    LJPotential, CoulombPotential, QuadrupoleEFGPotential, CompositePotential,
)
from porecdft.functional import (
    make_k_grid, make_fmt_weights_hat, compute_weighted_densities,
    compute_c1, bulk_c1,
)
from porecdft.eos import CO2_PCSAFT
from porecdft.solver import anderson_solve
from porecdft.structure import build_supercell
from porecdft.vext import build_vext_on_grid, fibonacci_rotations

T_K = 298.0
P_BAR = np.logspace(-2, 1.5, 14)        # 0.01 … ~32 bar
SIGMA_HS = 3.017
AVOGADRO = 6.022e23
ATOMIC_MASS = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999}

FIG_DIR = _REPO / "tutorials" / "figures"
CACHE   = Path(__file__).parent / "vext_dha_CO2_298K.npy"


def main():
    cif = STRUCT_DIR / "Dha_Tph_QEq.cif"
    print(f"Loading host: {cif}")
    host = read_cif(str(cif))
    print(f"  {len(host.species)} atoms; cell = {host.lattice.diagonal()}")
    print(f"  species set = {set(host.species)}")
    # The CIF carries QEq partial charges — read_cif extracts them if present.
    # If not, fall back to neutral.
    if np.allclose(host.charges, 0.0):
        print("  WARNING: CIF has no _atom_site_charge column — using LJ-only.")

    host_ff = load_dreiding_ff(FF_DIR / "DREIDING.dat")
    print(f"DREIDING FF: {list(host_ff)}")

    fluid = EPM2_CO2
    eos   = CO2_PCSAFT
    print(f"CO2 EPM2 (3-site) + PC-SAFT bulk EOS.")

    if CACHE.exists():
        vd = np.load(CACHE, allow_pickle=True).item()
        print(f"Vext loaded from cache: {CACHE}")
    else:
        potential = CompositePotential([
            LJPotential(host_ff=host_ff, fluid_ff=fluid.ff, cutoff=15.0),
            CoulombPotential(fluid_charges=fluid.charges,
                             method="smeared", gauss_width=2.0, cutoff=15.0),
            QuadrupoleEFGPotential(theta_zz=fluid.theta_zz, cutoff=15.0),
        ])
        print("Building Vext (LJ + smeared Coulomb + Q-EFG)...")
        vd = build_vext_on_grid(host, fluid, potential,
                                orientations=fibonacci_rotations(20),
                                spacing=1.2, pbc_supercell=(2, 2, 2),
                                temperature_K=T_K,
                                cache_path=str(CACHE),
                                v_reject_below_K=-10000.0,
                                v_cap_above_K=+5000.0,
                                averaging="boltzmann")

    Vext_K = vd["vext_avg"]
    dV     = vd["dV"]
    access = np.isfinite(Vext_K) & (Vext_K < 50.0 * T_K)

    Nx, Ny, Nz = Vext_K.shape
    Lx, Ly, Lz = np.linalg.norm(host.lattice, axis=1)
    KX, KY, KZ, K = make_k_grid((Nx, Ny, Nz),
                                 dx=Lx/Nx, dy=Ly/Ny, dz=Lz/Nz)
    w2_hat, w3_hat, w2vec_hat = make_fmt_weights_hat(K, KX, KY, KZ, SIGMA_HS)
    RHO_MAX = 0.45 * 6.0 / (np.pi * SIGMA_HS ** 3)

    def c1_fn(rho):
        wd = compute_weighted_densities(rho, w2_hat, w3_hat, w2vec_hat, SIGMA_HS)
        return np.asarray(compute_c1(rho, wd, w2_hat, w3_hat, w2vec_hat,
                                      SIGMA_HS, model="aWBII"))

    fw_amu = sum(ATOMIC_MASS.get(s, 0.0) for s in host.species)
    fw_g   = fw_amu / AVOGADRO
    to_mmol_per_g = (1.0 / AVOGADRO) * 1000.0 / fw_g

    N_arr = np.empty(len(P_BAR))
    rho_prev = None
    rho_prev_b = None
    print(f"\n{'P (bar)':>10}  {'iters':>6}  {'N (mmol/g)':>11}")
    for i, P in enumerate(P_BAR):
        rho_b = float(eos.bulk_density(P, T_K))
        c1_b  = bulk_c1(rho_b, SIGMA_HS, model="aWBII")
        if rho_prev is not None and rho_prev_b:
            rho0 = np.where(access, np.clip(rho_prev * (rho_b / rho_prev_b),
                                            1e-16, RHO_MAX), 1e-16)
        else:
            rho0 = np.minimum(rho_b * np.exp(np.clip(-Vext_K / T_K, -50, 20)) * access,
                              RHO_MAX)
        res = anderson_solve(rho0, rho_b, Vext_K, T_K, c1_fn, c1_b,
                             m=6, beta=0.3, max_iter=800, tol=1e-4,
                             accessibility_mask=access, log_clip=25.0,
                             safeguard_alpha=0.02, picard_warmup=30,
                             step_clip=2.0, rho_max=RHO_MAX)
        rho_prev, rho_prev_b = res.rho.copy(), rho_b
        N_arr[i] = float(res.rho.sum() * dV) * to_mmol_per_g
        print(f"{P:10.4e}  {res.iterations:6d}  {N_arr[i]:11.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(P_BAR, N_arr, "o-", color="#d62728", lw=2.0, ms=6,
                label="porecdft (FMT-aWBII + smeared Coulomb + Q-EFG)")
    ax.set_xlabel("Pressure (bar)", fontsize=12)
    ax.set_ylabel(r"CO$_2$ uptake (mmol g$^{-1}$)", fontsize=12)
    ax.set_title("CO$_2$ in Dha-Tph COF at 298 K\n"
                 "Tutorial 3 — validation vs Stierle 2024 (Fig. 9)",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, ls=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.9)
    out = FIG_DIR / "03_co2_in_dha_cof.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
