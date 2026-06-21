"""Tutorial 1 — Ar/ZIF-8 isotherm at 77 K.

Validates porecdft's FMT-aWBII engine against the Stierle & Gross 2024 SI
benchmark (cf. their Fig. 5).  All inputs (ZIF-8.cif, DREIDING.dat,
PC-SAFT argon) come from ``tutorials/data/``.

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \
        tutorials/01_argon_in_zif8/run.py
"""
from __future__ import annotations

import os
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

from data_loader import (
    STRUCT_DIR, FF_DIR,
    load_dreiding_ff, load_pcsaft_fluid, make_pcsaft_eos,
)

from porecdft.io import read_cif
from porecdft.io.forcefield import FFEntry
from porecdft.forcefield import LJPotential, CompositePotential
from porecdft.fluid.base import Fluid
from porecdft.functional import (
    make_k_grid, make_fmt_weights_hat, compute_weighted_densities,
    compute_c1, bulk_c1,
)
from porecdft.solver import anderson_solve
from porecdft.structure import build_supercell
from porecdft.vext import build_vext_on_grid, fibonacci_rotations

# ── constants ───────────────────────────────────────────────────────────────
T_K = 77.0
P_BAR = np.logspace(-4, 0, 16)     # 10⁻⁴ → 1 bar
SIGMA_HS = 3.405                   # Å — argon PC-SAFT sigma (= LJ σ_Ar)
AVOGADRO = 6.022e23

FIG_DIR  = _REPO / "tutorials" / "figures"
FIG_DIR.mkdir(exist_ok=True)
CACHE    = Path(__file__).parent / "vext_zif8_Ar_77K.npy"


# ── 1. Host + force field ───────────────────────────────────────────────────

def load_host_and_ff():
    cif = STRUCT_DIR / "ZIF-8.cif"
    print(f"Loading host:  {cif}")
    host = read_cif(str(cif))
    # ZIF-8 has Zn + N + C + H; DREIDING provides all of them but ZIF-8.cif
    # ships only the *framework* atoms (no extra-framework charges), so the
    # DREIDING LJ-only set is the right force field.
    host_ff = load_dreiding_ff(FF_DIR / "DREIDING.dat")
    print(f"  Host: {len(host.species)} atoms, "
          f"species = {set(host.species)}, "
          f"cell = {host.lattice.diagonal()}")
    print(f"  DREIDING FF parsed: {list(host_ff)}")
    # Set framework charges to zero (DREIDING + LJ-only).
    host = host.assign_charges(
        {s: 0.0 for s in set(host.species)},
        source="ZIF-8 + DREIDING LJ-only",
    )
    return host, host_ff


# ── 2. Argon as a single-site fluid ─────────────────────────────────────────

def build_fluid():
    m, sigma, eps, M = load_pcsaft_fluid("argon")
    print(f"\nArgon PC-SAFT: m={m}, σ={sigma}, ε/k={eps}, M={M}")
    # Argon is monatomic — a single body-frame site at the origin with no
    # charge and no quadrupole.
    ar = Fluid(
        name="Argon",
        body_sites=np.zeros((1, 3)),
        site_labels=["Ar"],
        ff={"Ar": FFEntry("Ar", sigma, eps, source="noble_gases.json")},
        charges={"Ar": 0.0},
        theta_zz=0.0,
        molar_mass=M,
    )
    eos = make_pcsaft_eos("argon")
    return ar, eos


# ── 3. Build Vext on a 3D grid ──────────────────────────────────────────────

def build_vext(host, fluid, host_ff, T_K, n_orient=20):
    if CACHE.exists():
        data = np.load(CACHE, allow_pickle=True).item()
        print(f"\nLoaded Vext cache: {CACHE}")
        return data
    # Let build_vext_on_grid do the PBC replication internally; use 2×2×2
    # because ZIF-8's 16.99 Å cell ≥ LJ cutoff.  Grid spacing 1.0 Å keeps
    # this affordable on CPU for the tutorial (≈ 30 s instead of hours).
    potential = CompositePotential([
        LJPotential(host_ff=host_ff, fluid_ff=fluid.ff, cutoff=15.0),
    ])
    print(f"\nBuilding Vext on 3D grid (≈ 30 s)...")
    data = build_vext_on_grid(
        host, fluid, potential,
        orientations=fibonacci_rotations(n_orient),
        spacing=1.0,
        pbc_supercell=(2, 2, 2),
        temperature_K=T_K,
        cache_path=str(CACHE),
        v_reject_below_K=-5000.0,
        v_cap_above_K=+10000.0,
        averaging="boltzmann",
    )
    return data


# ── 4. FMT-aWBII isotherm sweep ─────────────────────────────────────────────

def run_isotherm(Vext_K, dV, access, P_arr, T_K, eos, host_lattice, framework_mass_g):
    Nx, Ny, Nz = Vext_K.shape
    Lx, Ly, Lz = np.linalg.norm(host_lattice, axis=1)
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    KX, KY, KZ, K = make_k_grid((Nx, Ny, Nz), dx=dx, dy=dy, dz=dz)
    w2_hat, w3_hat, w2vec_hat = make_fmt_weights_hat(K, KX, KY, KZ, SIGMA_HS)
    RHO_MAX = 0.45 * 6.0 / (np.pi * SIGMA_HS ** 3)

    def c1_fn(rho):
        wd = compute_weighted_densities(rho, w2_hat, w3_hat, w2vec_hat, SIGMA_HS)
        return np.asarray(compute_c1(rho, wd, w2_hat, w3_hat, w2vec_hat,
                                      SIGMA_HS, model="aWBII"))

    to_mmol_per_g = (1.0 / AVOGADRO) * 1000.0 / framework_mass_g

    def boltz_init(rho_b, beta):
        ri = rho_b * np.exp(np.clip(-beta * Vext_K, -50.0, 20.0)) * access
        return np.minimum(ri, RHO_MAX)

    N_arr = np.empty(len(P_arr))
    rho_prev = None
    rho_prev_b = None
    print("\nIsotherm sweep:")
    print(f"  {'P (bar)':>10}  {'ρ_b (Å⁻³)':>12}  {'iters':>6}  {'N (mmol/g)':>11}")
    for i, P in enumerate(P_arr):
        rho_b = float(eos.bulk_density(P, T_K))
        c1_b  = bulk_c1(rho_b, SIGMA_HS, model="aWBII")
        beta  = 1.0 / T_K
        if rho_prev is not None and rho_prev_b:
            rho0 = np.where(access, np.clip(rho_prev * (rho_b / rho_prev_b),
                                            1e-16, RHO_MAX), 1e-16)
        else:
            rho0 = boltz_init(rho_b, beta)

        res = anderson_solve(
            rho_init=rho0, rho_bulk=rho_b,
            Vext_K=Vext_K, temperature_K=T_K,
            c1_callable=c1_fn, c1_bulk=c1_b,
            m=6, beta=0.3, max_iter=800, tol=1e-4,
            accessibility_mask=access, log_clip=25.0,
            safeguard_alpha=0.02, picard_warmup=30, step_clip=2.0,
            rho_max=RHO_MAX,
        )
        rho_prev, rho_prev_b = res.rho.copy(), rho_b
        N_arr[i] = float(res.rho.sum() * dV) * to_mmol_per_g
        print(f"  {P:10.4e}  {rho_b:12.4e}  {res.iterations:6d}  {N_arr[i]:11.3f}")
    return N_arr


# ── main ───────────────────────────────────────────────────────────────────

def main():
    host, host_ff = load_host_and_ff()
    fluid, eos    = build_fluid()
    vd            = build_vext(host, fluid, host_ff, T_K)
    Vext_K        = vd["vext_avg"]
    dV            = vd["dV"]
    access        = np.isfinite(Vext_K) & (Vext_K < 50.0 * T_K)

    atomic_mass = {"Zn": 65.38, "N": 14.007, "C": 12.011, "H": 1.008}
    fw_amu = sum(atomic_mass.get(s, 0.0) for s in host.species)
    fw_g   = fw_amu / AVOGADRO

    N_mmol_g = run_isotherm(Vext_K, dV, access, P_BAR, T_K, eos,
                            host.lattice, fw_g)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(P_BAR, N_mmol_g, "o-", color="#1f77b4", lw=2.0, ms=6,
                label="porecdft (FMT-aWBII + PC-SAFT)")
    ax.set_xlabel("Pressure (bar)", fontsize=12)
    ax.set_ylabel(r"Ar uptake (mmol g$^{-1}$)", fontsize=12)
    ax.set_title("Ar in ZIF-8 at 77 K\nTutorial 1 — validation vs Stierle 2024 (Fig. 5)",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, ls=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.9)
    out = FIG_DIR / "01_argon_in_zif8.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
