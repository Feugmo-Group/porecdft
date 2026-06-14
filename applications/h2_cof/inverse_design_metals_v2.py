"""Inverse design: H₂ deliverable capacity for 5 metal centres in COF-333.

Proper implementation using actual Morse parameters for each metal
(D_e, a, r_e all differ per metal, from Pramudya & Mendoza-Cortes 2016).

For each metal M in {Co, Fe, Ni, Cu, Mn}:
  1. In-silico substitute Co → M in COF-333-CoCl₂ (keep framework geometry fixed)
  2. Build Vext(r) using M's Morse params + DREIDING LJ for organic atoms
  3. Run full isotherm (16 P points, 1-500 bar) with Anderson solver at T=298 K
  4. Compute DC = N(100 bar) − N(5 bar) [deliverable capacity, on-board storage]

Key physics:
  Pramudya 2016 finds Co best because a_Co=0.850 Å⁻¹ (broadest well) captures
  more H₂ molecules in 3D pore — width matters more than depth.

Outputs
-------
  results/isotherm_h2_cof333_{metal}_v2.npz  — per-metal (P, N) arrays
  results/dc_metals_v2.npz                   — DC, Henry constants, timings
  figures/h2_inverse_design_5metals_v2.png   — 4-panel comparison figure
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARENT    = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try: sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

from pymatgen.core import Structure

from porecdft.structure.host import HostAtoms
from porecdft.structure.supercell import build_supercell
from porecdft.eos import H2_PR
from porecdft.functional import LJWDAFunctional
from porecdft.solver import anderson_solve

# ── constants ────────────────────────────────────────────────────────────────
KCAL_TO_K  = 503.228
SIGMA_H2   = 2.83
EPSILON_H2 = 59.7
RCUT_H2    = 5.0 * SIGMA_H2
T_K        = 298.0

MORSE_METALS = {"Co", "Fe", "Ni", "Cu", "Mn"}

MORSE_PARAMS = {
    "Co": dict(D_e=2*0.879*KCAL_TO_K, a=0.850, r_e=2.985, cutoff=12.0),
    "Fe": dict(D_e=2*1.092*KCAL_TO_K, a=1.180, r_e=3.015, cutoff=12.0),
    "Ni": dict(D_e=2*1.154*KCAL_TO_K, a=1.210, r_e=3.207, cutoff=12.0),
    "Cu": dict(D_e=2*0.818*KCAL_TO_K, a=1.462, r_e=2.931, cutoff=12.0),
    "Mn": dict(D_e=2*0.994*KCAL_TO_K, a=0.990, r_e=3.015, cutoff=12.0),
}

DREIDING = {
    "H":  (2.84642,   7.64893),
    "C":  (3.47299,  47.85620),
    "N":  (3.26256,  38.94920),
    "O":  (3.03315,  48.15810),
    "F":  (3.09320,  36.48345),
    "Al": (3.91104, 155.99820),
    "Si": (3.80414, 155.99820),
    "Br": (3.51905, 186.19140),
    "Cu": (3.11369,   2.51610),
    "Zn": (4.04468,  27.67710),
    "Co": (2.55800,   7.05000),
    "Cl": (3.52000, 114.23000),
}

MASS_MAP = {
    "H": 1.00784, "C": 12.0107, "N": 14.0067, "O": 15.999,
    "Co": 58.933, "Cl": 35.45,  "F": 18.998,  "Al": 26.9815,
    "Si": 28.0855,"Br": 79.904, "Cu": 63.546,  "Zn": 65.38,
    "Fe": 55.845, "Ni": 58.693, "Mn": 54.938,
}

METALS   = ["Co", "Fe", "Ni", "Cu", "Mn"]
COLORS   = {"Co": "#e41a1c", "Fe": "#ff7f00", "Ni": "#4daf4a",
            "Cu": "#984ea3", "Mn": "#377eb8"}
P_ISO    = np.array([1, 5, 10, 20, 40, 60, 80, 100, 120, 150,
                     200, 250, 300, 400, 450, 500], dtype=float)
P_LOW    = 5.0    # bar — charging pressure
P_HIGH   = 100.0  # bar — discharge pressure
STRUCTURES_DIR = _REPO_ROOT / "applications/h2_cof/structures"
RESULTS_DIR    = _REPO_ROOT / "applications/h2_cof/results"
FIGURES_DIR    = _REPO_ROOT / "applications/h2_cof/figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ── structure loading + metal substitution ────────────────────────────────────

def load_host_substituted(metal: str) -> HostAtoms:
    """Load COF-333-CoCl₂ and substitute Co → metal (in-silico)."""
    cif = STRUCTURES_DIR / "COF-333-CoCl2.cif"
    pmg = Structure.from_file(str(cif))
    # Replace every Co site with the target metal
    species = [metal if str(s) == "Co" else str(s) for s in pmg.species]
    return HostAtoms(
        positions=pmg.cart_coords.copy(),
        species=species,
        charges=np.zeros(pmg.num_sites),
        lattice=pmg.lattice.matrix.copy(),
        source=str(cif),
    )


# ── Vext builder ──────────────────────────────────────────────────────────────

def build_vext(host: HostAtoms, metal: str,
               grid_spacing: float = 0.25 * SIGMA_H2,
               supercell: tuple = (3, 3, 3),
               cache_path: Path | None = None) -> tuple:
    if cache_path and cache_path.exists():
        data = np.load(cache_path, allow_pickle=True).item()
        print(f"  [{metal}] Vext loaded from cache", flush=True)
        return (data["vext_3d"], tuple(data["n_pts"]),
                data["spacings"], float(data["dV"]))

    nx, ny, nz = supercell
    host_sc = build_supercell(host, nx, ny, nz)
    shift = (-(nx//2)*host.lattice[0] - (ny//2)*host.lattice[1]
             - (nz//2)*host.lattice[2])
    pos_sc  = host_sc.positions + shift
    spec_sc = host_sc.species

    lengths = np.linalg.norm(host.lattice, axis=1)
    n_pts   = tuple(max(2, int(np.ceil(L / grid_spacing))) for L in lengths)
    spacings = np.array([lengths[i] / n_pts[i] for i in range(3)])
    dV       = float(spacings.prod())

    fx = np.linspace(0, 1, n_pts[0], endpoint=False)
    fy = np.linspace(0, 1, n_pts[1], endpoint=False)
    fz = np.linspace(0, 1, n_pts[2], endpoint=False)
    Fx, Fy, Fz = np.meshgrid(fx, fy, fz, indexing="ij")
    grid_xyz = (np.stack([Fx, Fy, Fz], axis=-1).reshape(-1, 3) @ host.lattice)

    lj_params = {}
    for el in set(spec_sc):
        if el in MORSE_METALS or el not in DREIDING:
            continue
        s, e = DREIDING[el]
        lj_params[el] = (0.5*(SIGMA_H2 + s), float(np.sqrt(EPSILON_H2 * e)))

    vext = np.zeros(grid_xyz.shape[0])
    n_atoms = len(spec_sc)
    print(f"  [{metal}] Building Vext on {n_pts[0]}×{n_pts[1]}×{n_pts[2]} grid "
          f"({n_atoms} atoms in {nx}×{ny}×{nz} SC)...", flush=True)

    for i, (el, pos_i) in enumerate(zip(spec_sc, pos_sc)):
        dr = grid_xyz - pos_i
        r  = np.sqrt(np.einsum("gi,gi->g", dr, dr).clip(1e-8))
        if el in MORSE_METALS:
            mp = MORSE_PARAMS[el]
            mask = r < mp["cutoff"]
            if mask.any():
                x = np.exp(-mp["a"] * (r[mask] - mp["r_e"]))
                vext[mask] += np.clip(mp["D_e"] * ((1-x)**2 - 1), -mp["D_e"], 1e5)
        elif el in lj_params:
            sigma_sf, eps_sf = lj_params[el]
            mask = r < RCUT_H2
            if mask.any():
                sr6 = (sigma_sf / r[mask])**6
                vext[mask] += 4.0 * eps_sf * (sr6**2 - sr6)

    vext_3d = vext.reshape(n_pts)
    if cache_path:
        np.save(cache_path,
                {"vext_3d": vext_3d, "n_pts": np.array(n_pts),
                 "spacings": spacings, "dV": dV})
        print(f"  [{metal}] Vext cached to {cache_path.name}", flush=True)
    return vext_3d, n_pts, spacings, dV


# ── isotherm runner ───────────────────────────────────────────────────────────

def run_isotherm(metal: str, vext_3d: np.ndarray, dV: float,
                 dx: float, dy: float, dz: float,
                 wda: LJWDAFunctional,
                 cache_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    if cache_path and cache_path.exists():
        d = np.load(cache_path)
        print(f"  [{metal}] isotherm loaded from cache ({len(d['N'])} pts)",
              flush=True)
        return np.array(d["P"]), np.array(d["N"])

    import jax.numpy as jnp

    rho_b0  = H2_PR.bulk_density(1.0, T_K)
    c1_b0   = float(wda.c1_bulk(rho_b0))
    rho_max = float(np.exp(-c1_b0) * rho_b0 * 15)
    access  = vext_3d < 50.0 * T_K

    # Pre-warm weight cache before JIT loop
    wda._get_weights(vext_3d.shape, dx, dy, dz)
    c1_fn = lambda rho: np.asarray(wda.c1(jnp.asarray(rho), dx, dy, dz))

    N_arr, rho_prev, rho_prev_b = [], None, None
    t_start = time.perf_counter()

    for P in P_ISO:
        rho_b = H2_PR.bulk_density(P, T_K)
        c1_b  = float(wda.c1_bulk(rho_b))

        if rho_prev is not None:
            scale = np.clip(rho_b / max(rho_prev_b, 1e-30), 0.5, 4.0)
            rho0  = np.where(access,
                             np.clip(rho_prev * scale, 1e-16, rho_max), 1e-16)
        else:
            exp_arg = np.clip(-vext_3d / T_K, -50.0, 20.0)
            rho0 = np.where(access,
                            np.clip(rho_b * np.exp(exp_arg), 1e-16, rho_max),
                            1e-16)

        res = anderson_solve(
            rho0, rho_b, vext_3d, T_K, c1_fn, c1_b,
            m=8, beta=0.1, max_iter=10000, tol=1e-5,
            accessibility_mask=access, safeguard_alpha=0.01,
            picard_warmup=100, rho_max=rho_max,
        )
        N = float(res.rho.sum() * dV)

        # Monotonicity guard: if N drops >50% reject and re-init from Boltzmann
        if N_arr and N < 0.5 * N_arr[-1]:
            print(f"  [{metal}] P={P:.0f} bar: N={N:.1f} failed monotone "
                  f"(prev={N_arr[-1]:.1f}), re-init from Boltzmann...", flush=True)
            exp_arg = np.clip(-vext_3d / T_K, -50.0, 20.0)
            rho0 = np.where(access,
                            np.clip(rho_b * np.exp(exp_arg), 1e-16, rho_max),
                            1e-16)
            res = anderson_solve(
                rho0, rho_b, vext_3d, T_K, c1_fn, c1_b,
                m=8, beta=0.05, max_iter=20000, tol=1e-5,
                accessibility_mask=access, safeguard_alpha=0.005,
                picard_warmup=200, rho_max=rho_max,
            )
            N = float(res.rho.sum() * dV)

        rho_prev, rho_prev_b = res.rho.copy(), rho_b
        N_arr.append(N)
        t_el = time.perf_counter() - t_start
        print(f"  [{metal}] P={P:5.0f} bar  N={N:7.2f} mol/uc  "
              f"conv={res.converged}  iters={res.iterations}  t={t_el:.0f}s",
              flush=True)

    P_arr = P_ISO[:len(N_arr)]
    N_arr = np.array(N_arr)
    if cache_path:
        np.savez(cache_path, P=P_arr, N=N_arr)
        print(f"  [{metal}] isotherm cached", flush=True)
    return P_arr, N_arr


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(results: dict, out_path: Path):
    MASS_UC = {
        m: sum(MASS_MAP.get(el, 0)
               for el in load_host_substituted(m).species)
        for m in METALS
    }

    def to_wt(N_arr, m):
        m_h2 = N_arr * 2.016  # g/mol × mol/uc = g/uc
        m_tot = m_h2 + MASS_UC[m]
        return 100.0 * m_h2 / m_tot

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(
        "H$_2$/COF-333-MCl$_2$, $T=298\\,$K: 5 metal centres (proper Morse params)",
        fontsize=13, fontweight="bold"
    )

    # ── Panel A: isotherms in mol/uc ──
    ax = axes[0, 0]
    for m in METALS:
        P, N = results[m]["P"], results[m]["N"]
        ax.plot(P, N, color=COLORS[m], marker="o", ms=4, label=m)
    ax.set_xlabel("Pressure (bar)")
    ax.set_ylabel("$N_{\\rm ads}$ (mol / u.c.)")
    ax.set_xlim(0, 510)
    ax.legend(title="Metal")
    ax.set_title("(a) Adsorption isotherms")

    # ── Panel B: isotherms in wt% ──
    ax = axes[0, 1]
    for m in METALS:
        P, N = results[m]["P"], results[m]["N"]
        ax.plot(P, to_wt(N, m), color=COLORS[m], marker="o", ms=4, label=m)
    ax.set_xlabel("Pressure (bar)")
    ax.set_ylabel("H$_2$ uptake (wt %)")
    ax.set_xlim(0, 510)
    ax.legend(title="Metal")
    ax.set_title("(b) Gravimetric uptake")

    # ── Panel C: deliverable capacity bar ──
    ax = axes[1, 0]
    dc_vals = {m: results[m]["DC"] for m in METALS}
    sorted_m = sorted(dc_vals, key=lambda x: -dc_vals[x])
    bars = ax.bar([m for m in sorted_m],
                  [dc_vals[m] for m in sorted_m],
                  color=[COLORS[m] for m in sorted_m], edgecolor="k", lw=0.8)
    for bar, m in zip(bars, sorted_m):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{dc_vals[m]:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"DC = N({P_HIGH:.0f}) − N({P_LOW:.0f}) bar  (mol / u.c.)")
    ax.set_title("(c) Deliverable capacity")

    # ── Panel D: Morse well shapes ──
    ax = axes[1, 1]
    r_vals = np.linspace(2.0, 7.0, 300)
    for m in METALS:
        mp = MORSE_PARAMS[m]
        x = np.exp(-mp["a"] * (r_vals - mp["r_e"]))
        v = mp["D_e"] * ((1 - x)**2 - 1)
        ax.plot(r_vals, np.clip(v, -mp["D_e"]*1.05, 0),
                color=COLORS[m], label=f"{m} ($D_e$={mp['D_e']:.0f}K, "
                f"$a$={mp['a']:.3f}, $r_e$={mp['r_e']:.3f})")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("$r$ (Å)")
    ax.set_ylabel("$V_{\\rm Morse}$ (K)")
    ax.set_xlim(2.0, 7.0)
    ax.legend(fontsize=7, title="Metal")
    ax.set_title("(d) Morse potential shapes")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    wda = LJWDAFunctional(sigma=SIGMA_H2, epsilon=EPSILON_H2, temperature_K=T_K)
    results = {}

    for metal in METALS:
        print(f"\n{'='*60}", flush=True)
        print(f"  Metal: {metal}", flush=True)
        print(f"{'='*60}", flush=True)

        host = load_host_substituted(metal)
        n_metal = sum(1 for s in host.species if s == metal)
        print(f"  {len(host.species)} atoms, {n_metal} {metal} sites", flush=True)

        vext_cache = RESULTS_DIR / f"vext_cache_COF-333-{metal}Cl2_v2.npy"
        vext_3d, n_pts, spacings, dV = build_vext(
            host, metal, cache_path=vext_cache)
        dx, dy, dz = float(spacings[0]), float(spacings[1]), float(spacings[2])

        iso_cache = RESULTS_DIR / f"isotherm_h2_cof333_{metal.lower()}_v2.npz"
        P_arr, N_arr = run_isotherm(
            metal, vext_3d, dV, dx, dy, dz, wda, cache_path=iso_cache)

        dc = (np.interp(P_HIGH, P_arr, N_arr)
              - np.interp(P_LOW,  P_arr, N_arr))
        henry = float(np.interp(1.0, P_arr, N_arr) /
                      H2_PR.bulk_density(1.0, T_K) / dV / vext_3d.size)
        results[metal] = {"P": P_arr, "N": N_arr, "DC": dc, "K_H": henry}
        print(f"  DC = {dc:.2f} mol/u.c.  K_H = {henry:.2f}", flush=True)

    # Summary table
    print("\n" + "="*60)
    print(f"{'Metal':>6}  {'De(K)':>7}  {'a(Å⁻¹)':>7}  {'re(Å)':>6}  "
          f"{'N(5)':>7}  {'N(100)':>8}  {'DC':>8}")
    for m in METALS:
        mp = MORSE_PARAMS[m]
        n5   = np.interp(5,   results[m]["P"], results[m]["N"])
        n100 = np.interp(100, results[m]["P"], results[m]["N"])
        print(f"{m:>6}  {mp['D_e']:7.0f}  {mp['a']:7.3f}  {mp['r_e']:6.3f}  "
              f"{n5:7.2f}  {n100:8.2f}  {results[m]['DC']:8.2f}")

    # Save summary
    np.savez(RESULTS_DIR / "dc_metals_v2.npz",
             metals=np.array(METALS),
             de=np.array([MORSE_PARAMS[m]["D_e"] for m in METALS]),
             a=np.array([MORSE_PARAMS[m]["a"]    for m in METALS]),
             re=np.array([MORSE_PARAMS[m]["r_e"]  for m in METALS]),
             dc=np.array([results[m]["DC"]         for m in METALS]),
             kh=np.array([results[m]["K_H"]        for m in METALS]))

    make_figure(results, FIGURES_DIR / "h2_inverse_design_5metals_v2.png")


if __name__ == "__main__":
    main()
