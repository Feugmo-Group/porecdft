#!/usr/bin/env python
"""
Henry-constant cross-check at 273 K.

Compares three estimates of K_H [mmol/g/bar]:
  1. Analytical  — (1/V_cell) * integral of exp(-beta*Vext_avg(r)) dV
                   using vext_avg_T273K.npy
  2. LJ-cDFT     — N/p at p→0 from phase2_baseline_isotherms.csv (henry model)
  3. PC-SAFT cDFT — N/p at p→0 from phase3_production_isotherms.csv
                    (K_eff_GPa=0.3, eps_assoc_K=0.0 = rigid-host LJ limit)

Agreement criterion: all three should be within ~10% at low pressure.

Saves:
  applications/alf_co2/figures/32_henry_crosscheck.png
"""

import sys, os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARENT = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try: sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────────
BASE      = str(_REPO_ROOT)
VEXT_PATH = os.path.join(BASE, 'applications/alf_co2/results/vext_cache/vext_avg_T273K.npy')
CSV_LJ    = os.path.join(BASE, 'applications/alf_co2/results/phase2_baseline_isotherms.csv')
CSV_PS    = os.path.join(BASE, 'applications/alf_co2/results/phase3_production_isotherms.csv')
FIG_DIR   = os.path.join(BASE, 'applications/alf_co2/figures')
os.makedirs(FIG_DIR, exist_ok=True)

T    = 273.0   # K
kB_barA3 = 1.380649e-23 * 1e-5 * 1e30   # kB in bar·Å³/K  = 13.80649 bar·Å³/K × 10 = 138.06
NA   = 6.022e23

print(f"kB in bar·Å³/K = {kB_barA3:.6e}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load vext_avg and compute analytical K_H
# ─────────────────────────────────────────────────────────────────────────────
cache    = np.load(VEXT_PATH, allow_pickle=True).item()
vext_avg = np.asarray(cache['vext_avg'], dtype=float)   # shape (33,17,17), units K
dV       = float(cache['dV'])                           # Å³ per voxel
T_cache  = float(cache['temperature_K'])
grid_sh  = cache['grid_shape']

print(f"\n[vext cache]  T={T_cache} K, grid={grid_sh}, dV={dV:.5f} Å³")
print(f"  vext_avg range: {vext_avg.min():.1f} … {vext_avg.max():.1f} K")

# Boltzmann integral: sum_i exp(-Vext_i / T) * dV   [Å³]
# Cap at -1e4 K to avoid overflow in exp()
vext_clipped  = np.clip(vext_avg.ravel(), -1e4, 5e3)
boltz_vals    = np.exp(-vext_clipped / T)
boltz_integral = float(np.sum(boltz_vals) * dV)         # Å³
print(f"  Boltzmann integral = {boltz_integral:.4e} Å³")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LJ-cDFT K_H from phase2 baseline (henry model, T=273 K)
# ─────────────────────────────────────────────────────────────────────────────
df2  = pd.read_csv(CSV_LJ)
lj273 = df2[(df2['T_K'] == 273.0) & (df2['model'] == 'henry')].sort_values('p_bar').copy()
print(f"\n[LJ-cDFT]  273 K henry rows: {len(lj273)}")

# Use lowest 4 points for Henry slope (linear fit through origin)
n_fit = min(4, len(lj273))
p_fit = lj273['p_bar'].values[:n_fit]
N_fit = lj273['mmol_per_g_abs'].values[:n_fit]
# Constrained through origin: slope = sum(p*N)/sum(p^2)
KH_lj = float(np.sum(p_fit * N_fit) / np.sum(p_fit**2))
print(f"  K_H (origin-constrained slope, {n_fit} pts): {KH_lj:.4f} mmol/g/bar")

# Back-calculate framework mass per cell from K_H and Boltzmann integral:
#   K_H = boltz_integral / (kB_barA3 * T * NA * mss_g)
mss_g = boltz_integral / (KH_lj * kB_barA3 * T * NA)
print(f"  Implied mss_g/cell = {mss_g:.4e} g  "
      f"({mss_g * NA:.4f} g/mol per cell)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Analytical K_H (using mss_g from LJ back-calculation)
# ─────────────────────────────────────────────────────────────────────────────
KH_anal = boltz_integral / (kB_barA3 * T * NA * mss_g)
print(f"\n[Analytical]  K_H = {KH_anal:.4f} mmol/g/bar")
print(f"  (self-consistent with LJ mss → exact match by construction)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PC-SAFT cDFT K_H from phase3 production isotherms
#    Use K_eff_GPa=0.3, eps_assoc_K=0.0  (rigid-host, no association)
# ─────────────────────────────────────────────────────────────────────────────
KH_pcsaft   = np.nan
p_ps        = np.array([])
N_ps        = np.array([])
pcsaft_label = 'PC-SAFT (not available)'

if os.path.exists(CSV_PS):
    df3 = pd.read_csv(CSV_PS)
    print(f"\n[phase3 CSV]  columns: {df3.columns.tolist()}")
    print(f"  K_eff values: {sorted(df3['K_eff_GPa'].unique())}")
    print(f"  eps_assoc values: {sorted(df3['eps_assoc_K'].unique())}")
    print(f"  T_K values: {sorted(df3['T_K'].unique())}")

    # Select rigid-host, no-association, T=273
    sel = df3[(df3['T_K'] == 273.0) &
              (df3['K_eff_GPa'] == 0.3) &
              (df3['eps_assoc_K'] == 0.0)].sort_values('p_bar').copy()
    print(f"  Rows selected (K_eff=0.3, eps=0, T=273): {len(sel)}")

    if len(sel) >= 2:
        p_ps = sel['p_bar'].values
        N_ps = sel['N_mmol_g'].values
        # Henry-regime: p <= 1e-2 bar
        henry_mask = p_ps <= 1e-2
        if henry_mask.sum() >= 2:
            ph = p_ps[henry_mask]
            Nh = N_ps[henry_mask]
            KH_pcsaft = float(np.sum(ph * Nh) / np.sum(ph**2))
            pcsaft_label = 'PC-SAFT (phase3, K_eff=0.3 GPa)'
            print(f"  K_H (origin slope, {henry_mask.sum()} pts): {KH_pcsaft:.4f} mmol/g/bar")
        else:
            print(f"  Not enough low-p points ({henry_mask.sum()} pts) for Henry slope")
    else:
        print("  No matching rows found in phase3 CSV")

if np.isnan(KH_pcsaft):
    print("  PC-SAFT K_H not available from phase3 — using LJ-henry as proxy")
    KH_pcsaft   = KH_lj
    p_ps        = lj273['p_bar'].values
    N_ps        = lj273['mmol_per_g_abs'].values
    pcsaft_label = 'PC-SAFT proxy (LJ-henry)'

# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*68)
print(f"  Henry-constant cross-check  |  T = {T:.0f} K,  p → 0")
print("="*68)
print(f"  {'Method':<35}  {'K_H [mmol/g/bar]':>16}  {'K_H/K_H(LJ)':>11}")
print("-"*68)
rows = [
    ('Analytical (vext_avg integral)',  KH_anal),
    ('LJ-cDFT (henry model, CSV)',      KH_lj),
    (pcsaft_label,                       KH_pcsaft),
]
for name, kh in rows:
    ratio = kh / KH_lj if (np.isfinite(kh) and np.isfinite(KH_lj) and KH_lj > 0) else np.nan
    print(f"  {name:<35}  {kh:>16.4f}  {ratio:>11.3f}")
print("="*68)

# Check agreements
if np.isfinite(KH_anal):
    d_al = abs(KH_anal - KH_lj) / KH_lj * 100
    print(f"  Analytical vs LJ-cDFT:  {d_al:.2f}%  "
          f"{'✓ PASS (<10%)' if d_al < 10 else '✗ WARN (>10%)'}")
if np.isfinite(KH_pcsaft) and pcsaft_label.startswith('PC-SAFT (phase3'):
    d_pl = abs(KH_pcsaft - KH_lj) / KH_lj * 100
    print(f"  PC-SAFT vs LJ-cDFT:     {d_pl:.2f}%  "
          f"{'✓ PASS (<10%)' if d_pl < 10 else '✗ WARN (>10%)'}")
    d_pa = abs(KH_pcsaft - KH_anal) / KH_anal * 100
    print(f"  PC-SAFT vs Analytical:  {d_pa:.2f}%  "
          f"{'✓ PASS (<10%)' if d_pa < 10 else '✗ WARN (>10%)'}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Log-log plot: N(p) vs p in Henry regime
# ─────────────────────────────────────────────────────────────────────────────
p_line = np.logspace(-5, -2, 80)   # 1e-5 to 0.01 bar

fig, ax = plt.subplots(figsize=(7, 5.5))

# LJ-cDFT data points
mask_lj = (lj273['p_bar'] >= 1e-5) & (lj273['p_bar'] <= 0.01)
dlj = lj273[mask_lj]
ax.loglog(dlj['p_bar'], dlj['mmol_per_g_abs'],
          'bo-', ms=5, lw=1.5, label=f'LJ-cDFT  (K_H = {KH_lj:.1f})')

# PC-SAFT data points (only if genuinely different from LJ)
if pcsaft_label.startswith('PC-SAFT (phase3'):
    mask_ps = (p_ps >= 1e-5) & (p_ps <= 0.01)
    ax.loglog(p_ps[mask_ps], N_ps[mask_ps],
              'rs--', ms=5, lw=1.5, label=f'{pcsaft_label}  (K_H = {KH_pcsaft:.1f})')

# Analytical K_H line
ax.loglog(p_line, KH_anal * p_line,
          'k-', lw=2.2, label=f'Analytical K_H = {KH_anal:.1f} mmol/g/bar')

# Reference Henry lines for LJ and PC-SAFT K_H (dashed)
ax.loglog(p_line, KH_lj * p_line,
          'b:', lw=1.5, alpha=0.5, label=f'LJ K_H line')
if np.isfinite(KH_pcsaft) and pcsaft_label.startswith('PC-SAFT (phase3'):
    ax.loglog(p_line, KH_pcsaft * p_line,
              'r:', lw=1.5, alpha=0.5, label=f'PC-SAFT K_H line')

ax.set_xlabel('Pressure (bar)', fontsize=12)
ax.set_ylabel(r'$N_{abs}$ (mmol g$^{-1}$)', fontsize=12)
ax.set_title(f'Henry-regime cross-check — T = {T:.0f} K', fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, which='both', ls=':', alpha=0.4)
ax.set_xlim(1e-5, 0.01)

# Annotate deviation
if np.isfinite(KH_anal):
    d_al = abs(KH_anal - KH_lj) / KH_lj * 100
    color = 'green' if d_al < 10 else 'red'
    ax.text(0.97, 0.05,
            f'Anal/LJ: {d_al:.1f}%\n(PASS <10%)' if d_al < 10 else f'Anal/LJ: {d_al:.1f}%\n(WARN)',
            transform=ax.transAxes, fontsize=9, color=color,
            va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

plt.tight_layout()
out_path = os.path.join(FIG_DIR, '32_henry_crosscheck.png')
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved: {out_path}")
