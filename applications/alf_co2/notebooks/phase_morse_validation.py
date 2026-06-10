"""Morse potential validation script (task #18).

H2-graphite benchmark parameters:
  D_e = 52.0 K, r_e = 3.2 Ang, a = 1.8 Ang^-1

Validates:
  1. V(r_e) == -D_e
  2. V(2*r_e) ~ 0  (within 5% of D_e)
  3. V(r -> large) -> 0

Also compares to LJ 12-6 with equivalent epsilon and sigma.
Saves figure to applications/alf_co2/figures/morse_validation.png
"""

import sys
import os

# Ensure project root is on the path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from porecdft.forcefield import MorseScalarPotential

# -----------------------------------------------------------------
# Parameters: H2-graphite benchmark
# -----------------------------------------------------------------
D_e = 52.0     # K
r_e = 3.2      # Ang
a   = 1.8      # Ang^-1

morse = MorseScalarPotential(D_e_K=D_e, r_e_A=r_e, a_invA=a)

# -----------------------------------------------------------------
# Compute V(r) over [2.0, 8.0] Ang
# -----------------------------------------------------------------
r = np.linspace(2.0, 8.0, 100)
V_morse = morse(r)

# -----------------------------------------------------------------
# LJ 12-6 with equivalent epsilon and sigma
#   epsilon_LJ = D_e,  sigma_LJ = r_e * 2^(-1/6)
# -----------------------------------------------------------------
eps_lj = D_e
sig_lj = r_e * 2.0 ** (-1.0 / 6.0)

def lj(r_arr, eps, sig):
    sr6 = (sig / r_arr) ** 6
    return 4.0 * eps * (sr6 ** 2 - sr6)

V_lj = lj(r, eps_lj, sig_lj)

# -----------------------------------------------------------------
# Assertions / pass-fail
# -----------------------------------------------------------------
tol_exact = 1e-10
tol_5pct  = 0.05 * D_e   # 5% of D_e

# 1. V(r_e) = -D_e
V_at_re = float(morse(np.array([r_e]))[0])
err_re  = abs(V_at_re - (-D_e))
pass1   = err_re < tol_exact
print(f"[Test 1] V(r_e) = {V_at_re:.6f} K,  expected {-D_e:.6f} K,  error = {err_re:.2e} K  ->  {'PASS' if pass1 else 'FAIL'}")

# 2. V(2*r_e) ~ 0  (within 5% of D_e)
V_at_2re = float(morse(np.array([2.0 * r_e]))[0])
err_2re  = abs(V_at_2re)
pass2    = err_2re < tol_5pct
print(f"[Test 2] V(2*r_e) = {V_at_2re:.6f} K,  |V| = {err_2re:.4f} K,  tol = {tol_5pct:.4f} K  ->  {'PASS' if pass2 else 'FAIL'}")

# 3. V(r -> large) -> 0   (use r = 50 Ang)
r_large   = 50.0
V_at_large = float(morse(np.array([r_large]))[0])
pass3      = abs(V_at_large) < 1e-6
print(f"[Test 3] V({r_large} Ang) = {V_at_large:.2e} K  ->  {'PASS' if pass3 else 'FAIL'}")

# 4. Minimum is at r_e (find numerical minimum)
r_fine     = np.linspace(2.5, 5.0, 10000)
V_fine     = morse(r_fine)
r_min_num  = r_fine[np.argmin(V_fine)]
pass4      = abs(r_min_num - r_e) < 0.01
print(f"[Test 4] Numerical minimum at r = {r_min_num:.4f} Ang,  expected {r_e} Ang  ->  {'PASS' if pass4 else 'FAIL'}")

all_pass = pass1 and pass2 and pass3 and pass4
print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

# -----------------------------------------------------------------
# Plot
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(r, V_morse, "b-",  lw=2, label=f"Morse  (D_e={D_e} K, r_e={r_e} Å, a={a} Å⁻¹)")
ax.plot(r, V_lj,    "r--", lw=2, label=f"LJ 12-6 (ε={eps_lj} K, σ={sig_lj:.4f} Å)")
ax.axhline(-D_e, color="grey",  ls=":",  lw=1, label=f"−D_e = {-D_e} K")
ax.axhline(  0,  color="black", ls="-",  lw=0.8)
ax.axvline(r_e,  color="green", ls="--", lw=1, label=f"r_e = {r_e} Å")

ax.set_xlim(r[0], r[-1])
ax.set_ylim(-1.4 * D_e, 3.0 * D_e)
ax.set_xlabel("r  /  Å", fontsize=13)
ax.set_ylabel("V(r)  /  K", fontsize=13)
ax.set_title("Morse vs LJ 12-6 — H₂/graphite benchmark", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig_path = os.path.join(_root, "applications", "alf_co2", "figures", "morse_validation.png")
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to: {fig_path}")
