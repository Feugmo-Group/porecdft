# Paper reproduction guide

This directory contains all application scripts that reproduce the figures and benchmarks in:

> Roy, A.; Tetsassi Feugmo, C. G. *A modular classical density-functional framework for gas adsorption in nanoporous materials.* 2026, in preparation.

All scripts must be run **from the repository root** with the editable install active.
Generated figures are written to `applications/*/figures/`.
Intermediate results are cached under `applications/*/results/` (not tracked by git) and reused on subsequent runs.

---

## CO₂ / aluminum formate (ALF)

ALF (Al(HCOO)₃, cubic Im-3m, Evans et al. *Sci. Adv.* 2022) simultaneously exhibits cooperative pore filling, framework gate-opening, and kinetic molecular sieving.

### Force-field validation (Methods section, not paper figures)

```bash
uv run python applications/alf_co2/notebooks/phase0_evans_check.py        # digitised data check
uv run python applications/alf_co2/notebooks/phase1_vext_validation.py    # SC/LC site probe
uv run python applications/alf_co2/notebooks/phase1d_lj_tuning.py         # LJ ε scaling
uv run python applications/alf_co2/notebooks/phase1e_smeared_coulomb_tuning.py
```

### Paper figures — minimal pipeline

Run in order; each step reads CSV files from earlier steps.
Vext caches under `results/vext_cache_flex/` are built on first run (~2–4 h) and reused thereafter.

```bash
# Step 1 — Production isotherm: K_eff × ε_assoc × T sweep + FMT-aWBII baseline
# Runtime: ~2–4 h first run.
uv run python applications/alf_co2/notebooks/phase3_production_isotherm.py
# Output: figures/24_phase3_param_sweep.png
#         figures/25_phase3_best_model.png
#         figures/26_phase3_parity.png

# Step 2 — Isosteric heat Q_st (Clausius–Clapeyron from step 1 CSV)
uv run python applications/alf_co2/notebooks/phase3_qst.py
# Output: figures/27_phase3_qst.png

# Step 3 — Paper summary figures (reads steps 1–2 CSVs, no new cDFT runs)
uv run python applications/alf_co2/notebooks/phase3_final_summary.py
# Output: figures/31_phase3_final_summary.png

uv run python applications/alf_co2/henry_crosscheck.py
# Output: figures/32_henry_crosscheck.png

uv run python applications/alf_co2/notebooks/n2_isotherm_selectivity.py
# Output: figures/33_n2_isotherm_298K.png
#         figures/34_co2_n2_selectivity.png

uv run python applications/alf_co2/notebooks/phase_summary_figure.py
# Output: figures/35_co2_vs_experiment_final.png
```

---

## H₂ / metalated COFs

### Morse potential validation

```bash
uv run python applications/alf_co2/notebooks/phase_morse_validation.py
# Output: applications/h2_cof/figures/morse_validation.png
```

### Full COF benchmark: 4 frameworks × 5 metals

Builds Vext caches per framework/metal on first run (~1–2 h total).

```bash
uv run python applications/h2_cof/notebooks/make_h2_cof_benchmark.py
# Output: figures/h2_cof_benchmark.png
```

### COF-333-CoCl₂ H₂ isotherm

```bash
uv run python applications/h2_cof/notebooks/make_h2_isotherm.py
# Output: figures/h2_isotherm_cof333.png
```

---

## Solver and F_exc benchmark (paper — solver comparison section)

### F_exc mode comparison: endpoint vs RPA vs GL quadrature

Single-point benchmark at T = 298 K, P = 10 bar. Runs Anderson (reference), Adam, and FIRE2 in all three F_exc modes.

```bash
uv run python applications/h2_cof/benchmark_fexc_modes.py
# Output: figures/benchmark_fexc_modes.png
#         results/benchmark_fexc_modes.npy
```

### Adam convergence and loss-landscape probe

Demonstrates that Adam stalls at a non-convex plateau and that Picard polishing closes the gap.

```bash
uv run python applications/h2_cof/benchmark_adam_polish.py
# Output: results/benchmark_adam_polish.npy
```

### Full isotherm comparison: Adam variants vs Anderson

Runs Adam 3k and Adam 20k (no polish) across 16 pressure points (1–500 bar) and compares to cached Anderson and Adam+polish isotherms.

```bash
uv run python applications/h2_cof/run_adam_isotherms.py
# Output: figures/isotherm_adam_variants.png
#         results/isotherm_h2_cof333_adam3k_nopol.npy
#         results/isotherm_h2_cof333_adam20k_nopol.npy
```

---

## EOS comparison

Multi-EOS bulk-density comparison for CO₂ at 298 K (ideal gas, PR, SRK, Span-Wagner, PC-SAFT).

```bash
uv run python applications/eos_compare/co2_comparison.py
# Output: applications/eos_compare/figures/co2_density_comparison.png
```

---

## Benchmark summary (H₂/COF-333, T = 298 K, P = 10 bar)

Anderson mixing is the reference solver. All solvers use `f_exc_mode="endpoint"` unless noted.

| Solver | Mode | N_ads | Δ% | t (s) | Notes |
|--------|------|-------|----|-------|-------|
| Anderson | — | 16.94 | ref | 7.7 | Picard warm-start, m=8 |
| Adam 3k | endpoint | 16.63 | −1.8 | 13.6 | Plateau at ~step 1700 |
| Adam 9k | endpoint | 16.63 | −1.8 | 25.9 | Identical to 3k — landscape artifact |
| Adam 3k + polish | endpoint | 16.94 | 0.0 | 35.6 | Anderson from Adam output |
| Adam 3k | rpa | 18.32 | +8.2 | 13.6 | ½ factor invalid for WDA |
| Adam 3k | quadrature (4-pt GL) | 18.60 | +9.8 | 40.3 | 4 c⁽¹⁾ calls |
| FIRE2 | endpoint | 17.81 | +5.1 | 298 | Not converged at 3k steps |

**Full isotherm finding:** Adam 3k ≡ Adam 20k at every pressure point (1–500 bar). Gap grows from +1.8% at 1 bar to +31% at 500 bar. Only Picard polish from the Adam output recovers the Anderson reference.
