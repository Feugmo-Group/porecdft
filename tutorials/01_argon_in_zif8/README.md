# Tutorial 1 — Argon adsorption in ZIF-8 at 77 K

## What this tutorial does

Reproduces the Ar/ZIF-8 isotherm from Stierle & Gross 2024 (Fig. 5) using
porecdft's standard pipeline:

1. **Load host** — ZIF-8 cubic unit cell (16.99 Å, *Im*-43*m*) from
   `tutorials/data/structures/ZIF-8.cif`.
2. **Force field** — DREIDING Lennard-Jones parameters from
   `tutorials/data/forcefields/DREIDING.dat`. Single-site fluid model: σ_Ar
   from the same DREIDING table.
3. **Build *V*~ext~** — composite LJ-only potential on a 3D grid
   (the noble-gas system has no charges, so we skip the Coulomb / EFG
   modules).
4. **FMT-aWBII self-consistency** — Anderson solver with the `rho_max`
   packing-fraction cap.
5. **Bulk EOS** — PC-SAFT argon
   (m, σ, ε/k) = (1.0, 3.405, 119.8) from `noble_gases.json`.
6. **Plot isotherm** and compare to the published Stierle 2024 curve.

## Why this is a useful validation

* **Independent code path.** ZIF-8 has Zn ions with no partial charges in
  DREIDING — the entire interaction is LJ. This isolates the FMT-aWBII +
  PC-SAFT layer and decouples it from porecdft's smeared-Coulomb / Q-EFG
  modules. A clean cross-check.
* **Same input data** as Stierle & Gross 2024.
* **77 K, low pressure.** Ar at 77 K is highly subcritical, isotherm
  saturates at full pore filling — strong stress test for FMT close-packing
  behaviour (the `rho_max` cap is crucial here).

## Run

```bash
cd /path/to/porecdft
/opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \
    tutorials/01_argon_in_zif8/run.py
```

Runtime: ~3 min on a CPU (first call builds the cached *V*~ext~; reruns are
seconds).

## Expected output

* `tutorials/figures/01_argon_in_zif8.png` — Ar isotherm at 77 K from
  10⁻⁴ to 1 bar, mmol/g and wt %.
* Console: convergence trace per pressure point.
* Stored intermediate: `tutorials/01_argon_in_zif8/vext_zif8_Ar_77K.npy`.

The porecdft curve should sit on top of the Stierle 2024 reference data
(stored in `reference_stierle2024.csv` once you have it) to within ~5 %
across the full pressure range.

## Files

| File | Purpose |
|------|---------|
| `run.py`                    | The runnable script |
| `reference_stierle2024.csv` | (optional) digitised reference data |
