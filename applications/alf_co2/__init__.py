"""ALF/CO₂ application — the first case study using `porecdft`.

This package contains everything specific to the aluminum-formate + CO₂ system:
- the canonical CIF (`structures/alf.cif`),
- the force-field parameters and DDEC6 partial charges with provenance,
- the experimental and ab-initio reference data,
- diagnostic and isotherm notebooks.

The cDFT engine itself is host- and fluid-agnostic; nothing under `applications/`
is imported by `porecdft.*`.
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent
STRUCTURES_DIR = DATA_DIR / "structures"
PARAMETERS_DIR = DATA_DIR / "parameters"
REFERENCES_DIR = DATA_DIR / "references"

ALF_CIF = STRUCTURES_DIR / "alf.cif"        # canonical cubic unit cell (Im-3m, a=11.367 Å, 104 atoms)
ALF_CIF_DFT = STRUCTURES_DIR / "alf_dft.cif"  # CP2K MD geometry — 2×1×1 supercell, use only as reference
FORCEFIELD_CSV = PARAMETERS_DIR / "forcefield.csv"
CHARGES_CSV = PARAMETERS_DIR / "charges.csv"

# Evans et al. DFT binding energies (kJ/mol) — Table 1 of Evans 2022 Sci. Adv. ade1473.
# Negative = attractive. SC = small cavity (H-bond), LC = large cavity (no H-bond).
DFT_BINDING_KJ_PER_MOL = {"SC": -18.4, "LC": -8.1}

# Experimental CO₂/ALF isotherms — Evans 2022 Sci. Adv. ade1473, Fig. 2A
# Carefully re-digitized from the published figure.
# Pressure originally in mmHg; stored in bar (1 mmHg = 0.00133322 bar).
EXPERIMENTAL_ISOTHERM_CSV = REFERENCES_DIR / "experimental_isotherm_evans2022.csv"

_MMHg = 0.00133322
EXP_ISOTHERMS = {
    # Saturates early (~2.88 mmol/g by 300 mmHg) — formate dynamics kinetically
    # restrict SC window at 273 K, limiting accessible sites.
    273: {
        "p_bar": [p * _MMHg for p in [25, 50, 75, 100, 125, 150, 200, 300, 400, 500, 600, 700, 800, 900]],
        "N_mmol_g": [0.80, 1.70, 2.25, 2.60, 2.75, 2.81, 2.84, 2.87, 2.88, 2.88, 2.88, 2.88, 2.88, 2.89],
    },
    # Gradual rise to 4.15 mmol/g — best reproduced by cDFT (RMSE 0.33 mmol/g).
    298: {
        "p_bar": [p * _MMHg for p in [25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900]],
        "N_mmol_g": [0.20, 0.60, 1.00, 1.40, 2.10, 2.55, 3.15, 3.55, 3.80, 3.95, 4.05, 4.10, 4.15],
    },
    # Anomalously reaches same 4.15 mmol/g as 298 K — Evans anomaly driven by
    # formate librational dynamics widening the SC window at higher T.
    323: {
        "p_bar": [p * _MMHg for p in [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900]],
        "N_mmol_g": [0.15, 0.35, 0.65, 1.00, 1.65, 2.20, 2.75, 3.20, 3.60, 3.90, 4.15],
    },
    348: {
        "p_bar": [p * _MMHg for p in [100, 200, 300, 400, 500, 600, 700, 800, 900]],
        "N_mmol_g": [0.04, 0.13, 0.32, 0.62, 0.98, 1.38, 1.82, 2.22, 2.48],
    },
    # Near-zero at all pressures; ALF regenerated at 353 K under CO₂.
    398: {
        "p_bar": [p * _MMHg for p in [100, 200, 300, 400, 500, 600, 700, 800, 900]],
        "N_mmol_g": [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.11, 0.12],
    },
}

# Convenience alias in the legacy list-of-tuples format used by all notebooks.
# Import this instead of defining EXP_TARGETS locally in each notebook.
EXP_TARGETS = {T: list(zip(d["p_bar"], d["N_mmol_g"]))
               for T, d in EXP_ISOTHERMS.items()}
