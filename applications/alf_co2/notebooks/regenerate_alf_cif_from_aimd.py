"""Write a fresh `alf_dft.cif` using:
  - the CP2K cell vectors from the user's input (22.861 × 11.4305 × 11.4305 Å)
  - the 104 framework atoms from a CP2K NVT snapshot

This replaces our placeholder `alf.cif` (which was actually a different structure
from a different source — 11.367 Å lattice, different atom positions, atoms
displaced by ~1.94 Å mean from the AIMD geometry).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARENT = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try: sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

from applications.alf_co2 import STRUCTURES_DIR
from applications.alf_co2.notebooks.phase1c_aimd_co2_positions import read_xyz

AIMD_FILE = Path(
    "/Users/cgtetsas/Library/CloudStorage/OneDrive-UniversityofWaterloo/UWaterloo/"
    "Manuscript/2026/ALF/cDFT/ALF-CO2-N2_Manuscript_data/CP2K_calculations/"
    "3_abinitio_MD/2_NVT_abinitio_MD/ALF-S1/ALF-S1-1CO2_NVT_10ps.xyz"
)

# Cell from CP2K input files (1_CP2K_input-files/*.inp). a is 2× to make a 2x1x1
# supercell of the underlying ALF primitive cell.
CELL_A, CELL_B, CELL_C = 22.86100, 11.43050, 11.43050


def main():
    species, positions, _ = read_xyz(AIMD_FILE)
    fram_sp = species[:104]
    fram_pos = positions[:104]
    print(f"Read {len(fram_sp)} framework atoms from {AIMD_FILE.name}")
    print(f"  species count: "
          f"Al={fram_sp.count('Al')} C={fram_sp.count('C')} "
          f"O={fram_sp.count('O')} H={fram_sp.count('H')}")

    # Fractional coordinates wrt CP2K cell
    inv_a = 1.0 / CELL_A
    inv_b = 1.0 / CELL_B
    inv_c = 1.0 / CELL_C
    fract = []
    for r in fram_pos:
        fx = r[0] * inv_a; fy = r[1] * inv_b; fz = r[2] * inv_c
        # wrap into [0, 1)
        fx -= int(fx) if fx >= 0 else int(fx) - 1
        fy -= int(fy) if fy >= 0 else int(fy) - 1
        fz -= int(fz) if fz >= 0 else int(fz) - 1
        fx = fx - 1.0 if fx >= 1.0 else fx
        fy = fy - 1.0 if fy >= 1.0 else fy
        fz = fz - 1.0 if fz >= 1.0 else fz
        fract.append((fx, fy, fz))

    out_path = STRUCTURES_DIR / "alf_dft.cif"
    counts = {"Al": 0, "C": 0, "O": 0, "H": 0}
    lines = [
        "data_ALF_DFT",
        "# Framework geometry from CP2K NVT MD (ALF-S1-1CO2_NVT_10ps.xyz, t=10 ps),",
        "# stripped to the 104 framework atoms. Cell from CP2K input (2×1×1 supercell).",
        f"_cell_length_a {CELL_A:.5f}",
        f"_cell_length_b {CELL_B:.5f}",
        f"_cell_length_c {CELL_C:.5f}",
        "_cell_angle_alpha 90.0",
        "_cell_angle_beta 90.0",
        "_cell_angle_gamma 90.0",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "  'x, y, z'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]
    for sp, (fx, fy, fz) in zip(fram_sp, fract):
        counts[sp] = counts.get(sp, 0) + 1
        label = f"{sp}{counts[sp]}"
        lines.append(f"{label:6s} {sp:2s} {fx:15.10f} {fy:15.10f} {fz:15.10f}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    print(f"  cell: {CELL_A} x {CELL_B} x {CELL_C} Å (volume {CELL_A*CELL_B*CELL_C:.1f} Å³)")
    print(f"  per-element counts: {counts}")


if __name__ == "__main__":
    main()
