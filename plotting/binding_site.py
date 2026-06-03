"""Binding-site diagnostic plots."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from porecdft.diagnostics.binding_site import BindingSiteResult, K_TO_KJ_PER_MOL


def plot_binding_rose(
    result: BindingSiteResult,
    dft_reference_kJ_per_mol: float | None = None,
    title: str | None = None,
    ax=None,
):
    """Polar scatter of orientation-resolved binding energy.

    The angle is the polar angle (acos(uz)) of the molecular axis; the radius is
    energy in kJ/mol (negative pulled to the centre). The DFT reference, if
    given, is drawn as a dashed circle.
    """
    energies_kJ = result.energies_K * K_TO_KJ_PER_MOL
    theta = np.arccos(np.clip(result.directions[:, 2], -1.0, 1.0))
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="polar")
    sc = ax.scatter(theta, energies_kJ, c=energies_kJ, cmap="viridis", s=20)
    if dft_reference_kJ_per_mol is not None:
        ax.plot(np.linspace(0, 2 * np.pi, 200),
                np.full(200, dft_reference_kJ_per_mol),
                "r--", label=f"DFT ref = {dft_reference_kJ_per_mol:.1f} kJ/mol")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.15))
    title = title or f"Binding energies at {result.site_label}"
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="E (kJ/mol)", shrink=0.7)
    return ax


def plot_part_decomposition(
    results: list[BindingSiteResult],
    dft_references_kJ_per_mol: dict[str, float] | None = None,
    ax=None,
):
    """Stacked bar chart: LJ / Coulomb / Quad / ... contributions at each site's
    minimum-energy orientation, compared to DFT references.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    sites = [r.site_label for r in results]
    component_names: list[str] = []
    for r in results:
        for k in r.parts_at_min:
            if k not in component_names:
                component_names.append(k)
    x = np.arange(len(sites))
    bottom_pos = np.zeros(len(sites))
    bottom_neg = np.zeros(len(sites))
    for comp in component_names:
        vals_kJ = np.array([r.parts_at_min.get(comp, 0.0) for r in results]) * K_TO_KJ_PER_MOL
        bottoms = np.where(vals_kJ >= 0, bottom_pos, bottom_neg)
        ax.bar(x, vals_kJ, bottom=bottoms, label=comp)
        bottom_pos = bottom_pos + np.where(vals_kJ >= 0, vals_kJ, 0)
        bottom_neg = bottom_neg + np.where(vals_kJ < 0, vals_kJ, 0)
    totals = np.array([r.E_min_kJ_per_mol for r in results])
    ax.plot(x, totals, "ko-", label="Total min")
    if dft_references_kJ_per_mol is not None:
        for i, s in enumerate(sites):
            ref = dft_references_kJ_per_mol.get(s)
            if ref is not None:
                ax.plot([i - 0.4, i + 0.4], [ref, ref], "r--",
                        label="DFT ref" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.set_ylabel("Binding energy (kJ/mol)")
    ax.set_title("Vext component decomposition at minimum orientation")
    ax.legend(loc="best", fontsize=8)
    ax.axhline(0, color="grey", lw=0.5)
    return ax


def plot_orientation_histogram(result: BindingSiteResult, n_bins: int = 30, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    e_kJ = result.energies_K * K_TO_KJ_PER_MOL
    ax.hist(e_kJ, bins=n_bins, edgecolor="black")
    ax.axvline(e_kJ.min(), color="red", linestyle="--", label=f"min = {e_kJ.min():.2f}")
    ax.set_xlabel("Binding energy (kJ/mol)")
    ax.set_ylabel("# orientations")
    ax.set_title(f"Orientation histogram @ {result.site_label}")
    ax.legend()
    return ax
