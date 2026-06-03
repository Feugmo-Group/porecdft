"""Plots of external-potential slices and line cuts."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_vext_slice_2d(
    vext_grid: np.ndarray,
    axis: str = "z",
    index: int | None = None,
    extent: tuple[float, float, float, float] | None = None,
    vmin_kJ_per_mol: float = -25.0,
    vmax_kJ_per_mol: float = +25.0,
    title: str | None = None,
    ax=None,
):
    """2D heatmap of a Vext slice (in K) shown in kJ/mol for readability.

    Parameters
    ----------
    vext_grid : ndarray
        (Nx, Ny, Nz) Vext in K.
    axis : {"x", "y", "z"}
        Which axis to slice perpendicular to.
    index : int, optional
        Slice index. If None, takes the middle of that axis.
    extent : (x0, x1, y0, y1), optional
        Imshow extent in Å for the two remaining axes.
    """
    from porecdft.diagnostics.binding_site import K_TO_KJ_PER_MOL
    g = vext_grid
    if axis == "z":
        idx = index if index is not None else g.shape[2] // 2
        plane = g[:, :, idx]
    elif axis == "y":
        idx = index if index is not None else g.shape[1] // 2
        plane = g[:, idx, :]
    else:
        idx = index if index is not None else g.shape[0] // 2
        plane = g[idx, :, :]
    plane_kJ = plane * K_TO_KJ_PER_MOL
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        plane_kJ.T,
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin_kJ_per_mol,
        vmax=vmax_kJ_per_mol,
        extent=extent,
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="Vext (kJ/mol)")
    ax.set_title(title or f"Vext slice ⟂ {axis}, idx={idx}")
    return ax


def plot_vext_line_1d(
    line_xyz: np.ndarray,
    vext_values_K: np.ndarray,
    label: str = "",
    ax=None,
):
    """1D line cut of Vext along a path through the unit cell."""
    from porecdft.diagnostics.binding_site import K_TO_KJ_PER_MOL
    s = np.linalg.norm(line_xyz - line_xyz[0], axis=1)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(s, vext_values_K * K_TO_KJ_PER_MOL, label=label)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Path length (Å)")
    ax.set_ylabel("Vext (kJ/mol)")
    if label:
        ax.legend()
    return ax
