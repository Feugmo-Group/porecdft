"""Picard fixed-point iteration for self-consistent cDFT.

Solve:  ρ(r) = ρ_bulk · exp[ −β V_ext(r) + c¹(ρ; r) − c¹_bulk ]

with mixing:  ρ_{n+1} = (1−α) ρ_n + α · ρ_target,  α ∈ (0, 1].

Includes log-space clipping for numerical stability when V_ext is very large
or c¹ swings widely during early iterations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PicardResult:
    rho: np.ndarray
    converged: bool
    iterations: int
    error_history: list[float]


def picard_solve(
    rho_init: np.ndarray,
    rho_bulk: float,
    Vext_K: np.ndarray,                  # (grid,) external potential in K
    temperature_K: float,
    c1_callable: Callable[[np.ndarray], np.ndarray],
    c1_bulk: float,
    alpha: float = 0.05,
    max_iter: int = 2000,
    tol: float = 1e-4,
    accessibility_mask: np.ndarray | None = None,
    log_clip: float = 50.0,
) -> PicardResult:
    """Picard iteration for ρ(r).

    Parameters
    ----------
    rho_init : ndarray
        Initial density (e.g., ρ_bulk everywhere or capped Boltzmann profile).
    rho_bulk : float
        Reservoir density (molecules/Å³).
    Vext_K : ndarray
        External potential in K, same shape as rho_init.
    temperature_K : float
        Temperature in K.
    c1_callable : callable
        ``c1_callable(rho) → c¹(r)`` — typically `lambda r: compute_c1(r, ...)`.
    c1_bulk : float
        Bulk reference c¹(ρ_bulk).
    alpha : float
        Mixing factor.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on ‖ρ_new − ρ‖ / ‖ρ_bulk‖.
    accessibility_mask : ndarray, optional
        If given, ρ is forced to 0 outside the mask.
    log_clip : float
        Clip |β V + c¹ − c¹_bulk| at ±log_clip to keep exp() finite.
    """
    beta = 1.0 / temperature_K
    rho = np.asarray(rho_init, dtype=np.float64).copy()
    history: list[float] = []
    converged = False
    last_err = np.inf

    log_rho_bulk = np.log(rho_bulk + 1e-300)
    log_rho = np.log(np.maximum(rho, 1e-30))
    for it in range(max_iter):
        c1 = np.asarray(c1_callable(np.exp(log_rho)))
        # Target in log space — exact Picard step
        log_rho_target = -beta * Vext_K + c1 - c1_bulk + log_rho_bulk
        log_rho_target = np.clip(log_rho_target, -log_clip + log_rho_bulk, +log_clip + log_rho_bulk)
        # Log-space mixing (stable even with huge swings in log_rho_target)
        log_rho_new = (1.0 - alpha) * log_rho + alpha * log_rho_target
        if accessibility_mask is not None:
            log_rho_new = np.where(accessibility_mask, log_rho_new, -1e6)
        rho_new = np.exp(log_rho_new)
        err = float(np.max(np.abs(rho_new - np.exp(log_rho))) / (rho_bulk + 1e-30))
        history.append(err)
        log_rho = log_rho_new
        rho = rho_new
        if err < tol:
            converged = True
            return PicardResult(rho, converged, it + 1, history)
        if not np.isfinite(err) or err > 1e30:
            break
        last_err = err

    return PicardResult(rho, converged, max_iter, history)
