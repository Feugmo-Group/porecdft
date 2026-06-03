"""Anderson-accelerated fixed-point solver for cDFT.

Standard Anderson type-II in log-density space:

    u_n  = log ρ_n
    v_n  = -β V_ext + c¹(ρ_n) − c¹_bulk + log ρ_bulk      (target log-density)
    F_n  = v_n − u_n                                      (residual)

At step n with history depth m (typically 5–10), solve

    γ = argmin_γ ‖ F_n − ΔF · γ ‖²                 ΔF = [F_n − F_{n−1}, …, F_n − F_{n−m}]

and update

    u_{n+1} = u_n + β·F_n − (ΔU + β·ΔF) · γ        ΔU = [u_n − u_{n−1}, …, u_n − u_{n−m}]

`β` is a damping parameter (often 1.0; smaller values stabilise at the cost of speed).
This converges in O(10²) iterations even for FMT at saturation densities where
simple Picard oscillates indefinitely.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class AndersonResult:
    rho: np.ndarray
    converged: bool
    iterations: int
    error_history: list[float]


def anderson_solve(
    rho_init: np.ndarray,
    rho_bulk: float,
    Vext_K: np.ndarray,
    temperature_K: float,
    c1_callable: Callable[[np.ndarray], np.ndarray],
    c1_bulk: float,
    m: int = 8,
    beta: float = 0.3,
    max_iter: int = 400,
    tol: float = 1e-5,
    accessibility_mask: np.ndarray | None = None,
    log_clip: float = 30.0,
    safeguard_alpha: float = 0.05,
    picard_warmup: int = 30,
    step_clip: float = 5.0,               # max |Δlog ρ| per step in any voxel
) -> AndersonResult:
    """Anderson-accelerated solver for ρ(r).

    Parameters
    ----------
    rho_init, rho_bulk, Vext_K, temperature_K : see porecdft.solver.picard_solve.
    c1_callable, c1_bulk : c¹ function and bulk reference.
    m : int
        Anderson history depth. Larger m → faster convergence but more memory
        and risk of ill-conditioned LSQ.
    beta : float
        Anderson damping (1.0 = no damping; recommended for FMT).
    max_iter, tol, accessibility_mask, log_clip : standard solver knobs.
    safeguard_alpha : float
        Picard mixing fraction used as a fallback if Anderson update would
        diverge (e.g., LSQ ill-conditioned).
    """
    inv_T = 1.0 / temperature_K
    log_rho_bulk = float(np.log(rho_bulk + 1e-300))
    log_clip_lo = -log_clip + log_rho_bulk
    log_clip_hi = +log_clip + log_rho_bulk

    u = np.log(np.maximum(np.asarray(rho_init, dtype=np.float64), 1e-30))
    # Apply mask immediately so u never starts at a huge negative value that
    # creates spuriously large residuals at inaccessible voxels.
    if accessibility_mask is not None:
        u = np.where(accessibility_mask, u, log_clip_lo)

    u_hist: deque[np.ndarray] = deque(maxlen=m + 1)
    F_hist: deque[np.ndarray] = deque(maxlen=m + 1)
    history: list[float] = []
    converged = False

    for it in range(max_iter):
        rho = np.exp(u)
        c1 = np.asarray(c1_callable(rho))
        v = -inv_T * Vext_K + c1 - c1_bulk + log_rho_bulk
        v = np.clip(v, log_clip_lo, log_clip_hi)
        F = v - u
        # Zero residual at inaccessible voxels so they do not corrupt the
        # Anderson history (mask forces u=-1e6 ≪ log_clip_lo → F would be huge).
        if accessibility_mask is not None:
            F = np.where(accessibility_mask, F, 0.0)
        u_hist.append(u.copy())
        F_hist.append(F.copy())
        err = float(np.max(np.abs(F)))
        history.append(err)
        if err < tol:
            converged = True
            break
        if not np.isfinite(err) or err > 1e10:
            break

        # Warm-up with Picard for first few iterations so history collects
        # smooth(ish) residuals before Anderson kicks in.
        if it < picard_warmup or len(F_hist) < 3:
            step = safeguard_alpha * F
        else:
            n_pairs = min(m, len(F_hist) - 1)
            DU = np.stack([u_hist[-1] - u_hist[-1 - k] for k in range(1, n_pairs + 1)], axis=1).reshape(-1, n_pairs)
            DF = np.stack([F_hist[-1] - F_hist[-1 - k] for k in range(1, n_pairs + 1)], axis=1).reshape(-1, n_pairs)
            try:
                gamma, *_ = np.linalg.lstsq(DF, F.reshape(-1), rcond=1e-8)
                step_arr = (beta * F).reshape(-1) - (DU + beta * DF) @ gamma
                step = step_arr.reshape(u.shape)
                if not np.all(np.isfinite(step)):
                    raise np.linalg.LinAlgError("nonfinite Anderson step")
            except (np.linalg.LinAlgError, ValueError):
                step = safeguard_alpha * F

        # Trust-region clip on the step magnitude (per voxel)
        step = np.clip(step, -step_clip, +step_clip)
        u_new = u + step
        u_new = np.clip(u_new, log_clip_lo, log_clip_hi)
        if accessibility_mask is not None:
            u_new = np.where(accessibility_mask, u_new, -1e6)
        u = u_new

    return AndersonResult(np.exp(u), converged, it + 1, history)
