"""FIRE2Solver — Nonlinear-CG minimiser for the cDFT grand potential.

Uses ``optimistix.NonlinearCG`` (Polak-Ribières) to minimise Ω[ρ] directly.
Operates in log-density space (ψ = ln ρ) so ρ > 0 is automatic and the
ideal-gas logarithmic term is well-scaled.

This is complementary to the Picard/Anderson solvers:
  - Picard / Anderson: fixed-point iteration, fast per-step, can stall near
    saturation density or with shallow gradients.
  - FIRE2Solver (NonlinearCG): proper minimiser, monotone Ω decrease, more
    robust for stiff problems.  Slightly heavier per-step due to JAX JIT.

Requires: jax, optimistix >= 0.0.10 (``uv add optimistix``).

API mirrors jax_solver.py — the functional convenience wrapper is ``fire2_solve``,
the equinox Module is ``FIRE2Solver``.

Excess free-energy approximation (``f_exc_mode``)
--------------------------------------------------
The grand potential contains an excess free-energy term F_ex[ρ] that is only
known through its functional derivative c¹[ρ] = -δF_ex/δρ / k_BT.  Three
approximations are supported, controlled by the ``f_exc_mode`` argument:

``"endpoint"`` (default)
    λ = 1 endpoint rule: F_ex ≈ -k_BT ∫ c¹[ρ] ρ dV.
    Gradient w.r.t. ρ (via JAX autograd) is close to, but not exactly equal
    to, the Euler-Lagrange condition — the extra chain-rule term from ∂c¹/∂ρ
    vanishes only at uniform density.  Picard polishing is recommended to
    finish at the exact fixed point.

``"rpa"``
    Linear/RPA approximation: F_ex ≈ -½ k_BT ∫ c¹[ρ] ρ dV.  Exact only when
    F_ex is quadratic (ideal correction, mean-field).  Poor for FMT / WDA at
    high packing.  Included for comparison with Bao (2025) suggestion.

``"quadrature"``
    Gauss-Legendre quadrature over the adiabatic path λ ∈ [0,1] (Eq. 3,
    Bao 2025):  F_ex ≈ -k_BT ∑_i w_i ∫ c¹[λ_i ρ] ρ dV.
    Requires ``n_quad`` c¹ evaluations per Ω call (default 4).
    GPU/vmap compatible: if c¹ is JAX-traceable the λ-loop is unrolled at
    trace time into a single XLA graph; no Python overhead at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import jax.numpy as jnp
import numpy as np

FExcMode = Literal["endpoint", "rpa", "quadrature"]

# Pre-computed Gauss-Legendre nodes/weights on [0, 1] for n = 2, 3, 4, 5.
_GL_NODES: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gl_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """GL nodes and weights on [0, 1] for *n* quadrature points."""
    if n not in _GL_NODES:
        xi, wi = np.polynomial.legendre.leggauss(n)
        _GL_NODES[n] = (0.5 * (xi + 1), 0.5 * wi)
    return _GL_NODES[n]

try:
    import equinox as eqx
    EQX_AVAILABLE = True
except ImportError:
    eqx = None  # type: ignore
    EQX_AVAILABLE = False

try:
    import optimistix as optx
    OPTX_AVAILABLE = True
except ImportError:
    optx = None  # type: ignore
    OPTX_AVAILABLE = False


@dataclass
class FIRE2Result:
    rho: np.ndarray
    converged: bool
    iterations: int
    residual: float
    omega_final: float
    # Per-step ||∇Ω||_inf history. Populated only when collect_history=True.
    # None when using the fast optimistix path (no per-step access inside lax.while_loop).
    error_history: list | None = None


def _omega_fn(
    log_rho: jnp.ndarray,
    rho_bulk: float,
    Vext_K: jnp.ndarray,
    temperature_K: float,
    c1_callable: Callable,
    c1_bulk: float,
    dV: float,
    accessibility_mask: jnp.ndarray | None,
    rho_min: float,
    f_exc_mode: FExcMode = "endpoint",
    n_quad: int = 4,
) -> jnp.ndarray:
    """Grand potential Ω[exp(ψ)] as a JAX scalar.

    Parameters
    ----------
    f_exc_mode : ``"endpoint"`` | ``"rpa"`` | ``"quadrature"``
        Approximation used for F_ex (see module docstring).
    n_quad : int
        Number of Gauss-Legendre quadrature points (only for ``"quadrature"``).
    """
    rho = jnp.clip(jnp.exp(log_rho), rho_min)
    if accessibility_mask is not None:
        rho = jnp.where(accessibility_mask, rho, 0.0)

    beta = 1.0 / temperature_K
    log_rho_bulk = float(np.log(rho_bulk + 1e-300))

    rho_safe = jnp.where(rho > 0, rho, rho_min)
    # All three terms in kBT units — d(Ω/kBT)/dρ = 0 gives EL condition.
    f_id = jnp.sum(rho * (jnp.log(rho_safe) - log_rho_bulk - 1.0)) * dV
    f_ext = jnp.sum(beta * Vext_K * rho) * dV

    if f_exc_mode == "endpoint":
        # λ=1 endpoint rule — one c¹ evaluation.
        c1 = jnp.asarray(c1_callable(rho))
        f_exc = jnp.sum((-c1 + c1_bulk) * rho) * dV

    elif f_exc_mode == "rpa":
        # Linear (RPA) approximation: F_ex ≈ ½ × (endpoint value).
        c1 = jnp.asarray(c1_callable(rho))
        f_exc = 0.5 * jnp.sum((-c1 + c1_bulk) * rho) * dV

    else:  # "quadrature"
        # Gauss-Legendre quadrature over adiabatic path λ ∈ [0,1].
        # Unrolled at trace time → single XLA graph; GPU-compatible.
        lam_nodes, lam_weights = _gl_nodes(n_quad)
        f_exc = jnp.zeros(())
        for lam_i, w_i in zip(lam_nodes, lam_weights):
            lam_i_j = jnp.asarray(lam_i)
            c1_lam = jnp.asarray(c1_callable(lam_i_j * rho))
            f_exc = f_exc + w_i * jnp.sum((-c1_lam + c1_bulk) * rho) * dV

    return f_id + f_exc + f_ext


def fire2_solve(
    rho_init: np.ndarray,
    rho_bulk: float,
    Vext_K: np.ndarray,
    temperature_K: float,
    c1_callable: Callable[[np.ndarray], np.ndarray],
    c1_bulk: float,
    dV: float = 1.0,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_steps: int = 10_000,
    rho_min: float = 1e-14,
    log_clip: float = 25.0,
    accessibility_mask: np.ndarray | None = None,
    collect_history: bool = False,
    collect_max_steps: int = 400,
    f_exc_mode: FExcMode = "endpoint",
    n_quad: int = 4,
) -> FIRE2Result:
    """Minimise Ω[ρ] with NonlinearCG (Polak-Ribières).

    Two execution paths — chosen via ``collect_history``:

    **Option A — fast path** (``collect_history=False``, default):
        Delegates to ``optimistix.NonlinearCG``.  The loop runs inside
        ``lax.while_loop`` (single XLA kernel), so it is fast but the
        internal per-step residuals are inaccessible.  ``FIRE2Result.error_history``
        is ``None``.  Use this path for production isotherms.

    **Option B — history path** (``collect_history=True``):
        Runs a Python-level PR+ CG loop with Armijo backtracking.  Each step
        records ``‖∇Ω‖_∞`` into ``FIRE2Result.error_history``.  Up to
        ``collect_max_steps`` steps are taken (default 400).  JIT is still
        applied per gradient call so speed is acceptable for single-point
        convergence demonstrations.  ``FIRE2Result.error_history`` is a list
        of floats.

    Parameters
    ----------
    rho_init, rho_bulk, Vext_K, temperature_K : standard solver inputs.
    c1_callable : callable
        c¹(ρ) → array same shape as ρ.  Must be JAX-traceable (use jnp ops).
    c1_bulk : float  — bulk reference value c¹(ρ_bulk, uniform).
    rtol, atol : convergence tolerances (optimistix path) / stopping criterion (history path).
    max_steps : hard iteration limit for the fast optimistix path.
    collect_history : bool
        When True, use Option B (Python PR+CG loop) and populate error_history.
    collect_max_steps : int
        Maximum CG steps for the history path.
    """
    import jax

    log_rho_bulk = float(np.log(rho_bulk + 1e-300))
    lo = log_rho_bulk - log_clip
    hi = log_rho_bulk + log_clip

    Vext_j = jnp.asarray(Vext_K, dtype=jnp.float32)
    mask_j = jnp.asarray(accessibility_mask, dtype=bool) if accessibility_mask is not None else None

    psi_init = jnp.clip(
        jnp.log(jnp.maximum(jnp.asarray(rho_init, dtype=jnp.float32), rho_min)),
        lo, hi,
    )
    if mask_j is not None:
        psi_init = jnp.where(mask_j, psi_init, lo)

    def wrapped_c1(rho):
        return jnp.asarray(c1_callable(rho))

    def omega_scalar(psi):
        return _omega_fn(psi, rho_bulk, Vext_j, temperature_K,
                         wrapped_c1, c1_bulk, dV, mask_j, rho_min,
                         f_exc_mode=f_exc_mode, n_quad=n_quad)

    # ── Option B: Python PR+ CG loop with history ─────────────────────────
    if collect_history:
        grad_fn   = jax.jit(jax.grad(omega_scalar))
        omega_jit = jax.jit(omega_scalar)

        psi = psi_init
        g   = grad_fn(psi)
        d   = -g                              # initial steepest-descent direction
        history: list[float] = []
        grad_norm = float(jnp.max(jnp.abs(g)))

        for k in range(collect_max_steps):
            history.append(grad_norm)
            if grad_norm < atol:
                break

            # PR+ beta: restart to steepest descent if non-descent
            g_flat  = g.ravel()
            gp_flat = g.ravel() if k == 0 else g_prev_flat  # type: ignore[used-before-assignment]
            if k > 0:
                beta = float(jnp.maximum(
                    0.0,
                    jnp.dot(g_flat, g_flat - gp_flat)
                    / (jnp.dot(gp_flat, gp_flat) + 1e-30),
                ))
                d = -g + beta * d
            # Restart if d is not a descent direction
            slope = float(jnp.sum(g * d))
            if slope >= 0:
                d = -g
                slope = -float(jnp.dot(g_flat, g_flat))

            # Armijo backtracking (up to 20 halvings)
            omega0 = float(omega_jit(psi))
            alpha  = 0.1
            for _ in range(20):
                if float(omega_jit(psi + alpha * d)) <= omega0 + 1e-4 * alpha * slope:
                    break
                alpha *= 0.5

            psi = jnp.clip(psi + alpha * d, lo, hi)
            if mask_j is not None:
                psi = jnp.where(mask_j, psi, lo)

            g_prev_flat = g_flat
            g            = grad_fn(psi)
            grad_norm    = float(jnp.max(jnp.abs(g)))

        rho_out = np.asarray(jnp.clip(jnp.exp(psi), rho_min))
        if accessibility_mask is not None:
            rho_out = np.where(accessibility_mask, rho_out, 0.0)

        return FIRE2Result(
            rho=rho_out,
            converged=grad_norm < atol * 10,
            iterations=len(history),
            residual=grad_norm,
            omega_final=float(omega_jit(psi)),
            error_history=history,
        )

    # ── Option A: fast optimistix path ────────────────────────────────────
    if not OPTX_AVAILABLE:
        raise ImportError(
            "FIRE2Solver requires optimistix.  Install with:\n"
            "    uv add optimistix"
        )

    def omega(psi, _args):
        return omega_scalar(psi)

    solver = optx.NonlinearCG(rtol=rtol, atol=atol, method=optx.polak_ribiere)
    result = optx.minimise(omega, solver, psi_init, max_steps=max_steps, throw=False)

    psi_out = jnp.clip(result.value, lo, hi)
    rho_out = np.asarray(jnp.clip(jnp.exp(psi_out), rho_min))
    if accessibility_mask is not None:
        rho_out = np.where(accessibility_mask, rho_out, 0.0)

    converged  = bool(result.result == optx.RESULTS.successful)
    n_steps    = int(result.stats["num_steps"])
    grad_norm  = float(jnp.max(jnp.abs(result.state.f_info.grad)))
    omega_val  = float(omega(result.value, None))

    return FIRE2Result(
        rho=rho_out,
        converged=converged,
        iterations=n_steps,
        residual=grad_norm,
        omega_final=omega_val,
        error_history=None,  # not available from lax.while_loop
    )


def fire2_solve_scan(
    rho_init: np.ndarray,
    rho_bulk: float,
    Vext_K: np.ndarray,
    temperature_K: float,
    c1_callable: Callable[[np.ndarray], np.ndarray],
    c1_bulk: float,
    dV: float = 1.0,
    max_steps: int = 200,
    step_size: float = 1.0,
    rho_min: float = 1e-14,
    log_clip: float = 25.0,
    accessibility_mask: np.ndarray | None = None,
    f_exc_mode: FExcMode = "endpoint",
    n_quad: int = 4,
) -> FIRE2Result:
    """Normalised gradient descent with ``jax.lax.scan`` — per-step history.

    Runs exactly ``max_steps`` of normalised-step gradient descent:
        ψ ← ψ − (step_size / ‖∇Ω‖_∞) · ∇Ω

    The entire loop is compiled as a single XLA ``scan`` kernel (JIT'd on
    first call, fast afterwards).  ``FIRE2Result.error_history`` is populated
    with ``‖∇Ω‖_∞`` at every step, giving a clean loss curve for benchmarking.

    Unlike ``optx.minimise`` (``lax.while_loop``), ``lax.scan`` unrolls output
    across a fixed number of steps so the history array is materialised.

    Unlike the Python-loop path (``collect_history=True``), this path is
    fully JIT-compiled — no Python per-step overhead, realistic wall-clock.

    Parameters
    ----------
    max_steps : int
        Fixed number of scan steps (no early stopping inside XLA).
    step_size : float
        Scaling constant for the normalised step; 1.0 works for most cases.
    """
    import jax

    log_rho_bulk = float(np.log(rho_bulk + 1e-300))
    lo = log_rho_bulk - log_clip
    hi = log_rho_bulk + log_clip

    Vext_j = jnp.asarray(Vext_K, dtype=jnp.float32)
    mask_j = jnp.asarray(accessibility_mask, dtype=bool) if accessibility_mask is not None else None

    psi_init = jnp.clip(
        jnp.log(jnp.maximum(jnp.asarray(rho_init, dtype=jnp.float32), rho_min)),
        lo, hi,
    )
    if mask_j is not None:
        psi_init = jnp.where(mask_j, psi_init, lo)

    def wrapped_c1(rho):
        return jnp.asarray(c1_callable(rho))

    def omega_scalar(psi):
        return _omega_fn(psi, rho_bulk, Vext_j, temperature_K,
                         wrapped_c1, c1_bulk, dV, mask_j, rho_min,
                         f_exc_mode=f_exc_mode, n_quad=n_quad)

    grad_fn = jax.grad(omega_scalar)

    def scan_step(psi, _):
        g          = grad_fn(psi)
        grad_norm  = jnp.max(jnp.abs(g))
        alpha      = step_size / (grad_norm + 1e-8)
        psi_new    = jnp.clip(psi - alpha * g, lo, hi)
        if mask_j is not None:
            psi_new = jnp.where(mask_j, psi_new, lo)
        return psi_new, grad_norm

    psi_final, history_j = jax.lax.scan(scan_step, psi_init, None, length=max_steps)

    history_np = np.asarray(history_j)
    rho_out    = np.asarray(jnp.clip(jnp.exp(psi_final), rho_min))
    if accessibility_mask is not None:
        rho_out = np.where(accessibility_mask, rho_out, 0.0)

    grad_norm_final = float(history_np[-1])
    omega_final     = float(omega_scalar(psi_final))

    return FIRE2Result(
        rho=rho_out,
        converged=bool(grad_norm_final < 1e-5),
        iterations=max_steps,
        residual=grad_norm_final,
        omega_final=omega_final,
        error_history=history_np.tolist(),
    )


if EQX_AVAILABLE:
    class FIRE2Solver(eqx.Module):
        """equinox Module wrapper around fire2_solve.

        Stores solver hyperparameters as static fields so the instance is
        jit-/vmap-compatible (hyperparameters are Python scalars, not arrays).
        """
        rtol: float = eqx.field(static=True)
        atol: float = eqx.field(static=True)
        max_steps: int = eqx.field(static=True)
        rho_min: float = eqx.field(static=True)
        log_clip: float = eqx.field(static=True)

        def __init__(
            self,
            rtol: float = 1e-7,
            atol: float = 1e-9,
            max_steps: int = 10_000,
            rho_min: float = 1e-14,
            log_clip: float = 25.0,
        ):
            self.rtol = rtol
            self.atol = atol
            self.max_steps = max_steps
            self.rho_min = rho_min
            self.log_clip = log_clip

        def solve(
            self,
            rho_init: np.ndarray,
            rho_bulk: float,
            Vext_K: np.ndarray,
            temperature_K: float,
            c1_callable: Callable,
            c1_bulk: float,
            dV: float = 1.0,
            accessibility_mask: np.ndarray | None = None,
            f_exc_mode: FExcMode = "endpoint",
            n_quad: int = 4,
        ) -> FIRE2Result:
            return fire2_solve(
                rho_init, rho_bulk, Vext_K, temperature_K,
                c1_callable, c1_bulk, dV,
                rtol=self.rtol,
                atol=self.atol,
                max_steps=self.max_steps,
                rho_min=self.rho_min,
                log_clip=self.log_clip,
                accessibility_mask=accessibility_mask,
                f_exc_mode=f_exc_mode,
                n_quad=n_quad,
            )
else:
    FIRE2Solver = None  # type: ignore
