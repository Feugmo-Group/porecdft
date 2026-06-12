"""Fixed-point and gradient-based solvers for self-consistent cDFT.

Solvers
-------
picard_solve      — classical Picard mixing, NumPy, no extra deps.
anderson_solve    — Anderson-accelerated fixed-point, NumPy, no extra deps.
jax_solve         — Ω minimisation via any optax optimizer (Adam, SGD, …).
                    Requires: equinox, optax.
fire2_solve       — Ω minimisation via NonlinearCG (Polak-Ribières).
                    Requires: jax, optimistix.

All solvers share the same functional interface::

    result = <solver>_solve(
        rho_init, rho_bulk, Vext_K, temperature_K,
        c1_callable, c1_bulk, dV=..., accessibility_mask=...,
    )
    rho = result.rho
"""

from porecdft.solver.picard import picard_solve, PicardResult
from porecdft.solver.anderson import anderson_solve, AndersonResult
from porecdft.solver.jax_solver import (
    jax_solve,
    JaxSolverResult,
    grand_potential_jax,
    GrandPotentialSolver,
    OPTAX_AVAILABLE,
    EQX_AVAILABLE,
)
from porecdft.solver.fire2 import (
    fire2_solve,
    FIRE2Result,
    FIRE2Solver,
    OPTX_AVAILABLE,
)

__all__ = [
    "picard_solve",
    "PicardResult",
    "anderson_solve",
    "AndersonResult",
    "jax_solve",
    "JaxSolverResult",
    "grand_potential_jax",
    "GrandPotentialSolver",
    "OPTAX_AVAILABLE",
    "EQX_AVAILABLE",
    "fire2_solve",
    "FIRE2Result",
    "FIRE2Solver",
    "OPTX_AVAILABLE",
]
