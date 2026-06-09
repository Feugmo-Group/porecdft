"""Fixed-point and gradient-based solvers for self-consistent cDFT."""

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
]
