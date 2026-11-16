"""Dynamic classical DFT (DDFT) for gas uptake and release in frameworks. PROTOTYPE.

Same free energy and external potential as the static solvers; adds a mobility and time. See ``ddft`` for the
equations and the limitations of this prototype, and ``tests/test_dynamics.py`` for the exact-limit tests.
"""
# Finite-difference Jacobian-vector products in the implicit solver need float64 (float32 noise stalls GMRES),
# so enable it before any JAX array is created, as the ionax DDFT code does.
try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:  # the ideal-gas path is pure NumPy/SciPy
    pass

from porecdft.dynamics.flux import bernoulli, build_operator
from porecdft.dynamics.ddft import DDFTProblem, DDFTResult
from porecdft.dynamics.diagnostics import fractional_uptake, time_to_fraction, crank_slab

__all__ = [
    "bernoulli",
    "build_operator",
    "DDFTProblem",
    "DDFTResult",
    "fractional_uptake",
    "time_to_fraction",
    "crank_slab",
]
