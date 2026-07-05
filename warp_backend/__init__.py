"""Optional NVIDIA Warp GPU backend for porecdft.

Exposes:
  - WARP_AVAILABLE : bool — set at import time, never raises ImportError.
  - get_warp_device(prefer="cuda") — pick a Warp device matching JAX.
  - rho_bar_sphere_warp — JAX-callable Warp kernel for Wertheim association
    sphere integrals (the inner hot loop of _rho_bar_all).

Design
------
Warp is **strictly optional**.  All kernels fall back to the NumPy/SciPy path
when warp-lang is not installed.  Import this package anywhere without guards.
"""

WARP_AVAILABLE = False
WARP_VERSION = None

try:
    import warp as _wp  # noqa: F401
    WARP_AVAILABLE = True
    WARP_VERSION = getattr(_wp, "__version__", "unknown")
except ImportError:
    pass


def get_warp_device(prefer: str = "cuda"):
    """Return a Warp device, preferring CUDA if available, else CPU.

    Returns None if warp-lang is not installed.
    """
    if not WARP_AVAILABLE:
        return None
    import warp as wp
    devices = [d.alias for d in wp.get_devices()]
    if prefer == "cuda" and any(d.startswith("cuda") for d in devices):
        return next(d for d in wp.get_devices() if d.alias.startswith("cuda"))
    return wp.get_device("cpu")


from porecdft.warp_backend.kernels import rho_bar_sphere_warp  # noqa: E402
from porecdft.warp_backend.vext_kernels import (  # noqa: E402
    lj_vext_grid_warp,
    coulomb_vext_grid_warp,
    boltzmann_orient_avg_warp,
    coulomb_vext_grid_warp
)

__all__ = [
    "WARP_AVAILABLE",
    "WARP_VERSION",
    "get_warp_device",
    # Functional / association
    "rho_bar_sphere_warp",
    # Vext on 3D grid
    "lj_vext_grid_warp",
    "coulomb_vext_grid_warp",
    "boltzmann_orient_avg_warp",
]
