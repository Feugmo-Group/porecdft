"""JAX ↔ Warp DLPack interop helpers.

Warp 1.14+ ships ``warp.jax_kernel`` and ``warp.jax_callable``, which wrap
``@wp.kernel`` / Python launcher functions into JAX primitives.  They handle
DLPack zero-copy interchange automatically and work inside ``@jax.jit`` traces.

This module:
  - guards imports so porecdft still works without warp-lang,
  - provides a friendly ``warp_kernel(my_kernel)`` alias,
  - documents the in-place gotcha: JAX arrays are immutable so a Warp kernel
    must write to an *output* array rather than mutating an input.
"""

from porecdft.warp_backend import WARP_AVAILABLE


def warp_kernel(kernel, num_outputs: int = 1, in_out_argnames=None):
    """Wrap a Warp @wp.kernel as a JAX-callable primitive.

    Parameters
    ----------
    kernel : warp.Kernel
        Function decorated with ``@wp.kernel``.
    num_outputs : int
        Trailing arguments that are outputs (default 1).
    in_out_argnames : list[str] | None
        Arguments that are read *and* written in-place.

    Returns
    -------
    callable
        Safe to call from JAX; works inside ``@jax.jit``.
    """
    if not WARP_AVAILABLE:
        raise RuntimeError(
            "warp-lang is not installed.  Install with `uv add warp-lang`."
        )
    import warp as wp
    if in_out_argnames is not None:
        return wp.jax_kernel(kernel, num_outputs=num_outputs, in_out_argnames=in_out_argnames)
    return wp.jax_kernel(kernel, num_outputs=num_outputs)


def warp_callable(py_func, num_outputs: int = 1, in_out_argnames=None):
    """Wrap a Python function that launches Warp kernels as a JAX-callable.

    Useful when you need multiple kernels in sequence (e.g., multiple weighted
    density accumulation passes) and want JAX to see the composite as one op.
    """
    if not WARP_AVAILABLE:
        raise RuntimeError("warp-lang is not installed; cannot use warp_callable.")
    import warp as wp
    if in_out_argnames is not None:
        return wp.jax_callable(py_func, num_outputs=num_outputs, in_out_argnames=in_out_argnames)
    return wp.jax_callable(py_func, num_outputs=num_outputs)


def manual_jax_to_warp(jax_array):
    """Zero-copy JAX → Warp via DLPack (fallback when jax_kernel is insufficient).

    Caller is responsible for keeping the JAX array alive while the Warp
    array is in use — DLPack does not transfer ownership.
    """
    if not WARP_AVAILABLE:
        raise RuntimeError("warp-lang is not installed.")
    import jax.dlpack
    import warp as wp
    return wp.from_dlpack(jax.dlpack.to_dlpack(jax_array))


def manual_warp_to_jax(warp_array, force_refresh: bool = True):
    """Zero-copy Warp → JAX via DLPack.

    Parameters
    ----------
    force_refresh : bool
        Run ``j + 0`` after conversion to force JAX to re-read mutated data.
        Matches the NVIDIA forum recommendation.
    """
    if not WARP_AVAILABLE:
        raise RuntimeError("warp-lang is not installed.")
    import jax.dlpack
    j = jax.dlpack.from_dlpack(warp_array)
    if force_refresh:
        j = j + 0
    return j
