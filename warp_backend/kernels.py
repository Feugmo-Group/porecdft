"""Warp GPU kernels for porecdft's computational hot paths.

Kernels provided
----------------
rho_bar_sphere_kernel
    Accumulate ρ̄_s = (1/κ_s) ∫_{|r−r_s|<r_κ,s} ρ(r) dV for all M sites in
    parallel.  Each GPU thread handles one grid voxel; atomic adds scatter
    contributions to per-site accumulators.  This replaces the Python loop in
    ``WertheimAssociation._rho_bar_all``, which is O(M × Ng) in Python and the
    bottleneck for large 3D grids with many association sites.

rho_bar_sphere_warp(positions, rho, site_pos, site_r2, site_kappa, dV)
    JAX-callable wrapper.  On a CUDA device dispatches through
    ``warp.jax_kernel`` (GPU, zero-copy, autodiff-capable via enable_backward).
    On CPU falls back to a direct ``wp.launch`` + numpy roundtrip for
    correctness validation.
"""

from porecdft.warp_backend import WARP_AVAILABLE

_RHO_BAR_KERNEL = None
_RHO_BAR_JAX_KERNEL = None


def _build_rho_bar_kernel():
    """Build (and cache) the rho_bar_sphere Warp kernel + JAX wrapper."""
    global _RHO_BAR_KERNEL, _RHO_BAR_JAX_KERNEL
    if _RHO_BAR_KERNEL is not None:
        return _RHO_BAR_KERNEL, _RHO_BAR_JAX_KERNEL
    if not WARP_AVAILABLE:
        return None, None

    import warp as wp

    @wp.kernel
    def rho_bar_sphere_kernel(
        positions: wp.array(dtype=wp.vec3),    # (Ng,) grid positions  Å
        rho:       wp.array(dtype=wp.float32), # (Ng,) density mol/Å³
        site_pos:  wp.array(dtype=wp.vec3),    # (M,)  site positions  Å
        site_r2:   wp.array(dtype=wp.float32), # (M,)  r_κ² per site   Å²
        site_kappa:wp.array(dtype=wp.float32), # (M,)  κ_s             Å³
        dV:        float,                      # voxel volume          Å³
        rho_bar:   wp.array(dtype=wp.float32), # (M,)  OUTPUT (zeroed by caller)
    ):
        """Each thread = one voxel.  Contributes to every site whose sphere
        contains this voxel via atomic add."""
        tid = wp.tid()
        pos = positions[tid]
        rho_t = rho[tid]
        M = site_pos.shape[0]
        for s in range(M):
            dr = pos - site_pos[s]
            r2 = wp.dot(dr, dr)
            if r2 <= site_r2[s]:
                contrib = rho_t * dV / site_kappa[s]
                wp.atomic_add(rho_bar, s, contrib)

    _RHO_BAR_KERNEL = rho_bar_sphere_kernel

    has_cuda = any(d.alias.startswith("cuda") for d in wp.get_devices())
    if has_cuda:
        from warp import jax_kernel
        # enable_backward=True lets Warp auto-generate the kernel adjoint so
        # jax.grad flows through this call without a custom JVP rule.
        _RHO_BAR_JAX_KERNEL = jax_kernel(
            rho_bar_sphere_kernel, num_outputs=1, enable_backward=True
        )
    return _RHO_BAR_KERNEL, _RHO_BAR_JAX_KERNEL


def rho_bar_sphere_warp(positions, rho, site_pos, site_r2, site_kappa, dV):
    """Compute ρ̄_s for all M sites via a single Warp kernel launch.

    Parameters
    ----------
    positions : array (Ng, 3) float32 — Cartesian grid positions in Å.
    rho       : array (Ng,)   float32 — fluid density in mol/Å³.
    site_pos  : array (M, 3)  float32 — site Cartesian positions in Å.
    site_r2   : array (M,)    float32 — squared association radii r_κ² in Å².
    site_kappa: array (M,)    float32 — association volumes κ_s in Å³.
    dV        : float — voxel volume in Å³.

    Returns
    -------
    rho_bar : array (M,) float32
        Mean density × κ_s (dimensionless occupancy numerator) for each site,
        matching the quantity returned by WertheimAssociation._rho_bar_all.

    Raises
    ------
    RuntimeError
        If warp-lang is not installed.
    """
    import numpy as np

    if not WARP_AVAILABLE:
        raise RuntimeError(
            "warp-lang is not installed.  Install with `uv add warp-lang` "
            "to use the Warp backend for association kernels."
        )
    import warp as wp

    wk, jk = _build_rho_bar_kernel()
    Ng = rho.shape[0]
    M = site_pos.shape[0]

    # Flatten positions to (Ng, 3) and convert to float32 JAX/numpy arrays.
    try:
        import jax.numpy as jnp
        USE_JAX = True
    except ImportError:
        USE_JAX = False

    if USE_JAX and jk is not None:
        # GPU fast path: zero-copy DLPack through JAX
        pos_flat = jnp.asarray(positions, dtype=jnp.float32).reshape(-1, 3)
        rho_f    = jnp.asarray(rho,       dtype=jnp.float32).reshape(-1)
        sp_f     = jnp.asarray(site_pos,  dtype=jnp.float32)
        sr2_f    = jnp.asarray(site_r2,   dtype=jnp.float32)
        sk_f     = jnp.asarray(site_kappa,dtype=jnp.float32)
        rho_bar_init = jnp.zeros(M, dtype=jnp.float32)
        # jax_kernel returns a list [output_array]; dV is a Python float
        (rho_bar_out,) = jk(pos_flat, rho_f, sp_f, sr2_f, sk_f, float(dV), rho_bar_init)
        return rho_bar_out

    # CPU fallback: wp.launch on numpy buffers
    pos_np  = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    rho_np  = np.asarray(rho,       dtype=np.float32).reshape(-1)
    sp_np   = np.asarray(site_pos,  dtype=np.float32)
    sr2_np  = np.asarray(site_r2,   dtype=np.float32)
    sk_np   = np.asarray(site_kappa,dtype=np.float32)

    wp_pos  = wp.array(pos_np,  dtype=wp.vec3,    device="cpu")
    wp_rho  = wp.array(rho_np,  dtype=wp.float32, device="cpu")
    wp_sp   = wp.array(sp_np,   dtype=wp.vec3,    device="cpu")
    wp_sr2  = wp.array(sr2_np,  dtype=wp.float32, device="cpu")
    wp_sk   = wp.array(sk_np,   dtype=wp.float32, device="cpu")
    wp_out  = wp.zeros(M, dtype=wp.float32, device="cpu")

    wp.launch(wk, dim=Ng,
              inputs=[wp_pos, wp_rho, wp_sp, wp_sr2, wp_sk, float(dV)],
              outputs=[wp_out],
              device="cpu")
    return np.asarray(wp_out.numpy())
