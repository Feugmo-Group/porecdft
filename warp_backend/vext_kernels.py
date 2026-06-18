"""Warp GPU kernels for ``Vext`` evaluation on the 3D grid.

These kernels replace the heavy O(Ng × S × Na) Python/NumPy loops in
:mod:`porecdft.forcefield`.  Each thread handles one grid voxel and sweeps
the host atoms for one orientation.  Per (P, T) state-point we run the kernel
``N_orient`` times — typically 20 for Fibonacci-sphere averaging.

Kernels provided
----------------
lj_vext_grid_kernel
    Lennard–Jones (12-6) energy of a multi-site fluid at every grid voxel,
    for a single orientation.  Inputs are *already-rotated* fluid-site
    offsets (S, 3) and per-(site, host atom) σ_ij / ε_ij / mask arrays.

morse_vext_grid_kernel
    Morse-well energy ``D_e * ((1 - exp(-α (r - r_e)))^2 - 1)`` for
    transition-metal sites in metalated COFs (H₂ binding studies).  Only the
    flagged metal atoms contribute; the LJ kernel handles everything else.

smeared_coulomb_grid_kernel
    Gaussian-smeared Coulomb interaction
    ``Φ(r) = q · erf(r / (√2 σ_eff)) / (4π ε₀ r)`` evaluated for every fluid
    charge site against every host atom.

All three kernels accumulate into a single ``out`` array of shape ``(Ng,)``
(K-units, i.e. ε/k_B in K).  Run them in sequence with the same ``out``
buffer to sum across the composite force-field components.

Boltzmann orientation average
-----------------------------
``boltzmann_orient_avg_kernel`` does the final ``V(r;T) = -k_B T · ln (1/N_Ω
Σ_i exp(-β V_i))`` reduction across orientations *on device*, again with one
thread per voxel.  No further NumPy bookkeeping after the per-orientation
kernels have run.
"""

from porecdft.warp_backend import WARP_AVAILABLE

# Cached compiled kernels + JAX wrappers.  Built lazily on first use.
_LJ_KERNEL = None
_LJ_JAX_KERNEL = None
_MORSE_KERNEL = None
_MORSE_JAX_KERNEL = None
_COULOMB_KERNEL = None
_COULOMB_JAX_KERNEL = None
_BOLTZ_KERNEL = None
_BOLTZ_JAX_KERNEL = None


# ──────────────────────────────────────────────────────────────────────────
# 1. Lennard-Jones Vext kernel
# ──────────────────────────────────────────────────────────────────────────

def _build_lj_kernel():
    global _LJ_KERNEL, _LJ_JAX_KERNEL
    if _LJ_KERNEL is not None or not WARP_AVAILABLE:
        return _LJ_KERNEL, _LJ_JAX_KERNEL

    import warp as wp

    @wp.kernel
    def lj_vext_grid_kernel(
        grid_xyz:     wp.array(dtype=wp.vec3),      # (Ng,)  Å
        n_sites:      int,
        site_offset:  wp.array(dtype=wp.vec3),      # (S,)   lab-frame s_α already rotated, Å
        host_pos:     wp.array(dtype=wp.vec3),      # (Na,)  Å
        sigma_ij:     wp.array2d(dtype=wp.float32), # (S, Na) Å
        epsilon_ij:   wp.array2d(dtype=wp.float32), # (S, Na) K
        active:       wp.array2d(dtype=wp.int32),   # (S, Na) 1 = include, 0 = skip
        cutoff2:      float,                         # Å²
        out:          wp.array(dtype=wp.float32),    # (Ng,) accumulated V (K), zero-initialised
    ):
        g = wp.tid()
        grid = grid_xyz[g]
        Na = host_pos.shape[0]
        v_total = wp.float32(0.0)
        for s in range(n_sites):
            pos = grid + site_offset[s]
            for a in range(Na):
                if active[s, a] == 0:
                    continue
                dr = host_pos[a] - pos
                r2 = wp.dot(dr, dr)
                if r2 > 0.0 and r2 < cutoff2:
                    sig = sigma_ij[s, a]
                    eps = epsilon_ij[s, a]
                    sr2 = sig * sig / r2
                    sr6 = sr2 * sr2 * sr2
                    v_total = v_total + 4.0 * eps * (sr6 * sr6 - sr6)
        out[g] = out[g] + v_total

    _LJ_KERNEL = lj_vext_grid_kernel

    has_cuda = any(d.alias.startswith("cuda") for d in wp.get_devices())
    if has_cuda:
        from warp import jax_kernel
        _LJ_JAX_KERNEL = jax_kernel(
            lj_vext_grid_kernel, num_outputs=1, enable_backward=False
        )
    return _LJ_KERNEL, _LJ_JAX_KERNEL


# ──────────────────────────────────────────────────────────────────────────
# 2. Morse Vext kernel (transition-metal binding for H₂/COF studies)
# ──────────────────────────────────────────────────────────────────────────

def _build_morse_kernel():
    global _MORSE_KERNEL, _MORSE_JAX_KERNEL
    if _MORSE_KERNEL is not None or not WARP_AVAILABLE:
        return _MORSE_KERNEL, _MORSE_JAX_KERNEL

    import warp as wp

    @wp.kernel
    def morse_vext_grid_kernel(
        grid_xyz:    wp.array(dtype=wp.vec3),       # (Ng,)
        site_offset: wp.vec3,                       # single fluid site offset (s_α) Å (already rotated)
        host_pos:    wp.array(dtype=wp.vec3),       # (Na,)
        D_e:         wp.array(dtype=wp.float32),    # (Na,) well depth in K
        alpha_w:     wp.array(dtype=wp.float32),    # (Na,) curvature 1/Å
        r_e:         wp.array(dtype=wp.float32),    # (Na,) equilibrium distance Å
        active:      wp.array(dtype=wp.int32),      # (Na,) 1 = metal site, 0 = skip
        cutoff:      float,                          # Å
        out:         wp.array(dtype=wp.float32),     # (Ng,) accumulator (K)
    ):
        g = wp.tid()
        pos = grid_xyz[g] + site_offset
        v = wp.float32(0.0)
        Na = host_pos.shape[0]
        for a in range(Na):
            if active[a] == 0:
                continue
            dr = host_pos[a] - pos
            r2 = wp.dot(dr, dr)
            if r2 > 0.0 and r2 < cutoff * cutoff:
                r = wp.sqrt(r2)
                x = wp.exp(-alpha_w[a] * (r - r_e[a]))
                # Morse: D_e * ((1 - x)^2 - 1) — depth at r=r_e is -D_e
                v_pair = D_e[a] * ((1.0 - x) * (1.0 - x) - 1.0)
                # Clamp to physical range
                if v_pair < -D_e[a]:
                    v_pair = -D_e[a]
                v = v + v_pair
        out[g] = out[g] + v

    _MORSE_KERNEL = morse_vext_grid_kernel

    has_cuda = any(d.alias.startswith("cuda") for d in wp.get_devices())
    if has_cuda:
        from warp import jax_kernel
        _MORSE_JAX_KERNEL = jax_kernel(
            morse_vext_grid_kernel, num_outputs=1, enable_backward=False
        )
    return _MORSE_KERNEL, _MORSE_JAX_KERNEL


# ──────────────────────────────────────────────────────────────────────────
# 3. Gaussian-smeared Coulomb kernel
# ──────────────────────────────────────────────────────────────────────────

def _build_coulomb_kernel():
    global _COULOMB_KERNEL, _COULOMB_JAX_KERNEL
    if _COULOMB_KERNEL is not None or not WARP_AVAILABLE:
        return _COULOMB_KERNEL, _COULOMB_JAX_KERNEL

    import warp as wp

    @wp.kernel
    def smeared_coulomb_grid_kernel(
        grid_xyz:    wp.array(dtype=wp.vec3),       # (Ng,)
        n_sites:     int,
        site_offset: wp.array(dtype=wp.vec3),       # (S,) Å (already rotated)
        site_q:      wp.array(dtype=wp.float32),    # (S,) fluid-site charges (e)
        host_pos:    wp.array(dtype=wp.vec3),       # (Na,) — replicated over PBC by caller
        host_q:      wp.array(dtype=wp.float32),    # (Na,)
        sigma_eff:   float,                          # σ_eff (√2 σ) in Å
        cutoff2:     float,
        prefactor_K: float,                          # k_e · e² / k_B (K·Å) — converts q·q/r to K
        out:         wp.array(dtype=wp.float32),
    ):
        g = wp.tid()
        grid = grid_xyz[g]
        Na = host_pos.shape[0]
        v_total = wp.float32(0.0)
        for s in range(n_sites):
            pos = grid + site_offset[s]
            qs = site_q[s]
            if qs == 0.0:
                continue
            for a in range(Na):
                dr = host_pos[a] - pos
                r2 = wp.dot(dr, dr)
                if r2 > 0.0 and r2 < cutoff2:
                    r = wp.sqrt(r2)
                else:
                    r = 1.0
                erf_arg = r / sigma_eff
                v_total = v_total + prefactor_K * qs * host_q[a] / r * wp.erf(erf_arg)
        out[g] = out[g] + v_total

    _COULOMB_KERNEL = smeared_coulomb_grid_kernel

    has_cuda = any(d.alias.startswith("cuda") for d in wp.get_devices())
    if has_cuda:
        from warp import jax_kernel
        _COULOMB_JAX_KERNEL = jax_kernel(
            smeared_coulomb_grid_kernel, num_outputs=1, enable_backward=False
        )
    return _COULOMB_KERNEL, _COULOMB_JAX_KERNEL


# ──────────────────────────────────────────────────────────────────────────
# 4. Boltzmann orientation averaging
# ──────────────────────────────────────────────────────────────────────────

def _build_boltz_kernel():
    global _BOLTZ_KERNEL, _BOLTZ_JAX_KERNEL
    if _BOLTZ_KERNEL is not None or not WARP_AVAILABLE:
        return _BOLTZ_KERNEL, _BOLTZ_JAX_KERNEL

    import warp as wp

    @wp.kernel
    def boltzmann_orient_avg_kernel(
        v_per_orient: wp.array2d(dtype=wp.float32),  # (N_orient, Ng), in K (must be finite)
        T_K:          float,                          # temperature K
        v_min_clip:   float,                          # cap below this (e.g. -10000 K)
        v_max_clip:   float,                          # cap above this (e.g. +50000 K)
        out:          wp.array(dtype=wp.float32),     # (Ng,) free-energy averaged V_ext (K)
    ):
        """Per-voxel orientation averaging via -kT·log<exp(-βV)>_Ω.

        Uses the min-shift trick: subtract the per-voxel minimum before
        exp() to avoid overflow.  Each thread handles one voxel; the
        kernel performs the min-search and the log-sum-exp in-place.
        """
        g = wp.tid()
        n_orient = v_per_orient.shape[0]
        # min sweep
        v_min = wp.float32(1.0e30)
        for k in range(n_orient):
            v = v_per_orient[k, g]
            if v < v_min_clip:
                v = v_min_clip
            if v > v_max_clip:
                v = v_max_clip
            if v < v_min:
                v_min = v
        # log-sum-exp
        beta = 1.0 / T_K
        acc = wp.float32(0.0)
        for k in range(n_orient):
            v = v_per_orient[k, g]
            if v < v_min_clip:
                v = v_min_clip
            if v > v_max_clip:
                v = v_max_clip
            acc = acc + wp.exp(-beta * (v - v_min))
        # -kT ln (1/N · sum) = v_min - kT ln(acc / N)
        out[g] = v_min - T_K * wp.log(acc / wp.float32(n_orient))

    _BOLTZ_KERNEL = boltzmann_orient_avg_kernel

    has_cuda = any(d.alias.startswith("cuda") for d in wp.get_devices())
    if has_cuda:
        from warp import jax_kernel
        _BOLTZ_JAX_KERNEL = jax_kernel(
            boltzmann_orient_avg_kernel, num_outputs=1, enable_backward=False
        )
    return _BOLTZ_KERNEL, _BOLTZ_JAX_KERNEL


# ──────────────────────────────────────────────────────────────────────────
# Public Python wrappers (CPU fallback baked in)
# ──────────────────────────────────────────────────────────────────────────

def lj_vext_grid_warp(
    grid_xyz, site_offset, host_pos,
    sigma_ij, epsilon_ij, active,
    cutoff, out=None,
):
    """Compute LJ V_ext on a 3D grid for one orientation via Warp.

    See :func:`_build_lj_kernel` for the underlying kernel.

    Parameters
    ----------
    grid_xyz : (Ng, 3) array
    site_offset : (S, 3) array — already-rotated fluid-site lab offsets
    host_pos : (Na, 3) array
    sigma_ij, epsilon_ij : (S, Na) arrays (Å, K)
    active : (S, Na) int array (1=include, 0=skip)
    cutoff : float, Å
    out : optional (Ng,) array — accumulated in-place if given, else zero-init.

    Returns
    -------
    (Ng,) numpy or JAX array — V_ext contribution from LJ, in K.
    """
    if not WARP_AVAILABLE:
        raise RuntimeError("warp-lang not installed.")
    import numpy as np
    import warp as wp
    wk, _ = _build_lj_kernel()
    Ng = grid_xyz.shape[0]
    S = site_offset.shape[0]

    gp  = np.ascontiguousarray(np.asarray(grid_xyz,    dtype=np.float32).reshape(-1, 3))
    so  = np.ascontiguousarray(np.asarray(site_offset, dtype=np.float32).reshape(-1, 3))
    hp  = np.ascontiguousarray(np.asarray(host_pos,    dtype=np.float32).reshape(-1, 3))
    sij = np.ascontiguousarray(np.asarray(sigma_ij,    dtype=np.float32))
    eij = np.ascontiguousarray(np.asarray(epsilon_ij,  dtype=np.float32))
    act = np.ascontiguousarray(np.asarray(active,      dtype=np.int32))
    if out is None:
        out_np = np.zeros(Ng, dtype=np.float32)
    else:
        out_np = np.ascontiguousarray(np.asarray(out, dtype=np.float32))

    device = "cpu"
    wp_grid    = wp.array(gp,  dtype=wp.vec3,    device=device)
    wp_site    = wp.array(so,  dtype=wp.vec3,    device=device)
    wp_host    = wp.array(hp,  dtype=wp.vec3,    device=device)
    wp_sigma   = wp.array(sij, dtype=wp.float32, device=device)
    wp_eps     = wp.array(eij, dtype=wp.float32, device=device)
    wp_active  = wp.array(act, dtype=wp.int32,   device=device)
    wp_out     = wp.array(out_np, dtype=wp.float32, device=device)

    wp.launch(wk, dim=Ng,
              inputs=[wp_grid, S, wp_site, wp_host,
                      wp_sigma, wp_eps, wp_active,
                      float(cutoff * cutoff)],
              outputs=[wp_out], device=device)
    return wp_out.numpy()

def coulomb_vext_grid_warp(
    grid_xyz, 
    site_offset, 
    site_q,
    host_pos,
    host_q,  
    sigma_eff,
    prefactor_K,
    cutoff, 
    out=None,
):    
    """Compute Coulomb V_ext on a 3D grid for one orientation via Warp.

    See :func:`_build_lj_kernel` for the underlying kernel.

    Parameters
    ----------
    grid_xyz : (Ng, 3) array
    site_offset : (S, 3) array — already-rotated fluid-site lab offsets
    site_q, (S,) fluid-site charges (e)
    host_pos : (Na, 3) array
    host_q: (Na) host-site cÅharges(e)
    sigma_ff : float σ_eff (√2 σ) in Å
    prefactor_K: float
    cutoff : float, 
    out : optional (Ng,) array — accumulated in-place if given, else zero-init.

    Returns
    -------
    (Ng,) numpy or JAX array — V_ext contribution from LJ, in K.
    """
    if not WARP_AVAILABLE:
        raise RuntimeError("warp-lang not installed.")
    import numpy as np
    import warp as wp
    wk, _ = _build_coulomb_kernel()
    Ng = grid_xyz.shape[0]
    S = site_offset.shape[0]

    gp  = np.ascontiguousarray(np.asarray(grid_xyz,    dtype=np.float32).reshape(-1, 3))
    so  = np.ascontiguousarray(np.asarray(site_offset, dtype=np.float32).reshape(-1, 3))
    hp  = np.ascontiguousarray(np.asarray(host_pos,    dtype=np.float32).reshape(-1, 3))
    siteq = np.ascontiguousarray(np.asarray(site_q,    dtype=np.float32))
    hostq = np.ascontiguousarray(np.asarray(host_q,  dtype=np.float32))
    if out is None:
        out_np = np.zeros(Ng, dtype=np.float32)
    else:
        out_np = np.ascontiguousarray(np.asarray(out, dtype=np.float32))

    device = "cpu"
    wp_grid    = wp.array(gp,  dtype=wp.vec3,    device=device)
    wp_site    = wp.array(so,  dtype=wp.vec3,    device=device)
    wp_host    = wp.array(hp,  dtype=wp.vec3,    device=device)
    wp_siteq   = wp.array(siteq, dtype=wp.float32, device=device)
    wp_hostq     = wp.array(hostq, dtype=wp.float32, device=device)
    wp_out     = wp.array(out_np, dtype=wp.float32, device=device)

    wp.launch(wk, dim=Ng,
              inputs=[wp_grid, S, wp_site, wp_siteq, wp_host, wp_hostq, 
                      float(sigma_eff), float(cutoff * cutoff), float(prefactor_K)],
              outputs=[wp_out], device=device)
    return wp_out.numpy()


def boltzmann_orient_avg_warp(v_per_orient, T_K,
                              v_min_clip=-10000.0, v_max_clip=50000.0):
    """Free-energy orientation average ``V(r;T) = -kT·log<exp(-βV)>_Ω`` via Warp.

    Parameters
    ----------
    v_per_orient : (N_orient, Ng) array of V values (K)
    T_K : float
    v_min_clip, v_max_clip : float — per-voxel clamps to suppress singularities

    Returns
    -------
    (Ng,) numpy array of free-energy averaged V (K).
    """
    if not WARP_AVAILABLE:
        raise RuntimeError("warp-lang not installed.")
    import numpy as np
    import warp as wp
    wk, _ = _build_boltz_kernel()
    n_or, Ng = v_per_orient.shape
    vpo = np.ascontiguousarray(np.asarray(v_per_orient, dtype=np.float32))

    wp_v   = wp.array(vpo, dtype=wp.float32, device="cpu")
    wp_out = wp.zeros(Ng, dtype=wp.float32, device="cpu")
    wp.launch(wk, dim=Ng,
              inputs=[wp_v, float(T_K), float(v_min_clip), float(v_max_clip)],
              outputs=[wp_out], device="cpu")
    return wp_out.numpy()
