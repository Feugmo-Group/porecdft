"""Dynamic classical DFT (DDFT) on the porecdft grid. PROTOTYPE.

Equation solved (overdamped, adiabatic closure; Marconi-Tarazona, Archer-Evans):

    d rho / dt = div [ D (grad rho + rho grad v) ],     v = beta V_ext - c1[rho]

with c1 the one-body direct correlation function of the static functional (FMT, WDA, association...)
exactly as used by ``porecdft.solver.anderson_solve``. The stationary state with no flux is therefore the
same discrete Euler-Lagrange equation  ln rho - c1[rho] + beta V_ext = const  that the static solvers solve.

Scheme: finite volume with Scharfetter-Gummel face fluxes (``flux``), fully implicit backward Euler.
  * Ideal gas (``c1_fn=None``): one sparse linear solve per step (LU cached for a constant step).
  * Otherwise each step solves R(rho) = rho - rho_n - dt F(rho) = 0 by inexact Newton-Krylov: GMRES with the
    sparse preconditioner (I - dt A(v(rho))), A the frozen-v operator. (Treating c1 explicitly is only
    conditionally stable for hard-sphere fluids because the thermodynamic factor is large.)
  * ``jax_c1=True`` (c1_fn written in JAX, no NumPy conversion inside): F is a jit-compiled, matrix-free flux
    divergence and Jacobian-vector products are exact (``jax.jvp``). Without it, products are finite differences,
    which stall GMRES once the residual falls near the finite-difference noise floor.

Reservoirs: voxels in ``fixed`` are held at a prescribed density (a gas buffer for an open system); with
``fixed=None`` and no-flux or periodic faces the system is closed and N is conserved.

Limitations of this prototype: first-order in time; one species; constant, per-axis or voxel-field mobility
(no loading dependence, no Maxwell-Stefan coupling); isothermal; sparse direct factorisation of the preconditioner
(no AMG, so grids beyond ~1e5 voxels need a different preconditioner).

Units: A, ps, K (V_ext_K / T_K = beta V).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres, splu

from porecdft.dynamics.flux import build_operator, face_weights


@dataclass
class DDFTResult:
    t: np.ndarray                      # times of the recorded states (ps)
    N: np.ndarray                      # molecules in the accessible, non-fixed voxels at those times
    rho: np.ndarray                    # final density, grid shape
    snapshots: dict = field(default_factory=dict)   # step time -> density, for requested snapshot times
    newton_iterations: int = 0
    converged: bool = True


class DDFTProblem:
    """Landscape, mobility, functional and boundaries of one species in a rigid host.

    Parameters
    ----------
    Vext_K   : (Nx, Ny, Nz) external potential in K (non-finite values only allowed where `access` is False).
    T_K      : temperature.
    spacing  : (hx, hy, hz) in A.
    D        : mobility in A^2/ps: scalar, (3,), or a voxel field.
    c1_fn    : callable rho -> c1 (grid-shaped); None for an ideal gas. With ``jax_c1=True`` it must be traceable
               by JAX (return a jax array, no ``np.asarray`` inside).
    periodic : per-axis periodicity; non-periodic axes are no-flux walls.
    access   : bool mask of accessible voxels (as in the static solvers).
    fixed    : bool mask of voxels held at the reservoir density (open system).
    """

    def __init__(self, Vext_K, T_K, spacing, D, c1_fn: Callable | None = None, jax_c1: bool = False,
                 periodic=(True, True, True), access=None, fixed=None):
        Vext_K = np.asarray(Vext_K, dtype=float)
        self.shape = Vext_K.shape
        self.n = Vext_K.size
        self.T_K = float(T_K)
        self.spacing = tuple(float(h) for h in spacing)
        self.dV = float(np.prod(self.spacing))
        self.D = D
        self.c1_fn = c1_fn
        self.jax_c1 = bool(jax_c1) and c1_fn is not None
        self.periodic = tuple(periodic)
        self.access = np.ones(self.shape, bool) if access is None else np.asarray(access, bool)
        bV = Vext_K / self.T_K
        self.bV = np.where(self.access & np.isfinite(bV), bV, 0.0)
        self.fixed = np.zeros(self.shape, bool) if fixed is None else np.asarray(fixed, bool)
        self._fixed_f = self.fixed.ravel()
        self._free_f = ~self._fixed_f
        self._count_f = (self.access & ~self.fixed).ravel()
        self._lu_cache: dict = {}
        self.stats = {"newton": 0, "jvp": 0, "gmres_unconverged": 0}
        if self.jax_c1:
            self._setup_jax()

    # ---- JAX path ---------------------------------------------------------------------------------------
    def _setup_jax(self):
        import jax
        import jax.numpy as jnp
        from porecdft.dynamics.flux import flux_divergence_jax

        shape = self.shape
        w = [None if q is None else jnp.asarray(q)
             for q in face_weights(shape, self.D, self.spacing, self.periodic, self.access)]
        bV, acc, c1 = jnp.asarray(self.bV), jnp.asarray(self.access), self.c1_fn

        def v_of(rho_flat):
            rho = rho_flat.reshape(shape)
            return jnp.where(acc, bV - c1(jnp.maximum(rho, 1e-300)), 0.0)

        def F(rho_flat):
            return flux_divergence_jax(rho_flat.reshape(shape), v_of(rho_flat), w).ravel()

        self._v_jit = jax.jit(v_of)
        self._F_jit = jax.jit(F)
        self._jvp_jit = jax.jit(lambda rho, x: jax.jvp(F, (rho,), (x,))[1])

    # ---- field, operator, right-hand side -------------------------------------------------------------
    def v_field(self, rho):
        rho = np.asarray(rho, float).ravel()
        if self.jax_c1:
            return np.asarray(self._v_jit(rho)).reshape(self.shape)
        v = self.bV
        if self.c1_fn is not None:
            v = v - np.asarray(self.c1_fn(np.maximum(rho.reshape(self.shape), 1e-300)))
        return np.where(self.access, v, 0.0)

    def operator(self, rho):
        return build_operator(self.v_field(rho), self.D, self.spacing, self.periodic, self.access)

    def rhs(self, rho):
        """d rho / dt (flat array)."""
        rho = np.asarray(rho, float).ravel()
        if self.jax_c1:
            return np.asarray(self._F_jit(rho))
        return self.operator(rho) @ rho

    def count(self, rho):
        """Molecules in the accessible, non-fixed voxels."""
        return float(np.sum(np.asarray(rho).ravel()[self._count_f]) * self.dV)

    # ---- one implicit step ----------------------------------------------------------------------------
    def _fixed_values(self, rho_fixed, t):
        """Reservoir density on the fixed voxels: scalar, per-fixed-voxel vector, or full-grid array; may be a callable of t."""
        if rho_fixed is None:
            return None
        val = np.asarray(rho_fixed(t) if callable(rho_fixed) else rho_fixed, float)
        nfix = int(self._fixed_f.sum())
        if val.ndim == 0:
            return np.full(nfix, float(val))
        if val.size == nfix:
            return val.ravel()
        return val.ravel()[self._fixed_f]

    def step(self, rho_n, dt, rho_fixed=None, t_new=0.0, newton_tol=1e-9, max_newton=30, gmres_tol=1e-6):
        """Advance by dt (backward Euler). Returns (rho_new, newton_iterations)."""
        rho_n = np.asarray(rho_n, float).ravel().copy()
        fixed = self._fixed_f
        fv = self._fixed_values(rho_fixed, t_new) if fixed.any() else None
        if fv is not None:
            rho_n[fixed] = fv
        if self.c1_fn is None:
            return self._step_linear(rho_n, dt, fv), 1
        return self._step_newton(rho_n, dt, fv, newton_tol, max_newton, gmres_tol)

    def _step_linear(self, rho_n, dt, fv):
        fixed, free = self._fixed_f, self._free_f
        key = (float(dt), int(fixed.sum()))
        if key not in self._lu_cache:
            A = self.operator(rho_n)
            P = (sp.identity(self.n, format="csr") - dt * A)[free][:, free].tocsc()
            AfF = A[free][:, fixed] if fixed.any() else None
            self._lu_cache[key] = (splu(P), AfF)
        lu, AfF = self._lu_cache[key]
        b = rho_n[free].copy()
        if AfF is not None:
            b += dt * (AfF @ fv)
        out = rho_n.copy()
        out[free] = lu.solve(b)
        return out

    def _step_newton(self, rho_n, dt, fv, tol, max_newton, gmres_tol):
        fixed, free = self._fixed_f, self._free_f
        nf = int(free.sum())
        rho = rho_n.copy()
        scale = max(float(np.max(rho[free])), 1e-300)
        its = 0
        for _ in range(max_newton):
            A = self.operator(rho)
            F = A @ rho if not self.jax_c1 else np.asarray(self._F_jit(rho))
            R = (rho - rho_n - dt * F)[free]
            if np.max(np.abs(R)) / scale < tol:
                break
            its += 1
            P = (sp.identity(self.n, format="csr") - dt * A)[free][:, free].tocsc()
            lu = splu(P)
            Mop = LinearOperator((nf, nf), matvec=lu.solve)
            nrm = np.linalg.norm(rho)

            def jv(x, rho=rho, F=F, nrm=nrm):
                self.stats["jvp"] += 1
                xf = np.zeros(self.n)
                xf[free] = x
                if self.jax_c1:
                    dF = np.asarray(self._jvp_jit(rho, xf))
                else:
                    eps = np.sqrt(np.finfo(float).eps) * max(1.0, nrm) / max(np.linalg.norm(xf), 1e-300)
                    dF = (self.rhs(rho + eps * xf) - F) / eps
                return x - dt * dF[free]

            dx, info = gmres(LinearOperator((nf, nf), matvec=jv), -R, M=Mop, rtol=gmres_tol, restart=30, maxiter=10)
            if info != 0:
                self.stats["gmres_unconverged"] += 1   # inexact Newton: accept the step, the outer loop corrects
            alpha = 1.0
            while np.any(rho[free] + alpha * dx <= 0.0) and alpha > 1e-8:
                alpha *= 0.5
            rho[free] += alpha * dx
        else:
            raise RuntimeError("Newton iteration did not converge; reduce dt")
        self.stats["newton"] += its
        return rho, its

    # ---- drivers --------------------------------------------------------------------------------------
    def run(self, rho0, t_end, dt, rho_fixed=None, snapshots=()):
        """Integrate to t_end with a constant step; record N(t) at every step."""
        rho = np.asarray(rho0, float).ravel().copy()
        nsteps = int(round(t_end / dt))
        t = np.zeros(nsteps + 1)
        N = np.zeros(nsteps + 1)
        fv0 = self._fixed_values(rho_fixed, 0.0) if self._fixed_f.any() else None
        if fv0 is not None:
            rho[self._fixed_f] = fv0
        N[0] = self.count(rho)
        snaps, want, total_its = {}, sorted(snapshots), 0
        for k in range(1, nsteps + 1):
            rho, its = self.step(rho, dt, rho_fixed, t_new=k * dt)
            total_its += its
            t[k], N[k] = k * dt, self.count(rho)
            while want and t[k] + 0.5 * dt >= want[0]:
                snaps[want.pop(0)] = rho.reshape(self.shape).copy()
        return DDFTResult(t, N, rho.reshape(self.shape), snaps, total_its)

    def relax(self, rho0, rho_fixed=None, dt0=1e-2, growth=1.25, dt_max=np.inf, t_max=1e8,
              tol=1e-9, max_steps=100000):
        """Relax to the stationary state with geometrically growing steps (backward Euler is unconditionally stable).

        Stops when max |d rho| / (dt max rho) < tol over the free voxels.
        """
        rho = np.asarray(rho0, float).ravel().copy()
        dt, t, total_its, ts, Ns = dt0, 0.0, 0, [0.0], [self.count(rho)]
        free = self._free_f
        for _ in range(max_steps):
            new, its = self.step(rho, dt, rho_fixed, t_new=t + dt)
            total_its += its
            rate = np.max(np.abs(new[free] - rho[free])) / (dt * max(np.max(new[free]), 1e-300))
            rho, t = new, t + dt
            ts.append(t)
            Ns.append(self.count(rho))
            if rate < tol:
                return DDFTResult(np.array(ts), np.array(Ns), rho.reshape(self.shape), {}, total_its, True)
            if t > t_max:
                break
            dt = min(dt * growth, dt_max)
        return DDFTResult(np.array(ts), np.array(Ns), rho.reshape(self.shape), {}, total_its, False)
