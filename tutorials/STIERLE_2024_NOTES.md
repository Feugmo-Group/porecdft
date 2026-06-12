# Learnings from Stierle & Gross 2024 (`dft_ad_jax/`)

After importing the SI structures for the tutorials I read through their
`dft_ad_jax/` reference implementation.  Here are the ideas worth porting
back into porecdft.

## 1. Triclinic / skewed-cell support (`grid.py:_skewed2cart`)

Their `Grid` class accepts `skew_angles = [α, β, γ]` and transforms the
Fourier-space coordinates via

```
k_y' = (k_y − k_x cos γ) / sin γ
k_z' = ( |k_x|·(ζ cos γ − sin γ cos β) − |k_y|·ζ  +  k_z sin γ ) / det
det  = sin γ · √(1 − cos²β − ζ²)
ζ    = (cos α − cos β cos γ) / sin γ
```

with the Jacobian determinant `det` rolled into `dV`.  This lets the same
FFT machinery handle MIL-53 monoclinic, ZIF-8 cubic, COF hexagonal cells,
etc. — **today porecdft assumes orthorhombic** in
`make_k_grid` / `make_fmt_weights_hat`.

**Action:** add `triclinic` support to `porecdft.functional.fmt.make_k_grid`
guarded by a `skew_angles` keyword that defaults to `[π/2]*3` (no behaviour
change for existing scripts).

## 2. Real-FFT in the last axis (`grid.py:69-76`)

Their k-grid uses `jnp.fft.fftfreq` for the first two axes and
`jnp.fft.rfftfreq` for the last.  For real-valued ρ(r) this halves the
array size of every weighted density and roughly doubles FFT speed.

**Action:** in `functional/fmt.py` switch the last-axis k-grid to
`rfftfreq` and use `jnp.fft.rfftn` / `irfftn` instead of `fftn` / `ifftn`.
~2× speedup of the FMT convolutions.

## 3. Multi-component density convention `ρ[c, x, y, z]`

The component axis is **first**, so they can write
`density_bulk[:, None, None, None]` and `parameters.m[:, None, None, None]`
to broadcast PC-SAFT chain segments.  This makes mixture cDFT essentially
free — no IAST needed.

**Action:** add an optional 4-D `(C, Nx, Ny, Nz)` path through
`anderson_solve` / `picard_solve` so users get mixture cDFT directly.
Existing scalar calls keep working via single-component reshape.

## 4. Modular `HelmholtzFunctional` contributions

They decompose the PC-SAFT excess into three contributions, each subclassing
a common ABC:

```
contributions = [FMTAntiSym(p, g), HardChain(p, g), Dispersion(p, g)]
```

Each contribution exposes `_weight_functions(T)`, `weighted_density(ρ, ω)`,
`helmholtz_energy_density(n_α, T)`.  The DFT class just sums their gradients.

**Today** porecdft has `aWBII FMT`, `WertheimAssociation`,
`LJWDAFunctional` as separate modules but the composition is hand-wired
in each application script.  We could refactor toward a `FunctionalSum`
ABC mirroring their pattern; would simplify Tutorial 4-style mixture
scripts.

## 5. JIT pre-compilation with explicit shapes

```python
jit_shape = ShapeDtypeStruct(shape=[n_comp]+n_grid, dtype='float64')
self.df_drho = jit(self.df_drho).lower(jit_shape).compile()
```

Pre-compiles `df_drho` for fixed shape + dtype at solver init — no JIT
warm-up cost during the Picard loop.

**Action:** wrap `LJWDAFunctional.c1` in the same `lower().compile()` step
at solver entry; warm-up currently costs ~2 s on the first state point.

## 6. `InvalidIteration(Exception)` NaN detection

Their solver raises a `InvalidIteration(iter_index)` exception the instant
the residual goes NaN — the caller sees exactly which step blew up.

**Action:** lift the silent `break` in our `anderson_solve` (line 104:
`err > 1e10`) into a typed `SolverDiverged` exception; users currently get
a `converged=False` with no hint of where things blew up.

## 7. Lanczos σ stored on the grid, not the functional

Their `Grid.sigma` is the Lanczos sinc-product evaluated *once* at grid
construction time; every functional then multiplies its k-space weight
by `grid.sigma`.  porecdft currently recomputes Lanczos inside the FMT
weights — fine for single-functional runs, redundant when WDA + FMT +
Wertheim all evaluate it.

**Action:** move `lanczos_filter()` from `functional/fmt.py` to a
`functional/_grid.py` helper that produces a shared σ array, then
references it from every functional.

## 8. `Grid.dv` Jacobian for non-orthorhombic

They divide `dv` by the triclinic-Jacobian `det` so the integrals
`∫ ρ dV` remain correct for any cell shape.  Coupled with item 1, this
gives quantitative results on MIL-53 (during breathing) and any
non-cubic COF.

## 9. Their picard_iteration *real-space* form

```python
density += jnp.clip(damping * error, a_max=max_change * density_bulk)
```

with a per-component `max_change` cap.  We have the same idea via
`step_clip = 5.0` in `anderson_solve`, but their per-component
`density_bulk`-scaled cap is more robust for mixtures with very
different bulk densities.

**Action:** scale `step_clip` by `rho_bulk` when in 4-D component mode.

## 10. PC-SAFT bulk chemical-potential utility

`parameters.py` ships a `bulk_chemical_potential_residual` helper that
returns `μ_res / kT` exactly — porecdft has `bulk_c1` for FMT but no
unified bulk-μ helper for PC-SAFT chains.  We could add this to
`porecdft/eos/pc_saft.py`.

## Priority ranking for porecdft

| Idea | Impact | Effort |
|------|--------|--------|
| Real-FFT (rfftn) for FMT       | **High** (~2× speed)   | Low |
| Multi-component density axis   | **High** (mixtures!)   | Medium |
| Triclinic skew_angles          | High (MIL-53, COFs)    | Medium |
| Lanczos σ on Grid (shared)     | Low (cosmetic)         | Low |
| JIT pre-compile shapes         | Medium (warm-up gone)  | Low |
| `SolverDiverged` exception     | High (debugability)    | Low |
| Bulk μ_res utility for PC-SAFT | Low                    | Low |
| Modular `FunctionalSum` ABC    | Medium (cleanup)       | High |

The **top three for immediate porting** are `rfftn` + multi-component
density + triclinic, in that order.
