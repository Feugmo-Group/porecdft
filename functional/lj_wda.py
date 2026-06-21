"""Weighted-Density Approximation (WDA) for the LJ attractive contribution.

Adds the dispersive c¹_att on top of the aWBII hard-sphere c¹_HS from
``porecdft.functional.fmt``.  Together they form the full aWBII+WDA
functional used in the H₂/COF adsorption calculations.

Physics
-------
The WDA approximates the attractive free energy density as:

    F_att[ρ] = ∫ [f_LJ(ρ̄(r), T) − f_HS(ρ̄(r), T)] dV

where ρ̄(r) = ∫ ρ(r')·w(r−r') dr' is the WDA-weighted density and
w is a normalised sphere of radius ψ·d (ψ = 1.3862, d = BH diameter).

The first functional derivative (to be added to c¹_HS):

    c¹_att(r) = −β · IFFT{ FFT[μ_att(ρ̄(r))] · w_hat }

with  μ_att(ρ̄) = μ_exc_LJ(ρ̄, T) − k_B T · βμ_HS(ρ̄)

and the bulk reference:

    c¹_att_bulk = −β · μ_att(ρ_bulk)

Reference: Denton & Ashcroft, *Phys. Rev. A* 39 (1989) 4701 (WDA); Johnson,
Zollweg & Gubbins, *Mol. Phys.* 78 (1993) 591 (MBWR EOS).
"""
from __future__ import annotations

import jax.numpy as jnp

from porecdft.eos.lj_mbwr import LJEOS, bh_diameter
from porecdft.functional.fmt import (
    make_k_grid,
    lanczos_filter,
    make_fmt_weights_hat,
    compute_weighted_densities,
    compute_c1 as _c1_hs,
    bulk_c1 as _bulk_c1_hs,
)
from porecdft.functional.fmt_weights import (
    w3FT,
    phi1func, dphi1dnfunc,
    phi2func, dphi2dnfunc,
    phi3func, dphi3dnfunc,
)

# ψ from Denton & Ashcroft 1991 — minimises bulk EOS error for the WDA sphere
_PSI = 1.3862


def _hs_betamu(rho, sigma: float, model: str = "aWBII") -> jnp.ndarray:
    """Hard-sphere excess chemical potential / kT for the WDA reference.

    The model must match the FMT functional used (default ``"aWBII"``).
    """
    n3 = rho * jnp.pi * sigma**3 / 6
    n3 = jnp.where(n3 >= 1.0, 1.0 - 1e-12, n3)
    n2 = rho * jnp.pi * sigma**2
    n1 = rho * sigma / 2
    n0 = rho
    dphi_dn0 = phi1func(n3)
    dphi_dn1 = n2 * phi2func(n3, model=model)
    dphi_dn2 = n1 * phi2func(n3, model=model) + 3 * n2**2 * phi3func(n3, model=model)
    dphi_dn3 = (n0 * dphi1dnfunc(n3)
                + n1 * n2 * dphi2dnfunc(n3, model=model)
                + n2**3  * dphi3dnfunc(n3, model=model))
    return (dphi_dn0
            + dphi_dn1 * sigma / 2
            + dphi_dn2 * jnp.pi * sigma**2
            + dphi_dn3 * jnp.pi * sigma**3 / 6)


class LJWDAFunctional:
    """Full aWBII + WDA functional: hard-sphere (aWBII) + LJ attractive (WDA).

    Parameters
    ----------
    sigma : float
        LJ / fluid diameter in Å.
    epsilon : float
        LJ well depth in K (ε/k_B convention).
    temperature_K : float
        System temperature (K). Fixes the BH diameter d and the WDA weight.
    model_hs : str
        Hard-sphere functional variant (default ``"aWBII"``).
    model_lj : str
        MBWR coefficient set — ``"NewMBWR"`` (default) or ``"MBWR"``.
    """

    def __init__(
        self,
        sigma: float,
        epsilon: float,
        temperature_K: float,
        model_hs: str = "aWBII",
        model_lj: str = "NewMBWR",
    ):
        self.sigma = sigma
        self.epsilon = epsilon
        self.T = temperature_K
        self.beta = 1.0 / temperature_K
        self.model_hs = model_hs
        self._ljeos = LJEOS(sigma=sigma, epsilon=epsilon, model=model_lj)

        # Barker-Henderson effective hard-sphere diameter
        self.d = bh_diameter(temperature_K, sigma=sigma, epsilon=epsilon)

        # WDA sphere radius
        self._r_wda = 2.0 * _PSI * self.d
        # Cached per grid shape: shape → (w2_hat, w3_hat, w2vec_hat, w_hat)
        self._weight_cache: dict = {}

    def _get_weights(self, shape, dx, dy, dz):
        key = (shape, dx, dy, dz)
        if key not in self._weight_cache:
            KX, KY, KZ, K = make_k_grid(shape, dx, dy, dz)
            # FMT weights with Lanczos anti-aliasing filter (matches legacy reference)
            w2_hat, w3_hat, w2vec_hat = make_fmt_weights_hat(
                K, KX, KY, KZ, self.d, dx=dx, dy=dy, dz=dz
            )
            # WDA weight: normalised sphere of radius ψ·d, also Lanczos-filtered
            sigma_L = lanczos_filter(KX, KY, KZ, dx, dy, dz)
            w3_wda = w3FT(K, sigma=self._r_wda) * sigma_L
            w_hat  = w3_wda / (jnp.pi * self._r_wda**3 / 6)
            self._weight_cache[key] = (w2_hat, w3_hat, w2vec_hat, w_hat)
        return self._weight_cache[key]

    def _mu_att(self, rhobar):
        """Local attractive chemical potential μ_att = μ_exc_LJ − k_B T · βμ_HS."""
        return self._ljeos.muexc(rhobar, self.T) - self.T * _hs_betamu(rhobar, self.d, self.model_hs)

    def c1(
        self,
        rho: jnp.ndarray,
        dx: float,
        dy: float,
        dz: float,
        weights=None,
        w_hat=None,
    ) -> jnp.ndarray:
        """Total c¹(r) = c¹_HS(r) + c¹_att(r).

        Parameters
        ----------
        rho : jnp.ndarray, shape (Nx, Ny, Nz)
            Density field in molecules·Å⁻³.
        dx, dy, dz : float
            Grid spacings in Å.
        weights : FMTWeightsHat, optional
            Pre-computed FMT weight arrays.  When provided together with
            ``w_hat``, the internal ``_get_weights()`` call is skipped.
            Useful for GL quadrature loops where the same weights are reused
            across n_quad c¹ evaluations.  If None, weights are fetched from
            the internal cache (or computed on first call) as usual — all
            existing callers continue to work unchanged.
        w_hat : jnp.ndarray, optional
            Pre-computed WDA sphere weight in Fourier space.  Must be
            supplied together with ``weights``; ignored when ``weights`` is None.

        Returns
        -------
        jnp.ndarray
            Total direct correlation function field (dimensionless).

        Note
        ----
        When calling this method inside a ``jax.jit``-compiled function
        across multiple iterations (e.g., an isotherm pressure loop with
        ``jax_solve``), call ``self._get_weights(rho.shape, dx, dy, dz)``
        once *before* entering the loop to populate the weight cache with
        concrete arrays.  Without this, the cache stores JAX DynamicJaxprTracers
        from the first JIT trace which become stale on re-entry.
        """
        shape = rho.shape
        if weights is not None and w_hat is not None:
            w2_hat, w3_hat, w2vec_hat = weights
        else:
            w2_hat, w3_hat, w2vec_hat, w_hat = self._get_weights(shape, dx, dy, dz)

        # Hard-sphere part (aWBII)
        wd = compute_weighted_densities(rho, w2_hat, w3_hat, w2vec_hat, self.d)
        c1_hs = _c1_hs(rho, wd, w2_hat, w3_hat, w2vec_hat, self.d, self.model_hs)

        # WDA attractive part
        rho_hat = jnp.fft.fftn(rho)
        rhobar  = jnp.fft.ifftn(rho_hat * w_hat).real
        mu_att  = self._mu_att(rhobar)
        c1_att  = -self.beta * jnp.fft.ifftn(jnp.fft.fftn(mu_att) * w_hat).real

        return c1_hs + c1_att

    def c1_bulk(self, rho_bulk: float) -> float:
        """Bulk reference c¹(ρ_bulk) = c¹_HS_bulk + c¹_att_bulk.

        In the bulk ρ̄ = ρ_bulk, so c¹_att_bulk = −β · μ_att(ρ_bulk).
        """
        c1_hs_b  = _bulk_c1_hs(rho_bulk, self.d, self.model_hs)
        mu_att_b = float(self._mu_att(jnp.array(rho_bulk)))
        c1_att_b = -self.beta * mu_att_b
        return c1_hs_b + c1_att_b
