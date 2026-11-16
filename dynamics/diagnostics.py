"""Uptake diagnostics and analytic references for the dynamic module."""
from __future__ import annotations

import numpy as np


def fractional_uptake(N, N_initial=None, N_final=None):
    """m(t) = (N(t) - N(0)) / (N(inf) - N(0)); N_final defaults to the last recorded value."""
    N = np.asarray(N, float)
    n0 = N[0] if N_initial is None else N_initial
    n1 = N[-1] if N_final is None else N_final
    return (N - n0) / (n1 - n0)


def time_to_fraction(t, m, fraction=0.9):
    """First time at which m(t) reaches `fraction` (linear interpolation); NaN if never reached."""
    t, m = np.asarray(t, float), np.asarray(m, float)
    above = np.nonzero(m >= fraction)[0]
    if above.size == 0:
        return float("nan")
    k = above[0]
    if k == 0:
        return float(t[0])
    return float(t[k - 1] + (fraction - m[k - 1]) * (t[k] - t[k - 1]) / (m[k] - m[k - 1]))


def crank_slab(t, D, ell, nterms=200):
    """Fractional uptake of a plane sheet of half thickness `ell` open on both faces (equivalently a slab of
    thickness `ell` open on one face, closed on the other), constant diffusivity D, small step:

        m(t) = 1 - (8/pi^2) sum_{n>=0} (2n+1)^-2 exp[-(2n+1)^2 pi^2 D t / (4 ell^2)].
    """
    t = np.atleast_1d(np.asarray(t, float))
    n = np.arange(nterms)[:, None]
    k = (2 * n + 1) ** 2
    return 1.0 - (8.0 / np.pi**2) * np.sum(np.exp(-k * np.pi**2 * D * t[None, :] / (4.0 * ell**2)) / k, axis=0)
