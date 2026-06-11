"""Lennard-Jones equation of state via the Modified Benedict-Webb-Rubin (MBWR) expansion.

Provides the excess free energy density ``fexc`` and excess chemical potential
``muexc`` of a Lennard-Jones fluid as functions of number density and temperature,
using either the Johnson 1993 (``"MBWR"``) or May & Mausbach 2012 (``"NewMBWR"``)
coefficient sets.  These quantities feed directly into the WDA attractive
functional in ``porecdft.functional.lj_wda``.

Also provides ``bh_diameter`` — the Barker-Henderson effective hard-sphere
diameter at a given temperature, which sets the WDA convolution sphere radius.

References
----------
Johnson, Zollweg & Gubbins, *Mol. Phys.* 78 (1993) 591.
May & Mausbach, *Phys. Rev. E* 85 (2012) 031201.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import jit

# ─── MBWR coefficients ────────────────────────────────────────────────────────

# Johnson 1993
_XLMBWR = (
    0.862308, 2.976218, -8.402230, 0.105413, -0.856458,
    1.582759, 0.763942, 1.753173, 2.798e3, -4.839e-2,
    0.996326, -3.698e1, 2.084e1, 8.305e1, -9.574e2,
    -1.478e2, 6.398e1, 1.604e1, 6.806e1, -2.791e3,
    -6.245128, -8.117e3, 1.489e1, -1.059e4, -1.131607e2,
    -8.867771e3, -3.986982e1, -4.689270e3, 2.593535e2,
    -2.694523e3, -7.218487e2, 1.721802e2,
)

# May & Mausbach 2012 (more accurate near critical point)
_XLNEW = (
    0.8623085097507421, 2.976218765822098, -8.402230115796038,
    0.1054136629203555, -0.8564583828174598, 1.44787318813706322,
    -0.310267527929454501, 3.26700773856663408, 4402.40210429518902,
    0.0165375389359225696, 7.42150201869250559, -40.7967106914122298,
    16.4537825382141350, 12.8389071227935610, -1407.06580259642897,
    -33.2251738947705988, 17.8209627529619184, -331.646081795803070,
    331.495131943892488, -4399.44711295106300, -3.05878673562233238,
    -12849.6469455607240, 9.96912508326940738, -16399.8349720621627,
    -256.926076715047884, -14588.020393359636, 88.3082960748521799,
    -6417.29842088150144, 121.307436784732417, -4461.88332740913756,
    -507.183302372831804, 37.2385794546305178,
)


def _acoef(kTstar, xlj):
    return [
        xlj[0]*kTstar + xlj[1]*jnp.sqrt(kTstar) + xlj[2] + xlj[3]/kTstar + xlj[4]/kTstar**2,
        xlj[5]*kTstar + xlj[6] + xlj[7]/kTstar + xlj[8]/kTstar**2,
        xlj[9]*kTstar + xlj[10] + xlj[11]/kTstar,
        xlj[12],
        xlj[13]/kTstar + xlj[14]/kTstar**2,
        xlj[15]/kTstar,
        xlj[16]/kTstar + xlj[17]/kTstar**2,
        xlj[18]/kTstar**2,
    ]


def _bcoef(kTstar, xlj):
    return [
        xlj[19]/kTstar**2 + xlj[20]/kTstar**3,
        xlj[21]/kTstar**2 + xlj[22]/kTstar**4,
        xlj[23]/kTstar**2 + xlj[24]/kTstar**3,
        xlj[25]/kTstar**2 + xlj[26]/kTstar**4,
        xlj[27]/kTstar**2 + xlj[28]/kTstar**3,
        xlj[29]/kTstar**2 + xlj[30]/kTstar**3 + xlj[31]/kTstar**4,
    ]


@jit
def _Gfunc(rhostar):
    gamma = 3.0
    F = jnp.exp(-gamma * rhostar**2)
    G0 = (1 - F) / (2 * gamma)
    G1 = -(F * rhostar**2 - 2*G0) / (2*gamma)
    G2 = -(F * rhostar**4 - 4*G1) / (2*gamma)
    G3 = -(F * rhostar**6 - 6*G2) / (2*gamma)
    G4 = -(F * rhostar**8 - 8*G3) / (2*gamma)
    G5 = -(F * rhostar**10 - 10*G4) / (2*gamma)
    return [G0, G1, G2, G3, G4, G5]


@jit
def _dGdrhos(rhostar):
    gamma = 3.0
    F = jnp.exp(-gamma * rhostar**2)
    dF = -2*gamma*rhostar*F
    dG1 = -dF / (2*gamma)
    dG2 = -(dF*rhostar**2 + 2*F*rhostar - 2*dG1) / (2*gamma)
    dG3 = -(dF*rhostar**4 + 4*F*rhostar**3 - 4*dG2) / (2*gamma)
    dG4 = -(dF*rhostar**6 + 6*F*rhostar**5 - 6*dG3) / (2*gamma)
    dG5 = -(dF*rhostar**8 + 8*F*rhostar**7 - 8*dG4) / (2*gamma)
    dG6 = -(dF*rhostar**10 + 10*F*rhostar**9 - 10*dG5) / (2*gamma)
    return [dG1, dG2, dG3, dG4, dG5, dG6]


def bh_diameter(kT: float, sigma: float = 1.0, epsilon: float = 1.0) -> float:
    """Barker-Henderson effective hard-sphere diameter (Å)."""
    kTstar = kT / epsilon
    return float(sigma * (1 + 0.2977*kTstar) / (1 + 0.33163*kTstar + 1.0477e-3*kTstar**2))


class LJEOS:
    """Lennard-Jones excess free energy and chemical potential via MBWR EOS.

    Parameters
    ----------
    sigma : float
        LJ diameter in Å.
    epsilon : float
        LJ well depth in K (ε/k_B convention).
    model : str
        ``"MBWR"`` (Johnson 1993) or ``"NewMBWR"`` (May 2012, default).
    """

    def __init__(self, sigma: float = 1.0, epsilon: float = 1.0, model: str = "NewMBWR"):
        self.sigma = sigma
        self.epsilon = epsilon
        self._xlj = _XLMBWR if model == "MBWR" else _XLNEW

    def _fLJ(self, rhostar, kTstar):
        a = _acoef(kTstar, self._xlj)
        b = _bcoef(kTstar, self._xlj)
        G = _Gfunc(rhostar)
        f = (a[0]*rhostar + a[1]*rhostar**2/2 + a[2]*rhostar**3/3 +
             a[3]*rhostar**4/4 + a[4]*rhostar**5/5 + a[5]*rhostar**6/6 +
             a[6]*rhostar**7/7 + a[7]*rhostar**8/8)
        f = f + b[0]*G[0] + b[1]*G[1] + b[2]*G[2] + b[3]*G[3] + b[4]*G[4] + b[5]*G[5]
        return f

    def _dfLJ_drhostar(self, rhostar, kTstar):
        a = _acoef(kTstar, self._xlj)
        b = _bcoef(kTstar, self._xlj)
        dG = _dGdrhos(rhostar)
        df = (a[0] + a[1]*rhostar + a[2]*rhostar**2 + a[3]*rhostar**3 +
              a[4]*rhostar**4 + a[5]*rhostar**5 + a[6]*rhostar**6 + a[7]*rhostar**7)
        df = df + b[0]*dG[0] + b[1]*dG[1] + b[2]*dG[2] + b[3]*dG[3] + b[4]*dG[4] + b[5]*dG[5]
        return df

    def fexc(self, rho, kT):
        """Excess (LJ) free energy density in K·Å⁻³."""
        kTstar = kT / self.epsilon
        rhostar = rho * self.sigma**3
        return self.epsilon * rho * self._fLJ(rhostar, kTstar)

    def muexc(self, rho, kT):
        """Excess (LJ) chemical potential in K."""
        kTstar = kT / self.epsilon
        rhostar = rho * self.sigma**3
        fLJ = self._fLJ(rhostar, kTstar)
        dfLJ = self._dfLJ_drhostar(rhostar, kTstar)
        return self.epsilon * (fLJ + rhostar * dfLJ)
