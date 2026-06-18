#!/usr/bin/env python
# coding: utf-8

import numpy as np
from jax import numpy as jnp, random as jr, lax, config as jconfig
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import validate_sample, promote_shapes
jconfig.update("jax_enable_x64", True)

_TWO_PI = 2.0 * np.pi
 
 
def _cbrt(x):
    return jnp.sign(x) * jnp.abs(x) ** (1.0 / 3.0)
 
 
class CubGaussian(Distribution):
    """Single-field cubic-in-Gaussian noise, built from (probabilists') Hermite
    polynomials so each term is mean-zero and orthogonal under the N(0,1) measure:
 
        obs = loc + scale1 * He1(eps) + scale2 * He2(eps) + scale3 * He3(eps)
            = loc + scale1*eps + scale2*(eps**2 - 1) + scale3*(eps**3 - 3*eps),
        eps ~ N(0, 1).
 
    The He_n are the normal-ordered (Wick-ordered) powers :eps^n:; in the LSS bias
    literature this same subtraction is the renormalized operator, e.g. the
    renormalized cubic [delta^3] = delta^3 - 3*sigma^2*delta (here sigma^2 = 1).
 
    Why He3 and not a bare eps**3: bare eps**3 overlaps the linear term,
    <eps**3 * eps> = 3, so it would inject a cross-term into the variance
    (Var would gain 6*scale1*scale3 + extra) and be degenerate with scale1.
    Subtracting 3*eps (= using He3) removes that overlap, leaving the clean
 
        E[obs]   = loc
        Var[obs] = scale1**2 + 2*scale2**2 + 6*scale3**2
        E[(obs-loc)**3] = 2*scale2*(3*scale1**2 + 4*scale2**2
                                    + 18*scale1*scale3 + 54*scale3**2)
 
    Orthogonality only decouples the *variance* (the L2(Gaussian) inner product);
    higher cumulants still mix the coefficients, which is physical.
 
    Support is all of R whenever scale3 != 0 (a cubic is surjective). The map
    eps -> obs is a bijection only when it is monotone
    (scale3 > 0 and scale2**2 <= 3*scale3*(scale1 - 3*scale3)); outside that
    regime the cubic folds, the density is a sum over 1 or 3 real preimages, and
    has integrable singularities at the fold values (far in the tails when scale3
    is a small perturbative correction). Reduces exactly to QuadGaussian as
    scale3 -> 0, and to Normal(loc, scale1) as scale2, scale3 -> 0.
 
    Sign gauge: eps -> -eps maps (scale1, scale3) -> (-scale1, -scale3) with the
    same distribution; fixing scale1 > 0 removes it.
    """
 
    arg_constraints = {"loc": constraints.real,
                       "scale1": constraints.positive,
                       "scale2": constraints.real,
                       "scale3": constraints.real}
    support = constraints.real
    reparametrized_params = ["loc", "scale1", "scale2", "scale3"]
 
    def __init__(self, loc=0.0, scale1=1.0, scale2=0.0, scale3=0.0, *,
                 tol=1e-8, validate_args=None):
        self.loc, self.scale1, self.scale2, self.scale3 = promote_shapes(
            loc, scale1, scale2, scale3)
        bs = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale1),
                                  jnp.shape(scale2), jnp.shape(scale3))
        self.tol = tol
        super().__init__(batch_shape=bs, validate_args=validate_args)
 
    # ---- the cubic g(eps) and its derivative (obs = loc + g(eps)) -----------
    def _g(self, e):
        a, b, c = self.scale1, self.scale2, self.scale3
        return c * e ** 3 + b * e ** 2 + (a - 3 * c) * e - b
 
    def _gp(self, e):
        a, b, c = self.scale1, self.scale2, self.scale3
        return 3 * c * e ** 2 + 2 * b * e + (a - 3 * c)
 
    def _implicit(self, r_formula, Y):
        # value = the (stopped) Cardano root; gradient = implicit-function theorem,
        # which avoids the arccos/cbrt branch-point gradient pathologies.
        r = lax.stop_gradient(r_formula)
        h = self._g(r) - Y
        hp = self._gp(r)
        hp = jnp.where(jnp.abs(hp) < 1e-12, jnp.where(hp >= 0, 1e-12, -1e-12), hp)
        return r - h / hp
 
    def _roots(self, value):
        a, b, c = self.scale1, self.scale2, self.scale3
        Y = value - self.loc
        c_safe = jnp.where(jnp.abs(c) < self.tol, 1.0, c)
        p2, p1, p0 = b / c_safe, (a - 3 * c) / c_safe, -(b + Y) / c_safe
        shift = p2 / 3.0
        P = p1 - p2 ** 2 / 3.0
        Q = 2 * p2 ** 3 / 27.0 - p2 * p1 / 3.0 + p0
        Delta = Q ** 2 / 4.0 + P ** 3 / 27.0
        three = Delta < 0                                   # 3 real roots
        # trig branch (P < 0 guaranteed when three)
        P_neg = jnp.where(three, P, -1.0)
        m = 2.0 * jnp.sqrt(-P_neg / 3.0)
        arg = jnp.where(three, jnp.clip((3 * Q) / (P_neg * m),
                                        -1.0 + 1e-9, 1.0 - 1e-9), 0.0)
        th = jnp.arccos(arg) / 3.0
        et0 = m * jnp.cos(th) - shift
        et1 = m * jnp.cos(th - _TWO_PI / 3.0) - shift
        et2 = m * jnp.cos(th - 2.0 * _TWO_PI / 3.0) - shift
        # radical branch (guarded so cbrt never sees the bad input when three)
        Dpos = jnp.where(three, 1.0, Delta)
        sq = jnp.sqrt(jnp.where(Dpos > 0, Dpos, 0.0))
        ua = jnp.where(three, 1.0, -Q / 2 + sq)
        va = jnp.where(three, 1.0, -Q / 2 - sq)
        er = jnp.where(three, 0.0, _cbrt(ua) + _cbrt(va) - shift)
        e0 = jnp.where(three, et0, er)
        e1 = jnp.where(three, et1, er)        # = single root when not 'three'
        e2 = jnp.where(three, et2, er)
        e0 = self._implicit(e0, Y)
        e1 = self._implicit(e1, Y)
        e2 = self._implicit(e2, Y)
        return e0, e1, e2, three
 
    def _quad_logpdf(self, value):
        # scale3 ~ 0 fallback: this is QuadGaussian (and Normal when scale2 ~ 0).
        a, b = self.scale1, self.scale2
        Y = value - self.loc
        D = a ** 2 + 4 * b * (b + Y)
        Ds = jnp.where(D > 0, D, 1.0)
        sq = jnp.sqrt(Ds)
        b_s = jnp.where(jnp.abs(b) < 1e-12, 1.0, b)
        ep, em = (-a + sq) / (2 * b_s), (-a - sq) / (2 * b_s)
        quad = jnp.where(D > 0,
                         -0.5 * jnp.log(_TWO_PI) - 0.5 * jnp.log(Ds)
                         + logsumexp(jnp.stack([-0.5 * ep ** 2, -0.5 * em ** 2], 0), 0),
                         -jnp.inf)
        gauss = -0.5 * jnp.log(_TWO_PI) - jnp.log(jnp.abs(a)) - 0.5 * (Y / a) ** 2
        return jnp.where(jnp.abs(b) < 1e-8, gauss, quad)
 
    @validate_sample
    def log_prob(self, value):
        c = self.scale3
        e0, e1, e2, three = self._roots(value)
 
        def term(e, valid):
            lp = (-0.5 * jnp.log(_TWO_PI) - 0.5 * e ** 2
                  - jnp.log(jnp.abs(self._gp(e)) + 1e-30))
            return jnp.where(valid, lp, -jnp.inf)
 
        cubic = logsumexp(jnp.stack([term(e0, True),
                                     term(e1, three),
                                     term(e2, three)], 0), axis=0)
        return jnp.where(jnp.abs(c) < self.tol, self._quad_logpdf(value), cubic)
 
    def log_cdf(self, value):
        a, b, c = self.scale1, self.scale2, self.scale3
        Y = value - self.loc
        e0, e1, e2, _ = self._roots(value)
        r = jnp.sort(jnp.stack([e0, e1, e2], 0), axis=0)
        r0, r1, r2 = r[0], r[1], r[2]
        Phi = norm.cdf
        # P(g(eps) <= Y); the 3-root interval formula collapses to the 1-root
        # case automatically since coincident roots make the extra terms cancel.
        cdf_pos = Phi(r0) + Phi(r2) - Phi(r1)
        cdf_neg = Phi(r1) - Phi(r0) + 1.0 - Phi(r2)
        cdf_cub = jnp.clip(jnp.where(c > 0, cdf_pos, cdf_neg), 1e-300, 1.0)
        # scale3 ~ 0 fallback (QuadGaussian / Normal CDF)
        D = a ** 2 + 4 * b * (b + Y)
        Ds = jnp.where(D > 0, D, 1.0)
        sq = jnp.sqrt(Ds)
        b_s = jnp.where(jnp.abs(b) < 1e-12, 1.0, b)
        ep, em = (-a + sq) / (2 * b_s), (-a - sq) / (2 * b_s)
        hi, lo = jnp.maximum(ep, em), jnp.minimum(ep, em)
        cdf_pos2 = jnp.where(D > 0, jnp.clip(Phi(hi) - Phi(lo), 0.0, 1.0), 0.0)
        cdf_neg2 = jnp.where(D > 0, jnp.clip(Phi(lo) + 1.0 - Phi(hi), 0.0, 1.0), 1.0)
        cdf_quad = jnp.where(b > 0, cdf_pos2, cdf_neg2)
        cdf_qb = jnp.where(jnp.abs(b) < 1e-8, Phi(Y / a), cdf_quad)
        cdf = jnp.where(jnp.abs(c) < self.tol, cdf_qb, cdf_cub)
        return jnp.log(jnp.clip(cdf, 1e-300, 1.0))
 
    def cdf(self, value):
        return jnp.exp(self.log_cdf(value))
 
    def sample(self, key, sample_shape=()):
        eps = jr.normal(key, sample_shape + self.batch_shape)
        return (self.loc + self.scale1 * eps + self.scale2 * (eps ** 2 - 1.0)
                + self.scale3 * (eps ** 3 - 3.0 * eps))
 
    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.batch_shape)
 
    @property
    def variance(self):
        return jnp.broadcast_to(
            self.scale1 ** 2 + 2 * self.scale2 ** 2 + 6 * self.scale3 ** 2,
            self.batch_shape)
 


import matplotlib.pyplot as plt

from montecosmo.utils import QuadGaussian, TwoQuadGaussian

loc = 0.0
std = 1.0   # fix total std for all distributions

# (scale2, scale3): scale1 is derived to keep Var[obs] = std^2 for each model.
#   QG / TQG : scale1 = sqrt(std^2 - 2*s2^2)
#   CubGaussian: scale1 = sqrt(std^2 - 2*s2^2 - 6*s3^2)
configs = [
    (0.03, 0.02),
    (0.1,  0.05),
    (0.15, 0.1),
    (0.2,  0.1),
    (0.3,  0.15),
    (0.49, 0.2),
]

fig, axes = plt.subplots(2, 3, figsize=(13, 7))

for ax, (s2, s3) in zip(axes.flat, configs):
    s1    = float(jnp.sqrt(std**2 - 2.0*s2**2))
    s1_cg = float(jnp.sqrt(std**2 - 2.0*s2**2 - 6.0*s3**2))

    qg  = QuadGaussian(loc=loc, scale1=s1,    scale2=s2)
    tqg = TwoQuadGaussian(loc=loc, scale1=s1, scale2=s2)
    cg  = CubGaussian(loc=loc, scale1=s1_cg,  scale2=s2, scale3=s3)

    x = jnp.linspace(loc - 4.0*std, loc + 4.0*std, 800)

    ax.plot(x, norm.pdf(x, loc, std),       'k:', lw=1.5, alpha=0.5, label=f'Normal(σ={std:.2f})')
    ax.plot(x, jnp.exp(qg.log_prob(x)),     lw=2,          label='QuadGaussian')
    ax.plot(x, jnp.exp(tqg.log_prob(x)),    lw=2, ls='--',  label='TwoQuadGaussian')
    ax.plot(x, jnp.exp(cg.log_prob(x)),     lw=2, ls='-.',  label='CubGaussian')

    ax.set_title(f's2={s2},  s3={s3}', fontsize=11)
    ax.set_xlabel('x')
    ax.set_yscale('log')
    ax.legend(fontsize=8)

for ax in axes[:, 0]:
    ax.set_ylabel('PDF')

plt.suptitle(
    r'QuadGaussian  |  TwoQuadGaussian  |  CubGaussian  —  all with $\mathrm{Var}=1$' '\n'
    r'QG: $\mu + s_1\varepsilon + s_2(\varepsilon^2-1)$'
    r'   |   TQG: $\mu + s_1\varepsilon_1 + s_2(\varepsilon_2^2-1)$'
    r'   |   CG: $\mu + s_1\varepsilon + s_2(\varepsilon^2-1) + s_3(\varepsilon^3-3\varepsilon)$',
    fontsize=10,
)
plt.tight_layout()
plt.savefig('plot_gxy_stoch.png', dpi=150, bbox_inches='tight')
plt.show()
