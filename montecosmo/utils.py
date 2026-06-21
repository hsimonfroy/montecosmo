from __future__ import annotations # for Union typing | in python<3.10

import pickle
import yaml

from functools import partial, wraps

import numpy as np
from jax import jit, numpy as jnp, random as jr, vmap, grad, tree, lax

from numpy.polynomial.hermite_e import hermegauss
from jax.scipy.special import logsumexp, gammaln
from jax.scipy.stats import norm


from numpyro.distributions import Distribution, constraints, TruncatedNormal, Uniform, util
from numpyro.distributions.util import validate_sample, promote_shapes



def safe_div(x, y):
    """
    Safe division, where division by zero is zero.
    Uses the "double-where" trick for safe gradient, 
    see https://github.com/jax-ml/jax/issues/5039
    """
    where_fn = jnp.where if isinstance(x, jnp.ndarray) or isinstance(y, jnp.ndarray) else np.where
    y_nozeros = where_fn(y==0, 1, y)
    return where_fn(y==0, 0, x / y_nozeros)

def nvmap(fun, n):
    """
    Nest vmap n times.
    """
    for _ in range(n):
        fun = vmap(fun)
    return fun

def vlim(a, level=1., scale=1., axis:int=None):
    """
    Return robust inferior and superior limit values of an array,
    i.e. discard quantiles bilateraly on some level, and scale the margins.
    """
    vmin, vmax = jnp.quantile(a, (1 - level) / 2, axis=axis), jnp.quantile(a, (1 + level) / 2, axis=axis)
    vmean, vdiff = (vmax + vmin) / 2, scale*(vmax - vmin) / 2
    return jnp.stack((vmean - vdiff, vmean + vdiff), axis=-1)

def get_jit(*args, **kwargs):
    """
    Return custom jit function that preserves function name and documentation.
    !!! example
        ```python
            @get_jit(static_argnums=(0))
            def my_func(x,y):
                return x+y
        ```
    """
    def custom_jit(fun):
        return wraps(fun)(jit(fun, *args, **kwargs))
    return custom_jit



#################
# Dump and Load #
#################
# class Path(type(Path()), Path):
#     """Pathlib path but with right-concatenation operator. Please tell me why it is not natively implemented."""
#     # See pathlib inheritance https://stackoverflow.com/questions/61689391/error-with-simple-subclassing-of-pathlib-path-no-flavour-attribute
#     def __add__(self, other):
#         if isinstance(other, (str, Path)):
#             return Path(str(self) + str(other))
#         return NotImplemented


def pdump(obj, path):
    """Pickle save"""
    with open(path, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def pload(path):
    """Pickle load"""
    with open(path, 'rb') as file:
        return pickle.load(file)    

def ydump(obj, path):
    """YAML dump"""
    with open(path, 'w') as file:
        yaml.dump(obj, file)

def yload(path):
    """YAML load"""
    with open(path, 'r') as file:
        return yaml.load(file, Loader=yaml.Loader)

def numpy_array_representer(dumper, data):
    return dumper.represent_list(data.tolist())

def numpy_array_constructor(loader, node):
    return np.array(loader.construct_sequence(node))

yaml.add_representer(np.ndarray, numpy_array_representer, Dumper=yaml.SafeDumper)
yaml.add_constructor(np.ndarray, numpy_array_constructor, Loader=yaml.SafeLoader)

def ysafe_dump(obj, path):
    """YAML safe dump"""
    with open(path, 'w') as file:
        yaml.safe_dump(obj, file)

def ysafe_load(path):
    """YAML safe load"""
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def h5save(path, data:dict):
    """
    Save a (possibly nested) dict to an HDF5 file. None values are skipped, nested
    dicts become groups, and everything else (arrays, scalars, strings, bools) is a
    dataset. Used for self-describing 'register' files (geometry, meshes, init, cosmo).
    """
    import h5py
    def _write(grp, d):
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, dict):
                _write(grp.create_group(k), v)
            else:
                grp[k] = v
    with h5py.File(str(path), 'w') as f:
        _write(f, data)


def h5load(path):
    """
    Load an HDF5 file written by `save_h5` into a (possibly nested) dict, with groups
    as sub-dicts and scalars/strings decoded to python.
    """
    import h5py
    def _read(grp):
        out = {}
        for k, item in grp.items():
            if isinstance(item, h5py.Group):
                out[k] = _read(item)
            else:
                v = item[()]
                if isinstance(v, bytes):
                    v = v.decode()
                out[k] = v
        return out
    with h5py.File(str(path), 'r') as f:
        return _read(f)



######################################
# Truncated Normal reparametrization #
######################################
def lowtail(x, low=-jnp.inf, high=None):
    temp = 1/6.2842226/2 # best temperature at 12 sigma
    energy = - jnp.stack(jnp.broadcast_arrays(x, low), axis=0)
    return temp * logsumexp( - energy / temp, axis=0)

def hightail(x, low=None, high=jnp.inf):
    temp = 1/6.2842226/2 # best temperature at 12 sigma
    energy = jnp.stack(jnp.broadcast_arrays(x, high), axis=0)
    return - temp * logsumexp( - energy / temp, axis=0)

def lowbody(x, low=-jnp.inf, high=jnp.inf):
    cdf_low, cdf_high = norm.cdf(low), norm.cdf(high)
    cdf_y = cdf_low + (cdf_high - cdf_low) * norm.cdf(x)
    return norm.ppf(cdf_y)

def highbody(x, low=-jnp.inf, high=jnp.inf):
    cdf_nlow, cdf_nhigh = norm.cdf(-low), norm.cdf(-high) # cdf(-x) = 1-cdf(x), more stable
    cdf_ny = cdf_nhigh - (cdf_nhigh - cdf_nlow) * norm.cdf(-x)
    return - norm.ppf(cdf_ny)

def body(x, low=-jnp.inf, high=jnp.inf):
    condlist = [x < 0.]
    funclist = [lowbody, highbody]
    return jnp.piecewise(x, condlist, funclist, low=low, high=high)    

def std2trunc(x, loc=0., scale=1., low=-jnp.inf, high=jnp.inf):
    """
    Transport standard normal variable to a general truncated normal variable. 
    """
    scale = jnp.asarray(scale)
    low, high = (low - loc) / scale, (high - loc) / scale
    lim = 12 # switch to a more stable approx at 12 sigma, for float32
    condlist = [(x < -lim) & (low < -lim), (lim < x) & (lim < high)]
    funclist = [lowtail, hightail, body]
    return loc + scale * jnp.piecewise(x, condlist, funclist, low=low, high=high)

 


def invlowbody(y, low=-jnp.inf, high=jnp.inf):
    cdf_low, cdf_high = norm.cdf(low), norm.cdf(high)
    cdf_x = (norm.cdf(y) - cdf_low) / (cdf_high - cdf_low)
    return norm.ppf(cdf_x)

def invhighbody(y, low=-jnp.inf, high=jnp.inf):
    cdf_nlow, cdf_nhigh = norm.cdf(-low), norm.cdf(-high) # cdf(-x) = 1-cdf(x), more stable
    cdf_nx = (cdf_nhigh - norm.cdf(-y)) / (cdf_nhigh - cdf_nlow)
    return - norm.ppf(cdf_nx)

def invbody(y, low=-jnp.inf, high=jnp.inf):
    condlist = [y < 0.]
    funclist = [invlowbody, invhighbody]
    return jnp.piecewise(y, condlist, funclist, low=low, high=high)   

def invhightail(y, low=None, high=jnp.inf):
    temp = 1/6.2842226/2 # best temperature at 12 sigma
    energy, b = jnp.split(jnp.stack(jnp.broadcast_arrays(y, high, 1, -1), axis=0), 2)
    return - temp * logsumexp( - energy / temp, axis=0, b=b)

def invlowtail(y, low=-jnp.inf, high=None):
    temp = 1/6.2842226/2 # best temperature at 12 sigma
    energy, b = jnp.split(jnp.stack(jnp.broadcast_arrays(-y, -low, 1, -1), axis=0), 2)
    return temp * logsumexp( - energy / temp, axis=0, b=b)

def trunc2std(y, loc=0., scale=1., low=-jnp.inf, high=jnp.inf):
    """
    Transport a general truncated normal variable to a standard normal variable.
    """
    y, low, high = (y - loc) / scale, (low - loc) / scale, (high - loc) / scale
    lim = 12 # switch to a more stable approx at 12 sigma, for float32
    condlist = [(y < -lim) & (low < -lim), (lim < y) & (lim < high)]
    funclist = [invlowtail, invhightail, invbody]
    return jnp.piecewise(y, condlist, funclist, low=low, high=high)


class DetruncTruncNorm(Distribution):
    """
    Detruncated Truncated Normal distribution.
    Detruncation is such that a truncated normal with fiducial parameters is transformed into a standard normal.

    This means `std2trunc(DetruncTruncNorm(loc, scale, low, high, loc_fid, scale_fid), loc_fid, scale_fid, low, high)` 
    is distributed as `TruncNorm(loc, scale, low, high)`.
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 
                       'low': constraints.real, 'high': constraints.real,
                       'loc_fid': constraints.real, 'scale_fid': constraints.positive, 
                       }
    support = constraints.real
    def __init__(self, loc=0., scale=1., 
                 low=-jnp.inf, high=jnp.inf, 
                 loc_fid:float=None, scale_fid:float=None, 
                 *, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
  
        self.loc_fid = loc if loc_fid is None else loc_fid
        self.scale_fid = scale if scale_fid is None else scale_fid

        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale), 
                                           jnp.shape(loc_fid), jnp.shape(scale_fid),
                                           jnp.shape(low), jnp.shape(high),)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        trunc = TruncatedNormal(self.loc, self.scale, low=self.low, high=self.high).sample(key, sample_shape)
        trunc, loc_fid, scale_fid, low, high = jnp.broadcast_arrays(trunc, self.loc_fid, self.scale_fid, self.low, self.high)
        return nvmap(trunc2std, trunc.ndim)(trunc, loc_fid, scale_fid, low, high)

    def _log_prob(self, value, loc, scale, low, high, loc_fid, scale_fid):
        fn = partial(std2trunc, loc=loc_fid, scale=scale_fid, low=low, high=high)
        log_abs_det_jac = lambda x: jnp.log(jnp.abs(grad(fn)(x)))
        # log_abs_det_jac = lambda x: analyt_log_abs_det_jac(x, self.loc_fid, self.scale_fid, self.low, self.high)
        log_pdf = TruncatedNormal(loc, scale, low=low, high=high).log_prob
        return log_pdf(fn(value)) + log_abs_det_jac(value)
    
    def log_prob(self, value):
        value, loc, scale, loc_fid, scale_fid, low, high = jnp.broadcast_arrays(value, self.loc, self.scale, 
                                                            self.loc_fid, self.scale_fid, self.low, self.high)
        return nvmap(self._log_prob, value.ndim)(value, loc, scale, low, high, loc_fid, scale_fid)

class DetruncUnif(Distribution):
    """
    Detruncated Uniform distribution.
    Detruncation is such that a truncated normal with fiducial parameters is transformed into a standard normal.

    This means `std2trunc(DetruncUnif(low, high, loc_fid, scale_fid), loc_fid, scale_fid, low, high)` 
    is distributed as `Unif(low, high)`.
    """
    arg_constraints = {'low': constraints.real, 'high': constraints.real,
                       'loc_fid': constraints.real, 'scale_fid': constraints.positive, 
                       }
    support = constraints.real
    def __init__(self, low=0., high=1., 
                 loc_fid:float=None, scale_fid:float=None, 
                 *, validate_args=None):
        self.low = low
        self.high = high
        
        self.loc_fid = (high + low) / 2 if loc_fid is None else loc_fid
        self.scale_fid = (high - low) / 12**.5 if scale_fid is None else scale_fid

        batch_shape = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high),
                                           jnp.shape(loc_fid), jnp.shape(scale_fid),)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        trunc = Uniform(self.low, self.high).sample(key, sample_shape)
        trunc, loc_fid, scale_fid, low, high = jnp.broadcast_arrays(trunc, self.loc_fid, self.scale_fid, self.low, self.high)
        return nvmap(trunc2std, trunc.ndim)(trunc, loc_fid, scale_fid, low, high)

    def _log_prob(self, value, low, high, loc_fid, scale_fid):
        fn = partial(std2trunc, loc=loc_fid, scale=scale_fid, low=low, high=high)
        log_abs_det_jac = lambda x: jnp.log(jnp.abs(grad(fn)(x)))
        # log_abs_det_jac = lambda x: analyt_log_abs_det_jac(x, self.loc_fid, self.scale_fid, self.low, self.high)
        log_pdf = Uniform(low, high).log_prob
        return log_pdf(fn(value)) + log_abs_det_jac(value)

    def log_prob(self, value):
        value, low, high, loc_fid, scale_fid = jnp.broadcast_arrays(value, self.low, self.high, self.loc_fid, self.scale_fid)
        return nvmap(self._log_prob, value.ndim)(value, low, high, loc_fid, scale_fid)

def analyt_log_abs_det_jac(x, loc, scale, low, high):
    # NOTE: this analytical logabsdetjac for std2trunc fails after 12sigma for float32
    low, high = (low - loc) / scale, (high - loc) / scale
    cdf_low, cdf_high = norm.cdf(low), norm.cdf(high)
    return jnp.log(scale * (cdf_high - cdf_low) * norm.pdf(x) / norm.pdf(std2trunc(x, 0., 1., low, high)))
    # cdf_y = cdf_low + (cdf_high - cdf_low) * norm.cdf(x)
    # return jnp.log(scale * (cdf_high - cdf_low) * norm.pdf(x) / norm.pdf(norm.ppf(cdf_y)))



def _log1mexp(x):
    """Numerically stable log(1 - exp(x)) for x <= 0."""
    return jnp.where(x > -jnp.log(2.0),
                     jnp.log(-jnp.expm1(x)),
                     jnp.log1p(-jnp.exp(x)))


def _log_diff_cdf(hi, lo):
    """log(Phi(hi) - Phi(lo)) for hi >= lo, evaluated on the more accurate tail."""
    use_upper = (hi + lo) > 0  # interval right of 0: work with survival function
    lower = norm.logcdf(hi) + _log1mexp(norm.logcdf(lo) - norm.logcdf(hi))
    upper = norm.logcdf(-lo) + _log1mexp(norm.logcdf(-hi) - norm.logcdf(-lo))
    return jnp.where(use_upper, upper, lower)


_SHASH_QUAD_DEG = 20
_shash_x, _shash_w = hermegauss(_SHASH_QUAD_DEG)
_shash_x = jnp.asarray(_shash_x)
_shash_w = jnp.asarray(_shash_w / np.sqrt(2 * np.pi))   # E_{N(0,1)}[f] = sum_i w_i f(x_i)
_shash_asinh_x = jnp.arcsinh(_shash_x)


class SinhArcsinh(Distribution):
    """Sinh-Arcsinh of Normal distribution, standardized so loc/scale ARE the mean/std.

    Raw transform (eps ~ N(0,1)):  Z = sinh((asinh(eps) + skewness) * tailweight),
    then x = mean + std * (Z - E[Z]) / sqrt(Var[Z]).  Hence E[x]=mean, SD[x]=std for
    every (skewness, tailweight); shape (skew/kurtosis) is orthogonal to mean/std,
    which removes the loc<->skew (and scale<->shape) sampling ridge of the raw form.
    tailweight>1 (resp. <1) -> heavier (resp. lighter) tails; skewness>0 -> right skew;
    skewness=0, tailweight=1 -> Normal(mean, std).
    E[Z], Var[Z] are computed by Gauss-Hermite quadrature (degree _SHASH_QUAD_DEG).
    """
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive,
                       "skewness": constraints.real, "tailweight": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale", "skewness", "tailweight"]

    def __init__(self, mean=0.0, std=1.0, skewness=0.0, tailweight=1.0, *, validate_args=None):
        batch_shape = lax.broadcast_shapes(jnp.shape(mean), jnp.shape(std),
                                           jnp.shape(skewness), jnp.shape(tailweight))
        # attribute names match arg_constraints keys (pytree fields); loc/scale = true mean/std
        self.loc, self.scale, self.skewness, self.tailweight = promote_shapes(
            mean, std, skewness, tailweight)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def _standardizer(self):
        # mean m and std s of the raw Z under eps ~ N(0,1), per-element via GH quadrature.
        # Computed lazily (not stored): numpyro tree_unflatten bypasses __init__.
        # Without standardizer, the local moment matching between 
        # QuadGauss(mean, scale1, scale2) and ShAsh(loc, scale, skew, tail) is
        # loc = mean - 4.8 * scale2
        # scale = scale1
        # skew = 3.5 * scale2 / scale1
        # tail = 1 + 5.9 * (scale2 / scale1)**2
        a = _shash_asinh_x.reshape((-1,) + (1,) * len(self.batch_shape))
        Z = jnp.sinh((a + self.skewness) * self.tailweight)        # (Q, *batch)
        m = jnp.tensordot(_shash_w, Z, axes=(0, 0))
        v = jnp.tensordot(_shash_w, Z ** 2, axes=(0, 0)) - m ** 2
        return m, jnp.sqrt(v)

    def sample(self, key, sample_shape=()):
        m, s = self._standardizer()
        eps = jr.normal(key, sample_shape + self.batch_shape + self.event_shape)
        Z = jnp.sinh((jnp.arcsinh(eps) + self.skewness) * self.tailweight)
        return self.loc + self.scale * (Z - m) / s

    def _to_normal(self, value):
        # Inverse of the monotone increasing transform -> standard normal variate eps.
        m, s = self._standardizer()
        Z = m + s * (value - self.loc) / self.scale
        eps = jnp.sinh(jnp.arcsinh(Z) / self.tailweight - self.skewness)
        return eps, Z, s

    @validate_sample
    def log_prob(self, value):
        eps, Z, s = self._to_normal(value)
        # log N(eps) + log|deps/dZ| + log|dZ/dvalue|, with the +log s - log scale Jacobian.
        return (-0.5 * jnp.log(2 * jnp.pi) - 0.5 * eps ** 2 + 0.5 * jnp.log1p(eps ** 2)
                - jnp.log(self.tailweight) - 0.5 * jnp.log1p(Z ** 2)
                + jnp.log(s) - jnp.log(self.scale))

    def cdf(self, value):
        return norm.cdf(self._to_normal(value)[0])

    def log_cdf(self, value):
        return norm.logcdf(self._to_normal(value)[0])

    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.batch_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(self.scale ** 2, self.batch_shape)









class QuadGaussian(Distribution):
    """Quadratic-in-Gaussian noise, mean-subtracted:
           obs = loc + scale1 * eps + scale2 * (eps**2 - 1),   eps ~ N(0,1)
       so E[obs] = loc and Var[obs] = scale1**2 + 2*scale2**2.
       Reduces to Normal(loc, scale1) as scale2 -> 0.
       Careful, has support bounded by loc - scale2 - scale1**2 / (4*scale2)
       (upper or lower bound depending on sign of scale2)"""
    arg_constraints = {"loc": constraints.real,
                       "scale1": constraints.positive,
                       "scale2": constraints.real}
    support = constraints.real
    reparametrized_params = ["loc", "scale1", "scale2"]

    def __init__(self, loc=0.0, scale1=1.0, scale2=0.0, *, validate_args=None):
        self.loc, self.scale1, self.scale2 = promote_shapes(loc, scale1, scale2)
        bs = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale1), jnp.shape(scale2))
        super().__init__(batch_shape=bs, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        eps = jr.normal(key, sample_shape + self.batch_shape)
        return self.loc + self.scale1 * eps + self.scale2 * (eps**2 - 1.0)

    @validate_sample
    def log_prob(self, value):
        a, b = self.scale2, self.scale1
        r = value - self.loc + a                       # a*eps^2 + b*eps = r
        D = b**2 + 4.0 * a * r                         # discriminant
        D_safe = jnp.where(D > 0, D, 1.0)
        sq = jnp.sqrt(D_safe)
        a_safe = jnp.where(jnp.abs(a) < 1e-12, 1.0, a)
        ep = (-b + sq) / (2.0 * a_safe)                # two Gaussian preimages
        em = (-b - sq) / (2.0 * a_safe)
        lp_quad = (-0.5*jnp.log(2*jnp.pi) - 0.5*jnp.log(D_safe)
                   + logsumexp(jnp.stack([-0.5*ep**2, -0.5*em**2], 0), axis=0))
        lp_quad = jnp.where(D > 0, lp_quad, -jnp.inf)  # outside support
        lp_gauss = -0.5*jnp.log(2*jnp.pi) - jnp.log(b) - 0.5*((value-self.loc)/b)**2
        return jnp.where(jnp.abs(a) < 1e-8, lp_gauss, lp_quad)

    def log_cdf(self, value):
        a, b = self.scale2, self.scale1
        r = value - self.loc + a                        # a*eps^2 + b*eps <= r
        D = b**2 + 4.0 * a * r                          # discriminant
        D_safe = jnp.where(D > 0, D, 1.0)
        sq = jnp.sqrt(D_safe)
        a_safe = jnp.where(jnp.abs(a) < 1e-12, 1.0, a)
        ep = (-b + sq) / (2.0 * a_safe)                 # two Gaussian preimages
        em = (-b - sq) / (2.0 * a_safe)
        # a > 0: parabola opens up, region is the interval [em, ep] (empty if D<0).
        lc_pos = jnp.where(D > 0, _log_diff_cdf(ep, em), -jnp.inf)
        # a < 0: parabola opens down, region is (-inf, ep] U [em, +inf) (all if D<0).
        lc_neg = jnp.where(D > 0, jnp.logaddexp(norm.logcdf(ep), norm.logcdf(-em)), 0.0)
        lc_quad = jnp.where(a > 0, lc_pos, lc_neg)
        lc_gauss = norm.logcdf((value - self.loc) / b)  # scale2 -> 0 limit
        return jnp.where(jnp.abs(a) < 1e-8, lc_gauss, lc_quad)

    def cdf(self, value):
        return jnp.exp(self.log_cdf(value))

    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.batch_shape)
    
    @property
    def variance(self): 
        return jnp.broadcast_to(self.scale1**2 + 2*self.scale2**2, self.batch_shape)


class TwoQuadGaussian(Distribution):
    """Two-field quadratic-in-Gaussian noise, mean-subtracted:
 
           obs = loc + scale1 * eps1 + scale2 * (eps2**2 - 1),
           eps1, eps2 ~ N(0, 1)   INDEPENDENT.
 
    This is the "naive expansion" counterpart of QuadGaussian: the linear and
    quadratic responses are driven by two *independent* noise fields instead of
    a single shared one. Same first two moments as the single-field model,
 
        E[obs]   = loc
        Var[obs] = scale1**2 + 2 * scale2**2,
 
    but a different third moment (see note below), hence a genuinely different
    distribution.
 
    The density has no elementary closed form: marginalizing eps1 analytically
    leaves obs | eps2 ~ N(loc + scale2*(eps2**2 - 1), scale1), and the remaining
    1-D integral over eps2 is a Gaussian (x) shifted/scaled chi^2_1 convolution
    (a parabolic-cylinder / Bessel-K_{1/4} function). We evaluate that one
    integral by Gauss-Hermite quadrature against the N(0,1) weight, which is
    smooth, fast, and fully differentiable.
 
    Accuracy note: the integrand sharpens as scale2/scale1 grows. For a
    perturbative non-Gaussian noise term (scale2 <~ scale1/3) n_quad=32 already
    gives ~1e-5 relative error in the bulk; for scale2 ~ scale1 use n_quad ~ 96-128.
    Reduces exactly to Normal(loc, scale1) as scale2 -> 0 (no special-casing needed).
 
    Third-moment contrast (with the single-field QuadGaussian):
        single field : E[(obs-loc)^3] = 2*scale2*(3*scale1**2 + 4*scale2**2)
        two   fields : E[(obs-loc)^3] = 8*scale2**3
    The difference, 6*scale1**2*scale2, is exactly the linear-quadratic
    covariance that exists only when both share the same eps.
    """
 
    arg_constraints = {"loc": constraints.real,
                       "scale1": constraints.positive,
                       "scale2": constraints.real}
    support = constraints.real
    reparametrized_params = ["loc", "scale1", "scale2"]
 
    def __init__(self, loc=0.0, scale1=1.0, scale2=0.0, *, n_quad=64,
                 validate_args=None):
        self.loc, self.scale1, self.scale2 = promote_shapes(loc, scale1, scale2)
        bs = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale1),
                                  jnp.shape(scale2))
        # Gauss-Hermite nodes/weights for E_{N(0,1)}[ f ] ~ sum wn_i f(z_i):
        #   hermegauss gives  int f(x) e^{-x^2/2} dx ~ sum w_i f(x_i),  sum w = sqrt(2pi)
        z, w = hermegauss(n_quad)
        self._gh_z = jnp.asarray(z)                          # (n_quad,)
        self._gh_logw = jnp.asarray(np.log(w) - 0.5 * np.log(2 * np.pi))  # log(wn)
        self.n_quad = n_quad
        super().__init__(batch_shape=bs, validate_args=validate_args)
 
    def sample(self, key, sample_shape=()):
        k1, k2 = jr.split(key)
        shp = sample_shape + self.batch_shape
        eps1 = jr.normal(k1, shp)
        eps2 = jr.normal(k2, shp)
        return self.loc + self.scale1 * eps1 + self.scale2 * (eps2 ** 2 - 1.0)
 
    def _quad_axes(self, value):
        # broadcast helpers: put the quadrature axis as a new leading axis 0
        nd = jnp.ndim(value)
        zr = self._gh_z.reshape((-1,) + (1,) * nd)           # (n_quad, 1...,1)
        logwr = self._gh_logw.reshape((-1,) + (1,) * nd)
        mu = self.loc + self.scale2 * (zr ** 2 - 1.0)        # (n_quad, *batch)
        return zr, logwr, mu
 
    @validate_sample
    def log_prob(self, value):
        _, logwr, mu = self._quad_axes(value)
        v = value[None, ...]
        s1 = self.scale1
        comp = logwr + norm.logpdf(v, loc=mu, scale=s1)       # (n_quad, *shape)
        return logsumexp(comp, axis=0)
 
    def log_cdf(self, value):
        _, logwr, mu = self._quad_axes(value)
        v = value[None, ...]
        s1 = self.scale1
        comp = logwr + norm.logcdf((v - mu) / s1)
        return logsumexp(comp, axis=0)
 
    def cdf(self, value):
        return jnp.exp(self.log_cdf(value))
 
    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.batch_shape)
 
    @property
    def variance(self):
        return jnp.broadcast_to(self.scale1 ** 2 + 2 * self.scale2 ** 2,
                                self.batch_shape)
 



_B = np.sqrt(2.0 / np.pi)
# Maximum |skewness| attainable by a skew-normal (delta -> 1):
_GAMMA_MAX = ((4.0 - np.pi) / 2.0) * (2.0 / (np.pi - 2.0)) ** 1.5  # ~0.9952717
 
from numpy.polynomial.legendre import leggauss
class SkewNormal(Distribution):
    """Azzalini skew-normal in the *centered* parametrization (mean, std, skew).
 
        density(x) = (2/omega) * phi(z) * Phi(alpha * z),   z = (x - xi)/omega
 
    but parametrized directly by its moments so location/scale are decoupled
    from shape (no MCMC ridge, unlike the raw (xi, omega, alpha) form):
 
        mean      = E[X]                       (the 'mean' argument)
        std       = sqrt(Var[X])               (the 'std' argument)
        skewness  = E[(X-mean)^3]/std^3        (the 'skew' argument)
 
    The mapping (mean, std, skew) -> (xi, omega, alpha) is CLOSED FORM (below),
    so there is no Gauss-Hermite quadrature anywhere in log_prob -- the field
    likelihood costs ~2 special-function evals per cell. Full support on R.
 
    A skew-normal can only realize |skew| < _GAMMA_MAX ~= 0.9953; the 'skew'
    argument is clipped into [-max_skew, max_skew] (default just inside the
    bound). Beyond that the mean/std are still matched exactly, the skewness
    saturates. log_cdf uses Owen's T (a small fixed quadrature, only in the cdf,
    never in log_prob).
    """
 
    arg_constraints = {"mean": constraints.real,
                       "std": constraints.positive,
                       "skew": constraints.real}
    support = constraints.real
    reparametrized_params = ["mean", "std", "skew"]
 
    def __init__(self, mean=0.0, std=1.0, skew=0.0, *,
                 max_skew=_GAMMA_MAX * (1.0 - 1e-6), n_owen=48,
                 validate_args=None):
        self.mean_, self.std, self.skew = promote_shapes(mean, std, skew)
        bs = lax.broadcast_shapes(jnp.shape(mean), jnp.shape(std),
                                  jnp.shape(skew))
        self.max_skew = float(min(max_skew, _GAMMA_MAX * (1.0 - 1e-6)))
        # Gauss-Legendre nodes on [0,1] for Owen's T (cdf only)
        x, w = leggauss(n_owen)
        self._gl_t = jnp.asarray(0.5 * (x + 1.0))
        self._gl_w = jnp.asarray(0.5 * w)
        # cache the direct parameters
        self._xi, self._omega, self._alpha, self._delta, self._gamma = \
            self._cp_to_dp(self.mean_, self.std, self.skew)
        super().__init__(batch_shape=bs, validate_args=validate_args)
 
    def _cp_to_dp(self, mean, std, skew):
        g = jnp.clip(skew, -self.max_skew, self.max_skew)
        A = (2.0 * jnp.abs(g) / (4.0 - np.pi)) ** (2.0 / 3.0)
        muz = jnp.sign(g) * jnp.sqrt(A / (1.0 + A))         # standardized mean = b*delta
        muz = jnp.clip(muz, -_B * (1 - 1e-7), _B * (1 - 1e-7))
        delta = muz / _B
        delta2 = jnp.clip(delta ** 2, 0.0, 1.0 - 1e-12)
        alpha = delta / jnp.sqrt(1.0 - delta2)
        omega = std / jnp.sqrt(1.0 - muz ** 2)
        xi = mean - omega * muz
        return xi, omega, alpha, delta, g
 
    @validate_sample
    def log_prob(self, value):
        z = (value - self._xi) / self._omega
        return (np.log(2.0) - jnp.log(self._omega)
                + norm.logpdf(z) + norm.logcdf(self._alpha * z))
 
    def sample(self, key, sample_shape=()):
        k0, k1 = jr.split(key)
        shp = sample_shape + self.batch_shape
        z0 = jr.normal(k0, shp)
        z1 = jr.normal(k1, shp)
        d = self._delta
        return self._xi + self._omega * (d * jnp.abs(z0)
                                         + jnp.sqrt(1.0 - d ** 2) * z1)
 
    def _owens_t(self, h, a):
        # T(h,a) = (1/2pi) int_0^{atan|a|} exp(-0.5 h^2 sec^2 th) dth, odd in a
        aa = jnp.abs(a)
        upper = jnp.arctan(aa)
        th = upper[..., None] * self._gl_t            # (..., n)
        sec2 = 1.0 / jnp.cos(th) ** 2
        integrand = jnp.exp(-0.5 * (h[..., None] ** 2) * sec2)
        integral = upper * jnp.sum(self._gl_w * integrand, axis=-1)
        return jnp.sign(a) * integral / (2.0 * np.pi)
 
    def cdf(self, value):
        z = (value - self._xi) / self._omega
        alpha = jnp.broadcast_to(self._alpha, jnp.shape(z))
        cdf = norm.cdf(z) - 2.0 * self._owens_t(z, alpha)
        return jnp.clip(cdf, 0.0, 1.0)
 
    def log_cdf(self, value):
        return jnp.log(jnp.clip(self.cdf(value), 1e-300, 1.0))
 
    @property
    def mean(self):
        return jnp.broadcast_to(self.mean_, self.batch_shape)
 
    @property
    def variance(self):
        return jnp.broadcast_to(self.std ** 2, self.batch_shape)
 
    @property
    def skewness(self):
        return jnp.broadcast_to(
            jnp.clip(self.skew, -self.max_skew, self.max_skew),
            self.batch_shape)
 
 
def match_quadratic_gaussian(loc, scale1, scale2):
    """(mean, std, skew) for a SkewNormal matching the first three moments of
    loc + scale1*eps + scale2*(eps**2 - 1),  eps ~ N(0,1).
 
        mean = loc
        std  = sqrt(scale1**2 + 2*scale2**2)
        skew = 2*scale2*(3*scale1**2 + 4*scale2**2) / std**3   (clipped by SkewNormal)
    """
    var = scale1 ** 2 + 2.0 * scale2 ** 2
    m3 = 2.0 * scale2 * (3.0 * scale1 ** 2 + 4.0 * scale2 ** 2)
    return loc, jnp.sqrt(var), m3 / var ** 1.5




##############################
# Fourier (Memory efficient) #
##############################
def ch2rshape(shape):
    """
    Complex Hermitian shape to real shape.
    
    Assume last real shape is even to lift the ambiguity
    (same convention as `fft.rfftn`).
    """
    return (*shape[:-1], 2*(shape[-1]-1))

def r2chshape(shape):
    """
    Real shape to complex Hermitian shape.
    """
    return (*shape[:-1], shape[-1]//2+1)


def _rg2cgh(mesh, part="real", norm="backward"):
    """
    Return the real and imaginary parts of a complex Gaussian Hermitian tensor
    obtained by permuting and reweighting a real Gaussian tensor (3D).
    Handle the Hermitian symmetry, specifically at border faces, edges, and vertices.
    """
    shape = np.array(mesh.shape)
    assert np.all(shape % 2 == 0), "dimension lengths must be even."
    
    hx, hy, hz = shape // 2
    meshk = jnp.zeros(r2chshape(shape))

    if part == "imag":
        slix, sliy, sliz = slice(hx + 1, None), slice(hy + 1, None), slice(hz + 1, None)
    else:
        assert part == "real", "part must be either 'real' or 'imag'."
        slix, sliy, sliz = slice(1, hx), slice(1, hy), slice(1, hz)
    meshk = meshk.at[:, :, 1:-1].set(mesh[:, :, sliz])
        
    for k in [0, hz]:  # two faces
        meshk = meshk.at[:, 1:hy, k].set(mesh[:, sliy, k])
        meshk = meshk.at[1:, hy + 1:, k].set(mesh[1:, sliy, k][::-1, ::-1])
        meshk = meshk.at[0, hy + 1:, k].set(mesh[0, sliy, k][::-1])  # handle the border
        if part == "imag" and norm != "amp":
            meshk = meshk.at[:, hy + 1:, k].multiply(-1.)

        for j in [0, hy]:  # two edges per face
            meshk = meshk.at[1:hx, j, k].set(mesh[slix, j, k])
            meshk = meshk.at[hx + 1:, j, k].set(mesh[slix, j, k][::-1])
            if part == "imag" and norm != "amp":
                meshk = meshk.at[hx + 1:, j, k].multiply(-1.)

            for i in [0, hx]:  # two points per edge
                if part == "real":
                    meshk = meshk.at[i, j, k].set(mesh[i, j, k])
                    if norm != "amp":
                        meshk = meshk.at[i, j, k].multiply(2**.5)

    shape = shape.astype(float)
    if norm == "backward":
        meshk /= (2 / shape.prod())**.5
    elif norm == "ortho":
        meshk /= 2**.5
    elif norm == "forward":
        meshk /= (2 * shape.prod())**.5
    else:
        assert norm == "amp", "norm must be either 'backward', 'forward', 'ortho', or 'amp'."

    return meshk


def _cgh2rg(meshk, part="real", norm="backward"):
    """
    Return the "real" and "imaginary" partitions of a real Gaussian tensor (3D)
    obtained by permuting and reweighting the real or imaginary parts of a 
    complex Gaussian Hermitian tensor.
    Handle the Hermitian symmetry, specifically at border faces, edges, and vertices.
    """
    shape = np.array(ch2rshape(meshk.shape))
    assert np.all(shape % 2 == 0), "dimension lengths must be even."

    hx, hy, hz = shape // 2
    mesh = jnp.zeros(shape)

    if part == "imag":
        slix, sliy, sliz = slice(hx + 1, None), slice(hy + 1, None), slice(hz + 1, None)
    else:
        assert part == "real", "part must be either 'real' or 'imag'."
        slix, sliy, sliz = slice(1, hx), slice(1, hy), slice(1, hz)
    mesh = mesh.at[:, :, sliz].set(meshk[:, :, 1:-1])
        
    for k in [0, hz]:  # two faces
        mesh = mesh.at[:, sliy, k].set(meshk[:, 1:hy, k])
        mesh = mesh.at[1:, sliy, k].set(meshk[1:, hy + 1:, k][::-1, ::-1])
        mesh = mesh.at[0, sliy, k].set(meshk[0, hy + 1:, k][::-1])  # handle the border
        if part == "imag" and norm != "amp":
            mesh = mesh.at[:, sliy, k].multiply(-1.)

        for j in [0, hy]:  # two edges per face
            mesh = mesh.at[slix, j, k].set(meshk[1:hx, j, k])
            mesh = mesh.at[slix, j, k].set(meshk[hx + 1:, j, k][::-1])
            if part == "imag" and norm != "amp":
                mesh = mesh.at[slix, j, k].multiply(-1.)

            for i in [0, hx]:  # two points per edge
                if part == "real":
                    mesh = mesh.at[i, j, k].set(meshk[i, j, k])
                    if norm != "amp":
                        mesh = mesh.at[i, j, k].divide(2**.5)

    shape = shape.astype(float)
    if norm == "backward":
        mesh *= (2 / shape.prod())**.5
    elif norm == "ortho":
        mesh *= 2**.5
    elif norm == "forward":
        mesh *= (2 * shape.prod())**.5
    else:
        assert norm == "amp", "norm must be either 'backward', 'forward', 'ortho', or 'amp'."

    return mesh


def rg2cgh(mesh, norm="backward"):
    """
    Permute and reweight a real Gaussian tensor (3D) into a complex Gaussian Hermitian tensor.
    The output would therefore be distributed as the real Fourier transform of a Gaussian tensor.

    This means that by setting `mean, amp = cgh2rg(meank, norm), cgh2rg(ampk, "amp")`\\
    then `rg2cgh(mean + amp * N(0,I), norm)` is distributed as `meank + ampk * rfftn(N(0,I), norm)`

    In particular `rg2cgh(N(0,I), norm)` is distributed as `rfftn(N(0,I), norm)`
    """
    real = _rg2cgh(mesh, part="real", norm=norm)
    if norm == "amp":
        return real
    else:
        imag = _rg2cgh(mesh, part="imag", norm=norm)
        return real + 1j * imag


def cgh2rg(meshk, norm="backward"):
    """
    Permute and reweight a complex Gaussian Hermitian tensor into a real Gaussian tensor (3D).

    This means that by setting `mean, amp = cgh2rg(meank, norm), cgh2rg(ampk, "amp")`\\
    then `rg2cgh(mean + amp * N(0,I), norm)` is distributed as `meank + ampk * rfftn(N(0,I), norm)`

    In particular `rg2cgh(N(0,I), norm)` is distributed as `rfftn(N(0,I), norm)`
    """
    real = _cgh2rg(meshk.real, part="real", norm=norm)
    if norm == "amp":
        # Give same amplitude to wavevector real and imaginary part
        imag = _cgh2rg(meshk.real, part="imag", norm=norm)
    else:
        imag = _cgh2rg(meshk.imag, part="imag", norm=norm)
    return real + imag


def _chreshape(mesh, shape):
    """
    Naively reshape a complex Hermitian tensor.
    Carefull, does not preserve Hermitian symmetry nor power.
    """
    scale = np.divide(ch2rshape(shape), ch2rshape(mesh.shape)).prod()

    # Center wavevectors in mesh to truncate or pad
    for ax, s in enumerate(mesh.shape[:-1]):
        mesh = jnp.roll(mesh, s//2, ax)
    
    slices = ()
    for ax, (ms, s) in enumerate(zip(mesh.shape, shape)):
        trunc = max(ms - s, 0)
        if ax < len(shape) - 1:
            trunc //= 2
            slices += (slice(trunc, None if trunc==0 else -trunc),)
        else:
            slices += (slice(0, None if trunc==0 else -trunc),)
    mesh = mesh[slices]

    pad_width = ()
    for ax, (ms, s) in enumerate(zip(mesh.shape, shape)):
        pad = max(s - ms, 0)
        if ax < len(shape) - 1:
            pad //= 2
            pad_width += ((pad, pad),)
        else:
            pad_width += ((0, pad),)
    mesh = jnp.pad(mesh, pad_width=pad_width)

    # Decenter wavevectors in mesh after truncate or pad
    for ax, s in enumerate(mesh.shape[:-1]):
        mesh = jnp.roll(mesh, -s//2, ax)
    return mesh * scale



def hermitian_symmetric(arr):
    """
    Return the Hermitian symmetric of a tensor (of any dimension and shape).
    A tensor has Hermitian symmetry if it is equal to its Hermitian symmetric.
    """
    dim = len(arr.shape)
    ids = dim * (slice(None, None, -1),)
    arr = arr[ids].conj()
    for ax in range(dim):
        arr = jnp.roll(arr, shift=1, axis=ax)
    return arr


def chreshape(mesh, shape):
    """
    Reshape a complex Hermitian tensor,
    with truncating or padding that preserve the Hermitian symmetry and the mean, 
    hence the average power.
    """
    mesh = jnp.asarray(mesh)
    # NOTE: reverse axes order to start with last axis, 
    # for the Hermitian symmetric of its only Nyquist hyperplane needs to be constructed first.
    for ax, (ms, s) in reversed(list(enumerate(zip(mesh.shape, shape)))): 
        if s < ms: # truncate axis
            if ax < len(shape) - 1:
                # Aggregate the 2 Nyquist hyperplanes
                neg_ids = (slice(None),) * ax + (-s//2,)
                pos_ids = (slice(None),) * ax + (s//2,)
                mesh = mesh.at[neg_ids].set((mesh[pos_ids] + mesh[neg_ids]) / 2**.5)
            else:
                # Aggregate the Nyquist hyperplane with its constructed Hermitian symmetric
                pos_ids = (slice(None),) * ax + (s - 1,)
                nyq_plane = mesh[pos_ids]
                nyq_plane_sym = hermitian_symmetric(nyq_plane)
                mesh = mesh.at[pos_ids].set((nyq_plane + nyq_plane_sym) / 2**.5)

    out = _chreshape(mesh, shape)

    for ax, (ms, s) in enumerate(zip(mesh.shape, shape)):
        if s > ms: # pad axis
            if ax < len(shape) - 1:
                # Reweight and duplicate the Nyquist hyperplane
                neg_ids = (slice(None),) * ax + (-ms//2,)
                pos_ids = (slice(None),) * ax + (ms//2,)
                out = out.at[neg_ids].divide(2**.5)
                out = out.at[pos_ids].set(out[neg_ids])
            else:
                # Reweight the Nyquist hyperplane
                pos_ids = (slice(None),) * ax + (ms - 1,)
                out = out.at[pos_ids].divide(2**.5)

    return out





##################################
# Fourier utils (Time efficient) #
##################################
def id_cgh(shape, part="real", norm="backward"):
    """
    Return indices and weights to permute a real Gaussian tensor of given shape (3D)
    into a complex Gaussian Hermitian tensor. 
    Handle the Hermitian symmetry, specificaly at border faces, edges, and vertices.
    """
    shape = np.asarray(shape)
    sx, sy, sz = shape
    assert sx%2 == sy%2 == sz%2 == 0, "dimension lengths must be even."
    
    hx, hy, hz = shape//2
    chshape = (sx, sy, hz+1)
    
    weights = np.ones(chshape)
    if norm == "backward":
        weights /= (2 / shape.prod())**.5
    elif norm == "ortho":
        weights /= 2**.5
    elif norm == "forward":
        weights /= (2 * shape.prod())**.5
    else:
        assert norm == "amp", "norm must be either 'backward', 'forward', 'ortho', or 'amp'."

    dtype = 'int16' # int16 -> +/- 32_767, trkl
    id = np.zeros((3, *chshape), dtype=dtype)
    xyz = np.indices(shape, dtype=dtype)

    if part == "imag":
        slix, sliy, sliz = slice(hx+1, None), slice(hy+1, None), slice(hz+1, None)
    else:
        assert part == "real", "part must be either 'real' or 'imag'."
        slix, sliy, sliz = slice(1,hx), slice(1,hy), slice(1,hz)
    id[...,1:-1] = xyz[...,sliz]
        
    for k in [0,hz]: # two faces
        id[...,1:hy,k] = xyz[...,sliy,k]
        id[...,1:,hy+1:,k] = xyz[...,1:,sliy,k][...,::-1,::-1]
        id[...,0,hy+1:,k] = xyz[...,0,sliy,k][...,::-1] # handle the border
        if part == "imag":
            weights[:,hy+1:,k] *= -1

        for j in [0,hy]: # two edges per faces
            id[...,1:hx,j,k] = xyz[...,slix,j,k]
            id[...,hx+1:,j,k] = xyz[...,slix,j,k][...,::-1]
            if part == "imag":
                weights[hx+1:,j,k] *= -1

            for i in [0,hx]: # two points per edges
                id[...,i,j,k] = xyz[...,i,j,k]
                if part == "imag":
                    weights[i,j,k] *= 0.
                else:
                    weights[i,j,k] *= 2**.5
    
    return tuple(id), weights



def rg2cgh2(mesh, norm="backward"):
    """
    Permute a real Gaussian tensor (3D) into a complex Gaussian Hermitian tensor.
    particular `rg2cgh(N(0,I), norm)` is distributed as `rfftn(N(0,I), norm)`
    """
    shape = mesh.shape
    id_real, w_real = id_cgh(shape, part="real", norm=norm)
    id_imag, w_imag = id_cgh(shape, part="imag", norm=norm)
    
    if norm == "amp":
        # Average wavevector real and imaginary power and return amplitude
        return ((mesh[id_real]**2 + mesh[id_imag]**2) / 2)**.5
    else:
        return mesh[id_real] * w_real + 1j * mesh[id_imag] * w_imag


def cgh2rg2(meshk, norm="backward"):
    """
    Permute a complex Gaussian Hermitian tensor into a real Gaussian tensor (3D).
    In particular `rg2cgh(N(0,I), norm)` is distributed as `rfftn(N(0,I), norm)`
    """
    shape = ch2rshape(meshk.shape)
    id_real, w_real = id_cgh(shape, part="real", norm=norm)
    id_imag, w_imag = id_cgh(shape, part="imag", norm=norm)
    
    mesh = jnp.zeros(shape)
    if norm == "amp":
        # Give same amplitude to wavevector real and imaginary part
        mesh = mesh.at[id_imag].set(meshk.real)
        mesh = mesh.at[id_real].set(meshk.real)
    else:
        # NOTE: w_imag can be zero, which is not safe for gradients
        mesh = mesh.at[id_imag].set(safe_div(meshk.imag, w_imag)) 
        mesh = mesh.at[id_real].set(meshk.real / w_real)
        # NOTE: real after imag to overwrite the 2^3=8 points
    return mesh


def _chreshape2(mesh, shape):
    """
    Naively reshape a complex Hermitian tensor.
    Carefull, does not preserve Hermitian symmetry nor power.
    """
    ids_shape = tuple(np.minimum(mesh.shape, shape))
    scale = np.divide(ch2rshape(shape), ch2rshape(mesh.shape)).prod()

    dtype = 'int16' # int16 -> +/- 32_767, trkl
    ids = tuple(np.roll(np.arange(-(s//2), (s+1)//2, dtype=dtype), -(s//2)) for s in ids_shape[:-1])
    ids += (np.arange(ids_shape[-1], dtype=dtype),)
    
    if ids_shape == shape: # downsample all axis
        out = mesh[np.ix_(*ids)]
    elif ids_shape == mesh.shape: # oversample all axis
        out = jnp.zeros(shape, dtype=complex)
        out = out.at[np.ix_(*ids)].set(mesh)
    else: # down or oversample
        out = jnp.zeros(shape, dtype=complex)
        ids = np.ix_(*ids)
        out = out.at[ids].set(mesh[ids])
    return out * scale



############
# Geometry #
############
def boxreshape(mesh, shape):
    """
    Reshape a tensor, with padding or truncating centered on each axis.
    """
    shape = np.array(shape)
    mesh_shape = np.array(mesh.shape)
    assert np.all(shape % 2 == 0) and np.all(mesh_shape % 2 == 0), "dimension lengths must be even."

    half_down = np.maximum(mesh_shape - shape, 0) // 2
    slices = tuple(slice(hd, None if hd==0 else -hd) for hd in half_down)
    mesh = mesh[slices]

    mesh_shape = np.array(mesh.shape)
    half_over = np.maximum(shape - mesh_shape, 0) // 2
    mesh = jnp.pad(mesh, pad_width=tuple((ho,ho) for ho in half_over))
    return mesh

def scale_shape(shape, scale=1.):
    """
    Return a valid scaled shape from a given mesh shape and a 1D scaling factor.
    """
    shape = np.asarray(shape)
    return 2 * np.rint(shape * scale / 2).astype(int)
    

def mesh2masked(mesh, mask=None):
    if mask is None:
        return mesh
    else:
        return mesh[...,mask]


def masked2mesh(masked, mask=None):
    if mask is None:
        return masked
    else:
        shape = jnp.shape(masked)[:-1] + jnp.shape(mask)
        return jnp.zeros(shape).at[...,mask].set(masked)


def radecrad2cart(ra, dec, radius):
    """
    Convert ra, dec (in degrees), and radius to cartesian coordinates.
    """
    ra = jnp.deg2rad(ra)
    dec = jnp.deg2rad(dec)
    x = jnp.cos(dec) * jnp.cos(ra)
    y = jnp.cos(dec) * jnp.sin(ra)
    z = jnp.sin(dec)
    cart = jnp.moveaxis(radius * jnp.stack((x, y, z)), 0, -1)
    return cart


def cart2radecrad(cart:jnp.ndarray):
    """
    Convert cartesian coordinates to ra, dec (in degrees), and radius.
    * ra \\in [0, 360]
    * dec \\in [-90, 90]
    * radius \\in [0, +\\infty[
    """
    radius = jnp.linalg.norm(cart, axis=-1)
    x, y, z = jnp.moveaxis(cart, -1, 0)
    ra = jnp.rad2deg(jnp.arctan2(y, x)) % 360.
    dec = jnp.rad2deg(jnp.arcsin(safe_div(z, radius)))
    return ra, dec, radius



def surface_hypersphere(d, R=1):
    """d is the embedding dimension"""
    log_surf = np.log(2) + d/2 * np.log(np.pi) + (d-1) * np.log(R) - gammaln(d/2)
    return np.exp(log_surf)

def volume_hypersphere(d, R=1):
    """d is the embedding dimension"""
    log_vol = d/2 * np.log(np.pi) + d * np.log(R) - gammaln(d/2 + 1)
    return np.exp(log_vol)



# def get_noise_fn(t0, t1, noises, steps=False):
#     """
#     Given a noises list, starting and ending times, 
#     return a function that interpolate these noises between these times,
#     by steps or linearly.
#     """
#     n_noises = len(noises)-1
#     if steps:
#         def noise_fn(t):
#             i_t = n_noises*(t-t0)/(t1-t0)
#             i_t1 = jnp.floor(i_t).astype(int)
#             return noises[i_t1]
#     else:
#         def noise_fn(t):
#             i_t = n_noises*(t-t0)/(t1-t0)
#             i_t1 = jnp.floor(i_t).astype(int)
#             s1 = noises[i_t1]
#             s2 = noises[i_t1+1]
#             return (s2 - s1)*(i_t - i_t1) + s1
#     return noise_fn



