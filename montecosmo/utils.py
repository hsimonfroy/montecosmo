from __future__ import annotations # for Union typing | in python<3.10

import pickle
import yaml

from functools import partial, wraps

import numpy as np
from jax import jit, numpy as jnp, vmap, grad, tree, lax

from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from numpyro.distributions import Distribution, constraints, TruncatedNormal, Uniform, util



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
    """Pickle dump"""
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


##############################
# Fourier (Memory efficient) #
##############################
def ch2rshape(shape):
    """
    Complex Hermitian shape to real shape.
    
    Assume last real shape is even to lift the ambiguity.
    (same convention as `fft.rfftn`)
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


def chreshape(mesh, shape):
    """
    Reshape a complex Hermitian tensor,
    handling correctly zero-padding.
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


def chreshape2(mesh, shape):
    """
    Reshape a complex Hermitian tensor,
    handling correctly zero-padding.
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

