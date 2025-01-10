from __future__ import annotations # for Union typing | in python<3.10

from pickle import dump, load, HIGHEST_PROTOCOL
from functools import wraps

import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, grad, tree

from jax.scipy.special import logsumexp
from jax.scipy.stats import norm




def pdump(obj, path):
    with open(path, 'wb') as file:
        dump(obj, file, protocol=HIGHEST_PROTOCOL)

def pload(path):
    with open(path, 'rb') as file:
        return load(file)    



def get_vlim(level=1., scale=1., axis=0):
    """
    Return function computing robust inferior and superior limit values of an array, 
    i.e. discard quantiles bilateraly on some level, and scale the margins.
    """
    def vlim(a):
        """
        Return robust inferior and superior limit values of an array.
        """
        vmin, vmax = jnp.quantile(a, (1-level)/2, axis=axis), jnp.quantile(a, (1+level)/2, axis=axis)
        vmean, vdiff = (vmax+vmin)/2, scale*(vmax-vmin)/2
        return jnp.stack((vmean-vdiff, vmean+vdiff), axis=-1)
    return vlim


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


def nvmap(fun, n):
    """
    Nest vmap n times.
    """
    for _ in range(n):
        fun = vmap(fun)
    return fun






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
    scale_nonzero = jnp.where(scale==0, 1, scale)
    lowhigh = (jnp.stack([low, high]) - loc) / scale_nonzero
    low, high = jnp.where(scale==0, jnp.stack([-jnp.inf, jnp.inf]), lowhigh)
    lim = 12 # switch to a more stable approx at 12 sigma
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
    lim = 12 # switch to a more stable approx at 12 sigma
    condlist = [(y < -lim) & (low < -lim), (lim < y) & (lim < high)]
    funclist = [invlowtail, invhightail, invbody]
    return jnp.piecewise(y, condlist, funclist, low=low, high=high)




def id_cgh(shape, part="real", norm="backward"):
    """
    Return indices and weights to permute a real Gaussian tensor of shape ``mesh_shape`` (3D)
    into a complex Gaussian Hermitian tensor. 
    Handle the Hermitian symmetry, specificaly at border faces, edges, and vertices.
    """
    shape = np.asarray(shape)
    sx, sy, sz = shape
    assert sx%2 == sy%2 == sz%2 == 0, "dimensions lengths must be even."
    
    hx, hy, hz = shape//2
    kshape = (sx, sy, hz+1)
    
    weights = np.ones(kshape) / 2**.5
    if norm == "backward":
        weights *= shape.prod()**.5 
    elif norm == "forward":
        weights /= shape.prod()**.5
    else:
        assert norm=="ortho", "norm must be either 'backward', 'forward', or 'ortho'."

    id = np.zeros((3,*kshape), dtype=int)
    xyz = np.indices(shape)

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
                    weights[i,j,k] *= 0
                else:
                    weights[i,j,k] *= 2**.5
    
    return tuple(id), weights



def rg2cgh(mesh, amp:bool=False, norm="backward"):
    """
    Permute a real Gaussian tensor (3D) into a complex Gaussian Hermitian tensor.
    The output would therefore be distributed as the real Fourier transform of a Gaussian tensor.

    This means that by setting `mean, amp = cgh2rg(meank, norm), cgh2rg(ampk, True, norm)`\\
    then `rg2cgh(mean + amp * N(0,I), norm)` is distributed as `meank + ampk * rfftn(N(0,I), norm)`

    In particular `rg2cgh(N(0,I), norm)` is distributed as `rfftn(N(0,I), norm)`
    """
    shape = mesh.shape
    id_real, w_real = id_cgh(shape, part="real", norm=norm)
    id_imag, w_imag = id_cgh(shape, part="imag", norm=norm)
    if not amp:
        return mesh[id_real] * w_real + 1j * mesh[id_imag] * w_imag
    else:
        # Average wavevector real and imaginary power and return amplitude
        return ((mesh[id_real]**2 + mesh[id_imag]**2) / 2)**.5



def cgh2rg(meshk, amp:bool=False, norm="backward"):
    """
    Permute a complex Gaussian Hermitian tensor into a real Gaussian tensor (3D).

    This means that by setting `mean, amp = cgh2rg(meank, norm), cgh2rg(ampk, True, norm)`\\
    then `rg2cgh(mean + amp * N(0,I), norm)` is distributed as `meank + ampk * rfftn(N(0,I), norm)`

    In particular `rg2cgh(N(0,I), norm)` is distributed as `rfftn(N(0,I), norm)`
    """
    shape = ch2rshape(meshk.shape)
    id_real, w_real = id_cgh(shape, part="real", norm=norm)
    id_imag, w_imag = id_cgh(shape, part="imag", norm=norm)
    
    mesh = jnp.zeros(shape)
    if not amp:
        mesh = mesh.at[id_imag].set(meshk.imag / w_imag)
        mesh = mesh.at[id_real].set(meshk.real / w_real)
        # NOTE: real after imag to get rid of infs
    else:
        # Give same amplitude to wavevector real and imaginary part
        mesh = mesh.at[id_imag].set(meshk.real)
        mesh = mesh.at[id_real].set(meshk.real)
    return mesh


def ch2rshape(kshape):
    return (*kshape[:2], 2*(kshape[2]-1))

def r2chshape(shape):
    return (*shape[:2], shape[2]//2+1)








# def thin_array(a, thinning=None, moment:int|list=None, axis=0):
#     """
#     If moment is array-like, moment dimension is added as a last dimension.
#     If thinning is None, return last values.
#     """
#     a = jnp.moveaxis(a, axis, -1)
#     shape = a.shape
#     if thinning is None:
#         thinning = shape[-1]
#     n_split = max(np.rint(shape[-1]/thinning), 1)
#     a = jnp.array_split(a, n_split, axis=-1)

#     if moment is None:
#         fn = lambda x: x[...,-1]
#     else:
#         if isinstance(moment, int):
#             fn = lambda x: jnp.sum(x**moment, axis=-1)
#         else:
#             moment = jnp.asarray(moment)
#             fn = lambda x: jnp.sum(x[...,None]**moment, axis=-2)

#     a = tree.map(fn, a)
#     a = jnp.stack(a, axis=-1)
#     return jnp.moveaxis(a, -1, axis)


# def cumfn_array(a, fn, n, *args, axis=0):
#     """
#     Compute function on cumulative slices along given axis, with results along the first dimension.
#     """
#     filt_ends = jnp.rint(jnp.arange(1,n+1) / n * a.shape[axis]).astype(int)
#     filt_fn = lambda end: fn(a[axis*(slice(None),) + (slice(None,end),)], *args)
#     out = ()
#     for end in filt_ends:
#         out += (filt_fn(end),)
#     return jnp.stack(out) # stack on first dim since fn can destroy some dims







# def choice_array(rng_key, a, n, axis):
#     """
#     Chose n random coordinates from last axis, obtained by flatenning given axes. 
#     Ensure reproducibilty independently of ungiven axes.
#     """
#     # Move given axes at the end
#     axis = jnp.atleast_1d(axis)
#     dest = -1-jnp.arange(len(axis)) 
#     a = jnp.moveaxis(a, axis, dest)

#     # Remove axes to flatten from shape
#     shape = list(a.shape)
#     for ax in axis:
#         shape.pop(ax) 

#     return jr.choice(rng_key, a.reshape((*shape,-1)), shape=(n,), replace=False, axis=-1)


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

