from __future__ import annotations # for Union typing | in python<3.10

import os
from pickle import dump, load, HIGHEST_PROTOCOL

import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, grad
from jax.tree_util import tree_map
from functools import wraps, partial

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import to_rgba_array

from numpyro.infer import MCMC
from numpyro.diagnostics import print_summary
from getdist.gaussian_mixtures import GaussianND
from getdist import MCSamples
from typing import Iterable, Callable

from jax.scipy.special import logsumexp
from jax.scipy.stats import norm





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


def pickle_dump(obj, path):
    with open(path, 'wb') as file:
        dump(obj, file, protocol=HIGHEST_PROTOCOL)


def pickle_load(path):
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


def color_switch(color, reverse=False):
    """
    Select between color an its negative, or colormap and its reversed.
    Typically used to switch between light theme and dark theme. 

    `color` must be Matpotlib color, or array of colors, or colormap.

    No need to switch the default color cycle `f'C{i}'`, Matplotlib handles it already.
    """
    try:
        color = to_rgba_array(color)
    except:
        if isinstance(color, str): # handle cmap
            if reverse:
                if color.endswith('_r'): # in case provided cmap is alreday reversed
                    return color[:-2]
                else:
                    return color+'_r'# reverse cmap
            else:
                return color
        else:
            raise TypeError("`color` must be Matpotlib color, or array of colors, or colormap.")

    if reverse:
        color[...,:-1] = 1-color[...,:-1] # take color negative, does not affect alpha
    return color


def set_plotting_options(usetex=False, font_size=10):
    params = {'text.usetex': usetex,
            #   'ps.useafm': True,
            #   'pdf.use14corefonts': True,
              'font.family': 'roman' if usetex else None,
              'font.size':font_size,} # NOTE: 'ps.useafm' and 'pdf.use14corefonts' for PS and PDF font comptatibiliies
    plt.rcParams.update(params)
    # import matplotlib as mpl
    # mpl.rcParams.update(mpl.rcParamsDefault)


def theme_switch(dark_theme=False, usetex=False, font_size=10):
    """
    Set Matplotlib theme and return an adequate color switching function.
    """
    if dark_theme: 
        plt.style.use('dark_background')
    else: 
        plt.style.use('default')
    rc('animation', html='html5') # handle Matplotlib animations
    set_plotting_options(usetex, font_size)
    theme = partial(color_switch, reverse=dark_theme)
    return theme





def save_run(mcmc:MCMC, i_run:int, save_path:str, var_names:list=None, 
             extra_fields:list=[], group_by_chain:bool=True):
    """
    Save one run of MCMC sampling, with extra fields and last state.
    If `var_names` is None, save all the variables.
    """
    # Save samples (and extra fields)
    samples = mcmc.get_samples(group_by_chain)
    if var_names is not None:
        samples = {key: samples[key] for key in var_names}

    if extra_fields:
        extra = mcmc.get_extra_fields(group_by_chain)
        if "num_steps" in extra.keys(): # renaming num_steps into clearer n_evals
            n_evals = extra.pop("num_steps")
            samples.update(n_evals=n_evals)
        samples.update(extra)
        del extra

    pickle_dump(samples, save_path+f"_{i_run}.p")
    del samples

    # Save or overwrite last state
    pickle_dump(mcmc.last_state, save_path+f"_laststate.p") 


def sample_and_save(mcmc:MCMC, n_runs:int, save_path:str, var_names:list=None, 
                    extra_fields:list=[], rng_key=jr.key(0), group_by_chain:bool=True, init_params=None) -> MCMC:
    """
    Warmup and run MCMC, saving the specified variables and extra fields.
    Do `mcmc.num_warmup` warmup steps, followed by `n_runs` times `mcmc.num_samples` sampling steps.
    If `var_names` is None, save all the variables.
    """
    # Warmup sampling
    if mcmc.num_warmup>=1:
        print(f"run {0}/{n_runs} (warmup)")

        # Warmup
        mcmc.warmup(rng_key, collect_warmup=True, extra_fields=extra_fields, init_params=init_params)
        save_run(mcmc, 0, save_path, var_names, extra_fields, group_by_chain)

        # Handling rng key and destroy init_params
        key_run = mcmc.post_warmup_state.rng_key
        init_params = None
    else:
        key_run = rng_key

    # Run sampling
    for i_run in range(1, n_runs+1):
        print(f"run {i_run}/{n_runs}")
            
        # Run
        mcmc.run(key_run, extra_fields=extra_fields, init_params=init_params)
        save_run(mcmc, i_run, save_path, var_names, extra_fields)

        # Init next run at last state
        mcmc.post_warmup_state = mcmc.last_state
        key_run = mcmc.post_warmup_state.rng_key
    return mcmc


def _load_runs(load_path:str, start_run:int, end_run:int, 
               var_names:Iterable[str]=None, conc_axis:int|Iterable[int]=0, 
               transform:Callable|Iterable[Callable]=lambda x:x, verbose=False):
    if verbose:
        print(f"loading: {os.path.basename(load_path)}, from run {start_run} to run {end_run} (included)")
    transform = np.atleast_1d(transform)

    for i_run in range(start_run, end_run+1):
        # Load
        samples_part = pickle_load(load_path+f"_{i_run}.p")   
        if var_names is None: # NOTE: var_names should not be a consumable iterator
            var_names = list(samples_part.keys())
        samples_part = {key: samples_part[key] for key in var_names}
        for trans in transform:
            samples_part = trans(samples_part)

        # Init or append samples
        if i_run == start_run:
            samples = tree_map(lambda x: x[None], samples_part)
        else:
            # samples = {key: jnp.concatenate((samples[key], samples_part[key])) for key in var_names}
            samples = tree_map(lambda x,y: jnp.concatenate((x, y[None]), axis=0), samples, samples_part)
            del samples_part  
        
    for axis in jnp.atleast_1d(conc_axis):
        samples = tree_map(lambda x: jnp.concatenate(x, axis=axis), samples)
            
    if verbose:
        if 'n_evals' in samples.keys():
            n_samples, n_evals = samples['n_evals'].shape, samples['n_evals'].sum(axis=-1)
            print(f"total n_samples: {n_samples}, total n_evals: {n_evals}")
        else:
            print(f"first variable length: {len(samples[list(samples.keys())[0]])}")
        print("")
    return samples


def load_runs(load_path:str|Iterable[str], start_run:int|Iterable[int], end_run:int|Iterable[int], 
              var_names:Iterable[str]=None, conc_axis:int|Iterable[int]=0, transform:Callable=lambda x:x, verbose=False):
    """
    Load and append runs (or extra fields) saved in different files with same name except index.

    Both runs `start_run` and `end_run` are included.
    If `var_names` is None, load all the variables.
    """
    paths = np.atleast_1d(load_path)
    starts = np.atleast_1d(start_run)
    ends = np.atleast_1d(end_run)
    assert len(paths)==len(starts)==len(ends), "lists must have the same lengths."
    samples = []

    for path, start, end in zip(paths, starts, ends):
        samples.append(_load_runs(path, start, end, var_names, conc_axis, transform, verbose))

    if isinstance(load_path, str):
        return samples[0]
    else:
    # if paths is load_path:
        return samples 
    



# def get_gdprior(samples:dict, prior_config:dict, label:str="Prior",
#                    verbose:bool=False, **config):
#     """
#     Construct getdist MCSamples from prior config.

#     Only uses keys from samples.
#     """
#     names = list(samples.keys())
#     labels = []
#     means, stds = [], []
#     for name in samples:
#         if name.endswith('_'): # convention for a latent param 
#             lab = "\\overline"+prior_config[name[:-1]][0]
#             mean, std = 0, 1
#         else:
#             lab, mean, std = prior_config[name]
#         labels.append(lab)
#         means.append(mean)
#         stds.append(std)

#     means, stds = np.array(means), np.array(stds)
#     gdsamples = GaussianND(means, np.diag(stds**2), names=names, labels=labels, label=label)

#     if verbose:
#         if label is not None:
#             print('# '+label)
#         else:
#             print("# <unspecified label>")
#         print("GaussianND object has no samples.\n")
#     return gdsamples


def _get_gdsamples(samples:dict, prior_config:dict, label:str=None,
                   verbose:bool=False, **config):
    labels = []
    for name in samples:
        if name.endswith('_'): # convention for a standardized latent param 
            lab = "\\overline"+prior_config[name[:-1]][0]
        else:
            lab = prior_config[name][0]
        labels.append(lab)

    gdsamples = MCSamples(samples=list(samples.values()), names=list(samples.keys()), labels=labels, label=label)

    if verbose:
        if label is not None:
            print('# '+gdsamples.getLabel())
        else:
            print("# <unspecified label>")
        print(gdsamples.getNumSampleSummaryText())
        print_summary(samples, group_by_chain=False) # NOTE: group_by_chain if several chains

    return gdsamples


def get_gdsamples(samples:dict|Iterable[dict], prior_config:dict, label:str|Iterable[str]=None, 
                  verbose:bool=False, **config):
    """
    Construct getdist MCSamples from samples. 
    """
    params = np.atleast_1d(samples)
    label = np.atleast_1d(label)
    assert len(params)==len(label), "lists must have the same lengths."
    gdsamples = []

    for par, lab in zip(params, label):
        gdsamples.append(_get_gdsamples(par, prior_config, lab, verbose))

    if isinstance(samples, dict):
        return gdsamples[0]
    else:
    # if params is samples:
        return gdsamples 
    






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
    lowhigh = (jnp.stack((low, high)) - loc) / scale_nonzero
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




# def id_cgh(mesh_size, part="real"):
#     """
#     Return indices and weights to permute a real Gaussian tensor of size ``mesh_size`` (3D)
#     into a complex Gaussian Hermitian tensor. 
#     Handle the Hermitian symmetry, specificaly at border faces, edges, and vertices.
#     """
#     mesh_size = np.array(mesh_size)
#     sx, sy, sz = mesh_size
#     # assert sx%2 == sy%2 == sz%2 == 0, "dimensions lengths must be even."
#     hx, hy, hz = mesh_size//2
#     kmesh_size = (sx, sy, hz+1)
#     weights = jnp.ones(kmesh_size) * (mesh_size.prod() / 2)**.5
#     id = jnp.zeros((3,*kmesh_size), dtype=int)
#     xyz = jnp.indices(mesh_size)

#     if part == "imag":
#         slix, sliy, sliz = slice(hx+1, None), slice(hy+1, None), slice(hz+1, None)
#     else:
#         slix, sliy, sliz = slice(1,hx), slice(1,hy), slice(1,hz)
#     id = id.at[...,1:-1].set( xyz[...,sliz] )
        
#     for k in [0,hz]: # two faces
#         id = id.at[...,1:hy,k].set(xyz[...,sliy,k])
#         id = id.at[...,1:,hy+1:,k].set(xyz[...,1:,sliy,k][...,::-1,::-1])
#         id = id.at[...,0,hy+1:,k].set(xyz[...,0,sliy,k][...,::-1]) # handle the border
#         if part == "imag":
#             weights = weights.at[:,hy+1:,k].multiply(-1)

#         for j in [0,hy]: # two edges per faces
#             id = id.at[...,1:hx,j,k].set(xyz[...,slix,j,k])
#             id = id.at[...,hx+1:,j,k].set(xyz[...,slix,j,k][...,::-1])
#             if part == "imag":
#                 weights = weights.at[hx+1:,j,k].multiply(-1)

#             for i in [0,hx]: # two points per edges
#                 id = id.at[...,i,j,k].set(xyz[...,i,j,k])
#                 if part == "imag":
#                     weights = weights.at[i,j,k].multiply(0)
#                 else:
#                     weights = weights.at[i,j,k].multiply(2**.5)
    
#     return id, weights



def id_cgh(mesh_size, part="real"):
    """
    Return indices and weights to permute a real Gaussian tensor of size ``mesh_size`` (3D)
    into a complex Gaussian Hermitian tensor. 
    Handle the Hermitian symmetry, specificaly at border faces, edges, and vertices.
    """
    mesh_size = np.array(mesh_size)
    sx, sy, sz = mesh_size
    # assert sx%2 == sy%2 == sz%2 == 0, "dimensions lengths must be even."
    hx, hy, hz = mesh_size//2
    kmesh_size = (sx, sy, hz+1)
    weights = np.ones(kmesh_size) * (mesh_size.prod() / 2)**.5
    id = np.zeros((3,*kmesh_size), dtype=int)
    xyz = np.indices(mesh_size)

    if part == "imag":
        slix, sliy, sliz = slice(hx+1, None), slice(hy+1, None), slice(hz+1, None)
    else:
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
    
    return id, weights



def rg2cgh(mesh):
    """
    Permute a real Gaussian tensor (3D) into a complex Gaussian Hermitian tensor.
    The output would therefore be distributed as the real Fourier transform of a Gaussian tensor.
    """
    mesh_size = mesh.shape
    id_real, w_real = id_cgh(mesh_size, part="real")
    id_imag, w_imag = id_cgh(mesh_size, part="imag")
    return mesh[*id_real] * w_real + 1j * mesh[*id_imag] * w_imag



def cgh2rg(kmesh):
    """
    Permute a complex Gaussian Hermitian tensor into a real Gaussian tensor (3D).
    """
    kmesh_size = kmesh.shape
    mesh_size = *kmesh_size[:2], 2*(kmesh_size[2]-1)
    id_real, w_real = id_cgh(mesh_size, part="real")
    id_imag, w_imag = id_cgh(mesh_size, part="imag")
    
    mesh = jnp.zeros(mesh_size)
    mesh = mesh.at[*id_imag].set(kmesh.imag / w_imag)
    mesh = mesh.at[*id_real].set(kmesh.real / w_real)
    return mesh