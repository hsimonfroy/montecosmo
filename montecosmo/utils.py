import os
from pickle import dump, load, HIGHEST_PROTOCOL

import jax.numpy as jnp
import jax.random as jr
from jax import jit
from jax.tree_util import tree_map
from functools import wraps, partial

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import to_rgba_array

from numpyro.infer import MCMC



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


def get_ylim(a, scale=1.25, q=0.001):
    """
    Compute inferior and superior limit values of an array, 
    with some scaled margins, and discarding bilateraly on some quantile level.
    """
    ymin, ymax = jnp.quantile(a, q/2), jnp.quantile(a, 1-q/2)
    ymean, ydiff = (ymax+ymin)/2, scale*(ymax-ymin)/2
    return ymean-ydiff, ymean+ydiff


def color_switch(color, reverse=False):
    """
    Select between color an its negative, or colormap and its reversed.
    Typically used to switch between light theme and dark theme. 
    `color` must be Matpotlib color, or array of colors, or colormap.
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


def theme_switch(dark_theme=False):
    """
    Set Matplotlib theme and return an adequate color switching function.
    """
    if dark_theme: 
        plt.style.use('dark_background')
    else: 
        plt.style.use('default')
    rc('animation', html='html5') # handle Matplotlib animations
    theme = partial(color_switch, reverse=dark_theme)
    return theme


def save_run(mcmc:MCMC, i_run:int, save_path:str, var_names:list=None, extra_fields:list=[]):
    """
    Save one run of MCMC sampling, with extra fields and last state.
    If `var_names` is None, save all the variables.
    """
    samples = mcmc.get_samples()
    if var_names is not None:
        samples = {key: samples[key] for key in var_names}

    # Save run
    pickle_dump(samples, save_path+f"_{i_run}.p")
    del samples

    # Save extra fields
    if extra_fields:
        extra = mcmc.get_extra_fields()
        pickle_dump(extra, save_path+f"_extra_{i_run}.p")
        del extra

    # Save or overwrite last state
    pickle_dump(mcmc.last_state, save_path+f"_laststate.p") 


def sample_and_save(mcmc:MCMC, n_runs:int, save_path:str, var_names:list=None, extra_fields:list=[], rng_key=jr.PRNGKey(0))->MCMC:
    """
    Warmup and run MCMC, saving the specified variables and extra fields.
    Do `mcmc.num_warmup` warmup steps, followed by `n_runs` times `mcmc.num_samples` sampling steps.
    If `var_names` is None, save all the variables.
    """
    # Warmup sampling
    if mcmc.num_warmup>=1:
        print(f"run {0}/{n_runs} (warmup)")

        # Warmup
        mcmc.warmup(rng_key, collect_warmup=True, extra_fields=extra_fields)
        save_run(mcmc, 0, save_path, var_names, extra_fields)

        # Handling rng key
        key_run = mcmc.post_warmup_state.rng_key
    else:
        key_run = rng_key

    # Run sampling
    for i_run in range(1, n_runs+1):
        print(f"run {i_run}/{n_runs}")
            
        # Run
        mcmc.run(key_run, extra_fields=extra_fields)
        save_run(mcmc, i_run, save_path, var_names, extra_fields)

        # Init next run at last state
        mcmc.post_warmup_state = mcmc.last_state
        key_run = mcmc.post_warmup_state.rng_key
    return mcmc


def load_runs(start_run, end_run, load_path, var_names=None):
    """
    Load and append runs (or extra fields) saved in different files with same name.
    Both runs `start_run` and `end_run` are included.
    If `var_names` is None, load all the variables.
    """
    print(f"loading: {os.path.basename(load_path)}")
    for i_run in range(start_run, end_run+1):
        # Load
        samples_part = pickle_load(load_path+f"_{i_run}.p")   
        if var_names is not None:
            samples_part = {key: samples_part[key] for key in var_names}

        # Init or append samples
        if i_run == start_run:
            samples = samples_part
        else:
            # samples = {key: jnp.concatenate((samples[key], samples_part[key])) for key in var_names}
            samples = tree_map(lambda x,y: jnp.concatenate((x, y)), samples, samples_part)
    print(f"total run length={samples[list(samples.keys())[0]].shape[0]}")
    return samples