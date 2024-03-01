import os
from pickle import dump, load, HIGHEST_PROTOCOL

import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from jax.tree_util import tree_map
from functools import wraps, partial

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import to_rgba_array

from numpyro.infer import MCMC
from numpyro.diagnostics import print_summary
from getdist.gaussian_mixtures import GaussianND
from getdist import MCSamples
from collections.abc import Iterable



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


def get_vlim(q=0., scale=1.):
    """
    Return function computing robust inferior and superior limit values of an array, 
    i.e. discard bilateraly on some quantile level, and scale the margins.
    """
    def vlim(a):
        """
        Return robust inferior and superior limit values of an array.
        """
        vmin, vmax = jnp.quantile(a, q/2), jnp.quantile(a, 1-q/2)
        vmean, vdiff = (vmax+vmin)/2, scale*(vmax-vmin)/2
        return vmean-vdiff, vmean+vdiff
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


def set_plotting_options(use_TeX):
    params = {'text.usetex': use_TeX,
            #   'ps.useafm': True,
            #   'pdf.use14corefonts': True,
              } # NOTE: 'ps.useafm' and 'pdf.use14corefonts' for PS and PDF font comptatibiliies
    plt.rcParams.update(params)


def theme_switch(dark_theme=False, use_TeX=False):
    """
    Set Matplotlib theme and return an adequate color switching function.
    """
    if dark_theme: 
        plt.style.use('dark_background')
    else: 
        plt.style.use('default')
    rc('animation', html='html5') # handle Matplotlib animations
    set_plotting_options(use_TeX)
    theme = partial(color_switch, reverse=dark_theme)
    return theme


# def save_run(mcmc:MCMC, i_run:int, save_path:str, var_names:list=None, extra_fields:list=[]):
#     """
#     Save one run of MCMC sampling, with extra fields and last state.
#     If `var_names` is None, save all the variables.
#     """
#     samples = mcmc.get_samples()
#     if var_names is not None:
#         samples = {key: samples[key] for key in var_names}

#     # Save run
#     pickle_dump(samples, save_path+f"_{i_run}.p")
#     del samples

#     # Save extra fields
#     if extra_fields:
#         extra = mcmc.get_extra_fields()
#         pickle_dump(extra, save_path+f"_extra_{i_run}.p")
#         del extra

#     # Save or overwrite last state
#     pickle_dump(mcmc.last_state, save_path+f"_laststate.p") 


def save_run(mcmc:MCMC, i_run:int, save_path:str, var_names:list=None, extra_fields:list=[]):
    """
    Save one run of MCMC sampling, with extra fields and last state.
    If `var_names` is None, save all the variables.
    """
    # Save samples (and extra fields)
    samples = mcmc.get_samples()
    if var_names is not None:
        samples = {key: samples[key] for key in var_names}

    if extra_fields:
        extra = mcmc.get_extra_fields()
        samples.update(extra)
        del extra

    pickle_dump(samples, save_path+f"_{i_run}.p")
    del samples

    # Save or overwrite last state
    pickle_dump(mcmc.last_state, save_path+f"_laststate.p") 


def sample_and_save(mcmc:MCMC, n_runs:int, save_path:str, var_names:list=None, extra_fields:list=[], rng_key=jr.PRNGKey(0)) -> MCMC:
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


def _load_runs(load_path:str, start_run:int, end_run:int, var_names:Iterable[str]=None, verbose=False):
    if verbose:
        print(f"loading: {os.path.basename(load_path)}")

    for i_run in range(start_run, end_run+1):
        # Load
        samples_part = pickle_load(load_path+f"_{i_run}.p")   
        if var_names is not None: # NOTE: var_names should not be a consumable iterator
            samples_part = {key: samples_part[key] for key in var_names}

        # Init or append samples
        if i_run == start_run:
            samples = samples_part
        else:
            # samples = {key: jnp.concatenate((samples[key], samples_part[key])) for key in var_names}
            samples = tree_map(lambda x,y: jnp.concatenate((x, y)), samples, samples_part)
            
    if verbose:
        print(f"total run length: {samples[list(samples.keys())[0]].shape[0]}")
    return samples


def load_runs(load_path:str|Iterable[str], start_run:int|Iterable[int], end_run:int|Iterable[int], 
              var_names:Iterable[str]=None, verbose=False):
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
        samples.append(_load_runs(path, start, end, var_names, verbose))

    if isinstance(load_path,str):
        return samples[0]
    else:
        return samples 
    



def get_gdprior(samples:dict, prior_config:dict, label:str="Prior",
                   verbose:bool=False, **config):
    """
    Construct getdist MCSamples from prior config.

    Only uses keys from samples.
    """
    names = list(samples.keys())
    labels = []
    means, stds = [], []
    for name in samples:
        if name.endswith('_'): # convention for a latent param 
            lab = "\overline"+prior_config[name[:-1]][0]
            mean, std = 0, 1
        else:
            lab, mean, std = prior_config[name]
        labels.append(lab)
        means.append(mean)
        stds.append(std)

    means, stds = np.array(means), np.array(stds)
    gdsamples = GaussianND(means, np.diag(stds**2), names=names, labels=labels, label=label)

    if verbose:
        if label is not None:
            print('# '+label)
        else:
            print("# <unspecified label>")
        print("GaussianND object has no samples.\n")
    return gdsamples


def _get_gdsamples(samples:dict, prior_config:dict, label:str=None,
                   verbose:bool=False, **config):
    labels = []
    for name in samples:
        if name.endswith('_'): # convention for a latent param 
            lab = "\overline"+prior_config[name[:-1]][0]
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
    ps_ = np.atleast_1d(samples)
    label = np.atleast_1d(label)
    # if label is not None:
    assert len(ps_)==len(label), "lists must have the same lengths."
    # else:
        # label = np.broadcast_to(label, ps_.shape)
    gdsamples = []

    for p_, lab in zip(ps_, label):
        gdsamples.append(_get_gdsamples(p_, prior_config, lab, verbose))

    if isinstance(samples,dict):
        return gdsamples[0]
    else:
        return gdsamples 
    

##### To plot a table ####
# plt.subplot(position=[0,-0.01,1,1]), plt.axis('off')
# labels = ["\overline"+config['prior_config'][name[:-1]][0] for name in post_samples_]
# # Define a custom formatting function to vectorize on summary array
# def format_value(value):
#     return f"{value:0.2f}"

# summary_dict = numpyro.diagnostics.summary(post_samples_, group_by_chain=False) # NOTE: group_by_chain if several chains
# summary_subdicts = list(summary_dict.values())
# summary_table = [list(summary_subdicts[i].values()) for i in range(len(summary_dict))]
# summary_cols = list(summary_subdicts[0].keys())

# # gd.fig.axes[-1]('tight'), plt.axis('tight'), plt.subplots_adjust(top=2), plt.gcf().patch.set_visible(False), 
# plt.table(cellText=np.vectorize(format_value)(summary_table),
#             # rowLabels=list(summary_dic.keys()),
#             rowLabels=["$"+label+"$" for label in labels], 
#             colLabels=summary_cols,)
# # plt.savefig(save_path+"_contour", bbox_inches='tight') # NOTE: tight bbox required for table
# # mlflow.log_figure(plt.gcf(), f"NUTS_contour.svg", bbox_inches='tight')  # NOTE: tight bbox required for table
# plt.show();