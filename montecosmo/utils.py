import pickle
import os

import jax.numpy as jnp
from jax import random, jit
from functools import wraps



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
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def save_mcmc(mcmc, i_run, save_path, save_var_names, extra_fields):
    """
    Save one run of MCMC sampling, with extra fields and last state.
    """
    samples = mcmc.get_samples()
    samples = {key: samples[key] for key in save_var_names}

    # Save warmup
    pickle_dump(save_path+f"_{i_run}.p", samples)
    del samples

    # Save extra fields
    if extra_fields:
        extra = mcmc.get_extra_fields()
        pickle_dump(save_path+f"_extra_{i_run}.p", extra)
        del extra

    # Save or overwrite last state
    pickle_dump(save_path+f"_extra_{i_run}.p", mcmc.last_state) 


def sample_and_save(mcmc, model_kwargs, n_runs, save_path, save_var_names, extra_fields:list=[], rng_key=random.PRNGKey(0)):
    """
    Warmup and run MCMC, saving the specified variables and extra fields.
    """
    # Warmup sampling
    if mcmc.num_warmup>=1:
        print(f"run {0}/{n_runs} (warmup)")

        # Warmup
        mcmc.warmup(rng_key, collect_warmup=True, **model_kwargs, extra_fields=extra_fields)
        save_mcmc(mcmc, 0, save_path, save_var_names, extra_fields)

        # Handling rng key
        key_run = mcmc.post_warmup_state.rng_key
    else:
        key_run = rng_key

    # Run sampling
    for i_run in range(1, n_runs+1):
        print(f"run {i_run}/{n_runs}")
            
        # Run
        mcmc.run(key_run, **model_kwargs, extra_fields=extra_fields)
        save_mcmc(mcmc, i_run, save_path, save_var_names, extra_fields)

        # Init next run at last state
        mcmc.post_warmup_state = mcmc.last_state
        key_run = mcmc.post_warmup_state.rng_key
    return mcmc


def load_runs(start_run, end_run, load_path, var_names=None):
    """
    Load and append runs saved in different files with same name.
    If var_names is None, load all the variables.
    """
    print(f"loading: {os.path.basename(load_path)}")
    samples = {}
    for i_run in range(start_run, end_run+1):
        # Load
        # post_samples_part = np.load(load_path+f"_{i_run}.npy", allow_pickle=True).item()
        with open(load_path+f"_{i_run}.p", 'rb') as file:
          post_samples_part = pickle.load(file)        

        # Init or append samples
        if samples == {}:
            if var_names is None:
                var_names = list(post_samples_part.keys())
            samples = {key: post_samples_part[key] for key in var_names}
        else:
            samples = {key: jnp.concatenate((samples[key], post_samples_part[key])) for key in var_names}
    print(f"total run length={samples[var_names[0]].shape[0]}")
    return samples