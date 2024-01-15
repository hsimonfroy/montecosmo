import pickle
import os

import jax.numpy as jnp
from jax import random
from numpyro.handlers import seed, condition, trace
from numpyro.infer.util import log_density

from jax import random, jit, vmap, grad
from functools import partial, wraps



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


def _simulator(model, rng_seed=0, model_kwargs={}):
    cond_trace = trace(seed(model, rng_seed=rng_seed)).get_trace(**model_kwargs)
    params = {name: cond_trace[name]['value'] for name in cond_trace.keys()}
    return params


def get_simulator(model):
    """
    Return a simulator that samples from a model.
    """
    def simulator(rng_seed=0, model_kwargs={}):
        """
        Sample batches from the model.
        """
        return partial(_simulator, model)(rng_seed, model_kwargs)
    return simulator


def _logp_fn(model, params, model_kwargs={}):
    logp = log_density(model=model, 
                model_args=(), 
                model_kwargs=model_kwargs, 
                params=params)[0]
    return logp


def get_logp_fn(model):
    """
    Return a model log probabilty function.
    """
    def logp_fn(params, model_kwargs={}):
        """
        Return the model log probabilty, evaluated on some parameters.
        """
        return partial(_logp_fn, model)(params, model_kwargs)
    
    return logp_fn


def get_score_fn(model):
    """
    Return a model score function, optionally with some fixed parameters.
    """
    def score_fn(params, model_kwargs={}):
        """
        Return the model score, evaluated on some parameters.
        """
        return grad(partial(_logp_fn, model), argnums=0)(params, model_kwargs)
    
    return score_fn 





# def get_logp_fn(model, cond_params={}):
#     """
#     Return a model log probabilty function, conditioned on some parameters.
#     """
#     vlogp_model = vmap(partial(logp_model, model, cond_params), in_axes=(0,None))
#     @get_jit
#     def logp_fn(params, model_kwargs={}):
#         """
#         Return the model log probabilty, evaluated on some parameters.
#         """
#         return vlogp_model(params, model_kwargs)

#     return logp_fn

# def get_score_fn(model, cond_params={}):
#     """
#     Return a model score function, conditioned on some parameters.
#     """
#     score_model = grad(partial(logp_model, model, cond_params), argnums=0)
#     vscore_model = vmap(score_model, in_axes=(0,None))
#     @get_jit()
#     def score_fn(params, model_kwargs={}):
#         """
#         Return the model score, evaluated on some parameters.
#         """
#         return vscore_model(params, model_kwargs)
    
#     return score_fn 

# def get_simulator(model, cond_params={}):
#     """
#     Return a simulator that samples from a model conditioned on some parameters.
#     """
#     def sample_model(model, cond_params, rng_seed=0, model_kwargs={}):
#         if len(model_kwargs)==0:
#             model_kwargs = {}
#         cond_model = condition(model, cond_params) # NOTE: Only condition on random sites
#         cond_trace = trace(seed(cond_model, rng_seed=rng_seed)).get_trace(**model_kwargs)
#         params = {name: cond_trace[name]['value'] for name in cond_trace.keys()}
#         return params

#     vsample_model = vmap(partial(sample_model, model, cond_params), in_axes=(None,0))
#     vvsample_model = vmap(vsample_model, in_axes=(0,None))

#     @get_jit(static_argnames=('batch_size'))
#     def simulator(batch_size=1, rng_key=random.PRNGKey(0), model_kwargs={}):
#         """
#         Sample batches from model. If they are both strict greater than one, 
#         batch size would be left-most dimension, and model arguments size the second left-most.
#         """
#         squeeze_axis = []
#         if batch_size==1:
#             squeeze_axis.append(0)
#         if len(model_kwargs)==0:
#             model_kwargs = jnp.array([[]]) # for vmap, because jnp.array([{}]) is not valid
#             squeeze_axis.append(1)
#         keys = random.split(rng_key, batch_size)
#         params = vvsample_model(keys, model_kwargs)
#         return {name: params[name].squeeze(axis=squeeze_axis) for name in params.keys()}

#     return simulator




def save_mcmc(mcmc, i_run, save_path, save_var_names, extra_fields):
    """
    Save one run of MCMC sampling, with extra fields and last state.
    """
    samples = mcmc.get_samples()
    samples = {key: samples[key] for key in save_var_names}

    # Save warmup
    with open(save_path+f"_{i_run}.p", 'wb') as file:
        pickle.dump(samples, file, protocol=pickle.HIGHEST_PROTOCOL)
    del samples

    # Save extra fields
    if extra_fields:
        extra = mcmc.get_extra_fields()
        with open(save_path+f"_extra_{i_run}.p", 'wb') as file:
            pickle.dump(extra, file, protocol=pickle.HIGHEST_PROTOCOL)
        del extra

    # Save or overwrite last state
    with open(save_path+f"_laststate.p", 'wb') as file:
        pickle.dump(mcmc.last_state, file, protocol=pickle.HIGHEST_PROTOCOL)   


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