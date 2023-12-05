import jax.numpy as jnp
from jaxpm.kernels import cic_compensation, fftk
from jax import random
import pickle
import numpy as np


def cic_compensate(mesh):
  """
  Compensate for CIC painting convolution.
  Only use for computing spectra, as it can increase numerical instability if used in modeling.
  """
  kmesh = jnp.fft.rfftn(mesh)
  kmesh = kmesh * cic_compensation(fftk(mesh.shape))
  comp_mesh = jnp.fft.irfftn(kmesh)
  return comp_mesh


def sample_and_save(mcmc, model_kwargs, n_runs, save_path, save_var_names):
    """
    Warmup and run MCMC, and save the specified variables.
    """
    # Warmup sampling
    if mcmc.num_warmup>=1:
        print(f"run {0}/{n_runs} (warmup)")
        key_warmup = random.PRNGKey(0)

        # Warmup
        mcmc.warmup(key_warmup, collect_warmup=True, **model_kwargs)
        warmup_samples = mcmc.get_samples()
        warmup_samples = {key: warmup_samples[key] for key in save_var_names}

        # Saving warmup
        with open(save_path+f"_{0}.p", 'wb') as file:
            pickle.dump(warmup_samples, file, protocol=pickle.HIGHEST_PROTOCOL)
        del warmup_samples

        # Handling rng key
        key_run = mcmc.post_warmup_state.rng_key

        # Save or overwrite last state
        with open(save_path+f"_laststate.p", 'wb') as file:
            pickle.dump(mcmc.last_state, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        key_run = random.PRNGKey(0)

    # Run sampling
    for i_run in range(1, n_runs+1):
        print(f"run {i_run}/{n_runs}")
            
        # Run
        mcmc.run(key_run, **model_kwargs)
        run_samples = mcmc.get_samples()
        run_samples = {key: run_samples[key] for key in save_var_names}
        
        # Saving run
        with open(save_path+f"_{i_run}.p", 'wb') as file:
            pickle.dump(run_samples, file, protocol=pickle.HIGHEST_PROTOCOL)
        del run_samples

        # Init next run at last state
        mcmc.post_warmup_state = mcmc.last_state
        key_run = mcmc.post_warmup_state.rng_key

        # Save or overwrite last state
        with open(save_path+f"_laststate.p", 'wb') as file:
            pickle.dump(mcmc.last_state, file, protocol=pickle.HIGHEST_PROTOCOL)
    return mcmc


def load_runs(var_names, load_path, start_run, end_run):
    """
    Load and append runs saved in different files with same name.
    """
    samples = {}
    for i_run in range(start_run, end_run+1):
        # Load
        # post_samples_part = np.load(load_path+f"_{i_run}.npy", allow_pickle=True).item()
        with open(load_path+f"_{i_run}.p", 'rb') as file:
          post_samples_part = pickle.load(file)

        # Init or append samples
        if not samples:
            samples = {key: post_samples_part[key] for key in var_names}
        else:
            samples = {key: jnp.concatenate((samples[key], post_samples_part[key])) for key in var_names}
    print(f"total num samples={samples[var_names[0]].shape[0]}")
    return samples