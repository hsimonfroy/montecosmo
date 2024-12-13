#!/usr/bin/env python
# coding: utf-8

# # Model Inference
# Infer from a cosmological model via MCMC samplers. 

# In[ ]:


import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.50' # NOTE: jax preallocates GPU (default 75%)

import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp, random as jr, jit, vmap, grad, debug, tree

from functools import partial
from getdist import plots
from numpyro import infer

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from montecosmo.model import FieldLevelModel, default_config
from montecosmo.utils import pdump, pload
from montecosmo.mcbench import sample_and_save

# import mlflow
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8081")
# mlflow.set_experiment("infer")
# !jupyter nbconvert --to script ./src/montecosmo/tests/infer_model.ipynb


# ## Config and fiduc

# In[ ]:



# Config
save_dir = os.path.expanduser("~/scratch/pickles/")
# save_dir = os.path.expanduser("/lustre/fsn1/projects/rech/fvg/uvs19wt/pickles/")
config = {
          # Mesh and box parameters
          'mesh_shape':3 * (64,), # int
          'box_shape':3 * (640.,), # in Mpc/h (aim for cell lengths between 1 and 10 Mpc/h)
          # LSS formation
          'a_lpt':0.1,
          'lpt_order':1,
          'fourier':True,
          }

# Load and save model
model = FieldLevelModel(**default_config | config)
save_dir += f"m{model.mesh_shape[0]:d}_b{model.box_shape[0]:.1f}"
save_dir += f"_al{model.a_lpt:.1f}_ao{model.a_obs:.1f}_lo{model.lpt_order:d}_f{model.precond:d}_o{model.obs}/"
# print(model)
# model.render()

if not os.path.exists(save_dir):
    # Predict and save fiducial
    truth = {'Omega_m': 0.31, 
            'sigma8': 0.81, 
            'b1': 1., 
            'b2':0., 
            'bs2':0., 
            'bn2': 0.}

    model.reset()
    truth = model.predict(samples=truth, hide_base=False, frombase=True)

    print(f"Saving model and truth at {save_dir}")
    os.mkdir(save_dir)
    model.save(save_dir)    
    pdump(truth, save_dir + "truth.p")
else:
    print(f"Loading truth from {save_dir}")
    truth = pload(save_dir + "truth.p")

model.condition({'obs': truth['obs']})
model.block()
# model.render()


# ## Run

# ### NUTS, HMC

# In[ ]:


sampler = "NUTS"
n_samples, max_tree_depth, n_runs, n_chains = 64, 10, 10, 8
save_path = save_dir + f"s{sampler}_nc{n_chains:d}_ns{n_samples:d}"


nuts_kernel = infer.NUTS(
    model=model.model,
    # init_strategy=numpyro.infer.init_to_value(values=fiduc_params)
    adapt_mass_matrix=True,
    step_size=1e-3, 
    adapt_step_size=True,
    max_tree_depth=max_tree_depth,
    target_accept_prob=0.65,)

hmc_kernel = infer.HMC(
    model=model.model,
    # init_strategy=numpyro.infer.init_to_value(values=fiduc_params),
    adapt_mass_matrix=True,
    step_size=1e-3, 
    adapt_step_size=True,
    # Rule of thumb (2**max_tree_depth-1)*step_size_NUTS/(2 to 4), compare with default 2pi.
    trajectory_length= 1023 * 1e-3 / 4, 
    target_accept_prob=0.65,)

mcmc = infer.MCMC(
    sampler=nuts_kernel,
    num_warmup=n_samples,
    num_samples=n_samples, # for each run
    num_chains=n_chains,
    chain_method="vectorized",
    progress_bar=True,)


continue_run = False
if continue_run:
    mcmc.num_warmup = 0
    last_state = pload(save_path + "_last_state.p")
    mcmc.post_warmup_state = last_state

# print("mean_acc_prob:", last_state.mean_accept_prob, "\nss:", last_state.adapt_state.step_size)
# invmm = list(last_state.adapt_state.inverse_mass_matrix.values())[0][0]
# invmm.min(),invmm.max(),invmm.mean(),invmm.std()

# Init params
init_model = model.copy()
init_model.partial(temp=1e-2)
init_params_ = init_model.predict(samples=n_chains)


# In[ ]:


# mlflow.log_metric('halt',0) # 31.46s/it 4chains, 37.59s/it 8chains
mcmc_runned = sample_and_save(mcmc, n_runs, save_path, extra_fields=['num_steps'], init_params=init_params_)
# mlflow.log_metric('halt',1)

