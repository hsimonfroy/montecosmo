#!/usr/bin/env python
# coding: utf-8

# # Model Inference
# Infer from a cosmological model via MCMC samplers. 

# In[ ]:


import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='1.' # NOTE: jax preallocates GPU (default 75%)
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp, random as jr, jit, vmap, grad, debug, tree

from functools import partial
from getdist import plots
from numpyro import infer

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

from montecosmo.model import FieldLevelModel, default_config
from montecosmo.utils import pdump, pload
from montecosmo.mcbench import sample_and_save

# import mlflow
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8081")
# mlflow.set_experiment("infer")
# get_ipython().system('jupyter nbconvert --to script ./src/montecosmo/tests/infer_model.ipynb')


# ## Config and fiduc

# In[14]:


def get_save_dir(**kwargs):
    # dir = os.path.expanduser("~/scratch/pickles/")
    dir = os.path.expanduser("/lustre/fsn1/projects/rech/fvg/uvs19wt/pickles/")

    dir += f"m{kwargs['mesh_shape'][0]:d}_b{kwargs['box_shape'][0]:.1f}"
    dir += f"_al{kwargs['a_lpt']:.1f}_ao{kwargs['a_obs']:.1f}_lo{kwargs['lpt_order']:d}_pc{kwargs['precond']:d}_ob{kwargs['obs']}/"
    return dir

class ParseSlurmId():
    def __init__(self, id):
        self.id = str(id)

        dic = {}
        dic['mesh_length'] = [32,64,128]
        dic['lpt_order'] = [0,1,2]
        dic['precond'] = [0,1,2,3]
        dic['target_accept_prob'] = [0.65, 0.8]

        dic['box_length'] = [None]
        dic['a_lpt'] = [0.1]
        dic['a_obs'] = [0.5]
        dic['obs'] = ['mesh']
        
        for i, (k, v) in enumerate(dic.items()):
            if i < len(self.id):
                setattr(self, k, v[int(self.id[i])])
            else:
                setattr(self, k, v[0])


# In[ ]:


task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
# task_id = 0
args = ParseSlurmId(task_id)

config = {
          'mesh_shape':3 * (args.mesh_length,),
          'box_shape':3 * (args.box_length if args.box_length is not None else 5 * args.mesh_length,), 
          'a_lpt':args.a_obs if args.lpt_order > 0 else args.a_lpt,
          'a_obs':args.a_obs,
          'lpt_order':1 if args.lpt_order==1 else 2, # 2lpt + pm for 0
          'precond':args.precond,
          'obs':args.obs
          }
target_accept_prob = args.target_accept_prob
save_dir = get_save_dir(**config)

sampler = "NUTS"
n_samples, max_tree_depth, n_runs, n_chains = 64, 10, 10, 8
save_path = save_dir + f"s{sampler}_nc{n_chains:d}_ns{n_samples:d}_mt{max_tree_depth:d}_ta{target_accept_prob}"

os.makedirs(save_dir, exist_ok=True)
import sys
tempstdout = sys.stdout
tempstderr = sys.stderr
sys.stdout = open(save_path+'.out', 'a')
sys.stderr = open(save_path+'.out', 'a')
# sys.stdout = tempstdout
# sys.stderr = tempstderr


# In[15]:


model = FieldLevelModel(**default_config | config)
print(model)
# model.render()

if not os.path.exists(save_dir+"truth.p"):
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
    model.save(save_dir)    
    pdump(truth, save_dir+"truth.p")
else:
    print(f"Loading truth from {save_dir}")
    truth = pload(save_dir+"truth.p")

model.condition({'obs': truth['obs']})
model.obs_meshk = truth['obs']
model.block()
# model.render()


# ## Run

# ### NUTS, HMC

# In[ ]:


def get_mcmc(model, name="NUTS"):
    if name == "NUTS":
        kernel = infer.NUTS(
            model=model,
            # init_strategy=numpyro.infer.init_to_value(values=fiduc_params)
            step_size=1e-3, 
            max_tree_depth=max_tree_depth,
            target_accept_prob=target_accept_prob,)
        
    elif name == "HMC":
        kernel = infer.HMC(
            model=model,
            # init_strategy=numpyro.infer.init_to_value(values=fiduc_params),
            step_size=1e-3, 
            # Rule of thumb (2**max_tree_depth-1)*step_size_NUTS/(2 to 4), compare with default 2pi.
            trajectory_length= 1023 * 1e-3 / 4, 
            target_accept_prob=target_accept_prob,)

    mcmc = infer.MCMC(
        sampler=kernel,
        num_warmup=n_samples,
        num_samples=n_samples, # for each run
        num_chains=n_chains,
        chain_method="vectorized",
        progress_bar=True,)
    
    return mcmc

# print("mean_acc_prob:", last_state.mean_accept_prob, "\nss:", last_state.adapt_state.step_size)
# invmm = list(last_state.adapt_state.inverse_mass_matrix.values())[0][0]
# invmm.min(),invmm.max(),invmm.mean(),invmm.std()

# Init params
# init_model = model.copy()
# init_model.partial(temp=1e-2)
# init_params_ = init_model.predict(samples=n_chains)


# In[ ]:


continue_run = False
if continue_run:
    model.reset()
    model.condition({'obs': truth['obs']})
    model.block()
    mcmc = get_mcmc(model.model, name=sampler)

    last_state = pload(save_path + "_last_state.p")
    mcmc.num_warmup = 0
    mcmc.post_warmup_state = last_state
    init_params_ = last_state.z
else:
    model.reset()
    model.condition({'obs': truth['obs']} | model.prior_loc, frombase=True)
    model.block()
    mcmc = get_mcmc(model.model, name=sampler)
    
    init_params_ = jit(vmap(model.init_model))(jr.split(jr.key(43), n_chains))
    mcmc = sample_and_save(mcmc, 0, save_path+'_init', extra_fields=['num_steps'], init_params=init_params_)
    init_params_ = mcmc.last_state.z


# In[ ]:


mcmc_runned = sample_and_save(mcmc, n_runs, save_path, extra_fields=['num_steps'], init_params=init_params_)

