#!/usr/bin/env python
# coding: utf-8

# # Model Inference
# Infer from a cosmological model via MCMC samplers. 

# In[2]:


import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='1.' # NOTE: jax preallocates GPU (default 75%)
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp, random as jr, jit, vmap, grad, debug, tree

from functools import partial
from getdist import plots
from numpyro import infer

# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

from montecosmo.model import FieldLevelModel, default_config
from montecosmo.utils import pdump, pload
from montecosmo.mcbench import sample_and_save
from montecosmo.script import from_id, get_mcmc

# import mlflow
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8081")
# mlflow.set_experiment("infer")
# get_ipython().system('jupyter nbconvert --to script ./src/montecosmo/tests/infer_model.ipynb')


# ## Config and fiduc

# In[32]:


################## TO SET #######################
task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
# task_id = 4020
print("SLURM_ARRAY_TASK_ID:", task_id)
model, mcmc_config, save_dir, save_path = from_id(task_id)
os.makedirs(save_dir, exist_ok=True)

import sys
tempstdout, tempstderr = sys.stdout, sys.stderr
sys.stdout = sys.stderr = open(save_path+'.out', 'a')
job_id = int(os.environ['SLURM_ARRAY_JOB_ID'])
print("SLURM_ARRAY_JOB_ID:", job_id)
print("SLURM_ARRAY_TASK_ID:", task_id)


# In[33]:


print(model)
print(mcmc_config)
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
    truth = model.predict(samples=truth, hide_base=False, hide_samp=False, hide_det=False, frombase=True)
    
    print(f"Saving model and truth at {save_dir}")
    model.save(save_dir)    
    pdump(truth, save_dir+"truth.p")
else:
    print(f"Loading truth from {save_dir}")
    truth = pload(save_dir+"truth.p")

model.condition({'obs': truth['obs']})
model.delta_obs = truth['obs'] - 1
model.block()
# model.render()


# ## Run

# ### NUTS, HMC

# In[ ]:


continue_run = False
if continue_run:
    model.reset()
    model.condition({'obs': truth['obs']})
    model.block()
    mcmc = get_mcmc(model.model, mcmc_config)

    last_state = pload(save_path + "_last_state.p")
    mcmc.num_warmup = 0
    mcmc.post_warmup_state = last_state
    init_params_ = None
else:
    model.reset()
    model.condition({'obs': truth['obs']} | model.prior_loc, frombase=True)
    model.block()

    nuts_config = {
        'sampler':'NUTS',
        'target_accept_prob':0.65,
        'n_samples':64,
        'max_tree_depth':10,
        'n_runs':10,
        'n_chains':8
    }
    mcmc = get_mcmc(model.model, nuts_config)
    
    print("Init params")
    init_params_ = jit(vmap(model.init_model))(jr.split(jr.key(43), nuts_config['n_chains']))
    init_mesh_ = {k: init_params_[k] for k in ['init_mesh_']} # NOTE: !!!!!!!
    mcmc = sample_and_save(mcmc, save_path+'_init', 0, 0, extra_fields=['num_steps'], init_params=init_mesh_)



    from montecosmo.plot import plot_pk, plot_pktranscoh
    ls = mcmc.last_state
    mesh0 = jnp.fft.irfftn(truth['init_mesh'])
    kpks_ = vmap(lambda x: model.pktranscoh(mesh0, model.reparam(x, fourier=False)['init_mesh']))(init_params_)
    kpks = vmap(lambda x: model.pktranscoh(mesh0, model.reparam(x, fourier=False)['init_mesh']))(init_params_ | ls.z)
    kpk0 = model.spectrum(mesh0)
    kpkobs = model.spectrum(truth['obs']-1)
    print(ls.z.keys(), init_params_.keys())

    mse_ = jnp.mean((vmap(lambda x: model.reparam(x, fourier=False))(init_params_)['init_mesh']  - mesh0)**2, axis=(1,2,3))
    mse = jnp.mean((vmap(lambda x: model.reparam(x, fourier=False))(init_params_ | ls.z)['init_mesh']  - mesh0)**2, axis=(1,2,3))
    print("MSEs:", mse_, mse)

    plt.figure(figsize=(12, 4))
    plot_pktranscoh(*jnp.median(jnp.stack(kpks_), 1), label='start')
    plot_pktranscoh(*kpks_, fill=0.68)
    plot_pktranscoh(*jnp.median(jnp.stack(kpks), 1), label='end')
    plot_pktranscoh(*kpks, fill=0.68)
    plt.subplot(131)
    plot_pk(*kpk0, 'k', label='true')
    plot_pk(*kpkobs, 'k:', label='obs')
    plt.tight_layout()
    plt.savefig(save_dir+f'initpk_{task_id}.png')



    print("mean_acc_prob:", mcmc.last_state.mean_accept_prob, "\nss:", mcmc.last_state.adapt_state.step_size)
    init_params_ |= mcmc.last_state.z
    print(init_params_.keys())

    model.reset()
    model.condition({'obs': truth['obs']})
    model.block()


# In[ ]:


if mcmc_config['sampler'] != 'NUTSwG':
    mcmc = get_mcmc(model.model, mcmc_config)
    mcmc_runned = sample_and_save(mcmc, save_path, 0, mcmc_config['n_runs'], extra_fields=['num_steps'], init_params=init_params_)

else:
    from montecosmo.samplers import NUTSwG_init, get_NUTSwG_run

    step_fn, init_fn, parameters, init_state_fn = NUTSwG_init(model.logpdf)
    warmup_fn = jit(vmap(get_NUTSwG_run(model.logpdf, step_fn, init_fn, parameters, mcmc_config['n_samples'], warmup=True)))
    key = jr.key(42)
    last_state = jit(vmap(init_state_fn))(init_params_)
    # last_state = pload(save_dir+"NUTSGibbs/HMCGibbs_ns256_x_nc8_laststate32.p")


    (last_state, parameters), samples, infos = warmup_fn(jr.split(jr.key(43), mcmc_config['n_chains']), last_state)
    print(parameters,'\n=======\n')
    jnp.savez(save_path+f"_{0}.npz", **samples | {k:infos[k] for k in ['n_evals']})
    pdump(last_state, save_path+f"_last_state.p")
    # pdump(parameters, 'parameters.p'), pdump(tree.map(jnp.mean, infos), 'infos.p')

    run_fn = jit(vmap(get_NUTSwG_run(model.logpdf, step_fn, init_fn, parameters, mcmc_config['n_samples'])))
    start = 1
    end = start + mcmc_config['n_runs']-1
    for i_run in range(start, end+1):
        print(f"run {i_run}/{end}")
        key, run_key = jr.split(key, 2)
        # last_state, samples, infos = run_fn(jr.split(run_key, mcmc_config['n_chains']), last_state)
        last_state, samples, infos = run_fn(jr.split(run_key, mcmc_config['n_chains']), last_state)
        print("infos:", infos)
        jnp.savez(save_path+f"_{i_run}.npz", **samples | {k:infos[k] for k in ['n_evals']})
        pdump(last_state, save_path+f"_last_state.p")


# In[ ]:


# model.reset()
# model.condition({'obs': truth['obs']})
# model.block()
# mcmc = get_mcmc(model.model, mcmc_config)
# init_params_ = {k+'_': jnp.broadcast_to(truth[k+'_'], (mcmc_config['n_chains'], *jnp.shape(truth[k+'_']))) for k in ['Omega_m','sigma8','b1','b2','bs2','bn2','init_mesh']}

# mcmc_runned = sample_and_save(mcmc, mcmc_config['n_runs'], save_path, extra_fields=['num_steps'], init_params=init_params_)


# In[ ]:




