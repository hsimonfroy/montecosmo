#!/usr/bin/env python
# coding: utf-8

# # Model Inference
# Infer from a cosmological model via MCMC samplers. 

# In[ ]:


import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='1.' # NOTE: jax preallocates GPU (default 75%)
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices, jit, vmap, grad, debug, tree
jconfig.update("jax_enable_x64", True)
print(jdevices())

from functools import partial
from getdist import plots
from numpyro import infer

from montecosmo.model import FieldLevelModel, default_config
from montecosmo.utils import pdump, pload
from montecosmo.mcbench import sample_and_save
from montecosmo.script import from_id, get_mcmc, get_init_mcmc

# import mlflow
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8081")
# mlflow.set_experiment("infer")
# !jupyter nbconvert --to script ./src/montecosmo/tests/infer_model.ipynb/

# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().system('hostname')


# ## Config and fiduc

# In[ ]:

from montecosmo.mcbench import Chains
from montecosmo.plot import plot_pow, plot_powtranscoh, plot_coh, theme, SetDark2
# theme(usetex=False, font_size=14)
groups = ['cosmo','bias','init']
# groups = ['cosmo_','bias_','init_']
# groups = ['cosmo','init']
# groups = ['cosmo_','init_']
group_names = ['cosmology', 'galaxy bias', 'linear field']
# group_names = ['cosmology', 'linear field']

# pfx = 'precond_m64_1lpt_'
sli = slice(0, None)
tids = [
        2040,2041,
        2140,2141,
        2240,2241,
        2340,2341,
        ][sli]
labels = [
        "direct nomm", "direct mm"
        "fourier nomm", "fourier mm"
        "static nomm", "static mm"
        "dynamic nomm", "dynamic mm"
           ][sli]
ends = 10*[100]
# starts = [1,1,1,1,1,1,1,1,1,1,1,1][sli]
starts = [2,2,2,2,2,2,2,2,2][sli]


def from_id_(i_s, tid):
    model, mcmc_config, save_dir, save_path = from_id(tid)

    # if i_s in [0]:
    #     save_path = save_path.replace('ns128', 'ns64')

    # if i_s in [1]:
    #     save_dir = save_dir.replace('obfield', 'obfield_2')
    #     save_path = save_path.replace('obfield', 'obfield_2')

    # if i_s in [2]:
    #     save_dir = save_dir.replace('obfield', 'obfield_3')
    #     save_path = save_path.replace('obfield', 'obfield_3')
    # model = FieldLevelModel.load(save_dir+"model.p")
    # print(model)
        
    return model, mcmc_config, save_dir, save_path 
    

moms = []
for i_s, (start, end, tid, lab) in enumerate(zip(starts, ends, tids, labels)):
    model, mcmc_config, save_dir, save_path = from_id_(i_s, tid)

    # Load truth
    print(f"Loading truth from {save_dir}")
    truth = pload(save_dir+"truth.p")
    mesh0 = jnp.fft.irfftn(truth['init_mesh'])
    # pow0 = model.spectrum(mesh0)
    # kptc_obs = model.powtranscoh(mesh0, truth['obs'] - 1)

    # Load chains
    thinning = 1
    transforms = [
                lambda x: x[['*~diverging']],
                partial(Chains.thin, thinning=thinning), 
                model.reparam_chains, 
                partial(model.powtranscoh_chains, mesh0=mesh0),
                partial(Chains.choice, n=100, names=['init','init_']),
                ]
    chains = model.load_runs(save_path, start, end, transforms=transforms, batch_ndim=2)
    print(chains.shape)
    pdump(chains, save_path+'chains.p')

    # Load chains
    thinning = 1
    transforms = [
                lambda x: x[['*~diverging']],
                partial(Chains.thin, thinning=thinning), 
                partial(Chains.choice, n=100, names=['init','init_']),
                ]
    chains = model.load_runs(save_path, start, end, transforms=transforms, batch_ndim=2)
    print(chains.shape)
    pdump(chains, save_path+'chains_.p')

