#!/usr/bin/env python
# coding: utf-8

# Analyse cosmological model posterior samples.

# In[1]:


import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='1.' # NOTE: jax preallocates GPU (default 75%)
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices, jit, vmap, grad, debug, tree
jconfig.update("jax_enable_x64", True)
print(jdevices())

from montecosmo.model import FieldLevelModel, default_config
from montecosmo.utils import pdump, pload, Path
from getdist import plots



from montecosmo.chains import Chains
from montecosmo.plot import plot_pow, plot_trans, plot_coh, plot_powtranscoh, theme, SetDark2
theme(usetex=True, font_size=12)

ids = np.array([0,1,2,3])
# save_dir = Path("/feynman/home/dphp/hs276503/scratch/png")
save_dir = Path("/pscratch/sd/h/hsimfroy/png/")

save_dirs = np.array([save_dir / s for s in ["lpt_64_fnl_p50", "lpt_64_fnl_0", "lpt_64_fnl_m50", "lpt_fnl_64_kaiser_stat"]])[ids]
save_paths = np.array([save_dir / "test" for save_dir in save_dirs])

for sd, sp in zip(save_dirs, save_paths):

    model = FieldLevelModel.load(sd / "model.yaml")
    truth = dict(jnp.load(sd / 'truth.npz'))
    mesh_true = jnp.fft.irfftn(truth['init_mesh'])
    # kpow_true = model.spectrum(mesh_true)
    # delta_obs = model.count2delta(truth['obs'])
    # kptc_obs = model.powtranscoh(mesh_true, delta_obs)

    obs = ['obs','Omega_m','sigma8','b1','b2','bs2','bn2','fNL','ngbar','init_mesh']
    obs = {k: truth[k] for k in obs}
    model.condition(obs, from_base=True)

    transforms = [
                #   lambda x: x[:3],
                partial(Chains.thin, thinning=1),                     # thin the chains
                model.reparam_chains,                                 # reparametrize sample variables into base variables
                partial(model.powtranscoh_chains, mesh0=mesh_true),   # compute mesh statistics
                partial(Chains.choice, n=10, names=['init','init_']), # subsample mesh 
                ]
    chains = model.load_runs(sp, 1, 100, transforms=transforms, batch_ndim=2)
    pdump(chains, sp + "_chains.p")
    print(chains.shape, '\n')


    transforms = [
                #   lambda x: x[:3],
                partial(Chains.thin, thinning=1),                     # thin the chains
                partial(Chains.choice, n=10, names=['init','init_']), # subsample mesh 
                ]
    chains = model.load_runs(sp, 1, 100, transforms=transforms, batch_ndim=2)
    pdump(chains, sp + "_chains_.p")
    print(chains.shape, '\n')


    transforms = [
                #   lambda x: x[:3],
                partial(Chains.thin, thinning=64),
                model.reparam_chains,
                partial(model.powtranscoh_chains, mesh0=mesh_true),
                ]
    chains = model.load_runs(sp, 1, 100, transforms=transforms, batch_ndim=2)
    pdump(chains, sp + "_chains_mesh.p")
    print(chains.shape, '\n')





# In[ ]:




