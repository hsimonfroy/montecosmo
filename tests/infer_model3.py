#!/usr/bin/env python
# coding: utf-8
#!jupyter nbconvert --to python /dvs_ro/u1/h/hsimfroy/workspace/montecosmo_proj/src/montecosmo/tests/infer_model2.ipynb
import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.95' # NOTE: jax preallocates GPU (default 75%)
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices, jit, vmap, grad, debug, tree, lax, value_and_grad, pmap
jconfig.update("jax_enable_x64", True)
print(jdevices())
# vmap = pmap


from montecosmo.model import FieldLevelModel, default_config
from montecosmo.utils import pdump, pload , Path

save_dir = Path("./scratch")
# save_dir = Path("/feynman/home/dphp/hs276503/scratch/png/kaiser_test21")
# save_dir = Path("/pscratch/sd/h/hsimfroy/png/kaiser_test21")
save_path = save_dir / "test"
save_dir.mkdir(parents=True, exist_ok=True)

########
# Load #
########
box_shape = 3*(2000,)
cell_budget = 64**3
selection = None
mesh_length = round(cell_budget**(1/3))

model = FieldLevelModel(**default_config | 
                        {'mesh_shape': 3*(mesh_length,), 
                        'cell_length': box_shape[0] / mesh_length, # in Mpc/h
                        'box_center': (0.,0.,0.), # in Mpc/h
                        'box_rotvec': (0.,0.,0.,), # rotation vector in radians
                        'evolution': 'lpt',
                        'a_obs': 1., # light-cone if None
                        'curved_sky': False, # curved vs. flat sky
                        'ap_auto': None, # parametrized AP vs. auto AP
                        'selection': selection, # if float, padded fraction, if str or Path, path to window mesh file
                        'paint_order':2, # order of interpolation kernel
                        'init_oversamp':1., # initial mesh 1D oversampling factor
                        'ptcl_oversamp':1., # particle grid 1D oversampling factor
                        'paint_oversamp':1., # painted mesh 1D oversampling factor
                        'interlace_order':2, # interlacing order
                        'n_rbins': 1,
                        'k_cut': None,
                        } )

print(model)
# model.render()
truth = {
    'Omega_m': 0.3, 
    # 'Omega_c': 0.3-0.04860,
    # 'Omega_b': 0.04860,
    'sigma8': 0.8,
    'b1': 1.,
    'b2': 0.,
    'bs2': 0.,
    'bn2': 0.,
    'bnp': 0.,
    'fNL': 0.,
    'alpha_iso': 1.,
    'alpha_ap': 1.,
    'ngbars': 0.00084,
    }

tracer_mesh = jnp.load("./scratch/tracer_mesh_6746545.npy")
truth = truth | {'obs': tracer_mesh}
model.save(save_dir / "model.yaml")    
jnp.savez(save_dir / "truth.npz", **truth)
delta_obs = model.count2delta(truth['obs'])

# truth2 = model.predict(samples=truth, hide_base=False, hide_samp=False, from_base=True)
# delta_obs2 = model.count2delta(truth2['obs'])
# count_obs  = model.masked2mesh(truth['obs'])
# count_obs2  = model.masked2mesh(truth2['obs'])





##########
# Warmup #
##########
n_samples, n_runs, n_chains = 128, 64, 6
tune_mass = True

model.reset()
model.condition({'obs': truth['obs']} | model.loc_fid, from_base=True)
model.block()
params_start = jit(vmap(partial(model.kaiser_post, delta_obs=delta_obs, scale_field=1/10)))(jr.split(jr.key(45), n_chains))
print('start params:', params_start.keys())

# overwrite = True
overwrite = False
if not os.path.exists(save_path+"_warm_state.p") or overwrite:
    print("Warming up...")

    from montecosmo.samplers import get_mclmc_warmup
    warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=2**14, config=None, 
                                desired_energy_var=1e-6, diagonal_preconditioning=False)))
    state, config = warmup_fn(jr.split(jr.key(43), n_chains), params_start)
    pdump(state, save_path+"_warm_state.p")
    pdump(config, save_path+"_warm_conf.p")
else:
    state = pload(save_path+"_warm_state.p")
    config = pload(save_path+"_warm_conf.p")


params_warm = params_start | state.position
print('warm params:', params_warm.keys())
# params_warm = {k: params_warm[k] for k in params_warm.keys() - model.data.keys()}

from montecosmo.plot import plot_pow, plot_trans, plot_coh, plot_powtranscoh
from montecosmo.bricks import lin_power_interp

kptcs_init = vmap(lambda x: model.powtranscoh(delta_obs, model.reparam(x)['init_mesh']))(params_start)
kptcs_warm = vmap(lambda x: model.powtranscoh(delta_obs, model.reparam(x)['init_mesh']))(params_warm)
kpow_fid = kptcs_warm[0][0], lin_power_interp(model.cosmo_fid)(kptcs_warm[0][0])
prob = 0.95

plt.figure(figsize=(12, 4), layout='constrained')
def plot_kptcs(kptcs, label=None):
    plot_powtranscoh(*kptcs, fill=prob)
    plot_powtranscoh(*tree.map(lambda x: jnp.median(x, 0), kptcs), label=label)

plot_kptcs(kptcs_init, label='init')
# plot_kptcs(kptcs_init2, label='init2')
plot_kptcs(kptcs_warm, label='warm')

plt.subplot(131)
# plot_pow(*kpow_true, 'k:', label='true')
plot_pow(*kpow_fid, 'k--', label='fiducial')
plt.legend()
plt.subplot(132)
# plot_trans(kpow_true[0], (kpow_fid[1] / kpow_true[1])**.5, 'k--', label='fiducial')
plt.axhline(1., linestyle=':', color='k', alpha=0.5)
plt.subplot(133)
# plot_coh(kptc_obs[0], kptc_obs[3], 'k:', alpha=0.5, label='obs');
plt.axhline(model.selec_mesh.mean(), linestyle=':', color='k', alpha=0.5)
plt.savefig(save_path+f'_init_warm.png')   





###################
# Warmup2 and Run #
###################
# jconfig.update("jax_debug_nans", True)
from tqdm import tqdm
from montecosmo.samplers import get_mclmc_warmup, get_mclmc_run
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

obs = ['obs','fNL','bnp','alpha_iso','alpha_ap']
# obs = ['obs','Omega_m','sigma8','fNL','b1','b2','bs2','bn2','bnp','alpha_iso','alpha_ap','ngbars']
# obs = ['obs','fNL','b1','b2','bs2','bn2','bnp','alpha_iso','alpha_ap','ngbars']
# obs = ['obs','fNL','alpha_iso','alpha_ap']
# obs = ['obs','fNL','bnp','alpha_iso','alpha_ap']
obs = {k: truth[k] for k in obs}

model.reset()
model.condition(obs, from_base=True)
# model.render()
model.block()

# overwrite = True
overwrite = False
if not os.path.exists(save_path+"_warm2_state.p") or overwrite:
    print("Warming up...")
    warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=2**14, config=None, # 2**13
                                        desired_energy_var=3e-7, diagonal_preconditioning=tune_mass)))
    state, config = warmup_fn(jr.split(jr.key(43), n_chains), params_warm)

    eval_per_ess = 1e3
    ss = jnp.median(config.step_size)
    config = MCLMCAdaptationState(L=0.4 * eval_per_ess/2 * ss, 
                                step_size=ss, 
                                inverse_mass_matrix=jnp.median(config.inverse_mass_matrix, 0))
    config = tree.map(lambda x: jnp.broadcast_to(x, (n_chains, *jnp.shape(x))), config)
    
    print("ss: ", config.step_size)
    print("L: ", config.L)
    from jax.flatten_util import ravel_pytree
    flat, unrav_fn = ravel_pytree(tree.map(lambda x:x[0], state.position))
    print("inv_mm:", unrav_fn(config.inverse_mass_matrix[0]))
    print(tree.map(vmap(lambda x: jnp.isnan(x).sum()), state.position))

    pdump(state, save_path+"_warm2_state.p")
    pdump(config, save_path+"_conf.p")
    start = 1

elif not os.path.exists(save_path+"_last_state.p") or overwrite:
    state = pload(save_path+"_warm2_state.p")
    config = pload(save_path+"_conf.p")
    start = 1

else:
    state = pload(save_path+"_last_state.p")
    config = pload(save_path+"_conf.p")
    start = 1000 ###########


print("Running...")
run_fn = jit(vmap(get_mclmc_run(model.logpdf, n_samples, thinning=64, progress_bar=False)))
key = jr.key(42)

end = start + n_runs - 1
for i_run in tqdm(range(start, end + 1)):
    print(f"run {i_run}/{end}")
    key, run_key = jr.split(key, 2)
    state, samples = run_fn(jr.split(run_key, n_chains), state, config)
    
    print("MSE per dim:", jnp.mean(samples['mse_per_dim'], 1), '\n')
    jnp.savez(save_path+f"_{i_run}.npz", **samples)
    pdump(state, save_path+"_last_state.p")

