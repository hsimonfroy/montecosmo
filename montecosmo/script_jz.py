import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='1.' # NOTE: jax preallocates GPU (default 75%)
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices, jit, vmap, grad, debug, tree, pmap
jconfig.update("jax_enable_x64", True)
print('\n', jdevices())
vmap = pmap

from montecosmo.model import FieldLevelModel, default_config
from montecosmo.utils import pdump, pload, Path

# save_dir = Path(os.path.expanduser("~/scratch/png/"))
save_dir = Path("/lustre/fswork/projects/rech/fvg/uvs19wt/workspace/pickles/") # JZ
# save_dir = Path("/pscratch/sd/h/hsimfroy/png/abacs0") # Perlmutter
# load_dir = Path("./scratch/abacus_c0_i0_z08_lrg/")


task_id = os.environ['SLURM_ARRAY_TASK_ID']
print("SLURM_ARRAY_TASK_ID:", task_id)
task_id = int(task_id)
mesh_length = int(np.array([8, 16, 32, 64, 128, 256])[task_id % 10])
evolution = str(np.array(["lpt", "nbody"])[task_id // 10 % 10])
print("mesh_length:", mesh_length, "evolution:", evolution)

save_dir = save_dir / f"fast_{evolution}_{mesh_length:d}"
save_path = save_dir / "test"
save_dir.mkdir(parents=True, exist_ok=True)

jconfig.update("jax_compilation_cache_dir", str(save_dir / "jax_cache/"))
jconfig.update("jax_persistent_cache_min_entry_size_bytes", -1)
jconfig.update("jax_persistent_cache_min_compile_time_secs", 10)
jconfig.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")



########
# Load #
########
cell_length = 5.
model = FieldLevelModel(**default_config | 
                        {'mesh_shape': 3*(mesh_length,), 
                        'cell_length': cell_length, # in Mpc/h
                        'box_center': (0.,0.,cell_length * mesh_length), # in Mpc/h
                        'box_rotvec': (0.,0.,0.), # rotation vector in radians
                        'evolution': evolution,
                        'a_obs': 0.5, # light-cone if None
                        'curved_sky': True, # curved vs. flat sky
                        'ap_auto': None, # parametrized AP vs. auto AP
                        'selection': 0.2, # if float, padded fraction, if str or Path, path to window mesh file
                        'paint_order':2, # order of interpolation kernel
                        'interlace_order':1, # interlacing order
                        'n_rbins': 1,
                        'init_oversamp':1., # initial mesh 1D oversampling factor
                        'ptcl_oversamp':1., # particle grid 1D oversampling factor
                        'paint_oversamp':1., # painted mesh 1D oversampling factor
                        'k_cut':jnp.inf,
                        })

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
    'ngbars': 5e-4,
    }


truth = model.predict(samples=truth, hide_base=False, hide_samp=False, from_base=True)
model.save(save_dir / "model.yaml")    
jnp.savez(save_dir / "truth.npz", **truth)







##########
# Warmup #
##########
n_samples, n_runs, n_chains = 32 if model.mesh_shape[0]==128 else 128, 64 if model.mesh_shape[0]==128 else 32, 8
print(f"n_samples: {n_samples}, n_runs: {n_runs}, n_chains: {n_chains}")
tune_mass = True

model.reset()
model.condition({'obs': truth['obs']} | model.loc_fid, from_base=True)
model.block()
params_start = jit(vmap(partial(model.kaiser_post, delta_obs=model.count2delta(truth['obs']), scale_field=1/5)))(jr.split(jr.key(45), n_chains))
print('start params:', params_start.keys())

# overwrite = True
overwrite = False
if not os.path.exists(save_path+"_warm_state.p") or overwrite:
    print("Warming up...")

    from montecosmo.samplers import get_mclmc_warmup
    warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=2**14, config=None, 
    # warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=2**15, config=None, 
                                desired_energy_var=1e-6, diagonal_preconditioning=False)))
    state, config = warmup_fn(jr.split(jr.key(43), n_chains), params_start)
    pdump(state, save_path+"_warm_state.p")
    pdump(config, save_path+"_warm_conf.p")
else:
    state = pload(save_path+"_warm_state.p")
    config = pload(save_path+"_warm_conf.p")

from montecosmo.plot import plot_pow, plot_trans, plot_coh, plot_powtranscoh
from montecosmo.bricks import lin_power_interp

mesh_true = truth.pop('init_mesh')
kpow_true = model.spectrum(mesh_true)
kptcs_init = vmap(lambda x: model.powtranscoh(mesh_true, model.reparam(x)['init_mesh']))(params_start)
kptcs_warm = vmap(lambda x: model.powtranscoh(mesh_true, model.reparam(x)['init_mesh']))(state.position)
del mesh_true # We won't need it anymore
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
plot_pow(*kpow_true, 'k:', label='true')
plot_pow(*kpow_fid, 'k--', label='fiducial')
plt.legend()
plt.subplot(132)
plot_trans(kpow_true[0], (kpow_fid[1] / kpow_true[1])**.5, 'k--', label='fiducial')
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

obs = ['obs','fNL','bnp','alpha_iso','alpha_ap','ngbars']
# obs = ['obs','Omega_m','sigma8','fNL','b1','b2','bs2','bn2','bnp','alpha_iso','alpha_ap','ngbars']
# obs = ['obs','fNL','b1','b2','bs2','bn2','bnp','alpha_iso','alpha_ap','ngbars']
# obs = ['obs','fNL','alpha_iso','alpha_ap']
# obs = ['obs','fNL','bnp','alpha_iso','alpha_ap']
obs = {k: truth[k] for k in obs}

model.reset()
model.condition(obs, from_base=True)
# model.render()
model.block()
params_start = jit(vmap(partial(model.kaiser_post, delta_obs=model.count2delta(truth['obs']))))(jr.split(jr.key(45), n_chains))
params_warm = params_start | state.position
print('warm params:', params_warm.keys())

# overwrite = True
overwrite = False
start = 1
if not os.path.exists(save_path+"_warm2_state.p") or overwrite:
    print("Warming up 2...")
    warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=2**14, config=None,
    # warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=2**13, config=None,
                                        # desired_energy_var=3e-7, diagonal_preconditioning=tune_mass)))
                                        desired_energy_var=1e-6, diagonal_preconditioning=tune_mass)))
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
    _, unrav_fn = ravel_pytree(tree.map(lambda x:x[0], state.position))
    print("inv_mm:", unrav_fn(config.inverse_mass_matrix[0]))
    print(tree.map(vmap(lambda x: jnp.isnan(x).sum()), state.position))

    pdump(state, save_path+"_warm2_state.p")
    pdump(config, save_path+"_conf.p")

elif not os.path.exists(save_path+"_last_state.p") or overwrite:
    state = pload(save_path+"_warm2_state.p")
    config = pload(save_path+"_conf.p")

else:
    state = pload(save_path+"_last_state.p")
    config = pload(save_path+"_conf.p")
    while os.path.exists(save_path+f"_{start}.npz") and start <= n_runs:
        start += 1
    print(f"Resuming at run {start}...")


print("Running...")
run_fn = jit(vmap(get_mclmc_run(model.logpdf, n_samples, thinning=64, progress_bar=False)))
key = jr.key(42)

for i_run in tqdm(range(start, n_runs + 1)):
    print(f"run {i_run}/{n_runs}")
    key, run_key = jr.split(key, 2)
    state, samples = run_fn(jr.split(run_key, n_chains), state, config)
    
    print("MSE per dim:", jnp.mean(samples['mse_per_dim'], 1), '\n')
    jnp.savez(save_path+f"_{i_run}.npz", **samples)
    pdump(state, save_path+"_last_state.p")

from montecosmo.script import load_model, warmup1, warmup2run, make_chains
make_chains(save_path, start=1, end=100)

raise