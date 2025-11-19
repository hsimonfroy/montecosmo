
from desipipe import Queue, Environment, TaskManager, spawn
from desipipe.environment import BaseEnvironment

queue = Queue('test', base_dir='_test1')
queue.clear(kill=False)

environ = BaseEnvironment(command='source /global/homes/h/hsimfroy/miniforge3/bin/activate montenv')

output, error = './outs/slurm-%j.out', './outs/slurm-%j.err'
tm = TaskManager(queue=queue, environ=environ, 
                 scheduler=dict(max_workers=12), 
                 provider=dict(provider='nersc', time='04:00:00', 
                               mpiprocs_per_worker=1, nodes_per_worker=1, 
                               output=output, error=output, 
                               constraint='gpu', 
                               qos='regular',
                               ))









# @tm.python_app
def infer_model(mesh_length, eh_approx=True, oversamp=False):
    import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='1.' # NOTE: jax preallocates GPU (default 75%)
    from datetime import datetime
    print(f"Started running on {os.environ.get('HOSTNAME')} at {datetime.now().isoformat()}")
    print(f"Submitted from host {os.environ.get('SLURM_SUBMIT_HOST')} to node(s) {os.environ.get('SLURM_JOB_NODELIST')}")
    
    import numpy as np
    from functools import partial
    import matplotlib.pyplot as plt
    from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices, jit, vmap, grad, debug, tree, pmap
    jconfig.update("jax_enable_x64", True)
    print('\n', jdevices())
    vmap = pmap

    from montecosmo.model import FieldLevelModel, default_config
    from montecosmo.utils import pdump, pload, Path

    save_dir = Path("/pscratch/sd/h/hsimfroy/png/leave1out/lpt") # Perlmutter

    save_dir += f"_{mesh_length:d}_eh{eh_approx:d}_ovsamp{oversamp:d}"
    save_path = save_dir / "test"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"SAVE DIR: {save_dir}")

    jconfig.update("jax_compilation_cache_dir", str(save_dir / "jax_cache/"))
    jconfig.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jconfig.update("jax_persistent_cache_min_compile_time_secs", 10)
    jconfig.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    

    
    ########
    # Load #
    ########
    box_size = 3*(2000,)
    selection = None
    z_obs = 0.8

    oversamp_config = {
        'evol_oversamp':2.,
        'ptcl_oversamp':2.,
        'paint_oversamp':2.,
        'k_cut':jnp.inf,    
        } if oversamp else {}

    model = FieldLevelModel(**default_config | 
                            {'final_shape': 3*(mesh_length,), 
                            'cell_length': box_size[0] / mesh_length, # in Mpc/h
                            'box_center': (0.,0.,0.), # in Mpc/h
                            'box_rotvec': (0.,0.,0.,), # rotation vector in radians
                            'evolution': 'lpt',
                            'a_obs': 1 / (1 + z_obs), # light-cone if None
                            'curved_sky': False, # curved vs. flat sky
                            'ap_auto': None, # parametrized AP vs. auto AP
                            'selection': selection, # if float, padded fraction, if str or Path, path to window mesh file
                            'paint_order':2, # order of interpolation kernel
                            'paint_deconv': True, # whether to deconvolve painted field
                            'kernel_type':'rectangular', # 'rectangular', 'kaiser_bessel'
                            'init_oversamp':1., # initial mesh 1D oversampling factor
                            'evol_oversamp':1., # evolution mesh 1D oversampling factor
                            'ptcl_oversamp':1., # particle cloud 1D oversampling factor
                            'paint_oversamp':1., # painted mesh 1D oversampling factor
                            'interlace_order':2, # interlacing order
                            'n_rbins': 1,
                            'k_cut': np.inf,
                            'init_power': 'init_kpow.npy' if not eh_approx else None, # if None, use EH power
                            'lik_type': 'gaussian_delta',
                            } | oversamp_config)

    truth = {
        'Omega_m': 0.3137721, 
        'sigma8': 0.8076353990239834,
        'b1': 1.1,
        'b2': 0.,
        'bs2': 0.,
        'bn2': 0.,
        'bnp': 0.,
        'fNL': 0.,
        'alpha_iso': 1.,
        'alpha_ap': 1.,
        'ngbars': 0.000843318125,
        'sigma_0': 0.000843318125,
        'sigma_delta': 1.,
        }
    print(model)

    # Self-specified
    truth = model.predict(samples=truth, hide_base=False, hide_samp=False, from_base=True)
    model.save(save_dir / "model.yaml")    
    jnp.savez(save_dir / "truth.npz", **truth)







    ##########
    # Warmup #
    ##########
    n_samples, n_runs, n_chains = 128 * 64//model.final_shape[0], 8, 4
    print(f"n_samples: {n_samples}, n_runs: {n_runs}, n_chains: {n_chains}")
    tune_mass = True

    model.reset()
    model.condition({'obs': truth['obs']} | model.loc_fid, from_base=True)
    model.block()
    params_start = jit(vmap(partial(model.kaiser_post, delta_obs=model.count2delta(truth['obs']), scale_field=2/3)))(jr.split(jr.key(45), n_chains))
    print('start params:', params_start.keys())

    # overwrite = True
    overwrite = False
    if not os.path.exists(save_path+"_warm_state.p") or overwrite:
        print("Warming up...")

        from montecosmo.samplers import get_mclmc_warmup
        warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=2**13, config=None, 
                                    desired_energy_var=3e-6, diagonal_preconditioning=False)))
        state, config = warmup_fn(jr.split(jr.key(43), n_chains), params_start)
        pdump(state, save_path+"_warm_state.p")
        pdump(config, save_path+"_warm_conf.p")
    else:
        state = pload(save_path+"_warm_state.p")
        config = pload(save_path+"_warm_conf.p")

    ###############
    # Plot Warmup #
    ###############
    from montecosmo.plot import plot_pow, plot_trans, plot_coh, plot_powtranscoh
    from montecosmo.bricks import lin_power_interp

    init_mesh = truth.pop('init_mesh')
    kpow_true = model.spectrum(init_mesh)
    kptcs_init = vmap(lambda x: model.powtranscoh(init_mesh, model.reparam(x)['init_mesh']))(params_start)
    kptcs_warm = vmap(lambda x: model.powtranscoh(init_mesh, model.reparam(x)['init_mesh']))(state.position)
    del init_mesh # We won't need it anymore
    kpow_fid = kptcs_warm[0][0], lin_power_interp(model.cosmo_fid)(kptcs_warm[0][0])
    prob = [0.68, 0.95]

    plt.figure(figsize=(12, 4), layout='constrained')
    def plot_kptcs(kptcs, label=None):
        plot_powtranscoh(*kptcs, fill=prob)
        plot_powtranscoh(*tree.map(lambda x: jnp.median(x, 0), kptcs), label=label)

    plot_kptcs(kptcs_init, label='init')
    plot_kptcs(kptcs_warm, label='warm')

    plt.subplot(131)
    plot_pow(*kpow_true, 'k:', label='true')
    plot_pow(*kpow_fid, 'k--', label='fiducial')
    plt.legend()
    plt.subplot(132)
    plot_trans(kpow_true[0], (kpow_fid[1] / kpow_true[1])**.5, 'k--', label='fiducial')
    plt.axhline(1., linestyle=':', color='k', alpha=0.5)
    plt.subplot(133)
    plt.axhline(model.selec_mesh.mean(), linestyle=':', color='k', alpha=0.5)
    plt.savefig(save_path+f'_init_warm.png')   








    ###################
    # Warmup2 and Run #
    ###################
    # jconfig.update("jax_debug_nans", True)
    from tqdm import tqdm
    from montecosmo.samplers import get_mclmc_warmup, get_mclmc_run
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

    obs = ['obs','fNL','bnp',
            # 'b1','b2','bs2','bn2', 
            # 'ngbars', 
            # 'sigma_0',
            # 'sigma_delta', 
            'Omega_m',
            'sigma8',
            # 'init_mesh',
        'alpha_iso','alpha_ap',]
    # obs += ['Omega_m'] if not eh_approx else []
    obs = {k: truth[k] for k in obs}

    model.reset()
    model.condition(obs, from_base=True)
    model.block()
    params_start = jit(vmap(partial(model.kaiser_post, delta_obs=model.count2delta(truth['obs']))))(jr.split(jr.key(45), n_chains))
    params_warm = params_start | state.position
    print('warm params:', params_warm.keys())

    # overwrite = True
    overwrite = False
    start = 1
    if not os.path.exists(save_path+"_warm2_state.p") or overwrite:
        print("Warming up 2...")
        warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=2**13, config=None,
                                            desired_energy_var=3e-7, diagonal_preconditioning=tune_mass)))
        state, config = warmup_fn(jr.split(jr.key(43), n_chains), params_warm)

        eval_per_ess = 1e3
        ss = jnp.median(config.step_size)
        config = MCLMCAdaptationState(L=0.4 * eval_per_ess / 2 * ss, 
                                    step_size=ss, 
                                    inverse_mass_matrix=jnp.median(config.inverse_mass_matrix, 0))
        config = tree.map(lambda x: jnp.broadcast_to(x, (n_chains, *jnp.shape(x))), config)

        def print_mclmc_config(config, state):
            print("MCLMC Config:")
            print("ss: ", config.step_size)
            print("L: ", config.L)
            from jax.flatten_util import ravel_pytree
            _, unrav_fn = ravel_pytree(tree.map(lambda x:x[0], state.position))
            invmm = config.inverse_mass_matrix[0]
            print("inv_mm:", unrav_fn(invmm))
            print(f"inv_mm mean: {invmm.mean()}, std: {invmm.std()}")
            print("nan count:", tree.map(vmap(lambda x: jnp.isnan(x).sum()), state.position))
        
        print_mclmc_config(config, state)
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

    from montecosmo.script import make_chains
    make_chains(save_path, start=1, end=100)





def make_chains_dir(save_dir, start=1, end=100, thinning=1, overwrite=False):
    import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='1.' # NOTE: jax preallocates GPU (default 75%)
    from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices, jit, vmap, grad, debug, tree, pmap
    from montecosmo.script import make_chains
    from montecosmo.utils import Path
    jconfig.update("jax_enable_x64", True)
    print('\n', jdevices())

    save_dir = Path(save_dir)
    dirs = [dir for dir in sorted(os.listdir(save_dir)) if (save_dir / dir).is_dir()]
    dirs.append("") # also process save_dir itself
    for dir in dirs:
        save_path = save_dir / dir / "test"
        # Check if there are samples but no processed chains yet
        if (os.path.exists(save_path + f"_{start}.npz") and 
        (overwrite or not os.path.exists(save_path + "_chains.p"))):
            make_chains(save_path, start=start, end=end, thinning=thinning)



if __name__ == '__main__':
    print("Demat")
    mesh_lengths = [32]
    eh_approxs = [True]
    oversamps = [True]
    
    for mesh_length in mesh_lengths:
        for eh_approx in eh_approxs:
            for oversamp in oversamps:
                print(f"\n--- mesh_length {mesh_length}, eh {eh_approx}, osamp {oversamp} ---")
                infer_model(mesh_length, eh_approx=eh_approx, oversamp=oversamp)

    # save_dir = "/pscratch/sd/h/hsimfroy/png/abacus_c0_i0_z08_lrg/matter_eh0_ovsamp1_s8_s0/lpt_64"
    # make_chains_dir(save_dir, start=1, end=100, thinning=1, overwrite=True)


    spawn(queue, spawn=True)

    print("Kenavo")





