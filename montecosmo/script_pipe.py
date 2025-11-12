
from desipipe import Queue, Environment, TaskManager, spawn
from desipipe.environment import BaseEnvironment

queue = Queue('test', base_dir='_test1')
queue.clear(kill=False)

# environ = Environment("nersc-cosmodesi")  # or your environnment, see https://github.com/cosmodesi/desipipe/blob/f0e8cafe63f5aa4ca80cc5e40c6b2efa61bcbcb5/desipipe/environment.py#L196

# class MontEnv(BaseEnvironment):
#     name = 'montenv'
#     _defaults = dict(DESICFS='/global/cfs/cdirs/desi')
#     _command = 'export CRAY_ACCEL_TARGET=nvidia80 ; ' \
#                 'export MPICC="cc -shared" ; ' \
#                 'export SLURM_CPU_BIND="cores" ; ' \
#                 'source activate montenv'

environ = BaseEnvironment(command='source /global/homes/h/hsimfroy/miniforge3/bin/activate montenv')

output, error = './outs/slurm-%j.out', './outs/slurm-%j.err'
tm = TaskManager(queue=queue, environ=environ, 
                 scheduler=dict(max_workers=12), 
                 provider=dict(provider='nersc', time='04:00:00', 
                               mpiprocs_per_worker=1, nodes_per_worker=1, 
                               output=output, error=output, 
                               constraint='gpu', 
                            #    qos='debug',
                            #    qos='shared',
                               qos='regular',
                            #    qos='interactive', # can not sbatch, must do salloc
                            #    qos='premium',
                               ))









@tm.python_app
def infer_model(mesh_length, eh_approx=True, ovsamp=False, poisson=False):
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

    # save_dir = Path(os.path.expanduser("~/scratch/png/"))
    # save_dir = Path("/lustre/fsn1/projects/rech/fvg/uvs19wt/png/") # JZ
    # save_dir = Path("/lustre/fswork/projects/rech/fvg/uvs19wt/workspace/png/") # JZ
    save_dir = Path("/pscratch/sd/h/hsimfroy/png/abacs0") # Perlmutter
    load_dir = Path("./scratch/abacus_c0_i0_z08_lrg/")

    save_dir += f"_eh{eh_approx:d}_ovsamp{ovsamp:d}"
    save_dir = save_dir / f"lpt_{mesh_length:d}"
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
    z_obs = 0.8
    box_size = 3*(2000,)
    cell_budget = mesh_length**3
    selection = None
    mesh_length = round(cell_budget**(1/3))

    ovsamp_config = {
        'init_oversamp':7/4,
        'ptcl_oversamp':7/4,
        'paint_oversamp':3/2,
        'k_cut':jnp.inf,    
        } if ovsamp else {}

    model = FieldLevelModel(**default_config | 
                            {'mesh_shape': 3*(mesh_length,), 
                            'cell_length': box_size[0] / mesh_length, # in Mpc/h
                            'box_center': (0.,0.,0.), # in Mpc/h
                            'box_rotvec': (0.,0.,0.), # rotation vector in radians
                            'evolution': 'lpt',
                            'a_obs': 1 / (1 + z_obs), # light-cone if None
                            'curved_sky': False, # curved vs. flat sky
                            'ap_auto': None, # parametrized AP vs. auto AP
                            'selection': selection, # if float, padded fraction, if str or Path, path to window mesh file
                            'paint_order':2, # order of interpolation kernel
                            'interlace_order':2, # interlacing order
                            'n_rbins': 1,
                            'init_power': load_dir / f'init_kpow.npy' if not eh_approx else None, # if None, use EH power
                            'init_oversamp':1., # initial mesh 1D oversampling factor
                            'ptcl_oversamp':1., # particle grid 1D oversampling factor
                            'paint_oversamp':1., # painted mesh 1D oversampling factor
                            'k_cut':jnp.inf,
                            } | ovsamp_config)

    print(model)
    # model.render()
    truth = {
        'Omega_m': 0.3137721, 
        'sigma8': 0.8076353990239834,
        # 'b1': 1.,
        'b1': 0.,
        'b2': 0.,
        'bs2': 0.,
        'bn2': 0.,
        'bnp': 0.,
        'fNL': 0.,
        'alpha_iso': 1.,
        'alpha_ap': 1.,
        'ngbars': 0.00084,
        }
    init_mesh = jnp.load(load_dir / f'init_mesh_{mesh_length}.npy')
    truth |= {'init_mesh': jnp.fft.rfftn(init_mesh)}
    del init_mesh

    # Abacus-truth
    obs_mesh = jnp.load(load_dir / f'fin_paint_{mesh_length}.npy')
    # obs_mesh = jnp.load(load_dir / f'tracer_mesh_6746545_{mesh_length}.npy')

    obs_mesh -= 1
    mean_count = truth['ngbars'] * model.cell_length**3
    if poisson:
        obs_mesh = jr.poisson(jr.key(44), jnp.abs(obs_mesh + 1) * mean_count) / mean_count - 1
    else:
        obs_mesh += jr.normal(jr.key(44), obs_mesh.shape) / mean_count**.5
    truth |= {'obs': obs_mesh}
    del obs_mesh

    # # Self-specified
    # truth = model.predict(samples=truth, hide_base=False, hide_samp=False, from_base=True)

    model.save(save_dir / "model.yaml")    
    jnp.savez(save_dir / "truth.npz", **truth)







    ##########
    # Warmup #
    ##########
    n_samples, n_runs, n_chains = 128 * 64//model.mesh_shape[0], 16, 4
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

    obs = ['obs','fNL','bnp','alpha_iso','alpha_ap','b1','b2','bs2','bn2']
    # obs = ['obs','fNL','bnp','alpha_iso','alpha_ap']
    obs += ['Omega_m'] if not eh_approx else []
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
        # warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=2**14, config=None,
        warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=2**13, config=None,
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





# @tm.python_app
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


# @tm.python_app
def compare_chains_dir(save_dir, labels):
    import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='1.' # NOTE: jax preallocates GPU (default 75%)
    from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices, jit, vmap, grad, debug, tree, pmap
    from montecosmo.script import compare_chains
    from montecosmo.utils import Path
    jconfig.update("jax_enable_x64", True)
    print('\n', jdevices())

    save_paths = []
    save_dir = Path(save_dir)
    dirs = [dir for dir in sorted(os.listdir(save_dir)) if (save_dir / dir).is_dir()]
    dirs.append("") # also process save_dir itself
    for dir in dirs:
        save_path = save_dir / dir / "test"
        # Check if there are chains
        if os.path.exists(save_path + "_chains.p"): 
            save_paths.append(save_path)
            print(f"Adding {dir} to comparison")
    compare_chains(save_paths, labels, save_dir)  



if __name__ == '__main__':
    print("Demat")
    mesh_lengths = [32, 64, 128]
    eh_approxs = [False]
    ovsamps = [True, False]
    poissons = [True, False]
    
    for mesh_length in mesh_lengths:
        for eh_approx in eh_approxs:
            for ovsamp in ovsamps:
                for poisson in poissons:
                    
                    if not (not ovsamp and poisson): # avoid poisson noise and no oversampling
                        print(f"\n--- mesh_length {mesh_length}, eh_approx {eh_approx}, ovsamp {ovsamp}, poisson {poisson} ---")
                        infer_model(mesh_length, eh_approx=eh_approx, ovsamp=ovsamp, poisson=poisson)

    # overwrite = False
    # overwrite = True
    # save_dir = "/pscratch/sd/h/hsimfroy/png/abacs0_eh1_cut0/lpt_128"
    # make_chains_dir(save_dir, start=3, end=100, thinning=1, overwrite=overwrite)

    # save_dir = "/pscratch/sd/h/hsimfroy/png/abacs0_eh1_cut0/"
    # compare_chains_dir(save_dir, labels=["128", "32", "64"])

    spawn(queue, spawn=True)

    print("Kenavo")





