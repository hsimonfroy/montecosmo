
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








# @tm.python_app
def infer_model(mesh_length, eh_approx=True, oversamp=0, s8=False, select=None, png_type=None, fourier=False):
    from pathlib import Path
    from datetime import datetime
    import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='1.' # NOTE: jax preallocates GPU (default 75%)
    
    # save_dir = Path(os.path.expanduser("~/scratch/png/abacus_c0_i0_z0.8_lrg")) # FMN
    # save_dir = Path("/lustre/fsn1/projects/rech/fvg/uvs19wt/png/") # JZ
    # save_dir = Path("/lustre/fswork/projects/rech/fvg/uvs19wt/workspace/png/") # JZ
    # main_dir = Path("/pscratch/sd/h/hsimfroy/png/abacus_c0_i0_z0.8_lrg") # Perlmutter
    main_dir = Path("/pscratch/sd/h/hsimfroy/png/fpm_b2760_z1_lrg_fNL") # Perlmutter
    load_dir = main_dir / "load"

    fNL_true = 100

    # save_dir = main_dir / f"tracer_real_eh{eh_approx:d}_ovsamp{oversamp:d}_s8{s8:d}_fNL"
    save_dir = main_dir / (f"tracer_fpmred_eh{eh_approx:d}_ovsamp{oversamp:d}_s8{s8:d}" + ("_fNL" if png_type=='fNL' else "_fNLb" if png_type=='fNL_bias' else ""))
    # save_dir = main_dir / f"selfspec_red_eh{eh_approx:d}_ovsamp{oversamp:d}_s8{s8:d}_fNL"
    save_dir /= f"lpt_{mesh_length:d}" + (f"_sel{select}" if select is not None else "") + f"_fNL{fNL_true:.0f}" + ("_fourier" if fourier else "") + ""

    chains_dir = save_dir / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)
    import sys
    sys.stdout = sys.stderr = open(save_dir / "run.out", "a")
    print(f"Started running on {os.environ.get('HOSTNAME')} at {datetime.now().astimezone().isoformat()}")
    print(f"Submitted from host {os.environ.get('SLURM_SUBMIT_HOST')} to node(s) {os.environ.get('SLURM_JOB_NODELIST')}")
    print(f"SAVE DIR: {save_dir}")
    
    import numpy as np
    from functools import partial
    import matplotlib.pyplot as plt
    from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices, jit, vmap, grad, debug, tree, pmap
    jconfig.update("jax_enable_x64", True)
    print('\n', jdevices())
    vmap = pmap

    jconfig.update("jax_compilation_cache_dir", str(save_dir / "jax_cache/"))
    jconfig.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jconfig.update("jax_persistent_cache_min_compile_time_secs", 10)
    jconfig.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    from montecosmo.model import FieldLevelModel, default_config
    from montecosmo.utils import pdump, pload, chreshape, r2chshape, boxreshape, cgh2rg
    from montecosmo.bricks import top_hat_selection, gen_gauss_selection
    

    
    ########
    # Load #
    ########
    box_size = 3*(2760,)
    selection = None
    # mesh_length = 96
    z_obs = 1.

    oversamp_config = {
        'init_oversamp':1.,
        'evol_oversamp':2.,
        'ptcl_oversamp':2.,
        'paint_oversamp':2.,
        # 'evol_oversamp':7/4,
        # 'ptcl_oversamp':7/4,
        # 'paint_oversamp':3/2,
        'k_cut':jnp.inf,    
        } if oversamp==1 else {
        'init_oversamp':1.5,
        'evol_oversamp':2.,
        'ptcl_oversamp':2.,
        'paint_oversamp':2.,
        'k_cut':jnp.inf,
        } if oversamp==2 else {}

    model = FieldLevelModel(**default_config | 
                            {'final_shape': 3*(mesh_length,), 
                            'cell_length': box_size[0] / mesh_length, # in Mpc/h
                            # 'box_center': (0.,0.,0.), # in Mpc/h
                            'box_center': (0.,0.,1.), # in Mpc/h
                            # 'box_center': (0.,0.,1938.), # in Mpc/h # a2chi(model.cosmo_fid, a=1/(1+z_obs))
                            'box_rotvec': (0.,0.,0.,), # rotation vector in radians
                            'evolution': 'lpt',
                            # 'evolution': 'kaiser',
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
                            'init_power': load_dir / f'init_kpow.npy' if not eh_approx else None,
                            # 'init_power': None,
                            'lik_type': 'gaussian_fourier' if fourier else 'gaussian_delta',
                            'png_type': png_type,
                            # 'precond': 'kaiser_dyn'
                            } | oversamp_config)

    truth = {
        'Omega_m': 0.3137721, 
        'sigma8': 0.8076353990239834,
        # 'b1': 0.,
        # 'b2': 0.,
        # 'bs2': 0.,
        'b1': .8,
        'b2': 0.,
        'bs2': 0.,
        'bn2': 0.,
        'bnpar': 0.,
        'b3': 0.,
        'fNL': 0.,
        'fNL_bp':fNL_true,
        'fNL_bpd':0.,
        'alpha_iso': 1.,
        'alpha_ap': 1.,
        # 'ngbars': 8.43318125e-4,
        'ngbars': 1e-4,
        # 'ngbars': 10000., # neglect lik noise
        's_0': 0.2,
        's_2': 0.,
        's_2mu': 0.,
        's_delta': 0.7,
        's_phi': 0.,
        }
    
    # truth = {
    #     'Omega_m': 0.3137721, 
    #     'sigma8': 0.8076353990239834,
    #     'b1': 1.0,
    #     'b2': 0.32,
    #     'bs2': -0.71,
    #     'bn2': -41.,
    #     'bnpar': -22.,
    #     'fNL': 0.,
    #     'fNL_bp': 0.,
    #     'fNL_bpd': 0.,
    #     'ngbars': 1e-4,
    #     's_0': 0.23,
    #     's_2': 0.,
    #     's_2mu': 0.,
    #     's_delta': 0.35,
    # }

    latents = model.new_latents_from_loc(truth, update_prior=True)
    model = FieldLevelModel(**model.asdict() | {'latents': latents})
    print(model)
    # model.render()


    # # Abacus matter
    # # obs_mesh = jnp.load(load_dir / f'fin_paint2_interl2_deconv0_{mesh_length}.npy')
    # # obs_mesh = jnp.load(load_dir / f'fin_paint2_interl1_deconv1_{mesh_length}.npy')
    # obs_mesh = jnp.load(load_dir / f'fin_paint2_interl2_deconv1_{mesh_length}.npy')
    # # obs_mesh = (1 + truth['b1']) * (obs_mesh - 1) + 1
    # obs_mesh *= truth['ngbars'] * model.cell_length**3
    # var = truth['s_0'] * model.cell_length**3
    # obs_mesh += jr.normal(jr.key(44), obs_mesh.shape) * var**.5
    # # obs_mesh = jr.poisson(jr.key(44), jnp.abs(obs_mesh + 1) * mean_count)

    # Abacus tracer real or redshift-space
    # obs_mesh = jnp.load(load_dir / f'tracer_6746545_paint2_deconv1_{mesh_length}.npy')
    # obs_mesh = jnp.load(load_dir / f'tracer_6746545_rsdflat_paint2_deconv1_{mesh_length}.npy')

    if fNL_true == 0:
        obs_mesh = jnp.load(load_dir / f'tracer_2099282_fNL0_paint2_deconv1_{mesh_length}.npy')
    elif fNL_true == 100:
        obs_mesh = jnp.load(load_dir / f'tracer_2099376_fNL100_paint2_deconv1_{mesh_length}.npy')
    elif fNL_true == -100:
        obs_mesh = jnp.load(load_dir / f'tracer_2099359_fNL-100_paint2_deconv1_{mesh_length}.npy')
    obs_mesh *= truth['ngbars'] * model.cell_length**3
    

    # # Abacus initial
    init_mesh = jnp.fft.rfftn(jnp.load(load_dir / f"init_mesh_fake_2760_{256}.npy"))
    init_mesh = chreshape(init_mesh, r2chshape(model.init_shape))

    # # init_mesh = jnp.fft.rfftn(jnp.load(load_dir / f'init_mesh_{mesh_length}.npy'))
    # init_mesh = jnp.fft.rfftn(jnp.load(load_dir / f'init_mesh_{576}.npy'))
    # init_mesh = chreshape(init_mesh, r2chshape(model.init_shape))
    truth = truth | {'init_mesh': init_mesh} | {'obs': obs_mesh}
    del obs_mesh
    del init_mesh


    # # Abacus within bigger volume 
    # # /!\ Don't known init_mesh anymore, load a fake one
    # init_mesh = jnp.fft.rfftn(jnp.load(load_dir / f"init_mesh_fake_3000_{256}.npy"))
    # init_mesh = chreshape(init_mesh, r2chshape(model.init_shape))

    # # obs_mesh = jnp.load(load_dir / f'tracer_6746545_paint2_deconv1_{256}.npy')
    # obs_mesh = jnp.load(load_dir / f'tracer_6746545_rsdflat_paint2_deconv1_{256}.npy')
    # over_shape = 3*(int((1+overselect) * 256),)
    # selec_mesh = top_hat_selection(over_shape, model.selection, norm_order=np.inf)
    # selec_mesh *= top_hat_selection(over_shape, 1., norm_order=8., pow_order=8.)
    # # selec_mesh *= gen_gauss_selection(model.box_center, model.box_rot, model.box_size, over_shape, True, order=4.)
    # selec_mesh /= selec_mesh[selec_mesh > 0].mean()

    # obs_mesh = realreshape(obs_mesh, over_shape)
    # obs_mesh *= selec_mesh
    # obs_mesh = jnp.fft.rfftn(obs_mesh)
    # obs_mesh = jnp.fft.irfftn(chreshape(obs_mesh, r2chshape(model.final_shape)))
    # obs_mesh = model.mesh2masked(obs_mesh)
    # obs_mesh *= truth['ngbars'] * model.cell_length**3 / obs_mesh.mean()
    # truth = truth | {'init_mesh': init_mesh} | {'obs': obs_mesh}
    # del obs_mesh
    # del init_mesh
    # del selec_mesh


    # # Self-specified
    # # truth |= {'init_mesh': truth0['init_mesh']}
    # truth = model.predict(samples=truth, hide_base=False, hide_samp=False, from_base=True)

    model.save(save_dir / "model.yaml")    
    jnp.savez(save_dir / "truth.npz", **truth)
    sys.stdout.flush()




    ##########
    # Warmup #
    ##########
    # n_samples, n_runs, n_chains = 128 * 64 // model.final_shape[0], 16, 4
    n_samples, n_runs, n_chains = 128 * 64 // model.final_shape[0], 8, 4
    print(f"n_samples: {n_samples}, n_runs: {n_runs}, n_chains: {n_chains}")
    tune_mass = True

    model.reset()
    if fourier:
        model.substitute({'obs': cgh2rg(jnp.fft.rfftn(truth['obs']))} | model.loc_fid, from_base=True)
    else:
        model.substitute({'obs': truth['obs']} | model.loc_fid, from_base=True)

    print('data params:', model.data.keys())
    model.block()
    params_start = jit(vmap(partial(model.kaiser_post, delta_obs=model.count2delta(truth['obs']), scale_field=2/3)))(jr.split(jr.key(45), n_chains))
    print('start params:', params_start.keys())
    # model.logpdf = jit(model.logpdf) # TODO: test if pre-jitting helps

    # overwrite = True
    overwrite = False
    if not os.path.exists(chains_dir / "warm1_state.p") or overwrite:
        print("\nWarming up...")

        from montecosmo.samplers import get_mclmc_warmup
        # warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=2**13, config=None,
        warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=2**14, config=None,
                                    desired_energy_var=1e-6, diagonal_preconditioning=False)))
        state, config = warmup_fn(jr.split(jr.key(43), n_chains), params_start)
        pdump(state, chains_dir / "warm1_state.p")
        pdump(config, chains_dir / "warm1_conf.p")
    else:
        state = pload(chains_dir / "warm1_state.p")
        config = pload(chains_dir / "warm1_conf.p")

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
    prob = (0.68, 0.95)

    plt.figure(figsize=(12, 4), layout='constrained')
    def plot_kptcs(kptcs, label=None):
        plot_powtranscoh(*kptcs, fill=prob)
        plot_powtranscoh(*tree.map(lambda x: jnp.median(x, 0), kptcs), label=label)

    plot_kptcs(kptcs_init, label='init')
    # plot_kptcs(kptcs_init2, label='init2')
    plot_kptcs(kptcs_warm, label='warm')

    plt.subplot(131)
    plot_pow(*kpow_true, 'k:', label='true')
    plot_pow(*kpow_fid, 'k--', alpha=0.5, label='fiducial')
    plt.legend()
    plt.subplot(132)
    plt.axhline(1., linestyle=':', color='k', alpha=0.5)
    plot_trans(kpow_true[0], (kpow_fid[1] / kpow_true[1])**.5, 'k--', alpha=0.5, label='fiducial')
    plt.subplot(133)
    plt.axhline(model.selec_mesh.mean(), linestyle=':', color='k', alpha=0.5)
    plt.savefig(save_dir / 'init_warm.png')
    sys.stdout.flush()







    ###################
    # Warmup2 and Run #
    ###################
    # jconfig.update("jax_debug_nans", True)
    from tqdm import tqdm
    from montecosmo.samplers import get_mclmc_warmup, get_mclmc_run
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

    obs = ['obs',
        #    'fNL',
        # 'fNL_bp',
        # 'fNL_bpd',
            # 'b1',
        #     'b2','bs2','bn2', 
        #    'bnpar',
            #   'b3',
            # 'ngbars', 
            # 's_0',
            # 's_2',
            # 's_2mu',
            's_delta',
            's_phi',
            # 'Omega_m',
            # 'sigma8',
            # 'init_mesh',
        'alpha_iso','alpha_ap',]
    obs += ['Omega_m'] if not eh_approx else []
    obs += ['sigma8'] if not s8 else []
    obs += ['fNL_bp','fNL_bpd'] if not png_type=='fNL_bias' else []
    obs += ['fNL'] if not ('fNL'==png_type or 'fNL_bias'==png_type) else []
    obs += ['s_delta'] if fourier else ['s_2', 's_2mu']
    
    # obs += ['fNL']
    obs = {k: truth[k] for k in obs}
    obs |= {'obs': cgh2rg(jnp.fft.rfftn(truth['obs']))} if fourier else {}

    model.reset()
    model.substitute(obs, from_base=True)
    # model.render()
    model.block()
    params_start = jit(vmap(partial(model.kaiser_post, delta_obs=model.count2delta(truth['obs']))))(jr.split(jr.key(45), n_chains))
    params_warm = params_start | state.position
    print('warm params:', params_warm.keys())

    print('warm params:', vmap(model.reparam)({k:v for k,v in params_warm.items() if k != 'init_mesh_'}))
    
    # overwrite = True
    overwrite = False
    start = 1

    def print_mclmc_config(config, state):
        print("\nss: ", config.step_size)
        print("L: ", config.L)

        from jax.flatten_util import ravel_pytree
        _, unrav_fn = ravel_pytree(tree.map(lambda x:x[0], state.position))
        invmm = vmap(unrav_fn)(config.inverse_mass_matrix)
        print("invmm mean:", tree.map(lambda x: x.mean(range(1, x.ndim)), invmm))
        print("invmm init_mesh_ std:", tree.map(lambda x: x.std(range(1, x.ndim)), invmm)['init_mesh_'])
        # print("invmm nan count:", tree.map(lambda x: jnp.isnan(x).sum(range(1, x.ndim)), invmm))

    if not os.path.exists(chains_dir / "warm2_state.p") or overwrite:
        print("\nWarming up 2...")
        # warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=2**13, config=None,
        warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=2**14, config=None,
                                            # desired_energy_var=3e-7, diagonal_preconditioning=tune_mass)))
                                            desired_energy_var=1e-7, diagonal_preconditioning=tune_mass)))
                                            # desired_energy_var=1e-8, diagonal_preconditioning=tune_mass)))
        state, config = warmup_fn(jr.split(jr.key(43), n_chains), params_warm)
        
        print_mclmc_config(config, state)
        eval_per_ess = 1e3
        ss = jnp.median(config.step_size)
        config = MCLMCAdaptationState(L=0.4 * eval_per_ess / 2 * ss, 
                                    step_size=ss, 
                                    inverse_mass_matrix=jnp.median(config.inverse_mass_matrix, 0))
        config = tree.map(lambda x: jnp.broadcast_to(x, (n_chains, *jnp.shape(x))), config)
        print_mclmc_config(config, state)

        pdump(state, chains_dir / "warm2_state.p")
        pdump(config, chains_dir / "warm2_conf.p")

    elif not os.path.exists(chains_dir / "last_state.p") or overwrite:
        state = pload(chains_dir / "warm2_state.p")
        config = pload(chains_dir / "warm2_conf.p")

    else:
        state = pload(chains_dir / "last_state.p")
        config = pload(chains_dir / "warm2_conf.p")
        # print_mclmc_config(config, state)
        # config = config._replace(step_size=config.step_size / 3,)
        # print_mclmc_config(config, state)

        while os.path.exists(chains_dir / f"run_{start}.npz") and start <= n_runs:
            start += 1
        print(f"Resuming at run {start}...")
    sys.stdout.flush()

    print("\nRunning...")
    run_fn = jit(vmap(get_mclmc_run(model.logpdf, n_samples, thinning=64, progress_bar=False)))
    key = jr.key(42)

    for i_run in tqdm(range(start, n_runs + 1)):
        print(f"run {i_run}/{n_runs}")
        key, run_key = jr.split(key, 2)
        # TODO: from jax import shard_map, P, partial(shardmap, in_spec=(P(),P(),P()))
        # TODO: Falcutatif lax.with_sharding_constraint(state), ensure_compile_time_eval
        state, samples = run_fn(jr.split(run_key, n_chains), state, config)

        # TODO: process_allgather(state and samples, tiled=False), sync_global_devices
        
        print("MSE per dim:", jnp.mean(samples['mse_per_dim'], 1), '\n')
        jnp.savez(chains_dir / f"run_{i_run}.npz", **samples)
        pdump(state, chains_dir / "last_state.p")

    from montecosmo.script import load_model, warmup1, warmup2run, make_chains
    make_chains(save_dir, start=1, end=100, reparb=False)
    print(f"Finished running on {os.environ.get('HOSTNAME')} at {datetime.now().astimezone().isoformat()}")





# @tm.python_app
def make_chains_dir(main_dir, start=1, end=100, thinning=1, reparb=False, overwrite=False):
    import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='1.' # NOTE: jax preallocates GPU (default 75%)
    from pathlib import Path
    from jax import config as jconfig, devices as jdevices
    jconfig.update("jax_enable_x64", True)
    print('\n', jdevices())
    from montecosmo.script import make_chains

    main_dir = Path(main_dir)
    dirs = [dir for dir in sorted(os.listdir(main_dir)) if (main_dir / dir).is_dir()]
    dirs.append("") # also process save_dir itself
    for dir in dirs:
        save_dir = main_dir / dir
        chains_dir = save_dir / "chains"
        # Check if there are chain runs but no processed chains yet
        if (os.path.exists(chains_dir / f"run_{start}.npz") and 
        (overwrite or not os.path.exists(chains_dir / "chains.p"))):
            print(f"Processing runs in {chains_dir}")
            make_chains(save_dir, start=start, end=end, thinning=thinning, reparb=reparb, prefix='')


# @tm.python_app
def compare_chains_dir(main_dir, labels, names=None):
    import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='1.' # NOTE: jax preallocates GPU (default 75%)
    from pathlib import Path
    from jax import config as jconfig, devices as jdevices
    jconfig.update("jax_enable_x64", True)
    print('\n', jdevices())
    from montecosmo.script import compare_chains

    save_dirs = []
    main_dir = Path(main_dir)
    if names is not None:
        dirs = names
    else:
        dirs = [dir for dir in sorted(os.listdir(main_dir)) if (main_dir / dir).is_dir()]
        dirs.append("") # also process save_dir itself
    for dir in dirs:
        save_dir = main_dir / dir
        chains_dir = save_dir / "chains"
        # Check if there are processed chains
        if os.path.exists(chains_dir / "chains.p"): 
            save_dirs.append(save_dir)
            print(f"Adding {dir} to comparison")
    compare_chains(save_dirs, labels, main_dir)  



if __name__ == '__main__':
    print("Demat")
    mesh_lengths = [64]
    eh_approxs = [False]
    oversamps = [2]
    s8s = [False]
    selects = [None]
    png_types = ['fNL_bias']  # 'fNL', 'fNL_bias', None
    fouriers = [True]
    # infer_model = tm.python_app(infer_model)
    
    for mesh_length in mesh_lengths:
        for eh_approx in eh_approxs:
            for oversamp in oversamps:
                for s8 in s8s:
                    for select in selects:
                        for png_type in png_types:
                            for fourier in fouriers:
                                print(f"\n=== mesh_length: {mesh_length}, eh_approx: {eh_approx}, oversamp: {oversamp}, s8: {s8}, sel: {select}, png_type: {png_type}, fourier: {fourier} ===")
                                infer_model(mesh_length, eh_approx=eh_approx, oversamp=oversamp, s8=s8, 
                                            select=select, png_type=png_type, fourier=fourier)

    # # # overwrite = False
    # overwrite = True
    # save_dir = "/pscratch/sd/h/hsimfroy/png/fpm_b2760_z05_lrg_fNL/tracer_fpmred_eh0_ovsamp2_s80_fNLb/lpt_64_fNL-100"
    # make_chains_dir(save_dir, start=1, end=100, thinning=1, reparb=False, overwrite=overwrite)

    # save_dir = "/pscratch/sd/h/hsimfroy/png/fpm_b2760_z1_lrg_fNL/tracer_fpmred_eh0_ovsamp2_s80_fNLb"
    # compare_chains_dir(save_dir,
    #                    labels=["$f_\\mathrm{NL}=-100$","$f_\\mathrm{NL}=0$", "$f_\\mathrm{NL}=100$"],
    #                    names=["lpt_64_fNL-100_fix", "lpt_64_fNL0_fix", "lpt_64_fNL100_fix"])
    #                 #    names=["lpt_32_fNL-100", "lpt_32_fNL0", "lpt_32_fNL100"])

    # spawn(queue, spawn=True)
    print("Kenavo")




