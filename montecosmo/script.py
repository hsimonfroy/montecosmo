
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices, jit, vmap, grad, debug, tree

from montecosmo.model import FieldLevelModel, default_config
from montecosmo.utils import pdump, pload, Path
import os

def load_model(truth0, config, cell_budget, padding, save_dir, overwrite=False):

    if not os.path.exists(save_dir / "truth.npz") or overwrite:
        print("Generate truth...")
        model = FieldLevelModel(**default_config | config )
        
        fits_path = Path("/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_2/mock0/LRG_complete_SGC_1_clustering.ran.fits")
        model.add_window(fits_path, cell_budget, padding, save_dir / "window.npy")

        truth = model.predict(samples=truth0, hide_base=False, hide_samp=False, hide_det=False, from_base=True)
        model.save(save_dir / "model.yaml")    
        jnp.savez(save_dir / "truth.npz", **truth)

        # model2 = FieldLevelModel(**model.asdict() | {'evolution': 'kaiser', 'curved_sky':False, 'window':None})
        # truth2 = model2.predict(samples=truth0, hide_base=False, hide_samp=False, from_base=True)
        # model2.save(save_dir / "model2.yaml")    
        # jnp.savez(save_dir / "truth2.npz", **truth2)
    else:
        model = FieldLevelModel.load(save_dir / "model.yaml")
        truth = np.load(save_dir / "truth.npz")

        # model2 = FieldLevelModel.load(save_dir / "model2.yaml")
        # truth2 = np.load(save_dir / "truth2.npz")

    print(model)
    # model.render()

    # print(model2)
    # model2.render("bnet.png")
    return model, truth




def warmup1(save_path, n_chains, overwrite=False):
    save_dir = save_path.parent
    model = FieldLevelModel.load(save_dir / "model.yaml")
    truth = np.load(save_dir / "truth.npz")

    delta_obs  = model.count2delta(truth['obs'])
    params_init = jit(vmap(partial(model.kaiser_post, delta_obs=delta_obs, scale_field=1/10)))(jr.split(jr.key(45), n_chains))    
    # params_init2 = jit(vmap(partial(model2.kaiser_post, delta_obs=delta_obs2)))(jr.split(jr.key(45), n_chains))

    if not os.path.exists(save_path+"_warm_state.p") or overwrite:
        print("Warming up...")
        model.reset()
        model.condition({'obs': truth['obs']} | model.loc_fid, from_base=True)
        model.block()

        from montecosmo.samplers import get_mclmc_warmup
        warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=2**14, config=None, 
                                    desired_energy_var=3e-7, diagonal_preconditioning=False)))
        state, config = warmup_fn(jr.split(jr.key(43), n_chains), {k: params_init[k] for k in ['init_mesh_']})
        pdump(state, save_path+"_warm_state.p")
        pdump(config, save_path+"_warm_conf.p")
    else:
        state = pload(save_path+"_warm_state.p")
        config = pload(save_path+"_warm_conf.p")

    obs = ['obs','b1','b2','bs2','bn2','fNL','ngbar','alpha_iso','alpha_ap']
    # obs = ['obs','Omega_m','sigma8','b1','b2','bs2','bn2','ngbar']
    # obs = ['obs','b1','b2','bs2','bn2','fNL','ngbar']
    # obs = ['obs','b2','bs2','bn2','fNL','ngbar','alpha_iso','alpha_ap']
    # obs = ['obs','fNL','ngbar','alpha_iso','alpha_ap']
    # obs = ['obs','ngbar','alpha_iso','alpha_ap']
    # obs = ['obs','alpha_iso','alpha_ap']
    # obs = ['obs', 'ngbar']
    obs = {k: truth[k] for k in obs}

    model.reset()
    model.condition(obs, from_base=True)
    # model.render()
    model.block()

    params_warm = params_init | state.position
    params_warm = {k: params_warm[k] for k in params_warm.keys() - model.data.keys()}



    ########
    # Plot #
    ########
    from montecosmo.plot import plot_pow, plot_trans, plot_coh, plot_powtranscoh
    from montecosmo.bricks import lin_power_interp

    mesh_true = jnp.fft.irfftn(truth['init_mesh'])
    kpow_true = model.spectrum(mesh_true)
    kpow_fid = kpow_true[0], lin_power_interp(model.cosmo_fid)(kpow_true[0])
    kptc_obs = model.powtranscoh(mesh_true, delta_obs)
    kptcs_init = vmap(lambda x: model.powtranscoh(mesh_true, model.reparam(x, fourier=False)['init_mesh']))(params_init)
    kptcs_warm = vmap(lambda x: model.powtranscoh(mesh_true, model.reparam(x, fourier=False)['init_mesh']))(params_warm)
    # kptcs_run = vmap(lambda x: model.powtranscoh(mesh_true, model.reparam(x, fourier=False)['init_mesh']))(state.position)


    prob = 0.95

    plt.figure(figsize=(12, 4), layout='constrained')
    def plot_kptcs(kptcs, label=None):
        plot_powtranscoh(*kptcs, fill=prob)
        plot_powtranscoh(*tree.map(lambda x: jnp.median(x, 0), kptcs), label=label)

    plot_kptcs(kptcs_init, label='init')
    plot_kptcs(kptcs_warm, label='warm')
    # plot_kptcs(kptcs_run, label='run')

    plt.subplot(131)
    plot_pow(*kpow_true, 'k:', label='true')
    plot_pow(*kpow_fid, 'k--', label='fiducial')
    plt.legend()
    plt.subplot(132)
    plot_trans(kpow_true[0], (kpow_fid[1] / kpow_true[1])**.5, 'k--', label='fiducial')
    plt.axhline(1., linestyle=':', color='k', alpha=0.5)
    plt.subplot(133)
    plot_coh(kptc_obs[0], kptc_obs[3], 'k:', alpha=0.5, label='obs');
    plt.axhline(model.wind_mesh.mean(), linestyle=':', color='k', alpha=0.5)
    plt.savefig(save_path+f'_init_warm.png')   

    return model, params_warm




def warmup2run(model, params_warm, save_path, n_samples, n_runs, n_chains, tune_mass, overwrite=False):
    # jconfig.update("jax_debug_nans", True)
    from tqdm import tqdm
    from montecosmo.samplers import get_mclmc_warmup, get_mclmc_run
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

    if not os.path.exists(save_path+"_warm2_state.p") or overwrite:
        print("Warming up 2...")
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
        start = 100 ###########


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







def make_chains(save_path):
    from montecosmo.chains import Chains
    save_dir = save_path.parent

    model = FieldLevelModel.load(save_dir / "model.yaml")
    truth = dict(jnp.load(save_dir / 'truth.npz'))
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
    chains = model.load_runs(save_path, 1, 100, transforms=transforms, batch_ndim=2)
    pdump(chains, save_path + "_chains.p")
    print(chains.shape, '\n')


    transforms = [
                #   lambda x: x[:3],
                partial(Chains.thin, thinning=1),                     # thin the chains
                partial(Chains.choice, n=10, names=['init','init_']), # subsample mesh 
                ]
    chains = model.load_runs(save_path, 1, 100, transforms=transforms, batch_ndim=2)
    pdump(chains, save_path + "_chains_.p")
    print(chains.shape, '\n')


    transforms = [
                #   lambda x: x[:3],
                partial(Chains.thin, thinning=64),
                model.reparam_chains,
                partial(model.powtranscoh_chains, mesh0=mesh_true),
                ]
    chains = model.load_runs(save_path, 1, 100, transforms=transforms, batch_ndim=2)
    pdump(chains, save_path + "_chains_mesh.p")
    print(chains.shape, '\n')




def make_chains_dir(save_dir):
    save_dir = Path(save_dir)
    dirs = [dir for dir in os.listdir(save_dir) if (save_dir / dir).is_dir()]
    for dir in dirs:
        save_path = save_dir / dir / "test"
        if not os.path.exists(save_path + "_chains.p"):
            make_chains(save_path)

    
make_chains_dir("/pscratch/sd/h/hsimfroy/png/")

