
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from jax import numpy as jnp, random as jr, jit, vmap, tree
from pathlib import Path
import os

from montecosmo.model import FieldLevelModel
from montecosmo.utils import h5save, h5load, h5save_tree, h5load_tree


# ---------------------------------------------------------------------------
# Inference steps
#
# The three phases share one model and a fiducial location dict `loc_fid`
# (base params + 'init_mesh' true field + 'count_mesh' observed counts), as
# built and saved (loc_fid.h5) by run/infer.py. Sampler states/configs are
# saved as HDF5 (h5save_tree); per-run samples as HDF5 (h5save). Each phase is
# skipped (loaded) if its files already exist, unless `overwrite` is True.
# ---------------------------------------------------------------------------
def field_warmup(model, loc_fid, chains_dir, n_steps, desired_energy_var, n_chains,
                 scale_field=7/8, seed=43, overwrite=False):
    """
    Field-only warmup: sample the initial density field while every other latent is
    fixed to its fiducial value. Return (state, config, params_start), where params_start
    are the Kaiser-posterior starting points (also reused by `plot_field_warmup`).
    The model is left conditioned on the fixed params and observed counts.
    """
    from montecosmo.samplers import get_mclmc_warmup
    from blackjax.mcmc.integrators import IntegratorState
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
    chains_dir = Path(chains_dir)
    state_path, conf_path = chains_dir / "field_warm_state.h5", chains_dir / "field_warm_conf.h5"

    # Fix every latent except the initial field, and condition on the observed counts.
    obs_mesh = loc_fid['count_mesh']
    model.reset()
    model.substitute({'count_mesh': obs_mesh} | model.loc_fid, from_base=True)
    model.block()

    params_start = jit(vmap(partial(model.kaiser_post, delta_obs=model.count2delta(obs_mesh),
                                    scale_field=scale_field)))(jr.split(jr.key(45), n_chains))
    print('field warmup start params:', list(params_start))

    if not state_path.exists() or overwrite:
        print("Field-only warmup...")
        warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=n_steps, config=None,
                                              desired_energy_var=desired_energy_var,
                                              diagonal_preconditioning=False)))
        state, config = warmup_fn(jr.split(jr.key(seed), n_chains), params_start)
        h5save_tree(state_path, state)
        h5save_tree(conf_path, config)
    else:
        print("Loading field warmup...")
        state = h5load_tree(state_path, IntegratorState)
        config = h5load_tree(conf_path, MCLMCAdaptationState)
    return state, config, params_start


def plot_field_warmup(model, loc_fid, params_start, state, save_dir, prob=(0.68, 0.95)):
    """
    Plot power, transfer and coherence of the field-warmup chains against the true initial
    field `loc_fid['init_mesh']`. Must be called right after `field_warmup` (model still
    conditioned on the fixed params, so `reparam` can recover the base fields).
    """
    from montecosmo.plot import plot_pow, plot_trans, plot_powtranscoh
    from montecosmo.bricks import lin_power_interp
    save_dir = Path(save_dir)

    init_mesh = loc_fid['init_mesh']
    kpow_true = model.spectrum(init_mesh)
    kptcs_start = vmap(lambda x: model.powtranscoh(init_mesh, model.reparam(x)['init_mesh']))(params_start)
    kptcs_warm = vmap(lambda x: model.powtranscoh(init_mesh, model.reparam(x)['init_mesh']))(state.position)
    kpow_fid = kptcs_warm[0][0], lin_power_interp(model.cosmo_fid)(kptcs_warm[0][0])

    plt.figure(figsize=(12, 4), layout='constrained')
    def plot_kptcs(kptcs, label=None):
        plot_powtranscoh(*kptcs, fill=prob)
        plot_powtranscoh(*tree.map(lambda x: jnp.median(x, 0), kptcs), label=label)
    plot_kptcs(kptcs_start, label='start')
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
    plt.savefig(save_dir / 'field_warm.png', dpi=300)
    plt.close()


def full_warmup(model, loc_fid, obs, state_field, chains_dir, n_steps, desired_energy_var,
                n_chains, tune_mass, eval_per_ess=1e3, seed=43, overwrite=False):
    """
    Full warmup: fix the `obs` params (dict of base values, incl. the observed 'count_mesh' field)
    and sample every other latent, seeding the field from the field-warmup `state_field`. The tuned
    config is collapsed to a single (median) config shared across chains, with trajectory length
    L set from the step size and the target `eval_per_ess`. Leaves the model conditioned on `obs`.
    """
    from montecosmo.samplers import get_mclmc_warmup
    from blackjax.mcmc.integrators import IntegratorState
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
    chains_dir = Path(chains_dir)
    state_path, conf_path = chains_dir / "full_warm_state.h5", chains_dir / "full_warm_conf.h5"

    obs_mesh = loc_fid['count_mesh']
    model.reset()
    model.substitute(obs, from_base=True)
    model.block()

    if not state_path.exists() or overwrite:
        print("Full warmup...")
        params_warm = jit(vmap(partial(model.kaiser_post,
                            delta_obs=model.count2delta(obs_mesh))))(jr.split(jr.key(45), n_chains))
        params_warm |= state_field.position if 'init_mesh' not in model.data else {}
        print('full warmup params:', list(params_warm))

        warmup_fn = jit(vmap(get_mclmc_warmup(model.logpdf, n_steps=n_steps, config=None,
                                              desired_energy_var=desired_energy_var,
                                              diagonal_preconditioning=tune_mass)))
        state, config = warmup_fn(jr.split(jr.key(seed), n_chains), params_warm)
        print_mclmc_config(config, state)

        ss = jnp.median(config.step_size)
        config = MCLMCAdaptationState(L=0.4 * eval_per_ess / 2 * ss, step_size=ss,
                                      inverse_mass_matrix=jnp.median(config.inverse_mass_matrix, 0))
        config = tree.map(lambda x: jnp.broadcast_to(x, (n_chains, *jnp.shape(x))), config)
        print_mclmc_config(config, state)

        h5save_tree(state_path, state)
        h5save_tree(conf_path, config)
    else:
        print("Loading full warmup...")
        state = h5load_tree(state_path, IntegratorState)
        config = h5load_tree(conf_path, MCLMCAdaptationState)
    return state, config


def full_run(model, state, config, chains_dir, n_samples, n_runs, n_chains,
             thinning=64, seed=42, overwrite=False):
    """
    Full run: sample `n_runs` runs of `n_samples` (thinned) samples each, saving each run as
    `run_{i}.h5` and the latest state as `run_last_state.h5`. If a previous `run_last_state.h5`
    exists (and not `overwrite`), resume from it at the first missing run. The model must already
    be conditioned on `obs` (as left by `full_warmup`).
    """
    from tqdm import tqdm
    from montecosmo.samplers import get_mclmc_run
    from blackjax.mcmc.integrators import IntegratorState
    chains_dir = Path(chains_dir)
    last_path = chains_dir / "run_last_state.h5"

    start = 1
    if last_path.exists() and not overwrite:
        state = h5load_tree(last_path, IntegratorState)
        while (chains_dir / f"run_{start}.h5").exists() and start <= n_runs:
            start += 1
        print(f"Resuming at run {start}...")

    print("Running...")
    run_fn = jit(vmap(get_mclmc_run(model.logpdf, n_samples, thinning=thinning, progress_bar=False)))
    key = jr.key(seed)
    for _ in range(1, start): # advance key so resumed runs use fresh randomness
        key, _ = jr.split(key, 2)

    for i_run in tqdm(range(start, n_runs + 1)):
        print(f"run {i_run}/{n_runs}")
        key, run_key = jr.split(key, 2)
        state, samples = run_fn(jr.split(run_key, n_chains), state, config)

        print("MSE per dim:", jnp.mean(samples['mse_per_dim'], 1), '\n')
        h5save(chains_dir / f"run_{i_run}.h5", {k: np.asarray(v) for k, v in samples.items()})
        h5save_tree(last_path, state)
    return state


# ---------------------------------------------------------------------------
# Chains post-processing
# ---------------------------------------------------------------------------
def make_chains(save_dir, start=1, end=100, thinning=1, reparb=False, prefix=""):
    from montecosmo.chains import Chains
    from montecosmo.plot import plot_pow, plot_trans, plot_coh, plot_powtranscoh, theme, SetDark2
    from getdist import plots
    import sys
    save_dir = Path(save_dir)
    sys.stdout = sys.stderr = open(save_dir / "run.out", "a")
    chains_dir = save_dir / "chains"

    model = FieldLevelModel.load(save_dir / "model.yaml")
    loc_fid = h5load(save_dir / "loc_fid.h5")
    obs_mesh = loc_fid.pop('count_mesh')
    mesh_ref = loc_fid['init_mesh']
    markers = {k: float(v) for k, v in loc_fid.items() if np.ndim(v) == 0}
    model.substitute(loc_fid, from_base=True) # fix all params (chains override the sampled ones)
    # mask_chains = np.array([0,2,3])
    mask_chains = ...

    transforms = [
                #   lambda x: x[:,30:],
                  lambda x: x[mask_chains],
                partial(Chains.thin, thinning=thinning),                     # thin the chains
                model.reparam_chains,                                 # reparametrize sample variables into base variables
                # model.reparam_bias if reparb else lambda x: x,        # reparametrize bias parameters
                partial(model.powtranscoh_chains, mesh0=mesh_ref),   # compute mesh statistics
                partial(Chains.choice, n=10, names=['init','init_']), # subsample mesh
                ]
    chains = model.load_runs(chains_dir, start, end, transforms=transforms, batch_ndim=2)
    chains.save(chains_dir / f"{prefix}chains.h5")
    print(chains.shape, '\n')

    # gdsamp = chains.prune()[list(model.groups)+['~init_mesh']].flatten().to_getdist()
    gdsamp = chains.prune()[list(model.groups)+['~init_mesh']].to_getdist()
    gdplt = plots.get_subplot_plotter(width_inch=7)
    gdplt.triangle_plot(roots=[gdsamp],
                    title_limit=1,
                    filled=True,
                    markers=markers,
                    contour_colors=[SetDark2(0)],)
    plt.savefig(save_dir / f"{prefix}triangle.png", dpi=300)



    from montecosmo.bricks import lin_power_interp
    from montecosmo.utils import chreshape, r2chshape
    mesh_obs = jnp.fft.rfftn(model.count2delta(obs_mesh))
    mesh_obs = jnp.fft.irfftn(chreshape(mesh_obs, r2chshape(model.init_shape)))
    kptc_obs = model.powtranscoh(mesh_ref, mesh_obs)

    kpow_ref = model.spectrum(mesh_ref)
    kpow_fid = kptc_obs[0], lin_power_interp(model.cosmo_fid)(kptc_obs[0])
    plt.figure(figsize=(12, 4), layout='constrained')
    def plot_kptcs(kptcs, label=None, i_color=0):
        plot_powtranscoh(*kptcs, fill=(0.68, 0.95), color=SetDark2(i_color))
        plot_powtranscoh(*tree.map(lambda x: jnp.median(x, 0), kptcs),
                         color=SetDark2(i_color), label=label)

    plt.subplot(131)
    plot_pow(*kpow_ref, 'k:', label='true')
    plot_pow(*kpow_fid, 'k--', alpha=0.5, label='fiducial')
    plt.subplot(132)
    plt.axhline(1., linestyle=':', color='k', alpha=0.5)
    plot_trans(kpow_ref[0], (kpow_fid[1] / kpow_ref[1])**.5, 'k--', alpha=0.5, label='fiducial')
    plt.subplot(133)
    plt.axhline(model.selec_mesh.mean(), linestyle=':', color='k', alpha=0.5)
    plot_coh(kptc_obs[0], kptc_obs[3], 'k--', alpha=0.5, label='obs')

    kptcs = tree.map(jnp.concatenate, chains['kptc'])
    plot_kptcs(kptcs, label='post')
    plt.subplot(131)
    plt.legend()
    plt.savefig(save_dir / f"{prefix}kptc.png", dpi=300)



    transforms = [
                  lambda x: x[mask_chains],
                partial(Chains.thin, thinning=thinning),                     # thin the chains
                partial(Chains.choice, n=10, names=['init','init_']), # subsample mesh
                ]
    chains = model.load_runs(chains_dir, 1, 100, transforms=transforms, batch_ndim=2)
    chains.save(chains_dir / f"{prefix}chains_.h5")
    print(chains.shape, '\n')

    plt.figure(figsize=(12,12))
    chains.print_summary()
    chains.prune().flatten().plot(list(model.groups_)+['logdensity'])
    # chains.prune().flatten().plot(['fNL_','fNL_bp_', 'fNL_bpd_'])
    plt.savefig(save_dir / f"{prefix}chains_.png", dpi=300)



    transforms = [
                partial(Chains.thin, thinning=64),
                model.reparam_chains,
                # model.reparam_bias if reparb else lambda x: x,
                partial(model.powtranscoh_chains, mesh0=mesh_ref),
                ]
    chains = model.load_runs(chains_dir, 1, 100, transforms=transforms, batch_ndim=2)
    chains.save(chains_dir / f"{prefix}chains_mesh.h5")
    print(chains.shape, '\n')




def compare_chains(load_dirs, labels, save_dir="./"):
    from montecosmo.chains import Chains
    from montecosmo.plot import plot_pow, plot_trans, plot_coh, plot_powtranscoh, theme, SetDark2
    from getdist import plots

    save_dir = Path(save_dir)
    chainss = []
    gdsamps = []
    for load_dir, label in zip(load_dirs, labels):
        model = FieldLevelModel.load(load_dir / "model.yaml")
        loc_fid = h5load(load_dir / 'loc_fid.h5')
        chains = Chains.load(load_dir / "chains/chains.h5")
        print('\n', chains.shape)
        gdsamp = chains.prune()[list(model.groups)+['~init_mesh']].to_getdist(label)
        # gdsamp = chains.prune()[['bias','png']+['~init_mesh']].to_getdist(label)
        chainss.append(chains)
        gdsamps.append(gdsamp)


    gdplt = plots.get_subplot_plotter(width_inch=7)
    gdplt.triangle_plot(roots=gdsamps,
                    title_limit=1,
                    # filled=True,
                    filled=3*[True]+3*[False],
                    # markers=loc_fid,
                    # markers={k:v for k,v in loc_fid.items() if k in ['fNL', 'fNL_bp', 'fNL_bpd']},
                    contour_colors=[SetDark2(i) for i in range(3)]+[SetDark2(i) for i in range(3)],
                    contour_ls=3*['-']+3*['--']+3*[':']+3*['-.'],
                    )
    plt.savefig(save_dir / f"triangle_{'_'.join(labels)[:200]}.png", dpi=300)



    mesh_ref = loc_fid['init_mesh']
    kpow_ref = model.spectrum(mesh_ref)
    plt.figure(figsize=(12, 4), layout='constrained')
    def plot_kptcs(kptcs, label=None, i_color=0):
        color = SetDark2(i_color)
        plot_powtranscoh(*kptcs, fill=0.68, color=color)
        plot_powtranscoh(*kptcs, fill=0.95, color=color)
        plot_powtranscoh(*tree.map(lambda x: jnp.median(x, 0), kptcs), color=color, label=label)

    plt.subplot(131)
    plot_pow(*kpow_ref, 'k:', label='true')
    plt.subplot(132)
    plt.axhline(1., linestyle=':', color='k', alpha=0.5)
    plt.subplot(133)
    plt.axhline(model.selec_mesh.mean(), linestyle=':', color='k', alpha=0.5)

    for i, (chains, label) in enumerate(zip(chainss, labels)):
        kptcs = tree.map(jnp.concatenate, chains['kptc'])
        plot_kptcs(kptcs, label=label, i_color=i)
    plt.subplot(131)
    plt.legend()
    plt.savefig(save_dir / f"kptc_{'_'.join(labels)[:200]}.png", dpi=300)



def print_mclmc_config(config, state):

    print("\nss: ", config.step_size)
    print("L: ", config.L)

    from jax.flatten_util import ravel_pytree
    _, unrav_fn = ravel_pytree(tree.map(lambda x:x[0], state.position))
    invmm = vmap(unrav_fn)(config.inverse_mass_matrix)
    print("invmm mean:", tree.map(lambda x: x.mean(range(1, x.ndim)), invmm))
    if 'init_mesh_' in invmm:
        print("invmm init_mesh_ std:", tree.map(lambda x: x.std(range(1, x.ndim)), invmm)['init_mesh_'])
    # print("invmm nan count:", tree.map(lambda x: jnp.isnan(x).sum(range(1, x.ndim)), invmm))
