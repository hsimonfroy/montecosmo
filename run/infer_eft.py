#!/usr/bin/env python
"""
Run field-level inference on a registered mock (see run/register.py), the modern replacement
for the run/script_pipe_*.py scripts. Either run directly after `salloc` (a GPU node) or submit
with desipipe by uncommenting the TaskManager block below and decorating `infer` with `@tm.python_app`.

Pipeline (per call to `infer`):
  1. Setup: jax config + compilation cache, build the save/load directories, redirect stdout/stderr
     to run.out, print host/job/time. The load dir is the `registered` folder; results are saved in
     png/<folder1>/<folder2> where folder1 = mock tag (+ png suffix) and folder2 = experiment.
  2. Fiducial: set bias/png/stoch locs, build the model from its register file (which fixes the
     fiducial cosmology + ngbars), optionally self-predict synthetic data, and save obs.h5.
  3. Inference, in three phases (run/script.py): field-only warmup, full warmup, full run -- each
     skipped/loaded if already done unless `overwrite`. States/configs/runs are saved as HDF5.
  4. Post-process the chains (make_chains).
"""

# from desipipe import Queue, Environment, TaskManager, spawn
# from desipipe.environment import BaseEnvironment
#
# queue = Queue('test', base_dir='_test1')
# queue.clear(kill=False)
#
# environ = BaseEnvironment(command='source /global/homes/h/hsimfroy/miniforge3/bin/activate montenv')
#
# output, error = './outs/slurm-%j.out', './outs/slurm-%j.err'
# tm = TaskManager(queue=queue, environ=environ,
#                  scheduler=dict(max_workers=12),
#                  provider=dict(provider='nersc', time='04:00:00',
#                                mpiprocs_per_worker=1, nodes_per_worker=1,
#                                output=output, error=output,
#                                constraint='gpu',
#                             #    qos='debug',
#                             #    qos='shared',
#                                qos='regular',
#                             #    qos='interactive', # can not sbatch, must do salloc
#                             #    qos='premium',
#                                ))
# To submit with desipipe: uncomment the block above, add `infer = tm.python_app(infer)` after the
# definition (or decorate it), call `infer(...)` in __main__, then `spawn(queue, spawn=True)`.

from pathlib import Path

REGISTERED_DIR = Path("/pscratch/sd/h/hsimfroy/png/registered")
PNG_DIR = Path("/pscratch/sd/h/hsimfroy/png")


# @tm.python_app
def infer(register_name, png_type=None, lik_type='shash', evolution='lpt',
          self_data=False, expe='', overwrite=False,
          obs_names=[],
          n_chains=4, tune_mass=True,
          n_steps_field=2**12, dev_field=1e-5,
          n_steps_full=2**13, dev_full=1e-7,
        #   n_samples=None, n_runs=10, thinning=64,
          n_samples=None, n_runs=6, thinning=32,
          scale_fid_fac=1., pred_thinning=128):
    """
    Run inference for the mock registered in `<REGISTERED_DIR>/<register_name>`.

    register_name : register HDF5 filename, e.g. 'register_abacus_c0_ph0_z0.800_LRG_redshiftspace_N32_pad0.h5'
                    (N<n> = mesh_length, pad<pad> = padding). mesh_length and the fiducial f_NL are
                    parsed from the name (f_NL from the '_fNL<value>' piece, else 0).
    png_type      : None, 'fNL' (universal mass relation -> infer fNL, fix fNL_bp/fNL_bpd),
                    or 'fNL_bias' (infer fNL_bp/fNL_bpd, fix fNL).
    lik_type      : likelihood, e.g. 'shash', 'quad_gauss', 'fourier_gauss' (adds '_fourier' to the dir).
    self_data     : if True, infer synthetic data self-predicted from the fiducial loc + true ICs.
    expe          : extra experiment-name suffix for folder2.
    overwrite     : redo phases whose files already exist (else they are loaded/resumed).
    obs_names     : base latents to observe; every other base latent is fixed to its fiducial value.
    pred_thinning : chain thinning for the posterior-predictive meshes (file grows fast).
    """
    import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99' # NOTE: jax preallocates GPU (default 75%)
    import re
    import sys
    from datetime import datetime
    import numpy as np
    from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices
    jconfig.update("jax_enable_x64", True) # 64-bit precision

    from montecosmo.model import FieldLevelModel, default_config
    from montecosmo.utils import h5save
    from montecosmo.script import (field_warmup, plot_field_warmup, full_warmup, full_run,
                                   make_chains, make_logdf_mesh, make_posterior_pred, plot_posterior_pred)

    ############
    # Saving   #
    ############
    # Name schema: register_<tag>_N<mesh_length>_pad<padding>.h5 (see run/register.py).
    stem = Path(register_name).stem
    tag = re.match(r"register_(.+)_N\d+_pad[\d.]+", stem).group(1)
    mesh_length = int(re.search(r"_N(\d+)_pad", stem).group(1))  # cells per side
    fnl = float(m.group(1)) if (m := re.search(r"_fNL(-?\d+)", stem)) else 0.  # from the file name, else 0
    tag = re.sub(r"_fNL-?\d+", "", tag) # remove fNL from tag: folder1 is shared across fNL, folder2 keeps it
    png_suffix = {'fNL': '_fNL', 'fNL_bias': '_fNLb'}.get(png_type, '')
    folder1 = tag + png_suffix
    folder2 = (f"{evolution}_{mesh_length}_fNL{fnl:.0f}"
               + ("_fourier" if lik_type == 'fourier_gauss' else "") + ("_self" if self_data else "")
               + (f"_{expe}" if expe else ""))
    save_dir = PNG_DIR / folder1 / folder2
    chains_dir = save_dir / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSAVE DIR: {save_dir}")
    sys.stdout = sys.stderr = open(save_dir / "run.out", "a", buffering=1)
    print(f"Started running on {os.environ.get('HOSTNAME')} at {datetime.now().astimezone().isoformat()}")
    print(f"Submitted from host {os.environ.get('SLURM_SUBMIT_HOST')} to node(s) {os.environ.get('SLURM_JOB_NODELIST')} as job {os.environ.get('SLURM_JOB_ID')}")
    print(f"SAVE DIR: {save_dir}")
    import shutil, subprocess
    shutil.copy(__file__, save_dir / Path(__file__).name) # snapshot the exact driver next to outputs
    commit = subprocess.run(['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    print(f"montecosmo commit: {commit}")
    print('\n', jdevices())

    jconfig.update("jax_compilation_cache_dir", str(save_dir / "jax_cache/"))
    jconfig.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jconfig.update("jax_persistent_cache_min_compile_time_secs", 10)
    jconfig.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    ###############################################
    # Fiducial location and model (from register) #
    ###############################################
    # Fiducial location of the inferred bias/png/stoch/AP params (cosmology + ngbars come from
    # the register file). Update prior locs too, so the model is centered on these values.

    # pp = 1. if fnl in (-20., 0.) else 1.4 # PNGUnit-XL fNL20 was run with low-passed PNGs, inducing higher p_phi.
    # NOTE: PNGUnit-XL fNL20 was run with low-passed PNGs, inducing higher p_phi.
    PPS = {'LRG': {-20: 0.80, 0: 0.80, 20: 1.13}, 'QSO': {-20: 0.83, 0: 0.83, 20: 1.55}}
    pp = PPS[tracer][fnl]

    fiduc = {
        'b1': 1., 'b2': 0., 'bs2': 0., 'b3': 0., 'bds2': 0., 'bs3': 0., 'bn2': 0., 'bnpar': 0.,
        'fNL': fnl, 'fNL_bp': 0, 'fNL_bpd': 0., 'pp':pp, 'fNL_bpd2': 0., 'fNL_bps2': 0., 'fNL_bn2p': 0.,
        's_e': 1.0, 's_k2e': 0., 's_kmu2e': 0.,
        's_ed': 0., 's_e2': 0., 's_ep': 0.,
        'alpha_iso': 1., 'alpha_ap': 1.,
        }
    latents = FieldLevelModel.new_latents_from_loc(default_config['latents'], fiduc, update_prior=True)
    for name in fiduc:
        latents[name] |= {'scale_fid': latents[name]['scale_fid'] * scale_fid_fac}

    register = REGISTERED_DIR / register_name
    model = FieldLevelModel(**default_config | {
        'evolution': evolution,
        'lik_type': lik_type,
        'png_type': png_type,
        'register': register, # overrides geometry/painting/cosmo/ngbars + loads the meshes
        'latents': latents,
        'n_rbins': 1,
        'evol_oversamp': 2.,
        'ptcl_oversamp': 2.,
        })

    print(model)

    # Fiducial location: model.fiduc (bias/png/stoch from `loc`, cosmo + ngbars from register)
    # plus the true initial field and the observed counts. Optionally self-predict synthetic data.
    if self_data:
        print("\nSelf-predicting synthetic data from the fiducial loc...")
        pred = model.predict(samples=model.fiduc | {'white_mesh': model.white_mesh}, hide_base=False, hide_samp=False, from_base=True)
        model.count_mesh = pred['count_mesh']
        del pred

    model.save(save_dir / "model.yaml")
    print("Setup done.")
    logpdf_fid = model.logpdf(model.reparam(
        model.fiduc | {'white_mesh': model.white_mesh, 'count_mesh': model.count_mesh}, inv=True))
    print("logpdf of fiduc:", logpdf_fid, "\n")
    if jnp.isinf(logpdf_fid) or jnp.isnan(logpdf_fid):
        raise ValueError("fiducial logpdf is infinite or nan")



    #############
    # Inference #
    #############
    params = model.fiduc | {'white_mesh': model.white_mesh} | {'count_mesh': model.count_mesh}
    obs = {k: params[k] for k in obs_names}
    h5save(save_dir / "obs.h5", obs)
    print(f"Inferring: {sorted(set(params) - set(obs))}")
    if n_samples is None:
        n_samples = 128 * 64 // mesh_length
    print(f"n_samples: {n_samples}, n_runs: {n_runs}, n_chains: {n_chains}, tune_mass: {tune_mass}")

    # 1. Field-only warmup (skipped if the init field is observed -> straight to full warmup)
    if 'white_mesh' not in obs:
        state, config, params_start = field_warmup(
            model, chains_dir, n_steps=n_steps_field, desired_energy_var=dev_field,
            n_chains=n_chains, overwrite=overwrite)
        plot_field_warmup(model, params_start, state, save_dir)
    else:
        state = None

    # 2. Full warmup
    state, config = full_warmup(
        model, obs, state, chains_dir, n_steps=n_steps_full, desired_energy_var=dev_full,
        n_chains=n_chains, tune_mass=tune_mass, overwrite=overwrite)

    # 3. Full run
    full_run(model, state, config, chains_dir, n_samples=n_samples, n_runs=n_runs,
             n_chains=n_chains, thinning=thinning, overwrite=overwrite)

    make_chains(save_dir, start=1, end=100)
    # make_logdf_mesh(save_dir, start=1, end=100, thinning=4)
    make_posterior_pred(save_dir, start=1, end=100, thinning=pred_thinning)
    plot_posterior_pred(save_dir)
    print(f"Finished running on {os.environ.get('HOSTNAME')} at {datetime.now().astimezone().isoformat()}")
    # sys.stdout.flush()


if __name__ == '__main__':
    import sys
    print("Demat")

    # FixIC EFT order battery on the pngunitbox HOD mocks, N32.
    #   usage: python infer_eft_fixic.py <order:1|2|3> [<tracers:LRG,QSO>] [<fnls:0,-20>] [<se:sefix|sefree>]
    order = int(sys.argv[1]) if len(sys.argv) > 1 else 3    # 1, 2, or 3
    assert order in [1,2,3]

    budgets = [64] # 32, 48, 64, 96, 128
    tracers = ['QSO'] # 'LRG', 'QSO'
    fnls = [0., -20., 20.] # -20., 0., 20.
    # png_type = 'fNL'
    png_type = 'fNL_bias'
    # lik_type = 'shash'
    lik_type = 'fourier_gauss'
    evolution = 'lpt'
    # evolution = 'kaiser'
    ZS = {'LRG': 0.74, 'QSO': 1.83}


    # Observed (FIXED) parameters; everything else (incl. white_mesh) is inferred.
    obs_names = ['count_mesh',
                #  'white_mesh',         # freeIC vs. fixIC
                'alpha_iso', 'alpha_ap', # AP fixed
                'Omega_m', 'sigma8',     # cosmology fixed
                #  'b1',                 # 1st order bias (sampled)
                #  'b2', 'bs2',          # 2nd order bias (sampled)
                #  'b3', 'bds2', 'bs3',  # 3rd order bias: FIXED for o2, sampled for o3 (added below)
                #  'bn2', 'bnpar',          # higher-derivative bias (any order)
                #  'fNL', 'fNL_bp', 'fNL_bpd', # PNG 1st+2nd order (sampled fNL, fNL_bp, fNL_bpd, if png_type='fNL_bias', else fNL and pp)
                 'pp',
                #  'fNL_bpd2', 'fNL_bps2',      # PNG 3rd order: FIXED for o2, sampled for o3 (added below)
                'fNL_bn2p',              # PNG higher-derivative (any order, degenerated on large scales with b1, on small with bn2)
                's_e',                   # s_e = 1 for shot noise
                's_ed',                  
                's_e2',
                's_ep',
                's_k2e','s_kmu2e'
                #  'ngbars',
                ]
    # Some automatic handling just in case
    obs_names += ['s_ed', 's_e2', 's_ep'] if lik_type == 'fourier_gauss' else ['s_k2e', 's_kmu2e']
    obs_names += ['fNL_bp', 'fNL_bpd'] if png_type == 'fNL' else ['pp'] # if fNL mode, fNL and pp determines fNL_bp and fNL_bpd, otherwise pp is unused
    obs_names += ['fNL', 'fNL_bp', 'fNL_bpd', 'fNL_bpd2', 'fNL_bps2', 'fNL_bn2p'] if png_type is None else [] # no png inference

    if evolution == 'kaiser':
        obs_names += ['bn2', 'bnpar', 'fNL_bn2p'] # no higher-derivative bias
        order = 1
    if order <= 2:  # nth order: fix the <nth-order bias and PNG terms
        obs_names += ['b3', 'bds2', 'bs3', 'fNL_bpd2', 'fNL_bps2']
    if order <= 1:
        obs_names += ['b2', 'bs2', 'fNL_bpd']

    # expe = 'o'+str(order) + '_' + lik_type + '_fixic_ppprior0.3'
    expe = 'o'+str(order) + '_' + lik_type + '_ppunit_s0'
    for tracer in tracers:
        for budget in budgets:
            for fnl in fnls:
                register_name = f'register_pngunitbox_fNL{fnl:.0f}_{tracer}_z{ZS[tracer]}_redshiftspace_N{budget}_pad0.h5'
                infer(register_name=register_name,
                    png_type=png_type, lik_type=lik_type, evolution=evolution,
                    self_data=False, overwrite=False,
                    obs_names=obs_names, expe=expe,
                    scale_fid_fac=1.)

    print("Kenavo")