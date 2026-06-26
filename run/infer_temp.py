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
     fiducial cosmology + ngbars), optionally self-predict synthetic data.
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
          self_data=False, fnl=0., expe='', overwrite=False,
          obs_names=[],
          n_chains=4, tune_mass=True,
          n_steps_field=2**12, dev_field=1e-5,
          n_steps_full=2**13, dev_full=1e-7,
          n_samples=None, n_runs=8, thinning=64,
          scale_fid_fac=None):
    """
    Run inference for the mock registered in `<REGISTERED_DIR>/<register_name>`.

    register_name : register HDF5 filename, e.g. 'register_abacus_c0_ph0_z0.800_LRG_redshiftspace_b32_p0.h5'.
    png_type      : None, 'fNL' (universal mass relation -> infer fNL, fix fNL_bp/fNL_bpd),
                    or 'fNL_bias' (infer fNL_bp/fNL_bpd, fix fNL).
    lik_type      : likelihood, e.g. 'shash', 'quad_gauss', 'fourier_gauss' (adds '_fourier' to the dir).
    self_data     : if True, infer synthetic data self-predicted from the fiducial loc + true ICs.
    fnl           : fiducial f_NL location (and folder2 label).
    expe          : extra experiment-name suffix for folder2.
    overwrite     : redo phases whose files already exist (else they are loaded/resumed).
    obs_names     : base latents to observe; every other base latent is fixed to its fiducial value.
    """
    import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.' # NOTE: jax preallocates GPU (default 75%)
    import re
    import sys
    from datetime import datetime
    import numpy as np
    from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices
    jconfig.update("jax_enable_x64", True) # 64-bit precision

    from montecosmo.model import FieldLevelModel, default_config
    from montecosmo.utils import h5save
    sys.path.insert(0, str(Path(__file__).resolve().parent)) # run/ dir, to import sibling script.py
    from script import (field_warmup, plot_field_warmup, full_warmup, full_run, make_chains, make_logdf_mesh)

    ###############################################
    # Fiducial location and model (from register) #
    ###############################################
    # Fiducial location of the inferred bias/png/stoch/AP params (cosmology + ngbars come from
    # the register file). Update prior locs too, so the model is centered on these values.
    fiduc = {
        'b1': 0.7, 'b2': 0., 'bs2': 0., 'b3': 0., 'bds2': 0., 'bs3': 0., 'bn2': 0., 'bnpar': 0.,
        'fNL': fnl, 'fNL_bp': fnl, 'fNL_bpd': 0., 'fNL_bpd2': 0., 'fNL_bps2': 0., 'fNL_bn2p': 0.,
        's_e': 1., 's_k2e': 0., 's_kmu2e': 0.,
        's_ed': 0., 's_e2': 0.,
        'alpha_iso': 1., 'alpha_ap': 1.,
        }
    latents = FieldLevelModel.new_latents_from_loc(default_config['latents'], fiduc, update_prior=True)
    # Enlarge fiducial (posterior) scale of given latents, e.g. {'b1': 30, 'fNL_bp': 30} when
    # constraints are weak, so the warmup mass matrix / step size is not under-dispersed.
    for nm, fac in (scale_fid_fac or {}).items():
        latents[nm] = latents[nm] | {'scale_fid': latents[nm]['scale_fid'] * fac}

    register = REGISTERED_DIR / register_name
    model = FieldLevelModel(**default_config | {
        'evolution': evolution,
        'lik_type': lik_type,
        'png_type': png_type,
        'register': register, # overrides geometry/painting/cosmo/ngbars + loads the meshes
        'latents': latents,
        'n_rbins': 1,
        })

    ############
    # Saving   #
    ############
    mesh_length = int(round(np.prod(model.final_shape)**(1/3)))
    tag = re.match(r"register_(.+)_b\d+_p[\d.]+", Path(register_name).stem).group(1)
    png_suffix = {'fNL': '_fNL', 'fNL_bias': '_fNLb'}.get(png_type, '')
    folder1 = tag + png_suffix
    folder2 = (f"{evolution}_{mesh_length}_fNL{fnl:.0f}"
               + ("_fourier" if lik_type == 'fourier_gauss' else "") + ("_self" if self_data else "")
               + (f"_{expe}" if expe else ""))
    save_dir = PNG_DIR / folder1 / folder2
    chains_dir = save_dir / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)

    print(f"SAVE DIR: {save_dir}")
    sys.stdout = sys.stderr = open(save_dir / "run.out", "a", buffering=1)
    print(f"Started running on {os.environ.get('HOSTNAME')} at {datetime.now().astimezone().isoformat()}")
    print(f"Submitted from host {os.environ.get('SLURM_SUBMIT_HOST')} to node(s) {os.environ.get('SLURM_JOB_NODELIST')}")
    print(f"Job id: {os.environ.get('SLURM_JOB_ID')}")
    print(f"SAVE DIR: {save_dir}")
    print('\n', jdevices())

    import shutil, subprocess
    shutil.copy(__file__, save_dir / Path(__file__).name) # snapshot the exact driver next to outputs
    commit = subprocess.run(['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    print(f"montecosmo commit: {commit}")

    jconfig.update("jax_compilation_cache_dir", str(save_dir / "jax_cache/"))
    jconfig.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jconfig.update("jax_persistent_cache_min_compile_time_secs", 10)
    jconfig.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

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
    sys.stdout.flush()



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
    sys.stdout.flush()

    # 1. Field-only warmup (skipped if the init field is observed -> straight to full warmup)
    if 'white_mesh' not in obs:
        state, config, params_start = field_warmup(
            model, chains_dir, n_steps=n_steps_field, desired_energy_var=dev_field,
            n_chains=n_chains, overwrite=overwrite)
        plot_field_warmup(model, params_start, state, save_dir)
    else:
        state = None
    sys.stdout.flush()

    # 2. Full warmup
    state, config = full_warmup(
        model, obs, state, chains_dir, n_steps=n_steps_full, desired_energy_var=dev_full,
        n_chains=n_chains, tune_mass=tune_mass, overwrite=overwrite)
    sys.stdout.flush()

    # 3. Full run
    full_run(model, state, config, chains_dir, n_samples=n_samples, n_runs=n_runs,
             n_chains=n_chains, thinning=thinning, overwrite=overwrite)
    sys.stdout.flush()

    make_chains(save_dir, start=1, end=100)
    # Per-voxel logpdf/logcdf of the counts over the chains -> chains/logdf_mesh.h5 (thinned).
    make_logdf_mesh(save_dir, start=1, end=100, thinning=64)
    print(f"Finished running on {os.environ.get('HOSTNAME')} at {datetime.now().astimezone().isoformat()}")
    sys.stdout.flush()


if __name__ == '__main__':
    import sys
    print("Demat")
    # infer = tm.python_app(infer) # uncomment to submit with desipipe

    # Dispatch the two kaiser-32 fNL_bias experiments (cosmology fixed):
    #   'self'    : infer b1, fNL_bp, white_mesh        (expe='')
    #   'fixedic' : infer b1, fNL_bp only, fix white_mesh (expe='fixedic')
    which = sys.argv[1] if len(sys.argv) > 1 else 'self'
    if which not in ('self', 'fixedic'):
        raise SystemExit(f"usage: python infer_temp.py [self|fixedic]; got {which!r}")

    png_type = 'fNL_bias'   # fNL_bp/fNL_bpd are free biases (fNL fixed)
    lik_type = 'quad_gauss'

    # Infer ONLY these (+ white_mesh in 'self'); observe everything else (cosmology fixed).
    infer_names = {'b1', 'fNL_bp'}

    # Full set of base latents for this config; observe the complement of `infer_names` so
    # nothing leaks into the inferred set by omission.
    ALL_BASE = ['Omega_m', 'sigma8',                                  # cosmo (fixed)
                'b1', 'b2', 'bs2', 'b3', 'bds2', 'bs3', 'bn2', 'bnpar', # bias
                'fNL', 'fNL_bp', 'fNL_bpd', 'fNL_bpd2', 'fNL_bps2', 'fNL_bn2p', # png
                'ngbars',                                            # syst
                's_e', 's_e2', 's_ed', 's_k2e', 's_kmu2e', 's_phi',  # stoch
                'alpha_iso', 'alpha_ap']                             # AP
    obs_names = ['count_mesh'] + [n for n in ALL_BASE if n not in infer_names]

    expe = 'ov0.5' if which == 'self' else 'fixedic_ov0.5'  # init_oversamp=1/2 (16^3 low-res ICs)
    if which == 'fixedic':
        obs_names += ['white_mesh'] # also fix the initial field

    obs_names = sorted(set(obs_names))
    inferred = sorted(set(ALL_BASE + ['white_mesh']) - set(obs_names))
    print(f"Run '{which}' (expe={expe!r}); inferring {inferred}")

    # abacus c0 ph0 z0.800 LRG, 32^3, redshift-space, self-predicted data, kaiser evolution,
    # quad_gauss lik, fNL_bias PNG, cosmology fixed, init_oversamp=1/2 (16^3 ICs < 32^3 counts, so
    # fewer inferred init values than observations -- for the LOO-PSIS test). Infer b1, fNL_bp (+ white_mesh).
    # Constraints are weak here -> widen the fiducial scale of b1 and fNL_bp by 30x.
    infer(register_name='register_abacus_c0_ph0_z0.800_LRG_redshiftspace_b32_p0_ov0.5.h5',
        png_type=png_type, lik_type=lik_type, evolution='kaiser',
        self_data=True, fnl=0., overwrite=False,
        obs_names=obs_names, expe=expe,
        scale_fid_fac={'b1': 30, 'fNL_bp': 30})

    # spawn(queue, spawn=True) # uncomment to submit with desipipe
    print("Kenavo")
