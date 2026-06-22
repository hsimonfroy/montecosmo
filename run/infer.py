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
     fiducial cosmology + ngbars), optionally self-predict synthetic data, and save loc_fid.h5.
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
          n_chains=4, tune_mass=True,
          n_steps_field=2**12, dev_field=1e-5,
          n_steps_full=2**13, dev_full=1e-7,
          n_samples=None, n_runs=8, thinning=64):
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
    """
    import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.' # NOTE: jax preallocates GPU (default 75%)
    import re
    import sys
    from datetime import datetime
    import numpy as np
    from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices
    jconfig.update("jax_enable_x64", True) # 64-bit precision

    from montecosmo.model import FieldLevelModel, default_config
    sys.path.insert(0, str(Path(__file__).resolve().parent)) # run/ dir, to import sibling script.py
    from script import (field_warmup, plot_field_warmup, full_warmup, full_run, make_chains)

    fourier = (lik_type == 'fourier_gauss')

    ###############################################
    # Fiducial location and model (from register) #
    ###############################################
    # Fiducial location of the inferred bias/png/stoch/AP params (cosmology + ngbars come from
    # the register file). Update prior locs too, so the model is centered on these values.
    loc = {
        'b1': 0.7, 'b2': 0., 'bs2': 0., 'b3': 0., 'bds2': 0., 'bs3': 0., 'bn2': 0., 'bnpar': 0.,
        'fNL': fnl, 'fNL_bp': fnl, 'fNL_bpd': 0., 'fNL_bpd2': 0., 'fNL_bps2': 0., 'fNL_bn2p': 0.,
        's_e': 1., 's_k2e': 0., 's_kmu2e': 0., 's_ed': 0.1, 's_e2': 0.1,
        'alpha_iso': 1., 'alpha_ap': 1.,
        }
    latents = FieldLevelModel.new_latents_from_loc(default_config['latents'], loc, update_prior=True)

    register = REGISTERED_DIR / register_name
    model = FieldLevelModel(**default_config | {
        'evolution': evolution,
        'lik_type': lik_type,
        'png_type': png_type,
        'register': register, # overrides geometry/painting/cosmo/ngbars + loads the meshes
        'latents': latents,
        })

    ############
    # Saving   #
    ############
    mesh_length = np.round(np.prod(model.final_shape)**(1/3), dtype=int)
    tag = re.match(r"register_(.+)_b\d+_p[\d.]+", Path(register_name).stem).group(1)
    png_suffix = {'fNL': '_fNL', 'fNL_bias': '_fNLb'}.get(png_type, '')
    folder1 = tag + png_suffix
    folder2 = (f"{evolution}_{mesh_length}_fNL{fnl:.0f}"
               + ("_fourier" if fourier else "") + ("_self" if self_data else "")
               + (f"_{expe}" if expe else ""))
    save_dir = PNG_DIR / folder1 / folder2
    chains_dir = save_dir / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)

    print(f"SAVE DIR: {save_dir}")
    sys.stdout = sys.stderr = open(save_dir / "run.out", "a")
    print(f"Started running on {os.environ.get('HOSTNAME')} at {datetime.now().astimezone().isoformat()}")
    print(f"Submitted from host {os.environ.get('SLURM_SUBMIT_HOST')} to node(s) {os.environ.get('SLURM_JOB_NODELIST')}")
    print(f"Job id: {os.environ.get('SLURM_JOB_ID')}")
    print(f"SAVE DIR: {save_dir}")
    print('\n', jdevices())

    jconfig.update("jax_compilation_cache_dir", str(save_dir / "jax_cache/"))
    jconfig.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jconfig.update("jax_persistent_cache_min_compile_time_secs", 10)
    jconfig.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    print(model)

    # Fiducial location: model.loc_fid (bias/png/stoch from `loc`, cosmo + ngbars from register)
    # plus the true initial field and the observed counts. Optionally self-predict synthetic data.
    loc_fid = dict(model.loc_fid)
    if self_data:
        print("\nSelf-predicting synthetic data from the fiducial loc and true ICs...")
        pred = model.predict(samples=loc_fid | {'init_mesh': model.init_mesh},
                             hide_base=False, hide_samp=False, from_base=True)
        count_mesh = pred['obs']
    else:
        count_mesh = model.count_mesh
    if model.init_mesh is not None:
        loc_fid['init_mesh'] = model.init_mesh
    loc_fid['count_mesh'] = count_mesh

    from montecosmo.utils import h5save
    model.save(save_dir / "model.yaml")
    h5save(save_dir / "loc_fid.h5", loc_fid)
    print("Setup done.")
    sys.stdout.flush()

    #############
    # Inference #
    #############
    # Observed (fixed) params: the others are inferred. `init_mesh` is always inferred.
    obs_names = ['alpha_iso', 'alpha_ap',                              # AP fixed
                 'b2', 'bs2', 'b3', 'bds2', 'bs3', 'bn2', 'bnpar',     # higher-order bias fixed
                 's_k2e', 's_kmu2e'] + (['s_ed', 's_e2'] if fourier else ['s_e'])
    obs_names += ['fNL_bpd2', 'fNL_bps2', 'fNL_bn2p']                  # higher-order png fixed
    if png_type == 'fNL':           # universal mass relation: infer fNL, fix fNL_bp, fNL_bpd
        obs_names += ['fNL_bp', 'fNL_bpd']
    elif png_type == 'fNL_bias':    # infer fNL_bp, fNL_bpd, fix fNL
        obs_names += ['fNL']
    else:                           # no png inference
        obs_names += ['fNL', 'fNL_bp', 'fNL_bpd']
    obs = {k: loc_fid[k] for k in obs_names if k in loc_fid}
    obs['obs'] = count_mesh
    if n_samples is None:
        n_samples = 128 * 64 // mesh_length
    print(f"Inferring: {sorted(set(model.loc_fid) - set(obs))} + init_mesh")
    print(f"n_samples: {n_samples}, n_runs: {n_runs}, n_chains: {n_chains}, tune_mass: {tune_mass}")

    # 1. Field-only warmup
    state, config, params_start = field_warmup(
        model, loc_fid, chains_dir, n_steps=n_steps_field, desired_energy_var=dev_field,
        n_chains=n_chains, overwrite=overwrite)
    plot_field_warmup(model, loc_fid, params_start, state, save_dir)
    sys.stdout.flush()

    # 2. Full warmup
    state, config = full_warmup(
        model, loc_fid, obs, state, chains_dir, n_steps=n_steps_full, desired_energy_var=dev_full,
        n_chains=n_chains, tune_mass=tune_mass, overwrite=overwrite)
    sys.stdout.flush()

    # 3. Full run
    full_run(model, state, config, chains_dir, n_samples=n_samples, n_runs=n_runs,
             n_chains=n_chains, thinning=thinning, overwrite=overwrite)
    sys.stdout.flush()

    make_chains(save_dir, start=1, end=100)
    print(f"Finished running on {os.environ.get('HOSTNAME')} at {datetime.now().astimezone().isoformat()}")
    sys.stdout.flush()


if __name__ == '__main__':
    print("Demat")
    # infer = tm.python_app(infer) # uncomment to submit with desipipe

    infer(register_name='register_abacus_c0_ph0_z0.800_LRG_redshiftspace_b32_p0.h5',
          png_type='fNL', self_data=True, overwrite=False)

    # spawn(queue, spawn=True) # uncomment to submit with desipipe
    print("Kenavo")
