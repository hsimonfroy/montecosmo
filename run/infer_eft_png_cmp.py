#!/usr/bin/env python
"""
Launcher for the 2nd-vs-3rd-order EFT PNG comparison on fastpm fNL=0 mocks.

Eight jobs = {32^3, 64^3} x {fixed white_mesh, inferred white_mesh} x {2nd order, 3rd order},
run 2 at a time. Each job calls `infer.infer` (run/infer.py, the canonical template, left
untouched) with the obs_names below. Selected by integer arg 0..7; `--dry` prints the config
of every job without running anything.

Held constant across all 8 jobs:
  png_type='fNL_bias', lik_type='quad_gauss', evolution='lpt', self_data=False, fnl=0.
  Fixed cosmology (Omega_m, sigma8, alpha_iso, alpha_ap).
  PNG: with png_type='fNL_bias' the matter-field fNL is a FREE amplitude, so {fNL, fNL_bp,
       fNL_bpd} are ALL sampled (1st+2nd order PNG; their fiducial is 0 since the data is fNL=0).
       (For png_type='fNL' only fNL is sampled and fNL_bp/fNL_bpd follow the universal mass relation.)
  Stoch: s_e fixed=1, s_ed fixed=0, s_e2 SAMPLED; s_k2e/s_kmu2e/s_phi fixed (unused by
         quad_gauss, but prior() samples every latent so they must be observed).
  ngbars left sampled (near-delta prior, as in the template).

Order toggle (higher-derivative ops bn2/bnpar/fNL_bn2p are part of 2nd-order EFT -> always sampled):
  2nd order: infer {b1,b2,bs2,bn2,bnpar, fNL,fNL_bp,fNL_bpd,fNL_bn2p};
             fix   {b3,bds2,bs3, fNL_bpd2,fNL_bps2}.
  3rd order: additionally infer {b3,bds2,bs3, fNL_bpd2,fNL_bps2}.

white_mesh toggle:
  fixed   -> white_mesh observed (field warmup skipped).
  inferred-> white_mesh sampled (full field-level inference).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run/ dir, to import sibling infer.py
from infer import infer

# (mesh budget, white_mesh handling, EFT order)
# NOTE: fastpm's registered white_mesh is a FAKE reference field (correct spectrum, NOT the true
# phases), so the 'fix' (fixed-IC) jobs are only a sanity/compile test on fastpm -- the real
# fixed-IC experiment is run later on abacus redshiftspace (known ICs). Here we ran 32^3 fix as a
# test and DROPPED 64^3 fix; the real fastpm runs are the inferred-white_mesh ('inf') jobs.
JOBS = [
    (32, 'fix', 2),
    (32, 'fix', 3),
    (32, 'inf', 2),
    (32, 'inf', 3),
    (64, 'fix', 2),
    (64, 'fix', 3),
    (64, 'inf', 2),
    (64, 'inf', 3),
]

# Observed (fixed) in every job. Sampled = every latent NOT listed here, i.e.
# {b1,b2,bs2,bn2,bnpar, fNL,fNL_bp,fNL_bpd,fNL_bn2p, s_e2, ngbars} (+ 3rd-order terms / white_mesh per toggles).
OBS_COMMON = [
    'count_mesh',                 # the fastpm data
    'Omega_m', 'sigma8',          # fixed cosmology
    'alpha_iso', 'alpha_ap',      # fixed AP / geometry
    # NB: fNL is NOT fixed -- with png_type='fNL_bias' the matter-field fNL is sampled (fiducial 0).
    's_e', 's_ed',                # s_e fixed=1, s_ed fixed=0 (-> scale1 = s_e const)
    's_k2e', 's_kmu2e',           # fourier-only stoch, unused by quad_gauss
    's_phi',                      # fixed
]
ORDER3_TERMS = ['b3', 'bds2', 'bs3', 'fNL_bpd2', 'fNL_bps2']  # fixed for 2nd order, sampled for 3rd


def build(mesh, white, order):
    register_name = f'register_fastpm_fNL0_z1.000_LRG_b{mesh}_p0.h5'
    obs = list(OBS_COMMON)
    if order == 2:
        obs += ORDER3_TERMS          # fix the 3rd-order bias/png terms
    if white == 'fix':
        obs += ['white_mesh']        # fix the initial field (skip field warmup)
    expe = f"o{order}_{'fixw' if white == 'fix' else 'infw'}"
    return register_name, obs, expe


def run_job(idx):
    mesh, white, order = JOBS[idx]
    register_name, obs, expe = build(mesh, white, order)
    print(f"[job {idx}] mesh={mesh} white={white} order={order} expe={expe}")
    print(f"[job {idx}] register={register_name}")
    print(f"[job {idx}] observed (fixed): {sorted(obs)}")
    infer(register_name=register_name,
          png_type='fNL_bias', lik_type='quad_gauss', evolution='lpt',
          self_data=False, fnl=0., expe=expe, overwrite=False,
          obs_names=obs)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == '--dry':
        ALL = ['Omega_m', 'sigma8', 'b1', 'b2', 'bs2', 'b3', 'bds2', 'bs3', 'bn2', 'bnpar',
               'fNL', 'fNL_bp', 'fNL_bpd', 'fNL_bpd2', 'fNL_bps2', 'fNL_bn2p',
               'alpha_iso', 'alpha_ap', 'ngbars',
               's_e', 's_k2e', 's_kmu2e', 's_ed', 's_e2', 's_phi', 'white_mesh']
        for i, (mesh, white, order) in enumerate(JOBS):
            register_name, obs, expe = build(mesh, white, order)
            sampled = [p for p in ALL if p not in obs]
            print(f"job {i}: mesh={mesh:>2} white={white} order={order}  expe={expe}")
            print(f"    register : {register_name}")
            print(f"    sampled  : {sampled}")
            print(f"    fixed    : {sorted(obs)}\n")
        sys.exit(0)

    idx = int(sys.argv[1])
    run_job(idx)
