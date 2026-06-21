#!/usr/bin/env python
"""
Verify that the refactored register pipeline reproduces the previously saved obs fields.

The saved fields are normalized counts (mean ~ 1, i.e. count / count_fid). register_catalog
returns count_mesh with count.sum() == n_tracers, so the comparable quantity is
count_mesh / count_mesh.mean(). We report n_tracers, the field correlation (alignment),
and the power-spectrum transfer + coherence (origin-independent).
"""
import os; os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.9')
from pathlib import Path
import numpy as np
from jax import numpy as jnp, config as jconfig
jconfig.update("jax_enable_x64", True)

from montecosmo.model import FieldLevelModel
from montecosmo.bricks import get_cosmology
from montecosmo.metrics import spectrum
import montecosmo.run.register as R

FPM_DIR = Path("/pscratch/sd/h/hsimfroy/png/fpm_b2760_z1_lrg_fNL/load")
ABA_DIR = Path("/pscratch/sd/h/hsimfroy/png/abacus_c0_i0_z0.8_lrg/load")


def register_obs(spec, cell_budget):
    cosmo_jax = get_cosmology(**R.cosmo2loc(spec['cosmo_fid']))
    obs = FieldLevelModel.register_catalog(
        cell_budget, cosmo_jax, spec['data'], random=spec.get('random'),
        box_size=spec.get('box_size'), box_center=spec.get('box_center'),
        a_obs=spec.get('a_obs'), padding=0.,
        init_oversamp=R.INIT_OVERSAMP, paint_oversamp=R.PAINT_OVERSAMP, **R.PAINT)
    return obs


def compare(name, count_mesh, saved_path, box_size):
    s = np.asarray(jnp.load(saved_path))
    c = np.asarray(count_mesh)
    cn = c / c.mean()                 # normalized counts, mean 1 (like saved)
    box = np.full(3, box_size)
    k, pc = spectrum(cn - 1, box_size=box, box_center=(0., 0., 1.))
    k, ps = spectrum(s - 1, box_size=box, box_center=(0., 0., 1.))
    k, px = spectrum(cn - 1, s - 1, box_size=box, box_center=(0., 0., 1.))
    pc, ps, px = np.asarray(pc), np.asarray(ps), np.asarray(px)
    transfer = (pc / ps)**.5
    coh = px / (pc * ps)**.5
    corr = np.corrcoef(cn.ravel(), s.ravel())[0, 1]
    print(f"\n{name}  shape={c.shape}")
    print(f"  saved mean={s.mean():.4f}  reg count.sum={c.sum():.0f}  reg/saved mean ratio={cn.mean()/s.mean():.4f}")
    print(f"  field corrcoef = {corr:.5f}")
    print(f"  transfer  (reg/saved)^.5 [6 low-k] = {np.round(transfer[:6], 4)}")
    print(f"  coherence (cross)        [6 low-k] = {np.round(coh[:6], 4)}")
    print(f"  transfer  [6 high-k]               = {np.round(transfer[-6:], 4)}")


if __name__ == "__main__":
    for mesh in [64, 96]:
        b = mesh**3
        # fastpm fNL0 (n_tracers 2099282), box 2760
        obs = register_obs(R.get_fastpm(fNL=0), b)
        compare(f"fastpm fNL0 mesh={mesh}", obs['count_mesh'],
                FPM_DIR / f"tracer_2099282_fNL0_paint2_deconv1_{mesh}.npy", R.FASTPM_BOX)

    for mesh in [96]:
        b = mesh**3
        # abacus realspace (n_tracers 6746545), box 2000
        obs = register_obs(R.get_abacus(field='realspace'), b)
        compare(f"abacus realspace mesh={mesh}", obs['count_mesh'],
                ABA_DIR / f"tracer_6746545_paint2_deconv1_{mesh}.npy", R.ABACUS_BOX)
        # abacus redshiftspace flat (n_tracers 6746545), box 2000
        obs = register_obs(R.get_abacus(field='redshiftspace'), b)
        compare(f"abacus redshiftspace mesh={mesh}", obs['count_mesh'],
                ABA_DIR / f"tracer_6746545_rsdflat_paint2_deconv1_{mesh}.npy", R.ABACUS_BOX)
