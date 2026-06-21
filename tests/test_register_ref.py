#!/usr/bin/env python
"""
Cross-check the register pipeline against trusted reference products.

(1) powtranscoh of the fastpm fNL0 count field vs the reference
    tracer_fNL0_paint2_deconv1_64.npy (painted at 576 with paint_order=2, deconv, then
    Fourier-downsampled to 64 -- see abacusing.ipynb), for paint_oversamp in {7/4, 2}.
    The reference is normalized counts (renormalized by ngbars*cell_length**3, i.e. mean ~ 1);
    register count_mesh has count.sum()==n_tracers, so both are compared as overdensity
    delta = field/field.mean() - 1 (the renormalization cancels). Expect transfer & coherence
    ~ 1 at low k, and paint_oversamp=2 to give no improvement over 7/4.

(2) init_kpow: register's compute_init_kpow(abacus_cosmo(0)) must match the on-disk
    fpm .../load/init_kpow.npy.

Run on a GPU node with the `montenv` env:
    export XLA_PYTHON_CLIENT_MEM_FRACTION='.9'
    python -u src/montecosmo/tests/test_register_ref.py
"""
import os, sys; os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.9')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'run'))  # register.py lives in run/
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jax import numpy as jnp, config as jconfig
jconfig.update("jax_enable_x64", True)

from montecosmo.model import FieldLevelModel
from montecosmo.bricks import get_cosmology
from montecosmo.metrics import powtranscoh, spectrum
from montecosmo.plot import plot_powtranscoh
import register as R

FPM_DIR = Path("/pscratch/sd/h/hsimfroy/png/fpm_b2760_z1_lrg_fNL/load")
OUT = Path("/pscratch/sd/h/hsimfroy/png/test_register_ref.png")
MESH = 64
OVERSAMPS = [7/4, 2.]


def register_count(spec, cell_budget, paint_oversamp):
    cosmo_jax = get_cosmology(**R.cosmo2loc(spec['cosmo_fid']))
    obs = FieldLevelModel.register_catalog(
        cell_budget, cosmo_jax, spec['data'], random=spec.get('random'),
        box_size=spec.get('box_size'), box_center=spec.get('box_center'),
        a_obs=spec.get('a_obs'), padding=0.,
        init_oversamp=R.INIT_OVERSAMP, paint_oversamp=paint_oversamp, **R.PAINT)
    return np.asarray(obs['count_mesh'])


# ---------------------------------------------------------------------------
# (2) init_kpow match
# ---------------------------------------------------------------------------
old_kpow = np.load(FPM_DIR / "init_kpow.npy")
new_kpow = R.compute_init_kpow(R.abacus_cosmo(0))
k_reldiff = float(np.abs(old_kpow[0] / new_kpow[0] - 1).max())
p_reldiff = float(np.abs(old_kpow[1] / new_kpow[1] - 1).max())
print(f"(2) init_kpow  k max reldiff = {k_reldiff:.2e}   P max reldiff = {p_reldiff:.2e}"
      f"   -> {'MATCH' if max(k_reldiff, p_reldiff) < 1e-10 else 'MISMATCH'}")

# ---------------------------------------------------------------------------
# (1) powtranscoh vs reference
# ---------------------------------------------------------------------------
box = np.full(3, R.FASTPM_BOX)
ref = np.asarray(jnp.load(FPM_DIR / f"tracer_fNL0_paint2_deconv1_{MESH}.npy"))
ref_delta = ref / ref.mean() - 1
print(f"(1) reference {ref.shape}  mean={ref.mean():.6f}")

spec = R.get_fastpm(fNL=0)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
kref, pref = spectrum(ref_delta, box_size=box)
axes[0].loglog(kref, pref, 'k--', label='reference (576->64)')
for ov in OVERSAMPS:
    reg_delta = register_count(spec, MESH**3, ov)
    reg_delta = reg_delta / reg_delta.mean() - 1
    k, pow1, trans, coh = powtranscoh(ref_delta, reg_delta, box_size=box)
    plot_powtranscoh(k, pow1, trans, coh, log=True, axes=axes, label=f'paint_oversamp={ov:.3g}')
    sel = k < 0.2
    print(f"  oversamp={ov:.3g}: transfer[k<0.2] in [{trans[sel].min():.4f},{trans[sel].max():.4f}]  "
          f"coherence[k<0.2] in [{coh[sel].min():.4f},{coh[sel].max():.4f}]")
for ax in axes:
    ax.axvline(np.pi * MESH / box[0], color='gray', ls=':', lw=1)  # k_nyquist
    ax.legend(fontsize=8)
axes[0].set_title(f'fastpm fNL0 {MESH}^3 vs reference')
fig.savefig(OUT, dpi=130)
print(f"saved {OUT}")
