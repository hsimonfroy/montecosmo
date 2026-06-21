#!/usr/bin/env python
"""
Check that `nufft` behaves like `rfftn(paint())` on large scales.

`nufft` paints at an oversampled `paint_shape`, deconvolves the painting kernel, and
Fourier-downsamples to `final_shape` (better anti-aliasing). It also applies the
final->paint units jacobian (`mesh *= (paint/final).prod()`), so that, like a plain
paint, the painted count conserves the total: count.sum() == n_tracers, i.e. the mesh
*mean* equals paint's mean (N / final.prod()).

This script paints the same positions two ways and compares power spectra:
  * reference : rfftn(paint(pos, final_shape))          (plain CIC/TSC paint, no oversampling)
  * nufft     : nufft(pos, final_shape, paint_shape)    (for several paint_oversamp)
If the jacobian normalization is right, P_nufft / P_reference ~ 1 on large scales (low k),
deviating only near the Nyquist frequency where anti-aliasing differs. A constant offset at
low k would mean the normalization (sum- vs mean-preservation) is off.

Run on a GPU node with the `montenv` env:
    export XLA_PYTHON_CLIENT_MEM_FRACTION='1.'
    python -u src/montecosmo/tests/check_nufft_paint.py
"""
import os; os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '1.')
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from jax import numpy as jnp, config as jconfig
jconfig.update("jax_enable_x64", True)

from montecosmo.bricks import phys2cell_pos
from montecosmo.nbody import paint, nufft, deconv_paint
from montecosmo.metrics import spectrum
from montecosmo.utils import scale_shape

import montecosmo.run.register as R  # getters (run from the tests/ dir or with it on sys.path)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
final = 64                       # final_shape per axis
paint_oversamps = [1., 1.25, 3/2, 7/4]  # nufft oversampling factors to compare
paint_order, interlace_order = 2, 2
out_path = Path("/pscratch/sd/h/hsimfroy/png/check_nufft_paint.png")

# Positions: a real clustered field (fastpm fNL=0).
spec = R.get_fastpm(fNL=0)
pos, pos_min, pos_max = spec['pos'], spec['pos_min'], spec['pos_max']
N = len(pos)

final_shape = np.full(3, final)
span = pos_max - pos_min
cell_length = span / final
box_size = final_shape * cell_length
geom_center = np.full(3, (pos_min + pos_max) / 2)
box_rot = Rotation.identity()

cell = phys2cell_pos(np.array(pos, float), geom_center, box_rot, box_size, final_shape)

# Reference: plain paint at final_shape (sum == N), window-convolved (not deconvolved).
mesh_ref = np.asarray(paint(cell.copy(), tuple(int(s) for s in final_shape), order=paint_order))
k_ref, p_ref = spectrum(mesh_ref / mesh_ref.mean() - 1, box_size=box_size)  # overdensity P(k)
# Same paint, but deconvolved by the painting kernel (the fair reference for nufft).
mesh_refd = np.asarray(jnp.fft.irfftn(deconv_paint(jnp.fft.rfftn(jnp.asarray(mesh_ref)), paint_order, oversamp=1.)))
k_refd, p_refd = spectrum(mesh_refd / mesh_refd.mean() - 1, box_size=box_size)
print(f"N={N}  reference paint: sum={mesh_ref.sum():.1f}  mean={mesh_ref.mean():.4f}")

# nufft variants.
results = {}
for ov in paint_oversamps:
    paint_shape = scale_shape(final_shape, ov)
    field = nufft(cell.copy(), final_shape, paint_shape, weights=1.,
                  paint_order=paint_order, interlace_order=interlace_order, paint_deconv=True)
    mesh = np.asarray(jnp.fft.irfftn(field))
    k, p = spectrum(mesh / mesh.mean() - 1, box_size=box_size)
    results[ov] = (k, p, mesh)
    print(f"nufft paint_oversamp={ov:.3g} paint_shape={tuple(int(s) for s in paint_shape)}: "
          f"sum={mesh.sum():.1f}  mean={mesh.mean():.4f}  (ref mean {mesh_ref.mean():.4f})")

k_nyq = np.pi * final / box_size[0]

# ---------------------------------------------------------------------------
# Plot: P(k) and ratio to the reference
# ---------------------------------------------------------------------------
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, height_ratios=[2, 1],
                               layout='constrained')
ax0.loglog(k_ref, p_ref, 'k-', label='rfftn(paint)')
ax0.loglog(k_refd, p_refd, 'k--', label='rfftn(paint), deconvolved')
for ov, (k, p, _) in results.items():
    ax0.loglog(k, p, label=f'nufft, paint_oversamp={ov:.3g}')
    ax1.semilogx(k, p / np.interp(k, k_refd, p_refd), label=f'oversamp={ov:.3g}')
for ax in (ax0, ax1):
    ax.axvline(k_nyq, color='gray', ls=':', lw=1)
ax0.set_ylabel(r'$P(k)$  [(Mpc/h)$^3$]'); ax0.legend(); ax0.set_title(f'fastpm fNL0, final={final}^3')
ax1.axhline(1., color='k', lw=0.8)
ax1.set_ylim(0.9, 1.1); ax1.set_ylabel(r'$P_\mathrm{nufft}/P_\mathrm{paint,deconv}$')
ax1.set_xlabel(r'$k$  [h/Mpc]  (dotted = $k_\mathrm{Nyq}$)')
fig.savefig(out_path, dpi=130)
print(f"saved {out_path}")

# Quick numeric verdict.
print(f"k_fund={2*np.pi/box_size[0]:.4f}  k_nyq={k_nyq:.4f} h/Mpc")
for ov, (k, p, _) in results.items():
    print(f"oversamp={ov:.3g}: P_nufft/P_rawpaint    at 6 lowest k = {np.round((p/np.interp(k,k_ref,p_ref))[:6], 4)}")
    print(f"oversamp={ov:.3g}: P_nufft/P_deconvpaint at 6 lowest k = {np.round((p/np.interp(k,k_refd,p_refd))[:6], 4)}")
