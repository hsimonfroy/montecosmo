#!/usr/bin/env python
"""
Download (read), paint, and save PNGUNITXL mock galaxy meshes for field-level inference.

PNGUNITXL is the "PNG-UNIT" simulation with local Primordial Non-Gaussianity (PNG).
The true f_NL is encoded in the directory name (no header field):
    fnl0 -> f_NL = 0,   fnl20 -> f_NL = +20,   fnlm20 -> f_NL = -20.

Catalogs are FITS binary tables with *sky* coordinates (RA, DEC in deg, Z redshift),
split by galactic cap (NGC / SGC), with a separate randoms catalog encoding the survey
selection. Cosmology is Planck 2015 (Table 4, last column).

For each cell budget (32^3, 64^3, 128^3), `FieldLevelModel.register_meshes` derives the box
geometry from the randoms (+ padding), paints the selection, mask and count meshes, computes
ngbars, and saves them into a single file (loadable via the model's `meshes` config). Mesh
shapes are box-proportional (isotropic cells). Two diagnostic figures are produced:
  * plot_meshes : rows = selection / mask / count, cols = the 3 budgets.
  * plot_delta  : rows = synthetic Gaussian IC (@ init_shape) / count2delta field (@ final_shape)
                  / their power spectra vs a Kaiser monopole + effective shot noise.

Run on a GPU node with the `montenv` env:
    export XLA_PYTHON_CLIENT_MEM_FRACTION='1.'
    python -u src/montecosmo/montecosmo/script_get_sims.py
"""
import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.'  # NOTE: jax preallocates GPU
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')  # headless compute node
import matplotlib.pyplot as plt

from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices
jconfig.update("jax_enable_x64", True)  # 64-bit precision
print(jdevices())

import fitsio
from montecosmo.model import FieldLevelModel, default_config
from montecosmo.bricks import Planck18, lin_power_mesh, lin_power_interp
from montecosmo.utils import chreshape, r2chshape
from montecosmo.metrics import kaiser_formula
from montecosmo.plot import plot_mesh, plot_pow


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
tracer, region, fnl = 'LRG', 'NGC', 'fnl0'
fNL_true = {'fnl0': 0., 'fnl20': 20., 'fnlm20': -20.}[fnl]
b1E = 2.0          # fiducial Eulerian linear bias for LRG (Kaiser-model reference)
seed = 0           # RNG seed for the synthetic Gaussian initial field
# padding = 0.5      # padded box fraction around the footprint
padding = 0.      # padded box fraction around the footprint
budgets = [128, 64, 32]  # cell budgets (~x^3 total cells, box-proportional shapes)
log = False        # plot_pow in k*P(k) (False) or loglog P(k) (True)

cat_dir = Path("/global/cfs/cdirs/desicollab/users/adrigut/PNGxHOD"
               "/dev_mocks/catalogs/DA2/v2.0/PNGUNITXL")
data_path = cat_dir / "complete" / fnl / f"{tracer}_complete_{region}_clustering.dat.fits"
rand_path = cat_dir / "randoms" / f"{tracer}_complete_{region}_0_clustering.ran.fits"

save_dir = Path("/pscratch/sd/h/hsimfroy/png/unitsim")
load_dir = save_dir / "load"
save_dir.mkdir(parents=True, exist_ok=True)
load_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load catalogs (field-level -> only WEIGHT, no FKP weights)
# ---------------------------------------------------------------------------
cols = ['RA', 'DEC', 'Z', 'WEIGHT']
print(f"Reading data    {data_path}")
data = fitsio.read(str(data_path), columns=cols)
print(f"Reading randoms {rand_path}")
rand = fitsio.read(str(rand_path), columns=cols)
print(f"n_data = {len(data):,}   n_rand = {len(rand):,}")
w_data, w_rand = data['WEIGHT'].astype(float), rand['WEIGHT'].astype(float)
sw2_data, sw2_rand = float((w_data**2).sum()), float((w_rand**2).sum())  # Poisson shot amplitudes
alpha_fkp = float(w_data.sum() / w_rand.sum())  # data/randoms weight ratio (FKP alpha)


# ---------------------------------------------------------------------------
# Model with fiducial = mock cosmology (Planck15). register_meshes will override the
# geometry per budget; cosmo_fid/a_fid follow from the latents loc and the footprint.
# ---------------------------------------------------------------------------
model = FieldLevelModel(**default_config | {
    'box_center': (0., 0., 2000.),  # placeholder, set by register_meshes
    'evolution': 'lpt',
    'a_obs': None,                  # light-cone
    'curved_sky': True,
    'png_type': None,
    'meshes': None,                 # set per budget by register_meshes
    'paint_order': 2,
    'paint_deconv': True,
    'interlace_order': 2,
    'init_oversamp': 1.,
    'evol_oversamp': 2.,
    'lik_type': 'gaussian_delta',
    })
latents = model.new_latents_from_loc({'Omega_m': 0.3075, 'sigma8': 0.8159}, update_prior=True)
model = FieldLevelModel(**model.asdict() | {'latents': latents})

# Theory curves use the *mock* cosmology (Planck15). model.cosmo_fid = get_cosmology(Om, s8)
# keeps AbacusSummit0's h/n_s/Omega_b, which differs from Planck15 by ~3% in P(k) shape.
cosmo_theory = Planck18()


def eff_shot_noise(count_mesh):
    """
    Effective (FKP) shot noise P_shot [(Mpc/h)^3] of count2delta(count_mesh, selec_mesh):
        P_shot = (sum_data w^2 + alpha_fkp^2 sum_rand w^2) / A,   A = int n_bar^2 dx = N^2 D^2 / V,
    with D the count2delta normalization and N the final cell count. The randoms term uses the
    *FKP* alpha = (sum_data w)/(sum_rand w): in (count - alpha_c2d*selec), the selec unit-mean
    normalization 1/m exactly turns count2delta's alpha_c2d into alpha_fkp for the noise. Uses the
    same downsized/masked selection as model.count2delta, so it matches the measured delta plateau.
    """
    selec = model.selec_mesh
    if np.ndim(selec) == 3 and selec.shape != count_mesh.shape:
        selec = jnp.fft.irfftn(chreshape(jnp.fft.rfftn(selec), r2chshape(count_mesh.shape)))
        selec = model.masked2mesh(model.mesh2masked(selec))
    alpha = count_mesh.mean() / selec.mean()
    D2 = ((alpha * selec)**2).mean()
    N, V = count_mesh.size, float(np.prod(model.box_size))
    return float((sw2_data + alpha_fkp**2 * sw2_rand) * V / (N**2 * D2))


def kaiser_ref(ks):
    """Kaiser redshift-space galaxy monopole P_0(k) at the model's a_fid (bias b1E, RSD)."""
    pk_lin0 = lin_power_interp(cosmo_theory, a=1.)(ks)  # z=0 linear; kaiser applies growth^2
    _, pk = kaiser_formula(cosmo_theory, model.a_fid, (ks, pk_lin0), b1E, ells=0)
    return pk[0]


# ---------------------------------------------------------------------------
# Register meshes per budget + collect everything for plotting (computed while the
# model is configured at that budget).
# ---------------------------------------------------------------------------
selecs, masks, counts, deltas, synths = {}, {}, {}, {}, {}
kdeltas, ksynths, krefs, box_sizes, ngbars = {}, {}, {}, {}, {}
for x in budgets:
    meshes_path = load_dir / f"meshes_{tracer}_{region}_{fnl}_{x}.npz"
    model.register_meshes(data, rand, meshes_path, cell_budget=x**3, padding=padding)
    box_sizes[x] = np.asarray(model.box_size)

    selecs[x] = np.asarray(model.selec_mesh)          # @ evol_shape, unit mean over footprint
    masks[x] = np.asarray(model.mask_mesh)            # @ final_shape (bool)
    counts[x] = np.asarray(model.count_mesh)          # @ final_shape (counts per cell)
    deltas[x] = model.count2delta(model.count_mesh, from_masked=False)  # @ final_shape

    # Synthetic Gaussian linear field at init_shape (a "before-PNG" reference; random phases,
    # NOT the sim's actual ICs -- only its linear P(k) is physical), saved per budget.
    pmesh = lin_power_mesh(cosmo_theory, model.init_shape, model.box_size, a=model.a_fid)
    synths[x] = jnp.fft.irfftn(jnp.fft.rfftn(jr.normal(jr.key(seed),
                               tuple(map(int, model.init_shape)))) * pmesh**0.5)
    np.save(load_dir / f"synth_{tracer}_{region}_{fnl}_{x}.npy", np.asarray(synths[x]))

    # Spectra + shot-noise-corrected Kaiser reference (computed at this budget's geometry).
    kdeltas[x] = model.spectrum(deltas[x])
    ksynths[x] = model.spectrum(synths[x])
    P_shot = eff_shot_noise(model.count_mesh)
    krefs[x] = (kdeltas[x][0], np.asarray(kaiser_ref(kdeltas[x][0]) + P_shot), P_shot)

    ngbars[x] = float(np.mean(model.latents['ngbars']['loc']))  # loc is per-radial-bin -> mean
    print(f"~{x}^3: final={(model.final_shape)} init={(model.init_shape)} "
          f"evol={tuple(model.evol_shape)} cell={model.cell_length:.1f} Mpc/h "
          f"a_fid={model.a_fid:.3f} (z={1/model.a_fid - 1:.2f}) ngbars={ngbars[x]:.3e} P_shot={P_shot:.1f}")
    print(f"saved {meshes_path.name} + synth_{tracer}_{region}_{fnl}_{x}.npy")


# ---------------------------------------------------------------------------
# Plot meshes: rows = selection / mask / count, cols = budgets.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 13), layout='constrained')
fig.suptitle(f"{tracer} {region} {fnl} (f_NL={fNL_true:g}) -- selection / mask / count")
rows = [('selection', selecs, 1e-2), ('mask', masks, None), ('count', counts, 1e-2)]
for r, (lab, dct, vlim) in enumerate(rows):
    for c, x in enumerate(budgets):
        plt.subplot(3, 3, r * 3 + c + 1)
        plot_mesh(dct[x].astype(float), box_sizes[x], vlim=vlim)
        plt.colorbar()
        plt.title(f"{lab}  ~{x}^3  {(dct[x].shape)}")
fig.savefig(save_dir / "plot_meshes.png", dpi=130)
plt.close(fig)
print(f"saved {save_dir / 'plot_meshes.png'}")


# ---------------------------------------------------------------------------
# Plot delta: rows = synthetic IC (@ init_shape) / count2delta (@ final_shape) / spectra.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 13), layout='constrained')
fig.suptitle(f"{tracer} {region} {fnl} (f_NL={fNL_true:g}) -- synthetic IC, delta, spectra")
for c, x in enumerate(budgets):
    plt.subplot(3, 3, c + 1)
    plot_mesh(synths[x], box_sizes[x], vlim=1e-3)
    plt.colorbar()
    plt.title(rf"synthetic IC  ~{x}^3  {(synths[x].shape)}")

    plt.subplot(3, 3, c + 4)
    plot_mesh(deltas[x], box_sizes[x], vlim=1e-3)
    plt.colorbar()
    plt.title(rf"$\delta$  ~{x}^3  {(deltas[x].shape)}, ngbars={ngbars[x]:.2e}$")

    plt.subplot(3, 3, c + 7)
    plot_pow(*kdeltas[x], log=log, label=rf"$\delta$ ~{x}^3")
    plot_pow(*ksynths[x], ':', log=log, label='synthetic IC field')
    plot_pow(krefs[x][0], krefs[x][1], 'k--', log=log, label=rf'Kaiser+shot ($b_1$={b1E:g})')
    plot_pow(krefs[x][0], krefs[x][2] * np.ones_like(krefs[x][0]), 'k:', alpha=0.4, log=log, label='shot noise')
    plt.legend()
fig.savefig(save_dir / "plot_delta.png", dpi=130)
plt.close(fig)
print(f"saved {save_dir / 'plot_delta.png'}")
