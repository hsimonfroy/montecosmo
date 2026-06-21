#!/usr/bin/env python
"""
Register mock data ready to be inferred with `montecosmo.model.FieldLevelModel`.

For each mock configuration and (cell_budget, padding), this writes ONE self-describing
HDF5 file in ``png/registered`` (loadable via the model's ``register`` config):

  register_<tag>_b<n>_p<pad>.h5   (schema of FieldLevelModel.register_catalog + init data)
    cell_length, box_center, box_rotvec                 # geometry (final_shape = count_mesh.shape)
    init_oversamp, paint_oversamp                       # mesh oversampling (-> model config)
    cosmo_fid/{Omega_m, sigma8}                         # mock fiducial cosmology (-> latents loc/loc_fid)
    count_mesh                                          # painted tracer counts @ final_shape (sum == n_tracers)
    selec_mesh, mask_mesh                               # selection @ paint_shape / footprint (cut-sky only)
    n_tracers, n_randoms                                # weighted catalog sizes (n_randoms cut-sky only)
    a_obs, curved_sky                                   # full-sky: 1/(1+z), False; cut-sky: None, True (light-cone, curved)
    paint_order, interlace_order, paint_deconv, kernel_type, cell_budget, padding
    init_kpow                                           # (2,N) k, P(k)/sigma8^2 from cosmoprimo (sigma8=1 normalized)
    init_mesh  OR  init_fake                            # complex rfft @ r2chshape(init_shape); fake only if no real ICs

`FieldLevelModel(register=path)` then overrides geometry/painting params, loads the meshes,
and sets the cosmology + ngbars latent locs (ngbars from n_tracers / footprint volume).

Every mock is a particle catalog, registered through `FieldLevelModel.register_catalog`, which
handles two cases (distinguished by whether a `random` catalog is given):
  * full-sky (no randoms): cartesian 'pos' (+ optional 'vel' -> RSD along z); provide box_size,
                           box_center, and a constant a_obs (flat sky).
  * cut-sky  (randoms):    'RA', 'DEC', 'Z', 'WEIGHT', light-cone (a_obs=None), curved sky.

Sources, each via a getter returning a uniform spec:
  * 'abacus'  : cubic box, choose cosmo, ic, z_obs, tracer, field in {matter, realspace, redshiftspace}.
  * 'fastpm'  : cubic box, redshift-space only (RSD baked in), choose fNL in {0, -100, 100}.
  * 'pngunit' : cut-sky catalog, choose fNL in {0, 20, -20}, tracer, region.

Run on a GPU node with the `montenv` env:
    export XLA_PYTHON_CLIENT_MEM_FRACTION='1.'
    python -u src/montecosmo/tests/register.py
"""
import os; os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '1.')
import glob
from pathlib import Path
import numpy as np

from jax import numpy as jnp, random as jr, config as jconfig
jconfig.update("jax_enable_x64", True)

from montecosmo.model import FieldLevelModel
from montecosmo.bricks import lin_power_mesh, get_cosmology
from montecosmo.utils import scale_shape, chreshape, r2chshape, h5save


# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------
REGISTERED_DIR = Path("/pscratch/sd/h/hsimfroy/png/registered")

# Painting defaults (stored in the file and applied by the loader).
PAINT = dict(paint_order=2, interlace_order=2, paint_deconv=True)
PAINT_OVERSAMP, INIT_OVERSAMP = 7/4, 3/2

# Cubic boxes (Mpc/h): size and geometric center (abacus in [-1000, 1000], fastpm in [0, 2760]).
# register_catalog paints full-sky RSD along the z axis (flat sky).
ABACUS_BOX, ABACUS_CENTER = 2000., (0., 0., 0.)
FASTPM_BOX, FASTPM_CENTER = 2760., (1380., 1380., 1380.)


# ---------------------------------------------------------------------------
# Cosmology and initial power spectrum (cosmoprimo)
# ---------------------------------------------------------------------------
def abacus_cosmo(cosmo: int = 0):
    """Return the cosmoprimo AbacusSummit fiducial cosmology of given index."""
    from cosmoprimo.fiducial import AbacusSummit
    return AbacusSummit(cosmo)


def unit_cosmo():
    """
    cosmoprimo UNIT simulation cosmology (Chuang+2019, arXiv:1811.01892), the PNGUNIT fiducial:
    flat LCDM Omega_m=0.3089, Omega_b=0.0486, h=0.6774, n_s=0.9667, sigma8=0.8147 (Planck 2015).
    (The cosmology is not stored in the PNGUNIT catalogs; it comes from the UNIT sim it is built on.)
    """
    from cosmoprimo import Cosmology
    return Cosmology(Omega_m=0.3089, Omega_b=0.0486, h=0.6774, n_s=0.9667, sigma8=0.8147, engine='class')


def cosmo2loc(cosmo):
    """
    cosmoprimo cosmology -> model latent loc {Omega_m, sigma8}, matching the model's jax_cosmo
    parametrization (Omega_m = cold dark matter + baryons, sigma8 = sigma8_m). For AbacusSummit(0)
    this reproduces AbacusSummit0 (Omega_m=0.3137721, sigma8=0.8076354).
    """
    return {'Omega_m': float(cosmo.Omega0_cdm + cosmo.Omega0_b),
            'sigma8': float(cosmo.get_fourier().sigma8_m)}


def compute_init_kpow(cosmo, n_interp: int = 256, kmin: float = 1e-4, kmax: float = 1e1):
    """
    Return (k, P(k)/sigma8_m^2) at z=0 as a (2, n_interp) array: the prior power spectrum
    normalized to sigma8=1, so the model recovers P(k) by scaling with the sampled sigma8.
    `cosmo` is a cosmoprimo cosmology.
    """
    pk_interpolator = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)
    ks = np.logspace(np.log10(kmin), np.log10(kmax), n_interp)
    pows = np.asarray(pk_interpolator(ks)) / cosmo.get_fourier().sigma8_m**2
    return np.stack((ks, pows))


# ---------------------------------------------------------------------------
# Initial conditions helpers
# ---------------------------------------------------------------------------
def read_abacus_ic(fn, z: int = 0):
    """Read an AbacusSummit ic_dens_N*.asdf density mesh, scaled to redshift index z."""
    import asdf
    with asdf.open(str(fn)) as af:
        growth = af['header']['GrowthTable'][z] if z is not None else 1.
        return np.asarray(af['data']['density']) * growth


def read_abacus_pos(fn):
    """Read particle positions from an Abacus *_rv_A_*.asdf file (for matter painting)."""
    from abacusnbody.data import read_abacus
    return np.asarray(read_abacus.read_asdf(str(fn), load=['pos'])['pos'])


def read_abacus_lagr_pos(fn):
    """Read Lagrangian (initial) particle positions from an Abacus *_rv_A_*.asdf file (PID-derived)."""
    from abacusnbody.data import read_abacus
    return np.asarray(read_abacus.read_asdf(str(fn), load=['lagr_pos'])['lagr_pos'])


def downsample_ic(real_mesh, init_shape):
    """rfft a real-space initial mesh and Fourier-downsample to r2chshape(init_shape)."""
    fmesh = jnp.fft.rfftn(jnp.asarray(real_mesh))
    return np.asarray(chreshape(fmesh, r2chshape(tuple(int(s) for s in np.asarray(init_shape)))))


def synth_ic(cosmo, init_shape, box_size, a=1., seed=0):
    """Synthetic Gaussian linear field @ init_shape (random phases, physical P(k)), jax_cosmo `cosmo`."""
    init_shape = tuple(int(s) for s in np.asarray(init_shape))
    pmesh = lin_power_mesh(cosmo, np.asarray(init_shape), np.asarray(box_size), a=a)
    return jnp.fft.irfftn(jnp.fft.rfftn(jr.normal(jr.key(seed), init_shape)) * pmesh**.5)


def build_init(spec, init_shape, cosmo_jax, box_size):
    """
    Return the init dict: init_kpow (always, from the cosmoprimo `cosmo_fid`) + init_mesh (real
    ICs, abacus) or, when no real ICs exist (fastpm, pngunit), a synthetic Gaussian init_fake
    drawn at the mock cosmology `cosmo_jax` (z=0 linear field). To be replaced by real ICs later.
    """
    init = {'init_kpow': compute_init_kpow(spec['cosmo_fid'])}
    if spec.get('ic_fn'):
        init['init_mesh'] = downsample_ic(read_abacus_ic(spec['ic_fn']), init_shape)
    else:
        init['init_fake'] = downsample_ic(synth_ic(cosmo_jax, init_shape, box_size, a=1.), init_shape)
    return init


# ---------------------------------------------------------------------------
# Source getters -> uniform spec dict (data/random dict-likes + cosmo_fid + box params)
# ---------------------------------------------------------------------------
def get_abacus(cosmo=0, ic=0, z_obs=0.8, tracer='LRG', field='redshiftspace'):
    """Spec for an AbacusSummit cubic-box mock. field in {matter, realspace, redshiftspace}."""
    base = f"AbacusSummit_base_c{cosmo:03d}_ph{ic:03d}"
    spec = dict(cosmo_fid=abacus_cosmo(cosmo), a_obs=1 / (1 + z_obs),
                box_size=np.full(3, ABACUS_BOX), box_center=ABACUS_CENTER,
                ic_fn=f"/dvs_ro/cfs/cdirs/desi/public/cosmosim/AbacusSummit/ic/{base}/ic_dens_N576.asdf",
                tag=f"abacus_c{cosmo}_ph{ic}_z{z_obs:.3f}_{tracer}_{field}")

    if field == 'matter':  # stream particle files as an iterator of dicts (real-space matter)
        fns = sorted(sum(
            [glob.glob(f"/dvs_ro/cfs/cdirs/desi/public/cosmosim/AbacusSummit/{base}/"
                       f"halos/z{z_obs:.3f}/{t}_rv_A/{t}_rv_A_*.asdf") for t in ['field', 'halo']], []))
        spec['data'] = ({'pos': read_abacus_pos(fn)} for fn in fns)
        return spec

    import fitsio
    fits = fitsio.read(f"/dvs_ro/cfs/cdirs/desi/cosmosim/SecondGenMocks/CubicBox/{tracer}/"
                       f"z{z_obs:.3f}/{base}/{tracer}_real_space.fits")
    data = {'pos': np.column_stack([fits[c] for c in ['x', 'y', 'z']]).astype(float)}
    if field == 'redshiftspace':  # RSD applied in register_catalog from the peculiar velocity
        data['vel'] = np.column_stack([fits[c] for c in ['vx', 'vy', 'vz']]).astype(float)
    elif field != 'realspace':
        raise ValueError(f"unknown abacus field {field!r}")
    spec['data'] = data
    return spec


def get_fastpm(fNL=0, z_obs=1.0, tracer='LRG'):
    """Spec for a FastPM cubic-box mock (redshift-space los-z, RSD baked in). fNL in {0, -100, 100}."""
    import fitsio
    cat_fn = ("/pscratch/sd/a/adematti/desi-fnl-standard-analysis/"
              f"run-knl-fnl-{fNL}-a0.5000/catalog_mcut2.25e+12_nbar1.00e-04_los-z.fits")
    pos = np.asarray(fitsio.read(cat_fn)['Position']).astype(float)
    return dict(data={'pos': pos}, cosmo_fid=abacus_cosmo(0),
                a_obs=1 / (1 + z_obs),
                box_size=np.full(3, FASTPM_BOX), box_center=FASTPM_CENTER,
                tag=f"fastpm_fNL{fNL}_z{z_obs:.3f}_{tracer}")


def get_pngunit(fNL=0, tracer='LRG', region='NGC'):
    """Spec for a PNGUNITXL cut-sky catalog. fNL in {0, 20, -20}."""
    import fitsio
    fnl_dir = {0: 'fnl0', 20: 'fnl20', -20: 'fnlm20'}[fNL]
    cat_dir = Path("/global/cfs/cdirs/desicollab/users/adrigut/PNGxHOD"
                   "/dev_mocks/catalogs/DA2/v2.0/PNGUNITXL")
    cols = ['RA', 'DEC', 'Z', 'WEIGHT']
    data = fitsio.read(str(cat_dir / "complete" / fnl_dir /
                           f"{tracer}_complete_{region}_clustering.dat.fits"), columns=cols)
    rand = fitsio.read(str(cat_dir / "randoms" /
                           f"{tracer}_complete_{region}_0_clustering.ran.fits"), columns=cols)
    return dict(data={c: data[c] for c in cols}, random={c: rand[c] for c in cols},
                cosmo_fid=unit_cosmo(),  # UNIT simulation cosmology (PNGUNIT fiducial)
                tag=f"pngunit_fNL{fNL}_{tracer}_{region}")


# ---------------------------------------------------------------------------
# Registration (one HDF5 written in a single pass)
# ---------------------------------------------------------------------------
def _register_path(spec, cell_budget, padding):
    n = int(round(cell_budget ** (1 / 3)))
    return REGISTERED_DIR / f"register_{spec['tag']}_b{n}_p{padding:g}.h5"


def register(spec, cell_budget, padding=0.):
    """Register one mock at a given budget/padding: paint (register_catalog) + init, write one HDF5."""
    REGISTERED_DIR.mkdir(parents=True, exist_ok=True)
    cosmo_jax = get_cosmology(**cosmo2loc(spec['cosmo_fid']))  # jax_cosmo for painting + fake IC

    obs = FieldLevelModel.register_catalog(
        cell_budget, cosmo_jax, spec['data'], random=spec.get('random'),
        box_size=spec.get('box_size'), box_center=spec.get('box_center'), box_rotvec=spec.get('box_rotvec'),
        a_obs=spec.get('a_obs'), padding=padding,
        init_oversamp=INIT_OVERSAMP, paint_oversamp=PAINT_OVERSAMP, **PAINT)

    final_shape = np.asarray(obs['count_mesh'].shape)
    box_size = final_shape * obs['cell_length']
    init_shape = scale_shape(final_shape, INIT_OVERSAMP)
    init = build_init(spec, init_shape, cosmo_jax, box_size)
    path = _register_path(spec, cell_budget, padding)
    h5save(path, {**obs, **init})
    print(f"registered {path.name}: final={tuple(int(s) for s in final_shape)} "
          f"cell={obs['cell_length']:.1f} Mpc/h n_tracers={obs['n_tracers']:.3e} "
          f"count.sum()={float(obs['count_mesh'].sum()):.3e} "
          f"init={'init_mesh' if 'init_mesh' in init else 'init_fake'}")
    return path


# ---------------------------------------------------------------------------
# Main: register selected mocks over cell budgets and paddings
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback
    cell_budgets = [32**3, 64**3, 96**3, 128**3]
    paddings = [0.]

    # 9 mocks x 4 budgets = 36 files. Light jobs first, heavy abacus matter (~71 GB I/O) last.
    jobs = [
        lambda: get_fastpm(fNL=-100),
        lambda: get_fastpm(fNL=0),
        lambda: get_fastpm(fNL=100),
        lambda: get_abacus(cosmo=0, ic=0, z_obs=0.8, tracer='LRG', field='realspace'),
        lambda: get_abacus(cosmo=0, ic=0, z_obs=0.8, tracer='LRG', field='redshiftspace'),
        lambda: get_pngunit(fNL=0, tracer='LRG', region='NGC'),
        lambda: get_pngunit(fNL=20, tracer='LRG', region='NGC'),
        lambda: get_pngunit(fNL=-20, tracer='LRG', region='NGC'),
        lambda: get_abacus(cosmo=0, ic=0, z_obs=0.8, tracer='LRG', field='matter'),  # heavy, last
    ]

    n_ok, n_fail = 0, 0
    for make_spec in jobs:
        for cell_budget in cell_budgets:
            for padding in paddings:
                # Rebuild the spec per budget: data may be a one-shot iterator (abacus matter).
                try:
                    register(make_spec(), cell_budget, padding)
                    n_ok += 1
                except Exception:
                    n_fail += 1
                    print(f"FAILED job at b{round(cell_budget**(1/3))} p{padding:g}:")
                    traceback.print_exc()
    print(f"\nDONE: {n_ok} registered, {n_fail} failed.")
