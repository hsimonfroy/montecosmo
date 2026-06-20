#!/usr/bin/env python
"""
Register mock data ready to be inferred with `montecosmo.model.FieldLevelModel`.

For each mock configuration and (cell_budget, padding), this writes two files in
``png/registered``:

* ``obs_<tag>.npz`` -- the observation, in the exact format produced by
  ``FieldLevelModel.register_catalog`` (loadable via the model's ``meshes`` config):
      selec_mesh, mask_mesh, count_mesh, n_tracers, n_randoms,
      final_shape, cell_length, box_center, box_rotvec.
  Catalog data (pngunit) carries a true selection (selec_mesh @ evol_shape, mask_mesh
  @ final_shape). Periodic-box data (abacus, fastpm) has no footprint, so it omits
  selec_mesh / mask_mesh / n_randoms: the model then uses a unit selection and no
  masking, and recomputes ngbars from count_mesh.

* ``init_<tag>.npz`` -- the initial conditions reference:
      init_kpow : (2, n) array (k, P(k)/sigma8_m^2) at z=0 from cosmoprimo, the prior
                  power spectrum to use (-> model `init_power`).
      init_mesh : if real ICs exist (abacus), the linear field rfft'd and Fourier
                  downsampled to r2chshape(init_shape) with `chreshape`.
      init_fake : otherwise (fastpm, pngunit), a fake field under this key (the loaded
                  fake box for fastpm, a synthetic Gaussian for pngunit).

Three sources are supported, each via a getter returning a uniform spec:
  * 'abacus'  : cubic box, choose cosmo, ic, z_obs, tracer, and field in
                {'matter', 'realspace', 'redshiftspace'}.
  * 'fastpm'  : cubic box, redshift-space only, choose fNL in {0, -100, 100}.
  * 'pngunit' : sky catalog, registered through `model.register_catalog`,
                choose fNL, tracer, region.

Run on a GPU node with the `montenv` env:
    export XLA_PYTHON_CLIENT_MEM_FRACTION='1.'
    python -u src/montecosmo/tests/register.py
"""
import os; os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '1.')
import glob
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from jax import numpy as jnp, random as jr, config as jconfig
jconfig.update("jax_enable_x64", True)

from montecosmo.bricks import phys2cell_pos, get_mesh_shape, lin_power_mesh, Planck15, AbacusSummit0
from montecosmo.nbody import nufft
from montecosmo.utils import scale_shape, chreshape, r2chshape


# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------
REGISTERED_DIR = Path("/pscratch/sd/h/hsimfroy/png/registered")

# Painting / box-model defaults (must match the inference model's settings).
PAINT = dict(paint_order=2, interlace_order=2, paint_deconv=True)
EVOL_OVERSAMP, INIT_OVERSAMP = 2., 1.

# Abacus cubic box: positions in [-1000, 1000] Mpc/h (box 2000, centered at origin).
ABACUS_POS_MIN, ABACUS_POS_MAX = -1000., 1000.
# FastPM cubic box: positions in [0, 2760] Mpc/h.
FASTPM_POS_MIN, FASTPM_POS_MAX = 0., 2760.
FASTPM_LOAD_DIR = Path("/pscratch/sd/h/hsimfroy/png/fpm_b2760_z1_lrg_fNL/load")
FASTPM_FAKE_IC = FASTPM_LOAD_DIR / "init_mesh_fake_2760_256.npy"

# Box-model line-of-sight center (flat sky, z-axis RSD), as in script_pipe_fpm.py.
BOX_CENTER = (0., 0., 1.)


# ---------------------------------------------------------------------------
# Initial power spectrum (cosmoprimo)
# ---------------------------------------------------------------------------
def _patch_cosmoprimo():
    """Work around an upstream cosmoprimo typo in the out=='rho' ncdm integrand."""
    import cosmoprimo.cosmology as cc
    if getattr(cc, "_montecosmo_patched", False):
        return
    orig = cc._make_phase_space_integrand
    def patched(jnp_, out, exp_sign):
        if out == 'rho':
            def integrand(q, m_over_T2, m2_over_T2):
                return q**2 * jnp_.sqrt(q**2 + m2_over_T2) / (1. + jnp_.exp(exp_sign * q))
            return integrand
        return orig(jnp_, out, exp_sign)
    cc._make_phase_space_integrand = patched
    cc._montecosmo_patched = True


def abacus_cosmo(cosmo: int = 0):
    """Return the cosmoprimo AbacusSummit fiducial cosmology of given index."""
    _patch_cosmoprimo()
    from cosmoprimo.fiducial import AbacusSummit
    return AbacusSummit(cosmo)


def compute_init_kpow(cosmo=None, n_interp: int = 256, kmin: float = 1e-4, kmax: float = 1e1):
    """
    Return the prior power spectrum (k, P(k)/sigma8_m^2) at z=0 as a (2, n_interp) array.
    `cosmo` is a cosmoprimo cosmology (default AbacusSummit(0)).
    """
    if cosmo is None:
        cosmo = abacus_cosmo(0)
    pk_interpolator = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)
    ks = np.logspace(np.log10(kmin), np.log10(kmax), n_interp)
    pows = np.asarray(pk_interpolator(ks)) / cosmo.get_fourier().sigma8_m**2
    return np.stack((ks, pows))


# ---------------------------------------------------------------------------
# Geometry and painting (periodic box)
# ---------------------------------------------------------------------------
def box_geometry(pos_min, pos_max, cell_budget, padding=0.):
    """
    Return (final_shape, cell_length, box_size, geom_center, evol_shape, init_shape)
    for a cubic box spanning [pos_min, pos_max] given a cell budget and padding.
    The mesh box (final_shape * cell_length) pads the catalog box, which is centered
    at geom_center within it.
    """
    span = float(pos_max - pos_min)
    geom_center = np.full(3, (pos_min + pos_max) / 2)
    final_shape, cell_length = get_mesh_shape(np.full(3, span), cell_budget, padding)
    box_size = final_shape * cell_length
    evol_shape = scale_shape(final_shape, EVOL_OVERSAMP)
    init_shape = scale_shape(final_shape, INIT_OVERSAMP)
    return final_shape, cell_length, box_size, geom_center, evol_shape, init_shape


def paint_count(pos_iter, geom_center, box_size, final_shape, paint_shape,
                weights=None, box_rotvec=(0., 0., 0.), **paint):
    """
    Paint a count mesh @ final_shape from cartesian positions, oversampling at
    paint_shape then Fourier-downsampling, exactly as `bricks.catalog2count` does.

    `pos_iter` is either an (N, 3) array, or an iterable of (N_i, 3) chunks (streamed
    and accumulated in Fourier space, for catalogs too large to hold at once, e.g. the
    abacus matter field). `weights` (single array) only applies to the array form.
    Returns (count_mesh @ final_shape, n_objects).
    """
    box_rot = Rotation.from_rotvec(np.asarray(box_rotvec, float))
    final_shape = np.asarray(final_shape)
    paint_shape = np.asarray(paint_shape)
    geom_center = np.asarray(geom_center, float)
    box_size = np.asarray(box_size, float)

    chunks = [pos_iter] if isinstance(pos_iter, np.ndarray) else pos_iter
    acc = jnp.zeros(r2chshape(tuple(int(s) for s in final_shape)), dtype=complex)
    n = 0.
    for chunk in chunks:
        chunk = np.array(chunk, dtype=float)
        cell = phys2cell_pos(chunk, geom_center, box_rot, box_size, final_shape)
        w = 1. if weights is None else jnp.asarray(weights)
        acc = acc + nufft(cell, final_shape, paint_shape, weights=w, **paint)
        n += chunk.shape[0] if weights is None else float(jnp.sum(jnp.asarray(weights)))
    count = jnp.fft.irfftn(acc) * np.divide(paint_shape, final_shape).prod()
    return np.asarray(count), n


# ---------------------------------------------------------------------------
# Initial conditions helpers
# ---------------------------------------------------------------------------
def read_abacus_ic(fn, z: int = 0):
    """Read an AbacusSummit ic_dens_N*.asdf density mesh, scaled to redshift index z."""
    import asdf
    with asdf.open(str(fn)) as af:
        growth = af['header']['GrowthTable'][z] if z is not None else 1.
        mesh = np.asarray(af['data']['density']) * growth
    return mesh


def read_abacus_pos(fn):
    """Read particle positions from an Abacus *_rv_A_*.asdf file (for matter painting)."""
    from abacusnbody.data import read_abacus
    return np.asarray(read_abacus.read_asdf(str(fn), load=['pos'])['pos'])


def downsample_ic(real_mesh, init_shape):
    """rfft a real-space initial mesh and Fourier-downsample to r2chshape(init_shape)."""
    fmesh = jnp.fft.rfftn(jnp.asarray(real_mesh))
    return np.asarray(chreshape(fmesh, r2chshape(tuple(int(s) for s in np.asarray(init_shape)))))


def synth_ic(cosmo, init_shape, box_size, a=1., seed=0):
    """Synthetic Gaussian linear field @ init_shape (random phases, physical P(k))."""
    init_shape = tuple(int(s) for s in np.asarray(init_shape))
    pmesh = lin_power_mesh(cosmo, np.asarray(init_shape), np.asarray(box_size), a=a)
    return jnp.fft.irfftn(jnp.fft.rfftn(jr.normal(jr.key(seed), init_shape)) * pmesh**.5)


def save_init(save_path, init_shape, kpow_cosmo=None, real_ic=None, fake_ic=None):
    """Save init_kpow plus init_mesh (real ICs) or init_fake (fallback) to one npz."""
    out = {'init_kpow': compute_init_kpow(kpow_cosmo)}
    if real_ic is not None:
        out['init_mesh'] = downsample_ic(real_ic, init_shape)
    if fake_ic is not None:
        out['init_fake'] = downsample_ic(fake_ic, init_shape)
    np.savez(str(save_path), **out)
    return out


# ---------------------------------------------------------------------------
# Source getters -> uniform spec dict
# ---------------------------------------------------------------------------
def get_abacus(cosmo=0, ic=0, z_obs=0.8, tracer='LRG', field='redshiftspace'):
    """
    Spec for an AbacusSummit cubic-box mock. field in
    {'matter', 'realspace', 'redshiftspace'}.
    """
    base = f"AbacusSummit_base_c{cosmo:03d}_ph{ic:03d}"
    spec = dict(source='abacus', kind='box', field=field, z_obs=z_obs,
                pos_min=ABACUS_POS_MIN, pos_max=ABACUS_POS_MAX, box_center=BOX_CENTER,
                kpow_cosmo=abacus_cosmo(cosmo),
                ic_fn=f"/dvs_ro/cfs/cdirs/desi/public/cosmosim/AbacusSummit/ic/{base}/ic_dens_N576.asdf",
                tag=f"abacus_c{cosmo}_ph{ic}_z{z_obs:.3f}_{tracer}_{field}")

    if field == 'matter':
        spec['particle_fns'] = sorted(sum(
            [glob.glob(f"/dvs_ro/cfs/cdirs/desi/public/cosmosim/AbacusSummit/{base}/"
                       f"halos/z{z_obs:.3f}/{t}_rv_A/{t}_rv_A_*.asdf") for t in ['field', 'halo']], []))
        return spec

    import fitsio
    cat_fn = (f"/dvs_ro/cfs/cdirs/desi/cosmosim/SecondGenMocks/CubicBox/{tracer}/"
              f"z{z_obs:.3f}/{base}/{tracer}_real_space.fits")
    fits = fitsio.read(cat_fn)
    pos = np.column_stack([fits[c] for c in ['x', 'y', 'z']]).astype(float)
    if field == 'redshiftspace':
        from jax_cosmo import background
        vel = np.column_stack([fits[c] for c in ['vx', 'vy', 'vz']]).astype(float)
        los = np.array([0., 0., 1.])
        a_obs = 1 / (1 + z_obs)
        E = float(background.Esqr(AbacusSummit0(), a_obs)**.5)
        vel = vel / (a_obs * 100 * E)         # peculiar velocity -> Mpc/h displacement
        pos = pos + (vel * los).sum(-1, keepdims=True) * los
    elif field != 'realspace':
        raise ValueError(f"unknown abacus field {field!r}")
    spec['pos'] = pos
    return spec


def get_fastpm(fNL=0, z_obs=1.0, tracer='LRG'):
    """
    Spec for a FastPM cubic-box mock (redshift-space, los-z). fNL in {0, -100, 100}.
    """
    import fitsio
    cat_fn = ("/pscratch/sd/a/adematti/desi-fnl-standard-analysis/"
              f"run-knl-fnl-{fNL}-a0.5000/catalog_mcut2.25e+12_nbar1.00e-04_los-z.fits")
    pos = np.asarray(fitsio.read(cat_fn)['Position']).astype(float)
    return dict(source='fastpm', kind='box', field='redshiftspace', z_obs=z_obs, pos=pos,
                pos_min=FASTPM_POS_MIN, pos_max=FASTPM_POS_MAX, box_center=BOX_CENTER,
                kpow_cosmo=abacus_cosmo(0), fake_ic_fn=FASTPM_FAKE_IC,
                tag=f"fastpm_fNL{fNL}_z{z_obs:.3f}_{tracer}")


def get_pngunit(fNL=0, tracer='LRG', region='NGC'):
    """
    Spec for a PNGUNITXL sky catalog (registered through model.register_catalog).
    fNL in {0, 20, -20}; NGC/SGC region; Planck15 mock cosmology.
    """
    import fitsio
    fnl_dir = {0: 'fnl0', 20: 'fnl20', -20: 'fnlm20'}[fNL]
    cat_dir = Path("/global/cfs/cdirs/desicollab/users/adrigut/PNGxHOD"
                   "/dev_mocks/catalogs/DA2/v2.0/PNGUNITXL")
    cols = ['RA', 'DEC', 'Z', 'WEIGHT']
    data = fitsio.read(str(cat_dir / "complete" / fnl_dir /
                           f"{tracer}_complete_{region}_clustering.dat.fits"), columns=cols)
    rand = fitsio.read(str(cat_dir / "randoms" /
                           f"{tracer}_complete_{region}_0_clustering.ran.fits"), columns=cols)
    return dict(source='pngunit', kind='catalog', data=data, random=rand, planck15=True,
                kpow_cosmo=abacus_cosmo(0),  # shape only; Planck15 mock, refine later
                tag=f"pngunit_fNL{fNL}_{tracer}_{region}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def _obs_path(spec, cell_budget, padding):
    n = int(round(cell_budget ** (1 / 3)))
    return REGISTERED_DIR / f"obs_{spec['tag']}_b{n}_p{padding:g}.npz"


def _init_path(spec, cell_budget, padding):
    n = int(round(cell_budget ** (1 / 3)))
    return REGISTERED_DIR / f"init_{spec['tag']}_b{n}_p{padding:g}.npz"


def register_box(spec, cell_budget, padding=0.):
    """Register a periodic-box spec (abacus / fastpm): write obs and init npz files."""
    REGISTERED_DIR.mkdir(parents=True, exist_ok=True)
    final_shape, cell_length, box_size, geom_center, evol_shape, init_shape = box_geometry(
        spec['pos_min'], spec['pos_max'], cell_budget, padding)

    if spec.get('particle_fns'):  # abacus matter: stream particle files
        pos_iter = (read_abacus_pos(fn) for fn in spec['particle_fns'])
    else:
        pos_iter = spec['pos']
    count_mesh, n = paint_count(pos_iter, geom_center, box_size, final_shape, evol_shape, **PAINT)

    obs_path = _obs_path(spec, cell_budget, padding)
    np.savez(str(obs_path), count_mesh=count_mesh, n_tracers=float(n),
             final_shape=final_shape, cell_length=cell_length,
             box_center=np.asarray(spec['box_center'], float), box_rotvec=np.zeros(3))

    real_ic = read_abacus_ic(spec['ic_fn']) if spec.get('ic_fn') else None
    fake_ic = np.load(spec['fake_ic_fn']) if spec.get('fake_ic_fn') else None
    save_init(_init_path(spec, cell_budget, padding), init_shape,
              kpow_cosmo=spec['kpow_cosmo'], real_ic=real_ic, fake_ic=fake_ic)
    print(f"registered {obs_path.name}: final={tuple(int(s) for s in final_shape)} "
          f"cell={cell_length:.1f} Mpc/h n={n:.3e} count_mean={count_mesh.mean():.3e}")
    return obs_path


def register_catalog_source(spec, cell_budget, padding=0., seed=0):
    """Register a sky-catalog spec (pngunit) through model.register_catalog + init npz."""
    from montecosmo.model import FieldLevelModel, default_config
    REGISTERED_DIR.mkdir(parents=True, exist_ok=True)

    model = FieldLevelModel(**default_config | {
        'box_center': (0., 0., 2000.), 'evolution': 'lpt', 'a_obs': None,
        'curved_sky': True, 'png_type': None, 'meshes': None,
        'paint_order': PAINT['paint_order'], 'paint_deconv': PAINT['paint_deconv'],
        'interlace_order': PAINT['interlace_order'],
        'init_oversamp': INIT_OVERSAMP, 'evol_oversamp': EVOL_OVERSAMP,
        'lik_type': 'gaussian_delta'})
    if spec.get('planck15'):  # set fiducial = mock cosmology
        latents = model.new_latents_from_loc({'Omega_m': 0.3075, 'sigma8': 0.8159}, update_prior=True)
        model = FieldLevelModel(**model.asdict() | {'latents': latents})

    obs_path = _obs_path(spec, cell_budget, padding)
    model.register_catalog(spec['data'], spec['random'], obs_path, cell_budget=cell_budget, padding=padding)

    cosmo_theory = Planck15() if spec.get('planck15') else model.cosmo_fid
    fake = synth_ic(cosmo_theory, model.init_shape, model.box_size, a=model.a_fid, seed=seed)
    save_init(_init_path(spec, cell_budget, padding), model.init_shape,
              kpow_cosmo=spec['kpow_cosmo'], fake_ic=fake)
    print(f"registered {obs_path.name}: final={tuple(int(s) for s in model.final_shape)} "
          f"cell={model.cell_length:.1f} Mpc/h n_tracers={model.n_tracers:.3e}")
    return obs_path


def register(spec, cell_budget, padding=0.):
    """Dispatch a spec to the right registration routine."""
    if spec['kind'] == 'box':
        return register_box(spec, cell_budget, padding)
    return register_catalog_source(spec, cell_budget, padding)


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------
def load_obs(path):
    return dict(np.load(str(path), allow_pickle=True))


def load_init(path):
    return dict(np.load(str(path)))


# ---------------------------------------------------------------------------
# Main: register selected mocks over cell budgets and paddings
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cell_budgets = [32**3, 64**3, 128**3]
    paddings = [0.]

    # Edit this list to choose which mocks to register.
    jobs = [
        lambda: get_fastpm(fNL=0),
        lambda: get_fastpm(fNL=-100),
        lambda: get_fastpm(fNL=100),
        # lambda: get_abacus(cosmo=0, ic=0, z_obs=0.8, tracer='LRG', field='redshiftspace'),
        # lambda: get_abacus(cosmo=0, ic=0, z_obs=0.8, tracer='LRG', field='realspace'),
        # lambda: get_abacus(cosmo=0, ic=0, z_obs=0.8, tracer='LRG', field='matter'),  # heavy
        # lambda: get_pngunit(fNL=0, tracer='LRG', region='NGC'),
    ]

    for make_spec in jobs:
        spec = make_spec()
        for cell_budget in cell_budgets:
            for padding in paddings:
                register(spec, cell_budget, padding)
