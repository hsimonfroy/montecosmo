#!/usr/bin/env python
"""
Standard P_ell(k) (P0, P2) PNG analysis of the v1.0 PNG-UNIT HOD boxes (QSO z=1.83, redshift space),
to compare with the field-level inference runs lpt_64_fNL{0,20,-20}_o3_shash_pp0 living on the same
register folder png/pngunitbox_QSO_z1.83_redshiftspace_fNL/ (see run/infer_eft.py).

Pipeline per fNL (adapted from src/flfnl/scripts/fastpm_fnl_standard_analysis.py):
  1. Read the ascii catalog, apply plane-parallel RSD along z at a_obs (same convention as
     register.py/fullsky2count), and measure P_{0,2,4}(k) with jaxpower on a 512^3 mesh.
  2. Compute the (periodic box) window matrix.
  3. Compute the Gaussian covariance from a smooth best-fit PNG theory + shot noise.
  4. Fit {fnl_loc, b1, sn0, sigmas} with desilike PNGTracerPowerSpectrumMultipoles(mode='b-p'),
     ells (0, 2), k in (0.002, 0.08), with the mass-relation p FIXED to 1.0 (fNL 0, -20) or
     1.4 (fNL 20), exactly as the FLI fiducials (infer_eft.py: PNGUnit fNL20 low-passed PNGs).
     4 NUTS chains, then export a getdist text root for plotting (run/plot_fli_vs_powell.py).

Environment: cosmodesi 2026_02 (desilike + jaxpower + lsstypes), NOT montenv:
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh 2026_02
Just source that env and run this script -- the fastprogress shim below is self-contained, so no
PYTHONPATH juggling is needed.
Run on a GPU compute node (jaxpower FFTs + NUTS); each step is skipped if its output exists.
Usage: python powell_pngunitbox.py [test]   ('test': small mesh + subsampled catalog, saved in _test)
"""

import shutil
import sys
import subprocess
from pathlib import Path


def _fix_blackjax_fastprogress():
    """
    The cosmodesi 2026_02 conda env ships a fastprogress whose top-level import pulls `fasthtml`
    / `fastcore` (both absent), which crashes `import blackjax` -> desilike's NUTSSampler with
    `ModuleNotFoundError: No module named 'fasthtml'`. A plain `pip install --user fastprogress`
    does NOT help: user site-packages is LAST on that env's sys.path, so it can't shadow the
    conda copy. Fix: keep a private `fastprogress==1.0.3` (no fasthtml import) and PREPEND it to
    sys.path here, installing it on first run if missing. Must run before any `import blackjax`.
    """
    fixes = Path.home() / ".local" / "cosmodesi_2026_02_fixes"
    if not (fixes / "fastprogress").is_dir():
        print(f"Installing fastprogress==1.0.3 into {fixes} (one-time blackjax fix)...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(fixes),
                        "--no-deps", "fastprogress==1.0.3"], check=True)
    if str(fixes) not in sys.path:
        sys.path.insert(0, str(fixes))


_fix_blackjax_fastprogress()

import numpy as np

from mockfactory import setup_logging
import lsstypes as types

BOXSIZE = 3000.  # Mpc/h, periodic
# TRACER = 'QSO'
TRACER = 'LRG'
ZS = {'LRG': 0.74, 'QSO': 1.83}
LOS = 'z'  # plane-parallel RSD axis, as register.py (los=(0,0,1))
MESHSIZE = 512
ELLS_FIT, KLIM = (0, 2), (0.002, 0.08)
PPS = {'LRG': {-20: 0.80, 0: 0.80, 20: 1.13}, 'QSO': {-20: 0.83, 0: 0.83, 20: 1.55}}

CATALOG_DIR = Path('/global/cfs/cdirs/desicollab/users/adrigut/PNGxHOD/boxes/catalogs/v1.0')
SAVE_ROOT = Path(f'/pscratch/sd/h/hsimfroy/png/pngunitbox_{TRACER}_z{ZS[TRACER]}_redshiftspace_fNL')

# Fit parametrization (desilike PNGTracerPowerSpectrumMultipoles mode):
#  'bp'   -> mass relation assumed: bphi = 2*deltac*(b1 - p), p FIXED per fNL (sample fnl_loc). [default]
#  'bfnl' -> NO mass relation: sample bfnl_loc = fNL*bphi directly (the only PNG combo P constrains;
#            bare fNL and fNL*bphidelta are prior-dominated, cf. arXiv:2605.21436 Fig.3, added at plot time).
MODE = 'bp'
TEST = False
for _arg in sys.argv[1:]:
    if _arg == 'test': TEST = True
    elif _arg in ('bp', 'bfnl'): MODE = _arg
if TEST:
    MESHSIZE = 128
    SAVE_ROOT = SAVE_ROOT / '_test'


def unit_cosmo():
    """UNIT simulation cosmology (Chuang+2019), the PNGUNIT fiducial (as register.py)."""
    from cosmoprimo import Cosmology
    return Cosmology(Omega_m=0.3089, Omega_b=0.0486, h=0.6774, n_s=0.9667, sigma8=0.8147, engine='class')


def get_fn(kind, fnl, pp=None, ichain=0, mode='bp'):
    dirname = SAVE_ROOT / f'powell_fNL{fnl:d}'
    if kind in ('mesh2_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum'):  # mode-independent
        return dirname / f'{kind.replace("_mesh2_spectrum", "").replace("mesh2_spectrum", "spectrum")}.h5'
    basename = f'bfnl_kmax{KLIM[1]:g}' if mode == 'bfnl' else f'pp{pp:.1f}_kmax{KLIM[1]:g}'
    if kind == 'mcmc': return dirname / f'mcmc_{basename}_{ichain:d}.npy'
    if kind == 'getdist': return dirname / f'mcmc_getdist_{basename}'
    raise NotImplementedError(f'kind {kind} not implemented')


def read_catalog(fnl):
    """Redshift-space positions (n, 3) in [0, BOXSIZE): plane-parallel RSD along LOS at a_obs,
    peculiar velocity in km/s, same convention as register.py get_pngunitbox/fullsky2count."""
    tag = {0: 'fnl0', 20: 'fnl20', -20: 'fnlm20'}[fnl]
    fn = CATALOG_DIR / f'PNGUNIT_{tag}_{TRACER}_box_z{ZS[TRACER]}.txt'
    print(f'Reading {fn}', flush=True)
    arr = np.fromfile(str(fn), dtype=float, sep=' ').reshape(-1, 15)
    if TEST: arr = arr[::10]  # uniform thinning (the file is position-sorted, do NOT truncate)
    pos, vel = arr[:, 0:3], arr[:, 3:6]
    a_obs = 1. / (1. + ZS[TRACER])
    rsd_factor = 1. / (a_obs * 100 * unit_cosmo().get_background().efunc(ZS[TRACER]))
    vlos = np.zeros(3)
    vlos['xyz'.index(LOS)] = 1.
    pos = pos + rsd_factor * (vel @ vlos)[:, None] * vlos
    return pos % BOXSIZE


def compute_spectrum(output_fn, fnl):
    from jaxpower import (ParticleField, MeshAttrs, compute_box2_normalization, compute_fkp2_shotnoise,
                          BinMesh2SpectrumPoles, compute_mesh2_spectrum)
    positions = np.asarray(read_catalog(fnl), dtype='f8')
    csize = len(positions)
    print(f'{csize} tracers, nbar = {csize / BOXSIZE**3:.3e} (h/Mpc)^3', flush=True)
    mattrs = MeshAttrs(boxsize=BOXSIZE, boxcenter=BOXSIZE / 2., meshsize=MESHSIZE)
    data = ParticleField(positions, attrs=mattrs, exchange=True, backend='jax')
    kw = {'z': ZS[TRACER], 'size_data1': csize, 'wsum_data1': data.sum()}
    bin = BinMesh2SpectrumPoles(mattrs, edges=np.arange(1e-3, 6e-1, 1e-3), ells=(0, 2, 4))
    norm = compute_box2_normalization(data, bin=bin)
    num_shotnoise = compute_fkp2_shotnoise(data, bin=bin)
    mesh = data.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    mesh = mesh - mesh.mean()
    del data
    spectrum = compute_mesh2_spectrum(mesh, bin=bin, los=LOS).clone(norm=norm, num_shotnoise=num_shotnoise)
    mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    spectrum = spectrum.clone(attrs=dict(los=LOS, **kw, **mattrs))
    spectrum.write(output_fn)


def compute_window(output_fn, spectrum_fn):
    from jaxpower import compute_mesh2_spectrum_window, BinMesh2SpectrumPoles, MeshAttrs, read
    spectrum = read(spectrum_fn)
    mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    ells = spectrum.ells
    bin = BinMesh2SpectrumPoles(mattrs, edges=spectrum.get(0).edges('k'), ells=ells)
    window = compute_mesh2_spectrum_window(mattrs, edgesin=bin.edges, ellsin=ells, los=spectrum.attrs['los'], bin=bin)
    for pole in window.theory: pole._meta['z'] = spectrum.attrs['z']
    window.write(output_fn)


def compute_covariance(output_fn, spectrum_fn, window_fn):
    """Gaussian box covariance from a smooth best-fit PNG theory + shot noise (as the example)."""
    from jaxpower import MeshAttrs, compute_spectrum2_covariance
    mean = types.read(spectrum_fn)
    smooth = mean
    window = types.read(window_fn)

    mattrs = MeshAttrs(**{name: mean.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    shotnoise = mean.get(0).values('shotnoise').mean()
    covariance = compute_spectrum2_covariance(mattrs, mean.at(0).clone(value=mean.get(0).value() + shotnoise))

    sl = (0.002, 0.1)
    mean = mean.select(k=sl)
    window = window.at.observable.select(k=sl)
    covariance = covariance.at.observable.select(k=sl)

    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler

    template = FixedPowerSpectrumTemplate(fiducial=unit_cosmo(), z=window.theory.get(ells=0).z)
    theory = PNGTracerPowerSpectrumMultipoles(template=template, mode='bfnl')
    observable = TracerPowerSpectrumMultipolesObservable(data=mean, wmatrix=window, theory=theory, covariance=covariance)
    likelihood = ObservablesGaussianLikelihood(observable)

    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.maximize()
    print(profiles.to_stats(tablefmt='pretty'), flush=True)

    theory.init.update(k=smooth.get(0).coords('k'))
    poles = theory(**profiles.bestfit.choice(index='argmax', input=True))
    smooth = smooth.clone(value=poles.ravel())
    smooth = smooth.at(0).clone(value=smooth.get(0).value() + shotnoise)
    covariance = compute_spectrum2_covariance(mattrs, smooth)
    covariance.write(output_fn)


def get_likelihood(spectrum_fn, window_fn, covariance_fn, pp=1., mode='bp'):
    """
    PNG Kaiser likelihood. mode='bp': mass relation bphi = 2*deltac*(b1 - p) with p FIXED to pp,
    so fnl_loc is sampled (as the FLI pp runs). mode='bfnl': NO mass relation, sample bfnl_loc = fNL*bphi
    directly -- the only PNG combination the power spectrum constrains (bare fNL degenerate with bphi).
    """
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    data = types.read(spectrum_fn)
    window = types.read(window_fn)
    covariance = types.read(covariance_fn)
    shotnoise = 1. / (data.attrs['size_data1'] / BOXSIZE**3)  # Poisson shot noise, sets the sn0 scale
    data = data.get(ells=list(ELLS_FIT)).select(k=KLIM)
    window = window.at.observable.get(ells=data.ells).at.observable.match(data)
    covariance = covariance.at.observable.get(ells=data.ells).at.observable.match(data)
    template = FixedPowerSpectrumTemplate(z=data.attrs['z'], fiducial=unit_cosmo())
    dmode = 'bfnl' if mode == 'bfnl' else 'b-p'  # desilike mode name
    theory = PNGTracerPowerSpectrumMultipoles(template=template, mode=dmode, method='else', shotnoise=shotnoise)
    if mode == 'bp':  # fix the mass-relation parameter p (universal mass relation)
        theory.init.params['p'].update(value=pp, fixed=True)
    observable = TracerPowerSpectrumMultipolesObservable(data=data, theory=theory, wmatrix=window, covariance=covariance)
    likelihood = ObservablesGaussianLikelihood(observables=observable)
    return likelihood


def fit(mcmc_fns, spectrum_fn, window_fn, covariance_fn, pp=1., mode='bp'):
    from desilike.profilers import MinuitProfiler
    from desilike.samplers import NUTSSampler

    likelihood = get_likelihood(spectrum_fn, window_fn, covariance_fn, pp=pp, mode=mode)

    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.maximize()
    profiles = profiler.interval()
    print(profiles.to_stats(tablefmt='pretty'), flush=True)

    sampler = NUTSSampler(likelihood, seed=42, save_fn=mcmc_fns)
    sampler.run(check={'max_eigen_gr': 0.02})


def export_getdist(getdist_fn, mcmc_fns, burnin=0.5):
    """Concatenate the chains (after burn-in) into a getdist text root for plotting."""
    from desilike.samples import Chain
    chain = Chain.concatenate([Chain.load(fn).remove_burnin(burnin) for fn in mcmc_fns])
    samples = chain.to_getdist()
    samples.saveAsText(str(getdist_fn))
    print(f'Saved getdist root {getdist_fn}', flush=True)


if __name__ == '__main__':

    print(f'Demat (MODE={MODE}{", TEST" if TEST else ""})', flush=True)
    setup_logging()

    for fnl in ([0] if TEST else [0, 20, -20]):
        pp = PPS[TRACER][fnl]
        print(f'\n=== fNL = {fnl}, p = {pp}, mode = {MODE} ===', flush=True)
        save_dir = get_fn('mesh2_spectrum', fnl).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(__file__, save_dir / Path(__file__).name)

        # spectrum/window/covariance are mode-independent -> reused across bp and bfnl fits
        spectrum_fn = get_fn('mesh2_spectrum', fnl)
        if not spectrum_fn.exists(): compute_spectrum(spectrum_fn, fnl)
        window_fn = get_fn('window_mesh2_spectrum', fnl)
        if not window_fn.exists(): compute_window(window_fn, spectrum_fn=spectrum_fn)
        covariance_fn = get_fn('covariance_mesh2_spectrum', fnl)
        if not covariance_fn.exists(): compute_covariance(covariance_fn, spectrum_fn=spectrum_fn, window_fn=window_fn)

        mcmc_fns = [get_fn('mcmc', fnl, pp=pp, ichain=i, mode=MODE) for i in range(4)]
        if not all(Path(fn).exists() for fn in mcmc_fns):
            fit(mcmc_fns, spectrum_fn=spectrum_fn, window_fn=window_fn, covariance_fn=covariance_fn, pp=pp, mode=MODE)
        getdist_fn = get_fn('getdist', fnl, pp=pp, mode=MODE)
        if not getdist_fn.with_suffix('.txt').exists():
            export_getdist(getdist_fn, mcmc_fns)

    print('Kenavo', flush=True)
