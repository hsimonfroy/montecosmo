import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.95'

import h5py
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax_cosmo.scipy.integrate import simps
import astropy.units as u
import numpy as np
from bpcosmo.pm import get_density_planes
from jaxpm.lensing import convergence_Born
import jax
import jax.numpy as jnp

import jax_cosmo as jc

import numpyro
import numpyro.distributions as dist

from jax.experimental.ode import odeint
from jaxpm.pm import lpt, make_ode_fn
from jaxpm.kernels import fftk
from jaxpm.lensing import density_plane
import haiku as hk

from jaxpm.painting import cic_paint, cic_read, cic_paint_2d
from jaxpm.kernels import gradient_kernel, laplace_kernel, longrange_kernel
from jaxpm.nn import NeuralSplineFourierFilter
from jaxpm.utils import gaussian_smoothing

import diffrax
from diffrax import diffeqsolve, ODETerm, Dopri5, LeapfrogMidpoint, PIDController, SaveAt

import numpyro
import numpyro.distributions as dist

# Reading the DC2 tomographic bins into redshift distribution objects
with h5py.File("shear_photoz_stack.hdf5") as f:
    group = f["n_of_z"]
    # Read the z grid
    source = group["source"]
    z_shear = source['z'][::]
    # Read the true n(z)
    nz_shear = [jc.redshift.kde_nz(z_shear,
                                   source[f"bin_{i}"][:],
                                   bw=0.01, zmax=2.5) for i in range(4)]


# Loads some correction factors to improve the resolution of the simulation
import pickle
params = pickle.load( open( "camels_25_64_pkloss.params", "rb" ) )



model = hk.without_apply_rng(
    hk.transform(lambda x, a: NeuralSplineFourierFilter(n_knots=16,
                                                        latent_size=32)(x, a)))


def linear_field(mesh_shape, box_size, pk):
  """
    Generate initial conditions.
    """
  kvec = fftk(mesh_shape)
  kmesh = sum(
      (kk / box_size[i] * mesh_shape[i])**2 for i, kk in enumerate(kvec))**0.5
  pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (
      box_size[0] * box_size[1] * box_size[2])

  field = numpyro.sample(
      'initial_conditions',
      dist.Normal(jnp.zeros(mesh_shape), jnp.ones(mesh_shape)))

  field = jnp.fft.rfftn(field) * pkmesh**0.5
  field = jnp.fft.irfftn(field)
  return field


def get_density_planes(
    cosmology,
    density_plane_width=100.,  # In Mpc/h
    density_plane_npix=256,  # Number of pixels
    density_plane_smoothing=3.,  # In Mpc/h
    box_size=[400., 400., 4000.],  # In Mpc/h
    nc=[32, 32, 320],
    neural_spline_params=None):
  """Function that returns tomographic density planes
  for a given cosmology from a lightcone.

  Args:
    cosmology: jax-cosmo object
    density_plane_width: width of the output density slices
    density_plane_npix: size of the output density slices
    density_plane_smoothing: Gaussian scale of plane smoothing
    box_size: [sx,sy,sz] size in Mpc/h of the simulation volume
    nc: number of particles/voxels in the PM scheme
    neural_spline_params: optional parameters for neural correction of PM scheme
  Returns:
    list of [r, a, plane], slices through the lightcone along with their
        comoving distance (r) and scale factors (a). Each slice "plane" is a
        2d array of size density_plane_npix^2
  """
  # Initial scale factor for the simulation
  a_init = 0.01

  # Planning out the scale factor stepping to extract desired lensplanes
  n_lens = int(box_size[-1] // density_plane_width)
  r = jnp.linspace(0., box_size[-1], n_lens + 1)
  r_center = 0.5 * (r[1:] + r[:-1])
  a_center = jc.background.a_of_chi(cosmology, r_center)

  # Create a small function to generate the matter power spectrum
  k = jnp.logspace(-4, 1, 256)
  pk = jc.power.linear_matter_power(cosmology, k)
  pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk
                                                ).reshape(x.shape)

  # Create initial conditions
  initial_conditions = linear_field(nc, box_size, pk_fn)

  # Create particles
  particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in nc]),
                        axis=-1).reshape([-1, 3])

  # Initial displacement
  cosmology._workspace = {}  # FIX ME: this a temporary fix
  dx, p, f = lpt(cosmology, initial_conditions, particles, a=a_init)

  @jax.jit
  def neural_nbody_ode(a, state, args):
    """
      state is a tuple (position, velocities ) in internal units: [grid units, v=\frac{a^2}{H_0}\dot{x}]
      See this link for conversion rules: https://github.com/fastpm/fastpm#units
      """
    cosmo, params = args
    pos = state[0]
    vel = state[1]

    kvec = fftk(nc)

    delta = cic_paint(jnp.zeros(nc), pos)

    delta_k = jnp.fft.rfftn(delta)

    # Computes gravitational potential
    pot_k = delta_k * laplace_kernel(kvec) * longrange_kernel(kvec, r_split=0)

    # Apply a correction filter
    if params is not None:
      kk = jnp.sqrt(sum((ki / jnp.pi)**2 for ki in kvec))
      pot_k = pot_k * (1. + model.apply(params, kk, jnp.atleast_1d(a)))

    # Computes gravitational forces
    forces = jnp.stack([
        cic_read(jnp.fft.irfftn(gradient_kernel(kvec, i) * pot_k), pos)
        for i in range(3)
    ],
                       axis=-1)

    forces = forces * 1.5 * cosmo.Omega_m

    # Computes the update of position (drift)
    dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

    # Computes the update of velocity (kick)
    dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

    return jnp.stack([dpos, dvel], axis=0)

  # Define the function that will save the density planes as we are going 
  # through the lightcone
  def density_plane_fn(t, y, args):
    cosmo, _ = args
    positions = y[0]
    nx, ny, nz = nc

    # Converts time t to comoving distance in voxel coordinates
    w = density_plane_width / box_size[2] * nc[2]
    center = jc.background.radial_comoving_distance(cosmo, t) / box_size[2] * nc[2]

    xy = positions[..., :2]
    d = positions[..., 2]

    # Apply 2d periodic conditions
    xy = jnp.mod(xy, nx)

    # Rescaling positions to target grid
    xy = xy / nx * density_plane_npix

    # Selecting only particles that fall inside the volume of interest
    weight = jnp.where((d > (center - w / 2)) & (d <= (center + w / 2)), 1., 0.)

    # Painting density plane
    density_plane = cic_paint_2d(jnp.zeros([density_plane_npix, density_plane_npix]), xy, weight)

    # Apply density normalization
    density_plane = density_plane / ((nx / density_plane_npix) *
                                     (ny / density_plane_npix) * w)

    return density_plane

  # Evolve the simulation forward
  term = ODETerm(neural_nbody_ode)
  solver = Dopri5()
  saveat = SaveAt(ts=a_center[::-1], fn=density_plane_fn)
  # stepsize_controller = PIDController(rtol=1e-3, atol=1e-3)

  solution = diffeqsolve(term, solver, t0=a_init, t1=1., dt0=0.05,
                       y0=jnp.stack([particles+dx, p], axis=0),
                       args=(cosmology, neural_spline_params),
                       saveat=saveat,
                       adjoint=diffrax.RecursiveCheckpointAdjoint(5),
                       max_steps=32)
                      #  stepsize_controller=stepsize_controller) 

  dx = box_size[0] / density_plane_npix
  dz = density_plane_width

  # Apply some amount of gaussian smoothing defining the effective resolution of
  # the density planes
  density_plane = jax.vmap(lambda x: gaussian_smoothing(x, 
                                           density_plane_smoothing / dx ))(solution.ys)
  return {'planes': density_plane[::-1],
          'a': solution.ts[::-1],
          'a2': a_center,
          'r': r_center,
          'dx':dx,
          'dz':dz}


from jax.scipy.ndimage import map_coordinates
from jaxpm.utils import gaussian_smoothing
import jax_cosmo.constants as constants

def convergence_Born(cosmo,
                     density_planes,
                     coords,
                     z_source):
  """
  Compute the Born convergence
  Args:
    cosmo: `Cosmology`, cosmology object.
    density_planes: list of dictionaries (r, a, density_plane, dx, dz), lens planes to use 
    coords: a 3-D array of angular coordinates in radians of N points with shape [batch, N, 2].
    z_source: 1-D `Tensor` of source redshifts with shape [Nz] .
    name: `string`, name of the operation.
  Returns:
    `Tensor` of shape [batch_size, N, Nz], of convergence values.
  """
  # Compute constant prefactor:
  constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
  # Compute comoving distance of source galaxies
  r_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))

  convergence = 0
  n_planes = len(density_planes['planes'])
  for i in range(n_planes):

    r = density_planes['r'][i]; a = density_planes['a'][i]; p = density_planes['planes'][i]
    dx = density_planes['dx']; dz = density_planes['dz']
    # Normalize density planes
    density_normalization = dz * r / a
    p = (p - p.mean()) * constant_factor * density_normalization

    # Interpolate at the density plane coordinates
    im = map_coordinates(p,
                         coords * r / dx - 0.5,
                         order=1, mode="wrap")

    convergence += im * jnp.clip(1. - (r / r_s), 0, 1000).reshape([-1, 1, 1])

  return convergence

def forward_model(box_size=[200., 200., 2000.], # In Mpc/h
                  nc = [100, 100, 1000],         # Number of pixels
                  field_size = 5,            # Size of the lensing field in degrees
                  field_npix = 256,           # Number of pixels in the lensing field
                  sigma_e = 0.25,             # Standard deviation of galaxy ellipticities
                  galaxy_density = 27./5,       # Galaxy density per arcmin^2, per redshift bin
                  ):
  """
  This function defines the top-level forward model for our observations
  """
  # Sampling cosmological parameters and defines cosmology
  Omega_c = numpyro.sample('omega_c', dist.TruncatedNormal(0.,1, low=-1))*0.2 + 0.25
  sigma_8 = numpyro.sample('sigma_8', dist.Normal(0., 1.))*0.14 + 0.831
  Omega_b = 0.04
  h = 0.7
  n_s = 0.96
  w0 = -1
  cosmo = jc.Cosmology(Omega_c=Omega_c, sigma8=sigma_8, Omega_b=Omega_b,
                       h=h, n_s=n_s, w0=w0, Omega_k=0., wa=0.)

  # Generate lightcone density planes through an nbody
  density_planes = get_density_planes(cosmo, box_size=box_size, nc=nc,
                                      neural_spline_params=params,
                                      density_plane_npix=512,
                                      density_plane_smoothing=0.75,
                                      density_plane_width=100.
                                      )

  # # Create photoz systematics parameters, and create derived nz
  # nzs_s_sys = [jc.redshift.systematic_shift(nzi,
  #                                           numpyro.sample('dz%d'%i, dist.Normal(0., 0.01)),
  #                                           zmax=2.5)
  #               for i, nzi in enumerate(nz_shear)]

  # Defining the coordinate grid for lensing map
  xgrid, ygrid = np.meshgrid(np.linspace(0, field_size, field_npix, endpoint=False), # range of X coordinates
                             np.linspace(0, field_size, field_npix, endpoint=False)) # range of Y coordinates
  coords = jnp.array((np.stack([xgrid, ygrid], axis=0)*u.deg).to(u.rad))

  # Generate convergence maps by integrating over nz and source planes
  convergence_maps = [simps(lambda z: nz(z).reshape([-1,1,1]) *
                              convergence_Born(cosmo, density_planes, coords, z), 0.01, 1.0, N=32)
                      for nz in nz_shear]

  # Apply noise to the maps (this defines the likelihood)
  observed_maps = [numpyro.sample('kappa_%d'%i,
                                  dist.Normal(k, sigma_e/jnp.sqrt(galaxy_density*(field_size*60/field_npix)**2)))
                   for i,k in enumerate(convergence_maps)]

  return observed_maps

# condition the model on a given set of parameters
fiducial_model = numpyro.handlers.condition(forward_model, {'omega_c': 0., # remember this is reparamd
                                                            'sigma_8': 0.})

# sample a mass map and save corresponding true parameters
model_trace = numpyro.handlers.trace(numpyro.handlers.seed(fiducial_model, jax.random.PRNGKey(42))).get_trace()

import numpyro
from numpyro.handlers import seed, trace, condition, reparam
from numpyro.infer.reparam import LocScaleReparam, TransformReparam
import numpyro.distributions as dist
from functools import partial

def config(x):
    if type(x['fn']) is dist.TransformedDistribution:
        return TransformReparam()
    elif type(x['fn']) is dist.Normal and ('decentered' not in x['name']) and ('kappa' not in x['name']):
        return LocScaleReparam(centered=0)
    else:
        return None

# ok, cool, now let's sample this posterior
observed_model = condition(forward_model, {'kappa_0': model_trace['kappa_0']['value'],
                                           'kappa_1': model_trace['kappa_1']['value'],
                                           'kappa_2': model_trace['kappa_2']['value'],
                                           'kappa_3': model_trace['kappa_3']['value']})
observed_model_reparam = observed_model # reparam(observed_model, config=config)


nuts_kernel = numpyro.infer.NUTS(
    model=observed_model_reparam,
    init_strategy=partial(numpyro.infer.init_to_value, values={'omega_c': 0.,
                                                               'sigma_8': 0.,
							       'initial_conditions': model_trace['initial_conditions']['value']}),
    max_tree_depth=3,
    step_size=2e-2)

mcmc = numpyro.infer.MCMC(
       nuts_kernel,
       num_warmup=0,
       num_samples=10,
       # chain_method="parallel", num_chains=8,
       # thinning=2,
       progress_bar=True
    )

mcmc.run(jax.random.PRNGKey(0))
import pickle
# First 50 samples
res = mcmc.get_samples()
with open('lensing_fwd_mdl_nbody_0.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

for i in range(4):
    print('round',i,'done')
    mcmc.post_warmup_state = mcmc.last_state
    mcmc.run(mcmc.post_warmup_state.rng_key)
    res = mcmc.get_samples()
    with open('lensing_fwd_mdl_nbody_%d.pickle'%(i+1), 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

