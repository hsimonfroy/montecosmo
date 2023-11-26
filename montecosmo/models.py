import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import jax_cosmo as jc
from jaxpm.kernels import fftk



def cosmo_prior(trace_deterministic=False):
  """
  Defines a cosmological prior to sample from.
  """
  Omega_c_base = numpyro.sample('Omega_c_base', dist.TruncatedNormal(0,1, low=-1))
  sigma8_base = numpyro.sample('sigma8_base', dist.Normal(0, 1))
  Omega_c = Omega_c_base * 0.2 + 0.25
  sigma8 = sigma8_base * 0.14 + 0.831

  if trace_deterministic:
    Omega_c = numpyro.deterministic('Omega_c', Omega_c)
    sigma8 = numpyro.deterministic('sigma8', sigma8)

  cosmo_params = {'Omega_c':Omega_c, 'sigma8':sigma8}
  # numpyro.deterministic('cosmo_params', cosmo_params)
  cosmology = jc.Planck15(**cosmo_params)
  return cosmology

# from numpyro.infer.reparam import LocScaleReparam
#   reparam_config = {'Omega_c': LocScaleReparam(centered=0),
#                     'sigma8': LocScaleReparam(centered=0)}
#   with numpyro.handlers.reparam(config=reparam_config):
#     Omega_c = numpyro.sample('Omega_c', dist.Normal(0.25, 0.2**2))
#     sigma8 = numpyro.sample('sigma8', dist.Normal(0.831, 0.14**2))


def linear_field(mesh_size, box_size, pk):
  """
  Generate initial conditions.
  """
  kvec = fftk(mesh_size)
  kmesh = sum((kk / box_size[i] * mesh_size[i])**2 for i, kk in enumerate(kvec))**0.5
  pkmesh = pk(kmesh) * (mesh_size.prod() / box_size.prod())

  field = numpyro.sample('init_mesh_base', dist.Normal(jnp.zeros(mesh_size), jnp.ones(mesh_size)))

  field = jnp.fft.rfftn(field) * pkmesh**0.5
  field = jnp.fft.irfftn(field)
  return field


def linear_pk_interp(cosmology, scale_factor=1, n_interp=256):
  """
  Return a light emulation of the linear matter power spectrum
  """
  k = jnp.logspace(-4, 1, n_interp)
  pk = jc.power.linear_matter_power(cosmology, k, a=scale_factor)
  pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)
  return pk_fn


