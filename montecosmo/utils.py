import jax.numpy as jnp
from jaxpm.kernels import cic_compensation, fftk


def cic_compensate(mesh):
  """
  Compensate for CIC painting convolution.
  Only use for computing spectra, as it can increase numerical instability if used in modeling.
  """
  kmesh = jnp.fft.rfftn(mesh)
  kmesh = kmesh * cic_compensation(fftk(mesh.shape))
  comp_mesh = jnp.fft.irfftn(kmesh)
  return comp_mesh


# TODO: Run MCMC function here