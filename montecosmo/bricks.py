import numpyro
import numpyro.distributions as dist
from numpyro import sample, deterministic
import jax.numpy as jnp
import jax_cosmo as jc
from jaxpm.kernels import fftk
from jaxpm.painting import cic_read
from jaxpm.growth import growth_factor, growth_rate

import jax

def cosmo_prior(trace_reparam=False):
    """
    Defines a cosmological prior to sample from.
    """
    # Omega_c_base = sample('Omega_c_base', dist.TruncatedNormal(0,1, low=-1))
    Omega_c_base = sample('Omega_c_base', dist.Normal(0,1))
    sigma8_base = sample('sigma8_base', dist.Normal(0, 1))
    Omega_c = Omega_c_base * 0.2 + 0.25
    sigma8 = sigma8_base * 0.14 + 0.831

    if trace_reparam:
        Omega_c = deterministic('Omega_c', Omega_c)
        sigma8 = deterministic('sigma8', sigma8)

    cosmo_params = {'Omega_c':Omega_c, 'sigma8':sigma8}
    # deterministic('cosmo_params', cosmo_params) # NOTE: does not seem to work properly
    cosmology = jc.Planck15(**cosmo_params)
    return cosmology

# from numpyro.infer.reparam import LocScaleReparam
#     reparam_config = {'Omega_c': LocScaleReparam(centered=0),
#                       'sigma8': LocScaleReparam(centered=0)}
#     with numpyro.handlers.reparam(config=reparam_config):
#         Omega_c = sample('Omega_c', dist.Normal(0.25, 0.2**2))
#         sigma8 = sample('sigma8', dist.Normal(0.831, 0.14**2))


def linear_field(mesh_size, box_size, pk, trace_reparam=False):
    """
    Generate initial conditions.
    """
    kvec = fftk(mesh_size)
    kmesh = sum((ki  * (m / l))**2 for ki, m, l in zip(kvec, mesh_size, box_size))**0.5
    pkmesh = pk(kmesh) * (mesh_size.prod() / box_size.prod()) # NOTE: convert from (Mpc/h)^3 to cell units

    field = sample('init_mesh_base', dist.Normal(jnp.zeros(mesh_size), jnp.ones(mesh_size)))

    field = jnp.fft.rfftn(field) * pkmesh**0.5
    field = jnp.fft.irfftn(field)

    if trace_reparam:
        field = deterministic('init_mesh', field)
    return field


def lagrangian_bias(cosmo, a, init_mesh, pos, box_size, trace_reparam=False):
    """
    Compute Lagrangian bias expansion weights as in [Modi+2020](http://arxiv.org/abs/1910.07097).
    .. math::
        
        w = 1 + b_1 \delta + b_2 \left(\delta^2 - \braket{\delta^2}\right) + b_s \left(s^2 - \braket{s^2}\right) + b_{\text{nl}} \nabla^2 delta
    """
    b1_base = sample('b1_base', dist.Normal(0,1))
    b2_base = sample('b2_base', dist.Normal(0, 1))
    bs_base = sample('bs_base', dist.Normal(0, 1))
    bnl_base = sample('bnl_base', dist.Normal(0, 1))
    b1 = 0.5 * b1_base + 1
    b2 = 5 * b2_base
    bs = 5 * bs_base
    bnl = 5 * bnl_base

    if trace_reparam:
        b1 = deterministic('b1', b1)
        b2 = deterministic('b2', b2)
        bs = deterministic('bs', bs)
        bnl = deterministic('bnl', bnl)

    # Get init_mesh at observation scale factor
    a = jnp.atleast_1d(a)
    init_mesh = init_mesh * growth_factor(cosmo, a)


    # Apply b1
    weights = 1
    delta_part = cic_read(init_mesh, pos)
    weights = weights + b1 * delta_part

    # Apply b2
    delta_sqr_part = delta_part**2
    weights = weights + b2 * (delta_sqr_part - delta_sqr_part.mean())

    # Apply bshear
    delta_k = jnp.fft.rfftn(init_mesh)
    mesh_size = init_mesh.shape
    kvec = fftk(mesh_size)

    kk = sum(ki**2 for ki in kvec)
    kk_nozeros = jnp.where(kk==0, 1, kk) 
    pot_k = delta_k / kk_nozeros 
    pot_k = jnp.where(kk==0, 0, pot_k) # inverse laplace kernel

    shear_sqr = 0  
    for i, ki in enumerate(kvec):
        # Add diagonal terms
        shear_sqr = shear_sqr + jnp.fft.irfftn(ki**2 * pot_k - delta_k / 3)**2
        for kj in kvec[i+1:]:
            # Add upper triangle terms (counted twice)
            shear_sqr = shear_sqr + 2 * jnp.fft.irfftn(ki * kj * pot_k)**2

    shear_sqr_part = cic_read(shear_sqr, pos)
    weights = weights + bs * (shear_sqr_part - shear_sqr_part.mean())

    # Apply bnl
    kk_box = sum((ki  * (m / l))**2 
                 for ki, m, l in zip(kvec, mesh_size, box_size)) # laplace kernel in physical units
    delta_nl = jnp.fft.irfftn(kk_box * delta_k)

    delta_nl_part = cic_read(delta_nl, pos)
    weights = weights + bnl * delta_nl_part

    return weights


def linear_pk_interp(cosmo, a=1, n_interp=256):
    """
    Return a light emulation of the linear matter power spectrum.
    """
    k = jnp.logspace(-4, 1, n_interp)
    pk = jc.power.linear_matter_power(cosmo, k, a=a)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape(-1), k, pk).reshape(x.shape)
    return pk_fn


def rsd(cosmo, a, p, los=jnp.array([0,0,1])):
    """
    Redshift-Space Distortion (RSD) displacement from cosmology and Particle Mesh (PM) momentum.
    Computed with respect scale factor and line-of-sight.
    """
    a = jnp.atleast_1d(a)
    los = los / jnp.linalg.norm(los)
    # Divide PM momentum by `a` once to retrieve velocity, and once again for comobile velocity  
    dx_rsd = p / (jnp.sqrt(jc.background.Esqr(cosmo, a)) * a**2)
    # Project velocity on line-of-sight
    dx_rsd = dx_rsd * los
    return dx_rsd


def kaiser_bias(cosmo, a, mesh_size, los):
    b = sample('b', dist.Normal(2, 0.25))
    a = jnp.atleast_1d(a)

    kshapes = jnp.eye(len(mesh_size), dtype=jnp.int16) * -2 + 1
    # kvec = fftk(mesh_size)
    kvec = [2 * jnp.pi *jnp.fft.fftfreq(m).reshape(kshape)
            for m, kshape in zip(mesh_size, kshapes)]
    kmesh = sum(kk**2 for kk in kvec)**0.5

    mumesh = sum(ki*losi for ki, losi in zip(kvec, los))
    kmesh_nozeros = jnp.where(kmesh==0, 1, kmesh) 
    mumesh = mumesh / kmesh_nozeros 
    mumesh = jnp.where(kmesh==0, 0, mumesh)

    return b + growth_rate(cosmo, a) * mumesh**2


def apply_kaiser_bias(cosmo, a, init_mesh, los=jnp.array([0,0,1])):
    # Get init_mesh at observation scale factor
    a = jnp.atleast_1d(a)
    init_mesh = init_mesh * growth_factor(cosmo, a)

    # Apply eulerian kaiser bias weights
    kaiser_weights = kaiser_bias(cosmo, a, init_mesh.shape, los)
    delta_k = jnp.fft.fftn(init_mesh)
    kaiser_mesh = jnp.fft.ifftn(kaiser_weights * delta_k)
    return kaiser_mesh