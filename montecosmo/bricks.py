import numpyro
import numpyro.distributions as dist
from numpyro import sample, deterministic
import jax.numpy as jnp
import jax_cosmo as jc
from jax_cosmo import Cosmology
from jaxpm.kernels import fftk
from jaxpm.painting import cic_read
from jaxpm.growth import growth_factor, growth_rate
from jaxpm.pm import pm_forces




def get_cosmo(prior_config, trace_reparam=False, **params_) -> dict:
    """
    Return cosmology from latent values.
    """
    cosmo = {}
    for name in ['Omega_c', 'sigma8']:
        _, mean, std = prior_config[name]
        value = params_[name+'_'] * std + mean
        if trace_reparam:
            value = deterministic(name, value)
        cosmo[name] = value
    return cosmo


## To reparametrize automaticaly
# from numpyro.infer.reparam import LocScaleReparam
#     reparam_config = {'Omega_c': LocScaleReparam(centered=0),
#                       'sigma8': LocScaleReparam(centered=0)}
#     with numpyro.handlers.reparam(config=reparam_config):
#         Omega_c = sample('Omega_c', dist.Normal(0.25, 0.2**2))
#         sigma8 = sample('sigma8', dist.Normal(0.831, 0.14**2))


def get_init_mesh(cosmo:Cosmology, mesh_size, box_size, trace_reparam=False, **params_):
    """
    Return initial conditions at a=1 from latent values.
    """
    # Compute initial power spectrum
    pk_fn = linear_pk_interp(cosmo, n_interp=256)
    kvec = fftk(mesh_size)
    kmesh = sum((ki  * (m / l))**2 for ki, m, l in zip(kvec, mesh_size, box_size))**0.5
    pkmesh = pk_fn(kmesh) * (mesh_size.prod() / box_size.prod()) # NOTE: convert from (Mpc/h)^3 to cell units

    # Parametrize
    init_mesh = jnp.fft.rfftn(params_['init_mesh_']) * pkmesh**0.5

    # k_nyquist = jnp.pi * jnp.min(mesh_size / box_size)
    # init_mesh = init_mesh * jnp.exp(-.5 * kmesh**2 / k_nyquist**2)
    # from jax import debug
    # debug.print('init smoothed')
    
    init_mesh = jnp.fft.irfftn(init_mesh)

    if trace_reparam:
        init_mesh = deterministic('init_mesh', init_mesh)
    return dict(init_mesh=init_mesh)


def get_biases(prior_config, trace_reparam=False, **params_) -> dict:
    """
    Return biases from latent values.
    """
    biases = {}
    for name in ['b1', 'b2', 'bs2', 'bn2']:
        _, mean, std = prior_config[name]
        value = params_[name+'_'] * std + mean
        if trace_reparam:
            value = deterministic(name, value)
        biases[name] = value
    return biases


def lagrangian_weights(cosmo:Cosmology, a, pos, box_size, 
                       b1, b2, bs2, bn2, init_mesh, **params):
    """
    Return Lagrangian bias expansion weight as in [Modi+2020](http://arxiv.org/abs/1910.07097).
    .. math::
        
        w = 1 + b_1 \delta + b_2 \left(\delta^2 - \braket{\delta^2}\right) + b_{s^2} \left(s^2 - \braket{s^2}\right) + b_{\nabla^2} \nabla^2 delta
    """    
    # Get init_mesh at observation scale factor
    a = jnp.atleast_1d(a)
    init_mesh = init_mesh * growth_factor(cosmo, a)

    # mesh_size = init_mesh.shape
    # delta_k = jnp.fft.rfftn(init_mesh)
    # kvec = fftk(mesh_size)
    # k_nyquist = jnp.pi * jnp.min(mesh_size / box_size)
    # kk_box = sum((ki  * (m / l))**2 for ki, m, l in zip(kvec, mesh_size, box_size))
    # delta_k = delta_k * jnp.exp(-.5 * kk_box / k_nyquist**2)
    # init_mesh = jnp.fft.irfftn(delta_k)

    weights = 1
    
    # Apply b1
    delta_part = cic_read(init_mesh, pos)
    weights = weights + b1 * delta_part

    # Apply b2
    delta_sqr_part = delta_part**2
    weights = weights + b2 * (delta_sqr_part - delta_sqr_part.mean())

    # Apply bshear2
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
    weights = weights + bs2 * (shear_sqr_part - shear_sqr_part.mean())

    # Apply bnabla2
    kk_box = sum((ki  * (m / l))**2 
                 for ki, m, l in zip(kvec, mesh_size, box_size)) # laplace kernel in h/Mpc physical units
    delta_nl = jnp.fft.irfftn(kk_box * delta_k)

    delta_nl_part = cic_read(delta_nl, pos)
    weights = weights + bn2 * delta_nl_part

    # jax.debug.print('Number of strict negative weights={i}', i=(weights<0).sum())
    return weights


def linear_pk_interp(cosmo:Cosmology, a=1., n_interp=256):
    """
    Return a light emulation of the linear matter power spectrum.
    """
    k = jnp.logspace(-4, 1, n_interp)
    pk = jc.power.linear_matter_power(cosmo, k, a=a)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape(-1), k, pk).reshape(x.shape)
    return pk_fn


def get_ode_fn(cosmo:Cosmology, mesh_size):

    def nbody_ode(a, state, args):
        """
        state is a phase space state array [*position, *velocities]
        """
        pos, vel = state[:,:3], state[:,3:]
        forces = pm_forces(pos, mesh_shape=mesh_size) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel
        
        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return jnp.concatenate((dpos, dvel), axis=-1)

    return nbody_ode


def rsd(cosmo:Cosmology, a, p, los=jnp.array([0,0,1])):
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


def kaiser_weights(cosmo:Cosmology, a, mesh_size, los):
    b = sample('b', dist.Normal(2, 0.25))
    a = jnp.atleast_1d(a)

    kvec = fftk(mesh_size)
    kmesh = sum(kk**2 for kk in kvec)**0.5

    mumesh = sum(ki*losi for ki, losi in zip(kvec, los))
    kmesh_nozeros = jnp.where(kmesh==0, 1, kmesh) 
    mumesh = mumesh / kmesh_nozeros 
    mumesh = jnp.where(kmesh==0, 0, mumesh)

    return b + growth_rate(cosmo, a) * mumesh**2


def apply_kaiser_bias(cosmo:Cosmology, a, init_mesh, los=jnp.array([0,0,1])):
    # Get init_mesh at observation scale factor
    a = jnp.atleast_1d(a)
    init_mesh = init_mesh * growth_factor(cosmo, a)

    # Apply eulerian kaiser bias weights
    weights = kaiser_weights(cosmo, a, init_mesh.shape, los)
    delta_k = jnp.fft.rfftn(init_mesh)
    kaiser_mesh = jnp.fft.irfftn(weights * delta_k)
    return kaiser_mesh





