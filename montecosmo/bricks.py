from functools import partial
import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro import sample, deterministic

from jax import numpy as jnp, debug
import jax_cosmo as jc
from jax_cosmo import Cosmology
from jaxpm.kernels import fftk
from jaxpm.painting import cic_read
from jaxpm.growth import growth_factor, growth_rate
from jaxpm.pm import pm_forces

from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController, Euler, Heun, Dopri5
from montecosmo.utils import std2trunc, trunc2std, rg2cgh, cgh2rg




   

def get_cosmo(prior_config, 
              trace_reparam=False, inverse=False, scaling=1., **params_) -> dict:
    """
    Return cosmological params from latent params.
    """
    cosmo = {}
    for name in ['Omega_m', 'sigma8']:
        label, loc, scale = prior_config[name]
        scale *= scaling

        if not inverse:
            input_name, output_name = name+'_', name
            trunc_push = std2trunc # truncate value in interval
            notrunc_push = lambda x : x * scale + loc
        else:
            input_name, output_name = name, name+'_'
            trunc_push = trunc2std
            notrunc_push = lambda x : (x - loc) / scale

        value = params_[input_name]
        if name == 'Omega_m':
            value = trunc_push(value, loc, scale, Planck18.keywords['Omega_b'], 1) # Omega_m > 0.05 > Omega_b
        elif name == 'sigma8':
            value = trunc_push(value, loc, scale, 0)
        else:
            value = notrunc_push(value)

        if trace_reparam:
            value = deterministic(output_name, value)
        cosmo[output_name] = value
    return cosmo

## To reparametrize automaticaly
# from numpyro.infer.reparam import LocScaleReparam
#     reparam_config = {'Omega_m': LocScaleReparam(centered=0),
#                       'sigma8': LocScaleReparam(centered=0)}
#     with numpyro.handlers.reparam(config=reparam_config):
#         Omega_m = sample('Omega_m', dist.Normal(0.25, 0.2**2))
#         sigma8 = sample('sigma8', dist.Normal(0.831, 0.14**2))


def get_cosmology(**cosmo) -> Cosmology:
    """
    Return full cosmology object from cosmological params.
    """
    return Planck18(Omega_c = cosmo['Omega_m'] - Planck18.keywords['Omega_b'], 
                    sigma8 = cosmo['sigma8'])


def get_init_mesh(cosmo:Cosmology, mesh_shape, box_shape, fourier=False,
                  trace_reparam=False, inverse=False, scaling=1., **params_) -> dict:
    """
    Return initial conditions at a=1 from latent params.
    """
    # Compute initial power spectrum
    pk_fn = linear_pk_interp(cosmo, n_interp=256)
    kvec = fftk(mesh_shape)
    k_box = sum((ki  * (m / l))**2 for ki, m, l in zip(kvec, mesh_shape, box_shape))**0.5
    pk_mesh = pk_fn(k_box) * (mesh_shape / box_shape).prod() # NOTE: convert from (Mpc/h)^3 to cell units
    pk_mesh *= scaling**2


    # Parametrize
    name = 'init_mesh'
    if not inverse:
        input_name, output_name = name+'_', name
        init = params_[input_name]
        if fourier:
            delta_k = rg2cgh(init)
        else:
            delta_k = jnp.fft.rfftn(init)
        delta_k *= pk_mesh**0.5
        init = jnp.fft.irfftn(delta_k)
    
    else:
        input_name, output_name = name, name+'_'
        delta_k = jnp.fft.rfftn(params_[input_name])
        delta_k /= pk_mesh**0.5   
        if fourier:
            init = cgh2rg(delta_k)
        else:
            init = jnp.fft.irfftn(delta_k)

    if trace_reparam:
        init = deterministic(output_name, init)
    return {output_name:init}


def get_biases(prior_config, 
               trace_reparam=False, inverse=False, scaling=1., **params_) -> dict:
    """
    Return biases params from latent params.
    """
    biases = {}
    for name in ['b1', 'b2', 'bs2', 'bn2']:
        _, loc, scale = prior_config[name]
        scale *= scaling

        if not inverse:
            input_name, output_name = name+'_', name
            notrunc_push = lambda x : x * scale + loc
        else:
            input_name, output_name = name, name+'_'
            notrunc_push = lambda x : (x - loc) / scale

        value = notrunc_push(params_[input_name])

        if trace_reparam:
            value = deterministic(output_name, value)
        biases[output_name] = value
    return biases


def lagrangian_weights(cosmo:Cosmology, a, pos, box_shape, 
                       b1, b2, bs2, bn2, init_mesh, **params):
    """
    Return Lagrangian bias expansion weights as in [Modi+2020](http://arxiv.org/abs/1910.07097).
    .. math::
        
        w = 1 + b_1 \\delta + b_2 \\left(\\delta^2 - \\braket{\\delta^2}\\right) + b_{s^2} \\left(s^2 - \\braket{s^2}\\right) + b_{\\nabla^2} \\nabla^2 \\delta
    """    
    # Get init_mesh at observation scale factor
    init_mesh = init_mesh * growth_factor(cosmo, jnp.atleast_1d(a))
    if jnp.isrealobj(init_mesh):
        delta = init_mesh
        delta_k = jnp.fft.rfftn(delta)
    else:
        delta_k = init_mesh
        delta = jnp.fft.irfftn(delta_k)
    # Smooth field to mitigate negative weights or TODO: use gaussian lagrangian biases
    # k_nyquist = jnp.pi * jnp.min(mesh_shape / box_shape)
    # delta_k = delta_k * jnp.exp( - kk_box / k_nyquist**2)
    # delta = jnp.fft.irfftn(delta_k)

    mesh_shape = delta.shape
    kvec = fftk(mesh_shape)
    kk_box = sum((ki  * (m / l))**2 
                 for ki, m, l in zip(kvec, mesh_shape, box_shape)) # minus laplace kernel in h/Mpc physical units

    # Init weights
    weights = 1
    
    # Apply b1, punctual term
    delta_part = cic_read(delta, pos)
    weights = weights + b1 * delta_part

    # Apply b2, punctual term
    delta2_part = delta_part**2
    weights = weights + b2 * (delta2_part - delta2_part.mean())

    # Apply bshear2, non-punctual term
    pot_k = delta_k * invlaplace_kernel(kvec)

    shear2 = 0  
    for i, ki in enumerate(kvec):
        # Add diagonal terms
        shear2 = shear2 + jnp.fft.irfftn( - ki**2 * pot_k - delta_k / 3)**2
        for kj in kvec[i+1:]:
            # Add strict-up-triangle terms (counted twice)
            shear2 = shear2 + 2 * jnp.fft.irfftn( - ki * kj * pot_k)**2

    shear2_part = cic_read(shear2, pos)
    weights = weights + bs2 * (shear2_part - shear2_part.mean())

    # Apply bnabla2, non-punctual term
    delta_nl = jnp.fft.irfftn( - kk_box * delta_k)

    delta_nl_part = cic_read(delta_nl, pos)
    weights = weights + bn2 * delta_nl_part

    return weights


def linear_pk_interp(cosmo:Cosmology, a=1., n_interp=256):
    """
    Return a light emulation of the linear matter power spectrum.
    """
    k = jnp.logspace(-4, 1, n_interp)
    pk = jc.power.linear_matter_power(cosmo, k, a=a)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape(-1), k, pk).reshape(x.shape)
    return pk_fn


def nbody(cosmo:Cosmology, mesh_shape, particles, a_lpt, a_obs, trace_meshes):
    terms = ODETerm(get_ode_fn(cosmo, mesh_shape))
    solver = Dopri5()
    # controller = PIDController(rtol=1e-5, atol=1e-5, pcoeff=0.4, icoeff=1, dcoeff=0)
    controller = PIDController(rtol=1e-2, atol=1e-2, pcoeff=0.4, icoeff=1, dcoeff=0)
    if trace_meshes < 2: 
        saveat = SaveAt(t1=True)
    else: 
        saveat = SaveAt(ts=jnp.linspace(a_lpt, a_obs, trace_meshes))      
    sol = diffeqsolve(terms, solver, a_lpt, a_obs, dt0=None, y0=particles,
                            stepsize_controller=controller, max_steps=8, saveat=saveat)
    particles = sol.ys
    # debug.print("n_solvsteps: {n}", n=sol.stats['num_steps'])
    return particles




# from jaxpm.pm import pm_forces
import jax_cosmo as jc
from jaxpm.growth import growth_factor, growth_rate, dGfa, growth_factor_second, growth_rate_second, dGf2a
from jaxpm.kernels import fftk, longrange_kernel
from jaxpm.painting import cic_paint, cic_read

# Planck 2015 paper XIII Table 4 final column (best fit)
Planck15 = partial(Cosmology,
    Omega_c=0.2589,
    Omega_b=0.04860,
    Omega_k=0.0,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    w0=-1.0,
    wa=0.0,)

# Planck 2018 paper VI Table 2 final column (best fit)
Planck18 = partial(Cosmology,
    # Omega_m = 0.3111
    Omega_c=0.2607,
    Omega_b=0.0490,
    Omega_k=0.0,
    h=0.6766,
    n_s=0.9665,
    sigma8=0.8102,
    w0=-1.0,
    wa=0.0,)


def get_ode_fn(cosmo:Cosmology, mesh_shape, grad_order=1):

    def nbody_ode(a, state, args):
        """
        state is a phase space state array [*position, *velocities]
        """
        pos, vel = state
        forces = pm_forces(pos, mesh_shape, grad_order=grad_order) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel
        
        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return jnp.stack([dpos, dvel])

    return nbody_ode


def make_ode_fn(mesh_shape, grad_order=1, lap_order=1):
    
    def nbody_ode(state, a, cosmo):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state

        forces = pm_forces(pos, mesh_shape, grad_order=grad_order, lap_order=lap_order) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel
        
        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces
        
        return dpos, dvel

    return nbody_ode



def invlaplace_kernel(kvec, order=1):
    """
    Compute the inverse Laplace kernel.

    cf. [Feng+2016](https://arxiv.org/pdf/1603.00476)

    Parameters
    -----------
    kvec: list
        List of wave-vectors

    Returns
    --------
    wts: array
        Complex kernel values
    """
    if order == 0:
        kk = sum(ki**2 for ki in kvec)
    elif order == 1:
        kk = sum((ki * np.sinc(ki / (2 * np.pi)))**2 for ki in kvec)
    kk_nozeros = np.where(kk==0, 1, kk) 
    return - np.where(kk==0, 0, 1 / kk_nozeros)


def gradient_kernel(kvec, direction, order=1):
    """
    Computes the gradient kernel in the requested direction
    
    Parameters
    -----------
    kvec: list
        List of wave-vectors in Fourier space
    direction: int
        Index of the direction in which to take the gradient

    Returns
    --------
    wts: array
        Complex kernel values
    """
    ki = kvec[direction]
    if order == 0:
        pass
    elif order ==1:
        ki = (8. * np.sin(ki) - np.sin(2 * ki)) / 6.
    return 1j * ki




def pm_forces(positions, mesh_shape, mesh=None, grad_order=1, lap_order=1, r_split=0):
    """
    Computes gravitational forces on particles using a PM scheme
    """
    if mesh is None:
        delta_k = jnp.fft.rfftn(cic_paint(jnp.zeros(mesh_shape), positions))
    elif jnp.isrealobj(mesh):
        delta_k = jnp.fft.rfftn(mesh)
    else:
        delta_k = mesh

    # Computes gravitational potential
    kvec = fftk(mesh_shape)
    pot_k = delta_k * invlaplace_kernel(kvec, lap_order) * longrange_kernel(kvec, r_split=r_split)
    # Computes gravitational forces
    return jnp.stack([cic_read(jnp.fft.irfftn(- gradient_kernel(kvec, i, grad_order) * pot_k), positions) 
                      for i in range(3)], axis=-1)


def lpt(cosmo:Cosmology, init_mesh, positions, a, order=1, grad_order=1, lap_order=1):
    """
    Computes first and second order LPT displacement, e.g. Eq. 2 and 3 [Jenkins2010](https://arxiv.org/pdf/0910.0258)
    """
    a = jnp.atleast_1d(a)
    E = jnp.sqrt(jc.background.Esqr(cosmo, a)) 
    delta_k = jnp.fft.rfftn(init_mesh)
    mesh_shape = init_mesh.shape

    init_force = pm_forces(positions, mesh_shape, mesh=delta_k, grad_order=grad_order, lap_order=lap_order)
    dx = growth_factor(cosmo, a) * init_force
    p = a**2 * growth_rate(cosmo, a) * E * dx
    f = a**2 * E * dGfa(cosmo, a) * init_force
    debug.print("grad_order: {grad_order}, lap_order: {lap}", grad_order=grad_order, lap=lap_order)

    if order == 2:
        kvec = fftk(mesh_shape)
        pot_k = delta_k * invlaplace_kernel(kvec, lap_order)

        delta2 = 0
        shear_acc = 0
        # for i, ki in enumerate(kvec):
        for i in range(3):
            # Add products of diagonal terms = 0 + s11*s00 + s22*(s11+s00)...
            # shear_ii = jnp.fft.irfftn(- ki**2 * pot_k)
            nabla_i_nabla_i = gradient_kernel(kvec, i, grad_order)**2
            shear_ii = jnp.fft.irfftn(nabla_i_nabla_i * pot_k)
            delta2 += shear_ii * shear_acc 
            shear_acc += shear_ii

            # for kj in kvec[i+1:]:
            for j in range(i+1, 3):
                # Substract squared strict-up-triangle terms
                # delta2 -= jnp.fft.irfftn(- ki * kj * pot_k)**2
                nabla_i_nabla_j = gradient_kernel(kvec, i, grad_order) * gradient_kernel(kvec, j, grad_order)
                delta2 -= jnp.fft.irfftn(nabla_i_nabla_j * pot_k)**2

        
        init_force2 = pm_forces(positions, mesh_shape, mesh=jnp.fft.rfftn(delta2), grad_order=grad_order)
        dx2 = 3/7 * growth_factor_second(cosmo, a) * init_force2 # D2 is renormalized: - D2 = 3/7 * growth_factor_second
        p2 = a**2 * growth_rate_second(cosmo, a) * E * dx2
        f2 = a**2 * E * dGf2a(cosmo, a) * init_force2

        dx += dx2
        p  += p2
        f  += f2

    return dx, p, f






def rsd(cosmo:Cosmology, a, p, los=[0,0,1]):
    """
    Redshift-Space Distortion (RSD) displacement from cosmology and Particle Mesh (PM) momentum.
    Computed with respect scale factor and line-of-sight.
    """
    a = jnp.atleast_1d(a)
    los = jnp.asarray(los)
    los = los / np.linalg.norm(los)
    # Divide PM momentum by `a` once to retrieve velocity, and once again for comobile velocity  
    dx_rsd = p / (jc.background.Esqr(cosmo, a)**.5 * a**2)
    # Project velocity on line-of-sight
    dx_rsd = dx_rsd * los
    return dx_rsd


def kaiser_weights(cosmo:Cosmology, a, mesh_shape, los):
    b = sample('b', dist.Normal(2, 0.25))
    a = jnp.atleast_1d(a)
    los = jnp.asarray(los)
    los = los / np.linalg.norm(los)

    kvec = fftk(mesh_shape)
    kmesh = sum(kk**2 for kk in kvec)**0.5

    mumesh = sum(ki*losi for ki, losi in zip(kvec, los))
    kmesh_nozeros = jnp.where(kmesh==0, 1, kmesh) 
    mumesh = mumesh / kmesh_nozeros 
    mumesh = jnp.where(kmesh==0, 0, mumesh)

    return b + growth_rate(cosmo, a) * mumesh**2


def apply_kaiser_bias(cosmo:Cosmology, a, init_mesh, los=[0,0,1]):
    # Get init_mesh at observation scale factor
    a = jnp.atleast_1d(a)
    init_mesh = init_mesh * growth_factor(cosmo, a)

    # Apply eulerian kaiser bias weights
    weights = kaiser_weights(cosmo, a, init_mesh.shape, los)
    delta_k = jnp.fft.rfftn(init_mesh)
    kaiser_mesh = jnp.fft.irfftn(weights * delta_k)
    return kaiser_mesh


