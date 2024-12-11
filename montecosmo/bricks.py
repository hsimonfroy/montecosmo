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

from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController, Euler, Heun, Dopri5, Tsit5
from montecosmo.utils import std2trunc, trunc2std, rg2cgh, cgh2rg, ch2rshape, r2chshape



def lin_power_interp(cosmo:Cosmology, a=1., n_interp=256):
    """
    Return a light emulation of the linear matter power spectrum.
    """
    k = jnp.logspace(-4, 1, n_interp)
    pk = jc.power.linear_matter_power(cosmo, k, a=a)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape(-1), k, pk).reshape(x.shape)
    return pk_fn

def lin_power_mesh(cosmo:Cosmology, mesh_shape, box_shape, a=1., n_interp=256):
    """
    Return linear matter power spectrum field.
    """
    pk_fn = lin_power_interp(cosmo, a=a, n_interp=n_interp)
    kvec = fftk(mesh_shape)
    k_box = sum((ki  * (m / l))**2 for ki, m, l in zip(kvec, mesh_shape, box_shape))**0.5
    return pk_fn(k_box) * (mesh_shape / box_shape).prod() # NOTE: convert from (Mpc/h)^3 to cell units

def gausslin_posterior(obs_meshk, cosmo:Cosmology, a, box_shape, gxy_count):
    """
    Return posterior mean and std fields of the linear matter field (at a=1) given the observed field,
    by assuming Gaussian linear model. All fields are in harmonic space.
    """
    # Compute linear matter power spectrum
    mesh_shape = ch2rshape(obs_meshk.shape)
    pmeshk = lin_power_mesh(cosmo, mesh_shape, box_shape)

    D1 = growth_factor(cosmo, jnp.atleast_1d(a))
    stds = (gxy_count * D1**2 + pmeshk**-1)**-.5
    means = stds**2 * gxy_count * D1 * obs_meshk
    return means, stds, pmeshk



def get_cosmology(**cosmo) -> Cosmology:
    """
    Return full cosmology object from cosmological params.
    """
    return Planck18(Omega_c = cosmo['Omega_m'] - Planck18.keywords['Omega_b'], 
                    sigma8 = cosmo['sigma8'])

def samp2base(params:dict, config, inv=False, temp=1.) -> dict:
    """
    Transform sample params into base params.
    """
    out = {}
    for in_name, value in params.items():
        name = in_name if inv else in_name[:-1]
        out_name = in_name+'_' if inv else in_name[:-1]

        loc, scale = config[name]['loc'], config[name]['scale']
        low, high = config[name].get('low', -jnp.inf), config[name].get('high', jnp.inf)
        scale *= temp**.5

        # Reparametrize
        if not inv:
            if low != -jnp.inf or high != jnp.inf:
                push = lambda x: std2trunc(x, loc, scale, low, high) # truncate value in interval
            else:
                push = lambda x: x * scale + loc
        else:
            if low != -jnp.inf or high != jnp.inf:
                push = lambda x: trunc2std(x, loc, scale, low, high)
            else:
                push = lambda x: (x - loc) / scale

        out[out_name] = push(value)
    return out

def samp2base_mesh(init:dict, cosmo:Cosmology, box_shape, precond=False, 
                   guide=None, inv=False, temp=1.) -> dict:
    """
    Transform sample mesh into base mesh, i.e. initial wavevectors at a=1.
    """
    assert len(init) <= 1, "init dict should only have one or zero key"
    for in_name, mesh in init.items():
        out_name = in_name+'_' if inv else in_name[:-1]
        mesh_shape = ch2rshape(mesh.shape) if inv else mesh.shape

        # Reparametrize
        if not inv:
            if precond in [0, 1, 2]:

                if precond==0:
                    # Sample in direct space
                    mesh = jnp.fft.rfftn(mesh) # ~ G(0, I)

                elif precond==1:
                    # Sample in harmonic space
                    mesh = rg2cgh(mesh) # ~ G(0, I)

                elif precond==2:
                    # Sample in harmonic space with
                    # partial (and static) posterior preconditioning assuming Gaussian linear model and fiducial cosmology
                    # as done in [Bayer+2023](http://arxiv.org/abs/2307.09504)
                    mesh = rg2cgh(mesh) # ~ G(0, I + n * P_fid(a_obs))
                    mesh /= guide # ~ G(0, I) ; guide = (I + n * P_fid(a_obs))^1/2)
                
                # Compute linear matter power spectrum
                pmeshk = lin_power_mesh(cosmo, mesh_shape, box_shape, a=1.)
                mesh *= pmeshk**.5 # ~ G(0, P)

            elif precond==3:
                # Sample in harmonic space with
                # complete (and dynamic) posterior preconditioning assuming Gaussian linear model
                means, stds = guide # sigma = (n * D^2 + P^-1)^-1/2 ; mu = sigma^2 * n * D * delta_obs
                mesh = rg2cgh(mesh) # ~ G( -mu * sigma^-1, sigma^-2 * P) 
                mesh = stds * mesh + means # ~ G(0, P)

            mesh *= temp**.5
        else:
            mesh /= temp**.5
            if precond in [0, 1, 2]:

                pmeshk = lin_power_mesh(cosmo, mesh_shape, box_shape, a=1.)
                mesh /= pmeshk**.5 # ~ G(0, I)

                if precond==0:
                    mesh = jnp.fft.irfftn(mesh)

                elif precond==1:
                    mesh = cgh2rg(mesh)

                elif precond==2:
                    mesh *= guide # ~ G(0, I + n * P_fid(a_obs))
                    mesh = cgh2rg(mesh)

            elif precond==3:          
                means, stds = guide # sigma = (n * D^2 + P^-1)^-1/2 ; mu = sigma^2 * n * D * delta_obs
                mesh = (mesh - means) / stds # ~ G( -mu * sigma^-1, sigma^-2 * P) 
                mesh = cgh2rg(mesh)

        return {out_name:mesh}
    return {}






def lagrangian_weights(cosmo:Cosmology, a, pos, box_shape, 
                       b1, b2, bs2, bn2, init_mesh):
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



def nbody(cosmo:Cosmology, mesh_shape, particles, a_lpt, a_obs, snapshots=None, tol=1e-3,
           grad_fd=True, lap_fd=False):
    if a_lpt == a_obs:
        return particles[None]
    else:
        terms = ODETerm(get_ode_fn(cosmo, mesh_shape, grad_fd, lap_fd))
        solver = Tsit5() # Tsit5 usually better than Dopri5
        controller = PIDController(rtol=tol, atol=tol, pcoeff=0.4, icoeff=1, dcoeff=0)

        if snapshots is None or (isinstance(snapshots, int) and snapshots < 2): 
            saveat = SaveAt(t1=True)
        elif isinstance(snapshots, int): 
            saveat = SaveAt(ts=jnp.linspace(a_lpt, a_obs, snapshots))   
        else: 
            saveat = SaveAt(ts=jnp.asarray(snapshots))   

        sol = diffeqsolve(terms, solver, a_lpt, a_obs, dt0=None, y0=particles,
                                stepsize_controller=controller, max_steps=10, saveat=saveat)
        particles = sol.ys
        debug.print("n_solvsteps: {n}", n=sol.stats['num_steps'])
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


def get_ode_fn(cosmo:Cosmology, mesh_shape,  grad_fd=True, lap_fd=False):

    def nbody_ode(a, state, args):
        """
        state is a phase space state array [*position, *velocities]
        """
        pos, vel = state
        forces = pm_forces(pos, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel
        
        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return jnp.stack([dpos, dvel])

    return nbody_ode


def make_ode_fn(mesh_shape, grad_fd=True, lap_fd=False):
    
    def nbody_ode(state, a, cosmo):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state

        forces = pm_forces(pos, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel
        
        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces
        
        return dpos, dvel

    return nbody_ode



def invlaplace_kernel(kvec, fd=False):
    """
    Compute the inverse Laplace kernel.

    cf. [Feng+2016](https://arxiv.org/pdf/1603.00476)

    Parameters
    -----------
    kvec: list
        List of wave-vectors
    fd: bool
        Finite difference kernel

    Returns
    --------
    wts: array
        Complex kernel values
    """
    if fd:
        kk = sum((ki * np.sinc(ki / (2 * np.pi)))**2 for ki in kvec)
    else:
        kk = sum(ki**2 for ki in kvec)
    kk_nozeros = np.where(kk==0, 1, kk) 
    return - np.where(kk==0, 0, 1 / kk_nozeros)


def gradient_kernel(kvec, direction, fd=False):
    """
    Computes the gradient kernel in the requested direction
    
    Parameters
    -----------
    kvec: list
        List of wave-vectors in Fourier space
    direction: int
        Index of the direction in which to take the gradient
    fd: bool
        Finite difference kernel

    Returns
    --------
    wts: array
        Complex kernel values
    """
    ki = kvec[direction]
    if fd:
        ki = (8. * np.sin(ki) - np.sin(2. * ki)) / 6.
    return 1j * ki




def pm_forces(positions, mesh_shape, mesh=None, grad_fd=True, lap_fd=False, r_split=0):
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
    pot_k = delta_k * invlaplace_kernel(kvec, lap_fd) * longrange_kernel(kvec, r_split=r_split)
    # Computes gravitational forces
    return jnp.stack([cic_read(jnp.fft.irfftn(- gradient_kernel(kvec, i, grad_fd) * pot_k), positions) 
                      for i in range(3)], axis=-1)


def lpt(cosmo:Cosmology, init_mesh, positions, a, order=1, grad_fd=True, lap_fd=False):
    """
    Computes first and second order LPT displacement, e.g. Eq. 2 and 3 [Jenkins2010](https://arxiv.org/pdf/0910.0258)
    """
    a = jnp.atleast_1d(a)
    E = jc.background.Esqr(cosmo, a)**.5
    if jnp.isrealobj(init_mesh):
        delta_k = jnp.fft.rfftn(init_mesh)
        mesh_shape = init_mesh.shape
    else:
        delta_k = init_mesh
        mesh_shape = ch2rshape(init_mesh.shape)

    init_force = pm_forces(positions, mesh_shape, mesh=delta_k, grad_fd=grad_fd, lap_fd=lap_fd)
    dq = growth_factor(cosmo, a) * init_force
    p = a**2 * growth_rate(cosmo, a) * E * dq
    f = a**2 * E * dGfa(cosmo, a) * init_force

    if order == 2:
        kvec = fftk(mesh_shape)
        pot_k = delta_k * invlaplace_kernel(kvec, lap_fd)

        delta2 = 0
        shear_acc = 0
        for i in range(3):
            # Add products of diagonal terms = 0 + s11*s00 + s22*(s11+s00)...
            shear_ii = gradient_kernel(kvec, i, grad_fd)**2
            shear_ii = jnp.fft.irfftn(shear_ii * pot_k)
            delta2 += shear_ii * shear_acc 
            shear_acc += shear_ii

            for j in range(i+1, 3):
                # Substract squared strict-up-triangle terms
                hess_ij = gradient_kernel(kvec, i, grad_fd) * gradient_kernel(kvec, j, grad_fd)
                delta2 -= jnp.fft.irfftn(hess_ij * pot_k)**2

        init_force2 = pm_forces(positions, mesh_shape, mesh=jnp.fft.rfftn(delta2), grad_fd=grad_fd)
        dq2 = (growth_factor_second(cosmo, a) * 3/7) * init_force2 # D2 is renormalized: - D2 = 3/7 * growth_factor_second
        p2 = (a**2 * growth_rate_second(cosmo, a) * E) * dq2
        f2 = (a**2 * E * dGf2a(cosmo, a) * 3/7) * init_force2

        dq += dq2
        p  += p2
        f  += f2

    return dq, p, f






def rsd(cosmo:Cosmology, a, p, los=[0,0,1]):
    """
    Redshift-Space Distortion (RSD) displacement from cosmology and Particle Mesh (PM) momentum.
    Computed with respect scale factor and line-of-sight.
    """
    a = jnp.atleast_1d(a)
    los = np.asarray(los)
    los = los / np.linalg.norm(los)
    # Divide PM momentum by scale factor once to retrieve velocity, and once again for comobile velocity  
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
    kmesh = sum(kk**2 for kk in kvec)**0.5 # in cell units

    mumesh = sum(ki*losi for ki, losi in zip(kvec, los))
    kmesh_nozeros = jnp.where(kmesh==0, 1, kmesh) 
    mumesh = jnp.where(kmesh==0, 0, mumesh / kmesh_nozeros )

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


