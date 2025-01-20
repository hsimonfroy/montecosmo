from functools import partial
import numpy as np

from jax import numpy as jnp, tree, debug, lax
import jax_cosmo as jc
from jax_cosmo import Cosmology
from montecosmo.utils import ch2rshape

# from jaxpm.pm import pm_forces
import jax_cosmo as jc
from jaxpm.growth import (growth_factor, growth_rate, 
                          growth_factor_second, growth_rate_second,
                          _growth_factor_ODE)
from jaxpm.kernels import longrange_kernel, cic_compensation
from jaxpm.painting import cic_paint, cic_read


def rfftk(shape):
    """
    Return wavevectors in cell units for rfftn.
    """
    kx = np.fft.fftfreq(shape[0]) * 2 * np.pi
    ky = np.fft.fftfreq(shape[1]) * 2 * np.pi
    kz = np.fft.rfftfreq(shape[2]) * 2 * np.pi

    kx = kx.reshape([-1, 1, 1])
    ky = ky.reshape([1, -1, 1])
    kz = kz.reshape([1, 1, -1])

    return kx, ky, kz


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


def pm_forces(pos, mesh_shape, mesh=None, grad_fd=True, lap_fd=False, r_split=0):
    """
    Computes gravitational forces on particles using a PM scheme
    """
    if mesh is None:
        delta_k = jnp.fft.rfftn(cic_paint(jnp.zeros(mesh_shape), pos))
    elif jnp.isrealobj(mesh):
        delta_k = jnp.fft.rfftn(mesh)
    else:
        delta_k = mesh

    # Computes gravitational potential
    kvec = rfftk(mesh_shape)
    pot_k = delta_k * invlaplace_kernel(kvec, lap_fd) * longrange_kernel(kvec, r_split=r_split)

    # If painted field, double deconvolution to account for both painting and reading 
    if mesh is None:
        pot_k *= cic_compensation(kvec)**2
        print("deconv")

    # Computes gravitational forces
    return jnp.stack([cic_read(jnp.fft.irfftn(- gradient_kernel(kvec, i, grad_fd) * pot_k), pos) 
                      for i in range(3)], axis=-1)



def lpt(cosmo:Cosmology, init_mesh, pos, a, order=1, grad_fd=True, lap_fd=False):
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

    init_force = pm_forces(pos, mesh_shape, mesh=delta_k, grad_fd=grad_fd, lap_fd=lap_fd)
    dq = growth_factor(cosmo, a) * init_force
    p = a**2 * growth_rate(cosmo, a) * E * dq
    # f = a**2 * E * dGfa(cosmo, a) * init_force

    if order == 2:
        kvec = rfftk(mesh_shape)
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

        init_force2 = pm_forces(pos, mesh_shape, mesh=jnp.fft.rfftn(delta2), grad_fd=grad_fd, lap_fd=lap_fd)
        dq2 = (3/7 * growth_factor_second(cosmo, a)) * init_force2 # D2 is renormalized: - D2 = 3/7 * growth_factor_second
        p2 = (a**2 * growth_rate_second(cosmo, a) * E) * dq2
        # f2 = (a**2 * E * dGf2a(cosmo, a) * 3/7) * init_force2

        dq += dq2
        p  += p2
        # f  += f2

    # return dq, p, f
    return dq, p






def g2a(cosmo, g):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0))
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(g, cache["g"], cache["a"])

def g2gg(cosmo, g):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0))
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(g, cache["g"], cache["g2"])

def g2f(cosmo, g):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0))
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(g, cache["g"], cache["f"])

def g2ff(cosmo, g):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0))
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(g, cache["g"], cache["f2"])



def get_bullfrog(cosmo:Cosmology, mesh_shape, dg, grad_fd=True, lap_fd=False):

    def dggdg(cosmo, g):
        gg, f, ff = g2gg(cosmo, g)*-3/7, g2f(cosmo, g), g2ff(cosmo, g)
        return jnp.where(g==0., 0., gg * ff / (g * f))
    
    def alpha(cosmo, g0, dg):
        '''See Eq. 2.3 in [List and Hahn, 2024](https://arxiv.org/abs/2106.00461)'''
        g1 = g0 + dg / 2
        g2 = g0 + dg

        dggdg0, dggdg2 = dggdg(cosmo, g0), dggdg(cosmo, g2)
        # NOTE: linearization of ratio (gg - g^2)/g aroung g0, evaluated at g1
        lin_ratio = (g2gg(cosmo, g0)*-3/7 + dggdg0 * dg / 2) / g1 - g1
        return (dggdg2 - lin_ratio) / (dggdg0 - lin_ratio)

    def kick(state, g0, cosmo, dg):
        pos, vel = state
        g1 = g0 + dg / 2
        forces = pm_forces(pos, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd)
        alph = alpha(cosmo, g0, dg)
        return pos, alph * vel + (1 - alph) * forces / g1
        # return pos, vel + (1 - alph) * (forces / g1 - vel) # equivalent

    def drift(state, dg):
        pos, vel = state
        return pos + vel * dg / 2, vel
    
    def step_fn(state, g0):
        state = drift(state, dg)
        state = kick(state, g0, cosmo, dg)
        state = drift(state, dg)
        return state, None
    
    return step_fn



def nbody_bf(cosmo:Cosmology, init_mesh, pos, a, snapshots=None, n_steps=5,
              grad_fd=True, lap_fd=False):
            #  mesh_shape, particles, a_obs, snapshots=None, n_steps=5):
    if jnp.isrealobj(init_mesh):
        delta_k = jnp.fft.rfftn(init_mesh)
        mesh_shape = init_mesh.shape
    else:
        delta_k = init_mesh
        mesh_shape = ch2rshape(init_mesh.shape)

    vel = pm_forces(pos, mesh_shape, mesh=delta_k, grad_fd=grad_fd, lap_fd=lap_fd)
    state = pos, vel
    
    g_obs = g2a(cosmo, a)
    dg = g_obs / n_steps
    step_fn = get_bullfrog(cosmo, mesh_shape, dg, grad_fd=grad_fd, lap_fd=lap_fd)
    gs = jnp.arange(n_steps) * dg

    state, _ = lax.scan(step_fn, state, gs)
    # for g in gs:
    #     state, _ = step_fn(state, g)
    return jnp.stack(state)



















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



from diffrax import diffeqsolve, ODETerm, SaveAt, Euler, Heun, Dopri5, Tsit5, PIDController, ConstantStepSize
def nbody_tsit5(cosmo:Cosmology, mesh_shape, particles, a_lpt, a_obs, snapshots=None, tol=1e-2,
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
                                stepsize_controller=controller, max_steps=100, saveat=saveat)
        particles = sol.ys
        debug.print("n_solvsteps: {n}", n=sol.stats['num_steps'])
        return particles


from montecosmo.fpm import EfficientLeapFrog, LeapFrogODETerm, symplectic_ode
def nbody_fpm(cosmo:Cosmology, mesh_shape, particles, a_lpt, a_obs, snapshots=None, n_steps=5,
           grad_fd=True, lap_fd=False):
    
    solver = EfficientLeapFrog(initial_t0=a_lpt, final_t1=a_obs, cosmo=cosmo)
    stepsize_controller = ConstantStepSize()
    terms = tree.map(
        LeapFrogODETerm,
        symplectic_ode(mesh_shape, paint_absolute_pos=False),
    )
    cosmo._workspace = {}
    args = cosmo

    if snapshots is None or (isinstance(snapshots, int) and snapshots < 2): 
        saveat = SaveAt(t1=True)
    elif isinstance(snapshots, int): 
        saveat = SaveAt(ts=jnp.linspace(a_lpt, a_obs, snapshots))   
    else: 
        saveat = SaveAt(ts=jnp.asarray(snapshots))   

    sol = diffeqsolve(
            terms,
            solver=solver,
            t0=a_lpt,
            t1=a_obs,
            dt0=(a_obs - a_lpt) / n_steps,
            y0=(*particles,),
            args=args,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=10,
            # progress_meter=TqdmProgressMeter(refresh_steps=2),
            # adjoint=BacksolveAdjoint(solver=solver),
        )

    particles = sol.ys
    return particles


