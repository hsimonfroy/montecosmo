# from argparse import ArgumentParser
# from pmesh.pm import ParticleMesh
# ap = ArgumentParser()
# ap.add_argument("config")

# from astropy.cosmology import Planck15
# # from nbodykit.cosmology import Planck15
# from nbodykit.cosmology import EHPower
# from nbodykit.cosmology import Cosmology
# from nbodykit.lab import FFTPower, FieldMesh

from numpy.testing import assert_allclose

from jax import numpy as jnp
import numpy as np
from pmesh.pm import ParticleMesh
from functools import partial
from jax_cosmo import Cosmology, background
import jax_cosmo as jc
from fastpm.core import Solver as Solver
from fastpm.core import leapfrog

from jax import random as jr, debug
from jax.experimental.ode import odeint
from jaxpm.painting import cic_paint
from jaxpm.pm import linear_field
from montecosmo.bricks import lpt as mylpt, get_ode_fn, make_ode_fn

from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController, Euler, Heun, Dopri5




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

a_obs = 1.0
mesh_shape = np.array([1, 1 ,1]) * 64
box_shape = np.array([1., 1. ,1.]) * 640


class Cosmo():
    def __init__(self, cosmo: Cosmology):
        self.cosmo = cosmo

    @property
    def H0(self):
        return self.cosmo.h * 100
    
    @property
    def Om0(self):
        return self.cosmo.Omega_m
    
    def efunc(self, z):
        return background.Esqr(self.cosmo, (1+z)**-1)**.5
    
    def Onu(self, z):
        return np.zeros_like(z)
    



def nbody(cosmo:Cosmology, mesh_shape, particles, a_lpt, a_obs, 
          grad_fd, lap_fd, trace_meshes=2, tol=1e-5):
    terms = ODETerm(get_ode_fn(cosmo, mesh_shape, grad_fd, lap_fd))
    solver = Dopri5()
    controller = PIDController(rtol=tol, atol=tol, pcoeff=0.4, icoeff=1, dcoeff=0)
    if trace_meshes < 2: 
        saveat = SaveAt(t1=True)
    else: 
        saveat = SaveAt(ts=jnp.linspace(a_lpt, a_obs, trace_meshes))      
    sol = diffeqsolve(terms, solver, a_lpt, a_obs, dt0=None, y0=particles,
                            stepsize_controller=controller, max_steps=8, saveat=saveat)
    particles = sol.ys
    debug.print("n_solvsteps: {n}", n=sol.stats['num_steps'])
    return particles


def run_jpm(cosmo, init_mesh, a_lpt, a_obs, lpt_order=1, grad_fd=True, lap_fd=False, tol=1e-5):
    # Initial displacement
    x_part = jnp.indices(mesh_shape).reshape(3,-1).T
    cosmo._workspace = {}  # FIX ME: this a temporary fix
    dx, p_part, f = mylpt(cosmo, init_mesh, x_part, a_lpt, lpt_order, grad_fd, lap_fd)

    if a_obs == a_lpt:
        pos = x_part + dx
    else:
        # With odeint
        # res = odeint(make_ode_fn(mesh_shape, grad_fd, lap_fd), jnp.stack([x_part+dx, p_part]), 
        #              jnp.linspace(a_lpt, a_obs, 2), cosmo, rtol=tol, atol=tol)
        
        # With diffrax
        res = nbody(cosmo, mesh_shape, jnp.stack([x_part + dx, p_part]), a_lpt, a_obs, 
                    grad_fd, lap_fd, trace_meshes=2, tol=tol)
        
        pos, p = res[-1]

    return cic_paint(jnp.zeros(mesh_shape), pos)
    




def test_nbody(a_lpt, a_obs, lpt_order, tol=1e-5):
    """
    Run end to end nbodies
    """
    meshes = []
    cosmo = Planck18()
    ref_cosmo = Cosmo(cosmo)

    pm = ParticleMesh(BoxSize=box_shape, Nmesh=mesh_shape, dtype='f4')
    grid = pm.generate_uniform_particle_grid(shift=0).astype(np.float32)
    solver = Solver(pm, ref_cosmo, B=1)
    stages = np.linspace(a_lpt, a_obs, 10, endpoint=True)

    # Initial power spectrum
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmo, k)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    # FastPM init to JaxPM init
    whitec = pm.generate_whitenoise(42, mode='complex', unitary=False)
    lineark = whitec.apply(lambda k, v: pk_fn(
        sum(ki**2 for ki in k)**0.5)**0.5 * v * (1 / v.BoxSize).prod()**0.5)
    init_mesh = lineark.c2r().value # XXX
    
    # JaxPM init to FastPM init
    # init_mesh = linear_field(mesh_shape, box_shape, pk_fn, seed=jr.key(0)) # XXX
    # lineark.value = np.fft.rfftn(init_mesh, norm='ortho') / np.prod(mesh_shape)**.5 # XXX



    # Run FastPM
    statelpt = solver.lpt(lineark, grid, a_lpt, order=lpt_order)
    finalstate = solver.nbody(statelpt, leapfrog(stages))
    fpm_mesh = pm.paint(finalstate.X)
    meshes.append(fpm_mesh)

    # Run JaxPM
    jpm_mesh = run_jpm(cosmo, init_mesh, a_lpt, a_obs, lpt_order, tol=tol)
    meshes.append(jpm_mesh)

    # assert_allclose(meshes[0], meshes[1], atol=1.2)

    return meshes




for lpt_order in [1,2]:
    for pm in [0,1]:
        if pm==0:
            a_lpt = a_obs
        else:
            a_lpt = 0.1
        print(f"CONFIG {lpt_order=}, {pm=}, {a_lpt=}, {a_obs=}")

        meshes = test_nbody(a_lpt, a_obs, lpt_order)
        jnp.save(f"meshes_lpt{lpt_order}_pm{pm}_{mesh_shape[0]}.npy", meshes)




