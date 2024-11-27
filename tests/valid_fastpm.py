# from argparse import ArgumentParser
# from pmesh.pm import ParticleMesh
# ap = ArgumentParser()
# ap.add_argument("config")

# from astropy.cosmology import Planck15
# # from nbodykit.cosmology import Planck15
# from nbodykit.cosmology import EHPower
# from nbodykit.cosmology import Cosmology
# from nbodykit.lab import FFTPower, FieldMesh
#
from jax import numpy as jnp
import numpy as np
from pmesh.pm import ParticleMesh
from functools import partial
from jax_cosmo import Cosmology, background
import jax_cosmo as jc
from fastpm.core import Solver as Solver
from fastpm.core import leapfrog

from jax import random as jr
from jax.experimental.ode import odeint
from jaxpm.painting import cic_paint
from jaxpm.pm import linear_field
from montecosmo.bricks import lpt as mylpt, make_ode_fn


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
    

def run_jpm(cosmo, init_mesh, a_lpt, a_obs, lpt_order=1, grad_order=1, lap_order=0):
    # Initial displacement
    particles = jnp.indices(mesh_shape).reshape(3,-1).T
    cosmo._workspace = {}  # FIX ME: this a temporary fix
    dx, p, f = mylpt(cosmo, init_mesh, particles, a_lpt, lpt_order, grad_order, lap_order)

    if a_obs == a_lpt:
        pos = particles + dx
    else:
        # Evolve the simulation forward
        snapshots = jnp.linspace(a_lpt, a_obs, 2)
        res = odeint(make_ode_fn(mesh_shape, grad_order, lap_order), jnp.stack([particles+dx, p]), snapshots, cosmo, rtol=1e-5, atol=1e-5)
        pos, p = res[-1]

    return cic_paint(jnp.zeros(mesh_shape), pos)
    




def test_nbody(a_lpt, a_obs, lpt_order):
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
    # go, lo = 1, 0
    for go in [0,1]:
        for lo in [0,1]:
            print(f"Running JaxPM with grad_order={go}, lap_order={lo}")
            jpm_mesh = run_jpm(cosmo, init_mesh, a_lpt, a_obs, lpt_order, go, lo)
            meshes.append(jpm_mesh)

    # assert_allclose(final_cube, tfread[0], atol=1.2)

    return meshes




for lpt_order in [2]:
    for pm in [0]:
        if pm==0:
            a_lpt = a_obs
        else:
            a_lpt = 0.1
        print(f"CONFIG {lpt_order=}, {pm=}, {a_lpt=}, {a_obs=}")

        meshes = test_nbody(a_lpt, a_obs, lpt_order)
        jnp.save(f"meshes_lpt{lpt_order}_pm{pm}_{mesh_shape[0]}_test.npy", meshes)




