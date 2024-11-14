# from argparse import ArgumentParser
# from pmesh.pm import ParticleMesh
# ap = ArgumentParser()
# ap.add_argument("config")

# from fastpm.core import Solver
# from fastpm.core import leapfrog
# from fastpm.core import autostages
# from fastpm.background import PerturbationGrowth

# from astropy.cosmology import Planck15
# # from nbodykit.cosmology import Planck15
# from nbodykit.cosmology import EHPower
# from nbodykit.cosmology import Cosmology
# from nbodykit.lab import FFTPower, FieldMesh
# import numpy

from jax import numpy as jnp
import numpy as np
from pmesh.pm import ParticleMesh
from functools import partial
from jax_cosmo import Cosmology, background
import jax_cosmo as jc
from fastpm.core import Solver as Solver
# import fastpm.force.lpt as fpmops
from fastpm.core import leapfrog
print("hey")



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

box_length = 640.
mesh_length = 64

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
    





def test_nbody():
    """ Checking end to end nbody
    """
    a0 = 0.1
    cosmo = Planck18()
    ref_cosmo = Cosmo(cosmo)

    pm = ParticleMesh(BoxSize=box_length, Nmesh=3*[mesh_length], dtype='f4')
    grid = pm.generate_uniform_particle_grid(shift=0).astype(np.float32)
    solver = Solver(pm, ref_cosmo, B=1)
    stages = np.linspace(0.1, 1.0, 10, endpoint=True)

    # Generate initial state with fastpm
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmo, k)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    whitec = pm.generate_whitenoise(42, mode='complex', unitary=False)
    lineark = whitec.apply(lambda k, v: pk_fn(
        sum(ki**2 for ki in k)**0.5)**0.5 * v / v.BoxSize.prod()**0.5)
    statelpt = solver.lpt(lineark, grid, a0, order=1)
    finalstate = solver.nbody(statelpt, leapfrog(stages))
    final_cube = pm.paint(finalstate.X)

    return final_cube

    # # Same thing with flowpm
    # tlinear = jnp.expand_dims(np.array(lineark.c2r()), 0)
    # state = tfpm.lpt_init(cosmo, tlinear, a0, order=1)
    # state = tfpm.nbody(cosmo, state, stages, nc)
    # tfread = pmutils.cic_paint(tf.zeros_like(tlinear), state[0]).numpy()

    # assert_allclose(final_cube, tfread[0], atol=1.2)


final_cube = test_nbody()


print("ho")






# _kws = {'ln10^{10}A_s':3.047, 'n_s':0.9665, 'k_pivot':0.05, 'tau_reio':0.066, 
#         # 'H0':67.66, 'Om0':0.3097, 'Ob0':0.0490,
#         }
# Planck18 = Cosmology.from_astropy(Planck15, **_kws)
# """Planck18 instance of FlatLambdaCDM cosmology
# Planck 2018 paper VI Table 2 final column (best fit)
# """

# class Config(dict):
#     def __init__(self, path):
#         self.prefix = '%s' % path
#         filename = self.makepath('config.py')

#         self['boxsize'] = 640.0
#         self['shift'] = 0.0
#         self['nc'] = 256
#         self['ndim'] = 3
#         self['seed'] = 1985
#         self['pm_nc_factor'] = 1 # 2
#         self['resampler'] = 'tsc'
#         self['cosmology'] = Planck18
#         self['powerspectrum'] = EHPower(Planck18, 0)
#         self['unitary'] = False
#         self['stages'] = numpy.linspace(0.1, 1.0, 5, endpoint=True)
#         self['aout'] = [1.0]

#         local = {} # these names will be usable in the config file
#         local['EHPower'] = EHPower
#         local['Cosmology'] = Cosmology
#         local['Planck15'] = Planck15
#         local['linspace'] = numpy.linspace
#         local['autostages'] = autostages

#         import nbodykit.lab as nlab
#         local['nlab'] = nlab

#         names = set(self.__dict__.keys())

#         exec(open(filename).read(), local, self)

#         unknown = set(self.__dict__.keys()) - names
#         assert len(unknown) == 0

#         self.finalize()
#         global _config
#         _config = self

#     def finalize(self):
#         self['aout'] = numpy.array(self['aout'])

#         self.pm = ParticleMesh(BoxSize=self['boxsize'], Nmesh= [self['nc']] * self['ndim'], resampler=self['resampler'])
#         mask = numpy.array([ a not in self['stages'] for a in self['aout']], dtype='?')
#         missing_stages = self['aout'][mask]
#         if len(missing_stages):
#             raise ValueError('Some stages are requested for output but missing: %s' % str(missing_stages))

#     def makepath(self, filename):
#         import os.path
#         return os.path.join(self.prefix, filename)

# def main(args=None):
#     ns = ap.parse_args(args)
#     config = Config(ns.config)

#     solver = Solver(config.pm, cosmology=config['cosmology'], B=config['pm_nc_factor'])
#     whitenoise = solver.whitenoise(seed=config['seed'], unitary=config['unitary'])
#     dlin = solver.linear(whitenoise, Pk=lambda k : config['powerspectrum'](k))

#     Q = config.pm.generate_uniform_particle_grid(shift=config['shift'])

#     state = solver.lpt(dlin, Q=Q, a=config['stages'][0], order=2)

#     def write_power(d, path, a):
#         meshsource = FieldMesh(d)
#         r = FFTPower(meshsource, mode='1d')
#         if config.pm.comm.rank == 0:
#             print('Writing matter power spectrum at %s' % path)
#             # only root rank saves
#             numpy.savetxt(path, 
#                 numpy.array([
#                   r.power['k'], r.power['power'].real, r.power['modes'],
#                   r.power['power'].real / solver.cosmology.scale_independent_growth_factor(1.0 / a - 1) ** 2,
#                 ]).T,
#                 comments='# k p N p/D**2')

#     write_power(dlin, config.makepath('power-linear.txt'), a=1.0)

#     def monitor(action, ai, ac, af, state, event):
#         if config.pm.comm.rank == 0:
#             print('Step %s %06.4f - (%06.4f) -> %06.4f' %( action, ai, ac, af),
#                   'S %(S)06.4f P %(P)06.4f F %(F)06.4f' % (state.a))

#         if action == 'F':
#             a = state.a['F']
#             path = config.makepath('power-%06.4f.txt' % a)
#             write_power(event['delta_k'], path, a)

#         if state.synchronized:
#             a = state.a['S']
#             if a in config['aout']:
#                 path = config.makepath('fpm-%06.4f' % a) % a
#                 if config.pm.comm.rank == 0:
#                     print('Writing a snapshot at %s' % path)
#                 # collective save
#                 state.save(path, attrs=config)

#     solver.nbody(state, stepping=leapfrog(config['stages']), monitor=monitor)

# if __name__ == '__main__':
#     main()