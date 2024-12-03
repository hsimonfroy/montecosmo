from __future__ import annotations # for Union typing with | in python<3.10

from functools import partial
from dataclasses import dataclass
from IPython.display import display
from pprint import pformat

import numpyro.distributions as dist
from numpyro import sample, deterministic, handlers, render_model
from numpyro.infer.util import log_density
import numpy as np

from jax import numpy as jnp, random as jr, jit, vmap, grad, debug
from jax.tree_util import tree_map
from jax.experimental.ode import odeint

# from jaxpm.pm import lpt, make_ode_fn
from jaxpm.painting import cic_paint

from montecosmo.bricks import base2samp, base2samp_mesh, get_cosmology, lpt, nbody, lagrangian_weights, rsd 
from montecosmo.metrics import power_spectrum




default_config={
            # Mesh and box parameters
            'mesh_shape':3 * (64,), # int
            'box_shape':3 * (640.,), # in Mpc/h (aim for cell lengths between 1 and 10 Mpc/h)
            # LSS formation
            'a_lpt':0.1, 
            'a_obs':0.5,
            'lpt_order':1,
            # Galaxies
            'gxy_density':1e-3, # in galaxy / (Mpc/h)^3
            # Debugging
            'trace_reparam':False, 
            'trace_meshes':False, # if int, number of PM mesh snapshots (LPT included)
            # Prior config {name: [group, label, loc, scale, low, high]}
            'latent': {'Omega_m': {'group':'cosmo', 
                                    'label':'{\\Omega}_m', 
                                    'loc':0.3111, 
                                    'scale':0.2,
                                    'low': 0.05, # XXX: Omega_m < Omega_c implies nan
                                    'high': 1.},
                        'sigma8': {'group':'cosmo',
                                    'label':'{\\sigma}_8',
                                    'loc':0.8102,
                                    'scale':0.2,
                                    'low': 0.,},
                        'b1': {'group':'bias',
                                    'label':'{b}_1',
                                    'loc':1.,
                                    'scale':0.5,},
                        'b2': {'group':'bias',
                                    'label':'{b}_2',
                                    'loc':0.,
                                    'scale':2.,},
                        'bs2': {'group':'bias',
                                    'label':'{b}_{s^2}',
                                    'loc':0.,
                                    'scale':2.,},
                        'bn2': {'group':'bias',
                                    'label':'{b}_{\\nabla^2}',
                                    'loc':0.,
                                    'scale':2.,},
                        'init_mesh': {'group':'init',
                                      'label':'{\\delta}_L',},},
            'fourier':True,                    
            'obs':'mesh', # 'mesh', 'pk', 'plk', 'bk' # TODO
            'snapshots':5, # number of PM snapshots
            }

bench_config = {
        # Chain subsampling
        'n_cell':None,
        'rng_key':jr.key(0),
        'thinning':1,
        # Power spectrum
        'multipoles':[0,2,4],
        }





class Model():
    def _model(self):
        raise NotImplementedError

    def model(self):
        return self._model()
    
    def reset(self):
        self.model = self._model

    def __call__(self):
        return self.model()

    def potential(self, params):
        return - log_density(self.model, (), {}, params)[0]
    
    def simulate(self, rng_seed=0):
        model_trace = handlers.trace(handlers.seed(self.model, rng_seed=rng_seed)).get_trace()
        params = {name: model_trace[name]['value'] for name in model_trace}
        return params
    
    def seed(self, rng_seed):
        self.model = handlers.seed(self.model, rng_seed=rng_seed)

    def condition(self, data=None):
        self.model = handlers.condition(self.model, data=data)

    def block(self, hide_fn=None, hide=None, expose_types=None, expose=None):
        self.model = handlers.block(self.model, 
                                    hide_fn=hide_fn, hide=hide, expose_types=expose_types, expose=expose)

    def render(self, render_dist=False, render_params=False):
        display(render_model(self.model, render_distributions=render_dist, render_params=render_params))




@dataclass
class FieldLevelModel(Model):
    """
    Field-level cosmological model, 
    with LPT and PM displacements, Lagrangian bias, and RSD.
    The relevant variables can be traced.

    Parameters
    ----------
    mesh_shape : array_like of int
        Shape of the mesh.
    box_shape : array_like
        Shape of the box in Mpc/h. Typically aim for cell lengths between 1 and 10 Mpc/h.
    a_lpt : float
        Scale factor to which compute Lagrangian Perturbation Theory (LPT) displacement.
        If equal to a_obs, no Particule Mesh (PM) step is computed.
    a_obs : float
        Scale factor of observations.
        If equal to a_lpt, no Particule Mesh (PM) step is computed.
    gxy_density : float
        Galaxy density in galaxy / (Mpc/h)^3
    """

    mesh_shape:np.ndarray
    box_shape:np.ndarray
    a_lpt:float
    a_obs:float
    lpt_order:int
    gxy_density:float
    trace_reparam:bool
    trace_meshes:bool|int
    latent:dict
    fourier:bool
    obs:dict
    snapshots:int|list

    def __post_init__(self):
        assert(self.a_lpt <= self.a_obs), "a_lpt must be less (<=) than a_obs"
        self.groups = self._get_groups(self.latent)
        self.prior_loc = self._get_prior_loc(self.latent)

        self.mesh_shape = np.asarray(self.mesh_shape)
        self.box_shape = np.asarray(self.box_shape)
        self.cell_shape = self.box_shape / self.mesh_shape

        # careful about int overflow, perform float cast before
        self.dk = 2*np.pi / np.min(self.box_shape) 
        self.k_nyquist = np.pi * np.min(self.mesh_shape / self.box_shape)
        # (2*pi factors because of Fourier transform definition)
        self.gxy_count = self.gxy_density * (self.box_shape / self.mesh_shape).prod()


    def __str__(self):
        out = ""
        out += f"# CONFIG\n"
        out += pformat(self.__dict__, width=1)
        out += "\n\n# INFOS\n"
        out += f"cell_shape:     {list(self.cell_shape)} Mpc/h\n"
        out += f"dk:             {self.dk:.5f} h/Mpc\n"
        out += f"k_nyquist:      {self.k_nyquist:.5f} h/Mpc\n"
        out += f"mean_gxy_count: {self.gxy_count:.3f} gxy/cell\n"
        return out


    def _model(self):
        x = self.prior()
        x = self.reparam(x)
        x = self.evolve(x)
        return self.likelihood(x)
    


    def prior(self):
        """
        A prior for cosmological model. 

        Return standardized params for computing cosmology, initial conditions, and Lagrangian biases.
        """
        # Sample standardized cosmology and biases
        for group in ['cosmo', 'bias']:
            params_ = {}
            for name in self.groups[group]:            
                name_ = name+'_'
                params_[name_] = sample(name_, dist.Normal(0, 1))
            yield params_

        # Sample standardized initial conditions
        name_ = self.groups['init'][0]+'_'         
        mesh = sample(name_, dist.Normal(jnp.zeros(self.mesh_shape), jnp.ones(self.mesh_shape)))
        yield {name_:mesh}

    
    def reparam(self, params, inv=False, fourier=True, temp=1.):
        """
        Transform sample params into base params.
        """
        cosmo, bias, init = params

        # Cosmology and Biases
        cosmo = base2samp(cosmo, self.latent, inv=inv, temp=temp)
        bias = base2samp(bias, self.latent, inv=inv, temp=temp)

        # Initial conditions
        cosmology = get_cosmology(**cosmo)
        init = base2samp_mesh(init, cosmology, self.mesh_shape, self.box_shape, inv=inv, fourier=fourier, temp=temp)
        return cosmo, bias, init


    def evolve(self, params):
        cosmo, bias, init = params
        cosmology = get_cosmology(**cosmo)

        # Create regular grid of particles
        q = jnp.indices(self.mesh_shape).reshape(3,-1).T

        # Lagrangian bias expansion weights at a_obs (but based on initial particules positions)
        lbe_weights = lagrangian_weights(cosmology, self.a_obs, q, self.box_shape, **bias, **init)

        # LPT displacement at a_lpt
        # NOTE: lpt supposes given mesh follows linear pk at a=1, and then correct by growth factor for target a_lpt
        cosmology._workspace = {}  # HACK: temporary fix
        dq, p, f = lpt(cosmology, init['init_mesh'], q, a=self.a_lpt, mesh_shape=self.mesh_shape, order=self.lpt_order)
        particles = jnp.stack([q + dq, p])

        # PM displacement from a_lpt to a_obs
        particles = nbody(cosmology, self.mesh_shape, particles, self.a_lpt, self.a_obs, self.snapshots)
        debug.print("particles: {i}", i=(particles.shape, particles[0].mean(), particles[0].std(), particles[0].min(), particles[0].max()))
        particles = deterministic('pm_part', particles)[-1]

        # RSD displacement at a_obs
        dq = rsd(cosmology, self.a_obs, particles[1])
        particles = particles.at[0].add(dq)
        particles = deterministic('rsd_part', particles)

        # CIC paint weighted by Lagrangian bias expansion weights
        biased_mesh = cic_paint(jnp.zeros(self.mesh_shape), particles[0], lbe_weights)
        biased_mesh = deterministic('bias_mesh', biased_mesh)        
        # debug.print("lbe_weights: {i}", i=(lbe_weights.mean(), lbe_weights.std(), lbe_weights.min(), lbe_weights.max()))
        # debug.print("biased mesh: {i}", i=(biased_mesh.mean(), biased_mesh.std(), biased_mesh.min(), biased_mesh.max()))
        # debug.print("frac of weights < 0: {i}", i=(lbe_weights < 0).sum()/len(lb,e_weights))
        return biased_mesh


    def likelihood(self, mesh, temp=1.):
        """
        A likelihood for cosmological model.

        Return an observed mesh sampled from a location mesh with observational variance.
        """
        mesh_shape, box_shape = np.asarray(self.mesh_shape), np.asarray(self.box_shape)

        if self.obs == 'mesh':
            gxy_count = (self.gxy_density * (box_shape / mesh_shape).prod())
            obs_mesh = sample('obs', dist.Normal(mesh, (temp / gxy_count)**.5)) # Gaussian noise
            return obs_mesh

        # elif self.obs == 'pk':
        #     # Anisotropic power spectrum covariance, cf. [Grieb+2016](http://arxiv.org/abs/1509.04293)
        #     multipoles = np.atleast_1d(self.lik_config['multipoles'])
        #     sli_multip = slice(1,1+len(multipoles))
        #     loc_pk, Nk = get_pk_fn(mesh_shape, box_shape, multipoles=multipoles, kcount=True, gxy_density=self.gxy_density)(mesh)
        #     sigma2 *= 2*(2*multipoles[:,None]+1) * (1 / self.gxy_density**2 + 2*loc_pk[1]/self.gxy_density) / Nk

        #     loc_pk = loc_pk.at[1].add(1/self.gxy_density) # add shot noise to the mean monopole
        #     # obs_pk = loc_pk.at[sli_multip].set(sample('obs', dist.MultivariateNormal(loc_pk[sli_multip], Nk)))
        #     obs_pk = loc_pk.at[sli_multip].set(sample('obs', dist.Normal(loc_pk[sli_multip], sigma2**.5)))
        #     obs_pk = deterministic('obs_pk', obs_pk)
        #     return obs_pk



    def spectrum(self, mesh, mesh2=None, kedges:int|float|list=None, multipoles=0, los=[0.,0.,1.]):
        return power_spectrum(mesh, mesh2=mesh2, box_shape=self.box_shape, 
                              kedges=kedges, multipoles=multipoles, los=los)


    def _get_prior_loc(self, latent):
        """
        Return location values of the prior config.
        """
        dic = {}
        for name in latent:
            loc = latent[name].get('loc')
            if loc is not None:
                dic[name] = loc
        return dic
    
    def _get_groups(self, latent):
        """
        Return groups from latent config.
        """
        groups = {}
        for name in latent:
            group = latent[name]['group']
            if group not in groups:
                groups[group] = []
            groups[group].append(name)
        return groups



