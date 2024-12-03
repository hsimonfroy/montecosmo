from __future__ import annotations # for Union typing with | in python<3.10

import numpyro.distributions as dist
from numpyro import sample, deterministic, handlers, render_model
from numpyro.infer.util import log_density
import numpy as np

from jax import numpy as jnp, random as jr, jit, vmap, grad, debug
from jax.tree_util import tree_map
from functools import partial
from dataclasses import dataclass
from IPython.display import display

from montecosmo.bricks import base2samp, base2samp_mesh, get_cosmology, lpt, nbody, lagrangian_weights, rsd 
from montecosmo.metrics import power_spectrum

from jax.experimental.ode import odeint
# from jaxpm.pm import lpt, make_ode_fn
from jaxpm.painting import cic_paint



default_config={
            # Mesh and box parameters
            'mesh_shape':3 * (64,), # int
            'box_shape':3 * (640.,), # in Mpc/h (aim for cell lengths between 1 and 10 Mpc/h)
            # LSS formation
            'a_lpt':0.5, 
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
                        'b1': {'group':'biases',
                                    'label':'{b}_1',
                                    'loc':1.,
                                    'scale':0.5,},
                        'b2': {'group':'biases',
                                    'label':'{b}_2',
                                    'loc':0.,
                                    'scale':2.,},
                        'bs2': {'group':'biases',
                                    'label':'{b}_{s^2}',
                                    'loc':0.,
                                    'scale':2.,},
                        'bn2': {'group':'biases',
                                    'label':'{b}_{\\nabla^2}',
                                    'loc':0.,
                                    'scale':2.,},
                        'init_mesh': {'group':'init',
                                      'label':'{\\delta}_L',},},
            'fourier':False,                    
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


def get_groups(latent:dict):
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

    def block(self, hide=None, expose=None):
        self.model = handlers.block(self.model, hide=hide, expose=expose)

    def render(self):
        display(render_model(self.model, render_distributions=True, render_params=True))




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
        self.groups = get_groups(self.latent)

        self.mesh_shape = np.asarray(self.mesh_shape)
        self.box_shape = np.asarray(self.box_shape)
        self.cell_shape = self.box_shape / self.mesh_shape

        # careful about int overflow, perform float cast before
        self.dk = 2*np.pi / np.min(self.box_shape) 
        self.k_nyquist = np.pi * np.min(self.mesh_shape / self.box_shape)
        # (2*pi factors because of Fourier transform definition)
        self.gxy_count = self.gxy_density * (self.box_shape / self.mesh_shape).prod()


    def __str__(self):
        print(f"# CONFIG\n{self.__dict__}\n")
        print("# INFOS")
        print(f"cell_shape:     {list(self.cell_shape)} Mpc/h")
        print(f"dk:             {self.dk:.5f} h/Mpc")
        print(f"k_nyquist:      {self.k_nyquist:.5f} h/Mpc")
        print(f"mean_gxy_count: {self.gxy_count:.3f} gxy/cell\n")


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
        for group in ['cosmo', 'biases', 'init']:
            params_ = {}
            for name in self.groups[group]:            
                name_ = name+'_'
                if name == 'init_mesh':
                    # Sample standardized initial conditions
                    params_[name_] = sample(name_, dist.Normal(jnp.zeros(self.mesh_shape), jnp.ones(self.mesh_shape)))
                else:
                    # Sample standardized cosmology and biases
                    params_[name_] = sample(name_, dist.Normal(0, 1))
            yield params_

    
    def reparam(self, params, inv=False, fourier=True, scaling=1.):
        """
        Transform sample params into base params.
        """
        cosmo, biases, init = params

        # Cosmology and Biases
        cosmo = base2samp(cosmo, self.latent, inv=inv, scaling=scaling)
        biases = base2samp(biases, self.latent, inv=inv, scaling=scaling)

        # Initial conditions
        init = base2samp_mesh(init, self.mesh_shape, inv=inv, fourier=fourier, scaling=scaling)
        return cosmo, biases, init


    def evolve(self, params):
        cosmo, biases, init = params
        cosmology = get_cosmology(**cosmo)

        # Create regular grid of particles
        q = jnp.indices(self.mesh_shape).reshape(3,-1).T

        # Lagrangian bias expansion weights at a_obs (but based on initial particules positions)
        lbe_weights = lagrangian_weights(cosmology, self.a_obs, q, self.box_shape, **biases, **init)

        # LPT displacement at a_lpt
        cosmology._workspace = {}  # HACK: temporary fix
        dq, p, f = lpt(cosmology, init['init_mesh'], q, a=self.a_lpt, order=self.lpt_order)
        # NOTE: lpt supposes given mesh follows linear pk at a=1, and then correct by growth factor for target a_lpt
        particles = jnp.stack([q + dq, p])

        # PM displacement from a_lpt to a_obs
        particles = nbody(cosmology, self.mesh_shape, particles, self.a_lpt, self.a_obs, self.snapshots)
        particles = deterministic('pm_part', particles)[-1]

        # # Uncomment only to trace bias mesh without rsd
        # biased_mesh = cic_paint(jnp.zeros(self.mesh_shape), particles[0], lbe_weights)
        # biased_mesh = deterministic('bias_prersd_mesh', biased_mesh)

        # RSD displacement at a_obs
        dq = rsd(cosmology, self.a_obs, particles[1])
        particles = particles.at[0].add(dq)
        particles = deterministic('rsd_part', particles)

        # CIC paint weighted by Lagrangian bias expansion weights
        biased_mesh = cic_paint(jnp.zeros(self.mesh_shape), particles[0], lbe_weights)

        # debug.print("lbe_weights: {i}", i=(lbe_weights.mean(), lbe_weights.std(), lbe_weights.min(), lbe_weights.max()))
        # debug.print("biased mesh: {i}", i=(biased_mesh.mean(), biased_mesh.std(), biased_mesh.min(), biased_mesh.max()))
        # debug.print("frac of weights < 0: {i}", i=(lbe_weights < 0).sum()/len(lb,e_weights))

        biased_mesh = deterministic('bias_mesh', biased_mesh)        
        return biased_mesh


    def likelihood(self, mesh, noise=0.):
        """
        A likelihood for cosmological model.

        Return an observed mesh sampled from a location mesh with observational variance.
        """
        base_var = 1 + noise**2
        mesh_shape, box_shape = np.asarray(self.mesh_shape), np.asarray(self.box_shape)

        if self.obs == 'mesh':
            gxy_count = (self.gxy_density * (box_shape / mesh_shape).prod())
            obs_mesh = sample('obs', dist.Normal(mesh, (base_var / gxy_count)**.5)) # Gaussian noise
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



    def spectrum(self, mesh, mesh2, kedges:int|float|list=None, multipoles=0, los=[0.,0.,1.]):
        return power_spectrum(mesh, mesh2, box_shape=self.box_shape, 
                              kedges=kedges, multipoles=multipoles, los=los)






def get_param_fn(mesh_shape, box_shape, prior_config, fourier=False,
                 trace_reparam=False, scaling=1., **config):
    """
    Return a partial replay model function for given config.
    """
    def param_fn(inverse=False, **params_):
        """
        Partially replay model, i.e. transform latent params into params of interest.
        """
        if not inverse:
            sufx = '_'
        else:
            sufx = ''

        if all([name+sufx in params_ for name in ['Omega_m', 'sigma8']]):
            cosmo = get_cosmo(prior_config, trace_reparam, inverse, scaling, **params_)

            if 'init_mesh'+sufx in params_:
                if not inverse:
                    cosmology = get_cosmology(**cosmo)
                else:
                    cosmology = get_cosmology(**params_)

                init_mesh = get_init_mesh(cosmology, mesh_shape, box_shape, fourier, 
                                          trace_reparam, inverse, scaling, **params_)
            else: init_mesh = {}
        else: cosmo, init_mesh = {}, {}

        if all([name+sufx in params_ for name in ['b1', 'b2', 'bs2', 'bn2']]):
            biases = get_biases(prior_config, trace_reparam, inverse, scaling, **params_)
        else: biases = {}

        # params = dict(**cosmo, **init_mesh, **biases)
        params = cosmo | init_mesh | biases  # XXX: python>=3.9
        return params
    return param_fn



def get_prior_loc(model:partial|dict):
    """
    Return location values of the prior config from a partial model.
    Alternatively, a config can directly be provided.
    """
    # Get prior config
    if isinstance(model, dict):
        config = model
    else:
        assert isinstance(model, partial), "No partial model or config provided."
        config = model.keywords
    prior_config = config['prior_config']

    # Get locs
    loc_dic = {}
    for name in prior_config:
        mean = prior_config[name][1]
        if mean is not None:
            loc_dic |= {name: mean}
    return loc_dic

