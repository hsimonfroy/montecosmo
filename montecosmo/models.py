from __future__ import annotations # for Union typing with | in python<3.10

import numpyro.distributions as dist
from numpyro import sample, deterministic
from numpyro.handlers import seed, condition, trace
from numpyro.infer.util import log_density
import numpy as np

from jax import numpy as jnp, random as jr, jit, vmap, grad, debug
from jax.tree_util import tree_map
from functools import partial

from montecosmo.bricks import get_cosmo, get_cosmology, get_init_mesh, get_biases, lpt, nbody, lagrangian_weights, rsd 
from montecosmo.metrics import power_spectrum, _initialize_pk

from jax.experimental.ode import odeint
# from jaxpm.pm import lpt, make_ode_fn
from jaxpm.painting import cic_paint



default_config={
            # Mesh and box parameters
            'mesh_shape':64 * np.array([1 ,1 ,1 ]), # int
            'box_shape':640 * np.array([1.,1.,1.]), # in Mpc/h (aim for cell lengths between 1 and 10 Mpc/h)
            # LSS formation
            'a_lpt':0.5, 
            'a_obs':0.5,
            'lpt_order':1,
            # Galaxies
            'galaxy_density':1e-3, # in galaxy / (Mpc/h)^3
            # Debugging
            'trace_reparam':False, 
            'trace_meshes':False, # if int, number of PM mesh snapshots (LPT included)
            # Prior config {name: [label, loc, scale]}
            'prior_config':{'Omega_m':['{\\Omega}_m', 0.3111, 0.2], # XXX: Omega_m<0 implies nan
                            'sigma8':['{\\sigma}_8', 0.8102, 0.2],
                            'b1':['{b}_1', 1., 0.5],
                            'b2':['{b}_2', 0., 2.],
                            'bs2':['{b}_{s^2}', 0., 2.],
                            'bn2':['{b}_{\\nabla^2}', 0., 2.],
                            'init_mesh':['{\\delta}_L', None, None]},
            'fourier':False,                    
            # Likelihood config
            'lik_config':{'obs_std':1.,
                          'obs':'mesh', # 'mesh', 'pk', 'bk' # TODO
                          'multipoles':[0,2,4], # when 'obs'=='pk'
                          },
            }

bench_config = {
        # Blocks to recombine variables
        'blocks_config':{'cosmo':['Omega_m','sigma8'], 
                         'biases':['b1','b2','bs2','bn2'], 
                         'init':['init_mesh']},
        # Chain subsampling
        'n_cell':None,
        'rng_key':jr.key(0),
        'thinning':1,
        # Power spectrum
        'multipoles':[0,2,4],
        }




def prior_model(mesh_shape, prior_config, **config):
    """
    A prior for cosmological model. 

    Return standardized params for computing cosmology, initial conditions, and Lagrangian biases.
    """
    params_ = {}
    
    for name in prior_config:
        name_ = name+'_'
        if name == 'init_mesh':
            # Sample standardized initial conditions
            params_[name_] = sample(name_, dist.Normal(jnp.zeros(mesh_shape), jnp.ones(mesh_shape)))
        else:
            # Sample standardized cosmology and biases
            params_[name_] = sample(name_, dist.Normal(0, 1))

    return params_



def likelihood_model(loc_mesh, mesh_shape, box_shape, galaxy_density, lik_config, noise=0., **config):
    """
    A likelihood for cosmological model.

    Return an observed mesh sampled from a location mesh with observational variance.
    """
    # TODO: prior on obs_std?
    sigma2 = lik_config['obs_std']**2+noise**2
    obs_name = lik_config['obs']
    mesh_shape, box_shape = np.asarray(mesh_shape), np.asarray(box_shape)

    if obs_name == 'mesh':
        # Normal noise
        sigma2 /= (galaxy_density * (box_shape / mesh_shape).prod())
        obs_mesh = sample('obs', dist.Normal(loc_mesh, jnp.sqrt(sigma2)))
        # Poisson noise
        # eps_var = 0.1 # add epsilon variance to prevent zero variance
        # obs_mesh = sample('obs_mesh', dist.Poisson(loc_mesh + eps_var)) 
        # obs_mesh = sample('obs_mesh', dist.Normal(loc_mesh, (loc_mesh  + eps_var)**.5)) # Normal approx
        return obs_mesh

    elif obs_name == 'pk':
        # Anisotropic power spectrum covariance, cf. [Grieb+2016](http://arxiv.org/abs/1509.04293)
        multipoles = np.atleast_1d(lik_config['multipoles'])
        sli_multip = slice(1,1+len(multipoles))
        loc_pk, Nk = get_pk_fn(mesh_shape, box_shape, multipoles=multipoles, kcount=True, galaxy_density=galaxy_density)(loc_mesh)
        # sigma2 *= ((2*multipoles[:,None]+1)/galaxy_density)**2 / Nk
        sigma2 *= 2*(2*multipoles[:,None]+1) * (1 / galaxy_density**2 + 2*loc_pk[1]/galaxy_density) / Nk

        loc_pk = loc_pk.at[1].add(1/galaxy_density) # add shot noise to the mean monopole
        # obs_pk = loc_pk.at[sli_multip].set(sample('obs', dist.MultivariateNormal(loc_pk[sli_multip], Nk)))
        obs_pk = loc_pk.at[sli_multip].set(sample('obs', dist.Normal(loc_pk[sli_multip], sigma2**.5)))
        obs_pk = deterministic('obs_pk', obs_pk)
        return obs_pk





def pmrsd_fn(params_, 
                mesh_shape,                 
                box_shape,
                a_lpt,
                a_obs,
                lpt_order, 
                trace_reparam, 
                trace_meshes,
                prior_config,
                fourier,):
    
    # Get cosmology, initial mesh, and biases from standardized latent params
    cosmo = get_cosmo(prior_config, trace_reparam, **params_)
    cosmology = get_cosmology(**cosmo)
    init_mesh = get_init_mesh(cosmology, mesh_shape, box_shape, fourier, trace_reparam, **params_)
    biases = get_biases(prior_config, trace_reparam, **params_)

    # Create regular grid of particles
    x_part = jnp.indices(mesh_shape).reshape(3,-1).T

    # Lagrangian bias expansion weights at a_obs (but based on initial particules positions)
    lbe_weights = lagrangian_weights(cosmology, a_obs, x_part, box_shape, **biases, **init_mesh)

    # LPT displacement at a_lpt
    cosmology._workspace = {}  # HACK: temporary fix
    dx, p_part, f = lpt(cosmology, init_mesh['init_mesh'], x_part, a=a_lpt, order=lpt_order)
    # NOTE: lpt supposes given mesh follows linear pk at a=1, 
    # and correct by growth factor to get forces at wanted scale factor
    particles = jnp.stack([x_part + dx, p_part])

    # PM displacement from a_lpt to a_obs
    assert(a_lpt <= a_obs), "a_lpt must be less (<=) than a_obs"
    assert(a_lpt < a_obs or 0 <= trace_meshes <= 1), \
        f"required trace_meshes={trace_meshes:d} LPT+PM snapshots, but a_lpt == a_obs == {a_lpt:.2f}"
    
    if trace_meshes == 1:
        particles = deterministic('pm_part', particles[None])[0]

    if a_lpt < a_obs:
        particles = nbody(cosmology, mesh_shape, particles, a_lpt, a_obs, trace_meshes)

        if trace_meshes >= 2:
            particles = deterministic('pm_part', particles)

        particles = particles[-1]
    
    # # Uncomment only to trace bias mesh without rsd
    # biased_mesh = cic_paint(jnp.zeros(mesh_shape), particles[0], lbe_weights)
    # if trace_meshes: 
    #     biased_mesh = deterministic('bias_prersd_mesh', biased_mesh)

    # RSD displacement at a_obs
    dx = rsd(cosmology, a_obs, particles[1])
    particles = particles.at[0].add(dx)

    if trace_meshes: 
        particles = deterministic('rsd_part', particles)
    
    # CIC paint weighted by Lagrangian bias expansion weights
    biased_mesh = cic_paint(jnp.zeros(mesh_shape), particles[0], lbe_weights)

    # debug.print("lbe_weights: {i}", i=(lbe_weights.mean(), lbe_weights.std(), lbe_weights.min(), lbe_weights.max()))
    # debug.print("biased mesh: {i}", i=(biased_mesh.mean(), biased_mesh.std(), biased_mesh.min(), biased_mesh.max()))
    # debug.print("frac of weights < 0: {i}", i=(lbe_weights < 0).sum()/len(lb,e_weights))

    if trace_meshes: 
        biased_mesh = deterministic('bias_mesh', biased_mesh)
    
    return biased_mesh



def pmrsd_model(mesh_shape,
                box_shape,
                a_lpt,
                a_obs, 
                lpt_order,
                galaxy_density, # in galaxy / (Mpc/h)^3
                trace_reparam, 
                trace_meshes,
                prior_config,
                fourier,
                lik_config,
                noise=0.):
    """
    A cosmological forward model, with LPT and PM displacements, Lagrangian bias, and RSD.
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

    galaxy_density : float
        Galaxy density in galaxy / (Mpc/h)^3

    trace_reparam : bool
        If True, trace reparametrized deterministic parameters.

    trace_meshes : bool, int
        If True, trace intermediary meshes.
        If int, number of PM mesh snapshots (LPT included) to trace.

    prior_config : dict
        Prior configuration.

    lik_config : dict 
        Likelihood configuration.

    Noise : float
        Noise level.
    """
    # Sample from prior
    params_ = prior_model(mesh_shape, prior_config)

    # Compute deterministic model function
    biased_mesh = pmrsd_fn(params_,
                            mesh_shape,
                            box_shape,
                            a_lpt,
                            a_obs,
                            lpt_order, 
                            trace_reparam, 
                            trace_meshes,
                            prior_config,
                            fourier,)

    # Sample from likelihood
    obs_mesh = likelihood_model(biased_mesh,
                                mesh_shape,
                                box_shape,
                                galaxy_density, # in galaxy / (Mpc/h)^3
                                lik_config, 
                                noise,) 
    return obs_mesh









def _simulator(model, rng_seed=0, model_kwargs={}):
    model_trace = trace(seed(model, rng_seed=rng_seed)).get_trace(**model_kwargs)
    params = {name: model_trace[name]['value'] for name in model_trace}
    return params


def get_simulator(model):
    """
    Return simulator that samples from model.
    """
    def simulator(rng_seed=0, model_kwargs={}):
        """
        Sample from the model.
        """
        return _simulator( model, rng_seed, model_kwargs)
    return simulator


def _logp_fn(model, params, model_kwargs={}):
    logp = log_density(model=model, 
                model_args=(), 
                model_kwargs=model_kwargs, 
                params=params)[0]
    return logp


def get_logp_fn(model):
    """
    Return model log probabilty functions.
    """
    def logp_fn(params, model_kwargs={}):
        """
        Return the model log probabilty, evaluated on parameters.
        """
        return _logp_fn(model, params, model_kwargs)
    return logp_fn
    

def get_score_fn(model):
    """
    Return model score functions.
    """
    def score_fn(params, model_kwargs={}):
        """
        Return the model score, evaluated on parameters.
        """
        return grad(_logp_fn, argnums=1)(model, params, model_kwargs)
    return score_fn


def get_pk_fn(mesh_shape, box_shape, kmin:float|None=None, dk:float|None=None, 
              los=[0,0,1], multipoles=0, kcount=False, **config):
    """
    Return power spectrum function for given config.
    """
    if kmin is None:
        kmin = 2*jnp.pi * np.max(1 / box_shape)
    if dk is None:
        kmax = 2*np.pi * np.min(mesh_shape / box_shape) / 2
        dk = kmax / 50 # about 50 wavenumber bins
        # dk = 2*np.pi * np.min(1 / box_shape) / 2 * 4

    def pk_fn(mesh):
        """
        Return mesh power spectrum.
        """
        return power_spectrum(mesh, box_shape, kmin, dk, los, multipoles, kcount)
    return pk_fn


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


def print_config(model:partial|dict):
    """
    Print config and infos from a partial model.
    Alternatively, a config can directly be provided.
    """
    if isinstance(model, dict):
        config = model
    else:
        assert isinstance(model, partial), "No partial model or config provided."
        config = model.keywords
    print(f"# CONFIG\n{config}\n")

    cell_shape = list( config['box_shape'] / config['mesh_shape'] )
    print("# INFOS")
    print(f"cell_shape:     {cell_shape} Mpc/h")

    dk = 2*np.pi / np.min(config['box_shape']) 
    knyquist = 2*np.pi * np.min(config['mesh_shape'] / config['box_shape']) / 2
    # (2*pi factor because of Fourier transform definition)
    print(f"dk:             {dk:.5f} h/Mpc")
    print(f"knyquist:       {knyquist:.5f} h/Mpc")

    mean_gxy_count = config['galaxy_density'] * (config['box_shape'] / config['mesh_shape']).prod()
    # NOTE: careful about mesh_shape int overflow, perform float cast before
    print(f"mean_gxy_count: {mean_gxy_count:.3f} gxy/cell\n")


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

