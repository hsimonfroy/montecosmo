from __future__ import annotations # for Union typing with | in python<3.10

import numpyro.distributions as dist
from numpyro import sample, deterministic
from numpyro.handlers import seed, condition, trace
from numpyro.infer.util import log_density
import numpy as np

import jax.numpy as jnp
from jax import random, jit, vmap, grad, debug
from jax.tree_util import tree_map
from functools import partial

from montecosmo.bricks import get_cosmo, get_cosmology, get_init_mesh, get_biases, lagrangian_weights, rsd, get_ode_fn
from montecosmo.metrics import power_spectrum

from jax.experimental.ode import odeint
from jaxpm.pm import lpt, make_ode_fn
from jaxpm.painting import cic_paint
from jax_cosmo import Cosmology

from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController, Euler, Heun, Dopri5


default_config={
            # Mesh and box parameters
            'mesh_size':64 * np.array([1 ,1 ,1 ]), # int
            'box_size':640 * np.array([1.,1.,1.]), # in Mpc/h (aim for cell lengths between 1 and 10 Mpc/h)
            # Scale factors
            'a_lpt':0.5, 
            'a_obs':0.5,
            # Galaxies
            'galaxy_density':1e-3, # in galaxy / (Mpc/h)^3
            # Debugging
            'trace_reparam':False, 
            'trace_meshes':False, # if int, number of PM mesh snapshots (LPT included)
            # Prior config {name: [label, mean, std]}
            'prior_config':{'Omega_m':['{\\Omega}_m', 0.3111, 0.2], # XXX: Omega_m<0 implies nan
                            'sigma8':['{\\sigma}_8', 0.8102, 0.2],
                            'b1':['{b}_1', 1., 0.5],
                            'b2':['{b}_2', 0., 2.],
                            'bs2':['{b}_{s^2}', 0., 2.],
                            'bn2':['{b}_{\\nabla^2}', 0., 2.]},
            'fourier':False,                    
            # Likelihood config
            'lik_config':{'obs_std':1.},
            }






def prior_model(mesh_size, prior_config, **config):
    """
    A prior for cosmological model. 

    Return latent params for computing cosmology, initial conditions, and Lagrangian biases.
    """
    # Sample latent cosmology and Lagrangian biases
    params_ = {}
    
    # Standard param
    for name in prior_config:
        name_ = name+'_'
        params_[name_] = sample(name_, dist.Normal(0, 1))

    # Sample latent initial conditions
    name_ = 'init_mesh_'
    params_[name_] = sample(name_, dist.Normal(jnp.zeros(mesh_size), jnp.ones(mesh_size)))

    return params_



def likelihood_model(mean_mesh, lik_config, noise=0., **config):
    """
    A likelihood for cosmological model.

    Return an observed mesh sampled from a mean mesh with observational variance.
    """
    # TODO: prior on obs_std?
    sigma = jnp.sqrt(lik_config['obs_std']**2+noise**2)

    # Normal noise
    obs_mesh = sample('obs_mesh', dist.Normal(mean_mesh, sigma))
    # Poisson noise
    # eps_var = 0.1 # add epsilon variance to prevent zero variance
    # obs_mesh = sample('obs_mesh', dist.Poisson(mean_mesh + eps_var)) 
    # obs_mesh = sample('obs_mesh', dist.Normal(mean_mesh, (mean_mesh  + eps_var)**.5)) # Normal approx
    return obs_mesh



def pmrsd_model_fn(latent_params, 
                mesh_size,                 
                box_size,
                a_lpt,
                a_obs, 
                galaxy_density,
                trace_reparam, 
                trace_meshes,
                prior_config,
                fourier,):
    """
    Parameters
    ----------
    latent_params : dict
        Latent parameters typically drawn from prior.

    mesh_size : array_like of int
        Size of the mesh.

    box_size : array_like
        Size of the box in Mpc/h. Typically aim for cell lengths between 1 and 10 Mpc/h.

    a_lpt : float
        Scale factor to which compute Lagrangian Perturbation Theory (LPT) displacement.
        If equal to a_obs, no Particule Mesh (PM) step is computed.

    a_obs : float
        Scale factor of observations.
        If equal to a_lpt, no Particule Mesh (PM) step is computed.

    galaxy_density : float
        Galaxy density in galaxy / (Mpc/h)^3.

    trace_reparam : bool
        If True, trace reparametrized deterministic parameters.

    trace_meshes : bool, int
        If True, trace intermediary meshes.
        If int, number of PM mesh snapshots (LPT included) to trace.

    prior_config : dict
        Prior configuration.
    """
    # Get cosmology, initial mesh, and biases from latent params
    cosmo = get_cosmo(prior_config, trace_reparam, **latent_params)
    cosmology = get_cosmology(**cosmo)
    init_mesh = get_init_mesh(cosmology, mesh_size, box_size, fourier, trace_reparam, **latent_params)
    biases = get_biases(prior_config, trace_reparam, **latent_params)

    # Create regular grid of particles
    x_part = jnp.indices(mesh_size).reshape(3,-1).T

    # Lagrangian bias expansion weights at a_obs (but based on initial particules positions)
    lbe_weights = lagrangian_weights(cosmology, a_obs, x_part, box_size, **biases, **init_mesh)

    # LPT displacement at a_lpt
    cosmology._workspace = {}  # HACK: temporary fix
    dx, p_part, f = lpt(cosmology, init_mesh['init_mesh'], x_part, a=a_lpt)
    # NOTE: lpt supposes given mesh follows linear pk at a=1, 
    # and correct by growth factor to get forces at wanted scale factor
    particles = jnp.concatenate((x_part + dx, p_part), axis=-1)

    # PM displacement from a_lpt to a_obs
    # assert(a_lpt <= a_obs), "a_lpt must be less (<=) than a_obs"
    # assert(a_lpt < a_obs or 0 <= trace_meshes <= 1), \
    #     "required trace_meshes={trace_meshes:d} LPT+PM snapshots, but a_lpt == a_obs == {a_lpt:.2f}"
    if trace_meshes == 1:
        particles = deterministic('pm_part', particles[None])[0]

    if a_lpt < a_obs:
        terms = ODETerm(get_ode_fn(cosmology, mesh_size))
        solver = Dopri5()
        controller = PIDController(rtol=1e-5, atol=1e-5, pcoeff=0.4, icoeff=1, dcoeff=0)
        if trace_meshes < 2: 
            saveat = SaveAt(t1=True)
        else: 
            saveat = SaveAt(ts=jnp.linspace(a_lpt, a_obs, trace_meshes))      
        sol = diffeqsolve(terms, solver, a_lpt, a_obs, dt0=None, y0=particles,
                             stepsize_controller=controller, max_steps=20, saveat=saveat)
        particles = sol.ys
        # debug.print("num_steps: {n}", n=sol.stats['num_steps'])

        if trace_meshes >= 2:
            particles = deterministic('pm_part', particles)

        particles = particles[-1]
    
    # Uncomment only to trace bias mesh without rsd
    biased_mesh = cic_paint(jnp.zeros(mesh_size), particles[:,:3], lbe_weights)
    if trace_meshes: 
        biased_mesh = deterministic('bias_prersd_mesh', biased_mesh)

    # RSD displacement at a_obs
    dx = rsd(cosmology, a_obs, particles[:,3:])
    particles = particles.at[:,:3].add(dx)

    # if trace_meshes: 
    #     particles = deterministic('rsd_part', particles)
    
    # CIC paint weighted by Lagrangian bias expansion weights
    biased_mesh = cic_paint(jnp.zeros(mesh_size), particles[:,:3], lbe_weights)

    # debug.print("lbe_weights: {i}", i=(lbe_weights.mean(), lbe_weights.std(), lbe_weights.min(), lbe_weights.max()))
    # debug.print("biased mesh: {i}", i=(biased_mesh.mean(), biased_mesh.std(), biased_mesh.min(), biased_mesh.max()))
    # debug.print("frac of weights < 0: {i}", i=(lbe_weights < 0).sum()/len(lbe_weights))

    if trace_meshes: 
        biased_mesh = deterministic('bias_mesh', biased_mesh)
    
    # Scale mesh by galaxy density
    gxy_mesh = biased_mesh * (galaxy_density * box_size.prod() / mesh_size.prod())
    return gxy_mesh


def pmrsd_model(mesh_size,
                  box_size,
                  a_lpt,
                  a_obs, 
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
    mesh_size : array_like of int
        Size of the mesh.

    box_size : array_like
        Size of the box in Mpc/h. Typically aim for cell lengths between 1 and 10 Mpc/h.

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
    latent_params = prior_model(mesh_size, prior_config)

    # Compute deterministic model function
    gxy_mesh = pmrsd_model_fn(latent_params,
                                mesh_size,
                                box_size,
                                a_lpt,
                                a_obs, 
                                galaxy_density, # in galaxy / (Mpc/h)^3
                                trace_reparam, 
                                trace_meshes,
                                prior_config,
                                fourier,)

    # Sample from likelihood
    obs_mesh = likelihood_model(gxy_mesh, lik_config, noise)
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


def get_pk_fn(mesh_size, box_size, kmin=0.001, dk=0.01, los=jnp.array([0.,0.,1.]), multipoles=0, **config):
    """
    Return power spectrum function for given config.
    """
    def pk_fn(mesh):
        """
        Return mesh power spectrum.
        """
        return power_spectrum(mesh, kmin, dk, mesh_size, box_size, los, multipoles)
    return pk_fn


def get_param_fn(mesh_size, box_size, prior_config, fourier=False,
                 trace_reparam=False, scale_std=1, **config):
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
        keys = params_.keys()

        if all([name+sufx in keys for name in ['Omega_m', 'sigma8']]):
            cosmo = get_cosmo(prior_config, trace_reparam, inverse, scale_std, **params_)

            if 'init_mesh'+sufx in keys:
                if not inverse:
                    cosmology = get_cosmology(**cosmo)
                else:
                    cosmology = get_cosmology(**params_)

                init_mesh = get_init_mesh(cosmology, mesh_size, box_size, fourier, 
                                          trace_reparam, inverse, scale_std, **params_)
            else: init_mesh = {}
        else: cosmo, init_mesh = {}, {}

        if all([name+sufx in keys for name in ['b1', 'b2', 'bs2', 'bn2']]):
            biases = get_biases(prior_config, trace_reparam, inverse, scale_std, **params_)
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

    cell_size = list( config['box_size'] / config['mesh_size'] )
    print("# INFOS")
    print(f"cell_size:        {cell_size} Mpc/h")

    delta_k = 2*jnp.pi * jnp.max(1 / config['box_size']) 
    k_nyquist = 2*jnp.pi * jnp.min(config['mesh_size'] / config['box_size']) / 2
    # (2*pi factor because of Fourier transform definition)
    print(f"delta_k:          {delta_k:.5f} h/Mpc")
    print(f"k_nyquist:        {k_nyquist:.5f} h/Mpc")

    mean_gxy_density = config['galaxy_density'] * config['box_size'].prod() / config['mesh_size'].prod()
    print(f"mean_gxy_density: {mean_gxy_density:.3f} gxy/cell\n")


def get_prior_mean(model:partial|dict):
    """
    Return mean values of the prior config from a partial model.
    Alternatively, a config can directly be provided.
    """
    if isinstance(model, dict):
        config = model
    else:
        assert isinstance(model, partial), "No partial model or config provided."
        config = model.keywords
    prior_config = config['prior_config']
    return {name: prior_config[name][1] for name in prior_config}


def get_noise_fn(t0, t1, noises, steps=False):
    """
    Given a noises list, starting and ending times, 
    return a function that interpolate these noises between these times,
    by steps or linearly.
    """
    n_noises = len(noises)-1
    if steps:
        def noise_fn(t):
            i_t = n_noises*(t-t0)/(t1-t0)
            i_t1 = jnp.floor(i_t).astype(int)
            return noises[i_t1]
    else:
        def noise_fn(t):
            i_t = n_noises*(t-t0)/(t1-t0)
            i_t1 = jnp.floor(i_t).astype(int)
            s1 = noises[i_t1]
            s2 = noises[i_t1+1]
            return (s2 - s1)*(i_t - i_t1) + s1
    return noise_fn

