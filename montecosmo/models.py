from __future__ import annotations # for Union typing with | in python<3.10

import numpyro.distributions as dist
from numpyro import sample, deterministic
from numpyro.handlers import seed, condition, trace
from numpyro.infer.util import log_density
import numpy as np

import jax.numpy as jnp
from jax import random, jit, vmap, grad
from jax.tree_util import tree_map
from functools import partial, wraps

from montecosmo.bricks import get_cosmo, get_init_mesh, get_biases, lagrangian_weights, rsd
from montecosmo.metrics import power_spectrum

from jax.experimental.ode import odeint
from jaxpm.pm import lpt, make_ode_fn
from jaxpm.painting import cic_paint
from jax_cosmo import Planck15




default_config={
            # Mesh and box parameters
            'mesh_size':64 * np.array([1 ,1 ,1 ]), # int
            'box_size':640 * np.array([1.,1.,1.]), # in Mpc/h (aim for cell lengths between 1 and 10 Mpc/h)
            # Scale factors
            'a_lpt':0.1, 
            'a_obs':0.5,
            # Galaxies
            'galaxy_density':1e-3, # in galaxy / (Mpc/h)^3
            # Debugging
            'trace_reparam':False, 
            'trace_meshes':False, # if int, number of PM mesh snapshots (LPT included)
            # Prior config {name: (label, mean, std)}
            'prior_config':{'Omega_c':['{\Omega}_c', 0.25, 0.1], # XXX: Omega_c<0 implies nan
                            'sigma8':['{\sigma}_8', 0.831, 0.14],
                            'b1':['{b}_1', 1, 0.5],
                            'b2':['{b}_2', 0, 0.5],
                            'bs2':['{b}_{s^2}', 0, 0.5],
                            'bn2':['{b}_{\\nabla^2}', 0, 0.5]},
            # Likelihood config
            'lik_config':{'obs_std':1}                    
            }


def prior_model(mesh_size, noise=0., **config):
    """
    A prior for cosmological model. 

    Return latent params for computing cosmology, initial conditions, and Lagrangian biases.
    """
    sigma = jnp.sqrt(1+noise**2)

    # Sample latent cosmology
    Omega_c_ = sample('Omega_c_', dist.Normal(0, sigma))
    sigma8_  = sample('sigma8_' , dist.Normal(0, sigma))

    # Sample latent initial conditions
    init_mesh_ = sample('init_mesh_', dist.Normal(jnp.zeros(mesh_size), sigma*jnp.ones(mesh_size)))

    # Sample latent Lagrangian biases
    b1_  = sample('b1_',  dist.Normal(0, sigma))
    b2_  = sample('b2_',  dist.Normal(0, sigma))
    bs2_  = sample('bs2_',  dist.Normal(0, sigma))
    bn2_ = sample('bn2_', dist.Normal(0, sigma))

    params_ = dict(Omega_c_=Omega_c_, sigma8_=sigma8_, 
                   init_mesh_=init_mesh_, 
                   b1_=b1_, b2_=b2_, bs2_=bs2_, bn2_=bn2_)

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
                prior_config,):
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
    cosmology = Planck15(**cosmo)
    init_mesh = get_init_mesh(cosmology, mesh_size, box_size, trace_reparam, **latent_params)
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
    x_part = x_part + dx

    # PM displacement from a_lpt to a_obs
    # assert(a_lpt <= a_obs), "a_lpt must be less (<=) than a_obs"
    # assert(a_lpt < a_obs or 0 <= trace_meshes <= 1), \
    #     "required trace_meshes={trace_meshes:d} LPT+PM snapshots, but a_lpt == a_obs == {a_lpt:.2f}"
    if trace_meshes == 1:
        x_part = deterministic('pm_part', x_part[None])[0]

    if a_lpt < a_obs:
        if trace_meshes < 2: trace_meshes = 2 # XXX: jnp.maximum would not jit
        snapshots = jnp.linspace(a_lpt, a_obs, trace_meshes)
        res = odeint(make_ode_fn(mesh_size), [x_part, p_part], snapshots, cosmology, rtol=1e-5, atol=1e-5)
        x_parts, p_parts = res

        if trace_meshes >= 2:
            x_parts = deterministic('pm_part', x_parts)

        x_part, p_part = x_parts[-1], p_parts[-1]

        
    biased_mesh = cic_paint(jnp.zeros(mesh_size), x_part, lbe_weights)
    if trace_meshes: 
        biased_mesh = deterministic('bias_prersd_mesh', biased_mesh)

    # RSD displacement at a_obs
    dx = rsd(cosmology, a_obs, p_part)
    # dx = rsd(cosmology, a_obs, jnp.zeros_like(x_part))
    x_part = x_part + dx

    if trace_meshes: 
        x_part = deterministic('rsd_part', x_part)

    # CIC paint weighted by Lagrangian bias expansion weights
    biased_mesh = cic_paint(jnp.zeros(mesh_size), x_part, lbe_weights)

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
    latent_params = prior_model(mesh_size)

    # Compute deterministic model function
    gxy_mesh = pmrsd_model_fn(latent_params,
                                mesh_size,
                                box_size,
                                a_lpt,
                                a_obs, 
                                galaxy_density, # in galaxy / (Mpc/h)^3
                                trace_reparam, 
                                trace_meshes,
                                prior_config,)

    # Sample from likelihood
    obs_mesh = likelihood_model(gxy_mesh, lik_config, noise)
    return obs_mesh









def _simulator(model, rng_seed=0, model_kwargs={}):
    model_trace = trace(seed(model, rng_seed=rng_seed)).get_trace(**model_kwargs)
    params = {name: model_trace[name]['value'] for name in model_trace.keys()}
    return params


def get_simulator(model):
    """
    Return simulator that samples from model.
    """
    def simulator(rng_seed=0, model_kwargs={}):
        """
        Sample from the model.
        """
        return partial(_simulator, model)(rng_seed, model_kwargs)
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
        return partial(_logp_fn, model)(params, model_kwargs)
    return logp_fn
    

def get_score_fn(model):
    """
    Return model score functions.
    """
    def score_fn(params, model_kwargs={}):
        """
        Return the model score, evaluated on parameters.
        """
        return grad(partial(_logp_fn, model), argnums=0)(params, model_kwargs)
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


def get_param_fn(mesh_size, box_size, prior_config, trace_reparam=False, **config):
    """
    Return a partial replay model function for given config.
    """
    def param_fn(Omega_c_=None, sigma8_=None, 
                   init_mesh_=None, 
                   b1_=None, b2_=None, bs2_=None, bn2_=None, 
                   **params_):
        """
        Partially replay model, i.e. transform latent params into params of interest.
        """
        if not any([v is None for v in [Omega_c_, sigma8_]]):
            cosmo = get_cosmo(prior_config, trace_reparam, Omega_c_=Omega_c_, sigma8_=sigma8_)

            if init_mesh_ is not None:
                cosmology = Planck15(**cosmo)
                init_mesh = get_init_mesh(cosmology, mesh_size, box_size, trace_reparam, init_mesh_=init_mesh_)
            else: init_mesh = {}
        else: cosmo, init_mesh = {}, {}

        if not any([v is None for v in [b1_, b2_, bs2_, bn2_]]):
            biases = get_biases(prior_config, trace_reparam, b1_=b1_, b2_=b2_, bs2_=bs2_, bn2_=bn2_)
        else: biases = {}

        params = dict(**cosmo, **init_mesh, **biases)
        # params = cosmo | init_mesh | biases  # XXX: python>=3.9
        return params
    return param_fn


def print_config(model:partial|dict):
    """
    Print config and infos from a partial model.
    Alternatively, config can be directly provided.
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


def condition_on_config_mean(model, prior_config=None, **config):
    """
    Condition model on the mean values of the prior config.
    If no `prior_config` is provided, extract it from partial model.
    """
    if prior_config is None:
        assert isinstance(model, partial), "No 'prior_config' found."
        prior_config = model.keywords['prior_config']
    params = {name+'_':0. for name in prior_config}
    return condition(model, params)


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



# def get_logp_fn(model, cond_params={}):
#     """
#     Return a model log probabilty function, conditioned on some parameters.
#     """
#     vlogp_model = vmap(partial(logp_model, model, cond_params), in_axes=(0,None))
#     @get_jit
#     def logp_fn(params, model_kwargs={}):
#         """
#         Return the model log probabilty, evaluated on some parameters.
#         """
#         return vlogp_model(params, model_kwargs)

#     return logp_fn

# def get_score_fn(model, cond_params={}):
#     """
#     Return a model score function, conditioned on some parameters.
#     """
#     score_model = grad(partial(logp_model, model, cond_params), argnums=0)
#     vscore_model = vmap(score_model, in_axes=(0,None))
#     @get_jit()
#     def score_fn(params, model_kwargs={}):
#         """
#         Return the model score, evaluated on some parameters.
#         """
#         return vscore_model(params, model_kwargs)
    
#     return score_fn 

# def get_simulator(model, cond_params={}):
#     """
#     Return a simulator that samples from a model conditioned on some parameters.
#     """
#     def sample_model(model, cond_params, rng_seed=0, model_kwargs={}):
#         if len(model_kwargs)==0:
#             model_kwargs = {}
#         cond_model = condition(model, cond_params) # NOTE: Only condition on random sites
#         cond_trace = trace(seed(cond_model, rng_seed=rng_seed)).get_trace(**model_kwargs)
#         params = {name: cond_trace[name]['value'] for name in cond_trace.keys()}
#         return params

#     vsample_model = vmap(partial(sample_model, model, cond_params), in_axes=(None,0))
#     vvsample_model = vmap(vsample_model, in_axes=(0,None))

#     @get_jit(static_argnames=('batch_size'))
#     def simulator(batch_size=1, rng_key=random.PRNGKey(0), model_kwargs={}):
#         """
#         Sample batches from model. If they are both strict greater than one, 
#         batch size would be left-most dimension, and model arguments size the second left-most.
#         """
#         squeeze_axis = []
#         if batch_size==1:
#             squeeze_axis.append(0)
#         if len(model_kwargs)==0:
#             model_kwargs = jnp.array([[]]) # for vmap, because jnp.array([{}]) is not valid
#             squeeze_axis.append(1)
#         keys = random.split(rng_key, batch_size)
#         params = vvsample_model(keys, model_kwargs)
#         return {name: params[name].squeeze(axis=squeeze_axis) for name in params.keys()}

#     return simulator