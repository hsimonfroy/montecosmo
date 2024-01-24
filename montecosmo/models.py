import numpyro.distributions as dist
from numpyro import sample, deterministic
from numpyro.handlers import seed, condition, trace
from numpyro.infer.util import log_density
import numpy as np

import jax.numpy as jnp
from jax import random, jit, vmap, grad
from jax.tree_util import tree_map
from functools import partial, wraps

from montecosmo.bricks import get_cosmology, get_linear_field, get_lagrangian_weights, rsd
from montecosmo.metrics import power_spectrum
from jaxpm.pm import lpt
from jaxpm.painting import cic_paint



model_config={
            # Mesh and box parameters
            'mesh_size':64 * np.array([1 ,1 ,1 ]), 
            'box_size':640 * np.array([1.,1.,1.]), # in Mpc/h (aim for cell lengths between 1 and 10 Mpc/h)
            # Scale factors
            'scale_factor_lpt':0.1, 
            'scale_factor_obs':0.5,
            # Galaxies
            'galaxy_density':1e-3, # in galaxy / (Mpc/h)^3
            # Debugging
            'trace_reparam':True, 
            'trace_deterministic':False}


def prior_model(mesh_size, noise=0., **config):
    """
    A prior for cosmological model. 
    Return base values for computing cosmology, initial conditions, and Lagrangian bias latent variables.
    """
    sigma = jnp.sqrt(1+noise**2)

    # Sample cosmology base
    Omega_c_base = sample('Omega_c_base', dist.Normal(0,sigma))
    sigma8_base = sample('sigma8_base', dist.Normal(0, sigma))
    cosmo_base = Omega_c_base, sigma8_base
    # cosmo_base = 0,0

    # Sample initial conditions base
    init_mesh_base = sample('init_mesh_base', dist.Normal(jnp.zeros(mesh_size), sigma*jnp.ones(mesh_size)))

    # Sample Lagrangian biases base
    b1_base  = sample('b1_base',  dist.Normal(0, sigma))
    b2_base  = sample('b2_base',  dist.Normal(0, sigma))
    bs_base  = sample('bs_base',  dist.Normal(0, sigma))
    bnl_base = sample('bnl_base', dist.Normal(0, sigma))
    biases_base = b1_base, b2_base, bs_base, bnl_base
    # biases_base = 0,0,0,0

    return cosmo_base, init_mesh_base, biases_base


def likelihood_model(mean_mesh, noise=0., **config):
    """
    A likelihood for cosmological model.
    Return an observed mesh sampled from a mean mesh with observational variance.
    """
    # TODO: prior on obs_var
    obs_var = 1
    sigma = jnp.sqrt(obs_var+noise**2)

    # Normal noise
    obs_mesh = sample('obs_mesh', dist.Normal(mean_mesh, sigma))
    # Poisson noise
    # eps_var = 0.1 # add epsilon variance to prevent zero variance
    # obs_mesh = sample('obs_mesh', dist.Poisson(gxy_intens_mesh + eps_var)) 
    # obs_mesh = sample('obs_mesh', dist.Normal(gxy_intens_mesh, (gxy_intens_mesh  + eps_var)**.5)) # Normal approx
    return obs_mesh


def pmrsd_model_fn(latent_values, 
                mesh_size,                 
                box_size,
                scale_factor_lpt,
                scale_factor_obs, 
                galaxy_density, # in galaxy / (Mpc/h)^3
                trace_reparam, 
                trace_deterministic):
    # Unpack latent variables
    cosmo_base, init_mesh_base, biases_base = latent_values

    # Get cosmology and initial conditions
    cosmology = get_cosmology(cosmo_base, trace_reparam)
    init_mesh = get_linear_field(cosmology, init_mesh_base, mesh_size, box_size, trace_reparam)

    # Create regular grid of particles
    x_part = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in mesh_size]),axis=-1).reshape([-1,3])

    # Get Lagrangian bias expansion weights
    lbe_weights = get_lagrangian_weights(biases_base, cosmology, scale_factor_obs, init_mesh, x_part, box_size, trace_reparam)

    # LPT displacement
    cosmology._workspace = {}  # HACK: temporary fix
    dx, p_part, f = lpt(cosmology, init_mesh, x_part, a=scale_factor_lpt)
    # NOTE: lpt supposes given mesh follows linear pk at a=1, 
    # and correct by growth factor to get forces at wanted scale factor
    x_part = x_part + dx

    # XXX: here N-body displacement
    # x_part, v_part = ... PM(scale_factor_lpt -> scale_factor_obs)

    if trace_deterministic: 
        x_part = deterministic('pm_part', x_part)

    # RSD displacement
    dx_rsd = rsd(cosmology, scale_factor_obs, p_part)
    x_part = x_part + dx_rsd

    if trace_deterministic: 
        x_part = deterministic('rsd_part', x_part)

    # CIC paint weighted by Lagrangian bias expansion
    biased_mesh = cic_paint(jnp.zeros(mesh_size), x_part, lbe_weights)

    if trace_deterministic: 
        biased_mesh = deterministic('biased_mesh', biased_mesh)

    # Observe
    gxy_intens_mesh = biased_mesh * (galaxy_density * box_size.prod() / mesh_size.prod())
    return gxy_intens_mesh



def pmrsd_model(mesh_size,
                  box_size,
                  scale_factor_lpt,
                  scale_factor_obs, 
                  galaxy_density, # in galaxy / (Mpc/h)^3
                  trace_reparam, 
                  trace_deterministic,
                  noise=0.):
    """
    A cosmological forward model, with LPT and PM displacements, Lagrangian bias, and RSD.
    The relevant variables can be traced.
    """
    # Sample from prior
    latent_values = prior_model(mesh_size, noise)

    # Compute deterministic model function
    gxy_intens_mesh = pmrsd_model_fn(latent_values,
                                        mesh_size,
                                        box_size,
                                        scale_factor_lpt,
                                        scale_factor_obs, 
                                        galaxy_density, # in galaxy / (Mpc/h)^3
                                        trace_reparam, 
                                        trace_deterministic,)

    # Sample from likelihood
    obs_mesh = likelihood_model(gxy_intens_mesh, noise)

    return obs_mesh




def lpt_model_fn(latent_values, 
                mesh_size,                 
                box_size,
                scale_factor_lpt,
                scale_factor_obs, 
                galaxy_density, # in galaxy / (Mpc/h)^3
                trace_reparam, 
                trace_deterministic):
    # Unpack latent variables
    cosmo_base, init_mesh_base, _ = latent_values

    # Get cosmology and initial conditions
    cosmology = get_cosmology(cosmo_base, trace_reparam)
    init_mesh = get_linear_field(cosmology, init_mesh_base, mesh_size, box_size, trace_reparam)

    # Create regular grid of particles
    x_part = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in mesh_size]),axis=-1).reshape([-1,3])

    # LPT displacement
    cosmology._workspace = {}  # HACK: temporary fix
    dx, p_part, f = lpt(cosmology, init_mesh, x_part, a=scale_factor_lpt)
    # NOTE: lpt supposes given mesh follows linear pk at a=1, 
    # and correct by growth factor to get forces at wanted scale factor
    x_part = x_part + dx

    # Cloud In Cell painting
    lpt_mesh = cic_paint(jnp.zeros(mesh_size), x_part)
    
    if trace_deterministic: 
       lpt_mesh = deterministic('lpt_mesh', lpt_mesh)

    # Observe
    gxy_intens_mesh = lpt_mesh * (galaxy_density * box_size.prod() / mesh_size.prod())
    return gxy_intens_mesh


def lpt_model(mesh_size,
                  box_size,
                  scale_factor_lpt,
                  scale_factor_obs, 
                  galaxy_density, # in galaxy / (Mpc/h)^3
                  trace_reparam, 
                  trace_deterministic):
    """
    A simple cosmological model, with LPT displacement.
    The relevant variables can be traced.
    """
    # Sample from prior
    latent_values = prior_model(mesh_size)

    gxy_intens_mesh = lpt_model_fn(latent_values,
                                    mesh_size,
                                    box_size,
                                    scale_factor_lpt,
                                    scale_factor_obs, 
                                    galaxy_density, # in galaxy / (Mpc/h)^3
                                    trace_reparam, 
                                    trace_deterministic,)

    # Sample from likelihood
    obs_mesh = likelihood_model(gxy_intens_mesh, obs_var=1)

    return obs_mesh





def _simulator(model, rng_seed=0, model_kwargs={}):
    model_trace = trace(seed(model, rng_seed=rng_seed)).get_trace(**model_kwargs)
    params = {name: model_trace[name]['value'] for name in model_trace.keys()}
    return params


def get_simulator(model):
    """
    Return a simulator that samples from a model.
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
    Return a model log probabilty functions.
    """
    def logp_fn(params, model_kwargs={}):
        """
        Return the model log probabilty, evaluated on parameters.
        """
        return partial(_logp_fn, model)(params, model_kwargs)
    return logp_fn
    

def get_score_fn(model):
    """
    Return a model score functions.
    """
    def score_fn(params, model_kwargs={}):
        """
        Return the model score, evaluated on parameters.
        """
        return grad(partial(_logp_fn, model), argnums=0)(params, model_kwargs)
    return score_fn


def get_pk_fn(config, kmin=0.001, dk=0.01):
    def pk_fn(mesh):
        pk = power_spectrum(mesh, kmin, dk, config['mesh_size'], config['box_size'])
        return pk
    return pk_fn


def get_init_mesh_fn(config):
    def init_mesh_fn(params_base):
        """
        Compute cosmology and initial conditions from base values.
        """
        cosmo_base = params_base['Omega_c_base'], params_base['sigma8']
        cosmology = get_cosmology(cosmo_base)
        init_mesh = get_linear_field(cosmology, init_base, config['mesh_size'], config['box_size'])
        return cosmology, init_mesh
    return init_mesh_fn




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