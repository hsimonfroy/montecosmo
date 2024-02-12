import numpyro.distributions as dist
from numpyro import sample, deterministic
from numpyro.handlers import seed, condition, trace
from numpyro.infer.util import log_density
import numpy as np

import jax.numpy as jnp
from jax import random, jit, vmap, grad
from jax.tree_util import tree_map
from functools import partial, wraps

from montecosmo.bricks import get_cosmology, get_init_mesh, get_biases, lagrangian_weights, rsd
from montecosmo.metrics import power_spectrum
from jaxpm.pm import lpt
from jaxpm.painting import cic_paint



default_config={
            # Mesh and box parameters
            'mesh_size':64 * np.array([1 ,1 ,1 ]), # int
            'box_size':640 * np.array([1.,1.,1.]), # in Mpc/h (aim for cell lengths between 1 and 10 Mpc/h)
            # Scale factors
            'scale_factor_lpt':0.1, 
            'scale_factor_obs':0.5,
            # Galaxies
            'galaxy_density':1e-3, # in galaxy / (Mpc/h)^3
            # Debugging
            'trace_reparam':True, 
            'trace_deterministic':False,
            # Prior config {name: (label, mean, std)}
            'prior_config':{'Omega_c':('\Omega_c', 0.25, 0.1), # XXX: Omega_c<0 implies nan
                            'sigma8':('\sigma_8', 0.831, 0.14),
                            'b1':('b_1', 1, 0.5),
                            'b2':('b_2', 0, 0.5),
                            'bs':('b_s', 0, 0.5),
                            'bnl':('b_{\text{nl}}', 0, 0.5)},
            # Likelihood config
            'lik_config':{'obs_std':1}                    
            }


def prior_model(mesh_size, noise=0., **config):
    """
    A prior for cosmological model. 
    Return latent values for computing cosmology, initial conditions, and Lagrangian biases variables.
    """
    sigma = jnp.sqrt(1+noise**2)

    # Sample latent cosmology
    Omega_c_ = sample('Omega_c_', dist.Normal(0, sigma))
    sigma8_  = sample('sigma8_' , dist.Normal(0, sigma))
    cosmo_ = Omega_c_, sigma8_

    # Sample latent initial conditions
    init_mesh_ = sample('init_mesh_', dist.Normal(jnp.zeros(mesh_size), sigma*jnp.ones(mesh_size)))

    # Sample latent Lagrangian biases
    b1_  = sample('b1_',  dist.Normal(0, sigma))
    b2_  = sample('b2_',  dist.Normal(0, sigma))
    bs_  = sample('bs_',  dist.Normal(0, sigma))
    bnl_ = sample('bnl_', dist.Normal(0, sigma))
    biases_ = b1_, b2_, bs_, bnl_

    return cosmo_, init_mesh_, biases_


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


def pmrsd_model_fn(latent_values, 
                mesh_size,                 
                box_size,
                scale_factor_lpt,
                scale_factor_obs, 
                galaxy_density, # in galaxy / (Mpc/h)^3
                trace_reparam, 
                trace_deterministic,
                prior_config,):
    # Unpack latent variables
    cosmo_, init_mesh_, biases_ = latent_values

    # Get cosmology, initial mesh, and biases from latent values
    cosmology = get_cosmology(cosmo_, prior_config, trace_reparam)
    init_mesh = get_init_mesh(cosmology, init_mesh_, mesh_size, box_size, trace_reparam)
    biases = get_biases(biases_, prior_config, trace_reparam)

    # Create regular grid of particles
    x_part = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in mesh_size]),axis=-1).reshape([-1,3])

    # Lagrangian bias expansion weights
    lbe_weights = lagrangian_weights(cosmology, scale_factor_obs, biases, init_mesh, x_part, box_size)

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

    # Scale mesh by galaxy density
    gxy_mesh = biased_mesh * (galaxy_density * box_size.prod() / mesh_size.prod())

    if trace_deterministic: 
        gxy_mesh = deterministic('gxy_mesh', gxy_mesh)
    return gxy_mesh


def pmrsd_model(mesh_size,
                  box_size,
                  scale_factor_lpt,
                  scale_factor_obs, 
                  galaxy_density, # in galaxy / (Mpc/h)^3
                  trace_reparam, 
                  trace_deterministic,
                  prior_config,
                  lik_config,
                  noise=0.):
    """
    A cosmological forward model, with LPT and PM displacements, Lagrangian bias, and RSD.
    The relevant variables can be traced.
    """
    # Sample from prior
    latent_values = prior_model(mesh_size)

    # Compute deterministic model function
    gxy_mesh = pmrsd_model_fn(latent_values,
                                mesh_size,
                                box_size,
                                scale_factor_lpt,
                                scale_factor_obs, 
                                galaxy_density, # in galaxy / (Mpc/h)^3
                                trace_reparam, 
                                trace_deterministic,
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


def get_pk_fn(mesh_size, box_size, kmin=0.001, dk=0.01, los=jnp.array([0.,0.,1.]), multipoles=0, **config):
    def pk_fn(mesh):
        pk = power_spectrum(mesh, kmin, dk, mesh_size, box_size, los, multipoles)
        return pk
    return pk_fn


def get_init_mesh_fn(mesh_size, box_size, prior_config, **config):
    def init_mesh_fn(**params_):
        """
        Compute cosmology and initial conditions from latent values.
        """
        cosmo_ = params_['Omega_c_'], params_['sigma8_']
        init_ = params_['init_mesh_']
        cosmo = get_cosmology(cosmo_, prior_config)
        init_mesh = get_init_mesh(cosmo, init_, mesh_size, box_size)
        return cosmo, init_mesh
    return init_mesh_fn


def get_noise_fn(t0, t1, noises, steps=False):
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


def print_config(config):
    """
    Print config and infos.
    """
    print(f"# CONFIG:\n{config}\n")

    cell_lengths = list( config['box_size'] / config['mesh_size'] )
    print(f"# INFOS:\n{cell_lengths=} Mpc/h")

    delta_k = 2*jnp.pi * jnp.max(1 / config['box_size']) 
    k_nyquist = 2*jnp.pi * jnp.min(config['mesh_size'] / config['box_size']) / 2
    # (2*pi factor because of Fourier transform definition)
    print(f"{delta_k=:.5f} h/Mpc, {k_nyquist=:.5f} h/Mpc")

    mean_gxy_density = config['galaxy_density'] * config['box_size'].prod() / config['mesh_size'].prod()
    print(f"{mean_gxy_density=:.3f} gxy/cell")


def condition_on_config_mean(model, prior_config, **config):
    params = {name+'_':0. for name in prior_config}
    return condition(model, params)



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