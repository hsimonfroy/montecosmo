
from jax import numpy as jnp, random as jr, jit, vmap, grad, debug, lax, tree
from functools import partial

import blackjax
from blackjax.progress_bar import gen_scan_fn # XXX: blackjax >= 1.2.3
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState


#########################
# HMC/NUTS within Gibbs #
#########################
def mwg_warmup(rng_key, state, logdf, config, n_samples=0, progress_bar=True):
    rng_keys = jr.split(rng_key, num=len(state))
    rng_keys = dict(zip(state.keys(), rng_keys))

    # avoid modifying argument state as JAX functions should be pure
    state = state.copy()
    infos = {}
    infos['n_evals'] = 0
    params = {}
    positions = {}

    for k in state.keys():
        # logdf of component k conditioned on all other components in state
        union = {}
        for _k in state.keys():
            union |= state[_k].position

        def logdf_k(value):
            return logdf(union | value) # update component k

        # give state[k] the right log_density 
        # NOTE: unnecessary if we only pass position to warmup
        # state[k] = init_fn[k](
        #     position=state[k].position,
        #     logdensity_fn=logdf_k
        # )

        wind_adapt = blackjax.window_adaptation(blackjax.nuts, logdf_k, **config[k], progress_bar=progress_bar) 
        # NOTE: window adapt progress bar yield "NotImplementedError: IO effect not supported in vmap-of-cond"
        # for blackjax==1.2.4 due to progress bar update
        rng_keys[k], warmup_key = jr.split(rng_keys[k], 2)
        (state[k], params[k]), info = wind_adapt.run(warmup_key, state[k].position, num_steps=n_samples)

        # register only relevant infos
        n_evals = info.info.num_integration_steps
        infos['infos_'+k] = {"acceptance_rate": info.info.acceptance_rate, 
                                "num_integration_steps": n_evals}
        infos['n_evals'] += n_evals
        # positions[k] = info.state.position
        positions |= info.state.position
    
    return (state, params), (positions, infos)



def mwg_kernel_general(rng_key, state, logdf, step_fn, init_fn, config):
    """
    General MWG kernel.

    Updates each component of ``state`` conditioned on all the others using a component-specific MCMC algorithm

    Parameters
    ----------
    rng_key
        The PRNG key.
    state
        Dictionary where each item is the state of an MCMC algorithm, i.e., an object of type ``AlgorithmState``.
    logdf
        A log-density function, where the arguments are the keys of ``state``.
    step_fn
        Dictionary with the same keys as ``state``,
        each element of which is an MCMC stepping functions on the corresponding component.
    init_fn
        Dictionary with the same keys as ``state``,
        each elemtn of chi is an MCMC initializer corresponding to the stepping functions in `step_fn`.
    config
        Dictionary with the same keys as ``state``, each of which is a dictionary of config parameters to
        the MCMC algorithm for the corresponding component.

    Returns
    -------
    Dictionary containing the updated ``state``.
    """
    rng_keys = jr.split(rng_key, num=len(state))
    rng_keys = dict(zip(state.keys(), rng_keys))

    # avoid modifying argument state as JAX functions should be pure
    state = state.copy()
    infos = {}
    infos['n_evals'] = 0

    for k in state.keys():
        # logdf of component k conditioned on all other components in state
        union = {}
        for _k in state.keys():
            union |= state[_k].position

        def logdf_k(value):
            return logdf(union | value) # update component k
        
        # give state[k] the right log_density
        state[k] = init_fn[k](
            position=state[k].position,
            logdensity_fn=logdf_k
        )

        # update state[k]
        state[k], info = step_fn[k](
            rng_key=rng_keys[k],
            state=state[k],
            logdensity_fn=logdf_k,
            **config[k]
        )

        # register only relevant infos
        n_evals = info.num_integration_steps
        infos['infos_'+k] = {"acceptance_rate": info.acceptance_rate, 
                                "num_integration_steps": n_evals}
        infos['n_evals'] += n_evals
    
    return state, infos
    


def sampling_loop_general(rng_key, initial_state, logdf, step_fn, init_fn, config, n_samples, progress_bar=True):
    
    # @blackjax.progress_bar.progress_bar_scan(n_samples)
    def one_step(state, xs):
        _, rng_key = xs
        state, infos = mwg_kernel_general(
            rng_key=rng_key,
            state=state,
            logdf=logdf,
            step_fn=step_fn,
            init_fn=init_fn,
            config=config
        )

        # Unionize the blocks
        union = {}
        for k in state.keys():
            union |= state[k].position
        return state, (union, infos)

    keys = jr.split(rng_key, n_samples)
    xs = jnp.arange(n_samples), keys

    # last_state, (positions, infos) = lax.scan(one_step, initial_state, xs) # XXX: blackjax < 1.2.3
  
    scan_fn = gen_scan_fn(n_samples, progress_bar)
    last_state, (positions, infos) = scan_fn(one_step, initial_state, xs)

    return last_state, (positions, infos)



def nutswg_init(logdf, kernel="NUTS"):
    init_ss = 1e-3
    target_acc_rate = 0.65

    if kernel == "HMC":
        ker_api = blackjax.hmc
        config = {
            "mesh_": {
                'target_acceptance_rate': target_acc_rate,
                'initial_step_size': init_ss,
                "num_integration_steps": 256,
                # "inverse_mass_matrix": jnp.ones(64**3),
                # "step_size": 3*1e-3
            },
            "rest_": {
                'target_acceptance_rate': target_acc_rate,
                'initial_step_size': init_ss,
                "num_integration_steps": 64,
                # "inverse_mass_matrix": jnp.ones(6),
                # "step_size": 3*1e-3
            }
        }
    elif kernel == "NUTS":
        ker_api = blackjax.nuts
        config = {
            "mesh_": {
                'target_acceptance_rate': target_acc_rate,
                'initial_step_size': init_ss,
                # 'max_num_doublings':10,
                # "inverse_mass_matrix": jnp.ones(64**3),
                # "step_size": 3*1e-3
            },
            "rest_": {
                'target_acceptance_rate': target_acc_rate,
                'initial_step_size': init_ss,
                # 'max_num_doublings':10,
                # "inverse_mass_matrix": jnp.ones(6),
                # "step_size": 3*1e-3
            }
        }
    mwg_init_x = ker_api.init
    mwg_init_y = ker_api.init
    mwg_step_fn_x = ker_api.build_kernel()
    mwg_step_fn_y = ker_api.build_kernel()

    step_fn = {
        "mesh_": mwg_step_fn_x,
        "rest_": mwg_step_fn_y
    }

    init_fn={
        "mesh_": mwg_init_x,
        "rest_": mwg_init_y
    }

    def init_state_fn(init_pos):
        return get_init_state(init_pos, logdf, init_fn)

    return step_fn, init_fn, config, init_state_fn


def get_init_state(init_pos, logdf, init_fn):
    init_pos_block1 = {name:init_pos[name] for name in ['init_mesh_']}
    init_pos_block2 = {name:init_pos[name] for name in ['Omega_m_','sigma8_','b1_','b2_','bs2_','bn2_']}
    init_state = {
        "mesh_": init_fn['mesh_'](
            position = init_pos_block1,
            logdensity_fn = lambda x: logdf(x |init_pos_block2)
        ),
        "rest_": init_fn['rest_'](
            position = init_pos_block2,
            logdensity_fn = lambda y: logdf(y | init_pos_block1)
        )
    }
    return init_state


def nutswg_run(rng_key, init_state, config, logdf, step_fn, init_fn, n_samples, progress_bar=True):
    last_state, (samples, infos) = sampling_loop_general(
                                rng_key=rng_key,
                                initial_state=init_state,
                                logdf=logdf,
                                step_fn=step_fn,
                                init_fn=init_fn,
                                config=config,
                                n_samples=n_samples,
                                progress_bar=progress_bar)
    return samples, infos, last_state

def get_nutswg_run(logdf, step_fn, init_fn, n_samples, progress_bar=True):
    return partial(nutswg_run, 
                   logdf=logdf, 
                   step_fn=step_fn, 
                   init_fn=init_fn, 
                   n_samples=n_samples,
                   progress_bar=progress_bar)


def nutswg_warm(rng_key, init_state, logdf, config, n_samples, progress_bar=True):
    (last_state, config), (samples, infos) = mwg_warmup(rng_key, init_state, logdf, config, n_samples, progress_bar=progress_bar)
    return samples, infos, last_state, config

def get_nutswg_warm(logdf, config, n_samples, progress_bar=True):
    return partial(nutswg_warm, 
                   logdf=logdf, 
                   config=config, 
                   n_samples=n_samples,
                   progress_bar=progress_bar)












#########
# MCLMC #
#########
def mclmc_warmup(rng, init_pos, logdf, n_samples, config=None, 
              desired_energy_variance=5e-4, diagonal_preconditioning=False):
    init_key, tune_key = jr.split(rng, 2)

    # Create an initial state for the sampler
    state = blackjax.mcmc.mclmc.init(
        position=init_pos, logdensity_fn=logdf, rng_key=init_key
    )

    if config is None:
        # Build the kernel
        kernel = lambda sqrt_diag_cov : blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=logdf,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            sqrt_diag_cov=sqrt_diag_cov,
        )

        # Find values for L and step_size
        print("finding L, ss, mm")
        state, config = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=n_samples,
            state=state,
            rng_key=tune_key,
            diagonal_preconditioning=diagonal_preconditioning,
            desired_energy_var=desired_energy_variance,
            # num_effective_samples=256, # NOTE: higher value implies slower averaging rate
            # frac_tune3=0.5
            )

    elif isinstance(config, dict):
        L = config['L']
        step_size = config['step_size']
        sqrt_diag_cov = config.get('sqrt_diag_cov', 1.0)
        config = MCLMCAdaptationState(L=L, step_size=step_size, sqrt_diag_cov=sqrt_diag_cov)

    else:
        assert isinstance(config, MCLMCAdaptationState), \
        "config must be either None, a dict, or a MCLMCAdaptationState"

    return state, config



def mclmc_run(rng, state, config:dict|MCLMCAdaptationState, logdf, n_samples,  
              transform=None, thinning=1, progress_bar=True):
    
    if transform is None:
        # transform = lambda state, info: state.position
        # transform = lambda state, info: (state.position, info)
        transform = lambda state, info: (state.position, tree.map(lambda x: jnp.mean(x**2)**.5, info)) # TODO: map_with_path to get mean logd and Kchange
        # transform = lambda state: state.position # XXX: blackjax < 1.2.3

    if isinstance(config, dict):
        L = config['L']
        step_size = config['step_size']
        sqrt_diag_cov = config.get('sqrt_diag_cov', 1.0)

    elif isinstance(config, MCLMCAdaptationState):
        L = config.L
        step_size = config.step_size
        sqrt_diag_cov = config.sqrt_diag_cov

    # Use the quick wrapper to build a new kernel with the tuned parameters
    sampler = blackjax.mclmc(logdf, L=L, step_size=step_size, sqrt_diag_cov=sqrt_diag_cov)

    # Run the sampler
    if thinning==1:
        state, samples = blackjax.util.run_inference_algorithm(
        # state, samples, info = blackjax.util.run_inference_algorithm( # XXX: blackjax < 1.2.3 but wrong state
            rng_key=rng,
            initial_state=state,
            inference_algorithm=sampler,
            num_steps=n_samples,
            transform=transform,
            progress_bar=progress_bar,
        )
    else:
        state, samples = run_with_thinning(
            rng_key=rng,
            inference_algorithm=sampler,
            num_steps=n_samples,
            initial_state=state,
            transform=transform,
            progress_bar=progress_bar,
            thinning=thinning
        )
    samples, info = samples
    
    # Register only relevant infos
    n_eval_per_steps = 2 # NOTE: 1 for velocity_verlet, 2 for mclachlan
    infos = {"n_evals": n_eval_per_steps * thinning * jnp.ones(n_samples)}

    return state, samples|infos, info



def get_mclmc_run(logdf, n_samples, transform=None, thinning=1, progress_bar=True):
    return partial(mclmc_run, 
                   logdf=logdf,
                   n_samples=n_samples,
                   transform=transform,
                   thinning=thinning,    
                   progress_bar=progress_bar)


def get_mclmc_warmup(logdf, n_samples, config=None,
              desired_energy_variance=5e-4, diagonal_preconditioning=False):
    return partial(mclmc_warmup,
                   logdf=logdf,
                   n_samples=n_samples,
                   config=config,
                   desired_energy_variance=desired_energy_variance,
                   diagonal_preconditioning=diagonal_preconditioning)


##################
# Adjusted MCLMC #
##################
# from blackjax.mcmc.adjusted_mclmc import rescale # NOTE: blackjax > 1.2.4
from blackjax.util import run_inference_algorithm

# def run_adjusted_mclmc(
#     logdensity_fn,
#     num_steps,
#     initial_position,
#     key,
#     transform=lambda state, _ : state.position,
#     diagonal_preconditioning=False,
#     random_trajectory_length=True,
#     L_proposal_factor=jnp.inf
# ):

#     init_key, tune_key, run_key = jr.split(key, 3)

#     initial_state = blackjax.mcmc.adjusted_mclmc.init(
#         position=initial_position,
#         logdensity_fn=logdensity_fn,
#         random_generator_arg=init_key,
#     )

#     if random_trajectory_length:
#         integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
#             jr.uniform(k) * rescale(avg_num_integration_steps))
#     else:
#         integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(avg_num_integration_steps)

#     kernel = lambda rng_key, state, avg_num_integration_steps, step_size, sqrt_diag_cov: blackjax.mcmc.adjusted_mclmc.build_kernel(
#         integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
#         sqrt_diag_cov=sqrt_diag_cov,
#     )(
#         rng_key=rng_key,
#         state=state,
#         step_size=step_size,
#         logdensity_fn=logdensity_fn,
#         L_proposal_factor=L_proposal_factor,
#     )

#     target_acc_rate = 0.9 # our recommendation

#     (
#         blackjax_state_after_tuning,
#         blackjax_mclmc_sampler_params,
#     ) = blackjax.adjusted_mclmc_find_L_and_step_size(
#         mclmc_kernel=kernel,
#         num_steps=num_steps,
#         state=initial_state,
#         rng_key=tune_key,
#         target=target_acc_rate,
#         frac_tune1=0.1,
#         frac_tune2=0.1,
#         frac_tune3=0.0, # our recommendation
#         diagonal_preconditioning=diagonal_preconditioning,
#     )

#     step_size = blackjax_mclmc_sampler_params.step_size
#     L = blackjax_mclmc_sampler_params.L

#     alg = blackjax.adjusted_mclmc(
#         logdensity_fn=logdensity_fn,
#         step_size=step_size,
#         integration_steps_fn=lambda key: jnp.ceil(
#             jr.uniform(key) * rescale(L / step_size)
#         ),
#         sqrt_diag_cov=blackjax_mclmc_sampler_params.sqrt_diag_cov,
#         L_proposal_factor=L_proposal_factor,
#     )

#     _, out = run_inference_algorithm(
#         rng_key=run_key,
#         initial_state=blackjax_state_after_tuning,
#         inference_algorithm=alg,
#         num_steps=num_steps,
#         transform=transform,
#         progress_bar=False,
#     )

#     return out











from functools import partial
from typing import Callable, Union

from blackjax.base import SamplingAlgorithm, VIAlgorithm
from blackjax.progress_bar import progress_bar_scan
from blackjax.types import ArrayLikeTree, PRNGKey

def run_with_thinning(
    rng_key: PRNGKey,
    inference_algorithm: SamplingAlgorithm|VIAlgorithm,
    num_steps: int,
    initial_state: ArrayLikeTree = None,
    initial_position: ArrayLikeTree = None,
    progress_bar: bool = False,
    thinning: int = 1,
    transform: Callable = lambda state, info: (state, info),
) -> tuple:
    """
    Wrapper to run an inference algorithm.

    Note that this utility function does not work for Stochastic Gradient MCMC samplers
    like sghmc, as SG-MCMC samplers require additional control flow for batches of data
    to be passed in during each sample.

    Parameters
    ----------
    rng_key
        The random state used by JAX's random numbers generator.
    initial_state
        The initial state of the inference algorithm.
    initial_position
        The initial position of the inference algorithm. This is used when the initial
        state is not provided.
    inference_algorithm
        One of blackjax's sampling algorithms or variational inference algorithms.
    num_steps
        Number of MCMC steps.
    progress_bar
        Whether to display a progress bar.
    transform
        A transformation of the trace of states to be returned. This is useful for
        computing determinstic variables, or returning a subset of the states.
        By default, the states are returned as is.
    """

    if initial_state is None and initial_position is None:
        raise ValueError(
            "Either `initial_state` or `initial_position` must be provided."
        )
    if initial_state is not None and initial_position is not None:
        raise ValueError(
            "Only one of `initial_state` or `initial_position` must be provided."
        )

    if initial_state is None:
        rng_key, init_key = jr.split(rng_key, 2)
        initial_state = inference_algorithm.init(initial_position, init_key)


    def one_sub_step(state, rng_key):
        state, info = inference_algorithm.step(rng_key, state)
        return state, info
    
    def one_step(state, xs):
        _, rng_key = xs
        keys = jr.split(rng_key, thinning)
        state, info = lax.scan(one_sub_step, state, keys)
        return state, transform(state, info)
        # return state, transform(state) # XXX: blackjax < 1.2.3

    keys = jr.split(rng_key, num_steps)
    xs = jnp.arange(num_steps), keys
    
    scan_fn = gen_scan_fn(num_steps, progress_bar)
    final_state, history = scan_fn(one_step, initial_state, xs)

    # if progress_bar:
    #     one_step = progress_bar_scan(num_steps)(one_step) # XXX: blackjax < 1.2.3

    final_state, history = lax.scan(one_step, initial_state, xs)

    return final_state, history







#############
# Optimizer #
#############
# NOTE: optimizers are just 0 Kelvin samplers
from jax.example_libraries.optimizers import adam
from tqdm import tqdm
from jax import value_and_grad

def optimize(potential, start, lr0=0.1, n_epochs=100):
    pots = []

    lr_fn = lambda i: lr0 / (1 + i)**.5
    opt_init, opt_update, get_params = adam(lr_fn)
    opt_state = opt_init(start)

    @jit
    def step(step, opt_state):
        value, grads = value_and_grad(potential)(get_params(opt_state))
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    for i_epoch in tqdm(range(n_epochs)):
        value, opt_state = step(i_epoch, opt_state)
        pots.append(value.astype(float))
    params = get_params(opt_state)
    return params, pots