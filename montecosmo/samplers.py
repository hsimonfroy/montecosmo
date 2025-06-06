from jax import numpy as jnp, random as jr, jit, vmap, grad, debug, lax, tree
from jax.flatten_util import ravel_pytree
from functools import partial

import blackjax
from blackjax.progress_bar import gen_scan_fn # XXX: blackjax >= 1.2.3
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from blackjax.util import run_inference_algorithm
from blackjax.mcmc.integrators import isokinetic_mclachlan, isokinetic_velocity_verlet, isokinetic_yoshida, isokinetic_omelyan
from numpyro.infer import MCMC



#########################
# HMC/NUTS within Gibbs #
#########################
def mwg_warmup(seed, state, logdf, config, n_samples=0, progress_bar=True):
    seeds = jr.split(seed, num=len(state))
    seeds = dict(zip(state.keys(), seeds))

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
        seeds[k], warmup_seed = jr.split(seeds[k], 2)
        (state[k], params[k]), info = wind_adapt.run(warmup_seed, state[k].position, num_steps=n_samples)

        # register only relevant infos
        n_evals = info.info.num_integration_steps
        infos['infos_'+k] = {"acceptance_rate": info.info.acceptance_rate, 
                                "num_integration_steps": n_evals}
        infos['n_evals'] += n_evals
        # positions[k] = info.state.position
        positions |= info.state.position
    
    return (state, params), (positions, infos)



def mwg_kernel_general(seed, state, logdf, step_fn, init_fn, config):
    """
    General MWG kernel.

    Updates each component of ``state`` conditioned on all the others using a component-specific MCMC algorithm

    Parameters
    ----------
    seed
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
    seeds = jr.split(seed, num=len(state))
    seeds = dict(zip(state.keys(), seeds))

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
            rng_key=seeds[k],
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
    


def sampling_loop_general(seed, initial_state, logdf, step_fn, init_fn, config, n_samples, progress_bar=True):
    
    def one_step(state, xs):
        _, seed = xs
        state, infos = mwg_kernel_general(
            seed=seed,
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

    seeds = jr.split(seed, n_samples)
    xs = jnp.arange(n_samples), seeds

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


def nutswg_run(seed, init_state, config, logdf, step_fn, init_fn, n_samples, progress_bar=True):
    last_state, (samples, infos) = sampling_loop_general(
                                seed=seed,
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


def nutswg_warm(seed, init_state, logdf, config, n_samples, progress_bar=True):
    (last_state, config), (samples, infos) = mwg_warmup(seed, init_state, logdf, config, n_samples, progress_bar=progress_bar)
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
def mclmc_warmup(seed, init_pos, logdf, n_steps=0, config=None, 
              desired_energy_var=5e-4, diagonal_preconditioning=False):
    init_seed, tune_seed = jr.split(seed, 2)

    # Create an initial state for the sampler
    state = blackjax.mcmc.mclmc.init(
        position=init_pos, logdensity_fn=logdf, rng_key=init_seed
    )

    if config is None:
        n_dim = len(ravel_pytree(state.position)[0])
        config = MCLMCAdaptationState(
            n_dim**.5, n_dim**.5 / 4, inverse_mass_matrix=jnp.ones(n_dim))
        
    elif isinstance(config, dict):
        L = config['L']
        step_size = config['step_size']
        inverse_mass_matrix = config.get('inverse_mass_matrix', 1.0)
        config = MCLMCAdaptationState(L=L, step_size=step_size, inverse_mass_matrix=inverse_mass_matrix)

    else:
        assert isinstance(config, MCLMCAdaptationState), \
        "config must be either None, a dict, or a MCLMCAdaptationState"

    if n_steps > 0:
        # Build the kernel
        kernel = lambda inverse_mass_matrix : blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=logdf,
            integrator=isokinetic_mclachlan,
            inverse_mass_matrix=inverse_mass_matrix,
        )

        # Find values for L and step_size
        frac_tune1 = 0.5
        frac_tune2 = 0.5
        frac_tune3 = 0. 
        # NOTE: can't afford to save every samples to estimate effective sample size, so adapt L afterwards
        num_steps = round(n_steps / (frac_tune1 + frac_tune2 * (1 + diagonal_preconditioning / 3) + frac_tune3))
        # NOTE: num_steps to pass to mclmc_find_L_and_step_size to actually perform n_steps adaptation steps

        state, config, n_steps_tot = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=num_steps,
            state=state,
            rng_key=tune_seed,
            desired_energy_var=desired_energy_var,
            diagonal_preconditioning=diagonal_preconditioning,
            frac_tune1=frac_tune1,
            frac_tune2=frac_tune2,
            frac_tune3=frac_tune3,
            num_effective_samples=256, # NOTE: higher value implies slower averaging rate
            # TODO: add config as first guess (next blackjax update)
            )
        debug.print("Perform {n_steps_tot} adaptation steps", n_steps_tot=n_steps_tot)  

    return state, config



def mclmc_run(seed, state, config:dict|MCLMCAdaptationState, logdf, n_samples,  
              transform=None, thinning=1, progress_bar=True):
    integrator = isokinetic_mclachlan

    if integrator == isokinetic_velocity_verlet:
        n_eval_per_steps = 1
    elif integrator == isokinetic_mclachlan:
        n_eval_per_steps = 2 
    elif integrator == isokinetic_yoshida:
        n_eval_per_steps = 3
    elif integrator == isokinetic_omelyan:
        n_eval_per_steps = 5
    
    if transform is None:
        n_dim = len(ravel_pytree(state.position)[0])
        transform = lambda state, info: (state.position,
                                    {'logdensity': state.logdensity, 
                                     'mse_per_dim': jnp.mean(info.energy_change**2) / n_dim})


    if isinstance(config, dict):
        L = config['L']
        step_size = config['step_size']
        inverse_mass_matrix = config.get('inverse_mass_matrix', 1.0)

    elif isinstance(config, MCLMCAdaptationState):
        L = config.L
        step_size = config.step_size
        inverse_mass_matrix = config.inverse_mass_matrix

    # Use the quick wrapper to build a new kernel with the tuned parameters
    sampler = blackjax.mclmc(logdf, L=L, step_size=step_size, 
                            inverse_mass_matrix=inverse_mass_matrix,
                            integrator=integrator,)

    # Run the sampler
    if thinning==1:
        state, history = run_inference_algorithm(
            rng_key=seed,
            initial_state=state,
            inference_algorithm=sampler,
            num_steps=n_samples,
            transform=transform,
            progress_bar=progress_bar,
        )
    else:
        state, history = run_with_thinning(
            seed=seed,
            inference_algorithm=sampler,
            num_steps=n_samples,
            initial_state=state,
            transform=transform,
            progress_bar=progress_bar,
            thinning=thinning
        )
    samples, infos = history
    
    # Register number of evaluations
    infos |= {"n_evals": n_eval_per_steps * thinning * jnp.ones(n_samples)}
    return state, samples | infos


def get_mclmc_warmup(logdf, n_steps=None, config=None,
              desired_energy_var=5e-4, diagonal_preconditioning=False):
    return partial(mclmc_warmup,
                   logdf=logdf,
                   n_steps=n_steps,
                   config=config,
                   desired_energy_var=desired_energy_var,
                   diagonal_preconditioning=diagonal_preconditioning)


def get_mclmc_run(logdf, n_samples, transform=None, thinning=1, progress_bar=True):
    return partial(mclmc_run, 
                   logdf=logdf,
                   n_samples=n_samples,
                   transform=transform,
                   thinning=thinning,    
                   progress_bar=progress_bar)





########
# MAMS #
########
from blackjax.mcmc.adjusted_mclmc_dynamic import rescale
from blackjax.util import run_inference_algorithm

def mams_warmup(seed, init_pos, logdf, n_steps=0, config=None, 
              diagonal_preconditioning=False, 
              random_trajectory_length=True,
                L_proposal_factor=jnp.inf):
    init_seed, tune_seed = jr.split(seed, 2)

    # Create an initial state for the sampler
    state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=init_pos, logdensity_fn=logdf, random_generator_arg=init_seed
    )

    if config is None:
        n_dim = len(ravel_pytree(state.position)[0])
        config = MCLMCAdaptationState(
            # n_dim**.5, n_dim**.5 / 5, inverse_mass_matrix=jnp.ones(n_dim))
            n_dim**.5, n_dim**.5 / 64, inverse_mass_matrix=jnp.ones(n_dim))
        
    elif isinstance(config, dict):
        L = config['L']
        step_size = config['step_size']
        inverse_mass_matrix = config.get('inverse_mass_matrix', 1.0)
        config = MCLMCAdaptationState(L=L, step_size=step_size, inverse_mass_matrix=inverse_mass_matrix)

    else:
        assert isinstance(config, MCLMCAdaptationState), \
        "config must be either None, a dict, or a MCLMCAdaptationState"

    if n_steps > 0:
        # Build the kernel
        if random_trajectory_length:
            integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
                jr.uniform(k) * rescale(avg_num_integration_steps))
        else:
            integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(avg_num_integration_steps)

        kernel = lambda seed, state, avg_num_integration_steps, step_size, inverse_mass_matrix: blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
            integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
            inverse_mass_matrix=inverse_mass_matrix,
            integrator=isokinetic_mclachlan,
            # integrator=isokinetic_omelyan,
        )(
            rng_key=seed,
            state=state,
            step_size=step_size,
            logdensity_fn=logdf,
            L_proposal_factor=L_proposal_factor,
        )

        # target_acc_rate = 0.9 # our recommendation
        target_acc_rate = 0.65

        state, config, n_steps_tot = blackjax.adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=n_steps,
            state=state,
            rng_key=tune_seed,
            target=target_acc_rate,
            frac_tune1=0.1,
            frac_tune2=0.1,
            frac_tune3=0., # our recommendation
            diagonal_preconditioning=diagonal_preconditioning,
            params=config,
            )
        debug.print("Perform {n_steps_tot} adaptation steps", n_steps_tot=n_steps_tot)

    return state, config


def mams_run(seed, state, config:dict|MCLMCAdaptationState, logdf, n_samples,  
              transform=None, thinning=1, progress_bar=True, L_proposal_factor=jnp.inf):
    integrator = isokinetic_mclachlan

    if integrator == isokinetic_velocity_verlet:
        n_eval_per_steps = 1
    elif integrator == isokinetic_mclachlan:
        n_eval_per_steps = 2 
    elif integrator == isokinetic_yoshida:
        n_eval_per_steps = 3
    elif integrator == isokinetic_omelyan:
        n_eval_per_steps = 5

    if transform is None:
        transform = lambda state, info: (state.position, 
                                    {'logdensity': state.logdensity,
                                     'acceptance_rate': jnp.mean(info.acceptance_rate), 
                                     'n_evals': jnp.sum(info.num_integration_steps) * n_eval_per_steps})

    if isinstance(config, dict):
        L = config['L']
        step_size = config['step_size']
        inverse_mass_matrix = config.get('inverse_mass_matrix', 1.0)

    elif isinstance(config, MCLMCAdaptationState):
        L = config.L
        step_size = config.step_size
        inverse_mass_matrix = config.inverse_mass_matrix

    sampler = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn=logdf,
        step_size=step_size,
        integration_steps_fn=lambda seed: jnp.ceil(
            jr.uniform(seed) * rescale(L / step_size)
        ),
        inverse_mass_matrix=inverse_mass_matrix,
        L_proposal_factor=L_proposal_factor,
        integrator=integrator,
    )

     # Run the sampler
    if thinning==1:
        state, history = run_inference_algorithm(
            rng_key=seed,
            initial_state=state,
            inference_algorithm=sampler,
            num_steps=n_samples,
            transform=transform,
            progress_bar=progress_bar,
        )
    else:
        state, history = run_with_thinning(
            seed=seed,
            inference_algorithm=sampler,
            num_steps=n_samples,
            initial_state=state,
            transform=transform,
            progress_bar=progress_bar,
            thinning=thinning
        )
    samples, infos = history

    return state, samples | infos



def get_mams_warmup(logdf, n_steps=None, config=None,
              diagonal_preconditioning=False):
    return partial(mams_warmup,
                   logdf=logdf,
                   n_steps=n_steps,
                   config=config,
                   diagonal_preconditioning=diagonal_preconditioning)


def get_mams_run(logdf, n_samples, transform=None, thinning=1, progress_bar=True):
    return partial(mams_run, 
                   logdf=logdf,
                   n_samples=n_samples,
                   transform=transform,
                   thinning=thinning,    
                   progress_bar=progress_bar)















from functools import partial
from typing import Callable, Union

from blackjax.base import SamplingAlgorithm, VIAlgorithm
from blackjax.progress_bar import progress_bar_scan
from blackjax.types import ArrayLikeTree, PRNGKey

def run_with_thinning(
    seed: PRNGKey,
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
    seed
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
    thinning
        The thinning factor, meaning a state is returned every `thinning` steps.
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
        seed, init_seed = jr.split(seed, 2)
        initial_state = inference_algorithm.init(initial_position, init_seed)


    def one_sub_step(state, seed):
        state, info = inference_algorithm.step(seed, state)
        return state, info
    
    def one_step(state, xs):
        _, seed = xs
        seeds = jr.split(seed, thinning)
        state, info = lax.scan(one_sub_step, state, seeds)
        return state, transform(state, info)

    seeds = jr.split(seed, num_steps)
    xs = jnp.arange(num_steps), seeds
    
    scan_fn = gen_scan_fn(num_steps, progress_bar)
    final_state, history = scan_fn(one_step, initial_state, xs)

    return final_state, history







############################
# NumPyro API for HMC/NUTS #
############################
from montecosmo.utils import pdump

def save_run(mcmc:MCMC, i_run:int, path:str, extra_fields:list=None, group_by_chain:bool=True):
    """
    Save one run of MCMC sampling, with extra fields and last state.
    """
    # Save samples (and extra fields)
    samples = mcmc.get_samples(group_by_chain)

    if extra_fields is not None:
        extra = mcmc.get_extra_fields(group_by_chain)
        if "num_steps" in extra: # renaming num_steps into clearer n_evals
            n_evals = extra.pop("num_steps")
            samples.update(n_evals=n_evals)
        samples |= extra
        del extra

    jnp.savez(path+f"_{i_run}.npz", **samples) # better than pickle for dict of array-like
    del samples

    # Save or overwrite last state
    pdump(mcmc.last_state, path+f"_last_state.p")


def sample_and_save(mcmc:MCMC, path:str, start:int=0, end:int=1, extra_fields=(),
                    seed=42, group_by_chain:bool=True, init_params=None) -> MCMC:
    """
    Warmup and run MCMC, saving the specified variables and extra fields.
    If `mcmc.num_warmup > 0`, first step is a warmup step.
    So to continue a run, simply do before:
    ```
    mcmc.num_warmup = 0
    mcmc.post_warmup_state = last_state
    ```
    """
    if isinstance(seed, int):
        seed = jr.key(seed)

    # Warmup sampling
    if mcmc.num_warmup > 0:
        print(f"\nrun {start}/{end} (warmup)")

        # Warmup
        mcmc.warmup(seed, collect_warmup=True, extra_fields=extra_fields, init_params=init_params)
        save_run(mcmc, start, path, extra_fields, group_by_chain)

        # Print warmup last state infos
        print("mean_acc_prob:", mcmc.last_state.mean_accept_prob,
            "\nstep_size:", mcmc.last_state.adapt_state.step_size,
            "\nsqrt_invmm:", mcmc.last_state.adapt_state.mass_matrix_sqrt_inv)

        # Handling rng seed and destroy init_params
        seed_run = mcmc.post_warmup_state.rng_key
        init_params = None
        start += 1
    else:
        seed_run = seed

    # Run sampling
    for i_run in range(start, end+1):
        print(f"\nrun {i_run}/{end}")

        # Run
        mcmc.run(seed_run, extra_fields=extra_fields, init_params=init_params)
        save_run(mcmc, i_run, path, extra_fields)

        # Init next run at last state
        mcmc.post_warmup_state = mcmc.last_state
        seed_run = mcmc.post_warmup_state.rng_key
    return mcmc







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





