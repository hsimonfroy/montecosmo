
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, grad, debug, lax, tree
from functools import partial

import blackjax
import blackjax.progress_bar
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

#########################
# HMC/NUTS within Gibbs #
#########################

def mwg_warmup(rng_key, state, logpdf, config, n_samples=0):
    rng_keys = jr.split(rng_key, num=len(state))
    rng_keys = dict(zip(state.keys(), rng_keys))

    # avoid modifying argument state as JAX functions should be pure
    state = state.copy()
    infos = {}
    infos['n_evals'] = 0
    params = {}
    positions = {}

    for k in state.keys():
        # logpdf of component k conditioned on all other components in state
        union = {}
        for _k in state.keys():
            union |= state[_k].position

        def logpdf_k(value):
            return logpdf(union | value) # update component k

        # give state[k] the right log_density 
        # NOTE: unnecessary if we only pass position to warmup
        # state[k] = init_fn[k](
        #     position=state[k].position,
        #     logdensity_fn=logpdf_k
        # )

        wind_adapt = blackjax.window_adaptation(blackjax.nuts, logpdf_k, **config[k], progress_bar=False) 
        # NOTE: window adapt progress bar can yield "NotImplementedError: IO effect not supported in vmap-of-cond"
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



def mwg_kernel_general(rng_key, state, logpdf, step_fn, init_fn, config):
    """
    General MWG kernel.

    Updates each component of ``state`` conditioned on all the others using a component-specific MCMC algorithm

    Parameters
    ----------
    rng_key
        The PRNG key.
    state
        Dictionary where each item is the state of an MCMC algorithm, i.e., an object of type ``AlgorithmState``.
    logpdf
        The log-density function on all components, where the arguments are the keys of ``state``.
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
        # logpdf of component k conditioned on all other components in state
        union = {}
        for _k in state.keys():
            union |= state[_k].position

        def logpdf_k(value):
            return logpdf(union | value) # update component k
        
        # give state[k] the right log_density
        state[k] = init_fn[k](
            position=state[k].position,
            logdensity_fn=logpdf_k
        )

        # update state[k]
        state[k], info = step_fn[k](
            rng_key=rng_keys[k],
            state=state[k],
            logdensity_fn=logpdf_k,
            **config[k]
        )

        # register only relevant infos
        n_evals = info.num_integration_steps
        infos['infos_'+k] = {"acceptance_rate": info.acceptance_rate, 
                                "num_integration_steps": n_evals}
        infos['n_evals'] += n_evals
    
    return state, infos
    


def sampling_loop_general(rng_key, initial_state, logpdf, step_fn, init_fn, config, n_samples):
    
    @blackjax.progress_bar.progress_bar_scan(n_samples)
    def one_step(state, xs):
        _, rng_key = xs
        state, infos = mwg_kernel_general(
            rng_key=rng_key,
            state=state,
            logpdf=logpdf,
            step_fn=step_fn,
            init_fn=init_fn,
            config=config
        )
        # positions = {k: state[k].position for k in state.keys()}
        union = {}
        for _k in state.keys():
            union |= state[_k].position
        # union = mult_tree(union, invmm**.5)
        return state, (union, infos)

    keys = jr.split(rng_key, n_samples)
    xs = (jnp.arange(n_samples), keys)
    last_state, (positions, infos) = lax.scan(one_step, initial_state, xs) # scan compile

    return last_state, (positions, infos)



def nutswg_init(logpdf, kernel="NUTS"):
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
        return get_init_state(init_pos, logpdf, init_fn)

    return step_fn, init_fn, config, init_state_fn


def get_init_state(init_pos, logpdf, init_fn):
    init_pos_block1 = {name:init_pos[name] for name in ['init_mesh_']}
    init_pos_block2 = {name:init_pos[name] for name in ['Omega_m_','sigma8_','b1_','b2_','bs2_','bn2_']}
    init_state = {
        "mesh_": init_fn['mesh_'](
            position = init_pos_block1,
            logdensity_fn = lambda x: logpdf(x |init_pos_block2)
        ),
        "rest_": init_fn['rest_'](
            position = init_pos_block2,
            logdensity_fn = lambda y: logpdf(y | init_pos_block1)
        )
    }
    return init_state


def nutswg_run(rng_key, init_state, config, logpdf, step_fn, init_fn, n_samples):
    last_state, (samples, infos) = sampling_loop_general(
                                rng_key=rng_key,
                                initial_state=init_state,
                                logpdf=logpdf,
                                step_fn=step_fn,
                                init_fn=init_fn,
                                config=config,
                                n_samples=n_samples,)
    return samples, infos, last_state, config

def get_nutswg_run(logpdf, step_fn, init_fn, n_samples):
    return partial(nutswg_run, 
                   logpdf=logpdf, 
                   step_fn=step_fn, 
                   init_fn=init_fn, 
                   n_samples=n_samples,)


def nutswg_warm(rng_key, init_state, logpdf, config, n_samples):
    (last_state, config), (samples, infos) = mwg_warmup(rng_key, init_state, logpdf, config, n_samples)
    return samples, infos, last_state, config

def get_nutswg_warm(logpdf, config, n_samples):
    return partial(nutswg_warm, 
                   logpdf=logpdf, 
                   config=config, 
                   n_samples=n_samples)












#########
# MCLMC #
#########
def mclmc_run(key, init_pos, logpdf, n_samples, config=None, transform=None, desired_energy_variance= 5e-4):
    if transform is None:
        transform = lambda x: x.position
    init_key, tune_key, run_key = jr.split(key, 3)

    # Create an initial state for the sampler
    state = blackjax.mcmc.mclmc.init(
        position=init_pos, logdensity_fn=logpdf, rng_key=init_key
    )


    if config is None:
        # Build the kernel
        kernel = blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=logpdf,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            # integrator=blackjax.mcmc.integrators.isokinetic_velocity_verlet,
            # integrator=blackjax.mcmc.integrators.isokinetic_leapfrog,
        )

        # Find values for L and step_size
        state, config = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=n_samples,
            state=state,
            rng_key=tune_key,
            num_effective_samples=1024,
        )

        # # Build the kernel
        # kernel = lambda sqrt_diag_cov : blackjax.mcmc.mclmc.build_kernel(
        #     logdensity_fn=logpdf,
        #     integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        #     sqrt_diag_cov=sqrt_diag_cov,
        # )

        # # Find values for L and step_size
        # state, config = blackjax.mclmc_find_L_and_step_size(
        #     mclmc_kernel=kernel,
        #     num_steps=n_samples,
        #     state=state,
        #     rng_key=tune_key,
        #     diagonal_preconditioning=False,
        #     desired_energy_var=desired_energy_variance
        #     # num_effective_samples=200,
        #     )


        L = config.L
        step_size = config.step_size
        return state, config

    elif isinstance(config, dict):
        L = config['L']
        step_size = config['step_size']

    elif isinstance(config, MCLMCAdaptationState):
        L = config.L
        step_size = config.step_size






    # Use the quick wrapper to build a new kernel with the tuned parameters
    sampler = blackjax.mclmc(logpdf,L=L, step_size=step_size,)

    # Run the sampler
    last_state, samples, info = blackjax.util.run_inference_algorithm(
    # last_state, samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        # initial_state=state,
        initial_state_or_position=state,
        inference_algorithm=sampler,
        num_steps=n_samples,
        transform=transform,
        progress_bar=True,
    )
    
    # Register only relevant infos
    n_eval_per_steps = 2 # 1 for velocity verlet, 2 for mclachlan
    infos = {"n_evals": n_eval_per_steps * jnp.ones(n_samples)}

    return samples, infos, last_state, config




def get_mclmc_run(logpdf, n_samples, config=None, transform=None, desired_energy_variance= 5e-4):
    return partial(mclmc_run, 
                   logpdf=logpdf,
                   n_samples=n_samples,
                   config=config,
                   transform=transform,
                   desired_energy_variance=desired_energy_variance,)











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