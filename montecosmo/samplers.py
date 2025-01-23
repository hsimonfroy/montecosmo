
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, grad, debug, lax, flatten_util
from jax.tree_util import tree_map
from functools import partial

import blackjax
import blackjax.progress_bar

#########################
# HMC/NUTS within Gibbs #
#########################

def mwg_warmup(rng_key, state, logpdf, parameters, n_samples=0):
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

        wind_adapt = blackjax.window_adaptation(blackjax.nuts, logpdf_k, **parameters[k], progress_bar=False) 
        # NOTE: Progress bar can yield "NotImplementedError: IO effect not supported in vmap-of-cond"
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



def mwg_kernel_general(rng_key, state, logpdf, step_fn, init_fn, parameters):
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
    parameters
        Dictionary with the same keys as ``state``, each of which is a dictionary of parameters to
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
            **parameters[k]
        )

        # register only relevant infos
        n_evals = info.num_integration_steps
        infos['infos_'+k] = {"acceptance_rate": info.acceptance_rate, 
                                "num_integration_steps": n_evals}
        infos['n_evals'] += n_evals
    
    return state, infos
    


def sampling_loop_general(rng_key, initial_state, logpdf, step_fn, init_fn, parameters, n_samples):
    
    @blackjax.progress_bar.progress_bar_scan(n_samples)
    def one_step(state, xs):
        _, rng_key = xs
        state, infos = mwg_kernel_general(
            rng_key=rng_key,
            state=state,
            logpdf=logpdf,
            step_fn=step_fn,
            init_fn=init_fn,
            parameters=parameters
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



def NUTSwG_init(logpdf, kernel="NUTS"):
    init_ss = 1e-3
    target_acc_rate = 0.65

    if kernel == "HMC":
        ker_api = blackjax.hmc
        parameters = {
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
        parameters = {
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

    return step_fn, init_fn, parameters, init_state_fn


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


def NUTSwG_run(rng_key, init_state, parameters, logpdf, step_fn, init_fn, n_samples):
    last_state, (samples, infos) = sampling_loop_general(
                                rng_key=rng_key,
                                initial_state=init_state,
                                logpdf=logpdf,
                                step_fn=step_fn,
                                init_fn=init_fn,
                                parameters=parameters,
                                n_samples=n_samples,)
    return last_state, samples, infos

def get_NUTSwG_run(logpdf, step_fn, init_fn, n_samples):
    return partial(NUTSwG_run, 
                   logpdf=logpdf, 
                   step_fn=step_fn, 
                   init_fn=init_fn, 
                   n_samples=n_samples,)


def NUTSwG_warm(rng_key, init_state, logpdf, parameters, n_samples):
    (last_state, parameters), (samples, infos) = mwg_warmup(rng_key, init_state, logpdf, parameters, n_samples)
    return (last_state, parameters), samples, infos

def get_NUTSwG_warm(logpdf, parameters, n_samples):
    return partial(NUTSwG_warm, 
                   logpdf=logpdf, 
                   parameters=parameters, 
                   n_samples=n_samples)












#########
# MCLMC #
#########
import blackjax

def MCLMC_run(key, init_state, logpdf, n_samples, transform):
    init_key, tune_key, run_key = jr.split(key, 3)

    # Create an initial state for the sampler
    initial_state = blackjax.mcmc.mclmc.init(
        position=init_state, logdensity_fn=logpdf, rng_key=init_key
    )

    # Build the kernel
    kernel = blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logpdf,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
    )

    ########
    # # Find values for L and step_size
    # tunning = blackjax.adaptation.mclmc_adaptation.mclmc_find_L_and_step_size(
    #     mclmc_kernel=kernel,
    #     num_steps=n_samples,
    #     state=initial_state,
    #     rng_key=tune_key,
    #     num_effective_samples=1024,
    # )
    # state_after_tuning, mclmc_sampler_params = tunning
    # L = mclmc_sampler_params.L
    # step_size = mclmc_sampler_params.step_size
    # initial_state = state_after_tuning
    ########

    ########
    L = 25
    step_size = 2
    initial_state = init_state
    ########

    # if not isinstance(initial_state, blackjax.mcmc.integrators.IntegratorState):
    #     from jax import debug
    #     debug.print("rep")
    #     initial_state = mult_tree(initial_state, invmm**(-.5))
    # tunning = (initial_state, {'L':L, 'step_size':step_size})


    # use the quick wrapper to build a new kernel with the tuned parameters
    sampling_alg = blackjax.mclmc(
        logpdf,
        L=L,
        step_size=step_size,
    )

    # run the sampler
    last_state, samples, info = blackjax.util.run_inference_algorithm(
        rng_key = run_key,
        initial_state_or_position = initial_state,
        inference_algorithm = sampling_alg,
        num_steps = n_samples,
        transform = transform,
        progress_bar = True,
    )

    # Register only relevant infos
    infos = {"num_steps":jnp.ones(n_samples)}
    return last_state, samples, infos

def get_MCLMC_run(logdensity, n_samples, transform):
    return partial(MCLMC_run, 
                   logdensity = logdensity,
                   n_samples = n_samples,
                   transform = transform,)








def run_mclmc(logdensity_fn, num_steps, initial_position, key, transform, desired_energy_variance= 5e-4):
    init_key, tune_key, run_key = jr.split(key, 3)

    # create an initial state for the sampler
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    # build the kernel
    kernel = lambda sqrt_diag_cov : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        sqrt_diag_cov=sqrt_diag_cov,
    )

    # find values for L and step_size
    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
    ) = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        diagonal_preconditioning=False,
        desired_energy_var=desired_energy_variance
    )

    # use the quick wrapper to build a new kernel with the tuned parameters
    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=blackjax_mclmc_sampler_params.L,
        step_size=blackjax_mclmc_sampler_params.step_size,
    )

    # run the sampler
    _, samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=True,
    )

    return samples, blackjax_state_after_tuning, blackjax_mclmc_sampler_params, run_key




logdensity = logp_fn
transform = lambda x: x.position
n_samples, n_runs, n_chains = 512, 5, 4
# n_samples, n_runs, n_chains = 512, 100, 8
save_path = save_dir + f"MCLMC_ns{n_samples:d}_test2"

run_fn = jit(vmap(get_MCLMC_run(logdensity, n_samples, transform=transform)))
key = jr.key(42)
# last_state = init_params_
last_state = tree_map(lambda x: x[:n_chains], init_params_)

for i_run in range(1, n_runs+1):
    print(f"run {i_run}/{n_runs}")
    key, run_key = jr.split(key, 2)
    last_state, samples, infos = run_fn(jr.split(run_key, n_chains), last_state)
    samples = tree_map(lambda x: x[:,::2], samples)
    infos = tree_map(lambda x: 2*x[:,::2], infos)
    pdump(samples | infos, save_path+f"_{i_run}.p")
    pdump(last_state, save_path+f"_laststate.p")