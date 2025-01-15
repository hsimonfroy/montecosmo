
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

def mwg_warmup(rng_key, state, logdensity_fn, init_fn, parameters, n_samples=0):
    rng_keys = jr.split(rng_key, num=len(state))
    rng_keys = dict(zip(state.keys(), rng_keys))

    # avoid modifying argument state as JAX functions should be pure
    state = state.copy()
    infos = {}
    infos['n_evals'] = 0
    params = {}
    positions = {}

    for k in state.keys():
        # logdensity of component k conditioned on all other components in state
        union = {}
        for _k in state.keys():
            union |= state[_k].position

        def logdensity_k(value):
            return logdensity_fn(union | value) # update component k

        # give state[k] the right log_density NOTE: unnecessary if we only pass position to warmup
        # state[k] = init_fn[k](
        #     position=state[k].position,
        #     logdensity_fn=logdensity_k
        # )

        wind_adapt = blackjax.window_adaptation(blackjax.nuts, logdensity_k, **parameters[k], progress_bar=True)
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



def mwg_kernel_general(rng_key, state, logdensity_fn, step_fn, init_fn, parameters):
    """
    General MWG kernel.

    Updates each component of ``state`` conditioned on all the others using a component-specific MCMC algorithm

    Parameters
    ----------
    rng_key
        The PRNG key.
    state
        Dictionary where each item is the state of an MCMC algorithm, i.e., an object of type ``AlgorithmState``.
    logdensity_fn
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
        # logdensity of component k conditioned on all other components in state
        union = {}
        for _k in state.keys():
            union |= state[_k].position

        def logdensity_k(value):
            return logdensity_fn(union | value) # update component k
        
        # give state[k] the right log_density
        state[k] = init_fn[k](
            position=state[k].position,
            logdensity_fn=logdensity_k
        )

        # update state[k]
        state[k], info = step_fn[k](
            rng_key=rng_keys[k],
            state=state[k],
            logdensity_fn=logdensity_k,
            **parameters[k]
        )

        # register only relevant infos
        n_evals = info.num_integration_steps
        infos['infos_'+k] = {"acceptance_rate": info.acceptance_rate, 
                                "num_integration_steps": n_evals}
        infos['n_evals'] += n_evals
    
    return state, infos
    


def sampling_loop_general(rng_key, initial_state, logdensity_fn, step_fn, init_fn, parameters, n_samples):
    
    @blackjax.progress_bar.progress_bar_scan(n_samples)
    def one_step(state, xs):
        _, rng_key = xs
        state, infos = mwg_kernel_general(
            rng_key=rng_key,
            state=state,
            logdensity_fn=logdensity_fn,
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



def NUTSwG_init(logdensity, kernel="NUTS"):

    if kernel == "HMC":
        ker_api = blackjax.hmc
        parameters = {
            "mesh_": {
                # "inverse_mass_matrix": jnp.ones(64**3),
                "num_integration_steps": 256,
                # "step_size": 3*1e-3
            },
            "rest_": {
                # "inverse_mass_matrix": jnp.ones(6),
                "num_integration_steps": 64,
                # "step_size": 3*1e-3
            }
        }
    elif kernel == "NUTS":
        ker_api = blackjax.nuts
        parameters = {
            "mesh_": {
                # "inverse_mass_matrix": jnp.ones(64**3),
                # "step_size": 3*1e-3
            },
            "rest_": {
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
        return get_init_state(init_pos, logdensity, init_fn)

    return step_fn, init_fn, parameters, init_state_fn


def get_init_state(init_pos, logdensity, init_fn):
    init_pos_block1 = {name:init_pos[name] for name in ['init_mesh_']}
    init_pos_block2 = {name:init_pos[name] for name in ['Omega_m_','sigma8_','b1_','b2_','bs2_','bn2_']}
    init_state = {
        "mesh_": init_fn['mesh_'](
            position = init_pos_block1,
            logdensity_fn = lambda x: logdensity(x |init_pos_block2)
        ),
        "rest_": init_fn['rest_'](
            position = init_pos_block2,
            logdensity_fn = lambda y: logdensity(y | init_pos_block1)
        )
    }
    return init_state


def NUTSwG_run(rng_key, init_state, logdensity, step_fn, init_fn, parameters, n_samples, warmup=False):
    if warmup:
        (last_state, parameters), (samples, infos) = mwg_warmup(rng_key, init_state, logdensity, init_fn, parameters, n_samples)
        return (last_state, parameters), samples, infos

    else:
        last_state, (samples, infos) = sampling_loop_general(
                                    rng_key = rng_key,
                                    initial_state = init_state,
                                    logdensity_fn = logdensity,
                                    step_fn = step_fn,
                                    init_fn = init_fn,
                                    parameters = parameters,
                                    n_samples = n_samples,)
        return last_state, samples, infos


def get_NUTSwG_run(logdensity, step_fn, init_fn, parameters, n_samples, warmup=False):
    return partial(NUTSwG_run, 
                   logdensity=logdensity, 
                   step_fn=step_fn, 
                   init_fn=init_fn, 
                   parameters=parameters, 
                   n_samples=n_samples,
                   warmup=warmup)