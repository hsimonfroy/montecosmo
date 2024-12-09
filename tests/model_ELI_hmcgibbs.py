#!/usr/bin/env python
# coding: utf-8

# # Model Explicit Likelihood Inference
# Infer from a cosmological model via MCMC samplers. 

# In[48]:


import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.99' # NOTE: jax preallocates GPU (default 75%)

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, grad, debug, lax, flatten_util
from jax.tree_util import tree_map

import numpyro
from numpyro.handlers import seed, condition, trace
from functools import partial
from getdist import plots


from montecosmo.utils import pdump, pload, get_vlim, theme_switch, sample_and_save, load_runs
save_dir = os.path.expanduser("~/scratch/pickles/")


# In[49]:




# ## Inference

# ### Import

# In[3]:


from montecosmo.models import pmrsd_model, prior_model, get_logp_fn, get_score_fn, get_simulator, get_pk_fn, get_param_fn
from montecosmo.models import print_config, get_prior_mean, default_config as config

# Build and render model
# config.update(a_lpt=0.5, mesh_size=8*np.ones(3, dtype=int))
# config.update(a_lpt=0.5, mesh_size=64*np.ones(3, dtype=int), fourier=True)
# config.update(a_lpt=0.1, mesh_size=64*np.ones(3, dtype=int), fourier=False)
model = partial(pmrsd_model, **config)
print_config(model)

# # Get fiducial parameters
# param_fn = get_param_fn(**config)
# fiduc_model = condition(partial(model, trace_reparam=True), param_fn(inverse=True, **get_prior_mean(model)))
# fiduc_params = get_simulator(fiduc_model)(rng_seed=0)

# # # Chain init
# @jit
# @vmap
# def sample_init_chains(rng_key, scale_std):
#     params_ = seed(prior_model, rng_key)(**config)
#     init_params = get_param_fn(scale_std=scale_std, **config)(**params_)
#     return get_param_fn(**config)(inverse=True, **init_params)

# init_params_ = sample_init_chains(jr.split(jr.key(1), 7), jnp.array([0]+6*[1/10]))
# init_params_ = tree_map(lambda x,y: jnp.concatenate((jnp.array(x)[None], y), axis=0), 
#                         get_param_fn(**config)(inverse=True, **fiduc_params), init_params_)
# pickle_dump(fiduc_params, save_dir+"fiduc_params_fourier.p")
# pickle_dump(init_params_, save_dir+"init_params_fourier_.p")

# Load fiducial and chain init params
fiduc_params = pload(save_dir+"fiduc_params.p")
init_params_ = pload(save_dir+"init_params_.p")
# fiduc_params = pickle_load(save_dir+"fiduc_params_pm.p")
# init_params_ = pickle_load(save_dir+"init_params_pm_.p")
# fiduc_params = pickle_load(save_dir+"fiduc_params_fourier.p")
# init_params_ = pickle_load(save_dir+"init_params_fourier_.p")

# Condition model on observables
obs_names = ['obs_mesh']
# obs_names = ['obs_mesh','Omega_m_','sigma8_','b1_','b2_','bs2_','bn2_']
obs_params = {name: fiduc_params[name] for name in obs_names}
obs_model = condition(model, obs_params)
logp_fn = get_logp_fn(obs_model)
param_fn = get_param_fn(**config)
# print(fiduc_params, init_params_)


# In[4]:


print(fiduc_params.keys(), '\n', init_params_['Omega_m_'], '\n', init_params_['init_mesh_'][:,0,0,0])


# In[5]:


print(fiduc_params.keys(), '\n', init_params_['Omega_m_'], '\n', init_params_['init_mesh_'][:,0,0,0])




# In[43]:


import blackjax
import blackjax.progress_bar

def mwg_warmup(rng_key, state, logdensity_fn, init_fn, parameters, n_samples=0):
    rng_keys = jr.split(rng_key, num=len(state))
    rng_keys = dict(zip(state.keys(), rng_keys))

    # avoid modifying argument state as JAX functions should be pure
    state = state.copy()
    infos = {}
    infos['num_steps'] = 0
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
        num_steps = info.info.num_integration_steps
        infos['infos_'+k] = {"acceptance_rate": info.info.acceptance_rate, 
                                "num_integration_steps": num_steps}
        infos['num_steps'] += num_steps
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
    init
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
    infos['num_steps'] = 0

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
        num_steps = info.num_integration_steps
        infos['infos_'+k] = {"acceptance_rate": info.acceptance_rate, 
                                "num_integration_steps": num_steps}
        infos['num_steps'] += num_steps
    
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




def HMCGibbs_init(logdensity, kernel="hmc"):

    if kernel == "hmc":
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
    elif kernel == "nuts":
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


def HMCGibbs_run(rng_key, init_state, logdensity, step_fn, init_fn, parameters, n_samples, warmup=True):
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


def get_HMCGibbs_run(logdensity, step_fn, init_fn, parameters, n_samples, warmup=0):
    return partial(HMCGibbs_run, 
                   logdensity=logdensity, 
                   step_fn=step_fn, 
                   init_fn=init_fn, 
                   parameters=parameters, 
                   n_samples=n_samples,
                   warmup=warmup)


# warmup = blackjax.window_adaptation(blackjax.hmc, logp_fn, num_integration_steps=10)
# rng_key, warmup_key, sample_key = jr.split(jr.key(0), 3)
# (state, parameters), infos = warmup.run(warmup_key, init_params_one_, num_steps=10)


# In[44]:


logdensity = logp_fn
n_samples, n_runs, n_chains = 256, 20, 8
save_path = save_dir + f"NUTSGibbs_ns{n_samples:d}_x_nc{n_chains}"

step_fn, init_fn, parameters, init_state_fn = HMCGibbs_init(logdensity, "nuts")
warmup_fn = jit(vmap(get_HMCGibbs_run(logdensity, step_fn, init_fn, parameters, n_samples, warmup=True)))
key = jr.key(42)
# last_state = jit(vmap(init_state_fn))(init_params_)
last_state = pload(save_dir+"NUTSGibbs/HMCGibbs_ns256_x_nc8_laststate32.p")


(last_state, parameters), samples, infos = warmup_fn(jr.split(jr.key(43), n_chains), last_state)
print(parameters,'\n=======\n')
pdump(samples | infos, save_path+f"_{0}.p")
pdump(last_state, save_path+f"_laststate.p")

run_fn = jit(vmap(get_HMCGibbs_run(logdensity, step_fn, init_fn, parameters, n_samples)))
i_shift = 0
for i_run in range(i_shift+1, i_shift+n_runs+1):
    print(f"run {i_run}/{n_runs}")
    key, run_key = jr.split(key, 2)
    # last_state, samples, infos = run_fn(jr.split(run_key, n_chains), last_state)
    last_state, samples, infos = run_fn(jr.split(run_key, n_chains), last_state, parameters=parameters)
    pdump(samples | infos, save_path+f"_{i_run}.p")
    pdump(last_state, save_path+f"_laststate.p")

raise

# ### NUTS, HMC

# In[108]:


num_samples, max_tree_depth, n_runs, num_chains = 256, 10, 44, 8
# num_samples, max_tree_depth, n_runs, num_chains = 128, 10, 10, 4
# num_samples, max_tree_depth, n_runs, num_chains = 128, 10, 5, 4
# num_samples, max_tree_depth, n_runs, num_chains = 64, 10, 5, 1

# Variables to save
extra_fields = ['num_steps'] # e.g. 'num_steps'
save_path = save_dir + f"HMC_ns{num_samples:d}_x_nc{num_chains}_2"
# save_path = save_dir + f"NUTS_ns{num_samples:d}_x_nc{num_chains}_pm"
# save_path = save_dir + f"HMCGibbs_ns{num_samples:d}_test"
# save_path = save_dir + f"NUTS_ns{num_samples:d}_test_fourier"

nuts_kernel = numpyro.infer.NUTS(
    model=obs_model,
    # init_strategy=numpyro.infer.init_to_value(values=fiduc_params)
    # inverse_mass_matrix=variance_as_invM, 
    adapt_mass_matrix=True,
    # dense_mass=[('Omega_c_base', 'sigma8_base')], # XXX: dense matrix for cosmo params joint, diagonal for the rest
    step_size=1e-4, 
    adapt_step_size=True,
    max_tree_depth=max_tree_depth,)

hmc_kernel = numpyro.infer.HMC(
    model=obs_model,
    # init_strategy=numpyro.infer.init_to_value(values=fiduc_params),
    adapt_mass_matrix=True,
    step_size=1e-3, 
    adapt_step_size=True,
    trajectory_length= 1023 * 3*1e-3 / 4, # (2**max_tree_depth-1)*step_size_NUTS/(2 to 4), compare with default 2pi.
    )


def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
    pass
hmcgibbs_kernel = numpyro.infer.HMCGibbs(hmc_kernel, 
                                         gibbs_fn=gibbs_fn, 
                                         mgibbs_sites=['Omega_m_','sigma8_','b1_','b2_','bs2_','bn2_'])

# # Propose MALA step size based on [Chen+2019](http://arxiv.org/abs/1801.02309)
# L_smoothness, m_strong_convex = 1, 1 # log density regularity properties
# condition_number = L_smoothness / m_strong_convex
# print(f"MALA step size proposal={1 / (L_smoothness * (config['mesh_size'].prod() * condition_number)**0.5):e}")

# from numpyro.contrib.tfp.mcmc import MetropolisAdjustedLangevinAlgorithm as MALA
# mala_kernel = MALA(model=obs_model,
#                     init_strategy=numpyro.infer.init_to_value(values=fiduc_params),
#                     step_size=0.001,)

mcmc = numpyro.infer.MCMC(
    sampler=nuts_kernel,
    num_warmup=0,
    # num_warmup=num_samples,
    num_samples=num_samples, # for each run
    num_chains=num_chains,
    chain_method="vectorized",
    progress_bar=True,)

last_state = pload(save_dir+"HMC/HMC_ns256_x_nc8"+"_laststate20.p")
print("mean_acc_prob:", last_state.mean_accept_prob, "\nss:", last_state.adapt_state.step_size)
mcmc.post_warmup_state = last_state
# invmm = list(last_state.adapt_state.inverse_mass_matrix.values())[0][0]
# invmm.min(),invmm.max(),invmm.mean(),invmm.std()


# In[16]:


# mlflow.end_run()
# mlflow.start_run(run_name="HMC, ss=1e-3")
# mlflow.log_params(config)
# mlflow.log_params({'n_runs':n_runs, 'num_samples':num_samples, 'max_tree_depth':max_tree_depth, 'num_chains':num_chains})
print({'n_runs':n_runs, 'num_samples':num_samples, 'max_tree_depth':max_tree_depth, 'num_chains':num_chains})
print(save_path)


# In[ ]:


# init_params_one_ = tree_map(lambda x: x[:num_chains], init_params_)
# mlflow.log_metric('halt',0) # 31.46s/it 4chains, 37.59s/it 8chains
# mcmc_runned = sample_and_save(mcmc, n_runs, save_path, extra_fields=extra_fields, init_params=init_params_one_)
mcmc_runned = sample_and_save(mcmc, n_runs, save_path, extra_fields=extra_fields, init_params=init_params_)
# mlflow.log_metric('halt',1)


# ## Analysis

# In[23]:


start_run, end_run = 0,1
var_names = [name+'_' for name in config['prior_config']] + ['num_steps']
# var_names = None

post_samples_ = load_runs(save_path, start_run, end_run, var_names, conc_axis=[1,0], verbose=True)
# post_samples_ = load_runs(save_path, start_run, end_run, var_names, conc_axis=[1], verbose=True)
# mlflow.log_params({'n_samples':n_samples, 'n_evals':n_evals})
# post_samples = [param_vfn(**s) for s in post_samples_]
post_samples = vmap(param_fn)(**post_samples_)


# In[49]:


post_samples_ = tree_map(lambda x: x[1], post_samples_)
post_samples = tree_map(lambda x: x[1], post_samples)


# ### Chain

# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
def _plot_chain(samples:dict, prior_config:dict, fiduc:dict, **config):
    labels = []
    for name in samples:
        if name.endswith('_'): # convention for a latent value 
            lab = "\\overline"+prior_config[name[:-1]][0]
        else:
            lab = prior_config[name][0]
        labels.append('$'+lab+'$')

    samples_arr = np.array(list((samples.values()))).T

    plt.plot(samples_arr, label=labels)
    plt.hlines([fiduc[name] for name in samples], 
            xmin=0, xmax=len(samples_arr), 
            ls="--", alpha=0.75,
            color=[f"C{i}" for i in range(len(samples))],)
# slice_toplot = np.concatenate([range(i,i+10) for i in [0,5*60-5, 6*60-5]])

def plot_chain(samples_:dict, prior_config:dict, fiduc:dict, **config):
    samples = get_param_fn(prior_config=prior_config, **config)(**samples_)
    plot_fn = partial(_plot_chain, prior_config=prior_config, fiduc=fiduc)

    plt.figure(figsize=(10,6))
    plt.subplot(221)
    plot_fn({name:samples_[name] for name in ['Omega_m_','sigma8_']})
    plt.legend(), 
    plt.subplot(222)
    plot_fn({name:samples_[name] for name in ['b1_','b2_','bs2_','bn2_']})
    plt.legend(), 
    plt.subplot(223)
    plot_fn({name:samples[name] for name in ['Omega_m','sigma8']})
    plt.legend(), 
    plt.subplot(224)
    plot_fn({name:samples[name] for name in ['b1', 'b2','bs2','bn2']})
    plt.legend(), 
    plt.tight_layout()
    # mlflow.log_figure(plt.gcf(), f"HMC_chain_L1o2_neval1102004.svg")
    plt.show();

# plot_chain(post_samples_[1], fiduc=fiduc_params, **config)
plot_chain(post_samples_, fiduc=fiduc_params, **config)


# ### Contours

# In[40]:


from numpyro.diagnostics import effective_sample_size, gelman_rubin

def get_metric_traj(metric_fn, samples, num, num_steps=None):
    def _get_metric_traj(samples):
        metrics = []
        length = len(samples[0])
        for i_filt in np.arange(length, 1, -length// num)[::-1]:

            if num_steps is not None:
                metrics.append([num_steps[:,:i_filt].sum(), metric_fn(samples[:,:i_filt])])
            else:
                metrics.append(metric_fn(samples[:,:i_filt]))
        return jnp.array(metrics).T
    return tree_map(_get_metric_traj, samples)

n_toplot = 300
ESSs = get_metric_traj(effective_sample_size, post_samples, n_toplot, post_samples_['num_steps'])
GRs = get_metric_traj(gelman_rubin, post_samples, n_toplot, post_samples_['num_steps'])


# In[46]:


plt.figure(figsize=(12,4))
plt.subplot(121)
i_tostart = 50
plot_fn = lambda x, **kwargs: plt.plot(x[0], x[1]/x[0], **kwargs)
for name, val in ESSs.items():
    plot_fn(val[:,i_tostart:], label='$'+config['prior_config'][name][0]+'$')
plt.xlabel("$n_{\\text{eval}}$"), plt.ylabel("$n_{\\text{ESS}}/n_{\\text{eval}}$")
plt.legend()

plt.subplot(122)
plot_fn = lambda x, **kwargs: plt.plot(*x, **kwargs)
for name, val in GRs.items():
    plot_fn(val[:,i_tostart:], label='$'+config['prior_config'][name][0]+'$')
plt.xlabel("$n_{\\text{eval}}$"), plt.ylabel("$\\hat R$")
plt.legend()
plt.tight_layout()
plt.show();


# In[44]:


paths = ["NUTS_ns256_x_nc8","HMC_ns256_x_nc8"]
mc_labels = ["NUTS","HMC"]
start_run, end_run = [2,2], [20,20]

load_paths = np.array([os.path.join(save_dir, path) for path in paths])
var_names = [name+'_' for name in config['prior_config']] + ['num_steps']

# post_samples_ = load_runs(load_paths, start_run, end_run, var_names, conc_axis=[1,0], verbose=True)
post_samples_ = load_runs(load_paths, start_run, end_run, var_names, conc_axis=[1], verbose=True)

post_samples = [param_fn(**s) for s in post_samples_]


# In[45]:


from numpyro.diagnostics import print_summary
for sample in post_samples:
    print_summary(sample, group_by_chain=True) # NOTE: group_by_chain if several chains


# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')
from montecosmo.utils import get_gdsamples, get_gdprior

gdsamples = get_gdsamples(post_samples, label=mc_labels, verbose=True, **config)
for i_gds, gds in enumerate(gdsamples):
    gdsamples[i_gds] = gds.copy(label=mc_labels[i_gds]+", 1$\\sigma$-smooth", settings={'smooth_scale_2D':1,'smooth_scale_1D':1,})

# gdsamples.append(get_gdprior(post_samples, verbose=True, **config))
g = plots.get_subplot_plotter(width_inch=9)
# g.settings.solid_colors='tab10_r's
g.triangle_plot(roots=gdsamples , 
                title_limit=1, 
                filled=True, 
                # param_limits={n:[m-2*s,m+2*s] for n,m,s in zip(names, mean, std)},
                markers=fiduc_params,
                )
# mlflow.log_figure(plt.gcf(), f"NUTS_contour_mtd3-8-10-12.svg", save_kwargs={'bbox_inches':'tight'}) # NOTE: tight bbox better
# plt.savefig('NUTS_mtd10_8192_unstandard_short.svg', bbox_inches='tight')
plt.show();


# In[8]:


from getdist import MCSamples

def get_gdsamples_mesh(samples:dict, fiduc:dict, n:int, stop:int=None, axis:int=0, label:str=None):
    mesh_samples = samples['init_mesh']
    mesh_fiduc = fiduc['init_mesh']

    subsamples = {}
    subfiduc = {}
    labels = []
    slices0 = len(mesh_fiduc.shape)*[slice(0,1)]
    name0 = len(mesh_fiduc.shape) * ["0"]
    if stop is None:
        stop = mesh_fiduc.shape[axis]
    for i in np.linspace(0, stop, n, endpoint=False, dtype=int):
        slices, name = slices0.copy(), name0.copy()
        slices[axis] = slice(i,i+1)
        # name[axis] = str(i)
        # name = "delta("+",".join(name)+")"
        # lab = "\\"+name
        name = f"delta({i})"
        lab = "\\"+name

        subsamples[name] = mesh_samples[:,*slices].squeeze()
        subfiduc[name] = mesh_fiduc[*slices].squeeze()
        labels.append(lab)
 
    gdsamples = MCSamples(samples=list(subsamples.values()), names=list(subsamples.keys()), labels=labels, label=label)
    return gdsamples, subfiduc

ntoplot = 8
stop = 8
gdsamplesX, subfiduc = get_gdsamples_mesh(post_samples, fiduc_params, ntoplot, stop, axis=0, label="NUTS, x")
gdsamplesY, subfiduc = get_gdsamples_mesh(post_samples, fiduc_params, ntoplot, stop, axis=1, label="NUTS, y")
gdsamplesZ, subfiduc = get_gdsamples_mesh(post_samples, fiduc_params, ntoplot, stop, axis=2, label="NUTS, z")

g.triangle_plot(roots=[gdsamplesX, gdsamplesY, gdsamplesZ] , 
                title_limit=1, 
                filled=True, 
                # markers=subfiduc,
                )
# plt.savefig('NUTS_mtd10_1560_meshtriangle.svg', dpi=200, bbox_inches='tight')
plt.show();


# ### Spectrum distribution

# In[9]:


from jaxpm.painting import cic_paint, cic_read, compensate_cic
pk_fiduc = pk_fn(fiduc_params['init_mesh'])
pk_post = vmap(pk_fn)(post_samples['init_mesh'])
qs = jnp.array([0.0015, 0.0250, 0.1600, 0.5, 0.840, 0.9750, 0.9985])
pk_0015, pk_0250, pk_1600, pk_5000, pk_8400, pk_9750, pk_9985 = jnp.quantile(pk_post, q=qs, axis=0)


# In[10]:


plot_fn = lambda pk, *args, **kwargs: plt.plot(pk[0], pk[0]*pk[1], *args, **kwargs)
plotfill_fn = lambda pklow, pkup, *args, **kwargs: plt.fill_between(pklow[0], pklow[0]*pklow[1], pklow[0]*pkup[1], *args, **kwargs)

plot_fn(pk_fiduc, 'k', label='fiduc')
plot_fn(pk_5000, 'r--', label='med')
plotfill_fn(pk_1600, pk_8400, alpha=0.15, color='red', label='68%')
plotfill_fn(pk_0250, pk_9750, alpha=0.10, color='red', label='95%')
plotfill_fn(pk_0015, pk_9985, alpha=0.05, color='red', label='99.7%')
plt.xlabel("$k$ [h/Mpc]"), plt.ylabel(f"$k P(k)$ [(Mpc/h)$^2$]")
plt.legend()
plt.tight_layout()
# plt.savefig('post_pk.svg', dpi=200, bbox_inches='tight')
plt.show()


# ### Mass matrix

# In[132]:


# Load mass matrix
import pickle
with open(save_path+f"_laststate16.p", 'rb') as file:
    last_state = pickle.load(file)
    
inverse_mass_matrix = last_state.adapt_state.inverse_mass_matrix
print(last_state.adapt_state.step_size, inverse_mass_matrix)
# np.cov(np.array([post_samples[var_name] for var_name in ['Omega_c_base', 'sigma8_base']]))


# In[143]:


# Plot inverse mass matrix vs. posterior sample variance
invM_arr = np.array(list(inverse_mass_matrix.values()))[0] # also jax.tree_util.tree_flatten(inverse_mass_matrix)[0][0]
var_names = list(inverse_mass_matrix.keys())[0]

post_variance_mesh, post_variance_cosmo = [], []
invM_mesh, invM_cosmo, invM_cosmo_name = [], [], []
invM_head = 0
for var_name in var_names:
    if var_name == 'bnl_':
        var_name = 'bn2_'
    if var_name == 'bs_':
        var_name = 'bs2_'
    var_variance = post_samples_[var_name].var(axis=0).flatten()
    new_invM_head = invM_head + len(var_variance)
    if var_name in ['init_mesh_']:
        post_variance_mesh = np.concatenate((post_variance_mesh, var_variance))
        invM_mesh = np.concatenate((invM_mesh, invM_arr[invM_head: new_invM_head]))
    else:
        post_variance_cosmo = np.concatenate((post_variance_cosmo, var_variance))
        invM_cosmo = np.concatenate((invM_cosmo, invM_arr[invM_head: new_invM_head]))
        invM_cosmo_name += [var_name]
    invM_head = new_invM_head


plt.figure(figsize=(14,6))
plt.subplot(1,5,(1,2))
x_pos = np.arange(len(invM_cosmo))
plt.bar(x_pos, invM_cosmo, width=.5, label="inverse mass")
plt.bar(x_pos+.4, post_variance_cosmo, width=.5, label="sample var")
plt.xticks(x_pos+.2, invM_cosmo_name)
plt.legend()

plt.subplot(1,5,(3,5))
# argsort_invM_mesh = np.argsort(invM_mesh) 
# plt.plot(invM_mesh[argsort_invM_mesh][::-1], label="inverse mass")
# plt.plot(post_variance_mesh[argsort_invM_mesh][::-1], label="sample var")
argsort_postvar_mesh = np.argsort(post_variance_mesh) 
plt.plot(invM_mesh[argsort_postvar_mesh][::-1], label="inverse mass")
plt.plot(post_variance_mesh[argsort_postvar_mesh][::-1], label="sample var")
plt.xlabel("init_mesh_")
plt.legend(), plt.tight_layout()
plt.savefig(save_path+"_invMvar.svg");


# In[ ]:


# # Save posterior variance as inverse mass matrix format
# post_variance = []
# for var_name in var_names:
#     if var_name == 'init_mesh_base':
#         var_name = 'init_mesh'
#         post_variance = np.concatenate((post_variance, np.ones(post_samples[var_name][0].flatten().shape)))
#     else:
#         post_variance = np.concatenate((post_variance, post_samples[var_name].var(axis=0).flatten()))
#         post_variance = np.concatenate((post_variance, ))
# variance_as_invM = {var_names: post_variance}
# print(variance_as_invM)

# with open(save_path+f"_invM.p", 'wb') as file:
#     pickle.dump(post_variance, file, protocol=pickle.HIGHEST_PROTOCOL)


# ### Init. cond. 

# In[ ]:


plt.figure(figsize=(15,4))
plt.subplot(131)
plt.imshow(post_samples['init_mesh'].mean(0).mean(0))
plt.title("sample mean"), plt.colorbar()
plt.subplot(132)
plt.imshow(post_samples['init_mesh'].std(0).mean(0))
plt.title("sample std"), plt.colorbar()
plt.subplot(133)
plt.imshow(fiducial_trace['init_mesh']['value'].mean(0))
plt.title("fiducial value"), plt.colorbar()
plt.show();

