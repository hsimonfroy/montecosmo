#!/usr/bin/env python
# coding: utf-8

# # Model Explicit Likelihood Inference
# Infer from a cosmological model via MCMC samplers. 

# In[1]:


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


# import mlflow
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8081")
# mlflow.set_experiment("ELI")
from montecosmo.utils import pickle_dump, pickle_load, get_vlim, theme_switch
from montecosmo.mcbench import sample_and_save
save_dir = os.path.expanduser("~/scratch/pickles/")


# In[2]:




# ## Import

# In[3]:


from montecosmo.models import pmrsd_model, prior_model, get_logp_fn, get_score_fn, get_simulator, get_pk_fn, get_param_fn
from montecosmo.models import print_config, get_prior_loc, default_config as config

# Build and render model
# config.update(a_lpt=0.5, mesh_shape=64*np.ones(3, dtype=int), box_size=256*np.ones(3))
config.update(a_lpt=0.1, mesh_shape=64*np.ones(3, dtype=int), fourier=True)
config['lik_config'].update(obs='pk')
# config.update(a_lpt=0.5, mesh_shape=64*np.ones(3, dtype=int), fourier=False)
model = partial(pmrsd_model, **config)
print_config(model)
expe_prefix = "fourier_pm_plk_"

# Get fiducial parameters
param_fn = get_param_fn(**config)
fiduc_model = condition(partial(model, trace_reparam=True), param_fn(inverse=True, **get_prior_loc(model)))
fiduc_trace = get_simulator(fiduc_model)(rng_seed=0)
fiduc_lat = param_fn(**fiduc_trace)
fiduc_lat_ = param_fn(inverse=True, **fiduc_lat)

# Chain init
@jit
@vmap
def sample_init_chains(rng_key, scale_std):
    params_ = seed(prior_model, rng_key)(**config)
    params = get_param_fn(scale_std=scale_std, **config)(**params_)
    return get_param_fn(**config)(inverse=True, **params)

# init_params_ = sample_init_chains(jr.split(jr.key(1), 7), jnp.array(7*[1/10]))
# init_params_ = tree_map(lambda x,y: jnp.concatenate((jnp.array(x)[None], y), axis=0), fiduc_params_, init_params_)
init_params_ = tree_map(lambda x: jnp.tile(x, (10,*len(jnp.shape(x))*[1])), fiduc_lat_)
pickle_dump(fiduc_trace, save_dir + expe_prefix + "fiduc_trace.p")
pickle_dump(init_params_, save_dir + expe_prefix + "init_params_.p")

# Load fiducial and chain init params
# fiduc_params = pickle_load(save_dir + expe_prefix + "fiduc_params.p")
# init_params_ = pickle_load(save_dir + expe_prefix + "init_params_.p")

# Condition model on observables
obs_names = ['obs']
# obs_names = ['obs','b1_','b2_','bs2_','bn2_']
# obs_names = ['obs','Omega_m_','sigma8_','b1_','b2_','bs2_','bn2_']
obs_params = {name: fiduc_trace[name] for name in obs_names}
obs_model = condition(model, obs_params)
logp_fn = get_logp_fn(obs_model)


# In[4]:


print(fiduc_trace.keys(), '\n', init_params_['Omega_m_'], '\n', init_params_['b1_'], '\n', init_params_['init_mesh_'][:,0,0,0])

# ### NUTS, HMC

# In[6]:


# num_samples, max_tree_depth, n_runs, num_chains = 256, 10, 20, 8
# num_samples, max_tree_depth, n_runs, num_chains = 128, 10, 10, 4
# num_samples, max_tree_depth, n_runs, num_chains = 128, 10, 5, 4
n_samples, max_tree_depth, n_runs, n_chains = 64, 9, 10, 10

# Variables to save
# save_path = save_dir + f"HMC_ns{num_samples:d}_x_nc{num_chains}_2"
# save_path = save_dir + f"NUTS_ns{num_samples:d}_x_nc{num_chains}_pm"
# save_path = save_dir + f"HMCGibbs_ns{num_samples:d}_test"
save_path = save_dir + expe_prefix + f"NUTS_nc{n_chains:d}_x_ns{n_samples:d}"

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
                                         gibbs_sites=['Omega_m_','sigma8_','b1_','b2_','bs2_','bn2_'])

# # Propose MALA step size based on [Chen+2019](http://arxiv.org/abs/1801.02309)
# L_smoothness, m_strong_convex = 1, 1 # log density regularity properties
# condition_number = L_smoothness / m_strong_convex
# print(f"MALA step size proposal={1 / (L_smoothness * (config['mesh_shape'].prod() * condition_number)**0.5):e}")

# from numpyro.contrib.tfp.mcmc import MetropolisAdjustedLangevinAlgorithm as MALA
# mala_kernel = MALA(model=obs_model,
#                     init_strategy=numpyro.infer.init_to_value(values=fiduc_params),
#                     step_size=0.001,)

mcmc = numpyro.infer.MCMC(
    sampler=nuts_kernel,
    # num_warmup=0,
    num_warmup=n_samples,
    num_samples=n_samples, # for each run
    num_chains=n_chains,
    chain_method="vectorized",
    progress_bar=True,)

# last_state = pickle_load(save_dir+"NUTS_nc8_x_ns64_test2_laststate.p")
# print("mean_acc_prob:", last_state.mean_accept_prob, "\nss:", last_state.adapt_state.step_size)
# mcmc.post_warmup_state = last_state
# invmm = list(last_state.adapt_state.inverse_mass_matrix.values())[0][0]
# invmm.min(),invmm.max(),invmm.mean(),invmm.std()


# In[7]:


# mlflow.end_run()
# mlflow.start_run(run_name="NUTS "+expe_prefix)
# mlflow.log_params(config)
# mlflow.log_params({'n_runs':n_runs, 'n_samples':n_samples, 'max_tree_depth':max_tree_depth, 'n_chains':n_chains})
print({'n_runs':n_runs, 'n_samples':n_samples, 'max_tree_depth':max_tree_depth, 'num_chains':n_chains})
print(save_path)


# In[8]:


# init_params_one_ = tree_map(lambda x: x[:num_chains], init_params_)
# mlflow.log_metric('halt',0) # 31.46s/it 4chains, 37.59s/it 8chains
mcmc_runned = sample_and_save(mcmc, n_runs, save_path, extra_fields=['num_steps'], init_params=init_params_)
# mcmc_runned = sample_and_save(mcmc, n_runs, save_path, extra_fields=['num_steps'])
# mlflow.log_metric('halt',1)

raise ValueError("Stop here")
# ## Analysis

# In[ ]:


def separate(samples, separ=['num_steps']):
    samples = samples.copy()
    samples2 = {name:samples.pop(name) for name in separ}
    return samples, samples2

def recombine(dic, block, block_name, rest=True, aggr_fn=jnp.stack):
    combined = {}
    if rest:
        dic = dic.copy()
        for b, bn in zip(block, block_name):
            combined[bn] = aggr_fn(jnp.stack([dic.pop(k) for k in b]))
        combined |= dic
    else:
        for b, bn in zip(block, block_name):
            combined[bn] = aggr_fn(jnp.stack([dic[k] for k in b]))
    return combined

def combine(moments, axis=0):
    moments, infos = separate(moments)
    moments = tree_map(lambda x: x.mean(axis=axis), moments)
    infos = tree_map(lambda x: x.sum(axis=axis), infos)
    return moments | infos

def get_moments(x_, axis=0):
    x_, infos = separate(x_)
    moments = tree_map(lambda x: jnp.stack([x, x**2], axis=2), vmap(vmap(param_fn))(**x_))
    return combine(moments | infos, axis=axis)

def choice_cells(x_, rng_key, n):
    x_, infos = separate(x_)
    x = vmap(vmap(param_fn))(**x_)
    name = 'init_mesh'
    init_mesh = x[name]
    x[name] = jr.choice(rng_key, init_mesh.reshape((*init_mesh.shape[:2],-1)), 
                         shape=(n,), replace=False, axis=2)
    return x | infos

## For Chains, ESS, GR
conc_axis = [1] # axis: run x chain x sample 
var_names = None
# var_names = [name+'_' for name in config['prior_config']] + ['num_steps']
# transform = lambda x:x
transform = lambda x: tree_map(lambda x_:x_[:,::16], x)
load_chains_ = partial(load_runs, var_names=var_names, conc_axis=conc_axis, transform=transform, verbose=True)

var_names = [name+'_' for name in config['prior_config']] + ['num_steps'] + ['init_mesh_']
transform = jit(partial(choice_cells, rng_key=jr.key(1), n=50))
load_chains = partial(load_runs, var_names=var_names, conc_axis=conc_axis, transform=transform, verbose=True)

## For Errors
conc_axis = []
var_names = [name+'_' for name in config['prior_config']] + ['num_steps'] + ['init_mesh_']
transform = jit(partial(get_moments, axis=(1)))
load_moments = partial(load_runs, var_names=var_names, conc_axis=conc_axis, transform=transform, verbose=True)


# In[ ]:


# load_path = save_dir + f"MCLMC/MCLMC_ns8192_x_nc8"
# load_path = save_dir + f"NUTSGibbs/HMCGibbs_ns256_x_nc8"
load_path = save_dir + f"NUTSGibbs_ns256_x_nc8"
# load_path = save_dir + f"NUTS/NUTS_ns256_x_nc8"
# load_path = save_dir + f"HMC/HMC_ns256_x_nc8"
start_run, end_run = 0,0

samples_ = load_chains_(load_path, start_run, end_run)
samples = load_chains(load_path, start_run, end_run)
# samples = jit(vmap(param_fn))(**samples_) | separate(samples_)[1]
moments = load_moments(load_path, start_run, end_run)
# TODO: modify to not do for loop but instead tree_map and first dim is sampler dim
# pickle_dump(combine(moments, axis=(0,1)), save_dir+"NUTS/NUTS_moments20.p")

# n_comb = 4
# for i in range(1,2):
#     start_run, end_run = 1+(i-1)*n_comb,i*n_comb
#     transform = lambda x: tree_map(lambda y : y[:,::n_comb], x)
#     var_names = None
#     samples_ = load_runs(load_path, start_run, end_run, var_names, conc_axis=[1], transform=transform, verbose=True)
#     pickle_dump(samples_, save_dir + f"HMC/HMC_ns1024_x_nc8_{i}.p")


# In[ ]:


samples_.keys()


# ### Chain

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
def _plot_chain(samples:dict, prior_config:dict, fiduc:dict, **config):
    labels = []
    for name in samples:
        if name.endswith('_'): # convention for a latent value 
            lab = "\\tilde"+prior_config[name[:-1]][0]
        else:
            lab = prior_config[name][0]
        labels.append('$'+lab+'$')

    samples_arr = np.array(list((samples.values()))).T

    plt.plot(samples_arr, label=labels)
    plt.hlines([fiduc[name] for name in samples], 
            xmin=0, xmax=len(samples_arr), 
            ls="--", alpha=0.75,
            color=[f"C{i}" for i in range(len(samples))],)

def plot_chain(samples_:dict, prior_config:dict, fiduc:dict, verbose=True, **config):
    # Print diagnostics
    samples = jit(vmap(param_fn))(**samples_)
    if verbose:
        from numpyro.diagnostics import print_summary
        print_summary(samples, group_by_chain=True) # NOTE: group_by_chain if several chains

    # Concatenate and reparam chains
    samples_ = tree_map(lambda x: jnp.concatenate(x, axis=0), samples_)
    samples = tree_map(lambda x: jnp.concatenate(x, axis=0), samples)

    # Plot chains
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

sli_toplot = slice(0,300)
# plot_chain(post_samples_[1], fiduc=fiduc_params, **config)
# plot_chain(samples_, fiduc=fiduc_params, **config)
plot_chain(tree_map(lambda x:x[:,sli_toplot], samples_), fiduc=fiduc_params, **config)
# mlflow.log_figure(plt.gcf(), f"MCLMC_chain_L25_ss2_invmm.svg")
plt.show();


# In[ ]:


from numpyro.diagnostics import effective_sample_size, gelman_rubin

def get_metric_traj(metric_fn, samples, n, axis=0, true=None):
    """
    """ # TODO: make such that no need to pass true, maybe by partial metrif_fn?
    samples, infos = separate(samples)
    if true is not None:
        true, true_infos = separate(true)

    def get_fn_traj(fn, samples, *args):
        metrics = []
        length = samples.shape[axis]
        for i_filt in np.arange(length, 1, -length// n)[::-1]:
            filt = jnp.arange(i_filt)
            metrics.append(fn(samples.take(filt, axis=axis), *args))
        return jnp.stack(metrics)
    
    get_infos_traj = partial(get_fn_traj, lambda x: x.sum())
    infos_traj = tree_map(get_infos_traj, infos)
    get_metric_traj = partial(get_fn_traj, metric_fn)
    if true is not None:
        metric_traj = tree_map(get_metric_traj, samples, true)
    else:
        metric_traj = tree_map(get_metric_traj, samples)
    return metric_traj | infos_traj


def geomean(x, axis=None):
    return jnp.exp( jnp.log(x).mean(axis=axis) )
multi_ess_fn = lambda x: geomean(effective_sample_size(x))
def grmean(x, axis=None):
    """cf. https://arxiv.org/pdf/1812.09384"""
    return (1 + geomean(x**2 - 1, axis=axis) )**.5
multi_gr_fn = lambda x: grmean(gelman_rubin(x))
def multi_gr_fn2(x):
    n_chains = x.shape[0]
    return geomean(gelman_rubin(x)**2 - 1) / n_chains # about 1/N_eff

sqrerr_moments_fn = lambda m, m_true: (m.mean(axis=(0,1))-m_true)**2
@jit
def sqrerr_locscale_fn(moments, moments_true):
    # Get mean and std from runs and chains
    m1_hat, m2_hat = moments.mean(axis=(0,1))
    m1, m2 = moments_true
    std_hat, std = (m2_hat - m1_hat**2)**.5, (m2 - m1**2)**.5 # Huygens formula
    # Compute normalized errors
    err_loc, err_scale = (m1_hat - m1) / std, (std_hat - std) / (std / 2**.5) # asymptotically N(0, 1/n_eff)
    mse_loc, mse_scale = (err_loc**2).mean(), (err_scale**2).mean() # asymptotically 1/n_eff * Chi^2(d)/d
    return jnp.stack([mse_loc, mse_scale])

@jit
def sqrerr_locscale_fn2(moments, moments_true):
    # Get mean and std from runs
    n_chains = moments.shape[1]
    m_hat = moments.mean(axis=(0))
    m1_hat, m2_hat = m_hat[:,0], m_hat[:,1]
    m1, m2 = moments_true
    std_hat, std = (m2_hat - m1_hat**2)**.5, (m2 - m1**2)**.5 # Huygens formula
    # Compute normalized errors
    err_loc, err_scale = (m1_hat - m1) / std, (std_hat - std) / (std / 2**.5) # asymptotically N(0, n_chain/n_eff)
    mse_loc, mse_scale = (err_loc**2).mean(), (err_scale**2).mean() # asymptotically n_chain/n_eff * Chi^2(d*n_chain)/(d*n_chain) 
    return jnp.stack([mse_loc, mse_scale]) / n_chains # asymptotically 1/n_eff * Chi^2(d*n_chain)/(d*n_chain) 


# In[ ]:


n_toplot = 100
# ESSs = get_metric_traj(effective_sample_size, samples, n_toplot, 1)
# GRs = get_metric_traj(gelman_rubin, samples, n_toplot, 1)
ESSs = get_metric_traj(multi_ess_fn, samples, n_toplot, 1)
# GRs = get_metric_traj(multi_gr_fn, samples, n_toplot, 1)
GRs = get_metric_traj(multi_gr_fn2, samples, n_toplot, 1)

moments_true = pickle_load(save_dir+"NUTS/NUTS_moments20.p")
# SEs = get_metric_traj(sqrerr_moments_fn, moments, n_toplot, 0, moments_true)
# NMSEs = get_metric_traj(sqrerr_locscale_fn, moments, n_toplot, 0, moments_true)
NMSEs = get_metric_traj(sqrerr_locscale_fn2, moments, n_toplot, 0, moments_true)


# In[ ]:


plt.figure(figsize=(12,4))
sli_plot = slice(20,None)

plt.subplot(121)
metric_traj, infos_traj = separate(ESSs)
num_steps = infos_traj['num_steps']
plot_fn = lambda x, **kwargs: plt.semilogy(num_steps[sli_plot], (num_steps / x.T).T[sli_plot], **kwargs)
for name, val in metric_traj.items():
    if name == 'init_mesh':
        label = "$\\delta_L$"
    else:
        label = '$'+config['prior_config'][name][0]+'$'
    plot_fn(val, label=label)
plt.xlabel("$N_{\\text{eval}}$"), plt.ylabel("$N_{\\text{eval}}\\;/\\;N_{\\text{eff}}$")
plt.legend()

plt.subplot(122)
metric_traj, infos_traj = separate(GRs)
num_steps = infos_traj['num_steps']
# plot_fn = lambda x, **kwargs: plt.plot(num_steps[sli_plot], x[sli_plot], **kwargs)
plot_fn = lambda x, **kwargs: plt.semilogy(num_steps[sli_plot], (num_steps * x)[sli_plot], **kwargs)
for name, val in metric_traj.items():
    if name == 'init_mesh':
        label = "$\\delta_L$"
    else:
        label = '$'+config['prior_config'][name][0]+'$'
    plot_fn(val, label=label)
# plt.xlabel("$N_{\\text{eval}}$"), plt.ylabel("$\\hat R$")
plt.xlabel("$N_{\\text{eval}}$"), plt.ylabel("$N_{\\text{eval}}\\;/\\;N_{\\text{chain}} \\times (\\hat R^2 - 1)$")
plt.legend()
plt.tight_layout()
plt.show();


# In[ ]:


plt.figure(figsize=(12,4))
sli_plot = slice(20,None)

plt.subplot(121)
metric_traj, infos_traj = separate(ESSs)
metric_traj = recombine(metric_traj, 
                        [['Omega_m','sigma8'],['b1','b2','bs2','bn2'],['init_mesh']], 
                        ['cosmo','biases','init'], 
                        aggr_fn=partial(geomean, axis=0))
num_steps = infos_traj['num_steps']
plot_fn = lambda x, **kwargs: plt.semilogy(num_steps[sli_plot], (num_steps / x.T).T[sli_plot], **kwargs)
for name, val in metric_traj.items():
    plot_fn(val, label=name)
plt.xlabel("$N_{\\text{eval}}$"), plt.ylabel("$N_{\\text{eval}}\\;/\\;N_{\\text{eff}}$")
plt.legend()

plt.subplot(122)
metric_traj, infos_traj = separate(GRs)
metric_traj = recombine(metric_traj, 
                        [['Omega_m','sigma8'],['b1','b2','bs2','bn2'],['init_mesh']], 
                        ['cosmo','biases','init'], 
                        # aggr_fn=partial(grmean, axis=0))
                        aggr_fn=partial(geomean, axis=0))
num_steps = infos_traj['num_steps']
# plot_fn = lambda x, **kwargs: plt.plot(num_steps[sli_plot], x[sli_plot], **kwargs)
plot_fn = lambda x, **kwargs: plt.semilogy(num_steps[sli_plot], (num_steps * x)[sli_plot], **kwargs)
for name, val in metric_traj.items():
    plot_fn(val, label=name)
# plt.xlabel("$N_{\\text{eval}}$"), plt.ylabel("$\\hat R$")
plt.xlabel("$N_{\\text{eval}}$"), plt.ylabel("$N_{\\text{eval}}\\;/\\;N_{\\text{chain}} \\times (\\hat R^2 - 1)$")
plt.legend()
plt.tight_layout()
plt.show();


# In[ ]:


plt.figure(figsize=(12,4))
sli_plot = slice(0,-1)
metric_traj, infos_traj = separate(NMSEs)
num_steps = infos_traj['num_steps']

for i_plot, name_stat in enumerate(['\\mu','\\sigma']) :
    plt.subplot(1, 2, i_plot+1)
    plot_fn = lambda x, **kwargs: plt.semilogy(num_steps[sli_plot], (num_steps * x.T[i_plot])[sli_plot], **kwargs)
    for name, val in metric_traj.items():
        if name == 'init_mesh':
            label = "$\\delta_L$"
        else:
            label = '$'+config['prior_config'][name][0]+'$'
        plot_fn(val, label=label)
    plt.xlabel("$N_{\\text{eval}}$"), plt.ylabel("$N_{\\text{eval}} \\times \\operatorname{NMSE}("+name_stat+")$")
    plt.legend()
plt.tight_layout()
plt.show();


# In[ ]:


plt.figure(figsize=(12,4))
sli_plot = slice(0,-1)
metric_traj = recombine(metric_traj, 
                        [['Omega_m','sigma8'],['b1','b2','bs2','bn2'],['init_mesh']], 
                        ['cosmo','biases','init'], 
                        aggr_fn=partial(jnp.mean, axis=0))
metric_traj, infos_traj = separate(NMSEs)
num_steps = infos_traj['num_steps']

for i_plot, name_stat in enumerate(['\\mu','\\sigma']) :

    plt.subplot(1, 2, i_plot+1)
    plot_fn = lambda x, **kwargs: plt.semilogy(num_steps[sli_plot], (num_steps * x.T[i_plot])[sli_plot], **kwargs)
    for name, val in metric_traj.items():
        plot_fn(val, label=name)
    plt.xlabel("$N_{\\text{eval}}$"), plt.ylabel("$N_{\\text{eval}} \\times \\operatorname{NMSE}("+name_stat+")$")
    plt.legend()
plt.tight_layout()
plt.show();


# ## Multiple runs analysis

# In[ ]:


paths = ["HMC/HMC_ns256_x_nc8","NUTS/NUTS_ns256_x_nc8","NUTSGibbs/HMCGibbs_ns256_x_nc8"]
load_paths = np.array([os.path.join(save_dir, path) for path in paths])
mc_labels = ["HMC","NUTS",'NUTSGibbs']
start_run, end_run = [2,1,2], [64,20,32]
# start_run, end_run = [1,1,1], [5,5,5]

moments = load_moments(load_paths, start_run, end_run)

# samples_ = load_chains_(load_paths, start_run, end_run)
samples = load_chains(load_paths, start_run, end_run)
# samples = [jit(vmap(param_fn))(**s_) | separate(s_)[1] for s_ in samples_]


# In[ ]:


from numpyro.diagnostics import print_summary
for lab, s in zip(mc_labels, samples):
    print(f"# {lab}")
    print_summary(separate(s)[0], group_by_chain=True) # NOTE: group_by_chain if several chains


# In[ ]:


n_toplot = 200
ESSs = [get_metric_traj(multi_ess_fn, s, n_toplot, 1) for s in samples]
GRs = [get_metric_traj(multi_gr_fn2, s, n_toplot, 1) for s in samples]
moments_true = pickle_load(save_dir+"NUTS/NUTS_moments20.p")
# NMSEs = [get_metric_traj(sqrerr_locscale_fn, m, n_toplot, 0, moments_true) for m in moments]
NMSEs = [get_metric_traj(sqrerr_locscale_fn2, m, n_toplot, 0, moments_true) for m in moments]


# In[ ]:


# sli_plot = slice(10,None)
# sli_plots = [slice(15,None), slice(10,68), slice(10,71)]
sli_plots = [slice(2*15,None), slice(2*10,2*68), slice(2*8,2*53)]
colors = [plt.get_cmap('Dark2')(i/7) for i in range(7)]
# colors = ['C'+str(i) for i in range(7)]
linestyles = ['-',':','--']
recomb_cbi = partial(recombine, 
					 block=[['Omega_m','sigma8'],['b1','b2','bs2','bn2'],['init_mesh']], 
					 block_name=['cosmo','biases','init']) 

plt.figure(figsize=(12,4))
plt.subplot(121)
trajs = []
for m in ESSs: # TODO: modify to not do for loop but instead tree_map and first dim is sampler dim
    traj = recomb_cbi(m, aggr_fn=partial(geomean, axis=0))
    trajs.append(traj)

theme_switch(usetex=True)
# plt.subplot(1, 2, 1)
for i_traj, traj in enumerate(trajs):
    metrics, infos = separate(traj)
    num_steps = infos['num_steps']
    sli_plot = sli_plots[i_traj]

    plot_fn = lambda x, **kwargs: plt.semilogy(num_steps[sli_plot], (num_steps / x.T).T[sli_plot], **kwargs)
    for i_val, (name, val) in enumerate(metrics.items()):
        plot_fn(val, label=name, color=colors[i_traj], linestyle=linestyles[i_val])

# plt.xlabel("$N_{\\text{eval}}$"), plt.ylabel("$N_{\\text{eval}}\\;/\\;N_{\\text{eff}}$")
plt.xlabel("$N_{\\textrm{eval}}$"), plt.ylabel("$N_{\\textrm{eval}}\\;/\\;N_{\\textrm{eff}}$")
from matplotlib.lines import Line2D; from matplotlib.patches import Patch
# handles, labels = plt.gca().get_legend_handles_labels()
handles = []
for i_traj in range(len(trajs)):
    handles.append(Patch(color=colors[i_traj], label=mc_labels[i_traj]))
for i_val, name in enumerate(metrics):
    handles.append(Line2D([], [], color='grey', linestyle=linestyles[i_val], label=name))
plt.legend(handles=handles, loc="lower left")




# plt.subplot(1, 2, 2)
# trajs = []
# for m in GRs:
#     traj = recomb_cbi(m, 
#                     # aggr_fn=partial(grmean, axis=0))
#                       aggr_fn=partial(geomean, axis=0))
#     trajs.append(traj)

# for i_traj, traj in enumerate(trajs):
#     metrics, infos = separate(traj)
#     num_steps = infos['num_steps']
#     sli_plot = sli_plots[i_traj]

#     # plot_fn = lambda x, **kwargs: plt.plot(num_steps[sli_plot], x[sli_plot], **kwargs)
#     plot_fn = lambda x, **kwargs: plt.semilogy(num_steps[sli_plot], (num_steps * x)[sli_plot], **kwargs)
#     for i_val, (name, val) in enumerate(metrics.items()):
#         plot_fn(val, label=name, color=colors[i_traj], linestyle=linestyles[i_val])

# # plt.xlabel("$N_{\\text{eval}}$"), plt.ylabel("$\\hat R$")
# plt.xlabel("$N_{\\text{eval}}$"), plt.ylabel("$N_{\\text{eval}}\\;/\\;N_{\\text{chain}} \\times (\\hat R^2 - 1)$")
# from matplotlib.lines import Line2D; from matplotlib.patches import Patch
# # handles, labels = plt.gca().get_legend_handles_labels()
# handles = []
# for i_traj in range(len(trajs)):
#     handles.append(Patch(color=colors[i_traj], label=mc_labels[i_traj]))
# for i_val, name in enumerate(metrics):
#     handles.append(Line2D([], [], color='grey', linestyle=linestyles[i_val], label=name))
# plt.legend(handles=handles)
# plt.tight_layout()
# plt.savefig('ess_traj.svg', dpi=200, bbox_inches='tight', transparent=True)
plt.show();


# In[ ]:


sli_plot = slice(0,None)
colors = [plt.get_cmap('Dark2')(i/7) for i in range(7)]
# colors = ['C'+str(i) for i in range(7)]
linestyles = ['-',':','--']

trajs = []
for m in NMSEs:
    traj = recomb_cbi(m, aggr_fn=partial(jnp.mean, axis=0))
    trajs.append(traj)

plt.figure(figsize=(12,4))
for i_plot, name_stat in enumerate(['\\mu','\\sigma']) :

    # markers = ['+','^']
    plt.subplot(1, 2, i_plot+1)
    markers = 2*[None]

    for i_traj, traj in enumerate(trajs):
        metrics, infos = separate(traj)
        num_steps = infos['num_steps']
        # if i_traj ==2: print(num_steps)

        plot_fn = lambda x, **kwargs: plt.semilogy(num_steps[sli_plot], (num_steps * x.T[i_plot])[sli_plot], **kwargs)
        for i_val, (name, val) in enumerate(metrics.items()):
            plot_fn(val, label=name, color=colors[i_traj], linestyle=linestyles[i_val], marker=markers[i_plot])

    plt.xlabel("$N_{\\text{eval}}$"), plt.ylabel("$N_{\\text{eval}} \\times \\operatorname{NMSE}("+name_stat+")$")
    from matplotlib.lines import Line2D; from matplotlib.patches import Patch
    # handles, labels = plt.gca().get_legend_handles_labels()
    handles = []
    for i_traj in range(len(trajs)):
        handles.append(Patch(color=colors[i_traj], label=mc_labels[i_traj]))
    for i_val, name in enumerate(metrics):
        handles.append(Line2D([], [], color='grey', linestyle=linestyles[i_val], label=name))
    plt.legend(handles=handles)
plt.tight_layout()
plt.show();


# In[ ]:


def get_first_after(dic, thres, verbose=False, mult=True):
	metrics, infos = separate(dic)
	num_steps = infos['num_steps']
	i_thres = (num_steps > thres).argmax(axis=0)
	ns = num_steps[i_thres]
	if verbose:
		print(f"relerr: {(ns - thres)/thres:.0e}")
	if mult:
		return tree_map(lambda x: ns * x[i_thres], metrics)
	else:
		return tree_map(lambda x: ns / x[i_thres], metrics)


# thres = 2.8*1e7
thres = 2.5*1e7
ESS1 = [recomb_cbi(get_first_after(m, thres, True, False), aggr_fn=partial(geomean, axis=0)) for m in ESSs]
NMSE1 = [recomb_cbi(get_first_after(m, thres, True), aggr_fn=partial(jnp.mean, axis=0)) for m in NMSEs]
metric1 = [tree_map(lambda x,y: jnp.concatenate((x[None], y)), m1, m2) for m1, m2 in zip(ESS1, NMSE1)]


# In[ ]:


plt.figure(figsize=(5,4))
colors = [plt.get_cmap('Dark2')(i/7) for i in range(len(mc_labels))]
markers = ['o','s','D']
ls = ""
alpha = 1
# ms = 10
# mec = None
ms = 12
mec = 'w'
from matplotlib.transforms import ScaledTranslation
offset = lambda mm: ScaledTranslation(mm/25.4,0, plt.gcf().dpi_scale_trans)
trans = plt.gca().transData



usetex = True
theme_switch(usetex=usetex)
for i_mc, (label, mets) in enumerate(zip(mc_labels, metric1)):
    for i_met in range(3):
        # block_names = ['cosmo','biases','init']
        block_names = ['biases','cosmo','init']
        # block_names_lat = ["$\\textrm{biases}$",'$\\textrm{cosmo}$','$\\textrm{init}$']
        met = [mets[k][i_met] for k in block_names]
        if i_met == 0:
            ls = ""
        else:
            ls = ""
        xshifts = 5*np.array([-1,0,1])
        # xshifts = *np.array([0,0,0])
        plt.semilogy(block_names, met, color=colors[i_mc], marker=markers[i_met], 
                     linestyle=ls, alpha=alpha, markersize=ms, markeredgecolor=mec, transform=trans+offset(xshifts[i_met]))

plt.xlim(-.3,2.3), plt.ylim((917, 190842))
from matplotlib.lines import Line2D; from matplotlib.patches import Patch
# # handles, labels = plt.gca().get_legend_handles_labels()
handles = []
for i_traj in range(len(mc_labels)):
    # mc_labels_lat = ["$\\textrm{NUTS}$","$\\textrm{HMC}$","$\\textrm{HMCGibbs}$"]
    handles.append(Patch(color=colors[i_traj], label=mc_labels[i_traj]))
if not usetex:
    metric_names = ["$N_{\\text{eval}}\\;/\\;N_{\\text{eff}}$",
                    "$N_{\\text{eval}} \\times \\operatorname{NMSE}(\\mu)$",
                    "$N_{\\text{eval}} \\times \\operatorname{NMSE}(\\sigma)$"]
else:
    metric_names = ["$N_{\\textrm{eval}}\\;/\\;N_{\\textrm{eff}}$",
                    "$N_{\\textrm{eval}} \\times \\textrm{NMSE}(\\mu)$",
                    "$N_{\\textrm{eval}} \\times \\textrm{NMSE}(\\sigma)$"]
for i_met, name in enumerate(metric_names):
    handles.append(Line2D([], [], color='grey', marker=markers[i_met], linestyle=ls, label=name, alpha=alpha, markersize=ms, markeredgecolor=mec))

plt.legend(handles=handles, loc="upper right")
plt.tight_layout()
plt.savefig('benchmark.svg', dpi=200, bbox_inches='tight', transparent=True)
plt.show();


# In[ ]:


# sli_plot = slice(10,None)
# sli_plots = [slice(15,None), slice(10,68), slice(10,71)]
sli_plots = [slice(15,None), slice(10,68), slice(8,53)]
colors = [plt.get_cmap('Dark2')(i/7) for i in range(7)]
# colors = ['C'+str(i) for i in range(7)]
linestyles = ['-',':','--']
recomb_cbi = partial(recombine, 
					 block=[['Omega_m','sigma8'],['b1','b2','bs2','bn2'],['init_mesh']], 
					 block_name=['cosmo','biases','init']) 

plt.figure(figsize=(10,4))

plt.subplot(121)



trajs = []
for m in ESSs: # TODO: modify to not do for loop but instead tree_map and first dim is sampler dim
    traj = recomb_cbi(m, aggr_fn=partial(geomean, axis=0))
    trajs.append(traj)


# plt.subplot(1, 2, 1)
for i_traj, traj in enumerate(trajs):
    metrics, infos = separate(traj)
    num_steps = infos['num_steps']
    sli_plot = sli_plots[i_traj]

    plot_fn = lambda x, **kwargs: plt.semilogy(num_steps[sli_plot], (num_steps / x.T).T[sli_plot], **kwargs)
    for i_val, (name, val) in enumerate(metrics.items()):
        plot_fn(val, label=name, color=colors[i_traj], linestyle=linestyles[i_val])

# plt.xlabel("$N_{\\text{eval}}$"), plt.ylabel("$N_{\\text{eval}}\\;/\\;N_{\\text{eff}}$")
plt.xlabel("$N_{\\textrm{eval}}$"), plt.ylabel("$N_{\\textrm{eval}}\\;/\\;N_{\\textrm{eff}}$")
from matplotlib.lines import Line2D; from matplotlib.patches import Patch
# handles, labels = plt.gca().get_legend_handles_labels()
handles = []
for i_traj in range(len(trajs)):
    handles.append(Patch(color=colors[i_traj], label=mc_labels[i_traj]))
for i_val, name in enumerate(metrics):
    handles.append(Line2D([], [], color='grey', linestyle=linestyles[i_val], label=name))
plt.legend(handles=handles, loc="lower left")
plt.ylim((917, 190842))



plt.subplot(122)
colors = [plt.get_cmap('Dark2')(i/7) for i in range(len(mc_labels))]
markers = ['o','s','D']
ls = ""
alpha = 1
# ms = 10
# mec = None
ms = 12
mec = 'w'
from matplotlib.transforms import ScaledTranslation
offset = lambda mm: ScaledTranslation(mm/25.4,0, plt.gcf().dpi_scale_trans)
trans = plt.gca().transData



usetex = True
theme_switch(usetex=usetex)
for i_mc, (label, mets) in enumerate(zip(mc_labels, metric1)):
    for i_met in range(3):
        # block_names = ['cosmo','biases','init']
        block_names = ['biases','cosmo','init']
        # block_names_lat = ["$\\textrm{biases}$",'$\\textrm{cosmo}$','$\\textrm{init}$']
        met = [mets[k][i_met] for k in block_names]
        if i_met == 0:
            ls = ""
        else:
            ls = ""
        xshifts = 5*np.array([-1,0,1])
        # xshifts = *np.array([0,0,0])
        plt.semilogy(block_names, met, color=colors[i_mc], marker=markers[i_met], 
                     linestyle=ls, alpha=alpha, markersize=ms, markeredgecolor=mec, transform=trans+offset(xshifts[i_met]))

plt.xlim(-.3,2.3), plt.ylim((917, 190842))
from matplotlib.lines import Line2D; from matplotlib.patches import Patch
# # handles, labels = plt.gca().get_legend_handles_labels()
handles = []
for i_traj in range(len(mc_labels)):
    # mc_labels_lat = ["$\\textrm{NUTS}$","$\\textrm{HMC}$","$\\textrm{HMCGibbs}$"]
    handles.append(Patch(color=colors[i_traj], label=mc_labels[i_traj]))
if not usetex:
    metric_names = ["$N_{\\text{eval}}\\;/\\;N_{\\text{eff}}$",
                    "$N_{\\text{eval}} \\times \\operatorname{NMSE}(\\mu)$",
                    "$N_{\\text{eval}} \\times \\operatorname{NMSE}(\\sigma)$"]
else:
    metric_names = ["$N_{\\textrm{eval}}\\;/\\;N_{\\textrm{eff}}$",
                    "$N_{\\textrm{eval}} \\times \\textrm{NMSE}(\\mu)$",
                    "$N_{\\textrm{eval}} \\times \\textrm{NMSE}(\\sigma)$"]
for i_met, name in enumerate(metric_names):
    handles.append(Line2D([], [], color='grey', marker=markers[i_met], linestyle=ls, label=name, alpha=alpha, markersize=ms, markeredgecolor=mec))

plt.legend(handles=handles, loc="upper right")
plt.tight_layout()
plt.savefig('ess_bench.svg', dpi=200, bbox_inches='tight', transparent=True)
plt.show();


# ### Highest Density Regions

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from montecosmo.utils import get_gdsamples, get_gdprior

conc_samples = [tree_map(lambda x: jnp.concatenate(x, axis=0), separate(s, ['num_steps','init_mesh'])[0]) for s in samples]
gdsamples = get_gdsamples(conc_samples, label=mc_labels, verbose=True, **config)
for i_gds, gds in enumerate(gdsamples): 
    gdsamples[i_gds] = gds.copy(label=mc_labels[i_gds]+", 1$\\sigma$-smooth", settings={'smooth_scale_2D':1,'smooth_scale_1D':1,})
# gdsamples.append(get_gdprior(post_samples, verbose=True, **config))

g = plots.get_subplot_plotter(width_inch=9)
# g.settings.solid_colors='tab10_r'
g.triangle_plot(roots=gdsamples, 
                title_limit=1, 
                filled=True, 
                # param_limits={n:[m-2*s,m+2*s] for n,m,s in zip(names, mean, std)},
                markers=fiduc_params,
                )
# mlflow.log_figure(plt.gcf(), f"NUTS_contour_mtd3-8-10-12.svg", save_kwargs={'bbox_inches':'tight'}) # NOTE: tight bbox better
# plt.savefig('NUTS_mtd10_8192_unstandard_short.svg', bbox_inches='tight')
plt.show();


# In[ ]:


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

# In[ ]:


from jaxpm.painting import cic_paint, cic_read, compensate_cic
pk_fiduc = pk_fn(fiduc_params['init_mesh'])
pk_post = vmap(pk_fn)(post_samples['init_mesh'])
qs = jnp.array([0.0015, 0.0250, 0.1600, 0.5, 0.840, 0.9750, 0.9985])
pk_0015, pk_0250, pk_1600, pk_5000, pk_8400, pk_9750, pk_9985 = jnp.quantile(pk_post, q=qs, axis=0)


# In[ ]:


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

# In[ ]:


# Load mass matrix
import pickle
with open(save_path+f"_laststate16.p", 'rb') as file:
    last_state = pickle.load(file)
    
inverse_mass_matrix = last_state.adapt_state.inverse_mass_matrix
print(last_state.adapt_state.step_size, inverse_mass_matrix)
# np.cov(np.array([post_samples[var_name] for var_name in ['Omega_c_base', 'sigma8_base']]))


# In[ ]:


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

