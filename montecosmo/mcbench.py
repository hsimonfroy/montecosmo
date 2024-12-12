
import os
from functools import wraps, partial
from itertools import product, cycle
from typing import Iterable, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba_array
from jax import numpy as jnp, random as jr, jit, vmap, grad
from jax.tree_util import tree_map

from numpyro.infer import MCMC
from numpyro import diagnostics
from getdist import MCSamples
# from getdist.gaussian_mixtures import GaussianND

from montecosmo.utils import pdump, pload
from montecosmo.metrics import hdi, qbi, multi_ess, multi_gr









from dataclasses import dataclass, asdict
from collections import UserDict
from jax import tree, tree_util


@tree_util.register_pytree_node_class
# @partial(tree_util.register_dataclass, data_fields=['data'], meta_fields=['groups']) # JAX >=0.4.27
@dataclass
class Samples(UserDict):
    # TODO: is UserDict slower than dict?
    """
    Global slicing and indexing s[1:3,2]
    Querying with groups s['abc', 'c', 'd'], s[['abc','c'],['d']]
    """
    data: dict
    groups: dict = None # dict of list of key

    def __post_init__(self):
        if isinstance(self.data, Samples):
            samples = self.data
            self.data = samples.data # avoid nested Samples
            if self.groups is None:
                self.groups = samples.groups # inherit groups
        if self.groups is None:
            self.groups = {}

    def __getitem__(self, key, default_fn=None):
        # Global indexing and slicing
        if self._istreeof(key, (int, slice, type(Ellipsis))):
            return tree.map(lambda x: x[key], self)

        # Querying with groups
        elif self._istreeof(key, str):
            if isinstance(key, str):
                if key in self.groups:
                    group = self.groups[key]
                    if len(group) == 1: # handle length 1 group
                        return self._get(group[0], default_fn)
                    else:
                        return tuple(self._get(k, default_fn) for k in group)
                else:
                    return self._get(key, default_fn)
                
            elif isinstance(key, list): # construct new instance
                data = {'data': {k:self._get(k, default_fn) for k in self._parse_key(key)}}
                return type(self)(**asdict(self) | data) 
            
            elif isinstance(key, tuple):
                return tuple(self.__getitem__(k, default_fn) for k in self._parse_key(key))
    
    def _parse_key(self, key):
        newkey = ()
        for k in key:
            if isinstance(k, str) and k.startswith('~'): # everything but
                k = k[1:]
                if k in self.groups:
                    newkey += tuple(self.data.keys() - set(self.groups[k]))
                else:
                    newkey += tuple(self.data.keys() - {k})
            else:
                if isinstance(k, str) and k in self.groups:
                    newkey += tuple(self.groups[k])
                else:
                    newkey += (k,) # handle str and list
        return newkey
    
    def _istreeof(self, obj, type):
        return tree.all(tree.map(lambda x: isinstance(x, type), obj))
    
    def _get(self, key, default_fn=None):
        """
        Rewrite dict get method to raise KeyError by default.
        """
        if key in self.data:
            return self.data[key]
        elif default_fn is not None:
            return default_fn(key)
        else:
            raise KeyError(key)

    def get(self, key, default=None, default_fn=None):
        return self.__getitem__(key, default_fn=(lambda k: default) if default_fn is None else default_fn)

    @property
    def shape(self):
        return tree.map(jnp.shape, self.data)
    
    @property
    def ndim(self):
        return tree.map(jnp.ndim, self.data)
    
    # NOTE: no need with register_dataclass JAX >=0.4.27
    def tree_flatten(self):
        return (self.data,), (self.groups,)
    
    @classmethod
    # NOTE: no need with register_dataclass JAX >=0.4.27
    def tree_unflatten(cls, aux, data):
        return cls(*data, *aux)
    
    def __or__(self, other):
        if isinstance(other, UserDict):
            data = {'data': self.data | other.data}
            return type(self)(**asdict(self) | data)
        if isinstance(other, dict):
            data = {'data': self.data | other}
            return type(self)(**asdict(self) | data)
        return NotImplemented

    def __ior__(self, other):
        new = super().__ior__(other)
        if isinstance(other, Samples):
            new.groups |= other.groups
        return self





@tree_util.register_pytree_node_class
@dataclass
class Chains(Samples):
    labels: dict = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.data, Chains):
            chains = self.data
            if self.labels is None:
                self.labels = chains.labels
        if self.labels is None:
            self.labels = {}

    # NOTE: no need with register_dataclass JAX >=0.4.27
    def tree_flatten(self):
        return (self.data,), (self.groups, self.labels)
    
    def __or__(self, other):
        new = super().__or__(other)
        if isinstance(other, Chains):
            print("ho")
            new.labels |= other.labels
        return new

    def __ior__(self, other):
        new = super().__ior__(other)
        if isinstance(other, Chains):
            new.labels |= other.labels
        return self

    def to_getdist(self, label=None):
        samples, names, labels = [], [], []
        for k, v in self.data.items():
            samples.append(v.reshape(-1))
            names.append(k)
            labels.append(self.labels.get(k, None))
        return MCSamples(samples=samples, names=names, labels=labels, label=label)
    

    def print_summary(self, group_by_chain=True):
        diagnostics.print_summary(self.data, group_by_chain=group_by_chain)

    @classmethod
    def load_runs(cls, path, start_run, end_run, transforms=None, groups=None, labels=None, batch_ndim=2):
        """
        Load and append runs (or extra fields) saved in different files with same name except index.

        Both runs `start_run` and `end_run` are included.
        Runs are concatenated along last batch dimension.
        """
        print(f"Loading: {os.path.basename(path)}, from run {start_run} to run {end_run} (included)")
        if transforms is None:
            transforms = []
        transforms = np.atleast_1d(transforms)

        # @jit
        def transform(samples):
            for trans in transforms:
                samples = trans(samples, batch_ndim=batch_ndim)
            return samples

        for i_run in range(start_run, end_run+1):
            # Load
            part = cls(pload(path+f"_{i_run}.p"), groups=groups, labels=labels)
            part = transform(part)

            # Init or append samples
            if batch_ndim == 0:
                part = tree.map(lambda x: x[None], part)

            if i_run == start_run:
                samples = part
            else:
                samples = tree.map(lambda x,y: jnp.concatenate((x, y), axis=max(batch_ndim-1, 0)), samples, part)
                del part  

        return samples







###########
# Loading #
###########
def _load_runs(load_path:str, start_run:int, end_run:int, 
               var_names:str|Iterable[str]=None, conc_axis:int|Iterable[int]=0, 
               transform:Callable|Iterable[Callable]=[], verbose=False):
    if verbose:
        print(f"loading: {os.path.basename(load_path)}, from run {start_run} to run {end_run} (included)")
    var_names = np.atleast_1d(var_names)
    transform = np.atleast_1d(transform)

    for i_run in range(start_run, end_run+1):
        # Load
        samples_part = pload(load_path+f"_{i_run}.p")   
        if None in var_names: # NOTE: var_names should not be a consumable iterator
            var_names = list(samples_part.keys())
        samples_part = {key: samples_part[key] for key in var_names}
        for trans in transform:
            samples_part = trans(samples_part)

        # Init or append samples
        if i_run == start_run:
            samples = tree_map(lambda x: x[None], samples_part)
        else:
            # samples = {key: jnp.concatenate((samples[key], samples_part[key])) for key in var_names}
            samples = tree_map(lambda x,y: jnp.concatenate((x, y[None]), axis=0), samples, samples_part)
            del samples_part  
        
    for axis in jnp.atleast_1d(conc_axis):
        samples = tree_map(lambda x: jnp.concatenate(x, axis=axis), samples)
            
    if verbose:
        if 'n_evals' in samples:
            n_samples, n_evals = samples['n_evals'].shape, samples['n_evals'].sum(axis=-1)
            print(f"total n_samples: {n_samples}, total n_evals: {n_evals}")
        else:
            print(f"first variable length: {len(samples[list(samples.keys())[0]])}")
        print("")
    return samples


def load_runs(load_path:str|Iterable[str], start_run:int|Iterable[int], end_run:int|Iterable[int], 
              var_names:str|Iterable[str]=None, conc_axis:int|Iterable[int]=0, 
              transform:Callable|Iterable[Callable]=[], verbose=False):
    """
    Load and append runs (or extra fields) saved in different files with same name except index.

    Both runs `start_run` and `end_run` are included.
    If `var_names` is None, load all the variables.
    """
    paths = np.atleast_1d(load_path)
    starts = np.atleast_1d(start_run)
    ends = np.atleast_1d(end_run)
    assert len(paths)==len(starts)==len(ends), "lists must have the same lengths."
    samples = []

    for path, start, end in zip(paths, starts, ends):
        samples.append(_load_runs(path, start, end, var_names, conc_axis, transform, verbose))

    if isinstance(load_path, str):
        return samples[0]
    else:
        return samples 
    


##############################
# Generic dict manipulations #
##############################
def separate(dic, blocks, rest=True):
    """
    Separate by block a dict into a list of dict.
    """
    parts = []
    dic = dic.copy()
    for b in blocks:
        parts.append({k:dic.pop(k) for k in b})
    if rest:
        parts.append(dic)
    return parts


def recombine(dic, blocks_dic, rest=True):
    """
    Recombine by block a dict via stacking.
    """
    combined = {}
    dic = dic.copy()
    for name, b in blocks_dic.items():
        if len(b) >= 1:
            parts = np.stack([dic.pop(k) for k in b], axis=-1)
            if len(b)==1:
                parts = np.squeeze(parts, axis=-1)
            combined[name] = parts
    if rest:
        combined |= dic
    return combined


def flatten_dic(dic, axis=0):
    """
    Flatten a dict starting at a specified axis.
    """
    flat_dic = {}
    for k, v in dic.items():
        shape = jnp.shape(v)[axis:]
        if len(shape) == 0:
            flat_dic[k] = v
        else:
            for ids in product(*map(range, shape)):
                sufx = "[{}]".format(",".join(map(str, ids)))
                slices = axis * [slice(None)]
                for id in ids:
                    slices.append(id)
                flat_dic[k + sufx] = v[tuple(slices)]
    return flat_dic


def thin(dic, thinning=1, moments=None, axis=0):
    """
    Thin a dict by splitting it and taking last value or moments of each split.
    """
    if moments is None:
        aggr_fn = lambda y: jnp.take(y, -1, axis%y.ndim)
    else:
        moments = jnp.atleast_1d(moments)
        _aggr_fn = lambda y: jnp.moveaxis(jnp.sum(y[...,None]**moments, axis%y.ndim), -1, axis%y.ndim)
        if len(moments) == 1:
            aggr_fn = lambda y: _aggr_fn(y).squeeze(axis%y.ndim) # squeeze raise error if axis is not 1-length
        else:
            aggr_fn = _aggr_fn

    dic = tree_map(lambda x: jnp.atleast_1d(x), dic)
    out = tree_map(lambda x:
                      jnp.stack(tree_map(aggr_fn, 
                            jnp.array_split(x, max(jnp.rint(x.shape[axis]/thinning), 1), axis%x.ndim)), 
                        axis%x.ndim), dic)
    return out


def choice_cell(rng_key, a, n, axis):
    """
    Chose n random coordinates from last axis, obtained by flatenning given axis. 
    Ensure reproducibilty independently of ungiven axes.
    """
    # Move given axes at the end
    axis = jnp.atleast_1d(axis)
    dest = -1-jnp.arange(len(axis)) 
    a = jnp.moveaxis(a, axis, dest)

    # Remove axes to flatten from shape
    shape = list(a.shape)
    for ax in axis:
        shape.pop(ax) 

    return jr.choice(rng_key, a.reshape((*shape,-1)), shape=(n,), replace=False, axis=-1)


##############################
# Specific dict manipulations #
##############################
def name_latent(samples, names):
    new_names = []
    for name in names:
        if name in samples:
            new_names.append(name)
        elif name + '_' in samples:    
            new_names.append(name + '_')
    return new_names

def separate_latent(samples, blocks, rest=True):
    blocks = blocks.copy()
    for i, b in enumerate(blocks):
        blocks[i] = name_latent(samples, b)
    return separate(samples, blocks, rest)

def recombine_latent(samples, blocks_dic, rest=True):
    blocks_dic = blocks_dic.copy()
    for k, b in blocks_dic.items():
        blocks_dic[k] = name_latent(samples, b)
    return recombine(samples, blocks_dic, rest)   

def label_latent(name, prior_config, **config):
    if name.endswith('_'): # convention for a standardized latent value 
        return "\\tilde"+prior_config[name[:-1]][0]
    else:
        return prior_config[name][0]
    
def fn_traj(fn, n, values, *args, axis=0):
    filt_ends = jnp.rint(jnp.arange(1,n+1)/n*values.shape[1]).astype(int)
    filt_fn = lambda end: fn(jnp.moveaxis(jnp.moveaxis(values, axis, 0)[:end], 0, axis), *args)
    metrics = []
    for end in filt_ends:
        metrics.append(filt_fn(end))
    return jnp.stack(metrics)






class MCBench:
    def __init__(self, model_config, bench_config, fiduc_trace=None):
        
        self.prior_config = model_config['prior_config']
        self.blocks_config = bench_config['blocks_config']
        self.n_cell = bench_config['n_cell']
        self.thinning = bench_config['thinning']
        self.rng_key = bench_config['rng_key']
        self.multipoles = bench_config['multipoles']

        self.separate_latent = partial(separate_latent, blocks=[list(self.prior_config.keys())])
        self.recombine_latent = partial(recombine_latent, blocks_dic=self.blocks_config)

        self.pk_fn = get_pk_fn(multipoles=self.multipoles, **model_config)
        self.param_fn = get_param_fn(**model_config)
        @jit
        def param_chains(samples):
            latent, rest = self.separate_latent(samples, rest=True)
            latent = vmap(vmap(self.param_fn))(**latent)
            return latent | rest

        def thin_chains(samples, thinning=1, moments=None, axis=1):
            # NOTE: Typically moments=[0,1,2]
            latent, rest = self.separate_latent(samples, rest=True)
            latent = thin(latent, thinning, moments, axis)

            # Handle infos
            infos = ['n_evals']
            for k in infos:
                if k in rest:
                    rest[k] = thin(rest[k], thinning, 1, axis) # sum n_evals
            return latent | rest

        def choice_cell_chains(rng_key, samples, n, axis=[-3,-2,-1]):
            if n is None:
                return samples
            blocks = [['init_mesh']]
            latent, rest = self.separate_latent(samples, blocks=blocks, rest=True)
            if n == 0:
                return rest
            else:
                for k in latent:
                    latent[k] = choice_cell(rng_key, latent[k], n, axis=axis)
                return latent | rest

        self.param_chains = param_chains
        self.thin_fn = partial(thin_chains, thinning=self.thinning)
        self.moments_fn = partial(thin_chains, thinning=self.thinning, moments=[0,1,2])
        self.choice_fn = partial(choice_cell_chains, self.rng_key, n=self.n_cell)

        conc_axis = [1] # axis: n_chain x n_sample x n_dim
        # var_names = None
        var_names = [name+'_' for name in self.prior_config] + ['n_evals']
        # transform = []
        transform = [self.thin_fn, self.choice_fn]
        self.load_chains_ = partial(load_runs, var_names=var_names, conc_axis=conc_axis, transform=transform, verbose=True)
        transform = [self.thin_fn, self.param_chains, self.choice_fn]
        self.load_chains = partial(load_runs, var_names=var_names, conc_axis=conc_axis, transform=transform, verbose=True)
        transform = [self.param_chains, self.choice_fn, self.moments_fn]
        self.load_moments = partial(load_runs, var_names=var_names, conc_axis=conc_axis, transform=transform, verbose=True)

        if fiduc_trace is not None:
            self.set_fiduc(fiduc_trace)

    def set_fiduc(self, fiduc_trace):
        self.fiduc_trace = fiduc_trace
        self.fiduc = self.param_fn(**self.fiduc_trace)
        self.fiduc_ = self.param_fn(inverse=True, **self.fiduc)

        if self.n_cell is None:
            self.fiduc_pk = self.pk_fn(self.fiduc['init_mesh'])
        else:
            self.fiduc = self.choice_fn(self.fiduc)
            self.fiduc_ = self.choice_fn(self.fiduc_)
            if self.n_cell <= 2**10:
                self.ffiduc = flatten_dic(self.fiduc)
                self.ffiduc_ = flatten_dic(self.fiduc_)
            else:
                print("dict quite large to flatten, try reducing n_cell")

    def label_chains(self, samples, axis=2, dollars=True):
        latent, = self.separate_latent(samples, rest=False)
        if dollars:
            presuf = "$"
        else:
            presuf = ""
        dic = {}
        for name in latent:
            shape = np.shape(latent[name])[axis:]
            label = label_latent(name, self.prior_config)
            if len(shape) == 0:
                labels = presuf + label + presuf
            else:
                labels = np.full(shape, "", dtype='<U32')
                for ids in product(*map(range, shape)):
                    sufx = "[{}]".format(",".join(map(str, ids)))
                    labels[ids] = presuf + label + sufx + presuf
            dic[name] = labels
        return dic


    @staticmethod
    def _get_gdsamples(samples:dict, labels:dict, label:str=None, verbose:bool=False):
        samples_conc = tree_map(lambda x: jnp.concatenate(x, 0), samples) # concatenate all chains
        samples_conc = flatten_dic(samples_conc, axis=1)
        labels = flatten_dic(labels, axis=0)
        values, names, labs = [], [], []
        for k in samples_conc:
            values.append(samples_conc[k])
            names.append(k)
            labs.append(labels[k])
        gdsamples = MCSamples(samples=values, names=names, labels=labs, label=label)

        if verbose:
            if label is not None:
                print('# '+gdsamples.getLabel())
            else:
                print("# <unspecified label>")
            print(gdsamples.getNumSampleSummaryText())
            print_summary(samples, group_by_chain=True) # NOTE: group_by_chain if several chains

        return gdsamples
    
    def get_gdsamples(self, samples:dict|Iterable[dict], label:str|Iterable[str]=None, verbose:bool=False):
        """
        Construct getdist MCSamples from samples. 
        """
        samps = np.atleast_1d(samples)
        label = np.atleast_1d(label)
        assert len(samps)==len(label), "lists must have the same lengths."
        gdsamples = []

        for samp, lab in zip(samps, label):
            latent, = self.separate_latent(samp, rest=False)
            labels = self.label_chains(latent, dollars=False)
            gdsamples.append(self._get_gdsamples(latent, labels, lab, verbose))

        if isinstance(samples, dict):
            return gdsamples[0]
        else:
            return gdsamples 



    @staticmethod
    def _plot_chains(values, fiduc, labels, cmap='tab10'):
        values = jnp.concatenate(values, axis=0) # concatenate all chains
        # In case values, fiduc, and labels have not been flattened already
        # max_lines = 10
        # values = values.reshape(len(values), -1)[:,:max_lines]
        # fiduc = fiduc.reshape(-1)[:max_lines]
        # labels = labels.reshape(-1)[:max_lines]

        cmap = plt.get_cmap(cmap)
        colors = [cmap(i) for i in range(values.shape[-1])]
        plt.gca().set_prop_cycle(color=colors)

        plt.plot(values, label=np.squeeze(labels))
        plt.hlines(fiduc, xmin=0, xmax=values.shape[0], 
            ls="--", alpha=0.75, color=colors,)

    def plot_chains(self, samples, fiduc, cmap='tab10'):
        labels = self.label_chains(samples)
        samples = self.recombine_latent(samples, rest=False)
        fiduc = self.recombine_latent(fiduc, rest=False)
        labels = self.recombine_latent(labels, rest=False)
        
        n_plot = len(samples)
        for i_k, k in enumerate(samples):
            plt.subplot(1, n_plot, i_k+1)
            plt.title(k)
            self._plot_chains(samples[k], fiduc[k], labels[k], cmap)
            plt.legend()







    def interval_pk(self, meshes, proba=.95):
        proba = jnp.atleast_1d(proba)
        pks = vmap(self.pk_fn)(meshes)
        med_pk = jnp.median(pks, 0)
        intervals = []
        for p in proba:
            intervals.append(hdi(pks, p, 0)) # hdi or qbi
        return med_pk, jnp.stack(intervals)

    def _plot_pk(self, samples, proba=.95, label=None, color=None):
        proba = jnp.atleast_1d(proba)
        name, = name_latent(samples, ['init_mesh'])
        # Concatenate all chains and get location and dispersion parameters
        med_pk, interv_pk = self.interval_pk(jnp.concatenate(samples[name]), proba)

        plot_fn = lambda pk, i_ell, **kwargs: plt.plot(pk[0], pk[0]*pk[i_ell+1], **kwargs)
        plotfill_fn = lambda pklow, pkup, i_ell, **kwargs: plt.fill_between(
            pklow[0], pklow[0]*pklow[i_ell+1], pklow[0]*pkup[i_ell+1], **kwargs)
        
        n_plot = len(self.multipoles)
        for i_ell, ell in enumerate(self.multipoles):
            plt.subplot(1, n_plot, i_ell+1)
            plot_fn(med_pk, i_ell, linestyle='--', color=color)

            for i_p, p in enumerate(proba):
                if i_p == 0: lab = label
                else: lab = None
                plotfill_fn(interv_pk[i_p,0], interv_pk[i_p,1], i_ell, 
                    label=lab, alpha=float(1-p)**(1/2), color=color)

            plt.xlabel("$k$ [$h$/Mpc]"), plt.ylabel(f"$k P_{ell}$ [Mpc/$h$]$^2$")
            plt.legend()

    def plot_fiduc_pk(self, label=None, color=None):
        plot_fn = lambda pk, i_ell, **kwargs: plt.plot(pk[0], pk[0]*pk[i_ell+1], **kwargs)
        n_plot = len(self.multipoles)
        for i_ell, ell in enumerate(self.multipoles):
            plt.subplot(1, n_plot, i_ell+1)
            plot_fn(self.fiduc_pk, i_ell, color=color, label=label)
            plt.xlabel("$k$ [$h$/Mpc]"), plt.ylabel(f"$k P_{ell}$ [Mpc/$h$]$^2$")
            plt.legend()

    def plot_pk(self, samples, proba=.95, label=None, cmap='tab10'):
        samples = np.atleast_1d(samples)
        label = np.atleast_1d(label)
        assert len(samples)==len(label), "lists must have the same lengths."
        cmap = plt.get_cmap(cmap)
        color = [cmap(i) for i in range(len(samples))]

        if hasattr(self, 'fiduc_pk'):
            self.plot_fiduc_pk(label='true', color='k')
        for samp, lab, col in zip(samples, label, color):
            self._plot_pk(samp, proba, label=lab, color=col)





    @staticmethod
    def _metric_traj(metric_fn, samples, true=None, n=1):
        samples = samples.copy()
        infos = {}
        for k in ['n_evals']:
            info = samples.pop(k, None)
            if info is not None:
                traj_fn = partial(fn_traj, lambda x:x.sum(), n, axis=1)
                infos[k] = traj_fn(info) # sum n_evals

        traj_fn = partial(fn_traj, metric_fn, n, axis=1)
        if true is None:
            samples = tree_map(traj_fn, samples)
        else:
            samples = tree_map(traj_fn, samples, true)

        return samples | infos

    def metric_traj(self, metric_fn, samples, true=None, n=1, comb=True, transform_fn=lambda x:x):
        samps = np.atleast_1d(samples)
        trajs = []
        for samp in samps:
            if comb:
                traj = self._metric_traj(metric_fn, self.recombine_latent(samp), true=true, n=n)
            else:
                traj = self._metric_traj(partial(metric_fn, axis=()), samp, true=true, n=n)
            trajs.append(transform_fn(traj))
        
        if isinstance(samples, dict):
            return trajs[0]
        else:
            return trajs 
        

    @staticmethod
    def _plot_traj(x, y, label, ylabel=None, c=None, ls=None):
        c = [None] if c is None else to_rgba_array(c)
        ls = np.atleast_1d(ls)
        label = np.atleast_1d(label)

        if x is None:
            xlabel = ""
            x = []
        else:
            xlabel = "$N_{\\textrm{eval}}$"
            x = [x]
        if not plt.rcParams['text.usetex']:
            xlabel = xlabel.replace('textrm', 'text')
            if ylabel is not None:
                ylabel = ylabel.replace('textrm', 'text')

        for yi, lbi, ci, lsi in zip(y.T, cycle(label), cycle(c), cycle(ls)):
            if isinstance(lsi, set):
                lsi = lsi.copy().pop()
            plt.semilogy(*x, yi, label=lbi, c=ci, linestyle=lsi)
        plt.xlabel(xlabel), plt.ylabel(ylabel)


    def plot_traj(self, traj, comb=True, ylabel=None, c=None, ls=None):
        traj = traj.copy()
        n_evals = traj.pop('n_evals', None)
    
        if comb:
            values = jnp.array(list(traj.values())).T
            labels = np.array(list(traj.keys()))

            self._plot_traj(n_evals, values, labels, ylabel, c, ls)
            plt.legend()
        else:
            labels = self.label_chains(traj, axis=1)
            traj = self.recombine_latent(traj, rest=False)
            labels = self.recombine_latent(labels, rest=False)
            
            n_plot = len(traj)
            for i_k, k in enumerate(traj):
                plt.subplot(1, n_plot, i_k+1)
                plt.title(k)
                self._plot_traj(n_evals, traj[k], labels[k], ylabel, c, ls)
                plt.legend()


    



    def plot_metric_traj(self, metric:str, samples:dict|Iterable[dict], label:str|Iterable[str]=None, 
                         true=None, n=1, comb=True, cmap='tab10'):
        samps = np.atleast_1d(samples)
        label = np.atleast_1d(label)
        assert len(samps)==len(label), "lists must have the same lengths."
        cmap = plt.get_cmap(cmap)
        c = [cmap(i) for i in range(len(samps))]
        ls = ['-', ':', '--', '-.', {(0, (1, 10))}]

        if metric == 'ess':
            ylabel = "$N_{\\textrm{eval}}\\;/\\;N_{\\textrm{eff}}$"
            metric_fn = multi_ess
            transform_fn = transform_ess
        
        trajs = self.metric_traj(metric_fn, samps, true=true, n=n, 
                                 comb=comb, transform_fn=transform_fn)

        for traj, ci in zip(trajs, c):
            self.plot_traj(traj, comb=comb, ylabel=ylabel, c=ci, ls=ls)

        handles = []
        names = trajs[0].copy()
        names.pop('n_evals', None)
        for ci, lab in zip(c, label):
            handles.append(Patch(color=ci, label=lab))
        for name, lsi in zip(names, cycle(ls)):
            if isinstance(lsi, set):
                lsi = lsi.copy().pop()
            handles.append(Line2D([], [], color='grey', linestyle=lsi, label=name))
        plt.legend(handles=handles)




    def plot_metric_last(self, metric:str, samples:dict|Iterable[dict], label:str|Iterable[str]=None, 
                         true=None, comb=True, cmap='tab10'):
        samps = np.atleast_1d(samples)
        label = np.atleast_1d(label)
        assert len(samps)==len(label), "lists must have the same lengths."
        cmap = plt.get_cmap(cmap)
        c = [cmap(i) for i in range(len(samps))]
        markers = ['o','s','D']
        ms = 12

        if metric == 'ess':
            ylabel = "$N_{\\textrm{eval}}\\;/\\;N_{\\textrm{eff}}$"
            metric_fn = multi_ess
            transform_fn = transform_ess
        
        n = 1
        trajs = self.metric_traj(metric_fn, samps, true=true, n=n, 
                                 comb=comb, transform_fn=transform_fn)

        for traj, ci in zip(trajs, c):
            traj = traj.copy()
            n_evals = traj.pop('n_evals', None)
            x = list(traj.keys())
            y = list(tree_map(lambda x:x[-1], traj).values())
            plt.semilogy(x, y, c=ci, ls='', marker=markers[0], markersize=ms)

            ylabel = np.atleast_1d(ylabel)
            handles = []
            for ci, lab in zip(c, label):
                handles.append(Patch(color=ci, label=lab))
            for name, mi in zip(ylabel, cycle(markers)):
                if not plt.rcParams['text.usetex']:
                    name = name.replace('textrm', 'text')
                handles.append(Line2D([], [], color='grey', ls='', marker=mi, markersize=ms, label=name))
            plt.legend(handles=handles, frameon=True)


def transform_ess(traj):
    traj_temp = traj.copy()
    n_evals = traj_temp.pop('n_evals', None)
    traj_temp = tree_map(lambda x: jnp.moveaxis(n_evals/jnp.moveaxis(x, 0, -1), -1, 0), traj_temp)
    return traj | traj_temp
    



###############
# NumPyro API #
###############

# TODO: can select var_names directly in numpyro run api

def save_run(mcmc:MCMC, i_run:int, path:str, names:list=None, 
             extra_fields:list=None, group_by_chain:bool=True):
    """
    Save one run of MCMC sampling, with extra fields and last state.
    If `var_names` is None, save all the variables.
    """
    # Save samples (and extra fields)
    samples = mcmc.get_samples(group_by_chain)
    if names is not None:
        samples = {key: samples[key] for key in names}

    if extra_fields is not None:
        extra = mcmc.get_extra_fields(group_by_chain)
        if "num_steps" in extra: # renaming num_steps into clearer n_evals
            n_evals = extra.pop("num_steps")
            samples.update(n_evals=n_evals)
        samples |= extra
        del extra

    pdump(samples, path+f"_{i_run}.p")
    # jnp.savez(save_path+f"_{i_run}.npz", **samples) # better for dict of array-likes
    del samples

    # Save or overwrite last state
    pdump(mcmc.last_state, path+f"_last_state.p") 


def sample_and_save(mcmc:MCMC, n_runs:int, path:str, names:list=None, 
                    extra_fields:list=[], rng=42, group_by_chain:bool=True, init_params=None) -> MCMC:
    """
    Warmup and run MCMC, saving the specified variables and extra fields.
    Do `mcmc.num_warmup` warmup steps, followed by `n_runs` times `mcmc.num_samples` sampling steps.
    If `var_names` is None, save all the variables.
    """
    if isinstance(rng, int):
        rng = jr.key(rng)

    # Warmup sampling
    if mcmc.num_warmup>=1:
        print(f"run {0}/{n_runs} (warmup)")

        # Warmup
        mcmc.warmup(rng, collect_warmup=True, extra_fields=extra_fields, init_params=init_params)
        save_run(mcmc, 0, path, names, extra_fields, group_by_chain)

        # Handling rng key and destroy init_params
        rng_run = mcmc.post_warmup_state.rng_key
        init_params = None
    else:
        rng_run = rng

    # Run sampling
    for i_run in range(1, n_runs+1):
        print(f"run {i_run}/{n_runs}")
            
        # Run
        mcmc.run(rng_run, extra_fields=extra_fields, init_params=init_params)
        save_run(mcmc, i_run, path, names, extra_fields)

        # Init next run at last state
        mcmc.post_warmup_state = mcmc.last_state
        rng_run = mcmc.post_warmup_state.rng_key
    return mcmc



    