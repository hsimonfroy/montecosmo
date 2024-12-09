from __future__ import annotations # for Union typing with | in python<3.10

from functools import partial
from dataclasses import dataclass, asdict
from IPython.display import display
from pprint import pformat
import os

import numpyro.distributions as dist
from numpyro import sample, deterministic, handlers, render_model
from numpyro.infer.util import log_density
import numpy as np

from jax import numpy as jnp, random as jr, vmap, grad, debug

# from jaxpm.pm import lpt, make_ode_fn
from jaxpm.painting import cic_paint
from montecosmo.bricks import samp2base, samp2base_mesh, get_cosmology, lpt, nbody, lagrangian_weights, rsd 
from montecosmo.metrics import power_spectrum
from montecosmo.utils import pdump, pload




default_config={
            # Mesh and box parameters
            'mesh_shape':3 * (64,), # int
            'box_shape':3 * (640.,), # in Mpc/h (aim for cell lengths between 1 and 10 Mpc/h)
            # LSS formation
            'a_lpt':0.1, 
            'a_obs':0.5,
            'lpt_order':1,
            # Galaxies
            'gxy_density':1e-3, # in galaxy / (Mpc/h)^3
            # Prior config {name: [group, label, loc, scale, low, high]}
            'latents': {'Omega_m': {'group':'cosmo', 
                                    'label':'{\\Omega}_m', 
                                    'loc':0.3111, 
                                    'scale':0.2,
                                    'low': 0.05, # XXX: Omega_m < Omega_c implies nan
                                    'high': 1.},
                        'sigma8': {'group':'cosmo',
                                    'label':'{\\sigma}_8',
                                    'loc':0.8102,
                                    'scale':0.2,
                                    'low': 0.,},
                        'b1': {'group':'bias',
                                    'label':'{b}_1',
                                    'loc':1.,
                                    'scale':0.5,},
                        'b2': {'group':'bias',
                                    'label':'{b}_2',
                                    'loc':0.,
                                    'scale':2.,},
                        'bs2': {'group':'bias',
                                    'label':'{b}_{s^2}',
                                    'loc':0.,
                                    'scale':2.,},
                        'bn2': {'group':'bias',
                                    'label':'{b}_{\\nabla^2}',
                                    'loc':0.,
                                    'scale':2.,},
                        'init_mesh': {'group':'init',
                                      'label':'{\\delta}_L',},}, # TODO: rajouter obs? so rename latent variables?
            'fourier':True,                    
            'obs':'mesh', # 'mesh', 'pk', 'plk', 'bk' # TODO
            'snapshots':None, # PM snapshots
            }

bench_config = {
        # Chain subsampling
        'n_cell':None,
        'rng_key':jr.key(0),
        'thinning':1,
        # Power spectrum
        'multipoles':[0,2,4],
        }





class Model():
    ###############
    # Model calls #
    ###############
    def _model(self, *args, **kwargs):
        raise NotImplementedError

    def model(self, *args, **kwargs):
        return self._model(*args, **kwargs)
    
    def reset(self):
        self.model = self._model

    def __call__(self):
        return self.model()
    
    def reparam(self, params, inv=False):
        return params
    
    def _block_det(self, model, hide_base=True, hide_det=True):
        base_name = self.latents.keys()
        if hide_base:
            if hide_det:
                hide_fn = lambda site: site['type'] == 'deterministic'
            else:
                hide_fn = lambda site: site['type'] == 'deterministic' and site['name'] in base_name
        else:
            if hide_det:
                hide_fn = lambda site: site['type'] == 'deterministic' and site['name'] not in base_name
            else:
                hide_fn = lambda site: False
        return handlers.block(model, hide_fn=hide_fn)

    def predict(self, rng=0, samples=None, batch_ndim=0, hide_base=True, hide_det=True, hide_samp=False, frombase=False):
        """
        Run model conditionned on samples.
        If samples is None, return a single prediction.
        If samples is an int or tuple, return a prediction of such shape.
        If samples is a dict, return a prediction for each sample, assuming batch_ndim batch dimensions.
        """
        if isinstance(rng, int):
            rng = jr.key(rng)

        def single_prediction(rng, sample={}):
            # Optionally reparametrize base to sample params
            if frombase:
                sample = self.reparam(sample, inv=True)

            # Condition then block
            model = handlers.condition(self.model, data=sample)
            if hide_samp:
                model = handlers.block(model, hide=sample.keys())
            model = self._block_det(model, hide_base=hide_base, hide_det=hide_det)

            # Trace and return values
            tr = handlers.trace(handlers.seed(model, rng_seed=rng)).get_trace()
            return {k: v['value'] for k, v in tr.items()}

        if samples is None:
            return single_prediction(rng)
        
        elif isinstance(samples, (int, tuple)):
            if isinstance(samples, int):
                samples = (samples,)
            rng = jr.split(rng, samples)
            # Nest vmaps
            pred_fn = single_prediction
            for _ in range(len(samples)):
                pred_fn = vmap(pred_fn)
            return pred_fn(rng)
        
        elif isinstance(samples, dict):
            # All item shapes should match on the first batch_ndim dimensions,
            # so take the first key shape
            shape = jnp.shape(samples[next(iter(samples))])[:batch_ndim]
            rng = jr.split(rng, shape)
            # Nest vmaps
            pred_fn = single_prediction
            for _ in range(len(shape)):
                pred_fn = vmap(pred_fn)
            return pred_fn(rng, samples)
    

    ############
    # Wrappers #
    ############
    def logp(self, params):
        return log_density(self.model, (), {}, params)[0]

    def potential(self, params):
        return - self.logp(params)
    
    def force(self, params):
        return grad(self.logp)(params) # force = - grad potential = grad logp
    
    def trace(self, rng=0):
        return handlers.trace(handlers.seed(self.model, rng_seed=rng)).get_trace()
    
    def seed(self, rng=0):
        self.model = handlers.seed(self.model, rng_seed=rng)

    def condition(self, data=None, frombase=False):
        # Optionally reparametrize base to sample params
        if frombase:
            data = self.reparam(data, inv=True)
        self.model = handlers.condition(self.model, data=data or {})

    def block(self, hide_fn=None, hide=None, expose_types=None, expose=None, hide_base=True, hide_det=True):
        """
        Precedence is given according to the order hide_fn, hide, expose_types, expose, (hide_base, hide_det).
        Only the set of parameters with the precedence is considered.
        The default call thus hides base and other deterministic sites, for sampling purposes.
        """
        if all(x is None for x in (hide_fn, hide, expose_types, expose)):
            self.model = self._block_det(self.model, hide_base=hide_base, hide_det=hide_det)
        else:
            self.model = handlers.block(self.model, hide_fn=hide_fn, hide=hide, expose_types=expose_types, expose=expose)

    def render(self, render_dist=False, render_params=False):
        display(render_model(self.model, render_distributions=render_dist, render_params=render_params))

    def partial(self, *args, **kwargs): # TODO: copy signature?
        self.model = partial(self.model, *args, **kwargs)

    def copy(self):
        return type(self)(**asdict(self))
    
    def deepcopy(self):
        import copy
        return copy.deepcopy(self)


    #################
    # Save and load #
    #################
    def save(self, dir_path): # XXX: path or dir_path?
        pdump(asdict(self), os.path.join(dir_path, "model.p"))

    @classmethod
    def load(cls, dir_path): # XXX: path or dir_path?
        return cls(**pload(os.path.join(dir_path, "model.p")))





@dataclass
class FieldLevelModel(Model):
    """
    Field-level cosmological model, 
    with LPT and PM displacements, Lagrangian bias, and RSD.
    The relevant variables can be traced.

    Parameters
    ----------
    mesh_shape : array_like of int
        Shape of the mesh.
    box_shape : array_like
        Shape of the box in Mpc/h. Typically aim for cell lengths between 1 and 10 Mpc/h.
    a_lpt : float
        Scale factor to which compute Lagrangian Perturbation Theory (LPT) displacement.
        If equal to a_obs, no Particule Mesh (PM) step is computed.
    a_obs : float
        Scale factor of observations.
        If equal to a_lpt, no Particule Mesh (PM) step is computed.
    gxy_density : float
        Galaxy density in galaxy / (Mpc/h)^3
    """

    mesh_shape:np.ndarray
    box_shape:np.ndarray
    a_lpt:float
    a_obs:float
    lpt_order:int
    gxy_density:float
    latents:dict
    fourier:bool
    obs:dict
    snapshots:int|list

    def __post_init__(self):
        assert(self.a_lpt <= self.a_obs), "a_lpt must be less than (<=) a_obs"
        self.groups = self._groups_config(base=True)
        self.groups_ = self._groups_config(base=False)
        self.prior_loc = self._prior_loc(base=True)

        self.mesh_shape = np.asarray(self.mesh_shape) # avoid int overflow
        self.box_shape = np.asarray(self.box_shape)
        self.cell_shape = self.box_shape / self.mesh_shape

        self.dk = 2*np.pi / np.min(self.box_shape) 
        self.k_nyquist = np.pi * np.min(self.mesh_shape / self.box_shape)
        # 2*pi factors because of Fourier transform definition
        self.gxy_count = self.gxy_density * (self.box_shape / self.mesh_shape).prod()

    def __str__(self):
        out = ""
        out += f"# CONFIG\n"
        out += pformat(asdict(self), width=1)
        out += "\n\n# INFOS\n"
        out += f"cell_shape:     {list(self.cell_shape)} Mpc/h\n"
        out += f"dk:             {self.dk:.5f} h/Mpc\n"
        out += f"k_nyquist:      {self.k_nyquist:.5f} h/Mpc\n"
        out += f"mean_gxy_count: {self.gxy_count:.3f} gxy/cell\n"
        return out

    def _model(self, temp=1.):
        x = self.prior(temp=temp)
        x = self.evolve(x)
        return self.likelihood(x, temp=temp)
    





    def prior(self, temp=1.):
        """
        A prior for cosmological model. 

        Return base parameters, as reparametrization of sample parameters.
        """
        # Sample, reparametrize, and register cosmology and biases
        tup = ()
        for g in ['cosmo', 'bias']:
            dic = self._sample_group(self.groups[g], base=False) # sample               
            dic = samp2base(dic, self.latents, inv=False, temp=temp) # reparametrize
            tup += ({k: deterministic(k, v) for k,v in dic.items()},) # register base params
        cosmo, bias = tup
        cosmology = get_cosmology(**cosmo)        

        # Sample, reparametrize, and register initial conditions
        init = {}
        name_ = self.groups['init'][0]+'_'
        init[name_] = sample(name_, dist.Normal(jnp.zeros(self.mesh_shape), jnp.ones(self.mesh_shape)))
        init = samp2base_mesh(init, cosmology, self.mesh_shape, self.box_shape, self.fourier, inv=False, temp=temp)
        init = {k: deterministic(k, v) for k,v in init.items()} # register base params
        return cosmology, bias, init


    def evolve(self, params:tuple):
        cosmology, bias, init = params

        # Create regular grid of particles
        q = jnp.indices(self.mesh_shape).reshape(3,-1).T

        # Lagrangian bias expansion weights at a_obs (but based on initial particules positions)
        lbe_weights = lagrangian_weights(cosmology, self.a_obs, q, self.box_shape, **bias, **init)

        # LPT displacement at a_lpt
        # NOTE: lpt assumes given mesh follows linear pk at a=1, and then correct by growth factor for target a_lpt
        cosmology._workspace = {}  # HACK: temporary fix
        dq, p, f = lpt(cosmology, **init, positions=q, a=self.a_lpt, mesh_shape=self.mesh_shape, order=self.lpt_order)
        particles = jnp.stack([q + dq, p])

        # PM displacement from a_lpt to a_obs
        particles = nbody(cosmology, self.mesh_shape, particles, self.a_lpt, self.a_obs, self.snapshots)
        particles = deterministic('pm_part', particles)[-1]

        # RSD displacement at a_obs
        dq = rsd(cosmology, self.a_obs, particles[1])
        particles = particles.at[0].add(dq)
        particles = deterministic('rsd_part', particles)

        # CIC paint weighted by Lagrangian bias expansion weights
        biased_mesh = cic_paint(jnp.zeros(self.mesh_shape), particles[0], lbe_weights)
        biased_mesh = deterministic('bias_mesh', biased_mesh)        
        # debug.print("lbe_weights: {i}", i=(lbe_weights.mean(), lbe_weights.std(), lbe_weights.min(), lbe_weights.max()))
        # debug.print("biased mesh: {i}", i=(biased_mesh.mean(), biased_mesh.std(), biased_mesh.min(), biased_mesh.max()))
        # debug.print("frac of weights < 0: {i}", i=(lbe_weights < 0).sum()/len(lb,e_weights))
        return biased_mesh


    def likelihood(self, mesh, temp=1.):
        """
        A likelihood for cosmological model.

        Return an observed mesh sampled from a location mesh with observational variance.
        """
        mesh_shape, box_shape = np.asarray(self.mesh_shape), np.asarray(self.box_shape)

        if self.obs == 'mesh':
            gxy_count = (self.gxy_density * (box_shape / mesh_shape).prod())
            obs_mesh = sample('obs', dist.Normal(mesh, (temp / gxy_count)**.5)) # Gaussian noise
            return obs_mesh

        # elif self.obs == 'pk':
        #     # Anisotropic power spectrum covariance, cf. [Grieb+2016](http://arxiv.org/abs/1509.04293)
        #     multipoles = np.atleast_1d(self.lik_config['multipoles'])
        #     sli_multip = slice(1,1+len(multipoles))
        #     loc_pk, Nk = get_pk_fn(mesh_shape, box_shape, multipoles=multipoles, kcount=True, gxy_density=self.gxy_density)(mesh)
        #     sigma2 *= 2*(2*multipoles[:,None]+1) * (1 / self.gxy_density**2 + 2*loc_pk[1]/self.gxy_density) / Nk

        #     loc_pk = loc_pk.at[1].add(1/self.gxy_density) # add shot noise to the mean monopole
        #     # obs_pk = loc_pk.at[sli_multip].set(sample('obs', dist.MultivariateNormal(loc_pk[sli_multip], Nk)))
        #     obs_pk = loc_pk.at[sli_multip].set(sample('obs', dist.Normal(loc_pk[sli_multip], sigma2**.5)))
        #     obs_pk = deterministic('obs_pk', obs_pk)
        #     return obs_pk




    def reparam(self, params:dict, inv=False, temp=1.):
        """
        Transform sample params into base params.
        """
        # Extract full groups from params
        cosmo_, bias, init = self._get_by_groups(params, ['cosmo','bias','init'], base=inv)

        # Cosmology and Biases
        cosmo = samp2base(cosmo_, self.latents, inv=inv, temp=temp)
        bias = samp2base(bias, self.latents, inv=inv, temp=temp)

        # Initial conditions
        if init != {}:
            cosmology = get_cosmology(**(cosmo_ if inv else cosmo))
            init = samp2base_mesh(init, cosmology, self.mesh_shape, self.box_shape, self.fourier, inv=inv, temp=temp)
        return cosmo | bias | init



    def _sample_group(self, group, base=False):
        dic = {}
        for name in group:
            name = name if base else name+'_'
            dic[name] = sample(name, dist.Normal(0, 1)) # sample
        return dic

    def _get_by_groups(self, params, groups, base=True):
        """
        Given group names, return corresponding params dict.
        """
        tup = ()
        for g in groups:
            dic = {}
            for k in self.groups[g]:
                k = k if base else k+'_'
                if k in params:
                    dic[k] = params[k]        
            tup += (dic,)
        return tup

    def _prior_loc(self, base=True):
        """
        Return location values of the latents config.
        """
        dic = {}
        for name, val in self.latents.items():
            loc = val.get('loc')
            if loc is not None:
                dic[name] = loc if base else jnp.zeros_like(loc)
        return dic
    
    def _groups_config(self, base=True):
        """
        Return groups config from latents config.
        """
        groups = {}
        for name, val in self.latents.items():
            group = val['group']
            group = group if base else group+'_'
            if group not in groups:
                groups[group] = []
            groups[group].append(name if base else name+'_')
        return groups

    @property
    def labels(self):
        labs = {}
        for name, val in self.latents.items():
            lab = val['label']
            labs[name] = lab
            labs[name+'_'] = "\\tilde"+lab
        return labs

    def spectrum(self, mesh, mesh2=None, kedges:int|float|list=None, multipoles=0, los=[0.,0.,1.]):
        return power_spectrum(mesh, mesh2=mesh2, box_shape=self.box_shape, 
                              kedges=kedges, multipoles=multipoles, los=los)
    




    # def load(self, path):

    # def prior(self):
    #     """
    #     A prior for cosmological model. 

    #     Return standardized params for computing cosmology, initial conditions, and Lagrangian biases.
    #     """
    #     # Sample reparametrized cosmology and biases
    #     for group in ['cosmo', 'bias']:
    #         params_ = {}
    #         for name in self.groups[group]:            
    #             name_ = name+'_'
    #             params_[name_] = sample(name_, dist.Normal(0, 1))
    #         yield params_

    #     # Sample reparametrized initial conditions
    #     name_ = self.groups['init'][0]+'_'         
    #     mesh = sample(name_, dist.Normal(jnp.zeros(self.mesh_shape), jnp.ones(self.mesh_shape)))
    #     yield {name_:mesh}
    
    # def reparam(self, params, inv=False, temp=1.):
    #     """
    #     Transform sample params into base params.
    #     """
    #     cosmo, bias, init = params

    #     # Cosmology and Biases
    #     cosmo = samp2base(cosmo, self.latent, inv=inv, temp=temp)
    #     cosmology = get_cosmology(**cosmo)
    #     bias = samp2base(bias, self.latent, inv=inv, temp=temp)

    #     # Initial conditions
    #     init = samp2base_mesh(init, cosmology, self.mesh_shape, self.box_shape, self.fourier, inv=inv, temp=temp)
    #     return cosmology, bias, init