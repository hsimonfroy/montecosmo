from __future__ import annotations # for Union typing with | in python<3.10

from functools import partial
from dataclasses import dataclass, asdict
from IPython.display import display
from pprint import pformat

import numpyro.distributions as dist
from numpyro import sample, deterministic, render_model
from numpyro.handlers import seed, condition, block, trace
from numpyro.infer.util import log_density
import numpy as np

from jax import numpy as jnp, random as jr, vmap, tree, grad, debug
from jax.scipy.spatial.transform import Rotation

# from jaxpm.painting import cic_paint
from jax_cosmo import Cosmology
from montecosmo.bricks import (samp2base, samp2base_mesh, get_cosmology, lin_power_mesh, add_png,
                               kaiser_boost, kaiser_model, kaiser_posterior,
                               lagrangian_weights,
                               simple_window, mesh2masked, masked2mesh,
                               tophysical_mesh, tophysical, toredshift_param, toredshift_auto, toredshift_auto2,
                               phys2cell_pos, cell2phys_pos, phys2cell_vel, cell2phys_vel,
                               catalog2mesh, catalog2window, pos_mesh, get_ptcl_shape)
from montecosmo.nbody import lpt, nbody_bf, nbody_bf_scan, chi2a, a2chi, a2g, g2a, a2f, paint
from montecosmo.metrics import spectrum, powtranscoh, deconv_paint
from montecosmo.utils import ysafe_dump, ysafe_load, Path

from montecosmo.utils import cgh2rg, rg2cgh, ch2rshape, nvmap, safe_div, DetruncTruncNorm, DetruncUnif, rg2cgh2
from montecosmo.chains import Chains



default_config={
            # Mesh and box parameters
            'mesh_shape':3 * (64,), # int
            'cell_length': 5., # in Mpc/h
            'box_center':(0.,0.,0.), # in Mpc/h
            'box_rotvec':(0.,0.,0.), # rotation vector in radians
            # 'box_shape':3 * (320.,), # in Mpc/h
            # Evolution
            'evolution':'lpt', # kaiser, lpt, nbody
            'nbody_steps':5, # number of N-body steps
            'nbody_snapshots':None, # number of N-body snapshots to save, if None, only save last
            'lpt_order':2, # order of LPT displacement
            'paint_order':2, # order of interpolation kernel
            'oversampling':1., # particle grid to mesh grid 1D ratio
            # Observables
            'observable':'field', # 'field', TODO: 'powspec' (with poles), 'bispec'
            'poles':(0,2,4), # multipoles order to compute, if observable is 'powspec'
            'a_obs':None, # light-cone if None
            'curved_sky':True, # curved vs. flat sky
            'ap_auto': True, # auto AP vs. parametric AP
            'window':None, # if float, padded fraction, if str or Path, path to window mesh file
            # 'save_dir':str,
            # Latents
            'precond':'kaiser', # real, fourier, kaiser, kaiser_dyn
            'latents': {'Omega_m': {'group':'cosmo', 
                                    'label':'{\\Omega}_m', 
                                    'loc':0.3111, 
                                    # 'loc':0.3, 
                                    'scale':0.5,
                                    'scale_fid':1e-2,
                                    'low': 0.05, # XXX: Omega_m < Omega_b implies nan
                                    'high': 1.},
                        'sigma8': {'group':'cosmo',
                                    'label':'{\\sigma}_8',
                                    'loc':0.8102,
                                    'scale':0.5,
                                    'scale_fid':1e-2,
                                    'low': 0.,
                                    'high':jnp.inf,},
                        'b1': {'group':'bias',
                                    'label':'{b}_1',
                                    'loc':1.,
                                    'scale':0.5,
                                    'scale_fid':1e-2,
                                    },
                        'b2': {'group':'bias',
                                    'label':'{b}_2',
                                    'loc':0.,
                                    'scale':5.,
                                    'scale_fid':1e-1,
                                    },
                        'bs2': {'group':'bias',
                                    'label':'{b}_{s^2}',
                                    'loc':0.,
                                    'scale':5.,
                                    'scale_fid':1e-1,
                                    },
                        'bn2': {'group':'bias',
                                    'label':'{b}_{\\nabla^2}',
                                    'loc':0.,
                                    'scale':5.,
                                    'scale_fid':1e0,
                                    },
                        'bnp': {'group':'bias',
                                    'label':'{b}_{\\nabla_\\parallel}',
                                    'loc':0.,
                                    'scale':5.,
                                    'scale_fid':1e0,
                                    },
                        'fNL': {'group':'bias',
                                    'label':'{f}_\\mathrm{NL}',
                                    'loc':0.,
                                    'scale':1e3,
                                    'scale_fid':1e1,
                                    },
                        'alpha_iso': {'group':'ap',
                                      'label':'{\\alpha_\\mathrm{iso}}',
                                      'loc':1.,
                                      'scale':.1,
                                      'scale_fid':1e-2,
                                      'low':0.,
                                      'high':jnp.inf,
                                      },
                        'alpha_ap': {'group':'ap',
                                      'label':'{\\alpha_\\mathrm{AP}}',
                                      'loc':1.,
                                      'scale':.1,
                                      'scale_fid':1e-2,
                                      'low':0.,
                                      'high':jnp.inf,
                                      },
                        'ngbar': {'group':'syst',
                                      'label':'{\\bar{n}_g}',
                                      'loc':1e-3, # in galaxy / (Mpc/h)^3
                                      'scale':.1,
                                      'scale_fid':1e-6,
                                      'low':0.,
                                      'high':jnp.inf,
                                      },
                        'init_mesh': {'group':'init',
                                      'label':'{\\delta_\\mathrm{L}}',},
                        },
            }




@dataclass
class Model():
    def __post_init__(self):
        self.data = {} # to store observed values
 
    ###############
    # Model calls #
    ###############
    def _model(self, *args, **kwargs):
        raise NotImplementedError

    def model(self, *args, **kwargs):
        return self._model(*args, **kwargs)
    
    def reset(self):
        self.model = self._model
        self.data = {}

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
        return block(model, hide_fn=hide_fn)

    def predict(self, rng=42, samples:int|tuple|dict=None, batch_ndim=0, hide_base=True, hide_det=True, hide_samp=True, from_base=False):
        """
        Run model conditioned on samples.
        * If samples is None, return a single prediction.
        * If samples is an int or tuple, return a prediction of such shape.
        * If samples is a dict, return a prediction for each sample, assuming batch_ndim batch dimensions.
        """
        if isinstance(rng, int):
            rng = jr.key(rng)

        def single_prediction(rng, sample={}):
            # Optionally reparametrize base to sample params
            if from_base:
                sample = self.reparam(sample, inv=True) 
                # NOTE: deterministic sites have no effects with handlers.condition, but do with handlers.subsitute

            # Condition then block
            model = condition(self.model, data=sample)
            if hide_samp:
                model = block(model, hide=sample.keys())
            model = self._block_det(model, hide_base=hide_base, hide_det=hide_det)

            # Trace and return values
            tr = trace(seed(model, rng_seed=rng)).get_trace()
            return {k: v['value'] for k, v in tr.items()}

        if samples is None:
            return single_prediction(rng)
        
        elif isinstance(samples, (int, tuple)):
            if isinstance(samples, int):
                samples = (samples,)
            rng = jr.split(rng, samples)
            return nvmap(single_prediction, len(samples))(rng)
        
        elif isinstance(samples, dict):
            if len(samples) == 0:
                return {}
            else:
                # All item shapes should match on the first batch_ndim dimensions,
                # so take the first item shape
                shape = jnp.shape(next(iter(samples.values())))[:batch_ndim]
                rng = jr.split(rng, shape)
                return nvmap(single_prediction, len(shape))(rng, samples)
        


    ############
    # Wrappers #
    ############
    def logpdf(self, params={}):
        """
        A log-density function of the model. In particular, it is the log-*probability*-density function 
        with respect to the full set of variables, i.e. E[e^logpdf] = 1.

        For unnormalized log-densities in numpyro, see https://forum.pyro.ai/t/unnormalized-densities/3251/9
        """
        return log_density(self.model, (), {}, params)[0]

    def potential(self, params={}):
        return - self.logpdf(params)
    
    def force(self, params={}):
        return grad(self.logpdf)(params) # force = - grad potential = grad logpdf
    
    def trace(self, rng):
        return trace(seed(self.model, rng_seed=rng)).get_trace()
    
    def seed(self, rng):
        self.model = seed(self.model, rng_seed=rng)

    def condition(self, data={}, from_base=False):
        """
        Substitue rnadom variables by their provided values, 
        optionally reparametrizing base values into sample values.
        Values are stored in attribute data.
        """
        if from_base:
            self.data |= data
            data = self.reparam(data, inv=True)
        self.data |= data
        self.model = condition(self.model, data=data)

    def block(self, hide_fn=None, hide=None, expose_types=None, expose=None, hide_base=True, hide_det=True):
        """
        Selectively hides parameters in the model. In particular, avoid them being returned in trace calls.
        
        Precedence is given according to the order: hide_fn, hide, expose_types, expose, (hide_base, hide_det).
        Only the set of parameters with the precedence is considered.
        The default call thus hides base and other deterministic sites, for sampling purposes.
        """
        if all(x is None for x in (hide_fn, hide, expose_types, expose)):
            self.model = self._block_det(self.model, hide_base=hide_base, hide_det=hide_det)
        else:
            self.model = block(self.model, hide_fn=hide_fn, hide=hide, expose_types=expose_types, expose=expose)

    def render(self, filename=None, render_dist=False, render_params=False):
        # NOTE: filename ignores path
        display(render_model(self.model, filename=filename, render_distributions=render_dist, render_params=render_params))

    def partial(self, *args, **kwargs):
        self.model = partial(self.model, *args, **kwargs)



    #################
    # Save and load #
    #################
    def asdict(self):
        return asdict(self)
    
    def save(self, path): # with yaml because not array-like
        ysafe_dump(asdict(self), path)

    @classmethod
    def load(cls, path):
        return cls(**ysafe_load(path))









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
        Shape of the box in Mpc/h. Typically such that cell lengths would be between 1 and 10 Mpc/h.
    evolution : str
        Evolution model: 'kaiser', 'lpt', 'nbody'.
    a_obs : float
        Scale factor of observations.
    nbody_steps : int
        Number of N-body steps.
        Only used for 'nbody' evolution.
    nbody_snapshots : int or list
        Number or list of N-body snapshots to save. If None, only save last.
        Only used for 'nbody' evolution.
    lpt_order : int
        Order of LPT displacement. 
        Only used for 'lpt' evolution.
    observable : str
        Observable: 'field', 'powspec'.
    los : array_like
        Line-of-sight direction for flat-sky simulation. 
        If None, no Redshift Space Distorsion is applied.
    box_center : array_like
        Center of the box, i.e. observer position, in Mpc/h for curved-sky simulation.
        If None, no Redshift Space Distorsion is applied.
    poles : array_like of int
        Power spectrum poles to compute.
        Only used for 'powspec' observable.
    precond : str
        Preconditioning method: 'real', 'fourier', 'kaiser', 'kaiser_dyn'.
    latents : dict
        Latent variables configuration.
    """
    # Mesh and box parameters
    mesh_shape:np.ndarray
    cell_length:float
    box_center:np.ndarray
    box_rotvec:np.ndarray
    # Evolution
    evolution:str
    nbody_steps:int
    nbody_snapshots:int|list
    lpt_order:int
    paint_order:int
    oversampling:float
    # Observable
    observable:str
    poles:tuple
    a_obs:float
    curved_sky:bool
    ap_auto:bool
    window:float|str
    # Latents
    precond:str
    latents:dict

    def __post_init__(self):
        super().__post_init__()
        self.latents = self._validate_latents()
        self.groups = self._groups(base=True)
        self.groups_ = self._groups(base=False)
        self.labels = self._labels()
        self.loc_fid = self._loc_fid()

        self.mesh_shape = np.asarray(self.mesh_shape)
        # NOTE: if x32, cast mesh_shape into float to avoid overflow when computing products
        self.cell_length = float(self.cell_length)
        self.box_center = np.asarray(self.box_center)
        self.box_rotvec = np.asarray(self.box_rotvec)
        self.box_rot = Rotation.from_rotvec(self.box_rotvec)
        self.box_shape = self.mesh_shape * self.cell_length
        self.ptcl_shape = get_ptcl_shape(self.mesh_shape, self.oversampling)

        self.k_funda = 2*np.pi / np.min(self.box_shape) 
        self.k_nyquist = np.pi * np.min(self.mesh_shape / self.box_shape)
        self.count_fid = self.loc_fid['ngbar'] * self.cell_length**3
        
        if self.window is None:
            self.wind_mesh = np.array(1.)
            self.mask = None
        elif isinstance(self.window, float):
            self.wind_mesh = simple_window(self.mesh_shape, self.window, ord=np.inf) 
            self.mask = self.wind_mesh > 0
        elif isinstance(self.window, (str, Path)):
            self.window = str(self.window) # cast to str to allow yaml saving
            self.wind_mesh = np.load(self.window)
            self.mask = self.wind_mesh > 0

        self.cosmo_fid = get_cosmology(**self.loc_fid)
        _, a = tophysical_mesh(self.box_center, self.box_rot, self.box_shape, self.mesh_shape,
                                self.cosmo_fid, self.a_obs, self.curved_sky)
        self.a_fid = g2a(self.cosmo_fid, jnp.mean(a2g(self.cosmo_fid, a)))

    def __str__(self):
        out = ""
        out += f"# CONFIG\n"
        out += pformat(asdict(self), width=1)
        out += "\n\n# INFOS\n"
        out += f"box_shape:      {list(self.box_shape)} Mpc/h\n"
        out += f"k_funda:        {self.k_funda:.5f} h/Mpc\n"
        out += f"k_nyquist:      {self.k_nyquist:.5f} h/Mpc\n"
        out += f"mean_count_fid: {self.count_fid:.3f} gxy/cell\n"
        out += f"ptcl_shape:     {list(self.ptcl_shape)} ptcl\n"
        return out

    def _model(self, temp_prior=1., temp_lik=1.):
        x = self.prior(temp=temp_prior)
        x = self.evolve(x)
        return self.likelihood(x, temp=temp_lik)
    




    def prior(self, temp=1.):
        """
        A prior for cosmological model. 

        Return base parameters, as reparametrization of sample parameters.
        """
        # Sample, reparametrize, and register cosmology and biases
        tup = ()
        for g in ['cosmo', 'bias', 'ap', 'syst']:
            dic = self._sample(self.groups[g]) # sample               
            dic = samp2base(dic, self.latents, inv=False, temp=temp) # reparametrize
            tup += ({k: deterministic(k, v) for k, v in dic.items()},) # register base params
        cosmo, bias, ap, syst = tup
        cosmology = get_cosmology(**cosmo)        

        # Sample, reparametrize, and register initial conditions
        init = {}
        name_ = self.groups['init'][0]+'_'

        scale, transfer = self._precond_scale_and_transfer(cosmology, bias, syst)
        init[name_] = sample(name_, dist.Normal(0., scale)) # sample
        init = samp2base_mesh(init, self.precond, transfer=transfer, inv=False, temp=temp) # reparametrize
        init = {k: deterministic(k, v) for k, v in init.items()} # register base params

        return cosmology, bias, ap, syst, init


    def evolve(self, params:tuple):
        cosmology, bias, ap, syst, init = params

        # init['init_mesh'] = add_png(cosmology, bias['fNL'], init['init_mesh'], self.box_shape) ######

        if self.evolution=='kaiser':
            # TODO: AP Kaiser?
            los, a = tophysical_mesh(self.box_center, self.box_rot, self.box_shape, self.mesh_shape,
                                cosmology, self.a_obs, self.curved_sky)
            gxy_mesh = kaiser_model(cosmology, a, bE=1 + bias['b1'], **init, los=los)
            gxy_mesh = deterministic('gxy_mesh', gxy_mesh)
            return gxy_mesh, syst
                    
        # Create regular grid of particles, and get their scale factors
        # pos = jnp.indices(self.mesh_shape, dtype=float).reshape(3,-1).T
        # print("oldy")

        pos = [np.linspace(0, m, p, endpoint=False) for m, p in zip(self.mesh_shape, self.ptcl_shape)]
        pos = jnp.stack(np.meshgrid(*pos, indexing='ij'), axis=-1).reshape(-1, 3)

        _, _, los, a = tophysical(pos, self.box_center, self.box_rot, self.box_shape, self.mesh_shape, 
                                cosmology, self.a_obs, self.curved_sky)

        # Lagrangian bias expansion weights at a_obs (but based on initial particules positions)
        lbe_weights = lagrangian_weights(cosmology, pos, los, a, self.box_shape, **bias, **init)
        # TODO: gaussian lagrangian weights?

        if self.evolution=='lpt':
            # NOTE: lpt assumes given mesh is at a=1
            cosmology._workspace = {}  # HACK: temporary fix
            dpos, vel = lpt(cosmology, **init, pos=pos, a=a, 
                            lpt_order=self.lpt_order, paint_order=self.paint_order, grad_fd=False, lap_fd=False)
            # print("ratio", (self.mesh_shape / self.ptcl_shape).prod() )
            # pos += dpos * (self.mesh_shape / self.ptcl_shape).prod()

            pos += dpos
            pos, vel = deterministic('lpt_ptcl', jnp.array((pos, vel)))

        elif self.evolution=='nbody':
            cosmology._workspace = {}  # HACK: temporary fix
            assert jnp.ndim(a) == 0, "N-body light-cone not implemented yet"
            pos, vel = nbody_bf(cosmology, **init, pos=pos, a=a, n_steps=self.nbody_steps, 
                                paint_order=self.paint_order, grad_fd=False, lap_fd=False, snapshots=self.nbody_snapshots)
            pos, vel = deterministic('nbody_ptcl', jnp.array((pos, vel)))
            pos, vel = tree.map(lambda x: x[-1], (pos, vel))

        pos, rpos, los, a = tophysical(pos, self.box_center, self.box_rot, self.box_shape, self.mesh_shape,
                                      cosmology, self.a_obs, self.curved_sky)        
        vel = cell2phys_vel(vel, self.box_rot, self.box_shape, self.mesh_shape)
        vel *= a2g(cosmology, a) * a2f(cosmology, a)
        # Growth-time integrator vel := dq / dg = v / (H * g * f), so dq_rsd := v / H = vel * g * f
        # v in (Mpc/h)*(km/s/(Mpc/h)) = km/s, so dq_rsd in Mpc/h

        # RSD and Alcock-Paczynski effects
        if self.ap_auto:
            pos = toredshift_auto2(pos, vel, rpos, los, a, cosmology, self.cosmo_fid, self.curved_sky) ########
            # pos = toredshift_auto(pos, vel, rpos, los, a, cosmology, self.cosmo_fid, self.curved_sky) 
        else:
            pos = toredshift_param(pos, vel, los, ap, self.curved_sky)

        # Painting weighted by Lagrangian bias expansion weights
        pos = phys2cell_pos(pos, self.box_center, self.box_rot, self.box_shape, self.mesh_shape)
        gxy_mesh = paint(pos, tuple(self.mesh_shape), lbe_weights, self.paint_order)
        gxy_mesh = deconv_paint(gxy_mesh, order=self.paint_order); print("fin deconv") # NOTE: final deconvolution amplifies AP-induced high-frequencies.
        gxy_mesh = deterministic('gxy_mesh', gxy_mesh)

        # debug.print("lbe_weights: {i}", i=(lbe_weights.mean(), lbe_weights.std(), lbe_weights.min(), lbe_weights.max()))
        # debug.print("biased mesh: {i}", i=(biased_mesh.mean(), biased_mesh.std(), biased_mesh.min(), biased_mesh.max()))
        # debug.print("frac of weights < 0: {i}", i=(lbe_weights < 0).sum()/len(lbe_weights))
        return gxy_mesh, syst # NOTE: mesh is 1+delta_obs


    def likelihood(self, params:tuple, temp=1.):
        """
        A likelihood for cosmological model.

        Return an observed mesh sampled from a location mesh with observational variance.
        """
        mesh, syst = params

        if self.observable == 'field':
            obs = mesh2masked(mesh, self.mask)
            obs /= obs.mean()
            # print("mesh", mesh.mean(), mesh.std(), mesh.min(), mesh.max())
            # print(obs.mean(), obs.std(), obs.min(), obs.max())

            wind = mesh2masked(self.wind_mesh, self.mask)
            obs *= wind * syst['ngbar'] * self.cell_length**3

            # Gaussian noise
            # obs = sample('obs', dist.Normal(obs, (temp * syst['ngbar'] * self.cell_length**3)**.5))
            obs = sample('obs', dist.Normal(obs, (temp * syst['ngbar'] * self.cell_length**3)**.5 / 1e9))
            # obs = sample('obs', dist.Normal(obs, jnp.abs(temp * obs)**.5))
            # obs = sample('obs', dist.Normal(obs, jnp.abs(jnp.mean(temp * obs))**.5))

            # Poisson noise
            # obs = sample('obs', dist.Poisson(jnp.abs(obs)**(1 / temp)))
            return obs 

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


    def reparam(self, params:dict, fourier=True, inv=False, temp=1.):
        """
        Transform sample params into base params.
        """
        # Retrieve potential substituted params
        params_ = self.data | params

        # Extract groups from params
        groups = ['cosmo','bias','ap','syst','init']
        key = tuple([k if inv else k+'_'] for k in groups) + tuple([['*'] + ['~'+k if inv else '~'+k+'_' for k in groups]])
        params_ = Chains(params_, self.groups | self.groups_).get(key) # use chain querying
        cosmo_, bias_, ap_, syst_, init, rest = (q.data for q in params_)

        # Cosmology and Biases
        cosmo = samp2base(cosmo_, self.latents, inv=inv, temp=temp)
        bias = samp2base(bias_, self.latents, inv=inv, temp=temp)
        ap = samp2base(ap_, self.latents, inv=inv, temp=temp)
        syst = samp2base(syst_, self.latents, inv=inv, temp=temp)

        # Initial conditions
        if len(init) > 0:
            cosmology = get_cosmology(**(cosmo_ if inv else cosmo))
            _, transfer = self._precond_scale_and_transfer(cosmology, 
                                                           bias_ if inv else bias, 
                                                           syst_ if inv else syst)

            if not fourier and inv:
                init = tree.map(lambda x: jnp.fft.rfftn(x), init)

            init = samp2base_mesh(init, self.precond, transfer=transfer, inv=inv, temp=temp)
            
            if not fourier and not inv:
                init = tree.map(lambda x: jnp.fft.irfftn(x), init)

        out = cosmo | bias | ap | syst | init
        out = {k:v for k,v in out.items() if (k[:-1] if inv else k+'_') in params}
        rest = {k:v for k,v in rest.items() if k in params} # do not return data
        out = rest | out # possibly update rest
        return out



    ###########
    # Getters #
    ###########   
    def _validate_latents(self):
        """
        Return a validated latents config.
        """
        new = {}
        for name, conf in self.latents.items():
            new[name] = conf.copy()
            loc, scale = conf.get('loc'), conf.get('scale')
            low, high = conf.get('low'), conf.get('high')
            loc_fid, scale_fid = conf.get('loc_fid'), conf.get('scale_fid')

            assert not (loc is None) ^ (scale is None),\
                f"latent '{name}' not valid: loc and scale must be both provided or both not provided"
            assert not (low is None) ^ (high is None),\
                f"latent '{name}' not valid: low and high must be both provided or both not provided"
            
            if loc is not None: # Normal or Truncated Normal prior
                if loc_fid is None:
                    new[name]['loc_fid'] = loc
                if scale_fid is None:
                    new[name]['scale_fid'] = scale
            
            elif low is not None: # Uniform prior
                assert low <= high,\
                    f"latent '{name}' not valid: low must be lower than high"
                assert low != -jnp.inf and high != jnp.inf,\
                    f"latent '{name}' not valid: low and high must be finite for uniform distribution"
                if loc_fid is None:
                    new[name]['loc_fid'] = (low + high) / 2
                if scale_fid is None:
                    new[name]['scale_fid'] = (high - low) / 12**.5
        return new

    def _sample(self, names:str|list):
        """
        Sample latent parameters from latents config.
        """
        dic = {}
        names = np.atleast_1d(names)
        for name in names:
            conf = self.latents[name]
            loc, scale = conf.get('loc', None), conf.get('scale', None)
            low, high = conf.get('low', -jnp.inf), conf.get('high', jnp.inf)
            loc_fid, scale_fid = conf['loc_fid'], conf['scale_fid']

            if loc is not None:
                if low == -jnp.inf and high == jnp.inf:
                    dic[name+'_'] = sample(name+'_', dist.Normal((loc - loc_fid) / scale_fid, scale / scale_fid))
                else:
                    dic[name+'_'] = sample(name+'_', DetruncTruncNorm(loc, scale, low, high, loc_fid, scale_fid))
            else:
                dic[name+'_'] = sample(name+'_', DetruncUnif(low, high, loc_fid, scale_fid))
        return dic            

    def _precond_scale_and_transfer(self, cosmo:Cosmology, bias, syst):
        """
        Return scale and transfer fields for linear matter field preconditioning.
        """
        pmesh = lin_power_mesh(cosmo, self.mesh_shape, self.box_shape)

        if self.precond in ['real', 'fourier']:
            scale = jnp.ones(self.mesh_shape)
            transfer = pmesh**.5

        elif self.precond=='kaiser':
            bE_fid = 1 + self.loc_fid['b1']
            boost_fid = kaiser_boost(self.cosmo_fid, self.a_fid, bE_fid, self.mesh_shape, self.box_center)
            pmesh_fid = lin_power_mesh(self.cosmo_fid, self.mesh_shape, self.box_shape)
            wind = (self.wind_mesh**2).mean()**.5

            scale = (1 + wind * self.count_fid * boost_fid**2 * pmesh_fid)**.5
            transfer = pmesh**.5 / scale
            scale = cgh2rg(scale, norm="amp")
        
        elif self.precond=='kaiser_dyn':
            bE = 1 + bias['b1']
            count = syst['ngbar'] * self.cell_length**3
            boost = kaiser_boost(cosmo, self.a_fid, bE, self.mesh_shape, self.box_center)
            wind = (self.wind_mesh**2).mean()**.5

            scale = (1 + wind * count * boost**2 * pmesh)**.5
            transfer = pmesh**.5 / scale
            scale = cgh2rg(scale, norm="amp")
        
        return scale, transfer
    
    def _groups(self, base=True):
        """
        Return groups from latents config.
        """
        groups = {}
        for name, val in self.latents.items():
            group = val['group']
            group = group if base else group+'_'
            if group not in groups:
                groups[group] = []
            groups[group].append(name if base else name+'_')
        return groups

    def _labels(self):
        """
        Return labels from latents config
        """
        labs = {}
        for name, val in self.latents.items():
            lab = val['label']
            labs[name] = lab
            labs[name+'_'] = "\\tilde"+lab
        return labs
    
    def _loc_fid(self):
        """
        Return fiducial location values from latents config.
        """
        return {k:v['loc_fid'] for k,v in self.latents.items() if 'loc_fid' in v}
    

    ########
    # Data #
    ########
    def pos_mesh(self):
        return pos_mesh(self.box_center, self.box_rot, self.box_shape, self.mesh_shape)
    
    def mesh2masked(self, mesh):
        return mesh2masked(mesh, self.mask)
    
    def masked2mesh(self, mesh):
        return masked2mesh(mesh, self.mask)

    def count2delta(self, mesh, from_masked=True):
        """
        Count mesh to delta mesh, by imposing global integral constraint.
        """
        if from_masked:
            mesh = self.masked2mesh(mesh)

        # NOTE: equivalent to:
        # mesh = self.mesh2masked(count_obs)
        # mesh = (mesh / mesh.mean() - self.mesh2masked(model.wind_mesh))
        # return self.masked2mesh(mesh) / (model.wind_mesh**2).mean()**.5

        alpha_wind = self.wind_mesh * mesh.mean() / self.wind_mesh.mean()
        return (mesh - alpha_wind) / (alpha_wind**2).mean()**.5 
    
    
    def add_window(self, load_path:str|Path, cell_budget, padding=0., save_path:str|Path=None):
        """
        Compute and save a painted window mesh from a RA, DEC, Z .fits catalog, then update model accordingly.
        """
        wind_mesh, cell_length, box_center, box_rotvec = catalog2window(load_path, self.cosmo_fid, cell_budget, padding)
        if save_path is None:
            save_path = Path(load_path).with_suffix('.npy')
        save_path = str(save_path) # cast to str to allow yaml saving
        np.save(save_path, wind_mesh)

        self.window = save_path
        self.mesh_shape = wind_mesh.shape
        self.cell_length = cell_length
        self.box_center = box_center
        self.box_rotvec = box_rotvec
        self.__post_init__() # update other model attributes

    def catalog2mesh(self, path:str|Path):
        """
        Compute a painted mesh from a RA, DEC, Z .fits catalog.
        """
        return catalog2mesh(path, self.cosmo_fid, self.box_center, self.box_rot, self.box_shape, self.mesh_shape)


    ###########
    # Metrics #
    ###########
    def spectrum(self, mesh, mesh2=None, kedges:int|float|list=None, deconv:int|tuple=(0, 0), poles:int|tuple=0):
        return spectrum(mesh, mesh2=mesh2, box_shape=self.box_shape, 
                            kedges=kedges, deconv=deconv, poles=poles, box_center=self.box_center)

    def powtranscoh(self, mesh0, mesh1, kedges:int|float|list=None, deconv=(0, 0)):
        """
        Return wavenumber, power spectrum, transfer function, and coherence of two meshes.
        """
        return powtranscoh(mesh0, mesh1, box_shape=self.box_shape, kedges=kedges, deconv=deconv)


    ##################
    # Chains process #
    ##################
    def load_runs(self, path:str, start:int, end:int, transforms=None, batch_ndim=2) -> Chains:
        return Chains.load_runs(path, start, end, transforms, 
                                groups=self.groups | self.groups_, labels=self.labels, batch_ndim=batch_ndim)

    def reparam_chains(self, chains:Chains, fourier=False, batch_ndim=2) -> Chains:
        chains = chains.copy() # TODO: tree.map
        chains.data = nvmap(partial(self.reparam, fourier=fourier), batch_ndim)(chains.data)
        return chains
    
    # def predict_chains(self, chains:Chains, rng=42, batch_ndim=2, 
    #                    hide_base=True, hide_det=True, hide_samp=True, from_base=False) -> Chains:
    #     chains = chains.copy() # TODO: tree.map
    #     chains.data = self.predict(rng=rng, samples=chains.data, batch_ndim=batch_ndim, 
    #                                hide_base=hide_base, hide_det=hide_det, hide_samp=hide_samp, from_base=from_base)
    #     return chains
    
    def powtranscoh_chains(self, chains:Chains, mesh0, name:str='init_mesh', 
                           kedges:int|float|list=None, deconv=(0, 0), batch_ndim=2) -> Chains:
        chains = chains.copy()
        fn = nvmap(lambda x: self.powtranscoh(mesh0, x, kedges=kedges, deconv=deconv), batch_ndim)
        chains.data['kptc'] = fn(chains.data[name])
        return chains
    
    def kaiser_post(self, rng, delta_obs, base=False, temp=1., scale_field=1.):
        if jnp.isrealobj(delta_obs):
            delta_obs = jnp.fft.rfftn(delta_obs)

        bE_fid = 1 + self.loc_fid['b1']
        means, stds = kaiser_posterior(delta_obs, self.cosmo_fid, bE_fid, self.count_fid, 
                                       self.wind_mesh, self.a_fid, self.box_shape, self.box_center)
        
        # HACK: rg2cgh has absurd problem with vmaped random arrays in CUDA11, so rely on rg2cgh2 until fully moved to CUDA12.
        # post_mesh = rg2cgh2(jr.normal(rng, ch2rshape(means.shape)))
        post_mesh = rg2cgh(jr.normal(rng, ch2rshape(means.shape)))
        post_mesh = temp**.5 * stds * post_mesh + means 
        # NOTE: scaling down the field is recommended when the Kaiser posterior approximation becomes less valid
        # because many high-wavevector amplitudes can be set to high.
        post_mesh *= scale_field 

        init_params = self.loc_fid | {'init_mesh': post_mesh}
        if base:
            return init_params
        else:
            return self.reparam(init_params, inv=True)

    

