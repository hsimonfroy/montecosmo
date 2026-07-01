from __future__ import annotations # for Union typing with | in python<3.10

from functools import partial
from dataclasses import dataclass, asdict
from IPython.display import display
from pprint import pformat
from pathlib import Path

from numpyro import sample, deterministic, render_model, handlers, distributions as dist
from numpyro.infer.util import log_density, compute_log_probs
import numpy as np

from jax import numpy as jnp, random as jr, vmap, tree, grad, debug, lax
from jax.scipy.spatial.transform import Rotation

from jax_cosmo import Cosmology
from montecosmo.bricks import (samp2base, samp2base_mesh, get_cosmology, lin_power_mesh, white2lin, lin2white, add_png,
                               kaiser_boost, kaiser_model, kaiser_posterior,
                               lagrangian_bias, eulerian_bias, b1_L2E, b2_L2E, b1_E2L, b2_E2L, fNL_bias,
                               top_hat_selection, gen_gauss_selection, los_scalefactor_mesh, los_scalefactor_pos, radius_mesh, phys2cell_pos, cell2phys_pos, phys2cell_vel, cell2phys_vel,
                               rsd, ap_auto, ap_param, rsd_ap_auto, ap_auto_absdetjac,
                               cutsky2count, cutsky2config, cutsky2selection, fullsky2count, get_mesh_shape, pos_mesh, regular_pos, sobol_pos, set_radial_count, count2delta)
from montecosmo.nbody import (lpt, nbody_bf, nbody_bf_scan, chi2a, a2chi, a2g, g2a, a2f, 
                              nbody_tsit5, 
                              paint, read, deconv_paint, nufft, rfftk, top_hat)
from montecosmo.metrics import spectrum, powtranscoh, distr_radial, distr_angular, mse_radius, mse_value, mse_wave
from montecosmo.utils import (ysave, yload, h5save, h5load,
                              cgh2rg, rg2cgh, ch2rshape, r2chshape, chreshape, masked2mesh, mesh2masked, scale_shape,
                              nvmap, safe_div, DetruncTruncNorm, DetruncUnif, SinhArcsinh, QuadGaussian, TwoQuadGaussian, rg2cgh2)
from montecosmo.chains import Chains



default_config={
        # Mesh and box parameters
        'final_shape': 3 * (64,), # int
        'cell_length': 20., # in Mpc/h
        'box_center':(0.,0.,0.), # in Mpc/h
        'box_rotvec':(0.,0.,0.), # rotation vector in radians
        # 'box_size': 3 * (1280.,), # in Mpc/h
        'k_cut': jnp.inf, # in h/Mpc, if None, k_nyquist
        # Init
        'png_type': None, # None, 'fNL', 'fNL_bias'
        # Evolution
        'evolution':'lpt', # kaiser, lpt, nbody
        'nbody_a_start':0., # starting scale factor for N-body, following LPT displacement
        'nbody_n_steps':10, # number of N-body steps
        'nbody_snapshots':None, # N-body snapshots to save, if int, number of snapshots, if None, only save last state
        'lpt_order':2, # order of LPT displacement
        'paint_order':2, # order of interpolation kernel
        'paint_deconv': True, # whether to deconvolve painted field
        'kernel_type':'rectangular', # 'rectangular', 'kaiser_bessel'
        'init_oversamp':3/2, # initial mesh 1D oversampling factor
        'evol_oversamp':7/4, # evolution mesh 1D oversampling factor
        'ptcl_oversamp':7/4, # particle cloud 1D oversampling factor
        # 'paint_oversamp':3/2, # painted mesh 1D oversampling factor
        'paint_oversamp':7/4, # painted mesh 1D oversampling factor (7/4 for selection function)
        'interlace_order':2, # interlacing order
        # Observables
        'observable':'field', # 'field', TODO: 'powspec' (with poles), 'bispec'
        'poles':(0,2,4), # multipoles order to compute, if observable is 'powspec'
        'a_obs':None, # if None then light-cone
        'curved_sky':True, # curved vs. flat sky
        'ap_auto': None, # auto AP vs. parametric AP, if None then no AP
        'register':None, # if None then no data; if str or Path then path to a register HDF5 file
        'n_rbins':None, # if None then set to maximum number of radial bins
        'lik_type': 'quad_gauss', # poisson, fourier_gauss, quad_gauss
        'bias_type': 'lagrangian', # lagrangian, eulerian
        # Latents
        'precond':'kaiser', # real, fourier, kaiser
        'latents': {
                'Omega_m': {'group':'cosmo', 
                            'label':r'{\Omega}_m', 
                            'loc':0.3111,
                            # 'loc': 0.3137721, 
                            'scale':0.1,
                            'scale_fid':1e-2,
                            'low': 0.05, # XXX: Omega_m < Omega_b implies nan
                            'high': 1.},
                # 'Omega_c': {'group':'cosmo', 
                #             'label':r'{\Omega}_c', 
                #             'loc':0.2607, 
                #             'scale':0.1,
                #             'scale_fid':1e-2,
                #             'low': 0.,
                #             'high': 1.},
                # 'Omega_b': {'group':'cosmo', 
                #             'label':r'{\Omega}_b', 
                #             'loc':0.0490, 
                #             'scale':0.1,
                #             'scale_fid':1e-2,
                #             'low': 0.,
                #             'high': 1.},
                'sigma8': {'group':'cosmo',
                            'label':r'{\sigma}_8',
                            'loc':0.8102,
                            # 'loc': 0.8076353990239834,
                            'scale':1e-1, # Redshift-space
                            'scale_fid':1e-2, # Redshift-space
                            # 'scale':3e-2, # Real-space
                            # 'scale_fid':3e-2, # Real-space
                            'low': 0.,
                            'high':jnp.inf,},
                'b1': {'group':'bias',
                            'label':r'{b}_1',
                            # 'label':'{b}_1 \\frac{\\s_8}{\\s_8^\\mathrm{fid}}',
                            'loc':1.,
                            'scale':1e2,
                            'scale_fid':1e-2,
                            },
                'b2': {'group':'bias',
                            'label':r'{b}_2',
                            'loc':0.,
                            'scale':1e2,
                            'scale_fid':3e-2,
                            },
                'bs2': {'group':'bias',
                            'label':r'{b}_{s^2}',
                            'loc':0.,
                            'scale':1e2,
                            'scale_fid':1e-1,
                            },
                'b3': {'group':'bias',
                            'label':r'{b}_{3}',
                            'loc':0.,
                            'scale':1e2,
                            'scale_fid':1e0,
                            },
                'bds2': {'group':'bias',
                            'label':r'{b}_{\delta s^2}',
                            'loc':0.,
                            'scale':1e2,
                            'scale_fid':1e0,
                            },
                'bs3': {'group':'bias',
                            'label':r'{b}_{s^3}',
                            'loc':0.,
                            'scale':1e2,
                            'scale_fid':1e0,
                            },
                'bn2': {'group':'bias',
                            'label':r'{b}_{\nabla^2}',
                            'loc':0.,
                            'scale':1e3,
                            'scale_fid':1e0,
                            },
                'bnpar': {'group':'bias',
                            'label':r'{b}_{\nabla_\parallel}',
                            'loc':0.,
                            'scale':1e2,
                            'scale_fid':1e0,
                            },
                'fNL': {'group':'png',
                            'label':r'{f}_\mathrm{NL}',
                            'loc':0.,
                            'scale':1e4,
                            'scale_fid':1e2,
                            },
                'fNL_bp': {'group':'png',
                            'label':r'{f}_\mathrm{NL} b_\phi',
                            'loc':0.,
                            'scale':1e4,
                            'scale_fid':3e1,
                            },
                'fNL_bpd': {'group':'png',
                            'label':r'{f}_\mathrm{NL} b_{\phi\delta}',
                            'loc':0.,
                            'scale':1e4,
                            'scale_fid':3e2,
                            },
                'fNL_bpd2': {'group':'png',
                            'label':r'{f}_\mathrm{NL} b_{\phi\delta^2}',
                            'loc':0.,
                            'scale':1e8,
                            'scale_fid':1e3,
                            },
                'fNL_bps2': {'group':'png',
                            'label':r'{f}_\mathrm{NL} b_{\phi s^2}',
                            'loc':0.,
                            'scale':1e8,
                            'scale_fid':1e4,
                            },
                'fNL_bn2p': {'group':'png',
                            'label':r'{f}_\mathrm{NL} b_{\nabla^2\phi}',
                            'loc':0.,
                            'scale':1e8,
                            'scale_fid':3e5,
                            },
                'alpha_iso': {'group':'ap',
                                'label':r'{\alpha}_\mathrm{iso}',
                                'loc':1.,
                                'scale':1e-1,
                                'scale_fid':1e-2,
                                'low':0.,
                                'high':jnp.inf,
                                },
                'alpha_ap': {'group':'ap',
                                'label':r'{\alpha}_\mathrm{AP}',
                                'loc':1.,
                                'scale':1e-1,
                                'scale_fid':1e-2,
                                'low':0.,
                                'high':jnp.inf,
                                },
                'ngbars': {'group':'syst',
                                'label':r'{\bar{n}}_g',
                                # 'loc':1e-3, # in galaxy / (Mpc/h)^3
                                'loc':0.000843318125, # in galaxy / (Mpc/h)^3
                                'scale':1e-2,
                                # 'scale_fid':3e-8,
                                'scale_fid':1e-7,
                                'low':0.,
                                'high':jnp.inf,
                                },
                's_e': {'group':'stoch',
                                'label':r'{s}_{\epsilon}',
                                'loc':1.,
                                # 'scale':1e-1,
                                'scale':1.,
                                'scale_fid':3e-3,
                                'low':0.,
                                'high':jnp.inf,
                                },
                's_k2e': {'group':'stoch',
                                'label':r'{s}_{k^2}',
                                'loc':0.,
                                'scale':3e2,
                                'scale_fid':1e1,
                                },
                's_kmu2e': {'group':'stoch',
                                'label':r'{s}_{k^2\mu^2}',
                                'loc':0.,
                                'scale':3e2,
                                'scale_fid':1e1,
                                },
                's_ed': {'group':'stoch',
                                'label':r'{s}_{\epsilon\delta}',
                                'loc':0.,
                                'scale':1e1,
                                'scale_fid':1e-2,
                                },
                's_e2': {'group':'stoch',
                                'label':r'{s}_{\epsilon^2}',
                                'loc':0.,
                                'scale':1e1,
                                'scale_fid':3e-3,
                                },
                's_ep': {'group':'stoch',
                                'label':r'{s}_{\epsilon\phi}',
                                'loc':0.,
                                'scale':1e5,
                                'scale_fid':1e2,
                                },
                'white_mesh': {'group':'init',
                                'label':r'{\delta}_\mathrm{w}',},
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
        return handlers.block(model, hide_fn=hide_fn)

    def predict(self, seed=42, samples:int|tuple|dict=None, batch_ndim=0, hide_base=True, hide_det=True, hide_samp=True, from_base=False):
        """
        Run model conditioned on samples.
        * If samples is None, return a single prediction.
        * If samples is an int or tuple, return a prediction of such shape.
        * If samples is a dict, return a prediction for each sample, assuming batch_ndim batch dimensions.
        """
        if isinstance(seed, int):
            seed = jr.key(seed)

        def single_prediction(seed, sample={}):
            # Optionally reparametrize base to sample params
            if from_base:
                sample = self.reparam(sample, inv=True) 
                # NOTE: deterministic sites have no effects with handlers.condition, but do with handlers.subsitute

            # Condition then block
            model = handlers.condition(self.model, data=sample)
            if hide_samp:
                model = handlers.block(model, hide=sample.keys())
            model = self._block_det(model, hide_base=hide_base, hide_det=hide_det)

            # Trace and return values
            tr = handlers.trace(handlers.seed(model, rng_seed=seed)).get_trace()
            return {k: v['value'] for k, v in tr.items()}

        if samples is None:
            return single_prediction(seed)
        
        elif isinstance(samples, (int, tuple)):
            if isinstance(samples, int):
                samples = (samples,)
            seed = jr.split(seed, samples)
            return nvmap(single_prediction, len(samples))(seed)
        
        elif isinstance(samples, dict):
            if len(samples) == 0:
                return {}
            else:
                # All item shapes should match on the first batch_ndim dimensions,
                # so take the first item shape
                shape = jnp.shape(next(iter(samples.values())))[:batch_ndim]
                seed = jr.split(seed, shape)
                return nvmap(single_prediction, len(shape))(seed, samples)
        


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

    def logdf_mesh(self, params={}, site='count_mesh'):
        """
        Element-wise likelihood and cumulative likelihood for ``site`` (default 'count_mesh').
        Return a tuple (logp(y_i | x), log F(y_i | x)), evaluated at latents x and observables y.

        For the per-voxel differentiation with respect to latent x (large y, possibly small x), use forward-mode:
            jax.jacfwd(lambda x: self.logdf_mesh({**y, **x})[0])(x)

        NOTE: logcdf requires the site distribution to implement log_cdf or cdf.
        """
        logpdfs_mesh, trace = compute_log_probs(self.model, (), {}, params, sum_log_prob=False)
        logpdf_mesh = logpdfs_mesh[site] # == d.log_prob(value)

        node = trace[site]
        d, value = node['fn'], node['value']
        logcdf_mesh = d.log_cdf(value) if hasattr(d, 'log_cdf') else jnp.log(d.cdf(value))
        return logpdf_mesh, logcdf_mesh

    def trace(self, seed):
        return handlers.trace(handlers.seed(self.model, rng_seed=seed)).get_trace()
    
    def seed(self, seed):
        self.model = handlers.seed(self.model, rng_seed=seed)

    def substitute(self, data={}, from_base=False):
        """
        Substitute random variables by their provided values, 
        optionally reparametrizing base values into sample values.
        Values are stored in attribute data.
        """
        if from_base:
            self.data |= data
            data = self.reparam(data, inv=True)
        self.data |= data
        self.model = handlers.condition(self.model, data=data)

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
            self.model = handlers.block(self.model, hide_fn=hide_fn, hide=hide, expose_types=expose_types, expose=expose)

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
        ysave(asdict(self), path)

    @classmethod
    def load(cls, path):
        return cls(**yload(path))









@dataclass
class FieldLevelModel(Model):
    """
    Field-level cosmological model, 
    with LPT and PM displacements, Lagrangian bias, and RSD.
    The relevant variables can be traced.

    Parameters
    ----------
    evolution : str
        Evolution model: 'kaiser', 'lpt', 'nbody'.
    a_obs : float
        Scale factor of observations.
    nbody_n_steps : int
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
    box_center : array_like
        Center of the box, i.e. observer position, in Mpc/h for curved-sky simulation.
        If None, no Redshift Space Distorsion is applied.
    poles : array_like of int
        Power spectrum poles to compute.
        Only used for 'powspec' observable.
    precond : str
        Preconditioning method: 'real', 'fourier', 'kaiser'.
    latents : dict
        Latent variables configuration.
    """
    # Mesh and box parameters
    final_shape:np.ndarray
    cell_length:float
    box_center:np.ndarray
    box_rotvec:np.ndarray
    k_cut:None|float
    # Init
    png_type:None|str
    # Evolution
    evolution:str
    nbody_a_start:float
    nbody_n_steps:int
    nbody_snapshots:int|list
    lpt_order:int
    paint_order:int
    paint_deconv:bool
    kernel_type:str
    init_oversamp:float
    evol_oversamp:float
    ptcl_oversamp:float
    paint_oversamp:float
    interlace_order:int
    # Observable
    observable:str
    poles:tuple
    a_obs:None|float
    curved_sky:bool
    ap_auto:bool
    register:None|float|Path
    n_rbins:None|int
    lik_type:str
    bias_type:str
    # Latents
    precond:str
    latents:dict

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.register, (str, Path.__base__)):
            # Load the register file overriding the corresponding config attributes.
            self.register = str(self.register) # cast to str to allow yaml saving
            reg = h5load(self.register)

            # Geometry and shapes (mandatory)
            for k in (
                'cell_length', 'box_center', 'box_rotvec',
                'init_oversamp', 'paint_oversamp', 
                ):
                setattr(self, k, reg[k])

            # Sky and painting (optional)
            for k in (
                'a_obs', 'curved_sky',
                'paint_order', 'interlace_order', 'paint_deconv', 'kernel_type'
                ):
                if k in reg:
                    setattr(self, k, reg[k])

            # Meshes (optionals except count_mesh)
            self.lin_kpow = reg.get('lin_kpow', None) # normalized to sigma8=1
            self.white_mesh = reg.get('white_mesh', reg.get('white_fake', None))
            self.selec_mesh = reg.get('selec_mesh', np.array(1.))
            self.mask_mesh = reg.get('mask_mesh', None)
            if self.lik_type=='fourier_gauss': 
                self.count_mesh = cgh2rg(jnp.fft.rfftn(reg['count_mesh']))
            else:
                self.count_mesh = mesh2masked(reg['count_mesh'], self.mask_mesh)
            self.final_shape = reg['count_mesh'].shape # original shape

            # Mean density and fiducial cosmology
            n_cells = self.count_mesh.size # works whether fourier or masked
            n_tracers = reg.get('n_tracers', self.count_mesh.sum())
            ngbar = n_tracers / (n_cells * self.cell_length**3) # in galaxy / (Mpc/h)^3
            self.latents = self.new_latents_from_loc(self.latents, reg['cosmo_fid'] | {'ngbars': ngbar}, update_prior=True)
        elif self.register is None:
            self.lin_kpow = None
            self.white_mesh = None
            self.count_mesh = None
            self.selec_mesh = np.array(1.)
            self.mask_mesh = None
        else:
            raise ValueError("register should be None, str, or Path.")

        # Geometry
        self.cell_length = float(self.cell_length)
        self.box_center = np.asarray(self.box_center)
        self.box_rotvec = np.asarray(self.box_rotvec)
        self.box_rot = Rotation.from_rotvec(self.box_rotvec)

        # Shapes
        self.final_shape = tuple(map(int, self.final_shape))
        self.box_size = np.multiply(self.final_shape, self.cell_length)
        self.init_shape = scale_shape(self.final_shape, self.init_oversamp)
        self.evol_shape = scale_shape(self.final_shape, self.evol_oversamp)
        self.ptcl_shape = scale_shape(self.final_shape, self.ptcl_oversamp)
        self.paint_shape = scale_shape(self.final_shape, self.paint_oversamp)

        # Scale cut
        self.k_funda = 2*np.pi / np.min(self.box_size)
        self.k_nyquist = np.pi * np.min(np.divide(self.final_shape, self.box_size))
        if self.k_cut == np.inf:
            self.cut_mask = None
        else:
            if self.k_cut is None:
                self.k_cut = float(self.k_nyquist)
            kvec = rfftk(self.init_shape, self.box_size) # in h/Mpc
            mask = top_hat(kvec, self.k_cut)
            self.cut_mask = np.array(cgh2rg(mask, norm="amp"), dtype=bool)


        # Imaging
        # TODO

        # Variables configuration
        self.latents = self._validate_latents()
        self.n_rbins, self.rmasked, self.redges, self.latents['ngbars'] = self._validate_rbins()
        self.groups = self._groups(base=True)
        self.groups_ = self._groups(base=False)
        self.labels = self._labels()

        # Fiducial quantities
        self.fiduc = self._fiduc()
        self.count_fid = self.fiduc['ngbars'].mean() * self.cell_length**3
        self.cosmo_fid = get_cosmology(**self.fiduc)
        _, a = los_scalefactor_mesh(self.box_center, self.box_rot, self.box_size, self.final_shape,
                                self.cosmo_fid, self.a_obs, self.curved_sky)
        self.a_fid = g2a(self.cosmo_fid, jnp.mean(a2g(self.cosmo_fid, a)))
        los = safe_div(self.box_center, np.linalg.norm(self.box_center))
        self.los_fid = self.box_rot.apply(los, inverse=True) # cell los
        self.selec_fid = (self.selec_mesh**2).mean()**.5 / self.selec_mesh.mean()


        # TODO: cell_length_init, cell_length_final, count_fid_init, count_fid_final 


    def __str__(self):
        out = ""
        out += f"# CONFIG\n"
        out += pformat(asdict(self), width=1)
        out += "\n\n# INFOS\n"
        out += f"box_size:       {self.box_size} Mpc/h\n"
        out += f"k_funda:        {self.k_funda:.5f} h/Mpc\n"
        out += f"k_nyquist:      {self.k_nyquist:.5f} h/Mpc\n"
        out += f"init_shape:     {self.init_shape} cell\n"
        out += f"evol_shape:     {self.evol_shape} cell\n"
        out += f"ptcl_shape:     {self.ptcl_shape} ptcl\n"
        out += f"paint_shape:    {self.paint_shape} cell\n"
        out += f"count_fid:      {self.count_fid:.3f} gxy/cell\n"
        out += f"a_fid:          {self.a_fid:.3f}\n"
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
        for g in ['cosmo', 'bias', 'png', 'stoch', 'ap', 'syst']:
            dic = self._sample(self.groups[g]) # sample               
            dic = samp2base(dic, self.latents, inv=False, temp=temp) # reparametrize
            tup += ({k: deterministic(k, v) for k, v in dic.items()},) # register base params
        cosmo, bias, png, stoch, ap, syst = tup
        cosmology = get_cosmology(**cosmo)

        # if 'b1' in bias:
        #     bias['b1'] = self.reparam_b1(bias['b1'], cosmo['sigma8'], eulerian=False)
        #     if 'b2' in bias:
        #         bias['b2'] = self.reparam_b2(bias['b2'], bias['b1'], cosmo['sigma8'], eulerian=False)

        # print(f"b1: {bias['b1']}, {cosmo['sigma8'] * b1_L2E(bias['b1'])}, ")  
        # print(f"b2: {bias['b2']}, {cosmo['sigma8']**2 * b2_L2E(bias['b2'], bias['b1'])}, ")  

        # Sample, reparametrize, and register initial conditions
        init = {}
        name_ = self.groups['init'][0]+'_' # 'white_mesh_'
        scale, transfer = self._precond_scale_and_transfer()

        if self.cut_mask is not None:
            samp = sample(name_, dist.Normal(0., scale[self.cut_mask])) # sample
            init[name_] = masked2mesh(samp, self.cut_mask)
        else:
            init[name_] = sample(name_, dist.Normal(0., scale)) # sample
        
        init = samp2base_mesh(init, self.precond, transfer=transfer, inv=False, temp=temp) # reparametrize
        # Limit fixed IC constant-folding through the model, which otherwise blows up GPU compilation.
        init = {k: lax.optimization_barrier(v) for k, v in init.items()}
        init = {k: deterministic(k, v) for k, v in init.items()} # register base params

        return cosmology, bias, png, stoch, ap, syst, init



    def evolve(self, params:tuple):
        cosmology, bias, png, stoch, ap, syst, init = params

        init_mesh = white2lin(cosmology, init['white_mesh'], self.init_shape, self.box_size, self.lin_kpow)
        init_mesh = chreshape(init_mesh, r2chshape(self.evol_shape))
        png = fNL_bias(png, bias, p=1., png_type=self.png_type)

        if self.evolution=='kaiser':
            los, a = los_scalefactor_mesh(self.box_center, self.box_rot, self.box_size, self.evol_shape,
                                cosmology, self.a_obs, self.curved_sky)
            cell_los = self.box_rot.apply(los, inverse=True) # cell los
            gxy_mesh = kaiser_model(cosmology, a, init_mesh, box_size=self.box_size, b1E=b1_L2E(bias['b1']), 
                                    fNL_bp=png['fNL_bp'], png_type=self.png_type, los=cell_los, kpow=self.lin_kpow)
            # NOTE: Kaiser model does not need any oversampling, even for curved-sky

            # print("kaiser:", gxy_mesh.mean(), gxy_mesh.std(), gxy_mesh.min(), gxy_mesh.max(), (gxy_mesh < 0).sum()/len(gxy_mesh.reshape(-1)))
            # gxy_mesh = jnp.abs(gxy_mesh)
            # gxy_mesh = 1 + 0.5 * jr.normal(jr.key(43), self.init_shape)
            # gxy_mesh = jnp.ones(self.init_shape)

            if self.ap_auto is not None:
                # Create regular grid of particles, and get their scale factors and line-of-sights
                # pos = sobol_pos(self.evol_shape, self.ptcl_shape, seed=43)
                print("ap_auto")
                pos = regular_pos(self.evol_shape, self.ptcl_shape)
                weights = read(pos, gxy_mesh, self.paint_order) ##########
                pos = cell2phys_pos(pos, self.box_center, self.box_rot, self.box_size, self.evol_shape)

                if self.ap_auto:
                    pos = ap_auto(pos, los, cosmology, self.cosmo_fid, self.curved_sky)
                    # pos, absdetjac = ap_auto_absdetjac(pos, los, cosmology, self.cosmo_fid, self.curved_sky)
                    # weights *= absdetjac
                else:
                    pos = ap_param(pos, los, ap, self.curved_sky)
                    # weights *= ap['alpha_iso']**3

                pos = phys2cell_pos(pos, self.box_center, self.box_rot, self.box_size, self.paint_shape)
                # gxy_mesh = interlace(pos, self.paint_shape, weights=lbe_weights, 
                #                      paint_order=self.paint_order, interlace_order=self.interlace_order, deconv=True)
                
                # weights = read2(pos, gxy_mesh, self.paint_order, 1/ap['alpha_iso'])
                # pos = regular_pos(self.mesh_shape, self.ptcl_shape)
                # gxy_mesh = paint2(pos, tuple(self.mesh_shape), weights, self.paint_order, ap['alpha_iso'])

                # gxy_mesh = jnp.fft.irfftn(interlace(pos, self.mesh_shape, weights, self.paint_order, self.interlace_order))
                # gxy_mesh = deconv_paint(gxy_mesh, order=self.paint_order); print("fin deconv") # NOTE: final deconvolution can amplify AP-induced high-frequencies.
                gxy_mesh *= np.divide(self.evol_shape, self.ptcl_shape).prod()
                
            if tuple(gxy_mesh.shape) != tuple(self.final_shape):
                gxy_mesh = jnp.fft.rfftn(gxy_mesh)
                gxy_mesh = chreshape(gxy_mesh, r2chshape(self.final_shape))
                gxy_mesh = jnp.fft.irfftn(gxy_mesh)

        else:
            # Create regular grid of particles, and get their scale factors
            pos = regular_pos(self.evol_shape, self.ptcl_shape)
            _, a = los_scalefactor_pos(pos, self.box_center, self.box_rot, self.box_size, self.evol_shape, 
                                    cosmology, self.a_obs, self.curved_sky)
            
            # if self.png_type is not None: #XXX XXX XXX XXX XXX
            #     # init_mesh = add_png(cosmology, png['fNL'], init_mesh, self.box_size)
            #     # init_mesh = chreshape(chreshape(init_mesh, r2chshape(self.init_shape)), r2chshape(self.evol_shape))
                
            #     init_mesh = chreshape(init_mesh, r2chshape(self.init_shape))
            #     init_mesh = add_png(cosmology, png['fNL'], init_mesh, self.box_size)
            #     init_mesh = chreshape(init_mesh, r2chshape(self.evol_shape))

            # Lagrangian bias expansion weights (based on initial particules positions)
            lbe_weights, dvel, phi = lagrangian_bias(cosmology, pos, a, self.box_size, init_mesh, bias, png,
                                                png_type=self.png_type, read_order=1)
            if self.bias_type == 'eulerian':
                phi_pos = read(pos, phi, order=1)
            
            if self.png_type is not None:
                init_mesh = add_png(cosmology, png['fNL'], init_mesh, self.box_size)
                init_mesh = chreshape(chreshape(init_mesh, r2chshape(self.init_shape)), r2chshape(self.evol_shape))

            if self.evolution=='lpt':
                # NOTE: lpt assumes given mesh is at a=1
                cosmology._workspace = {} # HACK: force recompute by jaxpm cosmo to get g2, f2 => TODO: add g2, f2 to jaxcosmo
                dpos, vel = lpt(cosmology, init_mesh, pos=pos, a=a, lpt_order=self.lpt_order, 
                                read_order=1, grad_fd=np.inf, lap_fd=np.inf)
                pos += dpos
                pos, vel = deterministic('lpt_ptcl', jnp.array((pos, vel)))

            elif self.evolution=='nbody':
                cosmology._workspace = {} # HACK: force recompute by jaxpm cosmo to get g2, f2 => TODO: add g2, f2 to jaxcosmo
                assert jnp.ndim(a) == 0, "N-body light-cone not implemented yet"
                pos, vel = nbody_bf(cosmology, init_mesh, pos=pos, a0=self.nbody_a_start, a1=a, n_steps=self.nbody_n_steps, 
                                    paint_order=self.paint_order, lpt_order=self.lpt_order, paint_deconv=False,
                                    grad_fd=np.inf, lap_fd=np.inf, snapshots=self.nbody_snapshots)
                # pos, vel = nbody_tsit5(cosmology, init_mesh, pos=pos, a0=self.nbody_a_start, a1=a,
                #                        grad_fd=np.inf, lap_fd=np.inf)

                pos, vel = deterministic('nbody_ptcl', jnp.array((pos, vel)))
                pos, vel = tree.map(lambda x: x[-1], (pos, vel))

            los, a = los_scalefactor_pos(pos, self.box_center, self.box_rot, self.box_size, self.evol_shape,
                                        cosmology, self.a_obs, self.curved_sky)
            pos = cell2phys_pos(pos, self.box_center, self.box_rot, self.box_size, self.evol_shape)      

            # RSD and Alcock-Paczynski effects
            dpos = rsd(cosmology, vel, los, a, self.box_rot, self.box_size, self.evol_shape, dvel)
            pos += dpos
            if self.ap_auto is not None:
                if self.ap_auto:
                    pos = ap_auto(pos, los, cosmology, self.cosmo_fid, self.curved_sky)
                    # pos, absdetjac = ap_auto_absdetjac(pos, los, cosmology, self.cosmo_fid, self.curved_sky)
                    # lbe_weights *= absdetjac
                else:
                    pos = ap_param(pos, los, ap, self.curved_sky)
                    # lbe_weights *= ap['alpha_iso']**3 

            # Paint weighted by Lagrangian bias expansion weights
            if self.bias_type == 'lagrangian':
                # pos = phys2cell_pos(pos, self.box_center, self.box_rot, self.box_size, self.final_shape)
                pos = phys2cell_pos(pos, self.box_center, self.box_rot, self.box_size, self.init_shape)
                # NOTE: final deconvolution can amplify AP-induced high-frequencies.
                # gxy_mesh = nufft(pos, self.final_shape, self.paint_shape, weights=lbe_weights, 
                gxy_mesh = nufft(pos, self.init_shape, self.paint_shape, weights=lbe_weights, 
                                paint_order=self.paint_order, interlace_order=self.interlace_order, 
                                kernel_type=self.kernel_type, paint_deconv=self.paint_deconv)
                # gxy_mesh *= np.divide(self.final_shape, self.ptcl_shape).prod() # jacobian of ptcl units to final units
                gxy_mesh *= np.divide(self.init_shape, self.ptcl_shape).prod()

                gxy_mesh = chreshape(gxy_mesh, r2chshape(self.paint_shape))
                gxy_mesh = jnp.fft.irfftn(gxy_mesh)

                # gxy_mesh = jnp.fft.rfftn(jnp.fft.irfftn(gxy_mesh) * (1 + bias['b1'] * jnp.fft.irfftn(init['init_mesh']))) ###XXX
                # gxy_mesh = jnp.fft.rfftn(1 + (jnp.fft.irfftn(gxy_mesh) - 1) * (1 + bias['b1'])) ###XXX
                # gxy_mesh = 1 + (gxy_mesh - 1) * (1 + bias['b1']) ###XXX

            elif self.bias_type == 'eulerian':
                pos = phys2cell_pos(pos, self.box_center, self.box_rot, self.box_size, self.init_shape)
                matter_mesh = nufft(pos, self.init_shape, self.paint_shape, weights=1., 
                                paint_order=self.paint_order, interlace_order=self.interlace_order, 
                                kernel_type=self.kernel_type, paint_deconv=self.paint_deconv)
                matter_mesh *= np.divide(self.paint_shape, self.ptcl_shape).prod()
                matter_mesh = chreshape(matter_mesh, r2chshape(self.paint_shape))
                print(jnp.fft.irfftn(matter_mesh).mean(), jnp.fft.irfftn(matter_mesh).std())

                phi_mesh = nufft(pos, self.init_shape, self.paint_shape, weights=phi_pos, 
                                paint_order=self.paint_order, interlace_order=self.interlace_order, 
                                kernel_type=self.kernel_type, paint_deconv=self.paint_deconv) # advected phi
                phi_mesh *= np.divide(self.paint_shape, self.ptcl_shape).prod()
                phi_mesh = chreshape(phi_mesh, r2chshape(self.paint_shape))
                print(jnp.fft.irfftn(phi_mesh).mean(), jnp.fft.irfftn(phi_mesh).std())

                gxy_mesh = eulerian_bias(matter_mesh, phi_mesh, self.box_size, bias, png, png_type=self.png_type)

        gxy_mesh = deterministic('gxy_mesh', gxy_mesh)
        # debug.print("lbe_weights: {i}", i=(lbe_weights.mean(), lbe_weights.std(), lbe_weights.min(), lbe_weights.max()))
        # debug.print("biased mesh: {i}", i=(biased_mesh.mean(), biased_mesh.std(), biased_mesh.min(), biased_mesh.max()))
        # debug.print("frac of weights < 0: {i}", i=(lbe_weights < 0).sum()/len(lbe_weights))
        return gxy_mesh, phi, stoch, syst # NOTE: mesh is 1+delta_obs


    def likelihood(self, params:tuple, temp=1.):
        """
        A likelihood for cosmological model.

        Return an observed mesh sampled from a location mesh with observational variance.
        """
        gxy_mesh, phi, stoch, syst = params

        if self.observable == 'field':
            # print("mesh", mesh.mean(), mesh.std(), mesh.min(), mesh.max())
            rcounts = syst['ngbars'] * self.cell_length**3
            # posit_fn = lambda x: jnp.maximum(x, 1e-9)
            # posit_fn = lambda x: jnp.log(1 + jnp.exp(x))
            posit_fn = lambda x: jnp.abs(x)

            count_mesh = jnp.fft.irfftn(chreshape(jnp.fft.rfftn(gxy_mesh * self.selec_mesh), r2chshape(self.final_shape)))
            count_mesh = mesh2masked(count_mesh, self.mask_mesh)
            count_mesh = set_radial_count(count_mesh, self.rmasked, self.redges, rcounts)
            # count_mesh = posit_fn(count_mesh)

            if self.selec_mesh.ndim == 3:
                selec_mesh = jnp.fft.irfftn(chreshape(jnp.fft.rfftn(self.selec_mesh), r2chshape(self.final_shape)))
                selec_mesh = mesh2masked(selec_mesh, self.mask_mesh)
                selec_mesh = set_radial_count(selec_mesh, self.rmasked, self.redges, rcounts)
                selec_mesh = posit_fn(selec_mesh)
            else:
                selec_mesh = jnp.mean(rcounts)

            if self.png_type is not None:
                phi = jnp.fft.irfftn(chreshape(jnp.fft.rfftn(phi), r2chshape(self.final_shape)))


            if self.lik_type == 'poisson':
                count_mesh = sample('count_mesh', dist.Poisson(posit_fn(count_mesh)**(1 / temp)))

            elif self.lik_type == 'fourier_gauss':
                assert self.mask_mesh is None, "Fourier likelihood not implemented for cut-sky."
                kvec = rfftk(self.final_shape, self.box_size) # in h/Mpc
                kmesh = sum(ki**2 for ki in kvec)**.5
                mumesh = sum(ki * losi for ki, losi in zip(kvec, self.los_fid))
                mumesh = safe_div(mumesh, kmesh)
                
                scale = posit_fn(stoch['s_e'] + stoch['s_k2e'] * kmesh**2 + stoch['s_kmu2e'] * (kmesh * mumesh)**2)
                scale *= selec_mesh**.5 * temp**.5
                scale = cgh2rg(scale, norm="amp")
                count_mesh = cgh2rg(jnp.fft.rfftn(count_mesh))
                count_mesh = sample('count_mesh', dist.Normal(count_mesh, scale))
            
            elif self.lik_type == 'quad_gauss':
                delta = count_mesh / selec_mesh - 1
                # delta = jnp.fft.irfftn(chreshape(jnp.fft.rfftn(gxy_mesh), r2chshape(self.final_shape))) - 1
                # debug.print("delta down {i}", i=(delta.mean(), delta.std(), delta.min(), delta.max()))
                
                # scale1 = posit_fn(stoch['s_e'] + stoch['s_ed'] * delta) + 1e-9
                scale1 = posit_fn(stoch['s_e'] + stoch['s_ed'] * delta + stoch['s_ep'] * phi) + 1e-9
                scale1 *= selec_mesh**.5 * temp**.5
                scale2 = stoch['s_e2']
                scale2 *= selec_mesh**.5

                # NOTE: QuadGaussian has a variable-dependent bounded support
                # that can make evaluations venture outside easily. 
                count_mesh = sample("count_mesh", QuadGaussian(count_mesh, scale1, scale2))

            elif self.lik_type == 'two_quad_gauss':
                delta = count_mesh / selec_mesh - 1
                scale1 = posit_fn(stoch['s_e'] + stoch['s_ed'] * delta + stoch['s_ep'] * phi) + 1e-9
                scale1 *= selec_mesh**.5 * temp**.5
                scale2 = stoch['s_e2']
                scale2 *= selec_mesh**.5
                count_mesh = sample("count_mesh", TwoQuadGaussian(count_mesh, scale1, scale2))
                
            elif self.lik_type == 'shash':
                delta = count_mesh / selec_mesh - 1
                # delta = jnp.fft.irfftn(chreshape(jnp.fft.rfftn(gxy_mesh), r2chshape(self.final_shape))) - 1
                # debug.print("delta down {i}", i=(delta.mean(), delta.std(), delta.min(), delta.max()))
                      
                # scale1 = posit_fn(stoch['s_e'] + stoch['s_ed'] * delta) + 1e-9
                scale1 = posit_fn(stoch['s_e'] + stoch['s_ed'] * delta + stoch['s_ep'] * phi) + 1e-9
                scale1 *= selec_mesh**.5 * temp**.5
                scale2 = stoch['s_e2']
                scale2 *= selec_mesh**.5 
            
                # NOTE: Local moment-match to QuadGaussian(count_mesh, scale1, scale2) 
                # (mean/std are exact and skew/tail match to first order in scale2/scale1):
                #   mean = count_mesh                            (exact)
                #   std  = (scale1**2 + 2*scale2**2)**.5         (exact: QuadGaussian variance)
                #   skewness   = 3.540 * (scale2/scale1)         (first-order skew match)
                #   tailweight = 1 + 5.884 * (scale2/scale1)**2  (first-order excess-kurtosis match)
                ratio = scale2 / scale1
                count_mesh = sample("count_mesh", SinhArcsinh(count_mesh,
                                                (scale1**2 + 2 * scale2**2)**.5,
                                                3.540 * ratio,
                                                1 + 5.884 * ratio**2))
            return count_mesh 

        # elif self.obs == 'pk':
        #     # Anisotropic power spectrum covariance, cf. [Grieb+2016](http://arxiv.org/abs/1509.04293)
        #     multipoles = np.atleast_1d(self.lik_config['multipoles'])
        #     sli_multip = slice(1,1+len(multipoles))
        #     loc_pk, Nk = get_pk_fn(final_shape, box_size, multipoles=multipoles, kcount=True, gxy_density=self.gxy_density)(mesh)
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
        groups = ['cosmo','bias','png','stoch','ap','syst','init'] # fixed order
        key = tuple([k if inv else k+'_'] for k in groups) 
        key += tuple([['*'] + ['~'+k if inv else '~'+k+'_' for k in groups]])
        params_ = Chains(params_, self.groups | self.groups_).get(key) # use chain querying
        cosmo_, bias_, png_, stoch_, ap_, syst_, init, rest = (q.data for q in params_)

        # All params except init
        cosmo = samp2base(cosmo_, self.latents, inv=inv, temp=temp)
        bias = samp2base(bias_, self.latents, inv=inv, temp=temp)
        png = samp2base(png_, self.latents, inv=inv, temp=temp)
        stoch = samp2base(stoch_, self.latents, inv=inv, temp=temp)
        ap = samp2base(ap_, self.latents, inv=inv, temp=temp)
        syst = samp2base(syst_, self.latents, inv=inv, temp=temp)

        # Initial conditions
        if len(init) > 0:
            _, transfer = self._precond_scale_and_transfer()

            if inv and not fourier:
                init = tree.map(jnp.fft.rfftn, init)
            if not inv and self.cut_mask is not None:       
                init = tree.map(lambda x: masked2mesh(x, self.cut_mask), init)

            init = samp2base_mesh(init, self.precond, transfer=transfer, inv=inv, temp=temp)
            
            if inv and self.cut_mask is not None:       
                init = tree.map(lambda x: mesh2masked(x, self.cut_mask), init)
            if not inv and not fourier:
                init = tree.map(jnp.fft.irfftn, init)

        out = cosmo | bias | png | stoch | ap | syst | init
        out = {k:v for k,v in out.items() if (k[:-1] if inv else k+'_') in params}
        rest = {k:v for k,v in rest.items() if k in params} # do not return data
        out = rest | out # possibly update rest
        return out

        
    def reparam_b1(self, b1, sigma8, eulerian=False, inv=False):
        """
        Transform sigma8-scaled b1 parameter into unscaled b1 parameter.
        """
        alpha = sigma8 / self.fiduc['sigma8']

        if not eulerian:
            b1 = b1_L2E(b1)
        if inv:
            b1 *= alpha
        else:
            b1 /= alpha
        if not eulerian:
            b1 = b1_E2L(b1)
        return b1

    def reparam_b2(self, b2, b1L, sigma8, eulerian=False, inv=False):
        """
        Transform sigma8-scaled b2 parameter into unscaled b2 parameter.
        """
        alpha = sigma8 / self.fiduc['sigma8']

        if not eulerian:
            b2 = b2_L2E(b2, b1L)
        if inv:
            b2 *= alpha**2
        else:
            b2 /= alpha**2
        if not eulerian:
            b2 = b2_E2L(b2, b1L)
        return b2

    def reparam_bias(self, params:dict, eulerian=False, inv=False):
        """
        Transform sigma8-scaled bias parameters into unscaled bias parameters.
        Consequently, params must contain 'sigma8' key.
        """
        out = self.data | params
        sigma8 = out['sigma8']
        if 'b1' in out:
            b1_ = out['b1']
            b1 = self.reparam_b1(b1_, sigma8, eulerian=eulerian, inv=inv)
            out['b1'] = b1

            if 'b2' in out:
                b1u = b1_ if inv else b1 # unscaled b1
                b1L = b1_E2L(b1u) if eulerian else b1u # unscaled b1L
                out['b2'] = self.reparam_b2(out['b2'], b1L, sigma8, eulerian=eulerian, inv=inv)
        
        # Do not return data, and potentially cast into Chains
        out = params | {k:out[k] for k in params} 
        return out

    ###########
    # Getters #
    ###########   
    def _validate_latents(self):
        """
        Validate latents config.
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

    def _validate_rbins(self):
        """
        Validate radial density and setup radial bins quantities. 
        """
        rmesh = np.array(self.radius_mesh())
        rmasked = mesh2masked(rmesh, self.mask_mesh)
        rmin, rmax = rmasked.min(), rmasked.max()
        dr = 3**.5 * self.cell_length # minimum dr to guarantee connected shell bins

        n_rbins = max(int((rmax - rmin) / dr), 1) if self.n_rbins is None else self.n_rbins
        redges = np.linspace(rmin - dr/1000, rmax + dr/1000, n_rbins + 1)

        ngbars_conf = self.latents['ngbars'].copy()
        for attr in ['loc','scale','loc_fid','scale_fid','low','high']:
            if attr in ngbars_conf:
                ngbars_conf[attr] = np.broadcast_to(ngbars_conf[attr], n_rbins)
        return n_rbins, rmasked, redges, ngbars_conf

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

            if None not in np.array(loc):
                if np.all(low == -jnp.inf) and np.all(high == jnp.inf):
                    samp = sample(name+'_', dist.Normal((loc - loc_fid) / scale_fid, scale / scale_fid))
                else:
                    samp = sample(name+'_', DetruncTruncNorm(loc, scale, low, high, loc_fid, scale_fid))
            else:
                samp = sample(name+'_', DetruncUnif(low, high, loc_fid, scale_fid))
            dic[name+'_'] = samp
        return dic

    def _precond_scale_and_transfer(self):
        """
        Return scale and transfer fields for white field preconditioning.
        """
        if self.precond in ['real', 'fourier']:
            scale = jnp.ones(self.init_shape)

        elif self.precond=='kaiser':
            b1E_fid = b1_L2E(self.fiduc['b1'])
            boost_fid = kaiser_boost(self.cosmo_fid, self.a_fid, self.init_shape, self.box_size, 
                                     b1E_fid, los=self.los_fid)
            pmesh_fid = lin_power_mesh(self.cosmo_fid, self.init_shape, self.box_size)
            pmesh_fid *= np.divide(self.init_shape, self.box_size).prod() # power in cell units
            var_fid = self.fiduc['s_e'] / (self.count_fid * self.selec_fid)
            scale = (1 + boost_fid**2 / var_fid * pmesh_fid)**.5

        else:
            raise ValueError(f"Unknown preconditioning type: {self.precond}")
        
        transfer = np.divide(self.init_shape, self.box_size).prod()**.5 / scale # transfer to unit-power white noise
        scale = cgh2rg(scale, norm="amp")
        return scale, transfer

    # def _precond_scale_and_transfer(self, cosmo:Cosmology, bias, stoch):
    #     """
    #     Return scale and transfer fields for linear matter field preconditioning.
    #     """
    #     if self.init_type == 'eh':
    #         pmesh = lin_power_mesh(cosmo, self.init_shape, self.box_size, kpow=None)
    #     elif self.init_type == 'init_kpow':
    #         pmesh = lin_power_mesh(cosmo, self.init_shape, self.box_size, kpow=self.init_kpow)
    #         # NOTE: init_kpow is normalized to sigma8=1, and scaled by cosmo sigma8**2
    #     else:
    #         raise ValueError(f"Unknown initial condition type: {self.init_type}")
        
    #     if self.precond in ['real', 'fourier']:
    #         scale = jnp.ones(self.init_shape)
    #         transfer = pmesh**.5

    #     elif self.precond=='kaiser':
    #         b1E_fid = b1_L2E(self.fiduc['b1'])
    #         boost_fid = kaiser_boost(self.cosmo_fid, self.a_fid, self.init_shape, self.box_size, b1E_fid, los=self.los_fid)
    #         pmesh_fid = lin_power_mesh(self.cosmo_fid, self.init_shape, self.box_size)
    #         var_fid = self.fiduc['s_e'] / (self.count_fid * self.selec_fid)

    #         scale = (1 + boost_fid**2 / var_fid * pmesh_fid)**.5
    #         transfer = pmesh**.5 / scale
    #         scale = cgh2rg(scale, norm="amp")
        
    #     elif self.precond=='kaiser_dyn':
    #         b1E = b1_L2E(bias['b1'])
    #         boost = kaiser_boost(cosmo, self.a_fid, self.init_shape, self.box_size, b1E, los=self.los_fid)
    #         var = stoch['s_e'] / (self.count_fid * self.selec_fid)

    #         scale = (1 + boost**2 / var * pmesh)**.5
    #         transfer = pmesh**.5 / scale
    #         scale = cgh2rg(scale, norm="amp")

    #     else:
    #         raise ValueError(f"Unknown preconditioning type: {self.precond}")
        
    #     return scale, transfer
    
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
    
    def _fiduc(self):
        """
        Return fiducial location values from latents config and meshes.
        """
        fiduc = {k:v['loc_fid'] for k,v in self.latents.items() if 'loc_fid' in v}
        # if self.white_mesh is not None:
        #     fiduc['white_mesh'] = self.white_mesh
        # if self.count_mesh is not None:
        #     fiduc['count_mesh'] = self.count_mesh
        return fiduc
    
    # @property
    # def rmasked(self):
    #     rmesh = np.array(radius_mesh(self.box_center, self.box_rot, self.box_size, self.final_shape, self.curved_sky))
    #     return mesh2masked(rmesh, self.mask)
    
    @classmethod
    def new_latents_from_loc(cls, latents, loc:dict, update_prior:bool=False):
        """
        Return a new latents config wih updated fiducial location based on given location dict.
        If `update_prior` is True, also update prior location (if it exists).
        """
        new = {}
        for name, conf in latents.items():
            new[name] = conf.copy()
            if name in loc:
                new[name]['loc_fid'] = loc[name]
                if update_prior and 'loc' in conf:
                    new[name]['loc'] = loc[name]
        return new


    ########
    # Data #
    ########
    def pos_mesh(self, shape=None):
        if shape is None:
            shape = self.final_shape
        return pos_mesh(self.box_center, self.box_rot, self.box_size, shape)

    def radius_mesh(self, shape=None):
        if shape is None:
            shape = self.final_shape
        return radius_mesh(self.box_center, self.box_rot, self.box_size, shape, self.curved_sky)
    
    def mesh2masked(self, mesh):
        return mesh2masked(mesh, self.mask_mesh)
    
    def masked2mesh(self, mesh):
        return masked2mesh(mesh, self.mask_mesh)

    def white2lin(self, cosmo:Cosmology, mesh):
        return white2lin(cosmo, mesh, self.init_shape, self.box_size, self.lin_kpow)

    def lin2white(self, cosmo:Cosmology, mesh):
        return lin2white(cosmo, mesh, self.init_shape, self.box_size, self.lin_kpow)

    def count2delta(self, mesh):
        """
        Count mesh to delta mesh, by imposing global integral constraint.
        """
        if self.lik_type == 'fourier_gauss':
            mesh = jnp.fft.irfftn(rg2cgh(mesh))
        else:
            mesh = self.masked2mesh(mesh)

        if self.selec_mesh.ndim == 3 and self.selec_mesh.shape != mesh.shape:
            selec_mesh = jnp.fft.irfftn(chreshape(jnp.fft.rfftn(self.selec_mesh), r2chshape(mesh.shape)))
            selec_mesh = self.masked2mesh(self.mesh2masked(selec_mesh))
        else:
            selec_mesh = np.asarray(self.selec_mesh)
        return count2delta(mesh, selec_mesh)
    
    @classmethod
    def register_catalog(cls, cell_budget:float, cosmo_fid:Cosmology, data, random=None,
                         box_size=None, box_center=None, box_rotvec=None, a_obs=None, los=None,
                         padding:float=0., init_oversamp:float=3/2, paint_oversamp:float=7/4,
                         paint_order:int=2, interlace_order:int=2, paint_deconv:bool=True, kernel_type:str='rectangular'):
        """
        Register a particle catalog into the meshes and metadata an inference-ready model needs,
        handling both sky and box catalogs (no model instance required):

        * cut-sky  (`random` given): `data` and `random` are (RA, DEC, Z, WEIGHT) dict-likes; the
          geometry is fit to the randoms, a selection and footprint mask are painted from the
          randoms, and the count is painted from the data. 
          `a_obs=None` (light-cone), `los=None` (curved-sky).
        * full-sky (`random` None): `data` is cartesian 'pos' (and optional 'vel',
          'WEIGHT') dict-like, or an iterable of such dicts (streamed, e.g. abacus matter over many files);
          `box_size` gives the periodic box, and there is no selection/mask. If 'vel' is present,
          RSD is applied at `a_obs` along the line of sight `los=(0,0,1)`.

        Return a register dict (geometry, meshes, weighted counts, fiducial cosmology, painting
        params) ready to be saved -- together with init data -- into a register HDF5 file via
        `montecosmo.utils.h5save` (None values are dropped, signaling "absent" to the loader).
        """
        cut_sky = random is not None
        # Geometry: fit to the randoms (cut-sky) or set the periodic box (full-sky)
        if cut_sky:
            assert a_obs is None and los is None, "For cut-sky catalog, a_obs and los must be None (light-cone, curved-sky)"
            curved_sky = True
            final_shape, cell_length, box_center, box_rotvec = cutsky2config(
                random, cosmo_fid, cell_budget, padding, box_size=box_size, box_center=box_center, box_rotvec=box_rotvec)
        else:
            assert a_obs is not None and los is not None and box_size is not None and box_center is not None,\
            "For full-sky catalog, a_obs, los, box_size, and box_center must be provided"
            box_rotvec = np.zeros(3) if box_rotvec is None else np.asarray(box_rotvec)
            final_shape, cell_length = get_mesh_shape(box_size, cell_budget, padding=0.)
            curved_sky = False
        paint = dict(paint_order=paint_order, interlace_order=interlace_order, paint_deconv=paint_deconv)
        box_size = np.multiply(final_shape, cell_length) # box_size update due to rounding and padding
        init_shape = scale_shape(final_shape, init_oversamp)
        paint_shape = scale_shape(final_shape, paint_oversamp)

        # Selection, mask, and count
        if cut_sky:
            selec_mesh, mask_mesh = cutsky2selection(
                random, cosmo_fid, mask_shape=final_shape, selec_shape=init_shape, paint_shape=paint_shape,
                box_size=box_size, box_center=box_center, box_rotvec=box_rotvec, **paint)
            selec_mesh = jnp.fft.irfftn(chreshape(jnp.fft.rfftn(selec_mesh), r2chshape(paint_shape)))
            selec_mesh, mask_mesh = np.asarray(selec_mesh), np.asarray(mask_mesh)

            count_mesh = cutsky2count(
                data, cosmo_fid, final_shape, paint_shape,
                box_size=box_size, box_center=box_center, box_rotvec=box_rotvec, **paint)
            n_tracers, n_randoms = float(np.sum(data['WEIGHT'])), float(np.sum(random['WEIGHT']))
        else:
            count_mesh = fullsky2count(
                data, cosmo_fid, a_obs, los=los,
                box_size=box_size, box_center=box_center, box_rotvec=box_rotvec,
                final_shape=final_shape, paint_shape=paint_shape, **paint)
            box_center = np.multiply(los, a2chi(cosmo_fid, a_obs)) # real box_center, coherent with the los
            n_tracers = float(count_mesh.sum())
            selec_mesh = mask_mesh = n_randoms = None

        return {
            # Geometry and painting (mandatory)
            'cell_length': float(cell_length), 'box_center': np.asarray(box_center), 'box_rotvec': np.asarray(box_rotvec),
            'init_oversamp': float(init_oversamp), 'paint_oversamp': float(paint_oversamp),
            'cosmo_fid': {'Omega_m': float(cosmo_fid.Omega_m), 'sigma8': float(cosmo_fid.sigma8)},
            'count_mesh': np.asarray(count_mesh),
            # Optional (None entries are dropped by h5save, signaling "absent")
            'selec_mesh': None if selec_mesh is None else np.asarray(selec_mesh),
            'mask_mesh': None if mask_mesh is None else np.asarray(mask_mesh),
            'n_tracers': n_tracers, 'n_randoms': n_randoms,
            'a_obs': a_obs, 'curved_sky': curved_sky,
            'paint_order': int(paint_order), 'interlace_order': int(interlace_order),
            'paint_deconv': bool(paint_deconv), 'kernel_type': kernel_type,
            'cell_budget': float(cell_budget), 'padding': float(padding),
        }




    ###########
    # Metrics #
    ###########
    def spectrum(self, mesh0, mesh1=None, ells:int|list=0, kedges:int|float|list=None, include_corners=True):
        return spectrum(mesh0, mesh1=mesh1, box_size=self.box_size, box_center=self.box_center,
                        ells=ells, kedges=kedges, include_corners=include_corners)

    def powtranscoh(self, mesh0, mesh1, kedges:int|float|list=None, include_corners=True):
        """
        Return wavenumber, power spectrum, transfer function, and coherence of two meshes.\\
        Precisely return a tuple (k, pow1,  (pow1 / pow0)^.5, pow01 / (pow0 * pow1)^.5).
        """
        return powtranscoh(mesh0, mesh1, box_size=self.box_size, kedges=kedges, include_corners=include_corners)
    
    def mse_radius(self, mesh0, mesh1, cell_length=None, redges:int|float|list=None, aggr_fn=None, from_masked=True):
        if cell_length is None:
            cell_length = self.cell_length
        if not from_masked:
            mesh0 = mesh2masked(mesh0, self.mask_mesh)
            mesh1 = mesh2masked(mesh1, self.mask_mesh)

        return mse_radius(mesh0, mesh1, self.rmasked, cell_length, redges=redges, aggr_fn=aggr_fn)

    def mse_value(self, mesh0, mesh1, cell_length=None, vedges:int|float|list=50, min_count=None, aggr_fn=None):
        if cell_length is None:
            cell_length = self.cell_length
            
        return mse_value(mesh0, mesh1, cell_length, vedges=vedges, min_count=min_count, aggr_fn=aggr_fn)

    def mse_wave(self, mesh0, mesh1, kedges:int|float|list=None, include_corners=True):            
        return mse_wave(mesh0, mesh1, self.box_size, kedges=kedges, include_corners=include_corners)

    def distr_radial(self, mesh, cell_length=None, redges:int|float|list=None, aggr_fn=None, from_masked=True):
        if cell_length is None:
            cell_length = self.cell_length
        if not from_masked:
            mesh = mesh2masked(mesh, self.mask_mesh)

        return distr_radial(mesh, self.rmasked, cell_length, redges=redges, aggr_fn=aggr_fn)

    def distr_angular(self):
        return distr_angular()

    ##################
    # Chains process #
    ##################
    def load_runs(self, path:str, start:int, end:int, transforms=None, batch_ndim=2) -> Chains:
        return Chains.load_runs(path, start, end, transforms, 
                                groups=self.groups | self.groups_, labels=self.labels, batch_ndim=batch_ndim)

    def reparam_chains(self, chains:Chains, fourier=False, inv=False, batch_ndim=2) -> Chains:
        chains = chains.copy()
        chains.data = nvmap(partial(self.reparam, fourier=fourier, inv=inv), batch_ndim)(chains.data)
        return chains
    
    # def predict_chains(self, chains:Chains, seed=42, batch_ndim=2, 
    #                    hide_base=True, hide_det=True, hide_samp=True, from_base=False) -> Chains:
    #     chains = chains.copy() # TODO: tree.map
    #     chains.data = self.predict(seed=seed, samples=chains.data, batch_ndim=batch_ndim, 
    #                                hide_base=hide_base, hide_det=hide_det, hide_samp=hide_samp, from_base=from_base)
    #     return chains
    
    def powtranscoh_chains(self, chains:Chains, mesh0, names:str|list=[], 
                           kedges:int|float|list=None, batch_ndim=2) -> Chains:
        """
        Return wavenumber, power spectrum, transfer function, and coherence 
        of meshes in chains compared to a reference mesh.
        Precisely, return under the key "kptc" a tuple 
        (k, pow1, (pow1 / pow0)^.5, pow01 / (pow0 * pow1)^.5).
        """
        chains = chains.copy()
        names = np.atleast_1d(names)
        fn = nvmap(lambda x: self.powtranscoh(mesh0, x, kedges=kedges), batch_ndim)
        for name in names:
            chains.data[f'kptc_{name}'] = fn(chains.data[name])
        return chains
    
    def kaiser_post(self, seed, base=False, temp=1., scale_field=1.):
        """
        Return posterior on init field given obs field, 
        as well as latent parameters fiducial values that are not in data.
        For MCMC initilization purposes.
        Assume a Kaiser linear Gaussian model.
        """
        delta_obs = self.count2delta(self.count_mesh)
        delta_obs = jnp.fft.rfftn(delta_obs)
        
        # Reshape in Fourier domain in case observed field shape != initial field shape
        delta_obs = chreshape(delta_obs, r2chshape(self.init_shape))

        b1E_fid = b1_L2E(self.fiduc['b1'])
        var_fid = self.fiduc['s_e'] / (self.count_fid * self.selec_fid)
        means, stds = kaiser_posterior(delta_obs, self.cosmo_fid, self.a_fid, self.box_size, 
                                       var_noise=var_fid, b1E=b1E_fid, los=self.los_fid)
        
        # HACK: rg2cgh has absurd problem with vmaped random arrays in CUDA11, so rely on rg2cgh2 until fully moved to CUDA12.
        # post_mesh = rg2cgh2(jr.normal(seed, ch2rshape(means.shape)))
        post_mesh = rg2cgh(jr.normal(seed, ch2rshape(means.shape)))
        post_mesh = temp**.5 * stds * post_mesh + means 
        post_mesh = lin2white(self.cosmo_fid, post_mesh, self.init_shape, self.box_size)
        # NOTE: scaling down the field is recommended when the Kaiser posterior approximation becomes less valid
        # because many high-wavevector amplitudes can be set to high.
        post_mesh *= scale_field 

        # Return starting values for all latent parameters except those in data
        start_params = {k: self.fiduc[k] for k in self.fiduc.keys() - self.data.keys()}
        start_params |= {k: post_mesh for k in {'white_mesh'} - self.data.keys()}
        if base:
            return start_params
        else:
            return self.reparam(start_params, inv=True)

    

