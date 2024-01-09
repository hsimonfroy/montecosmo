import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import sample, deterministic
import numpy as np

from montecosmo.bricks import cosmo_prior, linear_pk_interp, linear_field, lagrangian_bias, rsd
from jaxpm.pm import lpt
from jaxpm.painting import cic_paint


model_kwargs={
            # Mesh and box parameters
            'mesh_size':np.array([64, 64, 64]), 
            'box_size':np.array([640, 640, 640]), # in Mpc/h (aim for cell lengths between 1 and 10 Mpc/h)
            # Scale factors
            'scale_factor_lpt':0.1, 
            'scale_factor_obs':0.5,
            # Galaxies
            'galaxy_density':1e-3, # in galaxy / (Mpc/h)^3
            # Debugging
            'trace_reparam':True, 
            'trace_deterministic':False}


def forward_model(mesh_size=np.array([64, 64, 64]), 
                  box_size=np.array([640, 640, 640]), # in Mpc/h (aim for cell lengths between 1 and 10 Mpc/h)
                  scale_factor_lpt=0.1, 
                  scale_factor_obs=0.5, 
                  galaxy_density=1e-3, # in galaxy / (Mpc/h)^3
                  trace_reparam=True, 
                  trace_deterministic=False):
    """
    A cosmological forward model.
    The relevant variables can be traced.
    """
    # Sample cosmology
    cosmology = cosmo_prior(trace_reparam)

    # Sample initial conditions
    pk_fn = linear_pk_interp(cosmology, n_interp=256)
    init_mesh = linear_field(mesh_size, box_size, pk_fn, trace_reparam)

    # Create regular grid of particles
    x_part = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in mesh_size]),axis=-1).reshape([-1,3])

    # Compute Lagrangian bias expansion weights
    lbe_weights = lagrangian_bias(cosmology, scale_factor_obs, init_mesh, x_part, box_size)

    # LPT displacement
    cosmology._workspace = {}  # HACK: temporary fix
    dx, p_part, f = lpt(cosmology, init_mesh, x_part, a=scale_factor_lpt)
    # NOTE: lpt supposes given mesh follows linear pk at a=1, 
    # and correct by growth factor to get forces at wanted scale factor
    x_part = x_part + dx

    if trace_deterministic: 
        x_part = deterministic('lpt_part', x_part)

    # XXX: here N-body displacement
    # x_part, v_part = ... PM(scale_factor_lpt -> scale_factor_obs)

    # RSD displacement
    dx_rsd = rsd(cosmology, scale_factor_obs, p_part)
    x_part = x_part + dx_rsd

    if trace_deterministic: 
        x_part = deterministic('rsd_part', x_part)

    # CIC paint weighted by Lagrangian bias expansion
    biased_mesh = cic_paint(jnp.zeros(mesh_size), x_part, lbe_weights)

    if trace_deterministic: 
        biased_mesh = deterministic('biased_mesh', biased_mesh)

    # Observe
    gxy_intens_mesh = biased_mesh * (galaxy_density * box_size.prod() / mesh_size.prod())
    obs_mesh = sample('obs_mesh', dist.Poisson(gxy_intens_mesh))
    return obs_mesh



def lpt_model(mesh_size=np.array([64, 64, 64]), 
              box_size=np.array([640, 640, 640]), # in Mpc/h (aim for cell lengths between 1 and 10 Mpc/h)
              scale_factor_lpt=0.1, 
              scale_factor_obs=0.5, 
              galaxy_density=1e-3, # in galaxy / (Mpc/h)^3
              trace_reparam=True, 
              trace_deterministic=False):
    """
    A simple cosmological model, with LPT displacement and Poisson observation.
    The relevant variables can be traced.
    """
    # Sample cosmology
    cosmology = cosmo_prior(trace_reparam)

    # Sample initial conditions
    pk_fn = linear_pk_interp(cosmology, n_interp=256)
    init_mesh = linear_field(mesh_size, box_size, pk_fn, trace_reparam)
    
    if trace_deterministic: 
       init_mesh = deterministic('init_mesh', init_mesh)

    # Create regular grid of particles
    particles_pos = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in mesh_size]),axis=-1).reshape([-1,3])

    # Initial displacement with LPT
    cosmology._workspace = {}  # FIXME: this a temporary fix
    dx, p, f = lpt(cosmology, init_mesh, particles_pos, a=scale_factor_lpt)
    # NOTE: lpt supposes given mesh follows linear pk at a=1, 
    # and correct by growth factor to get forces at wanted scale factor
    particles_pos = particles_pos + dx

    # Cloud In Cell painting
    lpt_mesh = cic_paint(jnp.zeros(mesh_size), particles_pos)
    
    if trace_deterministic: 
       lpt_mesh = deterministic('lpt_mesh', lpt_mesh)

    # Observe
    ## Direct observation
    # obs_mesh = sample('obs_mesh', dist.Delta(lpt_mesh))
    ## Normal noise 
    # obs_mesh = sample('obs_mesh', dist.Normal(lpt_mesh, 0.1))
    ## Poisson noise
    gxy_intens_mesh = lpt_mesh * (galaxy_density * box_size.prod() / mesh_size.prod())
    obs_mesh = sample('obs_mesh', dist.Poisson(gxy_intens_mesh))
    return obs_mesh