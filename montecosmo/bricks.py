from functools import partial
import numpy as np

from jax import numpy as jnp, tree, debug
import jax_cosmo as jc
from jax_cosmo import Cosmology
from jaxpm.painting import cic_read
from jaxpm.growth import growth_factor, growth_rate

from montecosmo.utils import std2trunc, trunc2std, rg2cgh, cgh2rg, ch2rshape, r2chshape, safe_div
from montecosmo.nbody import rfftk, invlaplace_kernel


# Planck 2015 paper XIII Table 4 final column (best fit)
Planck15 = partial(Cosmology,
    Omega_c=0.2589,
    Omega_b=0.04860,
    Omega_k=0.0,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    w0=-1.0,
    wa=0.0,)

# Planck 2018 paper VI Table 2 final column (best fit)
Planck18 = partial(Cosmology,
    # Omega_m = 0.3111
    Omega_c=0.2607,
    Omega_b=0.0490,
    Omega_k=0.0,
    h=0.6766,
    n_s=0.9665,
    sigma8=0.8102,
    w0=-1.0,
    wa=0.0,)


def lin_power_interp(cosmo=Cosmology, a=1., n_interp=256):
    """
    Return a light emulation of the linear matter power spectrum.
    """
    k = jnp.logspace(-4, 1, n_interp)
    logpk = jnp.log(jc.power.linear_matter_power(cosmo, k, a=a))
    # Interpolate in semilogy space with logspaced k values, correctly handles k==0,
    # as interpolation in loglog space can produce nan gradients
    pk_fn = lambda x: jnp.exp(jnp.interp(x.reshape(-1), k, logpk, left=-jnp.inf, right=-jnp.inf)).reshape(x.shape)
    return pk_fn


def lin_power_mesh(cosmo:Cosmology, mesh_shape, box_shape, a=1., n_interp=256):
    """
    Return linear matter power spectrum field.
    """
    pk_fn = lin_power_interp(cosmo, a=a, n_interp=n_interp)
    kvec = rfftk(mesh_shape)
    k_box = sum((ki  * (m / l))**2 for ki, m, l in zip(kvec, mesh_shape, box_shape))**0.5
    return pk_fn(k_box) * (mesh_shape / box_shape).prod() # NOTE: convert from (Mpc/h)^3 to cell units
    

def gausslin_posterior(delta_obs, cosmo:Cosmology, b1, a, box_shape, gxy_count):
    """
    Return posterior mean, std, and preconditioning fields of the linear matter field (at a=1) given the observed field,
    by assuming Gaussian linear model. All fields are in fourier space.
    """
    # Compute linear matter power spectrum
    mesh_shape = ch2rshape(delta_obs.shape)
    pmeshk = lin_power_mesh(cosmo, mesh_shape, box_shape)
    evolve = (1 + b1) * growth_factor(cosmo, a) # Eulerian linear bias = 1 + Lagrangian linear bias
    # TODO: add rsd with kaiser model?

    stds = jnp.where(pmeshk==0., 0., pmeshk / (1 + gxy_count * evolve**2 * pmeshk))**.5
    # NOTE: gradient safe version of stds = (gxy_count * evolve**2 + pmeshk**-1)**-.5
    means = stds**2 * gxy_count * evolve * delta_obs

    scales = (1 + gxy_count * evolve**2 * pmeshk)**.5
    return means, stds, scales




def get_cosmology(**cosmo) -> Cosmology:
    """
    Return full cosmology object from cosmological params.
    """
    return Planck18(Omega_c=cosmo['Omega_m'] - Planck18.keywords['Omega_b'], 
                    sigma8=cosmo['sigma8'])


def samp2base(params:dict, config, inv=False, temp=1.) -> dict:
    """
    Transform sample params into base params.
    """
    out = {}
    for in_name, value in params.items():
        name = in_name if inv else in_name[:-1]
        out_name = in_name+'_' if inv else in_name[:-1]

        loc, scale = config[name]['loc'], config[name]['scale']
        low, high = config[name].get('low', -jnp.inf), config[name].get('high', jnp.inf)
        scale *= temp**.5

        # Reparametrize
        if not inv:
            if low != -jnp.inf or high != jnp.inf:
                push = lambda x: std2trunc(x, loc, scale, low, high) # truncate value in interval
            else:
                push = lambda x: x * scale + loc
        else:
            if low != -jnp.inf or high != jnp.inf:
                push = lambda x: trunc2std(x, loc, scale, low, high)
            else:
                push = lambda x: (x - loc) / scale

        out[out_name] = push(value)
    return out


def samp2base_mesh(init:dict, cosmo:Cosmology, box_shape, precond=False, 
                   transfer=None, inv=False, temp=1.) -> dict:
    """
    Transform sample mesh into base mesh, i.e. initial wavevectors at a=1.
    """
    assert len(init) <= 1, "init dict should only have one or zero key"
    for in_name, mesh in init.items():
        out_name = in_name+'_' if inv else in_name[:-1]
        mesh_shape = ch2rshape(mesh.shape) if inv else mesh.shape

        # Reparametrize
        if not inv:
            if precond in [0, 1, 2]:

                if precond==0:
                    # Sample in direct space
                    mesh = jnp.fft.rfftn(mesh) # ~ G(0, I)

                elif precond==1:
                    # Sample in fourier space
                    mesh = rg2cgh(mesh) # ~ G(0, I)

                elif precond==2:
                    # Sample in fourier space with
                    # partial (and static) posterior preconditioning assuming Gaussian linear model and fiducial cosmology
                    # as done in [Bayer+2023](http://arxiv.org/abs/2307.09504)
                    mesh = rg2cgh(mesh) # ~ G(0, I + n * P_fid(a_obs))
                    mesh *= transfer # ~ G(0, I) ; guide = (I + n * P_fid(a_obs))^-1/2
                
                # Compute linear matter power spectrum
                pmeshk = lin_power_mesh(cosmo, mesh_shape, box_shape, a=1.)
                mesh *= pmeshk**.5 # ~ G(0, P)

            elif precond>=3:
                # Sample in fourier space with
                # complete (and dynamic) posterior preconditioning assuming Gaussian linear model
                # means, stds = guide # sigma = (n * (bD)^2 + P^-1)^-1/2 ; mu = sigma^2 * nbD * delta_obs
                # mesh = rg2cgh(mesh) # ~ G( -mu * sigma^-1, sigma^-2 * P) 
                # mesh = stds * mesh + means # ~ G(0, P)

                mesh = rg2cgh(mesh) # ~ G(0, I + n * (bD)^2 * P)
                mesh *= transfer # ~ G(0, P) ; guide = (n * (bD)^2 + P^-1)^-1/2

            mesh *= temp**.5
        else:
            mesh /= temp**.5
            if precond in [0, 1, 2]:

                pmeshk = lin_power_mesh(cosmo, mesh_shape, box_shape, a=1.)
                mesh = safe_div(mesh, pmeshk**.5) # ~ G(0, I)

                if precond==0:
                    mesh = jnp.fft.irfftn(mesh)

                elif precond==1:
                    mesh = cgh2rg(mesh)

                elif precond==2:
                    mesh /= transfer # ~ G(0, I + n * P_fid(a_obs))
                    mesh = cgh2rg(mesh)

            elif precond>=3:        
                # means, stds = guide # sigma = (n * (bD)^2 + P^-1)^-1/2 ; mu = sigma^2 * nbD * delta_obs
                # mesh = safe_div(mesh - means, stds) # ~ G( -mu * sigma^-1, sigma^-2 * P)
                # mesh = cgh2rg(mesh)

                mesh = safe_div(mesh, transfer) # ~ G(0, I + n * (bD)^2 * P) ; guide = (n * (bD)^2 + P^-1)^-1/2
                mesh = cgh2rg(mesh)

                

        return {out_name:mesh}
    return {}






def lagrangian_weights(cosmo:Cosmology, a, pos, box_shape, 
                       b1, b2, bs2, bn2, init_mesh):
    """
    Return Lagrangian bias expansion weights as in [Modi+2020](http://arxiv.org/abs/1910.07097).
    .. math::
        
        w = 1 + b_1 \\delta + b_2 \\left(\\delta^2 - \\braket{\\delta^2}\\right) + b_{s^2} \\left(s^2 - \\braket{s^2}\\right) + b_{\\nabla^2} \\nabla^2 \\delta
    """    
    # Get init_mesh at observation scale factor
    init_mesh = init_mesh * growth_factor(cosmo, a)
    if jnp.isrealobj(init_mesh):
        delta = init_mesh
        delta_k = jnp.fft.rfftn(delta)
    else:
        delta_k = init_mesh
        delta = jnp.fft.irfftn(delta_k)
    # Smooth field to mitigate negative weights or TODO: use gaussian lagrangian biases
    # k_nyquist = jnp.pi * jnp.min(mesh_shape / box_shape)
    # delta_k = delta_k * jnp.exp( - kk_box / k_nyquist**2)
    # delta = jnp.fft.irfftn(delta_k)

    mesh_shape = delta.shape
    kvec = rfftk(mesh_shape)
    kk_box = sum((ki  * (m / l))**2 
            for ki, m, l in zip(kvec, mesh_shape, box_shape)) # minus laplace kernel in h/Mpc physical units

    # Init weights
    weights = 1.
    
    # Apply b1, punctual term
    delta_part = cic_read(delta, pos)
    weights = weights + b1 * delta_part

    # Apply b2, punctual term
    delta2_part = delta_part**2
    weights = weights + b2 * (delta2_part - delta2_part.mean())

    # Apply bshear2, non-punctual term
    pot_k = delta_k * invlaplace_kernel(kvec)

    shear2 = 0  
    for i, ki in enumerate(kvec):
        # Add diagonal terms
        shear2 = shear2 + jnp.fft.irfftn( - ki**2 * pot_k - delta_k / 3)**2
        for kj in kvec[i+1:]:
            # Add strict-up-triangle terms (counted twice)
            shear2 = shear2 + 2 * jnp.fft.irfftn( - ki * kj * pot_k)**2

    shear2_part = cic_read(shear2, pos)
    weights = weights + bs2 * (shear2_part - shear2_part.mean())

    # Apply bnabla2, non-punctual term
    delta_nl = jnp.fft.irfftn( - kk_box * delta_k)

    delta_nl_part = cic_read(delta_nl, pos)
    weights = weights + bn2 * delta_nl_part

    return weights


















def rsd_bf(cosmo:Cosmology, a, p, los:np.ndarray=None):
    """
    Redshift-Space Distortion (RSD) displacement from cosmology and Particle Mesh (PM) momentum.
    Computed with respect scale factor and line-of-sight.

    No RSD if los is None.
    """
    if los is None:
        return jnp.zeros_like(p)
    else:
        los = np.asarray(los)
        los /= jnp.linalg.norm(los)
        # Pi-Integrator momentum p = dx/dg
        dx_rsd = p * growth_factor(cosmo, a) * growth_rate(cosmo, a)
        # Project velocity on line-of-sight
        dx_rsd = dx_rsd * los
        return dx_rsd


def rsd_fpm(cosmo:Cosmology, a, p, los:np.ndarray=None):
    """
    Redshift-Space Distortion (RSD) displacement from cosmology and Particle Mesh (PM) momentum.
    Computed with respect scale factor and line-of-sight.
    
    No RSD if los is None.
    """
    if los is None:
        return jnp.zeros_like(p)
    else:
        los = np.asarray(los)
        los /= jnp.linalg.norm(los)
        # Divide PM momentum by scale factor once to retrieve velocity, and once again for comobile velocity  
        dx_rsd = p / (jc.background.Esqr(cosmo, a)**.5 * a**2)
        # Project velocity on line-of-sight
        dx_rsd = dx_rsd * los
        return dx_rsd




def kaiser_weights(cosmo:Cosmology, a, bE, mesh_shape, los:np.ndarray=None):
    """
    Return Kaiser Eulerian bias weights including linear bias and RSD.

    No RSD if los is None.
    """
    if los is None:
        return bE
    else:
        los = np.asarray(los)
        los /= jnp.linalg.norm(los)
        kvec = rfftk(mesh_shape)
        kmesh = sum(kk**2 for kk in kvec)**0.5 # in cell units
        mumesh = sum(ki * losi for ki, losi in zip(kvec, los))
        mumesh = safe_div(mumesh, kmesh)

        return bE + growth_rate(cosmo, a) * mumesh**2


def kaiser_model(cosmo:Cosmology, a, bE, init_mesh, los:np.ndarray=None):
    """
    Kaiser model, with linear growth, Eulerian linear bias, and RSD.
    """
    weights = kaiser_weights(cosmo, a, bE, init_mesh.shape, los)
    init_mesh *= growth_factor(cosmo, a) * weights
    return 1 + jnp.fft.irfftn(init_mesh) #  1 + delta


