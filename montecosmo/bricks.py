from functools import partial
import numpy as np
from jax import numpy as jnp, grad, vmap, lax
from jax.scipy.spatial.transform import Rotation

from jax_cosmo import Cosmology, background, constants, power
import fitsio
from scipy.interpolate import SmoothSphereBivariateSpline

from montecosmo.utils import std2trunc, trunc2std, rg2cgh, cgh2rg, ch2rshape, r2chshape, safe_div
from montecosmo.nbody import rfftk, invlaplace_kernel, gradient_kernel, a2g, g2a, a2f, a2chi, chi2a, paint, read

#############
# Cosmology #
#############
# [Planck2015 XIII](https://arxiv.org/abs/1502.01589) Table 4 final column (best fit)
Planck15 = partial(Cosmology,
    Omega_c=0.2589,
    Omega_b=0.04860,
    Omega_k=0.0,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    w0=-1.0,
    wa=0.0,)

# [Planck 2018 VI](https://arxiv.org/abs/1807.06209) Table 2 final column (best fit)
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


def get_cosmology(**cosmo) -> Cosmology:
    """
    Return full cosmology object from cosmological params.
    """
    return Planck18(Omega_c=cosmo['Omega_m'] - Planck18.keywords['Omega_b'], 
                    sigma8=cosmo['sigma8'])
    # return Planck18(Omega_c=cosmo['Omega_c'], Omega_b=cosmo['Omega_b'], 
    #                 sigma8=cosmo['sigma8'])



#########
# Power #
#########
def lin_power_interp(cosmo=Cosmology, a=1., n_interp=256):
    """
    Return a light emulation of the linear matter power spectrum.
    """
    ks = jnp.logspace(-4, 1, n_interp)
    # logpows = jnp.log(power.linear_matter_power(cosmo, ks, a=a))
    # Interpolate in semilogy space with logspaced k values, correctly handles k==0,
    # as interpolation in loglog space can produce nan gradients
    # pow_fn = lambda x: jnp.exp(jnp.interp(x.reshape(-1), ks, logpows, left=-jnp.inf, right=-jnp.inf)).reshape(x.shape)
    pows = power.linear_matter_power(cosmo, ks, a=a)
    pow_fn = lambda x: jnp.interp(x.reshape(-1), ks, pows, left=0., right=0.).reshape(x.shape)
    return pow_fn


def lin_power_mesh(cosmo:Cosmology, mesh_shape, box_shape, a=1., n_interp=256):
    """
    Return linear matter power spectrum field.
    """
    pow_fn = lin_power_interp(cosmo, a=a, n_interp=n_interp)
    kvec = rfftk(mesh_shape)
    kmesh = sum((ki  * (m / b))**2 for ki, m, b in zip(kvec, mesh_shape, box_shape))**.5
    return pow_fn(kmesh) * (mesh_shape / box_shape).prod() # from [Mpc/h]^3 to cell units


def trans_phi2delta_interp(cosmo:Cosmology, a=1., n_interp=256):
    """
    Return a light emulation of the transfer function from primordial potential to linear matter density field.
    """
    # NOTE: we could do as in
    # https://github.com/cosmodesi/desilike/blob/52f52698f7d901881724cd10f3bdd446e79a19f3/desilike/theories/galaxy_clustering/primordial_non_gaussianity.py#L84
    # but jax_cosmo has no A_s, so fallback on https://arxiv.org/pdf/1904.08859

    pow_fn = lin_power_interp(cosmo, a=a)
    ks = jnp.logspace(-4, 1, n_interp)
    pow_prim = ks**cosmo.n_s
    pow_lin = pow_fn(ks)
    trans_lin = (pow_lin / pow_prim / (pow_lin[0] / pow_prim[0]))**.5

    z_norm = 10. # in matter-dominated era
    a_norm = 1. / (1. + z_norm)
    normalized_growth_factor = a2g(cosmo, a) / a2g(cosmo, a_norm) * a_norm
    trans = - 2. * constants.rh**2 * ks**2 * trans_lin * normalized_growth_factor / (3. * cosmo.Omega)
    trans_fn = lambda x: jnp.interp(x.reshape(-1), ks, trans, left=0., right=0.).reshape(x.shape)
    return trans_fn


def add_png(cosmo:Cosmology, fNL, init_mesh, box_shape):
    """
    Add Primordial Non-Gaussianity (PNG) to the linear matter density field.
    """
    mesh_shape = ch2rshape(init_mesh.shape)
    kvec = rfftk(mesh_shape)
    kmesh = sum((ki  * (m / b))**2 for ki, m, b in zip(kvec, mesh_shape, box_shape))**.5
    trans_phi2delta = trans_phi2delta_interp(cosmo)(kmesh)

    phi = jnp.fft.irfftn(safe_div(init_mesh, trans_phi2delta))
    phi2 = phi**2
    phi += fNL * (phi2 - phi2.mean())
    init_mesh = trans_phi2delta * jnp.fft.rfftn(phi)

    return init_mesh



##########
# Kaiser #
##########
def kaiser_boost(cosmo:Cosmology, a, bE, mesh_shape, los=(0,0,0)):
    """
    Return Eulerian Kaiser boost including linear growth, Eulerian linear bias, and RSD.
    """
    kvec = rfftk(mesh_shape)
    kmesh = sum(kk**2 for kk in kvec)**.5 # in cell units
    mumesh = sum(ki * losi for ki, losi in zip(kvec, los))
    mumesh = safe_div(mumesh, kmesh)

    return a2g(cosmo, a) * (bE + a2f(cosmo, a) * mumesh**2)


def kaiser_model(cosmo:Cosmology, a, bE, init_mesh, los=(0,0,0)):
    """
    Kaiser model, i.e. growth, Eulerian bias, and RSD, are linear.
    For flat-sky with no light-cone, this linear model is moreover diagonal in Fourier space.
    """
    mesh_shape = ch2rshape(init_mesh.shape)

    if jnp.shape(los) == (3,) and jnp.shape(a) == (): # flat-sky, no light-cone
        init_mesh *= kaiser_boost(cosmo, a, bE, mesh_shape, los)
        return 1 + jnp.fft.irfftn(init_mesh) # 1 + delta
    
    elif jnp.shape(los) == (3,): # flat-sky, light-cone
        kvec = rfftk(mesh_shape)
        kmesh = sum(kk**2 for kk in kvec)**.5 # in cell units
        mumesh = sum(ki * losi for ki, losi in zip(kvec, los))
        mumesh = safe_div(mumesh, kmesh)

        delta = bE * jnp.fft.irfftn(init_mesh) + a2f(cosmo, a) * jnp.fft.irfftn(mumesh**2 * init_mesh)
        return 1 + a2g(cosmo, a) * delta # 1 + delta
    
    else: # curved-sky
        # kvec = rfftk(mesh_shape)
        # kmesh = sum(kk**2 for kk in kvec)**.5 # in cell units

        # mu_delta = jnp.stack([jnp.fft.irfftn(
        #         safe_div(kvec[i] * init_mesh, kmesh)
        #         ) for i in range(3)], axis=-1)
        # mu_delta = (mu_delta * los).sum(-1)
        # mu_delta = jnp.fft.rfftn(mu_delta)

        # mu2_delta = jnp.stack([jnp.fft.irfftn(
        #         safe_div(kvec[i] * mu_delta, kmesh)
        #         ) for i in range(3)], axis=-1)
        # mu2_delta = (mu2_delta * los).sum(-1)

        # delta = bE * jnp.fft.irfftn(init_mesh) + a2f(cosmo, a) * mu2_delta
        # return 1 + a2g(cosmo, a) * delta # 1 + delta
    
        print("kai")
        # return jnp.ones(mesh_shape)
        delta = bE * jnp.fft.irfftn(init_mesh)
        return 1 + 0.5 * delta # 1 + delta


def kaiser_posterior(delta_obs, cosmo:Cosmology, bE, count, wind_mesh, a, box_shape, los=(0,0,0)):
    """
    Return posterior mean and std fields of the linear matter field (at a=1) given the observed field,
    by assuming Kaiser model. All fields are in fourier space.
    """
    # Compute linear matter power spectrum
    mesh_shape = ch2rshape(delta_obs.shape)
    pmesh = lin_power_mesh(cosmo, mesh_shape, box_shape)
    boost = kaiser_boost(cosmo, a, bE, mesh_shape, los)
    wind = (wind_mesh**2).mean()**.5

    stds = (pmesh / (1 + wind * count * boost**2 * pmesh))**.5
    means = stds**2 * boost * count * wind * delta_obs
    return means, stds




#####################
# Reparametrization #
#####################
def samp2base(params:dict, config, inv=False, temp=1.) -> dict:
    """
    Transform sample params into base params.
    """
    out = {}
    for in_name, value in params.items():
        name = in_name if inv else in_name[:-1]
        out_name = in_name+'_' if inv else in_name[:-1]

        conf = config[name]
        low, high = conf.get('low', -jnp.inf), conf.get('high', jnp.inf)
        loc_fid, scale_fid = conf['loc_fid'], conf['scale_fid']
        scale_fid *= temp**.5

        # Reparametrize
        if not inv:
            if low != -jnp.inf or high != jnp.inf:
                push = lambda x: std2trunc(x, loc_fid, scale_fid, low, high)
            else:
                push = lambda x: x * scale_fid + loc_fid
        else:
            if low != -jnp.inf or high != jnp.inf:
                push = lambda x: trunc2std(x, loc_fid, scale_fid, low, high)
            else:
                push = lambda x: (x - loc_fid) / scale_fid

        out[out_name] = push(value)
    return out


def samp2base_mesh(init:dict, precond=False, transfer=None, inv=False, temp=1.) -> dict:
    """
    Transform sample mesh into base mesh, i.e. initial wavevector coefficients at a=1.
    """
    assert len(init) <= 1, "init dict should only have one or zero key"
    for in_name, mesh in init.items():
        out_name = in_name+'_' if inv else in_name[:-1]
        transfer *= temp**.5

        # Reparametrize
        if not inv:
            if precond=='real':
                # Sample in real space
                mesh = jnp.fft.rfftn(mesh)

            elif precond in ['fourier','kaiser','kaiser_dyn']:
                # Sample in fourier space
                mesh = rg2cgh(mesh)

            mesh *= transfer # ~ CN(0, P)
        else:
            mesh = safe_div(mesh, transfer)
            
            if precond=='real':
                mesh = jnp.fft.irfftn(mesh)

            elif precond in ['fourier','kaiser','kaiser_dyn']:
                mesh = cgh2rg(mesh)

        return {out_name:mesh}
    return {}



########
# Bias #
########
def lagrangian_bias(cosmo:Cosmology, pos, a, box_shape, 
                       b1, b2, bs2, bn2, fNL, bnp, init_mesh, read_order:int=2):
    """
    Return Lagrangian bias expansion weights as in [Modi+2020](http://arxiv.org/abs/1910.07097).
    .. math::
        
        w = 1 + b_1 \\delta_L + b_2 \\left(\\delta_L^2 - \\braket{\\delta_L^2}\\right) 
        + b_{s^2} \\left(s^2 - \\braket{s^2}\\right) + b_{\\nabla^2} \\nabla^2 \\delta_L
        + b_{\\phi} f_\\mathrm{NL} \\phi + b_{\\phi \\delta} f_\\mathrm{NL} (\\phi \\delta_L - \\braket{\\phi \\delta_L})
    """    
    # Smooth field to mitigate negative weights or TODO: use gaussian lagrangian biases?
    # k_nyquist = jnp.pi * jnp.min(mesh_shape / box_shape)
    # init_mesh *= gaussian_kernel(kvec, kcut=k_nyquist)

    delta = jnp.fft.irfftn(init_mesh)
    growths = a2g(cosmo, a)

    mesh_shape = delta.shape
    kvec = rfftk(mesh_shape)
    kmesh = sum((ki  * (m / b))**2 for ki, m, b in zip(kvec, mesh_shape, box_shape))**.5 # in h/Mpc 

    # Init weights
    weights = 1.
    
    # Apply b1, punctual term
    delta_pos = read(pos, delta, read_order) * growths.squeeze()
    weights += b1 * delta_pos

    # Apply b2, punctual term
    delta2_pos = delta_pos**2
    weights += b2 * (delta2_pos - delta2_pos.mean())

    # Apply bshear2, non-punctual term
    pot = init_mesh * invlaplace_kernel(kvec)
    dims = range(len(kvec))
    shear2 = 0.

    for i in dims:
        # Add diagonal terms
        nabi = gradient_kernel(kvec, i)
        shear2 += jnp.fft.irfftn(nabi**2 * pot - init_mesh / 3)**2
        for j in dims[i+1:]:
            # Add strict-up-triangle terms (counted twice)
            nabj = gradient_kernel(kvec, j)
            shear2 += 2 * jnp.fft.irfftn(nabi * nabj * pot)**2

    shear2_pos = read(pos, shear2, read_order) * growths.squeeze()**2
    weights += bs2 * (shear2_pos - shear2_pos.mean())

    # Apply bnabla2, higher-order term
    delta_nab2 = jnp.fft.irfftn( - kmesh**2 * init_mesh)

    delta_nab2_pos = read(pos, delta_nab2, read_order) * growths.squeeze()
    weights += bn2 * delta_nab2_pos

    # Apply bphi, primordial term
    trans_phi2delta = trans_phi2delta_interp(cosmo)(kmesh)
    phi = jnp.fft.irfftn(safe_div(init_mesh, trans_phi2delta))
    p = 1. # tracer parameter
    bp = b_phi(b1, p)

    phi_pos = read(pos, phi, read_order)
    weights += bp * fNL * phi_pos
    
    # Apply bphidelta, primordial term
    phi_delta_pos = phi_pos * delta_pos
    bpd = b_phi_delta(b1, b2, bp)

    weights += bpd * fNL * (phi_delta_pos - jnp.mean(phi_delta_pos))

    # Compute separatly bnablapar, velocity bias term
    delta_nabpar_pos = jnp.stack([
                read(pos, jnp.fft.irfftn(gradient_kernel(kvec, i) * (m / b) * init_mesh), read_order) 
                for i, (m, b) in enumerate(zip(mesh_shape, box_shape))], axis=-1) # in h/Mpc 
    dvel = bnp * delta_nabpar_pos * growths

    return weights, dvel



def b_phi(b1, p=1., delta_c=1.686):
    """
    Primordial scale-dependant bias parameter. See []()
    """
    # 2 * delta_c * (bE1 - p) and bE1 = 1 + b1
    return 2 * delta_c * (1 + b1 - p)

def b_phi_delta(b1, b2, bp, delta_c=1.686):
    """
    Primordial-density scale-dependant bias parameter. See []()
    """
    # bp - (bE1 - 1) + delta_c * (bE2 - 8 / 21 * (bE1 - 1)) and bE2 = b2 + 8/21 * b1
    # TODO: check for the factor 2
    return bp - b1 + delta_c * b2




##############################
# Distance and Line-Of-Sight #
##############################
def regular_pos(mesh_shape, ptcl_shape):
    """
    Return regularly spaced positions in cell coordinates.
    """
    pos = [np.linspace(0, m, p, endpoint=False) for m, p in zip(mesh_shape, ptcl_shape)]
    pos = jnp.stack(np.meshgrid(*pos, indexing='ij'), axis=-1).reshape(-1, 3)
    return pos

def unif_pos(mesh_shape, ptcl_shape, seed=42):
    """
    Return uniformly distributed positions in cell coordinates.
    """
    from jax import random as jr
    if isinstance(seed, int):
        seed = jr.key(seed)
    pos = jr.uniform(seed, shape=(ptcl_shape.prod(), 3), minval=0., maxval=mesh_shape)
    return pos

def sobol_pos(mesh_shape, ptcl_shape, seed=42):
    """
    Return Sobol sequence of positions in cell coordinates.
    """
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    return jnp.array(sampler.random(n=ptcl_shape.prod()) * mesh_shape)


def cell2phys_pos(pos, box_center, box_rot:Rotation, box_shape, mesh_shape):
    """
    Cell positions to physical positions.
    """
    pos *= (box_shape / mesh_shape)
    pos -= box_shape / 2
    pos = box_rot.apply(pos)
    pos += box_center
    return pos

def phys2cell_pos(pos, box_center, box_rot:Rotation, box_shape, mesh_shape):
    """
    Physical positions to cell positions.
    """
    pos -= box_center
    pos = box_rot.apply(pos, inverse=True)
    pos += box_shape / 2
    pos /= (box_shape / mesh_shape)
    return pos

def cell2phys_vel(vel, box_rot:Rotation, box_shape, mesh_shape):
    """
    Cell velocities to physical velocities.
    """
    vel *= (box_shape / mesh_shape)
    vel = box_rot.apply(vel)
    return vel

def phys2cell_vel(vel, box_rot:Rotation, box_shape, mesh_shape):
    """
    Physical velocities to cell velocities.
    """
    vel = box_rot.apply(vel, inverse=True)
    vel /= (box_shape / mesh_shape)
    return vel


def radius_mesh(box_center, box_rot:Rotation, box_shape, mesh_shape, curved_sky=True):
    """
    Return physical distances of the mesh cells.
    """
    # Only Nx*Ny*Nz memory instead of naive Nx*Ny*Nz*3 obtained from mesh of positions 
    rx = np.arange(mesh_shape[0]).reshape([-1, 1, 1])
    ry = np.arange(mesh_shape[1]).reshape([1, -1, 1])
    rz = np.arange(mesh_shape[2]).reshape([1, 1, -1])
    rvec = rx, ry, rz

    box_center = box_rot.apply(box_center, inverse=True)
    if curved_sky:
        # Use that ||Rx + c|| = ||x + R^T c|| to avoid computing Rx
        rvec = [r * b / m - b / 2 + c for r, m, b, c in zip(rvec, mesh_shape, box_shape, box_center)]
        rmesh = sum(ri**2 for ri in rvec)**.5
    else:
        # Use that l^T (Rx + c) = (R^T l)^T (x + R^T c) to avoid computing Rx
        # Here l = c / ||c|| so R^T l = R^T c / ||R^T c|| 
        los = safe_div(box_center, jnp.linalg.norm(box_center))
        rvec = [(r * b / m - b / 2 + c) * l for r, m, b, c, l in zip(rvec, mesh_shape, box_shape, box_center, los)]
        rmesh = jnp.abs(sum(ri for ri in rvec))
    return rmesh

def pos_mesh(box_center, box_rot:Rotation, box_shape, mesh_shape):
    """
    Return a mesh of the physical positions of the mesh cells.
    """
    pos = np.indices(mesh_shape, dtype=float).reshape(3,-1).T
    pos = cell2phys_pos(pos, box_center, box_rot, box_shape, mesh_shape)
    return pos.reshape(tuple(mesh_shape) + (3,))


def redges_and_scalefactors(cosmo:Cosmology, rmin, rmax, n_shells):
    """
    Return radius shell edges and their effective scale factors.
    Shell edges are linearly spaced in growth factor.
    """
    gmin, gmax = a2g(cosmo, chi2a(cosmo, rmax)), a2g(cosmo, chi2a(cosmo, rmin))
    gs = np.linspace(gmin, gmax, n_shells+1)
    redges = a2chi(cosmo, g2a(cosmo, gs)) # decreasing distance
    a = g2a(cosmo, (gs[:-1] + gs[1:]) / 2)
    return redges, a

def scale_pos(pos, los, scale_par, scale_perp):
    """
    Scale positions in parallel and perpendicular directions.
    """
    pos_par = (pos * los).sum(-1, keepdims=True) * los
    pos_perp = pos - pos_par
    pos_par *= scale_par
    pos_perp *= scale_perp
    return pos_par + pos_perp

def parperp2isoap(alpha_par, alpha_perp):
    """
    Convert parallel and perpendical scaling into isotropic and anisotropic scaling.
    """
    alpha_iso = (alpha_par * alpha_perp**2)**(1/3)
    alpha_ap = alpha_par / alpha_perp
    return alpha_iso, alpha_ap

def isoap2parperp(alpha_iso, alpha_ap):
    """
    Convert isotropic and anisotropic scaling into parallel and perpendical scaling.
    """
    alpha_par = alpha_iso * alpha_ap**(2/3)
    alpha_perp = alpha_iso * alpha_ap**(-1/3)
    return alpha_par, alpha_perp




################################
# Cell to Physical to Redshift #
################################
def tophysical_pos(pos, box_center, box_rot:Rotation, box_shape, mesh_shape, 
               cosmo:Cosmology, a_obs=None, curved_sky=True):
    """
    Return physical positions, distances, line-of-sight(s), and scale factor(s)
    for the different configurations of light-cone and sky.
    """
    pos = cell2phys_pos(pos, box_center, box_rot, box_shape, mesh_shape)
    if curved_sky:
        los = safe_div(pos, jnp.linalg.norm(pos, axis=-1, keepdims=True))
        rpos = jnp.linalg.norm(pos, axis=-1, keepdims=True)
    else:
        los = safe_div(box_center, np.linalg.norm(box_center))
        rpos = jnp.abs((pos * los).sum(-1, keepdims=True))

    if a_obs is None:
        a = chi2a(cosmo, rpos)
    else:
        a = a_obs
    return pos, rpos, los, a

def tophysical_mesh(box_center, box_rot:Rotation, box_shape, mesh_shape, 
                    cosmo:Cosmology, a_obs=None, curved_sky=True):
    """
    Return scale factor mesh for the different configurations of light-cone and sky.
    """
    if curved_sky:
        pomesh = pos_mesh(box_center, box_rot, box_shape, mesh_shape)
        los = safe_div(pomesh, jnp.linalg.norm(pomesh, axis=-1, keepdims=True))
        rmesh = jnp.linalg.norm(pomesh, axis=-1)
    else:
        los = safe_div(box_center, np.linalg.norm(box_center))
        rmesh = radius_mesh(box_center, box_rot, box_shape, mesh_shape, curved_sky)

    if a_obs is None:
        a = chi2a(cosmo, rmesh)
    else:
        a = a_obs
    return los, a
    

def rsd(cosmo:Cosmology, vel, los, a, box_rot, box_shape, mesh_shape, dvel=0.):
    """
    Redshift-Space Distortions.
    """
    # Growth-time integrator vel := dq / dg = v / (H * g * f), so Dq := v / H = vel * g * f
    # v in (Mpc/h)*(km/s/(Mpc/h)) = km/s, so Dq in Mpc/h
    vel = cell2phys_vel(vel, box_rot, box_shape, mesh_shape)
    vel *= a2g(cosmo, a) * a2f(cosmo, a)
    vel += dvel
    dpos = (vel * los).sum(-1, keepdims=True) * los
    return dpos


def ap_auto(pos, los, cosmo:Cosmology, cosmo_fid:Cosmology, curved_sky=True):
    """
    Automatic Alcock-Paczynski effect.
    """
    def alpha_fn(rpos):
        rpos_new = a2chi(cosmo_fid, chi2a(cosmo, rpos))
        return safe_div(rpos_new, rpos)
        
    # def alpha_fn(rpos):
    #     alpha = (1 + 0.2 * (cosmo.Omega_c - 0.2607))
    #     return alpha
    
    if curved_sky:
        rpos = jnp.linalg.norm(pos, axis=-1, keepdims=True)
        alpha = alpha_fn(rpos)
        pos *= alpha
    else:
        rpos = jnp.abs((pos * los).sum(-1, keepdims=True))
        alpha = alpha_fn(rpos)
        pos = scale_pos(pos, los, alpha, 1)
    return pos

def ap_auto_absdetjac(pos, los, cosmo:Cosmology, cosmo_fid:Cosmology, curved_sky=True):
    """
    Automatic Alcock-Paczynski effect.
    """
    def alpha_fn(rpos):
        rpos_new = a2chi(cosmo_fid, chi2a(cosmo, rpos))
        return safe_div(rpos_new, rpos)
        
    # def alpha_fn(rpos):
    #     alpha = (1 + 0.2 * (cosmo.Omega_c - 0.2607))
    #     return alpha
    
    if curved_sky:
        rpos = jnp.linalg.norm(pos, axis=-1, keepdims=True)
        alpha = alpha_fn(rpos)
        pos *= alpha
    else:
        rpos = jnp.abs((pos * los).sum(-1, keepdims=True))
        alpha = alpha_fn(rpos)
        pos = scale_pos(pos, los, alpha, 1)

    def absdetjac_fn(rpos):
        # NOTE: jac(alpha(r) * q) = alpha I + alpha' / r * q q^T
        # => absdetjac(alpha(r) * q) = alpha**(d-1) * (alpha + r * alpha')
        alpha = alpha_fn(rpos)
        absdetjac = alpha + rpos * grad(alpha_fn)(rpos)
        if curved_sky:
            absdetjac *= alpha**2
        return absdetjac

    return pos, vmap(absdetjac_fn)(rpos.squeeze())

def ap_param(pos, los, alphas, curved_sky=True):
    """
    Parametrized Alcock-Paczynski effect.
    """
    if curved_sky:
        pos *= alphas['alpha_iso']
    else:
        alpha_par, alpha_perp = isoap2parperp(alphas['alpha_iso'], alphas['alpha_ap'])
        pos = scale_pos(pos, los, alpha_par, alpha_perp)
    return pos

def rsd_ap_auto(pos, vel, rpos, los, a, cosmo:Cosmology, cosmo_fid:Cosmology, curved_sky=True):
    """
    Redshift-Space Distortions and automatic Alcock-Paczynski effect.
    """
    vel_los = (vel * los).sum(-1, keepdims=True)
    if not curved_sky: # handle positions behind line-of-sight
        vel_los *= jnp.sign((pos * los).sum(-1, keepdims=True))

    # Use that a_obs = 1 / (1 + z + v/c) = 1 / (1/a + v/H * H/c)
    a = (1 / a + vel_los * background.Esqr(cosmo, a)**.5 / constants.rh)**-1
    rpos_new = a2chi(cosmo_fid, a)
    alpha = safe_div(rpos_new, rpos)
    if curved_sky:
        pos *= alpha
    else:
        pos = scale_pos(pos, los, alpha, 1.)
    return pos



###################
# Mask and Window #
###################
def mesh2masked(mesh, mask=None):
    if mask is None:
        return mesh
    else:
        return mesh[...,mask]

def masked2mesh(masked, mask=None):
    if mask is None:
        return masked
    else:
        shape = jnp.shape(masked)[:-1] + jnp.shape(mask)
        return jnp.zeros(shape).at[...,mask].set(masked)

def radecrad2cart(ra, dec, radius):
    """
    Convert ra, dec (in degrees), and radius to cartesian coordinates.
    """
    ra = jnp.deg2rad(ra)
    dec = jnp.deg2rad(dec)
    x = jnp.cos(dec) * jnp.cos(ra)
    y = jnp.cos(dec) * jnp.sin(ra)
    z = jnp.sin(dec)
    cart = jnp.moveaxis(radius * jnp.stack((x, y, z)), 0, -1)
    return cart

def cart2radecrad(cart:jnp.ndarray):
    """
    Convert cartesian coordinates to ra, dec (in degrees), and radius.
    * ra \\in [0, 360]
    * dec \\in [-90, 90]
    * radius \\in [0, \\infty[
    """
    radius = jnp.linalg.norm(cart, axis=-1)
    x, y, z = jnp.moveaxis(cart, -1, 0)
    ra = jnp.rad2deg(jnp.arctan2(y, x)) % 360.
    dec = jnp.rad2deg(jnp.arcsin(z / radius))
    return ra, dec, radius

def radecz2cart(cosmo:Cosmology, radecz:dict):
    """
    Convert RA, DEC, Z dictionary (in degrees) to cartesian array (in Mpc/h).
    """
    ra = jnp.array(radecz['RA'])
    dec = jnp.array(radecz['DEC'])
    radius = a2chi(cosmo, 1 / jnp.array(1 + radecz['Z']))
    cart = radecrad2cart(ra, dec, radius)
    return cart

def cart2radecz(cosmo:Cosmology, cart:jnp.ndarray):
    """
    Convert cartesian array (in Mpc/h) to RA, DEC, Z dictionary (in degrees).
    """
    ra, dec, radius = cart2radecrad(cart)
    z = 1 / chi2a(cosmo, radius) - 1
    radecz = {'RA': ra, 'DEC': dec, 'Z': z}
    return radecz

def radec_interp(ra, dec, value, w=None, s=0., eps=1e-16):
    """
    Return an interpolator of a spherical field.
    """
    phi = np.deg2rad(ra).reshape(-1)
    theta = np.deg2rad(90. - dec).reshape(-1)
    value = value.reshape(-1)
    interp_fn = SmoothSphereBivariateSpline(theta, phi, value, w=w, s=s, eps=eps)

    def radec_interp_fn(ra, dec):
        shape = ra.shape
        phi = np.deg2rad(ra).reshape(-1)
        theta = np.deg2rad(90. - dec).reshape(-1)
        return interp_fn(theta, phi, grid=False).reshape(shape)
    return radec_interp_fn

# def radec_interp_mesh(ra, dec, value, w=None, s=0., eps=1e-16):
#     interp = radec_interp(ra, dec, value, w=w, s=s, eps=eps)

#     los = model.pos_mesh()
#     # los = pos_mesh(box_center, box_rot, box_shape, mesh_shape)
#     los = los / jnp.linalg.norm(los, axis=-1, keepdims=True)
#     ra_mesh, dec_mesh, _ = cart2radecrad(los)
#     val_mesh = interp(ra_mesh, dec_mesh)
#     return val_mesh



def simple_window(mesh_shape, padding=0., ord:float=np.inf):
    """
    Return an `ord-norm ball binary window mesh, with a `padding` 1D padded fraction.
    Therefore, `1/(1+padding)` is the mesh axes to ball axes ratio.
    """
    ord = float(ord)
    rx = jnp.abs((np.arange(mesh_shape[0]) + .5) * 2 / mesh_shape[0] - 1)
    ry = jnp.abs((np.arange(mesh_shape[1]) + .5) * 2 / mesh_shape[1] - 1)
    rz = jnp.abs((np.arange(mesh_shape[2]) + .5) * 2 / mesh_shape[2] - 1)

    rx = rx.reshape([-1, 1, 1])
    ry = ry.reshape([1, -1, 1])
    rz = rz.reshape([1, 1, -1])
    rvec = rx, ry, rz

    if ord == np.inf:
        rmesh = np.maximum(np.maximum(rvec[0], rvec[1]), rvec[2])
    elif ord == -np.inf:
        rmesh = np.minimum(np.minimum(rvec[0], rvec[1]), rvec[2])
    else:
        rmesh = sum(ri**ord for ri in rvec)**(1 / ord)

    wind_mesh = (rmesh < 1 / (1 + padding)).astype(float)
    # NOTE: window normalization to unit mean within its support.
    wind_mesh /= wind_mesh[wind_mesh > 0].mean()
    return wind_mesh

def simple_box(pos):
    """
    Return box configuration (center, rotvec, shape) for a given set of positions.
    The box is simply computed from the min and max of the positions along each axis.
    """
    low_corner, high_corner = pos.min(0), pos.max(0)
    center = (low_corner + high_corner) / 2
    shape = high_corner - low_corner
    rotvec = jnp.zeros(jnp.shape(pos)[-1])
    return center, rotvec, shape

def get_mesh_shape(box_shape, cell_budget, padding=0.):
    """
    Return mesh shape and cell length for a given box shape and cell budget, with optional padding.
    Mesh shape is rounded to the nearest even integers.
    """
    box_shape *= 1 + padding
    cell_length = float((box_shape.prod() / cell_budget)**(1/3))
    mesh_shape = 2 * np.rint(box_shape / cell_length / 2).astype(int)
    return mesh_shape, cell_length

def get_ptcl_shape(mesh_shape, oversampling=1.):
    """
    Return particle grid shape for a given mesh shape 
    and a 1D oversampling factor of the particle density by the mesh grid.
    """
    return np.rint(mesh_shape / oversampling).astype(int)


def catalog2mesh(path, cosmo:Cosmology, box_center, box_rot, box_shape, mesh_shape, paint_order:int=2):
    """
    Return painted mesh from a given path to RA, DEC, Z data.
    """
    data = fitsio.read(path, columns=['RA','DEC','Z'])
    pos = radecz2cart(cosmo, data)

    pos = phys2cell_pos(pos, box_center, box_rot, box_shape, mesh_shape)
    mesh = paint(pos, tuple(mesh_shape), paint_order)
    return mesh

def catalog2window(path, cosmo:Cosmology, cell_budget, padding=0., paint_order:int=2):
    """
    Return painted window mesh and box configuration from a given path to RA, DEC, Z data.
    """
    data = fitsio.read(path, columns=['RA','DEC','Z'])
    pos = radecz2cart(cosmo, data)
    box_center, box_rotvec, box_shape = simple_box(pos)
    mesh_shape, cell_length = get_mesh_shape(box_shape, cell_budget, padding)
    box_shape = mesh_shape * cell_length # box_shape update due to rounding and padding
    box_rot = Rotation.from_rotvec(box_rotvec)

    pos = phys2cell_pos(pos, box_center, box_rot, box_shape, mesh_shape)
    wind_mesh = paint(pos, tuple(mesh_shape), paint_order)

    # NOTE: window normalization to unit mean within its support.
    wind_mesh /= wind_mesh[wind_mesh > 0].mean()
    return wind_mesh, cell_length, box_center, box_rotvec


