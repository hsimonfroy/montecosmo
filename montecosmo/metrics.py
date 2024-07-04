import numpy as np
import jax.numpy as jnp
# from scipy.special import legendre
from jaxpm.growth import growth_rate, growth_factor


##################
# Power spectrum #
##################
def _initialize_pk(mesh_size, box_size, kmin, dk, los):

    W = jnp.ones(mesh_size)
    # W = np.empty(mesh_size, dtype='f4')
    # W[...] = 2.0
    # W[..., 0] = 1.0
    # W[..., -1] = 1.0 # XXX: Why?

    kmax = np.pi * np.min(mesh_size) / np.max(box_size) + dk / 2
    kedges = jnp.arange(kmin, kmax, dk)

    kshapes = np.eye(len(mesh_size), dtype=jnp.int32) * -2 + 1
    kvec = [(2 * jnp.pi * m / l) * jnp.fft.fftfreq(m).reshape(kshape)
            for m, l, kshape in zip(mesh_size, box_size, kshapes)]
    kmesh = sum(ki**2 for ki in kvec)**0.5

    dig = jnp.digitize(kmesh.reshape(-1), kedges)
    ksum = jnp.bincount(dig, weights=W.reshape(-1), length=len(kedges)+1)

    mumesh = sum(ki*losi for ki, losi in zip(kvec, los))
    kmesh_nozeros = jnp.where(kmesh==0, 1, kmesh) 
    mumesh = mumesh / kmesh_nozeros 
    mumesh = jnp.where(kmesh==0, 0, mumesh)
    
    return dig, ksum, W, kedges, mumesh


def power_spectrum(field, kmin, dk, mesh_size, box_size, los=jnp.array([0.,0.,1.]), multipoles=0, kcount=False):
    # Initialize values related to powerspectra (wavenumber bins and edges)
    los = los / jnp.linalg.norm(los)
    multipoles = jnp.atleast_1d(multipoles)
    dig, ksum, W, kedges, mumesh = _initialize_pk(mesh_size, box_size, kmin, dk, los)

    # Square modulus of FFT
    field_k = jnp.fft.fftn(field)
    field2_k = jnp.real(field_k * jnp.conj(field_k)) # TODO: cross pk

    Psum = jnp.empty((len(multipoles), *ksum.shape))
    for i_ell, ell in enumerate(multipoles):
        real_weights = W * field2_k * (2*ell+1) * legendre(ell, mumesh) # XXX: not implemented by jax.scipy.special.lpmm yet 
        Psum = Psum.at[i_ell].set(jnp.bincount(dig, weights=real_weights.reshape(-1), length=kedges.size+1))
    # Normalization for powerspectra
    P = (Psum / ksum).at[:,1:-1].get() * jnp.prod(box_size)
    norm = jnp.prod(mesh_size.astype(jnp.float32))**2

    # Find central values of each bin
    kbins = kedges[:-1] + (kedges[1:] - kedges[:-1]) / 2
    pk = jnp.concatenate([kbins[None], P / norm])
    if kcount:
        return pk, ksum[1:-1]
    else:
        return pk


def kaiser_formula(cosmo, a, pk_init, bias, multipoles=0):
    multipoles = jnp.atleast_1d(multipoles)
    a = jnp.atleast_1d(a)
    beta = growth_rate(cosmo, a) / bias
    k = pk_init[...,0,:]
    pk0 = pk_init[...,1,:] * growth_factor(cosmo, a)**2
    # f = growth_rate(cosmo, a)

    pk = np.empty((len(multipoles), *pk0.shape))
    for i_ell, ell in enumerate(multipoles):
        if ell==0:
            pk[i_ell] = (1 + beta * 2/3 + beta**2 /5) * bias**2 * pk0 
        elif ell==2:
            pk[i_ell] = (beta * 4/3 + beta**2 *4/7) * bias**2 * pk0 
        elif ell==4:
            pk[i_ell] = beta**2 * 8/35 * bias**2 * pk0 
        else: 
            raise NotImplementedError(
                "Handle only multipoles of order ell=0, 2 ,4. ell={ell} not implemented.") 
    return jnp.concatenate([k[None], pk])


def legendre(ell, x):
    """
    Return Legendre polynomial of given order.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Legendre_polynomials
    """
    P0 = lambda x: jnp.ones_like(x)
    P2 = lambda x: 1 / 2 * (3 * x**2 - 1)
    P4 = lambda x: 1 / 8 * (35 * x**4 - 30 * x**2 + 3)
    def error(x):
        return jnp.full_like(x, jnp.nan)
    # for vmaping on condition, see https://github.com/google/jax/issues/8409
    return jnp.piecewise(x, [ell==0, ell==2, ell==4], [P0,P2,P4,error])




###################
# Density regions #
###################
def qbi(x, proba=.95, axis=0, side='bi'):
    """
    Compute the Quantile Based Interval (QBI), 
    i.e. the interval of proba `proba` from quantile q1 to quantile q2, where:

    q1, q2 = (1-proba)/2, (1+proba)/2, for 'side==bi' Bilateral QBI (alias Equal-Tailed Interval)

    q1, q2 = 0, proba, for 'side==low' Low lateral QBI

    q1, q2 = 1-proba, 1, for 'side==high' High lateral QBI
    """
    if side == 'bi':
        p_low, p_high = (1-proba)/2, (1+proba)/2
    if side == 'low':
        p_low, p_high = 0, proba
    if side == 'high':
        p_low, p_high = 1-proba, 1
    q_low = jnp.quantile(x, p_low, axis=axis)
    q_high = jnp.quantile(x, p_high, axis=axis)
    return jnp.stack([q_low, q_high], axis=axis)
    

def hdi(x, proba=.95, axis=0):
    """
    Compute the Highest Density Interval (HDI),
    i.e. the smallest interval of proba `proba`.
    """
    x = np.moveaxis(x, axis, 0)
    x_sort = jnp.sort(x, axis=0)
    n = x.shape[0]
    # Round for better estimation at low number of sample, and handle also the case proba close to 1.
    i_length = min(int(jnp.round(proba * n)), n-1)

    intervals_low = x_sort[: (n - i_length)] # no need to consider all low bounds
    intervals_high = x_sort[i_length:]  # no need to consider all high bounds
    intervals_length = intervals_high - intervals_low # all intervals with given proba
    i_low = intervals_length.argmin(axis=0)
    i_high = i_low + i_length
    hdi_low = jnp.take_along_axis(x_sort, i_low[None], 0)[0]
    hdi_high = jnp.take_along_axis(x_sort, i_high[None], 0)[0]
    return jnp.stack([hdi_low, hdi_high], axis=axis)