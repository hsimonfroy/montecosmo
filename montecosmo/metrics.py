import numpy as np
import jax.numpy as jnp
from scipy.special import legendre
from jaxpm.growth import growth_rate, growth_factor

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
    Nsum = jnp.bincount(dig, weights=W.reshape(-1), length=len(kedges)+1)

    mumesh = sum(ki*losi for ki, losi in zip(kvec, los))
    kmesh_nozeros = jnp.where(kmesh==0, 1, kmesh) 
    mumesh = mumesh / kmesh_nozeros 
    mumesh = jnp.where(kmesh==0, 0, mumesh)
    
    return dig, Nsum, W, kedges, mumesh


def power_spectrum(field, kmin, dk, mesh_size, box_size, los=jnp.array([0.,0.,1.]), multipoles=0):
    # Initialize values related to powerspectra (mode bins and weights)
    los = los / jnp.linalg.norm(los)
    multipoles = jnp.atleast_1d(multipoles)
    dig, Nsum, W, kedges, mumesh = _initialize_pk(mesh_size, box_size, kmin, dk, los)

    # Absolute value of FFT
    fft_image = jnp.fft.fftn(field)
    pk = jnp.real(fft_image * jnp.conj(fft_image))

    # bincount_vfn = vmap(lambda w: jnp.bincount(dig, w, length=kedges.size+1))
    Psum = jnp.empty((len(multipoles), *Nsum.shape))
    for i_ell, ell in enumerate(multipoles):
        real_weights = W * pk * (2*ell+1) * legendre(ell)(mumesh) # XXX: jax.scipy.special.lpmm not completely implemented yet 
        Psum = Psum.at[i_ell].set(jnp.bincount(dig, weights=real_weights.reshape(-1), length=kedges.size+1))
    # Normalization for powerspectra
    P = (Psum / Nsum).at[:,1:-1].get() * jnp.prod(box_size)
    norm = jnp.prod(mesh_size.astype(jnp.float32))**2

    # Find central values of each bin
    kbins = kedges[:-1] + (kedges[1:] - kedges[:-1]) / 2
    return jnp.concatenate([kbins[None], P / norm])



def kaiser_formula(cosmo, a, pk_init, bias, multipoles=0):
    multipoles = jnp.atleast_1d(multipoles)
    a = jnp.atleast_1d(a)
    beta = growth_rate(cosmo, a) / bias
    pk_init = pk_init * growth_factor(cosmo, a)**2
    # f = growth_rate(cosmo, a)

    pk = np.empty((len(multipoles), *pk_init.shape))
    for i_ell, ell in enumerate(multipoles):
        if ell==0:
            pk[i_ell] = (1 + beta *2/3 + beta**2 /5) * bias**2 * pk_init 
        elif ell==2:
            pk[i_ell] = (beta *4/3 + beta**2 *4/7) * bias**2 * pk_init 
        elif ell==4:
            pk[i_ell] = beta**2 *8/35 * bias**2 * pk_init 
        else: 
            raise NotImplementedError("Handle only multipoles of order 0, 2 ,4") 
    return pk


# def legendre(ell):
#     """
#     Return Legendre polynomial of given order.

#     Reference
#     ---------
#     https://en.wikipedia.org/wiki/Legendre_polynomials
#     """
#     if ell == 0:
#         return lambda x: jnp.ones_like(x)
#     elif ell == 2:
#         return lambda x: 1. / 2. * (3 * x**2 - 1)
#     elif ell == 4:
#         return lambda x: 1. / 8. * (35 * x**4 - 30 * x**2 + 3)
#     else:
#         raise NotImplementedError(f"Legendre polynomial for ell={ell:d} not implemented")
    

# def legendre(ell, x):
#     P0 = lambda x: jnp.ones_like(x)
#     P2 = lambda x: 1 / 2 * (3 * x**2 - 1)
#     P4 = lambda x: 1 / 8 * (35 * x**4 - 30 * x**2 + 3)
#     def error(x):
#         return jnp.full_like(x, jnp.nan)
#     return jnp.piecewise(x, [ell==0, ell==2, ell==4], [P0,P2,P4,error])

