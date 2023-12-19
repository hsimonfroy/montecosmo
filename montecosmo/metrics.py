import numpy as np
import jax.numpy as jnp
from scipy.special import legendre
from jaxpm.growth import growth_rate, growth_factor

def _initialize_pk(mesh_size, box_size, kmin, dk, los):

    W = np.empty(mesh_size, dtype='f4')
    W[...] = 2.0
    W[..., 0] = 1.0
    W[..., -1] = 1.0 # XXX: Why?

    kmax = np.pi * np.min(np.array(mesh_size)) / np.max(np.array(box_size)) + dk / 2
    kedges = np.arange(kmin, kmax, dk)

    kshapes = np.eye(len(mesh_size), dtype='int') * -2 + 1
    kvec = [(2 * np.pi * m / l) * np.fft.fftfreq(m).reshape(kshape)
            for m, l, kshape in zip(mesh_size, box_size, kshapes)]
    kmesh = sum(ki**2 for ki in kvec)**0.5

    dig = np.digitize(kmesh.reshape(-1), kedges)
    Nsum = np.bincount(dig, weights=W.reshape(-1), minlength=len(kedges)+1)

    mumesh = sum(ki*losi for ki, losi in zip(kvec, los))
    kmesh[kmesh == 0] = jnp.inf # modify kmesh, so ensure digitizing kmesh before this
    mumesh = mumesh / kmesh
    return dig, Nsum, W, kedges, mumesh


def power_spectrum(field, kmin, dk, box_size, los=jnp.array([0,0,1]), multipoles=0):
    # Initialize values related to powerspectra (mode bins and weights)
    multipoles = jnp.atleast_1d(multipoles)
    mesh_size = field.shape
    dig, Nsum, W, kedges, mumesh = _initialize_pk(mesh_size, box_size, kmin, dk, los)

    # Absolute value of FFT
    fft_image = jnp.fft.fftn(field)
    pk = jnp.real(fft_image * jnp.conj(fft_image))

    Psum = np.empty((len(multipoles), *Nsum.shape))
    for i_ell, ell in enumerate(multipoles):
        real_weights = W * pk * (2*ell+1) * legendre(ell)(mumesh)
        Psum[i_ell] = jnp.bincount(dig, weights=real_weights.reshape(-1), length=kedges.size+1)

    # Normalization for powerspectra
    P = (Psum / Nsum)[:,1:-1] * np.prod(box_size)
    norm = np.prod(mesh_size)**2

    # Find central values of each bin
    kbins = kedges[:-1] + (kedges[1:] - kedges[:-1]) / 2
    return kbins, P / norm



def kaiser_formula(cosmo, a, pk_init, bias, multipoles=0):
    multipoles = jnp.atleast_1d(multipoles)
    a = jnp.atleast_1d(a)
    beta = growth_rate(cosmo, a) / bias
    pk_init = pk_init * growth_factor(cosmo, a)**2
    # f = growth_rate(cosmo, a)

    pk = np.empty((len(multipoles), *pk_init.shape))
    for i_ell, ell in enumerate(multipoles):
        if ell==0:
            # pk[i_ell] = (2*ell+1)/2 * (2*bias**2*pk_init + 4*bias*f*pk_init/3 + 2*f**2*pk_init/5)
            pk[i_ell] = (1 + beta *2/3 + beta**2 /5) * bias**2 * pk_init 

        elif ell==2:
            # pk[i_ell] = (2*ell+1)/2 * 8/105 * f * pk_init * (7*bias + 3*f)
            pk[i_ell] = (beta *4/3 + beta**2 *4/7) * bias**2 * pk_init 
        elif ell==4:
            # pk[i_ell] = (2*ell+1)/2 * 16/315 * f**2 * pk_init
            pk[i_ell] = beta**2 *8/35 * bias**2 * pk_init 

        else: raise Exception("Handle only multipoles of order 0, 2 ,4") 
    return pk