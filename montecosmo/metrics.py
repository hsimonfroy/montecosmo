import numpy as np
import jax.numpy as jnp
from scipy.special import legendre
from jaxpm.growth import growth_rate, growth_factor
from jaxpm.kernels import cic_compensation
from numpyro.diagnostics import effective_sample_size, gelman_rubin
from functools import partial
# from blackjax.diagnostics import effective_sample_size



##################
# Power spectrum #
##################
# def _initialize_pk(mesh_shape, box_shape, kmin, dk, los):
#     kmax = np.pi * np.min(mesh_shape / box_shape) + dk / 2
#     kedges = np.arange(kmin, kmax, dk)

#     kshapes = np.eye(len(mesh_shape), dtype=np.int32) * -2 + 1
#     kvec = [(2 * np.pi * m / l) * np.fft.fftfreq(m).reshape(kshape)
#             for m, l, kshape in zip(mesh_shape, box_shape, kshapes)]
#     kmesh = sum(ki**2 for ki in kvec)**0.5

#     dig = np.digitize(kmesh.reshape(-1), kedges)
#     ksum = np.bincount(dig, minlength=len(kedges)+1)

#     mumesh = sum(ki*losi for ki, losi in zip(kvec, los))
#     kmesh_nozeros = np.where(kmesh==0, 1, kmesh) 
#     mumesh = mumesh / kmesh_nozeros
#     mumesh = np.where(kmesh==0, 0, mumesh)
    
#     return dig, ksum, kedges, mumesh


# def power_spectrum(mesh, box_shape, kmin, dk, los=[0,0,1], multipoles=0, kcount=False, galaxy_density=1):
#     # Initialize values related to powerspectra (wavenumber bins and edges)
#     mesh_shape = np.array(mesh.shape)
#     box_shape, los = np.asarray(box_shape), np.asarray(los)
#     los /= np.linalg.norm(los)
#     multipoles = np.atleast_1d(multipoles)
#     dig, ksum, kedges, mumesh = _initialize_pk(mesh_shape, box_shape, kmin, dk, los)

#     # Square modulus of FFT
#     field_k = jnp.fft.fftn(mesh, norm='ortho')
#     field2_k = jnp.real(field_k * jnp.conj(field_k)) # TODO: cross pk

#     L = len(multipoles)
#     Psum = jnp.empty((L, *ksum.shape))
#     for i_ell, ell in enumerate(multipoles):
#         real_weights = field2_k * (2*ell+1) * legendre(ell)(mumesh)
#         Psum = Psum.at[i_ell].set(jnp.bincount(dig, weights=real_weights.reshape(-1), length=kedges.size+1))
#     # Normalization and convertion from cell units to (Mpc/h)^3
#     P = (Psum / ksum)[:,1:-1] * (box_shape / mesh_shape).prod()
    
#     covs = ksum[1:-1]
#     if False:
#         leges = np.moveaxis([legendre(ell)(mumesh) for ell in [0,2,4,6,8]], 0, -1)
#         covs = jnp.empty((P.shape[1], L, L))
#         for i_ell, elli in enumerate(multipoles):
#             for j_ell, ellj in enumerate(multipoles):
#                     wig_ell, wig_coeff = wigner3j_square(elli, ellj, prefactor=False)
#                     leg_ij = np.sum(leges[wig_ell]*wig_coeff)
#                     real_weights = 2*(2*multipoles[:,None]+1) * (1 / galaxy_density**2 + 2*field2_k*leg_ij/galaxy_density) / ksum[1:-1]
#                     covs[i_ell, j_ell] = covs.at[i_ell, j_ell].set(jnp.bincount(dig, weights=real_weights.reshape(-1), length=kedges.size+1))


#     # Find central values of each bin
#     kbins = kedges[:-1] + (kedges[1:] - kedges[:-1]) / 2
#     pk = jnp.concatenate([kbins[None], P])
#     if kcount:
#         return pk, covs
#     else:
#         return pk




def _initialize_pk(mesh_shape, box_shape, kedges, los):
    """
    Parameters
    ----------
    mesh_shape : tuple of int
        Shape of the mesh grid.
    box_shape : tuple of float
        Physical dimensions of the box.
    kedges : None, int, float, or list
        If None, set dk to twice the minimum.
        If int, specifies number of edges.
        If float, specifies dk.
    los : array_like
        Line-of-sight vector.

    Returns
    -------
    dig : ndarray
        Indices of the bins to which each value in input array belongs.
    kcount : ndarray
        Count of values in each bin.
    kedges : ndarray
        Edges of the bins.
    mumesh : ndarray
        Mu values for the mesh grid.
    """
    kmax = np.pi * np.min(mesh_shape / box_shape) # = knyquist

    if kedges is None or isinstance(kedges, (int, float)):
        if kedges is None:
            dk = 2*np.pi / np.min(box_shape) * 2 # twice the fundamental wavenumber
        if isinstance(kedges, int):
            dk = kmax / kedges # final number of bins will be kedges-1
        elif isinstance(kedges, float):
            dk = kedges
        kedges = np.arange(0, kmax, dk) + dk/2 # from dk/2 to kmax-dk/2

    kshapes = np.eye(len(mesh_shape), dtype=np.int32) * -2 + 1
    kvec = [(2 * np.pi * m / l) * np.fft.fftfreq(m).reshape(kshape)
            for m, l, kshape in zip(mesh_shape, box_shape, kshapes)] # h/Mpc physical units
    kmesh = sum(ki**2 for ki in kvec)**0.5

    dig = np.digitize(kmesh.reshape(-1), kedges)
    kcount = np.bincount(dig, minlength=len(kedges)+1)

    # Central value of each bin
    # kavg = (kedges[1:] + kedges[:-1]) / 2
    kavg = np.bincount(dig, weights=kmesh.reshape(-1), minlength=len(kedges)+1) / kcount
    kavg = kavg[1:-1]

    if los is None:
        mumesh = 1.
    else:
        mumesh = sum(ki*losi for ki, losi in zip(kvec, los))
        kmesh_nozeros = np.where(kmesh==0, 1, kmesh) 
        mumesh = np.where(kmesh==0, 0, mumesh / kmesh_nozeros)

    
    return dig, kcount, kavg, mumesh


def power_spectrum(mesh, mesh2=None, box_shape=None, kedges:int|float|list=None, comp=(False, False), multipoles=0, los=[0.,0.,1.]):
    """
    Compute the auto and cross spectrum of 3D fields, with multipoles.
    """
    # Initialize
    mesh_shape = np.array(mesh.shape)
    if box_shape is None:
        box_shape = mesh_shape
    else:
        box_shape = np.asarray(box_shape)

    if multipoles==0:
        los = None
    else:
        los = np.asarray(los)
        los /= np.linalg.norm(los)
    poles = np.atleast_1d(multipoles)

    if isinstance(comp, int):
        comp = (comp, comp)

    dig, kcount, kavg, mumesh = _initialize_pk(mesh_shape, box_shape, kedges, los)
    n_bins = len(kavg) + 2

    # FFTs
    mesh = jnp.fft.fftn(mesh, norm='ortho')
    if comp[0]:
        kshapes = np.eye(len(mesh_shape), dtype=np.int32) * -2 + 1
        kvec = [2 * np.pi * np.fft.fftfreq(m).reshape(kshape)
            for m, kshape in zip(mesh_shape, kshapes)] # cell units
        # kvec = fftk(mesh_shape)
        mesh *= cic_compensation(kvec) # TODO: rfftn, and remove shot noise before compensation 

    if mesh2 is None:
        mmk = mesh.real**2 + mesh.imag**2
    else:
        mesh2 = jnp.fft.fftn(mesh2, norm='ortho')
        if comp[1]:
            kshapes = np.eye(len(mesh_shape), dtype=np.int32) * -2 + 1
            kvec = [2 * np.pi * np.fft.fftfreq(m).reshape(kshape)
                for m, kshape in zip(mesh_shape, kshapes)] # cell units
            # kvec = fftk(mesh_shape)
            mesh2 *= cic_compensation(kvec)
        mmk = mesh * mesh2.conj()

    # Sum powers
    pk = jnp.empty((len(poles), n_bins))
    for i_ell, ell in enumerate(poles):
        weights = (mmk * (2*ell+1) * legendre(ell)(mumesh)).reshape(-1)
        if mesh2 is None:
            psum = jnp.bincount(dig, weights=weights, length=n_bins)
        else: # XXX: bincount is really slow with complex numbers
            psum_real = jnp.bincount(dig, weights=weights.real, length=n_bins)
            psum_imag = jnp.bincount(dig, weights=weights.imag, length=n_bins)
            psum = (psum_real**2 + psum_imag**2)**.5
        pk = pk.at[i_ell].set(psum)

    # Normalization and conversion from cell units to [Mpc/h]^3
    pk = (pk / kcount)[:,1:-1] * (box_shape / mesh_shape).prod()

    # pk = jnp.concatenate([kavg[None], pk])
    if np.ndim(multipoles)==0:
        return kavg, pk[0]
    else:
        return kavg, pk
  

def transfer(mesh0, mesh1, box_shape, kedges:int | float | list=None):
    pk_fn = partial(power_spectrum, box_shape=box_shape, kedges=kedges)   
    ks, pk0 = pk_fn(mesh0)
    ks, pk1 = pk_fn(mesh1)
    return ks, (pk1 / pk0)**.5


def coherence(mesh0, mesh1, box_shape, kedges:int | float | list=None):
    pk_fn = partial(power_spectrum, box_shape=box_shape, kedges=kedges)   
    ks, pk01 = pk_fn(mesh0, mesh1)
    ks, pk0 = pk_fn(mesh0)
    ks, pk1 = pk_fn(mesh1)
    return ks, pk01 / (pk0 * pk1)**.5


def pktranscoh(mesh0, mesh1, box_shape, kedges:int | float | list=None):
    pk_fn = partial(power_spectrum, box_shape=box_shape, kedges=kedges)
    ks, pk01 = pk_fn(mesh0, mesh1)  
    ks, pk0 = pk_fn(mesh0)
    ks, pk1 = pk_fn(mesh1)
    return ks, pk0, pk1, (pk1 / pk0)**.5, pk01 / (pk0 * pk1)**.5
    









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


# def legendre(ell, x):
#     """
#     Return Legendre polynomial of given order.

#     Reference
#     ---------
#     https://en.wikipedia.org/wiki/Legendre_polynomials
#     """
#     P0 = lambda x: jnp.ones_like(x)
#     P2 = lambda x: 1 / 2 * (3 * x**2 - 1)
#     P4 = lambda x: 1 / 8 * (35 * x**4 - 30 * x**2 + 3)
#     def error(x):
#         return jnp.full_like(x, jnp.nan)
#     # for vmaping on condition, see https://github.com/google/jax/issues/8409
#     return jnp.piecewise(x, [ell==0, ell==2, ell==4], [P0,P2,P4,error])



import math
def wigner3j_square(ellout, ellin, prefactor=True):
    """
    Return the coefficients corresponding to the product of two Legendre polynomials, corresponding to :math:`C_{\ell \ell^{\prime} L}`
    of e.g. eq. 2.2 of https://arxiv.org/pdf/2106.06324.pdf, with :math:`\ell` corresponding to ``ellout`` and :math:`\ell^{\prime}` to ``ellin``.

    Parameters
    ----------
    ellout : int
        Output order.

    ellin : int
        Input order.

    prefactor : bool, default=True
        Whether to include prefactor :math:`(2 \ell + 1)/(2 L + 1)` for window convolution.

    Returns
    -------
    ells : list
        List of mulipole orders :math:`L`.

    coeffs : list
        List of corresponding window coefficients.
    """
    qvals, coeffs = [], []

    def G(p):
        """
        Return the function G(p), as defined in Wilson et al 2015.
        See also: WA Al-Salam 1953
        Taken from https://github.com/nickhand/pyRSD.

        Parameters
        ----------
        p : int
            Multipole order.

        Returns
        -------
        numer, denom: int
            The numerator and denominator.
        """
        toret = 1
        for p in range(1, p + 1): toret *= (2 * p - 1)
        return toret, math.factorial(p)

    for p in range(min(ellin, ellout) + 1):

        numer, denom = [], []

        # numerator of product of G(x)
        for r in [G(ellout - p), G(p), G(ellin - p)]:
            numer.append(r[0])
            denom.append(r[1])

        # divide by this
        a, b = G(ellin + ellout - p)
        numer.append(b)
        denom.append(a)

        numer.append(2 * (ellin + ellout) - 4 * p + 1)
        denom.append(2 * (ellin + ellout) - 2 * p + 1)

        q = ellin + ellout - 2 * p
        if prefactor:
            numer.append(2 * ellout + 1)
            denom.append(2 * q + 1)

        coeffs.append(np.prod(numer, dtype='f8') * 1. / np.prod(denom, dtype='f8'))
        qvals.append(q)

    return qvals[::-1], coeffs[::-1]





#################
# Chain Metrics #
#################

def geomean(x, axis=None):
    return jnp.exp(jnp.mean(jnp.log(x), axis=axis))

def harmean(x, axis=None):
    return 1/jnp.mean(1/x, axis=axis)

def multi_ess(x, axis=None):
    return harmean(effective_sample_size(x), axis=axis)

def multi_gr(x, axis=None):
    """
    In the order of (1+nc/MESS)^(1/2), with nc the number of chains.
    cf. https://arxiv.org/pdf/1812.09384 and MultiESS := HarMean(ESS)
    """
    return jnp.mean(gelman_rubin(x)**2, axis=axis)**.5

