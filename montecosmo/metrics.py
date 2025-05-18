import numpy as np
import jax.numpy as jnp
from functools import partial

from scipy.special import legendre
from jaxpm.growth import growth_rate, growth_factor
from montecosmo.nbody import rfftk, paint_kernel
from montecosmo.utils import safe_div, ch2rshape

from numpyro.diagnostics import effective_sample_size, gelman_rubin
# from blackjax.diagnostics import effective_sample_size as effective_sample_size2
from jax_cosmo import Cosmology




############
# Spectrum #
############
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


def _waves(mesh_shape, box_shape, kedges, los):
    """
    Parameters
    ----------
    mesh_shape : tuple of int
        Shape of the mesh grid.
    box_shape : tuple of float
        Physical dimensions of the box.
    kedges : None, int, float, or list
        * If None, set dk to twice the fundamental wavenumber.
        * If int, specifies number of edges.
        * If float, specifies dk.
        * If list, specifies kedges.
    los : array_like
        Line-of-sight vector.

    Returns
    -------
    kedges : ndarray
        Edges of the bins.
    kmesh : ndarray
        Wavenumber mesh.
    mumesh : ndarray
        Cosine mesh.
    rfftw : ndarray
        RFFT weights accounting for Hermitian symmetry.
    """
    kmax = np.pi * np.min(mesh_shape / box_shape) # = knyquist

    if isinstance(kedges, (type(None), int, float)):
        if kedges is None:
            dk = 2*np.pi / np.min(box_shape) * 2 # twice the fundamental wavenumber
        if isinstance(kedges, int):
            dk = kmax / kedges # final number of bins will be kedges-1
        elif isinstance(kedges, float):
            dk = kedges
        kedges = np.arange(0, kmax, dk) + dk/2 # from dk/2 to kmax-dk/2

    kvec = rfftk(mesh_shape) # cell units
    kvec = [ki * (m / b) for ki, m, b in zip(kvec, mesh_shape, box_shape)] # h/Mpc physical units
    kmesh = sum(ki**2 for ki in kvec)**0.5

    mumesh = sum(ki * losi for ki, losi in zip(kvec, los))
    mumesh = safe_div(mumesh, kmesh)

    rfftw = np.full_like(kmesh, 2)
    rfftw[..., 0] = 1
    if mesh_shape[-1] % 2 == 0:
        rfftw[..., -1] = 1

    return kedges, kmesh, mumesh, rfftw


def spectrum(mesh, mesh2=None, box_shape=None, kedges:int|float|list=None, 
             deconv:int|tuple=(0, 0), poles:int|tuple=0, box_center:tuple=(0.,0.,0.)):
    """
    Compute the auto and cross spectrum of 3D fields, with multipole.
    """
    # Initialize
    mesh_shape = np.array(mesh.shape)
    if box_shape is None:
        box_shape = mesh_shape
    else:
        box_shape = np.asarray(box_shape)

    los = safe_div(np.asarray(box_center), np.linalg.norm(box_center))
    ells = np.atleast_1d(poles)

    # FFTs and deconvolution
    if isinstance(deconv, int):
        deconv = (deconv, deconv)

    mesh = jnp.fft.rfftn(mesh, norm='ortho')
    kvec = rfftk(mesh_shape) # cell units
    mesh /= paint_kernel(kvec, order=deconv[0])

    if mesh2 is None:
        mmk = mesh.real**2 + mesh.imag**2
    else:
        mesh2 = jnp.fft.rfftn(mesh2, norm='ortho')
        mesh2 /= paint_kernel(kvec, order=deconv[1])
        mmk = mesh * mesh2.conj()

    # Binning
    kedges, kmesh, mumesh, rfftw = _waves(mesh_shape, box_shape, kedges, los)
    n_bins = len(kedges) + 1
    dig = np.digitize(kmesh.reshape(-1), kedges)

    # Count wavenumber in bins
    kcount = np.bincount(dig, weights=rfftw.reshape(-1), minlength=n_bins)
    kcount = kcount[1:-1]

    # Average wavenumber values in bins
    # kavg = (kedges[1:] + kedges[:-1]) / 2
    kavg = np.bincount(dig, weights=(kmesh * rfftw).reshape(-1), minlength=n_bins)
    kavg = kavg[1:-1] / kcount

    # Average wavenumber power in bins
    pow = jnp.empty((len(ells), n_bins))
    for i_ell, ell in enumerate(ells):
        weights = (mmk * (2*ell+1) * legendre(ell)(mumesh) * rfftw).reshape(-1)
        if mesh2 is None:
            psum = jnp.bincount(dig, weights=weights, length=n_bins)
        else: 
            # NOTE: bincount is really slow with complex numbers, so bincount real and imag parts
            psum_real = jnp.bincount(dig, weights=weights.real, length=n_bins)
            psum_imag = jnp.bincount(dig, weights=weights.imag, length=n_bins)
            psum = (psum_real**2 + psum_imag**2)**.5
        pow = pow.at[i_ell].set(psum)
    pow = pow[:,1:-1] / kcount * (box_shape / mesh_shape).prod() # from cell units to [Mpc/h]^3

    # kpow = jnp.concatenate([kavg[None], pk])
    if poles==0:
        return kavg, pow[0]
    else:
        return kavg, pow



def transfer(mesh0, mesh1, box_shape, kedges:int|float|list=None, deconv=(0, 0)):
    if isinstance(deconv, int):
        deconv = (deconv, deconv)
    pow_fn = partial(spectrum, box_shape=box_shape, kedges=kedges)
    ks, pow0 = pow_fn(mesh0, deconv=deconv[0])
    ks, pow1 = pow_fn(mesh1, deconv=deconv[1])
    return ks, (pow1 / pow0)**.5


def coherence(mesh0, mesh1, box_shape, kedges:int|float|list=None, deconv=(0, 0)):
    if isinstance(deconv, int):
        deconv = (deconv, deconv)
    pow_fn = partial(spectrum, box_shape=box_shape, kedges=kedges)
    ks, pow01 = pow_fn(mesh0, mesh1, deconv=deconv)  
    ks, pow0 = pow_fn(mesh0, deconv=deconv[0])
    ks, pow1 = pow_fn(mesh1, deconv=deconv[1])
    return ks, pow01 / (pow0 * pow1)**.5


def powtranscoh(mesh0, mesh1, box_shape, kedges:int|float|list=None, deconv=(0, 0), scale_coh:float=1.):
    if isinstance(deconv, int):
        deconv = (deconv, deconv)
    pow_fn = partial(spectrum, box_shape=box_shape, kedges=kedges)
    ks, pow01 = pow_fn(mesh0, mesh1, deconv=deconv)  
    ks, pow0 = pow_fn(mesh0, deconv=deconv[0])
    ks, pow1 = pow_fn(mesh1, deconv=deconv[1])
    trans = (pow1 / pow0)**.5
    coh = scale_coh * pow01 / (pow0 * pow1)**.5
    return ks, pow1, trans, coh
    




def deconv_paint(mesh, order=2):
    """
    Deconvolve the mesh by the paint kernel of given order.
    """
    if jnp.isrealobj(mesh):
        kvec = rfftk(mesh.shape)
        mesh = jnp.fft.rfftn(mesh)
        mesh /= paint_kernel(kvec, order)
        mesh = jnp.fft.irfftn(mesh)
    else:
        kvec = rfftk(ch2rshape(mesh.shape))
        mesh /= paint_kernel(kvec, order)
    return mesh






def kaiser_formula(cosmo:Cosmology, a, lin_kpow, bE, poles=0):
    """
    bE is the Eulerien linear bias
    """
    poles = jnp.atleast_1d(poles)
    beta = growth_rate(cosmo, a) / bE
    k, pow = lin_kpow
    pow *= growth_factor(cosmo, a)**2

    weights = np.ones(len(poles)) * bE**2
    for i_ell, ell in enumerate(poles):
        if ell==0:
            weights[i_ell] *= (1 + beta * 2/3 + beta**2 /5)
        elif ell==2:
            weights[i_ell] *= (beta * 4/3 + beta**2 *4/7) 
        elif ell==4:
            weights[i_ell] *= beta**2 * 8/35
        else: 
            raise NotImplementedError(
                "Handle only poles of order ell=0, 2 ,4. ell={ell} not implemented.")
        
    pow = jnp.moveaxis(pow[...,None] * weights, -1, -2)
    return k, pow


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
    r"""
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
    return 1 / jnp.mean(1 / x, axis=axis)

def multi_ess(x, axis=None):
    return harmean(effective_sample_size(x), axis=axis)

def multi_gr(x, axis=None):
    """
    In the order of (1+nc/mESS)^(1/2), with nc the number of chains.
    cf. https://arxiv.org/pdf/1812.09384 and mESS := HarMean(ESS)
    """
    return jnp.mean(gelman_rubin(x)**2, axis=axis)**.5

