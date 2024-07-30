import numpy as np
import jax.numpy as jnp
from scipy.special import legendre
from jaxpm.growth import growth_rate, growth_factor
from numpyro.diagnostics import effective_sample_size, gelman_rubin
# from blackjax.diagnostics import effective_sample_size



##################
# Power spectrum #
##################
def _initialize_pk(mesh_shape, box_shape, kmin, dk, los):
    kmax = np.pi * np.min(mesh_shape) / np.max(box_shape) + dk / 2
    kedges = np.arange(kmin, kmax, dk)

    kshapes = np.eye(len(mesh_shape), dtype=np.int32) * -2 + 1
    kvec = [(2 * np.pi * m / l) * np.fft.fftfreq(m).reshape(kshape)
            for m, l, kshape in zip(mesh_shape, box_shape, kshapes)]
    kmesh = sum(ki**2 for ki in kvec)**0.5

    dig = np.digitize(kmesh.reshape(-1), kedges)
    ksum = np.bincount(dig, minlength=len(kedges)+1)

    mumesh = sum(ki*losi for ki, losi in zip(kvec, los))
    kmesh_nozeros = np.where(kmesh==0, 1, kmesh) 
    mumesh = mumesh / kmesh_nozeros
    mumesh = np.where(kmesh==0, 0, mumesh)
    
    return dig, ksum, kedges, mumesh


def power_spectrum(field, kmin, dk, mesh_shape, box_shape, los=np.array([0.,0.,1.]), multipoles=0, kcount=False, galaxy_density=1):
    # Initialize values related to powerspectra (wavenumber bins and edges)
    los = np.array(los) / np.linalg.norm(los)
    multipoles = np.atleast_1d(multipoles)
    mesh_shape, box_shape = np.array(mesh_shape), np.array(box_shape)
    dig, ksum, kedges, mumesh = _initialize_pk(mesh_shape, box_shape, kmin, dk, los)

    # Square modulus of FFT
    field_k = jnp.fft.fftn(field, norm='ortho')
    field2_k = jnp.real(field_k * jnp.conj(field_k)) # TODO: cross pk

    L = len(multipoles)
    Psum = jnp.empty((L, *ksum.shape))
    for i_ell, ell in enumerate(multipoles):
        real_weights = field2_k * (2*ell+1) * legendre(ell)(mumesh)
        Psum = Psum.at[i_ell].set(jnp.bincount(dig, weights=real_weights.reshape(-1), length=kedges.size+1))
    # Normalization and convertion from cell units to (Mpc/h)^3
    P = (Psum / ksum)[:,1:-1] * (box_shape / mesh_shape).prod()
    
    covs = ksum[1:-1]
    if False:
        leges = np.moveaxis([legendre(ell)(mumesh) for ell in [0,2,4,6,8]], 0, -1)
        covs = jnp.empty((P.shape[1], L, L))
        for i_ell, elli in enumerate(multipoles):
            for j_ell, ellj in enumerate(multipoles):
                    wig_ell, wig_coeff = wigner3j_square(elli, ellj, prefactor=False)
                    leg_ij = np.sum(leges[wig_ell]*wig_coeff)
                    real_weights = 2*(2*multipoles[:,None]+1) * (1 / galaxy_density**2 + 2*field2_k*leg_ij/galaxy_density) / ksum[1:-1]
                    covs[i_ell, j_ell] = covs.at[i_ell, j_ell].set(jnp.bincount(dig, weights=real_weights.reshape(-1), length=kedges.size+1))


    # Find central values of each bin
    kbins = kedges[:-1] + (kedges[1:] - kedges[:-1]) / 2
    pk = jnp.concatenate([kbins[None], P])
    if kcount:
        return pk, covs
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


###################
# Density regions #
###################
def qbi(x, proba=.95, axis=0, side='med'):
    """
    Compute the Quantile-Based Interval (QBI),
    i.e. the interval of proba `proba` which is

    * the lowest interval if `side=='low'` 
    * the median interval (alias equal-tail interval) if `side=='med'`
    * the highest interval if `side=='high'` 
    """
    if side == 'low':
        p_low = 0
    elif side == 'med':
        p_low = (1-proba)/2
    elif side == 'high':
        p_low = 1-proba

    p_high = p_low + proba
    q_low = jnp.quantile(x, p_low, axis=axis)
    q_high = jnp.quantile(x, p_high, axis=axis)
    return jnp.stack([q_low, q_high], axis=axis)


def qbr(x, proba=.95, side='med'):
    """
    Compute the Quantile-Based Region (QBR), 
    i.e. the spherical region of proba `proba`, where its center on dimension `i` is

    * the lowest value if `side[i]=='low'` 
    * the median value if `side[i]=='med'`
    * the highest value if `side[i]=='high'`

    Return both the region center and radius.
    
    `x` is assumed to be of shape (n_samples, n_dim), and `side` is broadcasted to shape (n_dim,).
    """    
    side = np.broadcast_to(side, x.shape[1])
    center = jnp.empty(x.shape[1])
    for i, s in enumerate(side):
        if s == 'low':
            p_center = 0.
        elif s == 'med':
            p_center = 1/2
        elif s == 'high':
            p_center = 1.
        center = center.at[i].set(jnp.quantile(x[:,i], p_center, axis=0))

    dists = ((x - center)**2).sum(axis=1)**.5
    radius = jnp.quantile(dists, proba, axis=0)
    return center, radius
    

def hdi(x, proba=.95, axis=0):
    """
    Compute the Highest Density Interval (HDI),
    i.e. the smallest interval of proba `proba`.
    """
    x = np.moveaxis(x, axis, 0)
    x_sort = jnp.sort(x, axis=0)
    n = x.shape[0]
    # Round for better estimation at low number of sample, and handle also the case proba close to 1.
    i_length = min(int(jnp.rint(proba * n)), n-1)

    intervals_low = x_sort[: (n - i_length)] # no need to consider all low bounds
    intervals_high = x_sort[i_length:]  # no need to consider all high bounds
    intervals_length = intervals_high - intervals_low # all intervals with given proba
    i_low = intervals_length.argmin(axis=0)
    i_high = i_low + i_length
    hdi_low = jnp.take_along_axis(x_sort, i_low[None], 0)[0]
    hdi_high = jnp.take_along_axis(x_sort, i_high[None], 0)[0]
    return jnp.stack([hdi_low, hdi_high], axis=axis)


def hdr(x, proba=.95):
    """
    Compute the Highest Density Region (HDR),
    i.e. the smallest region of proba `proba`.

    Return both a KDE mesh of the samples density, 
    and the density level corresponding to `proba`.
    """
    pass # TODO, and vectorize over proba






#################
# Chain metrics #
#################
def geomean(x, axis=None):
    return jnp.exp( jnp.log(x).mean(axis=axis) )

def grmean(x, axis=None):
    """cf. https://arxiv.org/pdf/1812.09384"""
    return (1 + geomean(x**2 - 1, axis=axis) )**.5

def multi_ess(x, axis=None):
    return geomean(effective_sample_size(x), axis=axis)

def multi_gr(x, axis=None):
    return grmean(gelman_rubin(x), axis=axis)




sqrerr_moments_fn = lambda m, m_true: (m.mean(axis=(0,1))-m_true)**2
def sqrerr_moments(moments, moments_true):
    # Get mean and std from runs and chains
    m1_hat, m2_hat = moments.mean(axis=(0,1))
    m1, m2 = moments_true
    std_hat, std = (m2_hat - m1_hat**2)**.5, (m2 - m1**2)**.5 # Huygens formula
    # Compute normalized errors
    err_loc, err_scale = (m1_hat - m1) / std, (std_hat - std) / (std / 2**.5) # asymptotically N(0, 1/n_eff)
    mse_loc, mse_scale = (err_loc**2).mean(), (err_scale**2).mean() # asymptotically 1/n_eff * chi^2(d)/d
    return jnp.stack([mse_loc, mse_scale])

def sqrerr_moments2(moments, moments_true):
    # Get mean and std from runs
    n_chains = moments.shape[1]
    m_hat = moments.mean(axis=(0))
    m1_hat, m2_hat = m_hat[:,0], m_hat[:,1]
    m1, m2 = moments_true
    std_hat, std = (m2_hat - m1_hat**2)**.5, (m2 - m1**2)**.5 # Huygens formula
    # Compute normalized errors
    err_loc, err_scale = (m1_hat - m1) / std, (std_hat - std) / (std / 2**.5) # asymptotically N(0, n_chain/n_eff)
    mse_loc, mse_scale = (err_loc**2).mean(), (err_scale**2).mean() # asymptotically n_chain/n_eff * chi^2(d*n_chain)/(d*n_chain) 
    return jnp.stack([mse_loc, mse_scale]) / n_chains # asymptotically 1/n_eff * chi^2(d*n_chain)/(d*n_chain) 