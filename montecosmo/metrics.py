import numpy as np
from jax import numpy as jnp
from functools import partial

from scipy.special import legendre, lpmv, factorial
from montecosmo.nbody import rfftk, rectangular_hat, a2g, a2f
from montecosmo.utils import safe_div, ch2rshape, cart2radecrad

from numpyro.diagnostics import effective_sample_size, gelman_rubin
# from blackjax.diagnostics import effective_sample_size as effective_sample_size2
from jax_cosmo import Cosmology




############
# Spectrum #
############
# def power_spectrum(mesh, box_size, kmin, dk, los=[0,0,1], multipoles=0, kcount=False, galaxy_density=1):
#     # Initialize values related to powerspectra (wavenumber bins and edges)
#     mesh_shape = np.array(mesh.shape)
#     box_size, los = np.asarray(box_size), np.asarray(los)
#     los /= np.linalg.norm(los)
#     multipoles = np.atleast_1d(multipoles)
#     dig, ksum, kedges, mumesh = _initialize_pk(mesh_shape, box_size, kmin, dk, los)

#     # Square modulus of FFT
#     field_k = jnp.fft.fftn(mesh, norm='ortho')
#     field2_k = jnp.real(field_k * jnp.conj(field_k)) # TODO: cross pk

#     L = len(multipoles)
#     Psum = jnp.empty((L, *ksum.shape))
#     for i_ell, ell in enumerate(multipoles):
#         real_weights = field2_k * (2*ell+1) * legendre(ell)(mumesh)
#         Psum = Psum.at[i_ell].set(jnp.bincount(dig, weights=real_weights.reshape(-1), length=kedges.size+1))
#     # Normalization and convertion from cell units to (Mpc/h)^3
#     P = (Psum / ksum)[:,1:-1] * (box_size / mesh_shape).prod()
    
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


def _waves(mesh_shape, box_size, kedges, include_corners, los):
    """
    Parameters
    ----------
    mesh_shape : tuple of int
        Shape of the mesh grid.
    box_size : tuple of float
        Physical dimensions of the box.
    kedges : None, int, float, or list
        * If None, set dk to sqrt(dim) times the fundamental wavenumber.
          It is the minimum dk to guarantee connected shell bins.
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
    if isinstance(kedges, (type(None), int, float)):
        dim = len(mesh_shape)
        kmin = 0.
        kmax = np.pi * (mesh_shape / box_size).min() # = k_nyquist
        if include_corners:
            kmax *= dim**.5 * 0.95
            
        if kedges is None:
            dk = dim**.5 * 2 * np.pi / box_size.min() # sqrt(d) times fundamental
            n_kedges = max(int((kmax - kmin) / dk), 1)
        elif isinstance(kedges, int):
            n_kedges = kedges # final number of bins will be nedges-1
        elif isinstance(kedges, float):
            n_kedges = max(int((kmax - kmin) / kedges), 1)
        dk = (kmax - kmin) / n_kedges
        kedges = np.linspace(kmin, kmax, n_kedges, endpoint=False)
        kedges += dk / 2 # from kmin+dk/2 to kmax-dk/2

    kvec = rfftk(mesh_shape, box_size) # in h/Mpc
    kmesh = sum(ki**2 for ki in kvec)**.5
    mumesh = sum(ki * losi for ki, losi in zip(kvec, los))
    mumesh = safe_div(mumesh, kmesh)

    rfftw = np.full_like(kmesh, 2)
    rfftw[..., 0] = 1
    if mesh_shape[-1] % 2 == 0:
        rfftw[..., -1] = 1

    return kedges, kmesh, mumesh, rfftw


def _spectrum(mesh0, mesh1=None, box_size=None, box_center:tuple=(0.,0.,0.), ells:int|list=0,
              kedges:int|float|list=None, include_corners=True, deconv:int|tuple=(0, 0)):
    """
    Compute the auto and cross spectrum of 3D fields, with multipole.
    """
    # Initialize
    box_center = np.asarray(box_center)
    los = safe_div(box_center, np.linalg.norm(box_center))

    if isinstance(deconv, int):
        deconv = (deconv, deconv)

    # FFTs and deconvolution
    if jnp.isrealobj(mesh0):
        mesh_shape = np.array(mesh0.shape)
        mesh0 = jnp.fft.rfftn(mesh0)
    else:
        mesh_shape = np.array(ch2rshape(mesh0.shape))

    kvec = rfftk(mesh_shape) # in cell units
    mesh0 /= rectangular_hat(kvec, order=deconv[0])

    if mesh1 is None:
        mmk = mesh0.real**2 + mesh0.imag**2
    else:
        if jnp.isrealobj(mesh1):
            mesh1 = jnp.fft.rfftn(mesh1)
        mesh1 /= rectangular_hat(kvec, order=deconv[1])
        mmk = mesh0 * mesh1.conj()

    # Binning
    box_size = mesh_shape if box_size is None else np.asarray(box_size)
    kedges, kmesh, mumesh, rfftw = _waves(mesh_shape, box_size, kedges, include_corners, los)
    n_bins = len(kedges) + 1
    dig = np.digitize(kmesh.reshape(-1), kedges)

    # Count wavenumber in bins
    kcount = np.bincount(dig, weights=rfftw.reshape(-1), minlength=n_bins)[1:-1]

    # Average wavenumber values in bins
    # kmean = (kedges[1:] + kedges[:-1]) / 2
    kmean = np.bincount(dig, weights=(kmesh * rfftw).reshape(-1), minlength=n_bins)[1:-1]
    kmean /= kcount

    # Average wavenumber power in bins
    for ell in np.atleast_1d(ells):
        weights = (mmk * (2*ell+1) * legendre(ell)(mumesh) * rfftw).reshape(-1)
        if mesh1 is None:
            pmean = jnp.bincount(dig, weights=weights, length=n_bins)[1:-1]
        else: 
            # NOTE: bincount is really slow with complex numbers, so bincount real and imag parts
            pmean_real = jnp.bincount(dig, weights=weights.real, length=n_bins)[1:-1]
            pmean_imag = jnp.bincount(dig, weights=weights.imag, length=n_bins)[1:-1]
            pmean = (pmean_real**2 + pmean_imag**2)**.5
        pmean *= (box_size / mesh_shape**2).prod() / kcount # from cell units to [Mpc/h]^3
        pow |= {ell: pmean}

    if isinstance(ells, int):
        return kcount, kmean, pow[ells]
    else:
        return kcount, kmean, pow

def spectrum(mesh0, mesh1=None, box_size=None, box_center:tuple=(0.,0.,0.), ells:int|list=0, 
             kedges:int|float|list=None, include_corners=True):
    kcount, kmean, pow = _spectrum(mesh0, mesh1, box_size, box_center, ells, kedges, include_corners)
    return kmean, pow


def transfer(mesh0, mesh1, box_size, kedges:int|float|list=None, deconv=(0, 0)):
    if isinstance(deconv, int):
        deconv = (deconv, deconv)
    pow_fn = partial(spectrum, box_size=box_size, kedges=kedges)
    ks, pow0 = pow_fn(mesh0, deconv=deconv[0])
    ks, pow1 = pow_fn(mesh1, deconv=deconv[1])
    return ks, (pow1 / pow0)**.5

def coherence(mesh0, mesh1, box_size, kedges:int|float|list=None, deconv=(0, 0)):
    if isinstance(deconv, int):
        deconv = (deconv, deconv)
    pow_fn = partial(spectrum, box_size=box_size, kedges=kedges)
    ks, pow01 = pow_fn(mesh0, mesh1, deconv=deconv)  
    ks, pow0 = pow_fn(mesh0, deconv=deconv[0])
    ks, pow1 = pow_fn(mesh1, deconv=deconv[1])
    return ks, pow01 / (pow0 * pow1)**.5

def powtranscoh(mesh0, mesh1, box_size, kedges:int|float|list=None, deconv=(0, 0)):
    if isinstance(deconv, int):
        deconv = (deconv, deconv)
    pow_fn = partial(spectrum, box_size=box_size, kedges=kedges)
    ks, pow01 = pow_fn(mesh0, mesh1, deconv=deconv)  
    ks, pow0 = pow_fn(mesh0, deconv=deconv[0])
    ks, pow1 = pow_fn(mesh1, deconv=deconv[1])
    trans = (pow1 / pow0)**.5
    coh = pow01 / (pow0 * pow1)**.5
    return ks, pow1, trans, coh
    


def bin_and_aggregate(targets, values, vedges:int|float|list, min_count=1, aggr_fn=None):
    """
    bin and aggregate target based on value.
    If min_count is None, vedges is interpreted in quantile space.
    If aggr_fn is None, aggregate by averaging.
    """
    targets, values = targets.reshape(-1), values.reshape(-1)
    assert len(targets) == len(values), "targets and values must have same length."

    if isinstance(vedges, (int, float)):
        vmin, vmax = (0., 1.) if min_count is None else (values.min(), values.max())
        if isinstance(vedges, int):
            n_vedges = vedges # final number of bins will be n_edges-1
        elif isinstance(vedges, float):
            n_vedges = max(int((vmax - vmin) / vedges), 1)
        dv = (vmax - vmin) / n_vedges
        vedges = np.linspace(vmin, vmax, n_vedges, endpoint=False)
        vedges += dv / 2 # from vmin+dv/2 to vmax-dv/2

    if min_count is None: # use quantile spacing
        vedges = np.quantile(values, q=vedges) # ~ len(mesh) / n_edges values per bin
        min_count = 1

    n_bins = len(vedges) + 1
    dig = np.digitize(values, vedges)
    vcount = np.bincount(dig, minlength=n_bins)[1:-1]
    count_mask = vcount >= min_count
    vcount = vcount[count_mask]

    vmean = np.bincount(dig, weights=values, minlength=n_bins)[1:-1]
    vmean = vmean[count_mask] / vcount

    if aggr_fn is None: # aggregate by averaging
        taggr = np.bincount(dig, weights=targets, minlength=n_bins)[1:-1]
        taggr = taggr[count_mask] / vcount
    else:
        taggr = []
        for i_bin in range(1, n_bins-1):
            bin_mask = dig == i_bin
            taggr.append(aggr_fn(targets[bin_mask]))
        taggr = np.array(taggr)[count_mask]

    return vcount, vmean, taggr


def mse_radius(mesh0, mesh1, rmesh, box_size, redges:int|float|list=None, aggr_fn=None):
    """
    Compute mean squared error between mesh0 and mesh1, binned by radius, in (Mpc/h)^3.
    """
    cell_vol = (np.array(box_size) / np.array(mesh0.shape)).prod()
    if redges is None:
        redges = 3**.5 * cell_vol**(1/3) # NOTE: minimum dr to guarantee connected shell bins.
    se = (mesh0 - mesh1)**2 * cell_vol

    rcount, rmean, mse = bin_and_aggregate(se, rmesh, redges, aggr_fn=aggr_fn)
    return rcount, rmean, mse

def mse_value(mesh0, mesh1, box_size, vedges:int|float|list, min_count=None, aggr_fn=None):
    """
    Compute mean squared error between mesh0 and mesh1, binned by value of mesh0, in (Mpc/h)^3.
    """
    cell_vol = (np.array(box_size) / np.array(mesh0.shape)).prod()
    se = (mesh0 - mesh1)**2 * cell_vol

    vcount, vmean, mse = bin_and_aggregate(se, mesh0, vedges, min_count=min_count, aggr_fn=aggr_fn)
    return vcount, vmean, mse

def mse_wave(mesh0, mesh1, box_size, kedges:int|float|list=None, include_corners=True):
    """
    Compute mean squared error between mesh0 and mesh1, binned by wave number, in (Mpc/h)^3.
    """
    # if min_count is None: # use linear spacing in quantile space
    #     from montecosmo.utils import volume_hypersphere, surface_hypersphere
    #     from montecosmo.nbody import rfftk
    #     mesh_shape = np.array(mesh0.shape)
    #     if isinstance(kedges, (type(None), int, float)):
    #         dim = len(mesh_shape)
    #         kmin = 0.
    #         if include_corners:
    #             kmax = 1.
    #         else:
    #             kmax = volume_hypersphere(dim) / 2**dim
    #             kmax *= (mesh_shape.min(keepdims=True) / mesh_shape).prod() # = k_nyquist quantile

    #         if kedges is None:
    #             min_count = surface_hypersphere(dim, R=mesh_shape.min() / 2)* dim**.5 # ~ count in a shell of thickness sqrt(d) at k_nyquist
    #             n_kedges = max(int(mesh0.size / min_count), 1)
    #         if isinstance(kedges, int):
    #             n_kedges = kedges # final number of bins will be n_edges-1
    #         elif isinstance(kedges, float):
    #             n_kedges = max(int((kmax - kmin) / kedges), 1)
    #         dk = (kmax - kmin) / n_kedges
    #         kedges = np.linspace(kmin, kmax, n_kedges, endpoint=False)
    #         kedges += dk / 2 # from kmin+dk/2 to kmax-dk/2
        
    #     kvec = rfftk(mesh_shape, box_size)
    #     kmesh = sum(ki**2 for ki in kvec)**.5
    #     kedges = np.quantile(kmesh.reshape(-1), q=kedges) # ~ len(mesh) / n_edges values per bin
    #     min_count = 1

    kcount, kmean, pow = _spectrum(mesh1 - mesh0, box_size=box_size, kedges=kedges, include_corners=include_corners)
    return kcount, kmean, pow


def mean_errorbar(count, std, confidence=0.95, gaussian_approx=False):
    """
    Error bar on mean assuming Gaussian variables.
    """
    from scipy.stats import norm, t
    if not gaussian_approx:
        df = count - 1
        low, high = t(df=df).interval(confidence)
        low, high = std / df**.5 * low, std / df**.5 * high
        yerr = jnp.stack((-low, high))
    else:
        high = norm.interval(confidence)[1] * std / count**.5
        yerr = jnp.stack((high, high))
    return yerr

def var_errorbar(count, var, confidence=0.95, gaussian_approx=False):
    """
    Error bar on variance assuming Gaussian variables.
    """
    from scipy.stats import chi2, norm
    if not gaussian_approx:
        low, high = chi2(df=count).interval(confidence)
        low, high = var * count / high, var * count / low
        low, high = var - low, high - var
        yerr = jnp.stack((low, high))
    else:
        high = var * (2 / count)**.5 * norm.interval(confidence)[1]
        yerr = jnp.stack((high, high))
    return yerr




def kaiser_formula(cosmo:Cosmology, a, lin_kpow, b1E, ells=0):
    """
    b1E is the Eulerien linear bias
    """
    ells = jnp.atleast_1d(ells)
    beta = a2f(cosmo, a) / b1E
    k, pow = lin_kpow
    pow *= a2g(cosmo, a)**2

    weights = np.ones(len(ells)) * b1E**2
    for i_ell, ell in enumerate(ells):
        if ell==0:
            weights[i_ell] *= (1 + beta * 2/3 + beta**2 /5)
        elif ell==2:
            weights[i_ell] *= (beta * 4/3 + beta**2 *4/7) 
        elif ell==4:
            weights[i_ell] *= beta**2 * 8/35
        else: 
            raise NotImplementedError(
                f"Handle only poles of degree ell=0, 2 ,4. ell={ell} not implemented.")
        
    pow = jnp.moveaxis(pow[...,None] * weights, -1, -2)
    return k, pow


def real_sph_harm(l, m, theta, phi):
    """
    Compute real spherical harmonics Y(l, m, theta, phi).
    l: degree/azimuthal number (l >= 0)
    m: order/magnetic number (-l <= m <= l)
    theta: polar angle/colatitude [0, pi]
    phi: azimuthal angle/longitude [0, 2*pi]
    """
    m_abs = abs(m)
    norm = ((2*l+1) / (4*np.pi) * factorial(l-m_abs) / factorial(l+m_abs))**.5
    asso_legendre = lpmv(m_abs, l, np.cos(theta))
    if m > 0:
        ylm = 2**.5 * norm * asso_legendre * np.cos(m * phi)
    elif m < 0:
        ylm = 2**.5 * norm * asso_legendre * np.sin(m_abs * phi)
    else:
        ylm = norm * asso_legendre
    return ylm


def naive_mu2_delta(mesh, los):
    mesh_shape = ch2rshape(mesh.shape)
    kvec = rfftk(mesh_shape)
    kmesh = sum(kk**2 for kk in kvec)**.5 # in cell units

    mu_delta = jnp.stack([jnp.fft.irfftn(
            safe_div(kvec[i] * mesh, kmesh)
            ) for i in range(3)], axis=-1)
    mu_delta = (mu_delta * los).sum(-1)
    mu_delta = jnp.fft.rfftn(mu_delta)

    mu2_delta = jnp.stack([jnp.fft.irfftn(
            safe_div(kvec[i] * mu_delta, kmesh)
            ) for i in range(3)], axis=-1)
    mu2_delta = (mu2_delta * los).sum(-1)
    return mu2_delta


def optim_mu2_delta(mesh, los):
    """
    Exploit the fact that mu^2 can be expressed as a sum of Legendre polynomials, 
    which themselves can be expressed as spherical harmonics.
    .. math::
        mu^2 = 1/3 L_0(mu) + 2/3 L_2(mu) = 1/3 + 8pi/15 sum_{m=-2}^{2} Y_{2m}(k) Y*_{2m}(r)

    For a related computation, see [Hand+2017](https://arxiv.org/pdf/1704.02357)
    """
    mesh_shape = ch2rshape(mesh.shape)
    kvec = rfftk(mesh_shape)

    ra, dec, _ = cart2radecrad(los)
    phi = np.deg2rad(ra).reshape(-1)
    theta = np.deg2rad(90. - dec).reshape(-1)

    kra, kdec, _ = cart2radecrad(jnp.stack(jnp.broadcast_arrays(*kvec), -1))
    kphi = np.deg2rad(kra).reshape(-1)
    ktheta = np.deg2rad(90. - kdec).reshape(-1)

    delta = jnp.fft.irfftn(mesh)
    mu2_delta = delta / 3
    for m in range(-2, 3):
        # In real space
        ylos = real_sph_harm(jnp.array([2]), jnp.array([m]), theta, phi).reshape(mesh_shape)
        yk = real_sph_harm(jnp.array([2]), jnp.array([m]), ktheta, kphi).reshape(mesh.shape)
        yk = jnp.fft.irfftn(yk * mesh)

        # In Fourier space
        # ylos = real_sph_harm(jnp.array([2]), jnp.array([m]), theta, phi).reshape(mesh_shape)
        # ylos = jnp.fft.rfftn(ylos * delta)
        # yk = real_sph_harm(jnp.array([2]), jnp.array([m]), ktheta, kphi).reshape(init_mesh.shape)
        mu2_delta += 8 * jnp.pi / 15 * ylos * yk
    return delta, mu2_delta


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
# Distributions #
#################
def distr_radial(mesh, rmesh, box_size, redges:int|float|list=None, aggr_fn=None):
    """
    Compute the radial distribution of the given mesh in (h/Mpc)^3.
    """
    cell_vol = (np.array(box_size) / np.array(mesh.shape)).prod()
    if redges is None:
        redges = 3**.5 * cell_vol**(1/3) # NOTE: minimum dr to guarantee connected shell bins.

    rcount, rmean, maggr = bin_and_aggregate(mesh, rmesh, redges, aggr_fn=aggr_fn)
    return rcount, rmean, maggr / cell_vol

def distr_angular():
    """
    Compute the angular distribution of the given mesh in (h/Mpc)^3.
    """
    pass


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

