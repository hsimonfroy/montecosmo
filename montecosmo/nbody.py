from functools import partial
from itertools import product
import numpy as np

from jax import numpy as jnp, tree, debug, lax
from jax_cosmo import Cosmology, background
from jax_cosmo.scipy.ode import odeint
from montecosmo.utils import ch2rshape, r2chshape, safe_div


###########
# Kernels #
###########
def rfftk(shape):
    """
    Return wavevectors in cell units for rfftn.
    """
    kx = np.fft.fftfreq(shape[0]) * 2 * np.pi
    ky = np.fft.fftfreq(shape[1]) * 2 * np.pi
    kz = np.fft.rfftfreq(shape[2]) * 2 * np.pi

    kx = kx.reshape([-1, 1, 1])
    ky = ky.reshape([1, -1, 1])
    kz = kz.reshape([1, 1, -1])
    return kx, ky, kz


def fftk(shape):
    """
    Return wavevectors in cell units for fftn.
    (You shouldn't need it)
    """
    kx = np.fft.fftfreq(shape[0]) * 2 * np.pi
    ky = np.fft.fftfreq(shape[1]) * 2 * np.pi
    kz = np.fft.fftfreq(shape[2]) * 2 * np.pi

    kx = kx.reshape([-1, 1, 1])
    ky = ky.reshape([1, -1, 1])
    kz = kz.reshape([1, 1, -1])
    return kx, ky, kz


def invlaplace_hat(kvec, fd=False):
    """
    Fourier transform of inverse Laplace kernel.

    Parameters
    -----------
    kvec: list
        List of wavevectors
    fd: bool
        Finite difference kernel

    Returns
    --------
    weights: array
        Complex kernel values
    """
    if fd:
        kk = sum((ki * np.sinc(ki / (2 * np.pi)))**2 for ki in kvec)
    else:
        kk = sum(ki**2 for ki in kvec)
    return - safe_div(1, kk)


def gradient_hat(kvec, direction:int, fd=False):
    """
    Fourier transform of gradient kernel along given direction.
    
    Parameters
    -----------
    kvec: list
        List of wavevectors
    direction: int
        Index of the direction in which to take the gradient
    fd: bool
        Finite difference kernel

    Returns
    --------
    weights: array
        Complex kernel values
    """
    ki = kvec[direction]
    if fd:
        ki = (8. * np.sin(ki) - np.sin(2. * ki)) / 6.
    return 1j * ki


def gaussian_hat(kvec, kcut=np.inf):
    """
    Fourier transform of gaussian kernel.
    
    Parameters
    -----------
    kvec: list
        List of wavevectors
    kcut: float
        Cutoff wavenumber for the gaussian kernel

    Returns
    --------
    weights: array
        Complex kernel values
    """
    if kcut == np.inf:
        return 1.
    else:
        kk = sum(ki**2 for ki in kvec)
        # return np.exp(-kk / kcut**2)
        rcut = 2 * np.pi / kcut
        return np.exp(-kk * rcut**2 / 2)


def top_hat(kvec, kcut=np.inf):
    """
    Top-hat kernel in Fourier domain (isotropic). 

    Note that it is more efficient to compute 
    `where(top_hat(...), mesh, 0.)`
    than `top_hat(...) * mesh`.

    NB: does this mean that the "top" function is the Airy pattern? #xptdr
    
    Parameters
    -----------
    kvec: list
        List of wavevectors
    kcut: float
        Cutoff wavenumber for the gaussian kernel

    Returns
    --------
    weights: array
        Complex kernel values
    """
    if kcut == np.inf:
        return 1.
    else:
        kk = sum(ki**2 for ki in kvec)
        return np.where(kk < kcut**2, True, False)


def rectangular(s, order):
    """
    Rectangular kernel with given order.

    Parameters
    ----------
    kvec: list
        List of wavevectors
    order: int
        order of the kernel
        * 0: Dirac
        * 1: Nearest Grid Point (NGP)
        * 2: Cloud-In-Cell (CIC)
        * 3: Triangular-Shape Cloud (TSC)
        * 4: Piecewise-Cubic Spline (PCS)

        cf. [Sefusatti+2017](http://arxiv.org/abs/1512.07295), 
        [List&Hahn2024](https://arxiv.org/abs/2309.10865)
    """
    funclist = [
        lambda s: jnp.full(jnp.shape(s)[-1:], jnp.inf), # Dirac
        lambda s: jnp.full(jnp.shape(s)[-1:], 1.), # NGP
        lambda s: 1 - s, # CIC
        lambda s: (s <= 1/2) * (3/4 - s**2) + (1/2 < s) / 2 * (3/2 - s)**2, # TSC
        lambda s: (s <= 1) / 6 * (4 - 6 * s**2 + 3 * s**3) + (1 < s) / 6 * (2 - s)**3, # PCS
    ]
    return funclist[order](jnp.abs(s))


def rectangular_hat(kvec, order:int=2):
    """
    Fourier transform of rectangular kernel with given order.

    Parameters
    ----------
    kvec: list
        List of wavevectors
    order: int
        order of the kernel
        * 0: Dirac
        * 1: Nearest Grid Point (NGP)
        * 2: Cloud-In-Cell (CIC)
        * 3: Triangular-Shape Cloud (TSC)
        * 4: Piecewise-Cubic Spline (PCS)

        cf. [Sefusatti+2017](http://arxiv.org/abs/1512.07295), 
        [List&Hahn2024](https://arxiv.org/abs/2309.10865)

    Returns
    -------
    weights: array
        Complex kernel values
    """
    kernel = lambda k: np.sinc(k / (2 * np.pi))**order
    out = 1.
    for ki in kvec:
        out = out * kernel(ki)
    return out


def kaiser_bessel(s, order, kcut):
    """
    Kaiser-Bessel kernel.

    See [Barnet+2019](http://arxiv.org/abs/1808.06736)
    """
    s = s * 2 / order
    kcut = kcut * order / 2
    out = jnp.i0(kcut * (1 - s**2)**.5)
    out /= order * jnp.sinh(kcut) / kcut
    return out


def kaiser_bessel_hat(kvec, order, kcut):
    """
    Fourier transform of Kaiser-Bessel kernel.

    See [Barnet+2019](http://arxiv.org/abs/1808.06736)
    """
    def kernel(k, kcut):
        k = k * order / 2
        kcut = kcut * order / 2
        dist = jnp.abs(kcut**2 - k**2)**.5
        bulk = jnp.sinh(dist) / dist
        tail = jnp.sin(dist) / dist
        out = jnp.where(jnp.abs(k) <= kcut, bulk, tail)
        out /= jnp.sinh(kcut) / kcut
        return out

    out = 1.
    for ki in kvec:
        out = out * kernel(ki, kcut)
    return out


def deconv_paint(mesh, order:int=2, kernel_type='rectangular', oversamp=1.):
    """
    Deconvolve the mesh by the paint kernel of given order and type.
    """
    if kernel_type == 'rectangular':
        kernel = lambda kvec: rectangular_hat(kvec, order)
    elif kernel_type == 'kaiser_bessel':
        kernel = lambda kvec: kaiser_bessel_hat(kvec, order, optim_kcut(oversamp))
    
    if jnp.isrealobj(mesh):
        kvec = rfftk(mesh.shape)
        mesh = jnp.fft.rfftn(mesh)
        mesh /= kernel(kvec)
        mesh = jnp.fft.irfftn(mesh)
    else:
        kvec = rfftk(ch2rshape(mesh.shape))
        mesh /= kernel(kvec)
    return mesh


###################
# Mass-assignment #
###################
# See also https://github.com/adematti/jax-power/blob/45edcda356d29f337fc276044e77cf3363b92820/jaxpower/resamplers.py

# def cic_alpha(s, alpha=1.):
#     m = (1 + alpha) / 2
#     d = jnp.abs(1 - alpha) / 2
#     out = (s <= d) * jnp.minimum(1, alpha) + ((d < s) & (s <= m)) * (m - s)
#     return out / alpha

# paint_kernels2 = [
#     lambda s: jnp.full(jnp.shape(s)[-1:], jnp.inf), # Dirac
#     lambda s: jnp.full(jnp.shape(s)[-1:], 1.), # NGP
#     cic_alpha, # CIC
#     lambda s: (s <= 1/2) * (3/4 - s**2) + (1/2 < s) / 2 * (3/2 - s)**2, # TSC
#     lambda s: (s <= 1) / 6 * (4 - 6 * s**2 + 3 * s**3) + (1 < s) / 6 * (2 - s)**3, # PCS
# ]


def optim_kcut(oversamp, safety=0.98):
    """
    Optimal wavenumber cutoff for Prolate Spheroidal Wave Function-like (PSWF-like) kernels.

    See [Barnet+2019](http://arxiv.org/abs/1808.06736)
    """
    return safety * jnp.pi * (2 - 1 / oversamp)

def paint(pos, shape:tuple, weights=1., order:int=2, kernel_type='rectangular', oversamp=1.):
    """
    Paint the positions onto a mesh of given shape. 
    """
    dtype = 'int16' # int16 -> +/- 32_767, trkl
    shape = np.asarray(shape, dtype=dtype)
    mesh = jnp.zeros(shape)
    def wrap(idx):
        return idx % shape
    
    id0 = (jnp.round if order % 2 else jnp.floor)(pos).astype(dtype)
    ishifts = np.arange(order) - (order - 1) // 2
    ishifts = np.array(list(product(* len(shape) * (ishifts,))), dtype=dtype)

    if kernel_type == 'rectangular':
        kernel = lambda s: rectangular(s, order)
    elif kernel_type == 'kaiser_bessel':
        kernel = lambda s: kaiser_bessel(s, order, optim_kcut(oversamp))

    def step(carry, ishift):
        idx = id0 + ishift
        idx, ker = wrap(idx), kernel(idx - pos).prod(-1)

        # idx = jnp.unstack(idx, axis=-1)
        idx = tuple(jnp.moveaxis(idx, -1, 0)) # TODO: JAX >= 0.4.28 for unstack
        carry = carry.at[idx].add(weights * ker)
        return carry, None

    mesh = lax.scan(step, mesh, ishifts)[0]
    return mesh

def read(pos, mesh:jnp.ndarray, order:int=2, kernel_type='rectangular', oversamp=1.):
    """
    Read the value at the positions from the mesh.
    """
    dtype = 'int16' # int16 -> +/- 32_767, trkl
    shape = np.asarray(mesh.shape, dtype=dtype)
    def wrap(idx):
        return idx % shape
    
    id0 = (jnp.round if order % 2 else jnp.floor)(pos).astype(dtype)
    ishifts = np.arange(order) - (order - 1) // 2
    ishifts = np.array(list(product(* len(shape) * (ishifts,))), dtype=dtype)

    if kernel_type == 'rectangular':
        kernel = lambda s: rectangular(s, order)
    elif kernel_type == 'kaiser_bessel':
        kernel = lambda s: kaiser_bessel(s, order, optim_kcut(oversamp))
    
    def step(carry, ishift):
        idx = id0 + ishift
        idx, ker = wrap(idx), kernel(idx - pos).prod(-1)

        # idx = jnp.unstack(idx, axis=-1)
        idx = tuple(jnp.moveaxis(idx, -1, 0)) # TODO: JAX >= 0.4.28 for unstack
        carry += mesh[idx] * ker
        return carry, None
    
    out = jnp.zeros(id0.shape[:-1])
    out = lax.scan(step, out, ishifts)[0]
    return out


# def mass_assignment2(pos, shape, order:int=2, alpha:float=1.):
#     """
#     Compute mass assignment of particles onto a mesh.
#     """
#     dtype = 'int16' # int16 -> +/- 32_767, trkl
#     shape = np.asarray(shape, dtype=dtype)
#     def wrap(idx):
#         return idx % shape

#     order2 = order+2
#     id0 = (jnp.round if order2 % 2 else jnp.floor)(pos).astype(dtype)
#     ishifts = np.arange(order2) - (order2 - 1) // 2

#     for ishift in product(* len(shape) * (ishifts,)):
#         idx = id0 + np.array(ishift, dtype=dtype)
#         s = jnp.abs(idx - pos)
#         yield wrap(idx), paint_kernels2[order](s, alpha).prod(-1)

# def paint2(pos, mesh:tuple|jnp.ndarray, weights=1., order:int=2, alpha:float=1.):
#     """
#     Paint the positions onto the mesh. 
#     If mesh is a tuple, paint on a zero mesh with such shape.
#     """
#     if isinstance(mesh, tuple):
#         mesh = jnp.zeros(mesh)
#     else:
#         mesh = jnp.asarray(mesh)

#     for idx, ker in mass_assignment(pos, mesh.shape, order, alpha):
#         # idx = jnp.unstack(idx, axis=-1)
#         idx = tuple(jnp.moveaxis(idx, -1, 0)) # TODO: JAX >= 0.4.28 for unstack
#         mesh = mesh.at[idx].add(weights * ker)
#     return mesh
    
# def read2(pos, mesh:jnp.ndarray, order:int=2, alpha:float=1.):
#     """
#     Read the value at the positions from the mesh.
#     """
#     out = 0.
#     for idx, ker in mass_assignment(pos, mesh.shape, order, alpha):
#         # idx = jnp.unstack(idx, axis=-1)
#         idx = tuple(jnp.moveaxis(idx, -1, 0)) # TODO: JAX >= 0.4.28 for unstack
#         out += mesh[idx] * ker
#     return out



def interlace(pos, shape:tuple, weights=1., paint_order:int=2, interlace_order:int=2, 
              kernel_type='rectangular', oversamp=1., deconv=True):
    """
    Equal-spacing interlacing. Carefull `interlace_order>=3` is not isotropic.
    See [Wang&Yu2024](https://arxiv.org/abs/2403.13561)
    """
    kvec = rfftk(shape)
    mesh = jnp.zeros(r2chshape(shape), dtype=complex)
    shifts = jnp.arange(interlace_order) / interlace_order

    def step(carry, shift):
        mesh = paint(pos + shift, shape, weights, paint_order, kernel_type, oversamp)
        carry += jnp.fft.rfftn(mesh) * jnp.exp(1j * shift * sum(kvec)) / interlace_order
        return carry, None

    mesh = lax.scan(step, mesh, shifts)[0]
    if deconv:
        mesh = deconv_paint(mesh, paint_order, kernel_type=kernel_type, oversamp=oversamp)
    return mesh



##########
# Forces #
##########
def pm_forces(pos, mesh:tuple|jnp.ndarray, read_order:int=2, 
              paint_deconv:bool=False, grad_fd=False, lap_fd=False, kcut=np.inf):
    """
    Compute gravitational forces on particles using a PM scheme
    """
    if isinstance(mesh, tuple):
        mesh = jnp.fft.rfftn(paint(pos, mesh, order=read_order))
        # If painted field, double deconv to account for both painting and reading
        if paint_deconv:
            kvec = rfftk(ch2rshape(mesh.shape))
            mesh /= rectangular_hat(kvec, order=read_order)**2

    # Compute gravitational potential
    kvec = rfftk(ch2rshape(mesh.shape))
    pot = mesh * invlaplace_hat(kvec, lap_fd) * gaussian_hat(kvec, kcut)

    # Compute gravitational forces
    # NOTE: for regularly spaced positions, reading with paint_order=2 <=> paint_order=1
    return jnp.stack([read(pos, jnp.fft.irfftn(- gradient_hat(kvec, i, grad_fd) * pot), read_order) 
                      for i in range(3)], axis=-1)


def pm_forces2(pos, mesh:jnp.ndarray, read_order:int=2, lap_fd=False, grad_fd=False):
    """
    Return 2LPT source term.
    """
    kvec = rfftk(ch2rshape(mesh.shape))
    pot = mesh * invlaplace_hat(kvec, lap_fd)

    delta2 = 0.
    hesses = 0.
    for i in range(3):
        # Add products of diagonal terms = 0 + h11*h00 + h22*(h11+h00)...
        hess_ii = gradient_hat(kvec, i, grad_fd)**2
        hess_ii = jnp.fft.irfftn(hess_ii * pot)
        delta2 += hess_ii * hesses 
        hesses += hess_ii

        for j in range(i+1, 3):
            # Substract squared strict-up-triangle terms
            hess_ij = gradient_hat(kvec, i, grad_fd) * gradient_hat(kvec, j, grad_fd)
            delta2 -= jnp.fft.irfftn(hess_ij * pot)**2

    force2 = pm_forces(pos, jnp.fft.rfftn(delta2), read_order, grad_fd=grad_fd, lap_fd=lap_fd)
    return force2


def lpt(
    cosmo: Cosmology,
    init_mesh: jnp.ndarray,
    pos: jnp.ndarray,
    a: float| jnp.ndarray,
    lpt_order: int = 2,
    read_order: int = 2,
    grad_fd: bool = False,
    lap_fd: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute first or second order LPT displacement, at given scale factor(s).
    See e.g. Eq. 3.5 and 3.7 [List&Hahn](https://arxiv.org/abs/2409.19049)
    or Eq. 2 and 3 [Jenkins2010](https://arxiv.org/pdf/0910.0258)

    If `init_mesh` is provided in real space, perform an additional rfftn.
    """
    if jnp.isrealobj(init_mesh):
        init_mesh = jnp.fft.rfftn(init_mesh)

    force1 = pm_forces(pos, init_mesh, read_order, grad_fd=grad_fd, lap_fd=lap_fd)
    dpos = a2g(cosmo, a) * force1
    vel = force1

    if lpt_order == 2:
        # NOTE: For 3LPT and more, it can be more efficient to compute force only twice, 
        # for dpos and vel, than to compute it for every term.
        force2 = pm_forces2(pos, init_mesh, read_order, grad_fd=grad_fd, lap_fd=lap_fd)
        dpos -= a2g2(cosmo, a) * force2
        vel  -= a2dg2dg(cosmo, a) * force2

    return dpos, vel




###########
# Growths #
###########
growth_log10_amin: float = -3.
growth_steps: int = 128

# Growth from scale factor
def _growth_factor_ODE(cosmo, a, log10_amin=growth_log10_amin, steps=growth_steps):
    """
    Compute linear growth factor D(a) at a given scale factor,
    normalised such that D(a=1) = 1.

    Parameters
    ----------
    a: array_like
        Scale factor
    log10_amin: float
        Minimum scale factor in log10
    steps: int
        Number of steps for integration

    Returns
    -------
    D:  ndarray, or float if input scalar
        Growth factor computed at requested scale factor
    """
    # Check if growth has already been computed
    if not "background.growth_factor" in cosmo._workspace.keys():
        # Compute tabulated array
        atab = jnp.logspace(log10_amin, 0.0, steps)

        def D_derivs(y, x):
            q = 2.0
            q -= (background.Omega_m_a(cosmo, x) + (1.0 + 3.0 * background.w(cosmo, x)) * background.Omega_de_a(cosmo, x)) / 2
            q /= x
            r = 1.5 * background.Omega_m_a(cosmo, x) / x**2

            g1, g2 = y[0]
            f1, f2 = y[1]
            dy1da = [f1, -q * f1 + r * g1]
            dy2da = [f2, -q * f2 + r * g2 - r * g1 ** 2]
            return jnp.array([[dy1da[0], dy2da[0]], [dy1da[1], dy2da[1]]])

        y0 = jnp.array([[atab[0], -3.0 / 7 * atab[0] ** 2], [1.0, -6.0 / 7 * atab[0]]])
        y = odeint(D_derivs, y0, atab)

        # compute second order derivatives growth
        dyda2 = D_derivs(jnp.transpose(y, (1, 2, 0)), atab)
        dyda2 = jnp.transpose(dyda2, (2, 0, 1))

        # Normalize results
        y1 = y[:, 0, 0]
        gtab = y1 / y1[-1]
        y2 = y[:, 0, 1]
        g2tab = y2 / y2[-1]
        # To transform from dD/da to dlnD/dlna: dlnD/dlna = a / D dD/da
        ftab = y[:, 1, 0] / y1[-1] * atab / gtab
        f2tab = y[:, 1, 1] / y2[-1] * atab / g2tab
        # Similarly for second order derivatives
        # Note: these factors are not accessible as parent functions yet
        # since it is unclear what to refer to them with.
        htab = dyda2[:, 1, 0] / y1[-1] * atab / gtab
        h2tab = dyda2[:, 1, 1] / y2[-1] * atab / g2tab

        cache = {
            "a": atab,
            "g": gtab,
            "f": ftab,
            "h": htab,
            "g2": g2tab,
            "f2": f2tab,
            "h2": h2tab,
        }
        cosmo._workspace["background.growth_factor"] = cache
    else:
        cache = cosmo._workspace["background.growth_factor"]
    return jnp.clip(jnp.interp(a, cache["a"], cache["g"]), 0.0, 1.0)

def a2g(cosmo, a):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=growth_log10_amin, steps=growth_steps)
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(a, cache["a"], cache["g"])

def a2g2(cosmo, a):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=growth_log10_amin, steps=growth_steps)
    cache = cosmo._workspace["background.growth_factor"]
    # NOTE: "g2" is normalized such that g2 = -3/7 * "g2" ~ -3/7 * g^2
    return jnp.interp(a, cache["a"], cache["g2"]) * -3/7

def a2f(cosmo, a):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=growth_log10_amin, steps=growth_steps)
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(a, cache["a"], cache["f"])

def a2f2(cosmo, a):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=growth_log10_amin, steps=growth_steps)
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(a, cache["a"], cache["f2"])

def a2dg2dg(cosmo, a):
    g, g2, f, f2 = a2g(cosmo, a), a2g2(cosmo, a), a2f(cosmo, a), a2f2(cosmo, a)
    return safe_div(g2 * f2, g * f) # NOTE: dggdg(0) = 0


# Growth from growth factor
def g2a(cosmo, g):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=growth_log10_amin, steps=growth_steps)
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(g, cache["g"], cache["a"])

def g2g2(cosmo, g):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=growth_log10_amin, steps=growth_steps)
    cache = cosmo._workspace["background.growth_factor"]
    # NOTE: "g2" is normalized such that g2 = -3/7 * "g2" ~ -3/7 * g^2
    return jnp.interp(g, cache["g"], cache["g2"]) * -3/7

def g2f(cosmo, g):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=growth_log10_amin, steps=growth_steps)
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(g, cache["g"], cache["f"])

def g2f2(cosmo, g):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=growth_log10_amin, steps=growth_steps)
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(g, cache["g"], cache["f2"])

def g2dg2dg(cosmo, g):
    g2, f, f2 = g2g2(cosmo, g), g2f(cosmo, g), g2f2(cosmo, g)
    return safe_div(g2 * f2, g * f) # NOTE: dggdg(0) = 0


#############
# Distances #
#############
dist_log10_amin: float = -3.
dist_steps: int = 256

def a2chi(cosmo, a, log10_amin=dist_log10_amin, steps=dist_steps):
    r"""
    Radial comoving distance in [Mpc/h] for a given scale factor.

    Parameters
    ----------
    a : array_like
        Scale factor

    Returns
    -------
    chi : ndarray, or float if input scalar
        Radial comoving distance corresponding to the specified scale
        factor.

    Notes
    -----
    The radial comoving distance is computed by performing the following
    integration:

    .. math::

        \chi(a) =  R_H \int_a^1 \frac{da^\prime}{{a^\prime}^2 E(a^\prime)}
    """
    # Check if distances have already been computed
    if not "background.radial_comoving_distance" in cosmo._workspace.keys():
        # Compute tabulated array
        atab = jnp.logspace(log10_amin, 0.0, steps)

        def dchioverdlna(y, x):
            xa = jnp.exp(x)
            return background.dchioverda(cosmo, xa) * xa

        chitab = odeint(dchioverdlna, 0.0, jnp.log(atab))
        chitab = chitab[-1] - chitab

        cache = {"a": atab, "chi": chitab}
        cosmo._workspace["background.radial_comoving_distance"] = cache
    else:
        cache = cosmo._workspace["background.radial_comoving_distance"]

    # Return the results as an interpolation of the table
    return jnp.clip(jnp.interp(a, cache["a"], cache["chi"]), 0.0)


def chi2a(cosmo, chi, log10_amin=dist_log10_amin, steps=dist_steps):
    r"""
    Computes the scale factor for corresponding (array) of radial comoving
    distance by reverse linear interpolation.

    Parameters:
    -----------
    cosmo: Cosmology
      Cosmological parameters

    chi: array-like
      radial comoving distance to query.

    Returns:
    --------
    a : array-like
      Scale factors corresponding to requested distances
    """
    # Check if distances have already been computed, force computation otherwise
    if not "background.radial_comoving_distance" in cosmo._workspace.keys():
        a2chi(cosmo, 1.0, log10_amin=log10_amin, steps=steps)
    cache = cosmo._workspace["background.radial_comoving_distance"]
    return jnp.interp(chi, cache["chi"][::-1], cache["a"][::-1]) # NOTE: chi is decreasing with a





###########
# Solvers #
###########
def bullfrog_vf(cosmo:Cosmology, dg, mesh_shape:tuple, paint_order:int=2, 
                paint_deconv=False, grad_fd=False, lap_fd=False):
    """
    BullFrog vector field.
    """
    def alpha_bf(cosmo, g0, dg):
        '''
        BullFrog growth-time integrator coefficient.

        See Eq. 2.3 in [List&Hahn2024](https://arxiv.org/abs/2106.00461)
        '''
        g1 = g0 + dg / 2
        g2 = g0 + dg

        dg2dg0, dg2dg2 = g2dg2dg(cosmo, g0), g2dg2dg(cosmo, g2)
        lin_ratio = (g2g2(cosmo, g0) + dg2dg0 * dg / 2) / g1 - g1
        # NOTE: linearization of ratio (g2 - g^2)/g aroung g0, evaluated at g1
        return (dg2dg2 - lin_ratio) / (dg2dg0 - lin_ratio)
    
    def alpha_fpm(cosmo, g0, dg):
        '''
        FastPM growth-time integrator coefficient.

        See Eq. 3.16 in [List&Hahn2024](https://arxiv.org/abs/2106.00461)
        '''
        g2 = g0 + dg
        a0, a2 = g2a(cosmo, g0), g2a(cosmo, g2)
        coeff0 = background.Esqr(cosmo, a0)**.5 * g0 * g2f(cosmo, g0) * a0**2
        coeff2 = background.Esqr(cosmo, a2)**.5 * g2 * g2f(cosmo, g2) * a2**2
        return coeff0 / coeff2

    def kick(state, g0, cosmo, dg):
        pos, vel = state
        g1 = g0 + dg / 2
        forces = pm_forces(pos, tuple(mesh_shape), paint_order, grad_fd=grad_fd, lap_fd=lap_fd)
        alpha = alpha_bf(cosmo, g0, dg)
        return pos, alpha * vel + (1 - alpha) * forces / g1
        # return pos, vel + (1 - alpha) * (forces / g1 - vel) # equivalent
        # return pos, vel + dg * forces

    def drift(state, dg):
        pos, vel = state
        return pos + vel * dg, vel
    
    def vector_field(g0, state, args):
        old = state
        state = drift(state, dg / 2)
        state = kick(state, g0, cosmo, dg)
        state = drift(state, dg / 2)
        return tree.map(lambda new, old: (new - old) / dg, state, old)
    
    # def step(state, g0):
    #     state = drift(state, dg / 2)
    #     state = kick(state, g0, cosmo, dg)
    #     state = drift(state, dg / 2)
    #     return state, None
    
    return vector_field
    # return step


from diffrax import diffeqsolve, ODETerm, SaveAt, Euler
def save_y(t, y, args):
    return y

def nbody_bf(cosmo:Cosmology, init_mesh, pos, a0=0., a1=1., n_steps=5, 
             paint_order:int=2, lpt_order:int=2, grad_fd=False, lap_fd=False, 
             snapshots:int|list=None, fn=save_y):
    """
    N-body simulation with BullFrog solver.
    """
    n_steps = int(n_steps)
    g0 = a2g(cosmo, a0)
    g1 = a2g(cosmo, a1)
    dg = (g1 - g0) / n_steps
    
    mesh_shape = ch2rshape(init_mesh.shape)
    terms = ODETerm(bullfrog_vf(cosmo, dg, mesh_shape, paint_order, grad_fd=grad_fd, lap_fd=lap_fd))
    solver = Euler()

    # vel = pm_forces(pos, init_mesh, read_order=1, grad_fd=grad_fd, lap_fd=lap_fd)
    # dpos = g0 * vel # 1LPT initial displacement
    dpos, vel = lpt(cosmo, init_mesh, pos=pos, a=a0, lpt_order=lpt_order, 
                read_order=1, grad_fd=grad_fd, lap_fd=lap_fd)
    state = pos + dpos, vel

    if snapshots is None: 
        saveat = SaveAt(t1=True)
    elif isinstance(snapshots, int):
        if snapshots <= 1:
            saveat = SaveAt(t1=True, fn=fn)
        else:
            # saveat = SaveAt(ts=a2g(cosmo, jnp.linspace(a0, a1, snapshots)), fn=fn)
            saveat = SaveAt(ts=jnp.linspace(g0, g1, snapshots), fn=fn)
    else: 
        saveat = SaveAt(ts=a2g(cosmo, jnp.asarray(snapshots)), fn=fn)   

    sol = diffeqsolve(terms, solver, g0, g1, dt0=dg, y0=state, max_steps=n_steps, saveat=saveat) # cosmo as args may leak
    states = sol.ys
    # debug.print("bullfrog n_steps: {n}", n=sol.stats['num_steps'])
    return states


def nbody_bf_scan(cosmo:Cosmology, init_mesh, pos, a, n_steps=5, paint_order:int=2,
              grad_fd=False, lap_fd=False, snapshots:int|list=None):
    """
    No-diffrax version of N-body simulation with BullFrog solver. 
    Simpler but does not optimize for memory usage with binomial checkpointing.
    """
    g = a2g(cosmo, a)
    dg = g / n_steps
    gs = jnp.arange(n_steps) * dg

    mesh_shape = ch2rshape(init_mesh.shape)
    # vector_field = bullfrog_vf(cosmo, dg, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd)

    # def step(state, g0):
    #     vf = vector_field(g0, state, None)
    #     state = tree.map(lambda x, y: x + dg * y, state, vf)
    #     return state, None
    
    step = bullfrog_vf(cosmo, dg, mesh_shape, paint_order, grad_fd=grad_fd, lap_fd=lap_fd)
    
    vel = pm_forces(pos, init_mesh, paint_order, grad_fd=grad_fd, lap_fd=lap_fd)
    state = pos, vel

    state, _ = lax.scan(step, state, gs)
    return tree.map(lambda x: x[None], state)



















# def lpt_fpm(cosmo:Cosmology, init_mesh, pos, a, lpt_order:int=1, paint_order:int=2, grad_fd=True, lap_fd=False):
#     """
#     Computes first and second order LPT displacement, e.g. Eq. 2 and 3 [Jenkins2010](https://arxiv.org/pdf/0910.0258)
#     """
#     a = jnp.atleast_1d(a)
#     E = background.Esqr(cosmo, a)**.5
#     if jnp.isrealobj(init_mesh):
#         delta_k = jnp.fft.rfftn(init_mesh)
#         mesh_shape = init_mesh.shape
#     else:
#         delta_k = init_mesh
#         mesh_shape = ch2rshape(init_mesh.shape)

#     init_force = pm_forces(pos, delta_k, paint_order, grad_fd=grad_fd, lap_fd=lap_fd)
#     dq = a2g(cosmo, a) * init_force
#     p = a**2 * a2f(cosmo, a) * E * dq

#     if lpt_order == 2:
#         kvec = rfftk(mesh_shape)
#         pot = delta_k * invlaplace_hat(kvec, lap_fd)

#         delta2 = 0
#         hess_acc = 0
#         for i in range(3):
#             # Add products of diagonal terms = 0 + h11*h00 + h22*(h11+h00)...
#             hess_ii = gradient_hat(kvec, i, grad_fd)**2
#             hess_ii = jnp.fft.irfftn(hess_ii * pot)
#             delta2 += hess_ii * hess_acc 
#             hess_acc += hess_ii

#             for j in range(i+1, 3):
#                 # Substract squared strict-up-triangle terms
#                 hess_ij = gradient_hat(kvec, i, grad_fd) * gradient_hat(kvec, j, grad_fd)
#                 delta2 -= jnp.fft.irfftn(hess_ij * pot)**2

#         init_force2 = pm_forces(pos, np.fft.rfftn(delta2), paint_order, grad_fd=grad_fd, lap_fd=lap_fd)
#         dq2 = a2g2(cosmo, a) * init_force2 # D2 is renormalized: - D2 = 3/7 * growth_factor_second
#         p2 = (a**2 * a2f2(cosmo, a) * E) * dq2

#         dq -= dq2
#         p  -= p2

#     return dq, p


# def diffrax_vf(cosmo:Cosmology, mesh_shape, paint_order, grad_fd=True, lap_fd=False):
#     """
#     N-body ODE vector field for diffrax, e.g. Tsit5 or Dopri5

#     vector field signature is (a, state, args) -> dstate, where state is a tuple (position, velocities)
#     """
#     def vector_field(a, state, args):
#         pos, vel = state
#         forces = pm_forces(pos, mesh_shape, paint_order, grad_fd=grad_fd, lap_fd=lap_fd) * 1.5 * cosmo.Omega_m

#         # Computes the update of position (drift)
#         dpos = 1. / (a**3 * jnp.sqrt(background.Esqr(cosmo, a))) * vel
#         # Computes the update of velocity (kick)
#         dvel = 1. / (a**2 * jnp.sqrt(background.Esqr(cosmo, a))) * forces
#         return dpos, dvel
#     return vector_field


# def jax_ode_vf(cosmo:Cosmology, mesh_shape, paint_order, grad_fd=True, lap_fd=False):
#     """
#     Return N-body ODE vector field for jax.experimental.ode.odeint

#     vector field signature is (state, a, *args) -> dstate, where state is a tuple (position, velocities)
#     """
#     vf = diffrax_vf(cosmo, mesh_shape, paint_order, grad_fd, lap_fd)
#     def vector_field(state, a, *args):
#         return vf(a, state, args)
#     return vector_field



# from diffrax import diffeqsolve, ODETerm, SaveAt, Euler, Heun, Dopri5, Tsit5, PIDController, ConstantStepSize
# def nbody_tsit5(cosmo:Cosmology, mesh_shape, particles, a_lpt, a_obs, tol=1e-2, 
#                 paint_order:int=2, grad_fd=True, lap_fd=False, snapshots:int|list=None):
#     if a_lpt == a_obs:
#         return tree.map(lambda x: x[None], particles)
#     else:
#         terms = ODETerm(diffrax_vf(cosmo, mesh_shape, paint_order, grad_fd, lap_fd))
#         solver = Tsit5() # Tsit5 usually better than Dopri5
#         controller = PIDController(rtol=tol, atol=tol, pcoeff=0.4, icoeff=1, dcoeff=0)

#         if snapshots is None or (isinstance(snapshots, int) and snapshots < 2): 
#             saveat = SaveAt(t1=True)
#         elif isinstance(snapshots, int): 
#             saveat = SaveAt(ts=jnp.linspace(a_lpt, a_obs, snapshots))   
#         else: 
#             saveat = SaveAt(ts=jnp.asarray(snapshots))   

#         sol = diffeqsolve(terms, solver, a_lpt, a_obs, dt0=None, y0=particles,
#                                 stepsize_controller=controller, max_steps=1000, saveat=saveat)
#         # NOTE: if max_steps > 50 for dopri5/tsit5, just quit :')
#         particles = sol.ys
#         debug.print("tsit5 n_steps: {n}", n=sol.stats['num_steps'])
#         return particles


# from montecosmo.fpm import EfficientLeapFrog, LeapFrogODETerm, symplectic_ode
# def nbody_fpm(cosmo:Cosmology, mesh_shape, particles, a_lpt, a_obs, n_steps=5, 
#               paint_order:int=2, grad_fd=True, lap_fd=False, snapshots=None):
#     if a_lpt == a_obs:
#         return tree.map(lambda x: x[None], particles)
#     else:
#         solver = EfficientLeapFrog(initial_t0=a_lpt, final_t1=a_obs, cosmo=cosmo)
#         stepsize_controller = ConstantStepSize()
#         terms = tree.map(
#             LeapFrogODETerm,
#             symplectic_ode(mesh_shape, paint_absolute_pos=False),
#         )
#         cosmo._workspace = {}
#         args = cosmo

#         if snapshots is None or (isinstance(snapshots, int) and snapshots < 2): 
#             saveat = SaveAt(t1=True)
#         elif isinstance(snapshots, int): 
#             saveat = SaveAt(ts=jnp.linspace(a_lpt, a_obs, snapshots))   
#         else: 
#             saveat = SaveAt(ts=jnp.asarray(snapshots))   

#         sol = diffeqsolve(
#                 terms,
#                 solver=solver,
#                 t0=a_lpt,
#                 t1=a_obs,
#                 dt0=(a_obs - a_lpt) / n_steps,
#                 y0=particles,
#                 args=args,
#                 stepsize_controller=stepsize_controller,
#                 saveat=saveat,
#                 max_steps=10,
#                 # progress_meter=TqdmProgressMeter(refresh_steps=2),
#                 # adjoint=BacksolveAdjoint(solver=solver),
#             )

#         particles = sol.ys
#         return particles




# def rsd_fpm(cosmo:Cosmology, a, vel, los:np.ndarray):
#     """
#     Redshift-Space Distortion (RSD) displacement from cosmology and FastPM momentum.
#     Computed with respect to scale factor(s) and line-of-sight(s).
#     """
#     # Divide PM momentum by scale factor once to retrieve velocity, and once again for comobile velocity  
#     a = jnp.expand_dims(a, -1)
#     dpos = vel / (background.Esqr(cosmo, a)**.5 * a**2)
#     # Project velocity on line-of-sight
#     dpos = (dpos * los).sum(-1, keepdims=True) * los
#     return dpos



