import numpy as np
from functools import partial

import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
from matplotlib import animation, rc
from matplotlib.colors import to_rgba_array
from matplotlib.colors import ListedColormap

from montecosmo.bdec import credint


###########
# General #
###########
def plot_bivar(fn, box=((-1,1),(-1,1)), n=50, mode='mesh', **kwargs):
    """
    Plot bivariate function fn, that should be vectorized first.
    mode can be 'mesh', 'contour', 'contourf', 'surf'.

    Example
    --------
    ```
    # pdf = lambda x, y: np.exp( -(x**2 + y**2) / 2) / (2 * np.pi)
    pdf = lambda x: np.exp( -(x**2).sum(-1) / 2) / (2 * np.pi)
    plt.subplot(121, projection="3d")
    plot_bivar(pdf, mode='surf')
    ```
    """
    if isinstance(box, (int, float)):
        box = ((-box, box), (-box, box))

    xs, ys = np.linspace(*box[0], n), np.linspace(*box[1], n)
    xx, yy = np.meshgrid(xs, ys)
    # zz = fn(xx.reshape(-1), yy.reshape(-1)).reshape(n, n)
    zz = fn(np.stack((xx, yy), -1).reshape(-1, 2)).reshape(n, n)

    if mode=='surf':
        out = plt.gca().plot_surface(xx, yy, zz, **kwargs)
    elif mode=='mesh':
        out = plt.pcolormesh(xx, yy, zz, **kwargs)
    elif mode=='contour':
        out = plt.contour(xx, yy, zz, **kwargs)
    elif mode=='contourf':
        out = plt.contourf(xx, yy, zz, **kwargs)
    return out


#############
# 3D Meshes #
#############
def mean_proj(mesh, ids:float|slice|np.ndarray=1., axis=-1):
    """
    Return a 2D mean projection of a 3D mesh along given axis.
    """
    mesh_shape = np.array(mesh.shape)
    if isinstance(ids, float):
        low = round(mesh_shape[axis] / 2 * (1 - ids))
        high = round(mesh_shape[axis] / 2 * (1 + ids))
        ids = slice(low, high)
    return np.moveaxis(mesh, axis, -1)[...,ids].mean(-1)


def plot_mesh(mesh, box_shape=None, ids:float|slice|np.ndarray=1., 
              axis=-1, vlim:float|tuple[float,float]=1e-4, transpose=False, **kwargs):
    """
    Plot a 2D mean projection of a 3D mesh along given axis.

    Parameters
    ----------
    mesh : ndarray
        The 3D mesh to be plotted.
    box_shape : tuple of int, optional
        The shape of the mesh physical box in Mpc/h. If None, defaults to mesh shape.
    ids : float, slice, or ndarray, optional
        Indices to be averaged along the given axis of the mesh. 
        If float, specifies the proportion of axis used starting from mesh center.
    axis : int, optional
        The axis along which to average the mesh. Default is -1 (last axis).
    vlim : float or tuple of float, optional
        The limit values for colormap. 
        * If float, specifies the proportion of values discarded bilateraly. 
        * If tuple, specifies (vmin, vmax).
    transpose : bool, optional
        If True, transpose the x and y axes in the plot.
    **kwargs : keyword arguments
        Additional arguments passed to `plt.pcolormesh`.

    Returns
    -------
    quad : QuadMesh
        The QuadMesh object created by `plt.pcolormesh`.
    """
    mesh_shape = np.array(mesh.shape)
    axids = [0,1,2]
    axids.remove(axids[axis])
    if box_shape is None:
        box_shape = mesh_shape
    else:
        box_shape = np.asarray(box_shape)
        xlab, ylab = np.array(["x", "y", "z"])[axids]
        if transpose:
            xlab, ylab = ylab, xlab
        plt.xlabel(f"${xlab}$ [Mpc/$h$]"), plt.ylabel(f"${ylab}$ [Mpc/$h$]")

    mesh2d = mean_proj(mesh, ids, axis)

    if vlim is None:
        vlim = None, None
    elif isinstance(vlim, float):
        vlim = np.quantile(mesh2d, [vlim/2, 1-vlim/2])
    vmin, vmax = vlim

    xb, yb = box_shape[axids]
    xm, ym = mesh_shape[axids]
    xs, ys = np.linspace(0, xb, xm, endpoint=False), np.linspace(0, yb, ym, endpoint=False)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    if transpose:
        xx, yy = yy, xx
    quad = plt.pcolormesh(xx, yy, mesh2d, vmin=vmin, vmax=vmax, **kwargs)
    plt.gca().set_aspect(1)
    return quad


def anim_meshes(meshes, box_shape=None, vlim:float|tuple[float,float]=1e-4, 
                cmap='viridis', pause=10):
    """
    Animate a list of 2D meshes.
    """
    rc('animation', html='html5')
    meshes = np.asarray(meshes)
    assert meshes.ndim == 3, "meshes must be a list of 2D arrays"

    if vlim is None:
        vlim = meshes.min(), meshes.max()
    elif isinstance(vlim, float):
        vlim = np.quantile(meshes, [vlim/2, 1-vlim/2])

    quad = plot_mesh(meshes[0,...,None], box_shape, 1., vlim, cmap)
    plt.colorbar()

    def update(i):
        if i < len(meshes):
            quad.set_array(meshes[i])
        return quad,

    anim = animation.FuncAnimation(plt.gcf(), update, frames=len(meshes)+pause, interval=100, blit=True)
    plt.close()
    display(anim)




def circ_conv(a, b, axis=-1):
    """
    Circular convolution of two arrays along a given axis.
    Returned array has the maximum length of the two arrays along axis.
    """
    a, b = np.moveaxis(a, axis, -1), np.moveaxis(b, axis, -1)
    n = max(a.shape[-1], b.shape[-1])
    ab = np.fft.rfft(a, n) * np.fft.rfft(b, n)
    ab = np.fft.irfft(ab, n)
    return np.moveaxis(ab, -1, axis)


def circ_mean(a, n=1, axis=-1):
    """
    Circular mean of array along a given axis.
    """
    a = np.moveaxis(a, axis, -1)
    out = circ_conv(a, np.ones(n)/n, axis=-1)
    return np.moveaxis(out, -1, axis)




def scan_mesh3d(mesh, n:int|float=1/16):
    """
    Return a list of 2D mesh, mean projected from a 3D mesh.
    """
    mesh_shape = np.array(mesh.shape)
    if isinstance(n, float):
        n = round(n*mesh_shape[-1])
    return np.moveaxis(circ_mean(mesh, n, axis=-1), -1, 0)


def anim_scan(mesh, box_shape=None, n:int|float=1/16, vlim:float|tuple[float,float]=1e-4, 
              cmap='viridis', pause=0):
    scan = scan_mesh3d(mesh, n)
    anim_meshes(scan, box_shape, vlim=vlim, cmap=cmap, pause=pause)   
    



##################
# Power Spectrum #
##################
def plot_pow(ks, pow, *args, ell=None, log=False, fill=None, **kwargs):
    """
    Plot power spectrum `pow` as a function of wavenumber `ks`.

    Parameters
    ----------
    ks : ndarray
        Wavenumbers in units of $h$/Mpc.
    pow : ndarray
        Power spectrum values, of shape `shape(ks)` if `ell` is None, 
        else `(shape(ks)[:-1], n_ell, shape(ks)[-1])`.
    args : tuple
        Additional positional arguments passed to `plt.plot` or `plt.fill_between`.
    ell : int, optional
        The multipole to plot.
    log : bool, optional
        If True, plot spectrum $P(k)$ in loglog scale, 
        else plot $k P(k)$ in linlin scale.
    fill : float, optional
        If provided, the credible interval probability to fill between.
    kwargs : dict
        Additional keyword arguments passed to `plt.plot` or `plt.fill_between`.

    Returns
    -------
    out : Line2D or PolyCollection
        The output of `plt.plot` or `plt.fill_between`, depending on `fill`.
    """
    if ell is None:
        sub = ""
    else:
        i_ell = [0, None, 1, None, 2][ell]
        sub = f"_{ell}"
        pow = pow[...,i_ell,:]

    if log:
        plt.xscale('log'), plt.yscale('log')
        plt.ylabel("$P"+sub+"(k)$ [Mpc/$h$]$^3$")
    else:
        plt.ylabel("$k P"+sub+"(k)$ [Mpc/$h$]$^2$")
        pow = ks * pow

    if fill is None:
        out = plt.plot(ks, pow, *args, **kwargs)
    else:
        out = [] 
        fill = np.atleast_1d(fill)
        color = plt.gca()._get_patches_for_fill.get_next_color()
        for f in fill:
            scis = credint(pow, f, axis=0)
            collec = plt.fill_between(ks[0], *scis.T, *args, **{'alpha':(1-f)**.5, 'color': color} | kwargs)
            color = collec.get_facecolor()
            out.append(collec)
    plt.xlabel("$k$ [$h$/Mpc]")
    return out


def plot_trans(ks, trans, *args, log=False, fill=None, **kwargs):
    if fill is None:
        out = plt.plot(ks, trans, *args, **kwargs)
    else:
        out = [] 
        fill = np.atleast_1d(fill)
        color = plt.gca()._get_patches_for_fill.get_next_color()
        for f in fill:
            scis = credint(trans, f, axis=0)
            collec = plt.fill_between(ks[0], *scis.T, *args, **{'alpha':(1-f)**.5, 'color': color} | kwargs)
            color = collec.get_facecolor()
            out.append(collec)
    if log:
        plt.xscale('log')
    plt.yscale('log'), plt.xlabel("$k$ [$h$/Mpc]"), plt.ylabel("transfer")
    return out


def plot_coh(ks, coh, *args, log=False, fill=None, **kwargs):
    if fill is None:
        out = plt.plot(ks, coh, *args, **kwargs)
    else:
        out = [] 
        fill = np.atleast_1d(fill)
        color = plt.gca()._get_patches_for_fill.get_next_color()
        for f in fill:
            scis = credint(coh, f, axis=0)
            collec = plt.fill_between(ks[0], *scis.T, *args, **{'alpha':(1-f)**.5, 'color': color} | kwargs)
            color = collec.get_facecolor()
            out.append(collec)
    if log:
        plt.xscale('log')
    plt.yscale('log'), plt.xlabel("$k$ [$h$/Mpc]"), plt.ylabel("coherence")
    return out


def plot_powtranscoh(ks, pow1, trans, coh, *args, 
                     log=False, fill:float=None, axes:list=None, **kwargs):
    outs = []

    plt.subplot(131) if axes is None else plt.sca(axes[0])
    out = plot_pow(ks, pow1, *args, log=log, fill=fill, **kwargs)
    outs.append(out)
    
    plt.subplot(132) if axes is None else plt.sca(axes[1])
    out = plot_trans(ks, trans, *args, log=log, fill=fill, **kwargs)
    outs.append(out)

    plt.subplot(133) if axes is None else plt.sca(axes[2])
    out = plot_coh(ks, coh, *args, log=log, fill=fill, **kwargs)
    outs.append(out)
    return outs



##########
# Colors #
##########
def alternate(a, b, axis=0):
    a, b = np.moveaxis(a, axis, 0), np.moveaxis(b, axis, 0)
    assert a.shape == b.shape
    mid = a.shape[0]
    out = np.zeros((2*mid, *a.shape[1:]))
    out[:mid:2] = a[::2]
    out[mid::2] = b[::2]
    out[1:mid:2] = b[1::2]
    out[mid+1::2] = a[1::2]
    return np.moveaxis(out, 0, axis)

def interlace(a, b, axis=0):
    a, b = np.moveaxis(a, axis, 0), np.moveaxis(b, axis, 0)
    assert a.shape == b.shape
    out = np.empty((a.shape[0] + b.shape[0], *a.shape[1:]))
    out[0::2] = a
    out[1::2] = b
    return np.moveaxis(out, 0, axis)

c1 = plt.get_cmap('Dark2').colors
c2 = plt.get_cmap('Set2').colors

SetDark2 = ListedColormap(alternate(c2, c1))
DarkSet2 = ListedColormap(alternate(c1, c2))
# SetDark2_k = ListedColormap(alternate(1-np.array(c2), 1-np.array(c1)))
# DarkSet2_k = ListedColormap(alternate(1-np.array(c1), 1-np.array(c2)))



def color_switch(color, reverse=False):
    """
    Select between color an its negative, or colormap and its reversed.
    Typically used to switch between light theme and dark theme. 

    `color` must be Matpotlib color, or array of colors, or colormap.

    No need to switch the default color cycle `f'C{i}'`, Matplotlib handles it already.
    """
    try:
        color = to_rgba_array(color)
    except:
        if isinstance(color, str): # handle cmap
            if reverse:
                if color.endswith('_r'): # in case provided cmap is alreday reversed
                    return color[:-2]
                else:
                    return color+'_r'# reverse cmap
            else:
                return color
        else:
            raise TypeError("`color` must be Matpotlib color, or array of colors, or colormap.")

    if reverse:
        color[...,:-1] = 1-color[...,:-1] # take color negative, does not affect alpha
    return color


def set_plotting_options(usetex=False, font_size=10):
    params = {'text.usetex': usetex,
            #   'ps.useafm': True,
            #   'pdf.use14corefonts': True,
              'font.family': 'roman' if usetex else 'sans-serif',
              'font.size':font_size,} 
            # NOTE: 'ps.useafm' and 'pdf.use14corefonts' for PS and PDF font comptatibiliies
    plt.rcParams.update(params)
    # import matplotlib as mpl
    # mpl.rcParams.update(mpl.rcParamsDefault)


def theme(dark=False, usetex=False, font_size=10, cmap='SetDark2'):
    """
    Set Matplotlib theme and return an adequate color switching function.
    """
    if dark: 
        plt.style.use('dark_background')
    else: 
        plt.style.use('default')

    if cmap is None:
        cmap = plt.get_cmap('tab10') # default cmap
    elif cmap == 'SetDark2':
        cmap = SetDark2
    else:
        cmap = plt.get_cmap(cmap) # cmap can be plt.get_cmap('viridis', 10) for instance
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cmap.colors)

    rc('animation', html='html5') # handle Matplotlib animations
    set_plotting_options(usetex, font_size)
    theme = partial(color_switch, reverse=dark)
    return theme


def invert_bw(path, eps=30):
    """
    Invert black and white in image, without affecting other colors.
    ```
    img = invert_bw(path)
    img.save(save_path)
    img.show()
    ```
    """
    img = Image.open(path).convert("RGB")
    img = np.array(img)

    white_mask = np.all(img >= 255 - eps, axis=-1)
    black_mask = np.all(img <= eps, axis=-1)
    img[white_mask] = 255 - img[white_mask]
    img[black_mask] = 255 - img[black_mask]
    
    # greys = img.mean(-1)
    # mask = np.linalg.norm(img - greys[..., None], axis=-1) < eps
    return Image.fromarray(img)