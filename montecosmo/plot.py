import numpy as np
import matplotlib.pyplot as plt


# TODO: needs to return surf/p3d to add colorbar?
# TODO: create another function to plot 3d scatter
def plot_bivar_func(fn, n_discr=10, box=[[-1,1],[-1,1]], surface=False, cmap='viridis'):
    """
    Plot bivariate function fn, that should be vectorized first.
    e.g.:
        plt.subplot(121, projection="3d")
        plot_bivar_func(my_pdf, 100, surface=True)
    """
    xxyy = np.mgrid[[slice(box_ax[0],box_ax[1],n_discr*1j) for box_ax in box]]
    xy = xxyy.transpose(1,2,0)
    zz = fn(xy.reshape(-1,2)).reshape(n_discr,n_discr)
    if surface:
        plt.gca().plot_surface(xxyy[0], xxyy[1], zz, cmap=cmap)
    else:
        plt.pcolormesh(xxyy[0], xxyy[1], zz, cmap=cmap)


#############
# 3D Meshes #
#############
def mean_slice(mesh, sli:int | float | slice=None):
    """
    Return a 2D mean projected slice from a 3D mesh.
    """
    mesh_shape = np.array(mesh.shape)
    if sli is None:
        sli = slice(None)
    elif isinstance(sli, int):
        sli = slice(None, sli)
    elif isinstance(sli, float):
        sli = slice(None, round(sli*mesh_shape[-1]))
    return mesh[...,sli].mean(-1)


def plot_mesh(mesh, box_shape=None, sli:int | float | slice=None, vlim:float | tuple[float,float]=1e-4, cmap='viridis'):
    """
    Plot a 2D mean projected slice from a 3D mesh.

    Parameters
    ----------
    mesh : ndarray
        The 3D mesh to be plotted.
    box_shape : tuple of int, optional
        The shape of the mesh physical box in Mpc/h. If None, it defaults to mesh shape.
    sli : int or slice, optional
        The slice to be averaged along the last axis of the mesh. 
        If None, entire axis is used. 
        If integer, specifies the number of slices used starting from 0. 
        If float, specifies the proportion of axis used starting from 0.
    vlim : float or tuple of float, optional
        The limit values for colormap. 
        If float, specifies the proportion of values discarded bilateraly. 
        If tuple, specifies (vmin, vmax).
    cmap : str, optional
        The colormap used for plotting. Default is 'viridis'.

    Returns
    -------
    quad : QuadMesh
        The QuadMesh object created by `plt.pcolormesh`.
    """
    mesh_shape = np.array(mesh.shape)
    if box_shape is None:
        box_shape = mesh_shape

    mesh2d = mean_slice(mesh, sli)

    if vlim is None:
        vlim = None, None
    elif isinstance(vlim, float):
        vlim = np.quantile(mesh2d, [vlim/2, 1-vlim/2])
    vmin, vmax = vlim

    # xx, yy = np.indices(mesh_shape[:2]) * (box_shape/mesh_shape)[:2,None,None]
    xs, ys = np.linspace(0, box_shape[0], mesh_shape[0]), np.linspace(0, box_shape[1], mesh_shape[1])
    xx, yy = np.meshgrid(xs, ys)
    quad = plt.pcolormesh(xx, yy, mesh2d, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xlabel("$x$ [Mpc/$h$]"), plt.ylabel("$y$ [Mpc/$h$]")
    plt.gca().set_aspect(1)
    return quad


from matplotlib import animation, rc
from IPython.display import display

def anim_meshes(meshes, box_shape=None, vlim:float | tuple[float,float]=1e-4, 
                cmap='viridis', pause=10):
    """
    Animate a list of 2D meshes.
    """
    rc('animation', html='html5')
    meshes = np.asarray(meshes)
    assert meshes.ndim == 3, "meshes must be a list of 2D arrays"

    if vlim is None:
        vlim = np.quantile(meshes, [0, 1])
    elif isinstance(vlim, float):
        vlim = np.quantile(meshes, [vlim/2, 1-vlim/2])

    quad = plot_mesh(meshes[0,...,None], box_shape, None, vlim, cmap)
    plt.colorbar()

    def update(i):
        if i < len(meshes):
            quad.set_array(meshes[i])
        return quad,

    anim = animation.FuncAnimation(plt.gcf(), update, frames=len(meshes)+pause, interval=100, blit=True)
    plt.close()
    display(anim)


from montecosmo.utils import circ_mean

def scan_mesh3d(mesh, sli:int | float=1/16):
    """
    Return a list of 2D mesh as the mean projected slices from a 3D mesh.
    """
    mesh_shape = np.array(mesh.shape)
    if isinstance(sli, float):
        sli = round(sli*mesh_shape[-1])
    return np.moveaxis(circ_mean(mesh, sli, axis=-1), -1, 0)


def anim_scan(mesh, box_shape=None, sli:int | float=1/16, vlim:float | tuple[float,float]=1e-4, 
              cmap='viridis', pause=0):
    scan = scan_mesh3d(mesh, sli)
    anim_meshes(scan, box_shape, vlim=vlim, cmap=cmap, pause=pause)   
    



##################
# Power Spectrum #
##################
def plot_pk(ks, pk, i_ell=None, log=False, **kwargs):
    if i_ell is None:
        sub = ""
    else:
        ell = [0, 2, 4][i_ell]
        sub = f"_{ell}"
        pk = pk[i_ell]

    if log:
        plt.loglog(ks, pk, **kwargs)
        plt.ylabel("$P"+sub+"(k)$ [Mpc/$h$]$^3$")
    else:
        plt.plot(ks, ks * pk, **kwargs)
        plt.ylabel("$k P"+sub+"(k)$ [Mpc/$h$]$^2$")
    plt.xlabel("$k$ [$h$/Mpc]")


def plot_trans(ks, trans, log=False, **kwargs):
    if log:
        plt.loglog(ks, trans, **kwargs)
    else:
        plt.semilogy(ks, trans, **kwargs)
    plt.xlabel("$k$ [$h$/Mpc]"), plt.ylabel("transfer")


def plot_coh(ks, coh, log=False, **kwargs):
    if log:
        plt.loglog(ks, coh, **kwargs)
    else:
        plt.semilogy(ks, coh, **kwargs)
    plt.xlabel("$k$ [$h$/Mpc]"), plt.ylabel("coherence")


def plot_pktranscoh(ks, pk1, trans, coh, log=False, **kwargs):
    plt.subplot(131)
    plot_pk(ks, pk1, log=log, **kwargs)
    plt.legend()

    plt.subplot(132)
    plot_trans(ks, trans, log=log, **kwargs)

    plt.subplot(133)
    plot_coh(ks, coh, log=log, **kwargs)