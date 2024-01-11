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