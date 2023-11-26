import numpy as np

# TODO: use mgrid with 1j instead of linspace+meshgrid
# TODO: input box instead of cube_...
# TODO: add an option to choose between 3D plot_surface or 2D pcolormesh
# TODO: needs to return surf/p3d to add colorbar?
# TODO: create another function to plot 3d scatter, and move both to a plot.py
def plot_bivar_func(ax, func, nb_discr=100, cube_center=0, cube_halfsize=1.5):
    """
    Plot bivariate function.
    e.g.:
        ax = plt.subplot(121, projection="3d")
        plot3d_func(ax, my_pdf, 100)
    """
    x, y = np.linspace(cube_center-cube_halfsize, cube_center+cube_halfsize, nb_discr), np.linspace(cube_center-cube_halfsize, cube_center+cube_halfsize, nb_discr)
    xx, yy = np.meshgrid(x, y)
    xy = np.array([xx, yy]).transpose(1,2,0)
    zz =  func(xy.reshape(-1,2)).reshape(nb_discr,nb_discr)
    surf = ax.plot_surface(xx, yy, zz, cmap="viridis")