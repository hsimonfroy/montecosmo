#!/usr/bin/env python
"""
Paper figure 1: painting kernels.

Rectangular (NGP/CIC/TSC/PCS) vs Kaiser-Bessel mass-assignment kernels, for orders 1..4,
shown in position space K(s) (left) and Fourier space K(k) (right).

Both kernel families are normalized so that K(k=0) = 1 (total-mass conservation), which is why
the position-space peaks differ between orders. The Kaiser-Bessel cutoff is the optimal PSWF-like
`kcut` for an oversampling factor of 3/2.

Pure plotting, runs on CPU:
    python -u src/montecosmo/tests/paper/fig_paint_kernels.py
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # trivial compute, no GPU needed
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from jax import numpy as jnp, config as jconfig
jconfig.update("jax_enable_x64", True)

from montecosmo.plot import theme, SetDark2, DarkSet2
from montecosmo.nbody import (rectangular, rectangular_hat,
                              kaiser_bessel, kaiser_bessel_hat, optim_kcut)

# --- config ---------------------------------------------------------------
orders = [1, 2, 3, 4]
oversamp = 3/2
kcut = float(optim_kcut(oversamp))
out_dir = Path('/global/homes/h/hsimfroy/workspace/montecosmo_proj/figs/plotted')

theme(usetex=True, font_size=15)  # serif + SetDark2 color cycle
# colors = ['indigo', 'red', 'gold', 'lightgreen']
# https://gist.github.com/thriveth/8560036
CB_color = ['#377eb8', '#ff7f00', '#4daf4a',
            '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00']

DESI_like = ['#0173b2', '#ff9d26', '#28ac88',
            '#d38cc6', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00']

tab10_perm = ['#1f77b4', '#ff7f0e', '#2ca02c',
            '#e377c2', '#8c564b', '#9467bd',
            '#7f7f7f', '#d62728', '#bcbd22']

# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.get_cmap('Dark2').colors)
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=tab10_perm)

# --- kernels (with explicit compact support, half-width = order/2) ---------
def rect_pos(s, order):
    val = np.asarray(rectangular(jnp.asarray(s), order))
    return np.where(np.abs(s) < order / 2, val, 0.0)

def kb_pos(s, order, kcut):
    inside = np.abs(s) < order / 2
    s_safe = np.where(inside, s, 0.0)  # keep arg in support to avoid nan
    val = np.asarray(kaiser_bessel(jnp.asarray(s_safe), order, kcut))
    return np.where(inside, val, 0.0)

def rect_hat(k, order):
    return np.asarray(rectangular_hat([jnp.asarray(k)], order))

def kb_hat(k, order, kcut):
    return np.asarray(kaiser_bessel_hat([jnp.asarray(k)], order, kcut))

s = np.linspace(0, 2, 800)
kmax = 4*np.pi
k = np.linspace(0, kmax, 800)
colors = [f'C{i}' for i in range(len(orders))]

# --- plot -----------------------------------------------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 4.), layout='constrained')

for order, c in zip(orders, colors):
    ax0.plot(s, rect_pos(s, order), '-', color=c)
    ax0.plot(s, kb_pos(s, order, kcut), '--', color=c)
    ax1.plot(k, rect_hat(k, order), '-', color=c)
    ax1.plot(k, kb_hat(k, order, kcut), '--', color=c)

ax0.set_xlabel(r'$s$')
ax0.set_ylabel(r'$K(s)$')
ax0.set_xlim(0, 2)
# ax0.set_ylim(bottom=0)

# ax1.axvspan(np.pi, kmax, color='grey', alpha=0.1, label=r'$k>\pi$')
ax1.fill_between(k, 0, 1, where=k > np.pi, color='grey', alpha=0.1, transform=ax1.get_xaxis_transform())
ax1.set_xlabel(r'$k$')
ax1.set_ylabel(r'$\hat{K}(k)$')
ax1.set_xlim(0, kmax)
# ax1.set_ylim(bottom=0)
ax1.set_xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi ],
               ['$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
# ax1.set_xticks([0, np.pi / 2, np.pi, 3*np.pi/2, 2*np.pi, 5*np.pi/2, 3*np.pi ],
#                ['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$', r'$5\pi/2$', r'$3\pi$'])

# legend in the left plot only
handles = [Line2D([], [], color=c, ls='-') for c in colors]
labels = [rf'$\mathrm{{order}}={o}$' for o in orders]
handles += [Line2D([], [], color='grey', ls='-'), Line2D([], [], color='grey', ls='--')]
labels += ['rectangular', 'Kaiser-Bessel']
ax0.legend(handles, labels, loc='upper right', frameon=False)
ax0.grid(alpha=0.2)
ax1.grid(alpha=0.2)

out_dir.mkdir(parents=True, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(out_dir / f'painting_kernels.{ext}', dpi=200)
print(f"kcut(oversamp={oversamp}) = {kcut:.4f}")
print(f"saved {out_dir / 'painting_kernels.{pdf,png}'}")
