#!/usr/bin/env python
"""
Triangle plot comparing field-level inference (lpt_64_fNL{0,20,-20}_o3_shash_pp0) with the
standard P_ell(k) (P0, P2) analysis (powell_fNL{...}, see run/powell_pngunitbox.py) on the
v1.0 PNG-UNIT QSO z=1.83 redshift-space HOD boxes: fNL and Eulerian b1, both with the mass
relation assumed (p fixed to 1.0 for fNL in {0, -20}, 1.4 for fNL=20).

Run in montenv (needs montecosmo + getdist): python plot_fli_vs_powell.py
"""
from pathlib import Path

from getdist import plots, loadMCSamples
from montecosmo.chains import Chains
from montecosmo.plot import SetDark2, theme

FNLS = [-20, 0, 20]
# TRACER = 'QSO'  # 'LRG' or 'QSO'
# TRACER = 'LRG'  # 'LRG' or 'QSO'
TRACER = 'QSO'  # 'LRG' or 'QSO'
ZS = {'LRG': 0.74, 'QSO': 1.83}
PPS = {'LRG': {-20: 1.0, 0: 1.0, 20: 1.4}, 'QSO': {-20: 1.0, 0: 1.0, 20: 1.4}}
# PPS = {'LRG': {-20: 0.80, 0: 0.80, 20: 1.13}, 'QSO': {-20: 0.83, 0: 0.83, 20: 1.55}}
KMAX = 0.08
# KMAX = 0.1
# KMAX = 0.12
USETEX = True
ROOT = Path(f"/pscratch/sd/h/hsimfroy/png/pngunitbox_{TRACER}_z{ZS[TRACER]}_redshiftspace_fNL")

theme(usetex=USETEX, font_size=16)

labels = [rf"$f_\mathrm{{NL}}={fnl}$" for fnl in FNLS]
gdsamps = []
methods, true_fnls = [], []

# FLI chains: rename fNL -> fnl_loc (desilike name) and Lagrangian b1 -> Eulerian b1 + 1
for fnl, label in zip(FNLS, labels):
    chains = Chains.load(ROOT / f"lpt_64_fNL{fnl}_o3_shash_ppunit_s0" / "chains" / "chains.h5")
    # chains = Chains.load(ROOT / f"lpt_64_fNL{fnl}_o3_shash_pp0" / "chains" / "chains.h5")
    print(f"FLI fNL={fnl}:", chains['fNL'].shape)
    chains['fnl_loc'] = chains.pop('fNL')
    chains['b1'] = chains['b1'] + 1
    gdsamp = chains.prune()[['fnl_loc', 'b1']].to_getdist(label)
    gdsamp.getParamNames().parWithName('fnl_loc').label = r'{f}_\mathrm{NL}'
    gdsamp.getParamNames().parWithName('b1').label = r'{b}_1'
    gdsamps.append(gdsamp)
    methods.append(r'\mathrm{FLI}')
    true_fnls.append(fnl)

# Standard P_ell analysis: desilike getdist roots (fnl_loc, b1, sn0, sigmas)
for fnl, label in zip(FNLS, labels):
    root = ROOT / f"powell_fNL{fnl}" / f"mcmc_getdist_pp{PPS[TRACER][fnl]:.1f}_kmax{KMAX:g}"
    # root = ROOT / f"powell_fNL{fnl}" / f"mcmc_getdist_pp{PPS[TRACER][fnl]:.1f}_kmax{KMAX:g}_kmin{0.004:g}"
    gdsamp = loadMCSamples(str(root))
    print(f"P_ell fNL={fnl}:", gdsamp.getParamNames().list())
    gdsamp.getParamNames().parWithName('fnl_loc').label = r'{f}_\mathrm{NL}'
    gdsamps.append(gdsamp)
    methods.append(r'P_{\ell}(k)')
    true_fnls.append(fnl)

# Save markdown table of fNL constraints (mean, sigma, median, 68% credible interval) per chain.
# gain = 1 - sigma(fNL)(run) / sigma(fNL)(P_ell): 0 for P_ell by construction, >0 if the run is tighter.
sig_ref = {fnl: gdsamps[3 + i].std('fnl_loc') for i, fnl in enumerate(FNLS)}  # gdsamps[3:6]=P_ell, FNLS order
md_lines = [
    # r"| $\mathrm{Run}$ | $f_\mathrm{NL}$ | $\mu \pm \sigma$ |  $\textrm{med }^{\Delta^+_{0.68}}_{\Delta^-_{0.68}}$ | $0.68\textrm{-Interval}$ |",
    r"| $\mathrm{Run}$ | $f_\mathrm{NL}$ | $\mu \pm \sigma$ | $\mathrm{gain}$ | $\mathrm{med} \pm \Delta_{0.68}$ | $0.68\textrm{-Interval}$ |",
    "|---:|---:|---:|---:|---:|---:|",
]
for gdsamp, method, fnl in zip(gdsamps, methods, true_fnls):
    mean = gdsamp.mean('fnl_loc')
    sigma = gdsamp.std('fnl_loc')
    med = gdsamp.confidence('fnl_loc', 0.5)
    # lo, hi = gdsamp.twoTailLimits('fnl_loc', 0.68)
    res = gdsamp.get1DDensity('fnl_loc').getLimits(0.68) # -> (lo, hi, tail_lo, tail_hi)
    lo, hi = res[0], res[1]
    gain = 1 - sigma / sig_ref[fnl]
    md_lines.append(
        rf"| ${method}$ | ${fnl}$ | ${mean:.2f} \pm {sigma:.2f}$ | ${gain:.2f}$ | ${med:.2f}^{{+{hi-med:.2f}}}_{{-{med-lo:.2f}}}$ | $[{lo:.2f}, {hi:.2f}]$ |"
    )
md_fn = ROOT / "fnl_constraints_pp1_1_1.4.md"
md_fn.write_text("\n".join(md_lines) + "\n")
print(f"Saved {md_fn}")

gdplt = plots.get_subplot_plotter(width_inch=6, scaling=False, rc_sizes=True)
gdplt.triangle_plot(roots=gdsamps,
                    params=['fnl_loc', 'b1'],
                    legend_labels=labels,
                    # title_limit=1,
                    filled=3 * [True] + 3 * [False],
                    contour_colors=2 * [SetDark2(i) for i in range(3)],
                    contour_ls=3 * ['-'] + 3 * ['--'],
                    contour_lws=6 * [1.5],
                    )

# # True fNL markers, colored per mock, on the fnl_loc diagonal and the 2D panel (x axis)
# for i, fnl in enumerate(FNLS):
#     for ax in (gdplt.subplots[0, 0], gdplt.subplots[1, 0]):
#         ax.axvline(fnl, color=SetDark2(i), ls=':', lw=1.)

# Append grey style entries (linestyle = method) to the existing colored legend (color = f_NL).
from matplotlib.lines import Line2D
from matplotlib.transforms import IdentityTransform
leg = gdplt.legend
handles = list(leg.legend_handles)
texts = [t.get_text() for t in leg.get_texts()]
handles += [Line2D([], [], color='grey', ls='-', lw=1.5),
            Line2D([], [], color='grey', ls='--', lw=1.5)]
texts += [r'FLI', r'$P_\ell(k)$']
# texts += [r'FLI $k_\mathrm{Nyq}=0.075$', r'$P_\ell(k)\ k_\mathrm{max}=0.08$']
loc, bbox, fs = leg._loc, leg.get_bbox_to_anchor(), leg.get_texts()[0].get_fontsize()
leg.remove()
gdplt.fig.legend(handles, texts, loc=loc, bbox_to_anchor=bbox,
                 bbox_transform=IdentityTransform(), prop={'size': fs}, frameon=False)

for ext in ['pdf', 'png']:
    fn = ROOT / f"fnl_b1_fli_vs_powell_pp1_1_1.4.{ext}"
    gdplt.export(str(fn))
    print(f"Saved {fn}")
