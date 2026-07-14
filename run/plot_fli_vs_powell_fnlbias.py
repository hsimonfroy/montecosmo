#!/usr/bin/env python
"""
FLI-vs-P_ell comparison WITHOUT the mass relation (reproduces arXiv:2605.21436 Fig. 3 layout).
4-param triangle on {fNL, fNL_bp (=fNL*bphi), fNL_bpd (=fNL*bphidelta), b1(Eulerian)} for the
v1.0 PNG-UNIT QSO z=1.83 redshift-space mocks at fNL = -20, 0, 20.

  * FLI  : lpt_64_fNL{fnl}_o3_shash_s0 (png_type='fNL_bias' -> samples fNL, fNL_bp, fNL_bpd; free IC;
           stoch fixed), under the '..._fNLb' register folder. Constrains all four.
  * P_ell : powell_fNL{fnl}/mcmc_getdist_bfnl_kmax0.08 (desilike mode='bfnl'): the power spectrum
           constrains only fNL_bp (=bfnl_loc) and b1. Bare fNL and fNL_bpd are NOT power-spectrum
           observables (fNL degenerate with bphi; fNL*bphidelta loop-suppressed and fixed), so we
           overlay them as draws from the FLI model priors -> flat/prior-dominated bands, exactly as
           the paper: "P effectively constrains only the product fNL bphi, leaving fNL and fNL bphidelta
           essentially unconstrained" (Fig. 3).

Run in montenv: python plot_fli_vs_powell_fnlbias.py
"""
from pathlib import Path

import numpy as np
from getdist import plots, loadMCSamples, MCSamples
from montecosmo.chains import Chains
from montecosmo.plot import SetDark2, theme

FNL_ROOT = Path("/pscratch/sd/h/hsimfroy/png/pngunitbox_QSO_z1.83_redshiftspace_fNL")   # P_ell lives here
FNLB_ROOT = Path("/pscratch/sd/h/hsimfroy/png/pngunitbox_QSO_z1.83_redshiftspace_fNLb")  # FLI lives here
FNLS = [-20, 0, 20]
TRACER = 'QSO'  # 'LRG' or 'QSO'
PPS = {'LRG': {-20: 0.80, 0: 0.80, 20: 1.13}, 'QSO': {-20: 0.83, 0: 0.83, 20: 1.55}}  # assumed p_phi per mock
B1S = {'LRG': 2.050, 'QSO': 2.674}
KMAX = 0.08
# FLI model base priors for the P_ell prior-overlay of the unconstrained params (Normal(0, scale)).
PRIOR_SCALE = {'fNL': 10000., 'fNL_bpd': 10000.}
PARAMS = ['fNL', 'fNL_bp', 'fNL_bpd', 'b1']
LABELS = {'fNL': r'{f}_\mathrm{NL}', 'fNL_bp': r'{f}_\mathrm{NL} b_\phi',
          'fNL_bpd': r'{f}_\mathrm{NL} b_{\phi\delta}', 'b1': r'{b}_1'}
USETEX = True

theme(usetex=USETEX, font_size=14)
legend_labels = [rf"$f_\mathrm{{NL}}={fnl}$" for fnl in FNLS]
gdsamps = []

# --- FLI: constrains all four params (b1 -> Eulerian +1) ---
for fnl, lab in zip(FNLS, legend_labels):
    chains = Chains.load(FNLB_ROOT / f"lpt_64_fNL{fnl}_o3_shash_s0" / "chains" / "chains.h5")
    print(f"FLI  fNL={fnl}:", {k: chains[k].shape for k in ['fNL', 'fNL_bp', 'fNL_bpd', 'b1']})
    chains['b1'] = chains['b1'] + 1
    chains['fNL_bpd'] = chains['fNL_bpd'] - 2 * chains['fNL'] # We sample fNL_bpd + 2 fNL
    chains['fNL_bpd'] = chains['fNL_bp'] + chains['fNL_bpd'] / 2
    gd = chains.prune()[PARAMS].to_getdist(lab)
    for p in PARAMS:
        gd.getParamNames().parWithName(p).label = LABELS[p]
    gdsamps.append(gd)

# --- P_ell: real fNL_bp (=bfnl_loc) + b1; fNL, fNL_bpd drawn from the FLI priors (prior-dominated) ---
rng = np.random.default_rng(42)
for fnl, lab in zip(FNLS, legend_labels):
    g = loadMCSamples(str(FNL_ROOT / f"powell_fNL{fnl}" / f"mcmc_getdist_bfnl_kmax{KMAX:g}"))
    names = g.getParamNames().list()
    n = g.samples.shape[0]
    cols = {
        'fNL': rng.normal(0., PRIOR_SCALE['fNL'], n),           # prior overlay (unconstrained by P)
        'fNL_bp': g.samples[:, names.index('bfnl_loc')],        # the genuine P constraint
        'fNL_bpd': rng.normal(0., PRIOR_SCALE['fNL_bpd'], n),   # prior overlay (unconstrained by P)
        'b1': g.samples[:, names.index('b1')],
    }
    print(f"P_ell fNL={fnl}: fNL_bp={cols['fNL_bp'].mean():.1f}+-{cols['fNL_bp'].std():.1f}, b1={cols['b1'].mean():.3f}")
    gd = MCSamples(samples=np.column_stack([cols[p] for p in PARAMS]), names=PARAMS,
                   labels=[LABELS[p] for p in PARAMS], weights=g.weights, label=lab)
    gdsamps.append(gd)

# Markdown table: measured fNL_bp, and the fNL implied by ASSUMING the mass relation,
# fNL(p_phi) = fNL_bp / (2 delta_c (b1E - p_phi)), computed per sample (b1E from chains, p_phi from PPS).
DELTAC = 1.686
methods = 3 * [r'\mathrm{FLI}'] + 3 * [r'P_{\ell}(k)']
true_fnls = 2 * FNLS
# add fNL(p_phi) = fNL_bp / (2 delta_c (b1E - p_phi)) per sample to every chain
for gd, fnl in zip(gdsamps, true_fnls):
    pp = PPS[TRACER][fnl]
    p = gd.getParams()
    gd.addDerived(p.fNL_bp / (2 * DELTAC * (p.b1 - pp)), name='fNL_pp', label=r'f_\mathrm{NL}(p_\phi)')
    # gd.addDerived(p.fNL_bp / (2 * DELTAC * (B1S[TRACER] - pp)), name='fNL_pp', label=r'f_\mathrm{NL}(p_\phi)')
# reference for the gain: P_ell(k) sigma(fNL(p_phi)) per mock (gdsamps[3:6] are P_ell in FNLS order)
ref_fNL_bp = {fnl: gdsamps[3 + i].std('fNL_bp') for i, fnl in enumerate(FNLS)}
ref_fNL_pp = {fnl: gdsamps[3 + i].std('fNL_pp') for i, fnl in enumerate(FNLS)}
# gain = 1 - sigma(fNL)(run) / sigma(fNL)(P_ell) : 0 for P_ell, >0 if the run is tighter.
md_lines = [
    r"| $\mathrm{Run}$ | $\mathrm{true}\ f_\mathrm{NL}$ | $f_\mathrm{NL} b_\phi$ | $\mathrm{gain}$ | $f_\mathrm{NL}(p_\phi)$ | $\mathrm{gain}$ |",
    "|---:|---:|---:|---:|---:|---:|",
]
for gd, method, fnl in zip(gdsamps, methods, true_fnls):
    gain_fNL_bp = 1 - gd.std('fNL_bp') / ref_fNL_bp[fnl]
    gain_fNL_pp = 1 - gd.std('fNL_pp') / ref_fNL_pp[fnl]
    md_lines.append(
        rf"| ${method}$ | ${fnl}$ | ${gd.mean('fNL_bp'):.1f} \pm {gd.std('fNL_bp'):.1f}$ | ${gain_fNL_bp:.2f}$ "
        rf"| ${gd.mean('fNL_pp'):.2f} \pm {gd.std('fNL_pp'):.2f}$ | ${gain_fNL_pp:.2f}$ |"
    )
md_fn = FNLB_ROOT / "fnl_bp_pp_constraints.md"
md_fn.write_text("\n".join(md_lines) + "\n")
print(f"Saved {md_fn}")

# Zoom axes to the FLI constraint scale (union of FLI mean +-4 sigma over mocks); P_ell priors show flat.
param_limits = {}
for p in PARAMS:
    los = [s.mean(p) - 4 * s.std(p) for s in gdsamps[:3]]
    his = [s.mean(p) + 4 * s.std(p) for s in gdsamps[:3]]
    param_limits[p] = (min(los), max(his))

gdplt = plots.get_subplot_plotter(width_inch=8, scaling=False, rc_sizes=True)
gdplt.triangle_plot(roots=gdsamps,
                    params=PARAMS,
                    legend_labels=legend_labels,
                    filled=3 * [True] + 3 * [False],
                    contour_colors=2 * [SetDark2(i) for i in range(3)],
                    contour_ls=3 * ['-'] + 3 * ['--'],
                    contour_lws=6 * [1.5],
                    param_limits=param_limits,
                    )

# Legend: color = f_NL mock, linestyle = method (FLI solid, P_ell dashed).
from matplotlib.lines import Line2D
from matplotlib.transforms import IdentityTransform
leg = gdplt.legend
handles = list(leg.legend_handles) + [Line2D([], [], color='grey', ls='-', lw=1.5),
                                       Line2D([], [], color='grey', ls='--', lw=1.5)]
texts = [t.get_text() for t in leg.get_texts()] + [r'FLI', r'$P_\ell(k)$']
loc, bbox, fs = leg._loc, leg.get_bbox_to_anchor(), leg.get_texts()[0].get_fontsize()
leg.remove()
gdplt.fig.legend(handles, texts, loc=loc, bbox_to_anchor=bbox,
                 bbox_transform=IdentityTransform(), prop={'size': fs}, frameon=False)

for ext in ['pdf', 'png']:
    fn = FNLB_ROOT / f"fnlb_fli_vs_powell.{ext}"
    gdplt.export(str(fn))
    print(f"Saved {fn}")
