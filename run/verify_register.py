#!/usr/bin/env python
"""
Verify registered mocks (run/register.py) against the FieldLevelModel forward model.

For a set of budgets N (mesh_length) of the SAME mock, produce one 2x3 figure:

  Row 1 : the mock ``count_mesh`` for each budget, via ``plot.plot_mesh``.
  Row 2 : power / transfer / coherence (``plot.plot_powtranscoh``, log=False), comparing the
          mock ``count_mesh`` (reference, mesh0) to the model prediction (mesh1) obtained with the
          TRUE ``white_mesh`` and a fiducial bias/PNG loc, in 3 SetDark2 colors (one per budget).
          - solid  (per budget): model prediction  (from powtranscoh)
          - dotted (per budget): mock spectrum      (spectrum(count2delta(count_mesh)))
          - black dashed        : theoretical Kaiser(+shot) monopole, without PNG (alpha=0.5)
                                  and with the fNL scale-dependent bias b_phi (alpha=1)
          - transfer black dashed : (P_Kaiser / P_mock)^.5 per budget
          - coherence dashed (per budget color): coherence(count_mesh, white_mesh downsampled)
          - axhline(1) dotted in the transfer and coherence subplots.

Fiducial: everything zero except the (Lagrangian) linear bias b1 and the PNG mass-relation pp,
and fNL from the mock name. b1 = 1.1 (LRG) / 1.674 (QSO). pp per (tracer, fNL):
  LRG {-20:0.80, 0:0.80, +20:1.13},  QSO {-20:0.83, 0:0.83, +20:1.55}   (PNGUNIT-XL fits).

Run on a GPU node with the montenv env:
    export XLA_PYTHON_CLIENT_MEM_FRACTION='1.'
    python -u src/montecosmo/run/verify_register.py
"""
import os; os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.99')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jax import config as jconfig
jconfig.update("jax_enable_x64", True)

from montecosmo.model import FieldLevelModel, default_config
from montecosmo.bricks import b1_L2E, b_phi, lin_power_interp, trans_phi2delta_interp
from montecosmo.nbody import a2g, a2f
from montecosmo.metrics import kaiser_formula
from montecosmo.plot import plot_mesh, plot_pow, plot_trans, plot_coh, plot_powtranscoh, SetDark2
from montecosmo.utils import chreshape, r2chshape

REGISTERED_DIR = Path("/pscratch/sd/h/hsimfroy/png/registered")
VERIF_DIR = Path("/pscratch/sd/h/hsimfroy/png/verif_register")

# Per-tracer fiducial Lagrangian linear bias b1 (b1E = 1 + b1) and per-(tracer, fNL) pp.
# B1_FID = {'LRG': 1.1, 'QSO': 1.674}
B1_FID = {'LRG': 1.05, 'QSO': 1.674}
PP_FID = {'LRG': {-20: 0.80, 0: 0.80, 20: 1.13},
          'QSO': {-20: 0.83, 0: 0.83, 20: 1.55}}


def kaiser_theory(model, ks, b1E, fNL_bp, with_png):
    """
    Kaiser monopole P0(k) + shot noise at the model fiducial scale factor, in (Mpc/h)^3.
    Without PNG this is metrics.kaiser_formula(b1E); with PNG the linear bias becomes the
    scale-dependent b_eff(k) = b1E + fNL_bp / M(k), with M = trans_phi2delta (phi -> delta).
    """
    cosmo, a = model.cosmo_fid, model.a_fid
    pk_lin = lin_power_interp(cosmo, kpow=model.lin_kpow)(ks)  # sigma8^2-normalized, z=0
    if not with_png:
        _, pk = kaiser_formula(cosmo, a, (ks, pk_lin), b1E, ells=0)
        p0 = np.asarray(pk[0])
    else:
        f = a2f(cosmo, a)
        m_k = trans_phi2delta_interp(cosmo, a=a, kpow=model.lin_kpow)(ks)
        b_eff = b1E + fNL_bp / m_k
        p0 = np.asarray((b_eff**2 + 2/3 * f * b_eff + 1/5 * f**2) * a2g(cosmo, a)**2 * pk_lin)
    shot = 1. / float(np.mean(model.fiduc['ngbars']))  # 1/nbar in (Mpc/h)^3
    return p0 + shot


def verify(name_tmpl, budgets, tracer, fnl, evolution='lpt', save_dir=VERIF_DIR):
    """
    name_tmpl : register filename with a '{N}' placeholder for the mesh_length, e.g.
                'register_pngunitbox_fNL-20_LRG_z0.74_redshiftspace_N{N}_pad0.h5'.
    budgets   : list of mesh_lengths, e.g. [32, 48, 64].
    tracer    : 'LRG' or 'QSO'.  fnl : mock f_NL in {-20, 0, 20}.
    """
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    pp = PP_FID[tracer][fnl]
    b1 = B1_FID[tracer]                 # Lagrangian
    b1E = b1_L2E(b1)                    # Eulerian, for the theory
    fNL_bp = fnl * b_phi(b1, pp=pp)     # fNL x b_phi(b1_L, pp), same as fNL_bias(png_type='fNL')

    fiduc = {  # all zero except b1 and the PNG mass-relation pp; sensible unit defaults elsewhere
        'b1': b1, 'b2': 0., 'bs2': 0., 'b3': 0., 'bds2': 0., 'bs3': 0., 'bn2': 0., 'bnpar': 0.,
        'fNL': float(fnl), 'fNL_bp': fNL_bp, 'fNL_bpd': 0., 'pp': pp,
        'fNL_bpd2': 0., 'fNL_bps2': 0., 'fNL_bn2p': 0.,
        's_e': 1., 's_k2e': 0., 's_kmu2e': 0., 's_ed': 0., 's_e2': 0., 's_ep': 0.,
        'alpha_iso': 1., 'alpha_ap': 1.,
        }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), layout='constrained')
    row2 = list(axes[1])

    model_ref, ks_list, powmock_list = None, [], []
    best_coh = 0.
    for i, N in enumerate(budgets):
        color = SetDark2(i)
        register = REGISTERED_DIR / name_tmpl.format(N=N)
        print(f"[N={N}] loading {register.name}")
        latents = FieldLevelModel.new_latents_from_loc(default_config['latents'], fiduc, update_prior=True)
        model = FieldLevelModel(**default_config | {
            'evolution': evolution,
            'lik_type': 'shash',
            'png_type': 'fNL',
            # 'png_type': 'fNL_bias',
            'register': register,  # overrides geometry/painting/cosmo/ngbars + loads the meshes
            'latents': latents,
            'n_rbins': 1,
            'evol_oversamp': 2.5,
            'ptcl_oversamp': 2.5,
            })

        # Model prediction with the TRUE white_mesh; DO NOT overwrite model.count_mesh.
        pred = model.predict(samples=model.fiduc | {'white_mesh': model.white_mesh},
                             hide_base=False, hide_samp=False, from_base=True)
        pred_count = pred['count_mesh']

        # --- Row 1: mock count_mesh ---
        plt.sca(axes[0, i])
        quad = plot_mesh(np.asarray(model.count_mesh), box_size=model.box_size, ids=0.1)
        plt.colorbar(quad, fraction=0.046, pad=0.02)
        plt.title(f"$N={N}$  ({model.cell_length:.0f} Mpc/$h$/cell, "
                  f"$n_g={float(np.mean(model.fiduc['ngbars'])):.1e}$)")

        # --- Row 2: pow / trans / coh, mock (ref, mesh0) vs prediction (mesh1) ---
        d_mock = model.count2delta(model.count_mesh)
        d_pred = model.count2delta(pred_count)
        ks, pow_pred, trans, coh = model.powtranscoh(d_mock, d_pred)
        ks_m, pow_mock = model.spectrum(d_mock)  # == pow0

        # prediction (solid) via powtranscoh, mock spectrum (dotted) added on the power subplot
        plot_powtranscoh(ks, pow_pred, trans, coh, color=color, log=False, axes=row2,
                         label=f"$N={N}$ pred")
        plt.sca(row2[0])
        plot_pow(ks_m, pow_mock, ':', color=color, log=False, label=f"$N={N}$ mock")

        # coherence(count_mesh, white_mesh downsampled) -> registration sanity check
        white_down = chreshape(model.white_mesh, r2chshape(model.final_shape))
        kw, _, _, coh_w = model.powtranscoh(d_mock, white_down)
        plt.sca(row2[2])
        plot_coh(kw, coh_w, '--', color=color, log=False)
        if coh_w.max() > best_coh:
            best_coh = coh_w.max()

        model_ref = model  # theory params (cosmo, a_fid, lin_kpow, ngbar) are budget-independent
        ks_list.append(np.asarray(ks_m)); powmock_list.append(np.asarray(pow_mock))

    # --- Black-dashed theoretical references ---
    # Power: Kaiser + shot (no PNG, alpha 0.5) and Kaiser + shot + b_phi (with PNG), on the widest ks.
    ks_th = ks_list[int(np.argmax([k.max() for k in ks_list]))]
    p_nopng = kaiser_theory(model_ref, ks_th, b1E, fNL_bp, with_png=False)
    p_png = kaiser_theory(model_ref, ks_th, b1E, fNL_bp, with_png=True)
    plt.sca(row2[0])
    plot_pow(ks_th, p_nopng, 'k--', alpha=0.5, log=False, label='Kaiser+shot')
    plot_pow(ks_th, p_png, 'k--', log=False, label=r'Kaiser+shot+$f_\mathrm{NL}b_\phi$')
    plt.legend(fontsize=8)

    # Transfer: (P_Kaiser / P_mock)^.5 per budget (with PNG solid-black-dashed, no PNG alpha 0.5).
    plt.sca(row2[1])
    for ks_i, pm_i in zip(ks_list, powmock_list):
        plot_trans(ks_i, (kaiser_theory(model_ref, ks_i, b1E, fNL_bp, True) / pm_i)**.5,
                   'k--', log=False)
        plot_trans(ks_i, (kaiser_theory(model_ref, ks_i, b1E, fNL_bp, False) / pm_i)**.5,
                   'k--', alpha=0.5, log=False)
    plt.axhline(1., linestyle=':', color='k', alpha=0.5)

    plt.sca(row2[2])
    plt.axhline(1., linestyle=':', color='k', alpha=0.5)

    row2[0].set_title("power (solid: pred, dashed: Kaiser, dotted: mock)")
    row2[1].set_title(r"transfer with mock (solid: pred, dashed: Kaiser)")
    row2[2].set_title(f"coherence with mock, best {best_coh:.3f}")

    stem = name_tmpl.format(N="x").replace("_Nx_pad0.h5", "").replace("register_", "")
    fig.suptitle(stem, fontsize=11)
    out = save_dir / f"verify_{stem}_b1_1.05.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")
    return out


if __name__ == "__main__":

    # verify(name_tmpl="register_abacus_c0_ph0_z0.800_LRG_redshiftspace_N{N}_pad0.h5",
    #        budgets=[32, 64, 128], tracer='LRG', fnl=0, evolution='lpt')
    
    # verify(name_tmpl="register_pngunitbox_fNL-20_LRG_z0.74_redshiftspace_N{N}_pad0.h5",
    #        budgets=[32, 48, 64], tracer='LRG', fnl=-20, evolution='lpt')

    verify(name_tmpl="register_pngunitbox_fNL-20_QSO_z1.83_redshiftspace_N{N}_pad0.h5",
           budgets=[32, 48, 64], tracer='QSO', fnl=-20, evolution='lpt')

    verify(name_tmpl="register_pngunitbox_fNL0_QSO_z1.83_redshiftspace_N{N}_pad0.h5",
           budgets=[32, 48, 64], tracer='QSO', fnl=0, evolution='lpt')

    verify(name_tmpl="register_pngunitbox_fNL20_QSO_z1.83_redshiftspace_N{N}_pad0.h5",
           budgets=[32, 48, 64], tracer='QSO', fnl=20, evolution='lpt')

