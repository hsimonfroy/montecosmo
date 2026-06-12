from montecosmo.model import FieldLevelModel, default_config
from montecosmo.utils import pdump, pload , chreshape, r2chshape, boxreshape, rg2cgh, cgh2rg
from montecosmo.bricks import top_hat_selection, gen_gauss_selection

# box_size = 3*(2000,)
box_size = 3*(2760,)
overselect = None
selection = None if overselect is None else overselect + 0.05
# mesh_length = 256
# mesh_length = 48
mesh_length = 96
# mesh_length = 128
# z_obs = 0.8
z_obs = 1.

model = FieldLevelModel(**default_config | 
                        {'final_shape': 3*(mesh_length,), 
                        # 'cell_length': box_size[0] / mesh_length, # in Mpc/h
                        'cell_length': (1 if overselect is None else 1+overselect) * box_size[0] / mesh_length, # in Mpc/h
                        # 'box_center': (0.,0.,0.), # in Mpc/h
                        # 'box_center': (0.,0.,1.), # in Mpc/h
                        'box_center': (0.,0.,1938.), # in Mpc/h # a2chi(model.cosmo_fid, a=1/(1+z_obs))
                        # 'box_center': (0.,0.,-1.), # in Mpc/h
                        # 'box_center': (0.,1.,0.), # in Mpc/h

                        'box_rotvec': (0.,0.,0.,), # rotation vector in radians
                        'evolution': 'lpt',
                        # 'evolution': 'kaiser',
                        'a_obs': 1 / (1 + z_obs), # light-cone if None
                        'curved_sky': False, # curved vs. flat sky
                        'ap_auto': None, # parametrized AP vs. auto AP
                        'selection': selection, # if float, padded fraction, if str or Path, path to window mesh file
                        'paint_order':2, # order of interpolation kernel
                        'paint_deconv': True, # whether to deconvolve painted field
                        'kernel_type':'rectangular', # 'rectangular', 'kaiser_bessel'
                        'init_oversamp':1.5, # initial mesh 1D oversampling factor
                        'evol_oversamp':2., # evolution mesh 1D oversampling factor
                        'ptcl_oversamp':2., # particle cloud 1D oversampling factor
                        'paint_oversamp':2., # painted mesh 1D oversampling factor

                        # 'init_oversamp':1.5, # initial mesh 1D oversampling factor
                        # 'evol_oversamp':1.5, # evolution mesh 1D oversampling factor
                        # 'ptcl_oversamp':1.5, # particle cloud 1D oversampling factor
                        # 'paint_oversamp':1.5, # painted mesh 1D oversampling factor

                        # 'init_oversamp':1., # initial mesh 1D oversampling factor
                        # 'evol_oversamp':1., # evolution mesh 1D oversampling factor
                        # 'ptcl_oversamp':1., # particle cloud 1D oversampling factor
                        # 'paint_oversamp':1., # painted mesh 1D oversampling factor
                        'interlace_order':2, # interlacing order
                        'n_rbins': 1,
                        'k_cut': np.inf,
                        'init_power': load_dir / f'init_kpow.npy',
                        # 'init_power': None,
                        # 'lik_type': 'gaussian_delta_power',
                        'lik_type': 'gaussian_fourier',
                        'png_type': 'fNL_bias',
                        } )

truth = {
    'Omega_m': 0.3137721, 
    'sigma8': 0.8076353990239834,
    'b1': 0.,
    'b2': 0.,
    'bs2': 0.,
    
    'b1': 1.15,
    'b2': 0.2,
    'bs2': -0.2,
    'bn2': 0.,
    'bnpar': 0.,
    'fNL': 0.,
    'fNL_bp':0.,
    'fNL_bpd':0.,
    'alpha_iso': 1.,
    'alpha_ap': 1.,
    # 'ngbars': 8.43318125e-4,
    'ngbars': 1e-5,
    # 'ngbars': 10000., # neglect lik noise
    'sigma_0': 0.5,
    'sigma_2': 1e-6,
    'sigma_mu2': 1e-6,
    'sigma_delta': 0.7,
    }


latents = model.new_latents_from_loc(truth)
model = FieldLevelModel(**model.asdict() | {'latents': latents})
print(model)
# model.render()



# # Abacus matter
# # obs_mesh = jnp.load(load_dir / f'fin_paint2_interl2_deconv0_{mesh_length}.npy')
# # obs_mesh = jnp.load(load_dir / f'fin_paint2_interl1_deconv1_{mesh_length}.npy')
# obs_mesh = jnp.load(load_dir / f'fin_paint2_interl2_deconv1_{mesh_length}.npy')
# # obs_mesh = (1 + truth['b1']) * (obs_mesh - 1) + 1
# obs_mesh *= truth['ngbars'] * model.cell_length**3
# # var = truth['sigma_0'] * model.cell_length**3
# # obs_mesh += jr.normal(jr.key(44), obs_mesh.shape) * var**.5
# # # obs_mesh = jr.poisson(jr.key(44), jnp.abs(obs_mesh + 1) * mean_count)

# Abacus tracer real or redshift-space
# obs_mesh = jnp.load(load_dir / f'tracer_6746545_paint2_deconv1_{mesh_length}.npy')
# obs_mesh = jnp.load(load_dir / f'tracer_6746545_rsdflat_paint2_deconv1_{mesh_length}.npy')

fNL_true = 100
if fNL_true == 0:
    obs_mesh = jnp.load(load_dir / f'tracer_2099282_fNL0_paint2_deconv1_{mesh_length}.npy')
elif fNL_true == 100:
    obs_mesh = jnp.load(load_dir / f'tracer_2099376_fNL100_paint2_deconv1_{mesh_length}.npy')
elif fNL_true == -100:
    obs_mesh = jnp.load(load_dir / f'tracer_2099359_fNL-100_paint2_deconv1_{mesh_length}.npy')
obs_mesh *= truth['ngbars'] * model.cell_length**3


# Abacus initial
# init_mesh = jnp.fft.rfftn(jnp.load(load_dir / f'init_mesh_{576}.npy'))
init_mesh = jnp.fft.rfftn(jnp.load(load_dir / f'init_mesh_fake_2760_{256}.npy'))
init_mesh = chreshape(init_mesh, r2chshape(model.init_shape))
truth0 = truth | {'init_mesh': init_mesh} | {'obs': obs_mesh}
del obs_mesh
del init_mesh

# # Abacus within bigger volume 
# # /!\ Don't known init_mesh anymore, load a fake one
# init_mesh = jnp.fft.rfftn(jnp.load(load_dir / f"init_mesh_fake_3000_{256}.npy"))
# init_mesh = chreshape(init_mesh, r2chshape(model.init_shape))

# # obs_mesh = jnp.load(load_dir / f'tracer_6746545_paint2_deconv1_{256}.npy')
# obs_mesh = jnp.load(load_dir / f'tracer_6746545_rsdflat_paint2_deconv1_{256}.npy')
# over_shape = 3*(int((1+overselect) * 256),)
# print(f"{over_shape=}")
# selec_mesh = top_hat_selection(over_shape, model.selection)
# selec_mesh *= top_hat_selection(over_shape, 1., norm_order=4., pow_order=4.)
# # selec_mesh *= gen_gauss_selection(model.box_center, model.box_rot, model.box_size, over_shape, True, order=4.)
# selec_mesh /= selec_mesh[selec_mesh > 0].mean()

# obs_mesh = realreshape(obs_mesh, over_shape)
# obs_mesh *= selec_mesh
# obs_mesh = jnp.fft.rfftn(obs_mesh)
# obs_mesh = jnp.fft.irfftn(chreshape(obs_mesh, r2chshape(model.final_shape)))
# obs_mesh = model.mesh2masked(obs_mesh)
# obs_mesh *= truth['ngbars'] * model.cell_length**3 / obs_mesh.mean()
# truth0 = truth | {'init_mesh': init_mesh} | {'obs': obs_mesh}
# del obs_mesh

# Self-specified
truth |= {'init_mesh': truth0['init_mesh']}
truth1 = model.predict(samples=truth, hide_base=False, hide_samp=False, from_base=True)
# truth1 = model.predict(samples=truth, hide_base=False, hide_samp=False, hide_det=False, from_base=True)
# jnp.save(load_dir / f"lpt_ptcl_osamp1.5_2_{mesh_length}", truth1["lpt_ptcl"])
# truth1 = model.predict(samples=truth, hide_base=False, from_base=True)
# jnp.save(load_dir / f"init_mesh_fake_2760_{mesh_length}", jnp.fft.irfftn(truth1["init_mesh"]))


model2 = FieldLevelModel(**model.asdict() | {
                        # 'init_oversamp':1.5, # initial mesh 1D oversampling factor
                        # 'evol_oversamp':2., # evolution mesh 1D oversampling factor
                        # 'ptcl_oversamp':2., # particle cloud 1D oversampling factor
                        # 'paint_oversamp':2., # painted mesh 1D oversampling factor
                        # 'kernel_type':'kaiser_bessel',
                        # 'paint_order':4,
                        # 'png': True,
                                                })
# # coeff = 1.5
# # truth |= {'sigma8': truth0['sigma8'] * coeff, 'b1': (1+truth0['b1']) / coeff - 1}
# # truth |= {'sigma8': truth0['sigma8'] * coeff, 'init_mesh': truth0['init_mesh'] * coeff}
# # init_mesh = jnp.fft.rfftn(jnp.load(load_dir / f'init_mesh_{576}.npy'))
# # init_mesh = chreshape(init_mesh, r2chshape(model2.init_shape))
# # truth |= {'init_mesh': init_mesh}
truth |= {'ngbars': 1e6}
truth2 = model2.predict(samples=truth, hide_base=False, hide_samp=False, from_base=True)

# model2 = FieldLevelModel(**model.asdict() | {
#                                                 })
# coeff = 1.5
# # truth |= {'sigma8': truth0['sigma8'] * coeff, 'b1': (1+truth0['b1']) / coeff - 1}
# # truth |= {'sigma8': truth0['sigma8'] * coeff}
# truth |= {'sigma8': truth0['sigma8'] * coeff, 'init_mesh': truth0['init_mesh'] * coeff}
# truth2 = model2.predict(samples=truth, hide_base=False, hide_samp=False, from_base=True)

model.save(save_dir / "model.yaml")    
jnp.savez(save_dir / "truth.npz", **truth)
delta_obs0 = model.count2delta(truth0['obs'])
# delta_det = model.count2delta(truth1['obs'])

# delta_obs1 = model.count2delta(truth1['obs'])
# delta_obs2 = model2.count2delta(truth2['obs'])

delta_obs1 = model.count2delta(jnp.fft.irfftn(rg2cgh(truth1['obs'])))
delta_obs2 = model2.count2delta(jnp.fft.irfftn(rg2cgh(truth2['obs'])))

# delta_obs3 = model3.count2delta(truth3['obs'])
# delta_obs1 = truth1['obs'] / model.count_fid - 1
print(f'obs0 mean: {truth0['obs'].mean():.5e}, std: {truth0['obs'].std():.5e}')
print(f'obs1 mean: {truth1['obs'].mean():.5e}, std: {truth1['obs'].std():.5e}')

obs = ['obs','fNL','bnpar',
    #    'b1','b2','bs2','bn2',
       'ngbars', 'sigma_0', 'sigma_delta', 
       'Omega_m',
    #    'sigma8',
    #    'init_mesh',
       'alpha_iso','alpha_ap',]
obs = {k: truth0[k] for k in obs}

# model.substitute(obs, from_base=True)
# model.block()
# # params_start_ = jit(vmap(partial(model.kaiser_post, delta_obs=delta_obs0, scale_field=1.)))(jr.split(jr.key(45), n_chains)) 
# params_start_ = model.kaiser_post(jr.key(45), delta_obs0, scale_field=1/5)
# print('start params:', params_start_.keys())
# potential_valgrad = jit(value_and_grad(model.potential))
# # model.render()
# # potential_valgrad(params_start_ | {'sigma8_':0.3})