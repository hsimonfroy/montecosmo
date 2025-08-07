
from desipipe import Queue, Environment, TaskManager, spawn
from desipipe.environment import BaseEnvironment

queue = Queue('test2', base_dir='_test2')
queue.clear(kill=False)

# environ = Environment("nersc-cosmodesi")  # or your environnment, see https://github.com/cosmodesi/desipipe/blob/f0e8cafe63f5aa4ca80cc5e40c6b2efa61bcbcb5/desipipe/environment.py#L196

# class MontEnv(BaseEnvironment):
#     name = 'montenv'
#     _defaults = dict(DESICFS='/global/cfs/cdirs/desi')
#     _command = 'export CRAY_ACCEL_TARGET=nvidia80 ; ' \
#                 'export MPICC="cc -shared" ; ' \
#                 'export SLURM_CPU_BIND="cores" ; ' \
#                 'source activate montenv'

environ = BaseEnvironment(command='source /global/homes/h/hsimfroy/miniforge3/bin/activate montenv')

output, error = './outs/slurm-%j.out', './outs/slurm-%j.err'
tm = TaskManager(queue=queue, environ=environ, 
                 scheduler=dict(max_workers=12), 
                 provider=dict(provider='nersc', time='04:00:00', 
                               mpiprocs_per_worker=1, nodes_per_worker=1, 
                               output=output, error=output, 
                               constraint='gpu', 
                            #    qos='debug',
                            #    qos='shared',
                               qos='regular',
                            #    qos='interactive',
                               ))









@tm.python_app
def infer_model(ap_auto):
    import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='1.' # NOTE: jax preallocates GPU (default 75%)
    import numpy as np
    from functools import partial
    import matplotlib.pyplot as plt
    from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices, jit, vmap, grad, debug, tree
    jconfig.update("jax_enable_x64", True)
    print('\n', jdevices())

    from montecosmo.model import FieldLevelModel, default_config
    from montecosmo.utils import pdump, pload, Path

    # save_dir = Path(os.path.expanduser("~/scratch/png/"))
    # save_dir = Path("/lustre/fsn1/projects/rech/fvg/uvs19wt/png/") # JZ
    # save_dir = Path("/lustre/fswork/projects/rech/fvg/uvs19wt/workspace/png/") # JZ
    save_dir = Path("/pscratch/sd/h/hsimfroy/png/") # Perlmutter
    
    save_dir = save_dir / f"lpt_32_apauto_{ap_auto:d}_nodec2"
    save_path = save_dir / "test"
    save_dir.mkdir(parents=True, exist_ok=True)

    truth0 = {'Omega_m': 0.3111, 
        'sigma8': 0.8102,
        'b1': 1.,
        'b2': 0., 
        'bs2': 0., 
        'bn2': 0.,
        'fNL': 0.,
        'alpha_iso': 1.,
        'alpha_ap': 1.,
        'ngbar': 1e-3,}
    cell_budget = 32**3
    padding = 0.2

    config = {'mesh_shape': 3*(64,), 
            'cell_length': 10., 
            'box_center': (0.,0.,2000.), # in Mpc/h
            'box_rotvec': (0.,0.,0.), # rotation vector in radians
            'evolution': 'lpt',
            'a_obs': None, # light-cone if None
            'curved_sky': True, # curved vs. flat sky
            'ap_auto': ap_auto, # parametrized AP vs. auto AP
            'window': padding, # if float, padded fraction, if str or Path, path to window mesh file
            }

    overwrite = False
    from montecosmo.script import load_model, warmup1, warmup2run, make_chains
    model, truth = load_model(truth0, config, cell_budget, padding, save_dir, overwrite)

    n_samples, n_runs, n_chains, tune_mass = 128, 64, 6, True  
    print(f"n_samples={n_samples}, n_runs={n_runs}, n_chains={n_chains}, tune_mass={tune_mass}")
    
    model, params_warm = warmup1(save_path, n_chains, overwrite)
    warmup2run(model, params_warm, save_path, n_samples, n_runs, n_chains, tune_mass, overwrite)

    make_chains(save_path)






if __name__ == '__main__':
    print("hey")
    
    infer_model(True)

    infer_model(False)

    spawn(queue, spawn=True)

    print("bye")





