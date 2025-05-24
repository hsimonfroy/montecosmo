


# from desipipe import Queue, Environment, TaskManager
# from desipipe.environment import BaseEnvironment

# # Let's instantiate a Queue, which records all tasks to be performed
# queue = Queue('test', base_dir='_tests')

# # class MontEnv(BaseEnvironment):
# #     name = 'montenv'
# #     _defaults = dict(DESICFS='/global/cfs/cdirs/desi')
# #     _command = 'export CRAY_ACCEL_TARGET=nvidia80 ; ' \
# #                 'export MPICC="cc -shared ; ' \
# #                 'export SLURM_CPU_BIND="cores" ; ' \
# #                 'source activate montenv'

# # environ = MontEnv
# # environ = Environment("montenv")
# environ = Environment("nersc-cosmodesi")  # or your environnment, see https://github.com/cosmodesi/desipipe/blob/f0e8cafe63f5aa4ca80cc5e40c6b2efa61bcbcb5/desipipe/environment.py#L196

# output, error = './outs/slurm-%j.out', './outs/slurm-%j.err'
# tm = TaskManager(queue=queue, environ=environ, 
#                  scheduler=dict(max_workers=20), 
#                  provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=8, nodes_per_worker=2, output=output, error=error, constraint='gpu'))


# @tm.python_app
# def run():
#     print("ho")
#     import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.75' # NOTE: jax preallocates GPU (default 75%)
#     import numpy as np
#     from functools import partial
#     import matplotlib.pyplot as plt
#     from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices, jit, vmap, grad, debug, tree
#     jconfig.update("jax_enable_x64", True)
#     print(jdevices())

#     from montecosmo.model import FieldLevelModel, default_config
#     from montecosmo.utils import pdump, pload, Path
#     import os

#     # save_dir = os.path.expanduser("~/scratch/png/")
#     # save_dir = os.path.expanduser("/lustre/fsn1/projects/rech/fvg/uvs19wt/png/")
#     # save_dir = os.path.expanduser("/lustre/fswork/projects/rech/fvg/uvs19wt/workspace/png/") ###############
#     save_dir = os.path.expanduser("/pscratch/sd/h/hsimfroy/png/")
    
#     save_dir = save_dir / "lpt_64_fnl_00"
#     save_path = save_dir / "test"
#     # save_dir = "./lpt_64_fnl_0"
#     # save_path = save_dir + "test"
#     save_dir.mkdir(parents=True, exist_ok=True)

#     print("ho")

# if __name__ == '__main__':
#     from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices, jit, vmap, grad, debug, tree
#     jconfig.update("jax_enable_x64", True)
#     print(jdevices())
#     run()




from desipipe import Queue, Environment, TaskManager, spawn

queue = Queue('test', base_dir='_tests')
environ = Environment("nersc-cosmodesi")  # or your environnment, see https://github.com/cosmodesi/desipipe/blob/f0e8cafe63f5aa4ca80cc5e40c6b2efa61bcbcb5/desipipe/environment.py#L196

output, error = './outs/slurm-%j.out', './outs/slurm-%j.err'
tm = TaskManager(queue=queue, environ=environ, 
                 scheduler=dict(max_workers=20), 
                 provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=8, nodes_per_worker=2, output=output, error=error, constraint='gpu'))


@tm.python_app
def run():
    print("ho")

if __name__ == '__main__':
    print("hey")
    run()
    print("bye")

    spawn(queue)
