



from desipipe import Queue, Environment, TaskManager, FileManager

# Let's instantiate a Queue, which records all tasks to be performed
queue = Queue('test', base_dir='_tests')
# Pool of 4 workers
# Any environment variable can be passed to Environment: it will be set when running the tasks below
tm = TaskManager(queue, 
                 environ=Environment(), 
                 provider=dict(provider='nersc', time='00:30:00'))

# # Left untouched, this is a helper function, not a standalone task
# def draw_random_numbers(size):
#     import numpy as np
#     return np.random.uniform(-1, 1, size)

# # We decorate the function (task) with tm.python_app
# @tm.python_app
# def fraction(seed=42, size=10000, draw_random_numbers=draw_random_numbers):
#     # All definitions, except input parameters, must be in the function itself, or in its arguments
#     # and this, recursively:
#     # draw_random_numbers is defined above and all definitions, except input parameters, are in the function itself
#     # This is required for the tasks to be pickelable (~ can be converted to bytes)
#     import time
#     time.sleep(5)  # wait 5 seconds, just to show jobs are indeed run in parallel
#     x, y = draw_random_numbers(size), draw_random_numbers(size)
#     return np.sum((x**2 + y**2) < 1.) * 1. / size  # fraction of points in the inner circle of radius 1

# # Here we use another task manager, with only 1 worker
# tm2 = tm.clone(scheduler=dict(max_workers=1))
# @tm2.python_app
# def average(fractions):
#     import numpy as np
#     return np.average(fractions) * 4.

# # Let's add another task, to be run with bash
# @tm2.bash_app
# def echo(avg):
#     return ['echo', '-n', 'bash app says pi is ~ {:.4f}'.format(avg)]

# # The following line stacks all the tasks in the queue
# fractions = [fraction(seed=i) for i in range(20)]
# # fractions is a list of Future instances
# # We can pass them to other tasks, which creates a dependency graph
# avg = average(fractions)
# ech = echo(avg)


@tm.python_app
def run():
    import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.75' # NOTE: jax preallocates GPU (default 75%)
    import numpy as np
    from functools import partial
    import matplotlib.pyplot as plt
    from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices, jit, vmap, grad, debug, tree
    jconfig.update("jax_enable_x64", True)
    print(jdevices())

    from montecosmo.model import FieldLevelModel, default_config
    from montecosmo.utils import pdump, pload, Path
    import os

    # save_dir = os.path.expanduser("~/scratch/png/")
    # save_dir = os.path.expanduser("/lustre/fsn1/projects/rech/fvg/uvs19wt/png/")
    # save_dir = os.path.expanduser("/lustre/fswork/projects/rech/fvg/uvs19wt/workspace/png/") ###############
    save_dir = os.path.expanduser("/pscratch/sd/h/hsimfroy/png/")
    
    save_dir = save_dir / "lpt_64_fnl_0"
    save_path = save_dir / "test"
    save_dir.mkdir(parents=True, exist_ok=True)