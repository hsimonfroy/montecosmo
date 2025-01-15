import os
from montecosmo.model import FieldLevelModel, default_config

def get_save_dir(**kwargs):
    dir = os.path.expanduser("~/scratch/pickles/")
    # dir = os.path.expanduser("/lustre/fsn1/projects/rech/fvg/uvs19wt/pickles/")
    # dir = os.path.expanduser("/lustre/fswork/projects/rech/fvg/uvs19wt/workspace/pickles/")

    dir += f"m{kwargs['mesh_shape'][0]:d}_b{kwargs['box_shape'][0]:.1f}"
    dir += f"_al{kwargs['a_lpt']:.1f}_ao{kwargs['a_obs']:.1f}_lo{kwargs['lpt_order']:d}_pc{kwargs['precond']:d}_ob{kwargs['obs']}/"
    return dir

def from_id(id):
    args = ParseSlurmId(id)
    config = {
          'mesh_shape':3 * (args.mesh_length,),
          'box_shape':3 * (args.box_length if args.box_length is not None else 5. * args.mesh_length,), 
          'a_lpt':args.a_obs if args.lpt_order > 0 else args.a_lpt,
          'a_obs':args.a_obs,
          'lpt_order':1 if args.lpt_order==1 else 2, # 2lpt + pm for 0
          'precond':args.precond,
          'obs':args.obs
          }
    save_dir = get_save_dir(**config)
    model = FieldLevelModel(**default_config | config)
    
    mcmc_config = {
        'sampler':args.sampler,
        'target_accept_prob':0.65,
        'n_samples':64 if args.sampler == 'NUTSwG' else 64, #
        'max_tree_depth':10,
        'n_runs':10,
        'n_chains':8
    }
    save_path = save_dir 
    save_path += f"s{mcmc_config['sampler']}_nc{mcmc_config['n_chains']:d}_ns{mcmc_config['n_samples']:d}"
    save_path += f"_mt{mcmc_config['max_tree_depth']:d}_ta{mcmc_config['target_accept_prob']}"

    return model, mcmc_config, save_dir, save_path

class ParseSlurmId():
    def __init__(self, id):
        self.id = str(id)

        dic = {}
        dic['mesh_length'] = [8,16,32,64,128]
        dic['lpt_order'] = [0,1,2]
        dic['precond'] = [0,1,2,3]
        dic['sampler'] = ['NUTS', 'HMC', 'NUTSwG', 'MCLMC']

        dic['box_length'] = [None]
        dic['a_lpt'] = [0.1]
        dic['a_obs'] = [0.5]
        dic['obs'] = ['mesh']
        
        for i, (k, v) in enumerate(dic.items()):
            if i < len(self.id):
                setattr(self, k, v[int(self.id[i])])
            else:
                setattr(self, k, v[0])