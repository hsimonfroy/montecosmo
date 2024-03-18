
import numpy as np
import jax.random as jr
from jax import jit, vmap, grad, debug, flatten_util, tree_util, eval_shape
from jax.tree_util import tree_map
from functools import partial

import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils

# disable internet connection
# gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: None
# gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False


from montecosmo.models import pmrsd_model, prior_model, get_logp_fn, get_score_fn, get_simulator, get_pk_fn, get_param_fn, get_noise_fn
from montecosmo.models import print_config, condition_on_config_mean, default_config as config
# Build and render model
config.update(a_lpt=0.5, mesh_size=16*np.ones(3, dtype=int))
model = partial(pmrsd_model, **config)
config['lik_config']['obs_std'] = 0.1
print_config(model)
fiduc_params = get_simulator(condition_on_config_mean(model))(rng_seed=0)

simulator = jit(vmap(get_simulator(model)))




# class MyDatasetConfig(tfds.core.BuilderConfig):
#     def __init__(self, name, description="",**kwargs):
#         v1 = tfds.core.Version("0.0.1")
#         super().__init__(name=name, description=description, version=v1)
#         for k, v in kwargs.items():
#             setattr(self, k ,v)

class mydataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("0.0.1")
    RELEASE_NOTES = {"0.0.1": "Initial release.",}
    # BUILDER_CONFIGS = [MyDatasetConfig(name="default", **config)]

    


    # def _info(self) -> tfds.core.DatasetInfo:
    #     shape_dtype_struct = eval_shape(lambda x:x, fiduc_params)
    #     feature_dict_fn = lambda x: tfds.features.Tensor(shape=x.shape, dtype=x.dtype)
    #     feature_dict = tree_util.tree_map(feature_dict_fn, shape_dtype_struct)
        
    #     return tfds.core.DatasetInfo(
    #         builder=self,
    #         description="""DESCRIPTION""",
    #         features=tfds.features.FeaturesDict(feature_dict),
    #         supervised_keys=None,
    #         homepage="https://dataset-homepage/",
    #         citation="""CITATION""",
    #         )

    def _info(self) -> tfds.core.DatasetInfo:
        shape_dtype_struct = eval_shape(lambda x:x, fiduc_params)
        feature_dict_fn = lambda x: tfds.features.Tensor(shape=x.shape, dtype=x.dtype)
        feature_dict = tree_util.tree_map(feature_dict_fn, shape_dtype_struct)

        return self.dataset_info_from_configs(features=tfds.features.FeaturesDict(feature_dict))
    
    def _split_generators(self, dl_manager:tfds.download.DownloadManager):
        return {'train': self._generate_examples(rng_key=jr.PRNGKey(42), 
                                                 size=100,
                                                 batch_size=10)}
        # return [
        #     tfds.core.SplitGenerator(name=tfds.Split.TRAIN, 
        #                              gen_kwargs={'rng_key': jr.PRNGKey(42), 'size': 10}),
        #     tfds.core.SplitGenerator(name=tfds.Split.TEST,  
        #                              gen_kwargs={'rng_key': jr.PRNGKey(43), 'size': 10}),]

    def _generate_examples(self, rng_key, size, batch_size):

        for i_b in range((size-1) // batch_size + 1):
            key, rng_key = jr.split(rng_key)
            simus = simulator(rng_seed=jr.split(key, batch_size))
            print(simus)
            # yield simus
            for j_b in range(batch_size):
                yield f"{i_b}-{j_b}",{key: simus[key][j_b] for key in simus}

