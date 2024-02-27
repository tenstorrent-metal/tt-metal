import transformers
import ttnn
import models


def create_custom_preprocessor(model_config):
    def rotary_embedding_custom_processor(torch_model, name):
        parameters = {}
        if isinstance(torch_model, transformers.models.falcon.modeling_falcon.FalconRotaryEmbedding):
            parameters["cos_cached"] = ttnn.unsqueeze_to_4D(
                ttnn.from_torch(
                    torch_model.cos_cached,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=model_config["COS_CACHED_WEIGHTS_DTYPE"],
                    # memory_config=model_config["COS_CACHED_WEIGHTS_MEMCFG"] #TODO: tensor should support memconfg constructor
                )
            )
            parameters["sin_cached"] = ttnn.unsqueeze_to_4D(
                ttnn.from_torch(
                    torch_model.sin_cached,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=model_config["SIN_CACHED_WEIGHTS_DTYPE"],
                    # memory_config=model_config["SIN_CACHED_WEIGHTS_MEMCFG"],  #TODO: tensor should support memconfg constructor
                )
            )
        return parameters

    return rotary_embedding_custom_processor
