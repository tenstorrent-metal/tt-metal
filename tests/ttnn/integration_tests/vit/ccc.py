import torch
import transformers

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_vit.reference import torch_functional_vit
from models.utility_functions import torch_random, skip_for_wormhole_b0

from tests.ttnn.utils_for_testing import assert_with_pcc

torch.manual_seed(0)
model_name = "google/vit-base-patch16-224"
batch_size = 1
sequence_size = 197

if 1:
    config = transformers.ViTConfig.from_pretrained(model_name)
    # config.position_embedding_type = "none"
    model = transformers.ViTModel.from_pretrained(model_name, config=config).eval()
    # model = transformers.models.vit.modeling_vit.ViTEmbeddings(config).eval()

    print(model)

    # print(model.state_dict())
    # print(model.state_dict()["cls_token"].shape)
    # print(model.state_dict()["cls_token"].shape)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )
    print(parameters)
    # print(parameters.embeddings.patch_embeddings.projection.bias)
    # print(parameters.embeddings.cls_token)
