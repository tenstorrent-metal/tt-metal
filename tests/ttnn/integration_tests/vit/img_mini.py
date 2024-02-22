import torch
import transformers

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_vit.reference import torch_functional_vit
from models.utility_functions import torch_random, skip_for_wormhole_b0

from tests.ttnn.utils_for_testing import assert_with_pcc


def vit_patch_emb():
    query = torch_random((1, 3, 6, 6), -0.1, 0.1, dtype=torch.float32)
    print(query.shape)
    print(query)
    query = torch.reshape(query, (1, 3, 6, 2, 3))
    # print(query.shape)
    query = torch.reshape(query, (1, 3, 2, 3, 2, 3))
    # print(query.shape)
    query = torch.permute(query, (0, 1, 2, 4, 3, -1))
    # print(query.shape)
    query = torch.reshape(query, (1, 3, 4, 9))
    # print(query.shape)a
    query = torch.permute(query, (0, 2, 1, 3))
    query = torch.reshape(query, (1, 4, 27))
    print(query.shape)
    print(query)


vit_patch_emb()
