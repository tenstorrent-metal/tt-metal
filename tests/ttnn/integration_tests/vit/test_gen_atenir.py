import transformers
from datasets import load_dataset
import torch
from typing import Optional
from torch.nn import functional as F
from torchview import draw_graph
from datasets import load_dataset

from torch.fx.experimental.proxy_tensor import make_fx
from torch.func import functionalize

###
from transformers import AutoImageProcessor, ViTForImageClassification

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
pixel_values = image_processor(image, return_tensors="pt").pixel_values

print(pixel_values)

###

# inpt = (input_features, attention_mask,decoder_input_ids)

f_traced = make_fx(model)(pixel_values)
f_no_mutations_traced = make_fx(functionalize(model))(pixel_values)
f_no_mutations_and_views_traced = make_fx(functionalize(model, remove="mutations_and_views"))(pixel_values)
# print(f_traced.code)
# print("----------")
# print(f_no_mutations_traced.code)
# print(f_no_mutations_and_views_traced.code)
