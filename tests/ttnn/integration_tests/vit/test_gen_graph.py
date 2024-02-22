import transformers

from datasets import load_dataset
import torch
from typing import Optional
from torch.nn import functional as F

from torchview import draw_graph

from transformers import AutoImageProcessor, ViTForImageClassification, ViTConfig

"""
config = transformers.ViTConfig.from_pretrained("google/vit-base-patch16-224")
#print(config)
#model = transformers.models.vit.modeling_vit.ViTAttention(config)
model = transformers.models.vit.modeling_vit(config)
#print(model)
"""


dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
inputs = image_processor(image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

model_graph = draw_graph(
    model,
    input_size=(1, 3, 224, 224),
    dtypes=[torch.float32],
    expand_nested=True,
    graph_name="vit_lvl_1",
    depth=1,
    directory=".",
)
model_graph.visual_graph.render(format="svg")


model_graph = draw_graph(
    model,
    input_size=(1, 3, 224, 224),
    dtypes=[torch.long],
    expand_nested=True,
    graph_name="vit_lvl_2",
    depth=2,
    directory=".",
)
model_graph.visual_graph.render(format="svg")


model_graph = draw_graph(
    model,
    input_size=(1, 3, 224, 224),
    dtypes=[torch.long],
    expand_nested=True,
    graph_name="vit_lvl_3",
    depth=3,
    directory=".",
)
model_graph.visual_graph.render(format="svg")


model_graph = draw_graph(
    model,
    input_size=(1, 3, 224, 224),
    dtypes=[torch.long],
    expand_nested=True,
    graph_name="vit_lvl_4",
    depth=4,
    directory=".",
)
model_graph.visual_graph.render(format="svg")


model_graph = draw_graph(
    model,
    input_size=(1, 3, 224, 224),
    dtypes=[torch.long],
    expand_nested=True,
    graph_name="vit_lvl_5",
    depth=5,
    directory=".",
)
model_graph.visual_graph.render(format="svg")
