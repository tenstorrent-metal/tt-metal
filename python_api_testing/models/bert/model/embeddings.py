import torch

class TtEmbeddings(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return

class PytorchEmbeddings(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.embeddings = hugging_face_reference_model.bert.embeddings

        # Disable dropout
        self.eval()

    def forward(self, input_ids, segment_ids):
        return self.embeddings(input_ids, segment_ids)

def run_embeddings_inference():
    return
