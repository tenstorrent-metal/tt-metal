import torch
from torch.utils.data import Dataset
from typing import Any
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class ImageNetDataset(Dataset):
    """Configurable ImageNet Dataset."""

    def __init__(self, dataset: Any, feature_extractor: Any, preproc_style=None):
        """Init and preprocess ImageNet-1k dataset.

        Parameters
        ----------
        dataset : Any
            ImageNet-1k dataset
        feature_extractor : Any
            feature extractor object from HuggingFace
        """
        self.imagenet = dataset

        if preproc_style in ["resnet", "vit", "deit"]:
            self.data = [
                (
                    feature_extractor(item["image"].convert("RGB"), return_tensors="pt")["pixel_values"].squeeze(0),
                    item["label"],
                )
                for item in self.imagenet
            ]

        elif preproc_style in ["vgg"]:
            self.data = [
                (
                    transforms.ToTensor()(item["image"].resize((224,224))),
                    item["label"],
                )
                for item in self.imagenet
            ]


        # for huggingface models
        else:
            self.data = [
                (
                    feature_extractor(item["image"].convert("RGB"), return_tensors="pt")["pixel_values"].squeeze(0),
                    item["label"],
                )
                for item in self.imagenet
            ]

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Return sample from dataset.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        Tuple
            Data sample in format of X, y
        """
        X, y = self.data[index]
        return X, y


def imagenet_1K_samples_input(image_processor, preproc_style, microbatch=1, sample_count=1000):
    imagenet_dataset = load_dataset("imagenet-1k", split="validation", use_auth_token=True, streaming=True)
    dataset_iter = iter(imagenet_dataset)
    dataset = []
    for _ in range(sample_count):
        dataset.append(next(dataset_iter))
    dataset = ImageNetDataset(dataset=dataset, feature_extractor=image_processor, preproc_style=preproc_style)

    generator = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=True)
    input_data = []
    for batch, labels in generator:
        input_data.append((batch, [labels]))

    return input_data
