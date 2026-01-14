# data.py
# Exact translation of data.rs

import math
import random
from enum import Enum
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import functional as F


# -----------------------------
# Transform enum (Rust: enum Transform)
# -----------------------------
class Transform(Enum):
    Translate = "Tr"
    Shear = "Sr"
    Scale = "Sc"
    Rotation = "Rot"

    def __str__(self):
        return self.value


# -----------------------------
# Prepared MNIST item
# -----------------------------
class MnistItemPrepared:
    def __init__(self, image: Tensor, label: int):
        self.image = image          # [1, 28, 28]
        self.label = label          # int


# -----------------------------
# Mapper (Rust: MnistMapper)
# -----------------------------
class MnistMapper:
    def __init__(self):
        self.transforms: List[Transform] = []

    def transform(self, transforms: List[Transform]):
        self.transforms.extend(transforms)
        return self

    def map(self, image: Tensor, label: int) -> MnistItemPrepared:
        return prepare_image(self.transforms, image, label)


# -----------------------------
# Image preparation + augmentation
# -----------------------------
def prepare_image(
    transforms: List[Transform],
    image: Tensor,
    label: int,
) -> MnistItemPrepared:
    # image: [28, 28], uint8
    image = image.float()
    image = image.unsqueeze(0)  # [1, 28, 28]

    # Normalize exactly like Rust / PyTorch example
    image = (image / 255.0 - 0.1307) / 0.3081

    if transforms:
        image = mangle_image(transforms, image)

    return MnistItemPrepared(image=image, label=label)


def mangle_image(transforms: List[Transform], image: Tensor) -> Tensor:
    # Rust uses Transform2D::composed → PyTorch affine
    angle = 0.0
    translate = [0.0, 0.0]
    scale = 1.0
    shear = [0.0, 0.0]

    for t in transforms:
        if t == Transform.Translate:
            translate[0] += random.uniform(-0.2, 0.2) * image.shape[-1]
            translate[1] += random.uniform(-0.2, 0.2) * image.shape[-2]
        elif t == Transform.Shear:
            shear[0] += random.uniform(-0.6, 0.6)
            shear[1] += random.uniform(-0.6, 0.6)
        elif t == Transform.Scale:
            scale *= random.uniform(0.6, 1.5)
        elif t == Transform.Rotation:
            angle += random.uniform(-math.pi / 4, math.pi / 4) * 180 / math.pi

    return F.affine(
        image,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
    )


# -----------------------------
# Dataset wrappers (Rust: DatasetIdent, MapperDataset, SamplerDataset)
# -----------------------------
class MappedMnistDataset(Dataset):
    def __init__(self, base: Dataset, mapper: MnistMapper):
        self.base = base
        self.mapper = mapper

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        return self.mapper.map(image, label)


class SampledDataset(Dataset):
    def __init__(self, dataset: Dataset, size: int, with_replacement: bool):
        self.dataset = dataset
        self.indices = [
            random.randrange(len(dataset)) if with_replacement else i
            for i in range(size)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


# -----------------------------
# Collate function (Rust: MnistBatcher)
# -----------------------------
def mnist_collate(batch: List[MnistItemPrepared]):
    images = torch.stack([item.image for item in batch], dim=0)   # [B, 1, 28, 28]
    targets = torch.tensor([item.label for item in batch], dtype=torch.long)
    return images.squeeze(1), targets  # [B,28,28], [B]