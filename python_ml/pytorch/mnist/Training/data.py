# data.py
# Burn-equivalent MNIST data pipeline

import torch
from torch.utils.data import Dataset


class MnistItemPrepared:
    def __init__(self, image, label):
        self.image = image          # [1, 28, 28]
        self.label = label


def prepare_image(image, label):
    # image: [28, 28], uint8
    image = image.float()
    image = image.unsqueeze(0)     # [1, 28, 28]

    # Same normalization as Burn
    image = (image - 0.1307) / 0.3081

    return MnistItemPrepared(image, label)


class SimpleMnistDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        return prepare_image(image, label)


def mnist_collate(batch):
    images = torch.stack([item.image for item in batch], dim=0)
    targets = torch.tensor(
        [item.label for item in batch],
        dtype=torch.long
    )
    return images.squeeze(1), targets
