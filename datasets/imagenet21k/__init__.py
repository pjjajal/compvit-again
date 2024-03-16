from typing import Sequence

import torch
import torchvision.transforms.v2 as tvt
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import ImageFolder


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> tvt.Normalize:
    return tvt.Normalize(mean=mean, std=std)

def create_imagenet21k_dataset(data_dir, augmentations=False):
    augments = []
    if augmentations:
        augments.extend(
            [
                tvt.RandomChoice([
                    tvt.GaussianBlur(),
                    tvt.RandomSolarize(threshold=0.5, p=1), 
                    tvt.RandomGrayscale(p=1)
                ])
            ]
        )

    transform = tvt.Compose(
        [
            tvt.ToImage(),
            tvt.ToDtype(torch.float32, scale=True),
            tvt.RandomResizedCrop(224),
            tvt.RandomHorizontalFlip(),
            augments,
            make_normalize_transform(),
        ]
    )

    return ImageFolder(root=data_dir, transform=transform)
