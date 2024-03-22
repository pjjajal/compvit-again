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

def create_imagenet21k_dataset(args):
    if args.augmentations:
        transform = tvt.Compose(
            [
                tvt.ToImage(),
                tvt.ToDtype(torch.float32, scale=True),
                tvt.RandomResizedCrop(224),
                tvt.RandomHorizontalFlip(),
                tvt.RandomChoice([
                    tvt.GaussianBlur(7),
                    tvt.RandomSolarize(threshold=0.5, p=1), 
                    tvt.RandomGrayscale(p=1)
                ]),
                make_normalize_transform(),
            ]
        )
    else: 
        transform = tvt.Compose(
            [
                tvt.ToImage(),
                tvt.ToDtype(torch.float32, scale=True),
                tvt.RandomResizedCrop(224),
                tvt.RandomHorizontalFlip(),
                make_normalize_transform(),
            ]
        )

    return ImageFolder(root=args.data_dir, transform=transform)
