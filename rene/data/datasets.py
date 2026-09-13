"""Datasets of Section 5.1: CIFAR-10, CIFAR-100 and ImageNet-1K.

The search stage (Eq. 7) consumes a *validation* split that is held out of the
training set, so no training data is used to pick the ranks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

CIFAR10_STATS = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
CIFAR100_STATS = ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


@dataclass
class DataConfig:
    dataset: str = 'cifar10'
    root: str = './data'
    batch_size: int = 128
    search_batch_size: int = 256
    workers: int = 4
    val_fraction: float = 0.1        # held out of train, used by Eq. (7)
    seed: int = 0
    download: bool = True
    image_size: int = 224            # ImageNet only
    search_augment: bool = False     # training augmentation on the search split


def _cifar_transforms(stats, train: bool):
    from torchvision import transforms
    mean, std = stats
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def _imagenet_transforms(train: bool, size: int):
    from torchvision import transforms
    mean, std = IMAGENET_STATS
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    return transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])


def build_datasets(cfg: DataConfig):
    """Returns ``(train_set, search_val_set, test_set, num_classes)``.

    ``search_val_set`` is a held-out slice of the training data with *test-time*
    transforms; it is what the rank-coefficient update of Eq. (7) sees.
    """
    import os
    from torchvision import datasets

    name = cfg.dataset.lower()
    if name in ('cifar10', 'cifar100'):
        cls = datasets.CIFAR10 if name == 'cifar10' else datasets.CIFAR100
        stats = CIFAR10_STATS if name == 'cifar10' else CIFAR100_STATS
        n_cls = 10 if name == 'cifar10' else 100
        train_full = cls(cfg.root, train=True, download=cfg.download,
                         transform=_cifar_transforms(stats, True))
        val_full = cls(cfg.root, train=True, download=False,
                       transform=_cifar_transforms(stats, cfg.search_augment))
        test = cls(cfg.root, train=False, download=cfg.download,
                   transform=_cifar_transforms(stats, False))
    elif name == 'imagenet':
        train_full = datasets.ImageFolder(os.path.join(cfg.root, 'train'),
                                          _imagenet_transforms(True, cfg.image_size))
        val_full = datasets.ImageFolder(os.path.join(cfg.root, 'train'),
                                        _imagenet_transforms(cfg.search_augment, cfg.image_size))
        test = datasets.ImageFolder(os.path.join(cfg.root, 'val'),
                                    _imagenet_transforms(False, cfg.image_size))
        n_cls = 1000
    else:
        raise ValueError(f'unknown dataset {cfg.dataset!r}')

    g = torch.Generator().manual_seed(cfg.seed)
    n = len(train_full)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(round(cfg.val_fraction * n))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return Subset(train_full, train_idx), Subset(val_full, val_idx), test, n_cls


def build_loaders(cfg: DataConfig, pin_memory: bool = False):
    """Returns ``(train_loader, search_val_loader, test_loader, num_classes)``."""
    train_set, val_set, test_set, n_cls = build_datasets(cfg)
    common = dict(num_workers=cfg.workers, pin_memory=pin_memory,
                  persistent_workers=cfg.workers > 0)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(val_set, batch_size=cfg.search_batch_size, shuffle=True,
                            drop_last=False, **common)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader, n_cls


def input_size_for(dataset: str, image_size: int = 224) -> Tuple[int, int, int, int]:
    return (1, 3, 32, 32) if dataset.lower().startswith('cifar') else (1, 3, image_size, image_size)
