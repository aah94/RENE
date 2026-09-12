"""Model factory covering every backbone used in the paper's experiments."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .cifar import resnet20, resnet32, resnet56, vgg16

_CIFAR_BUILDERS = {
    'resnet20': resnet20, 'resnet32': resnet32, 'resnet56': resnet56, 'vgg16': vgg16,
}

# ImageNet backbones come from torchvision so we can load the official
# pre-trained weights that the paper's Table 2 baselines are measured against.
_IMAGENET_BUILDERS = {
    'resnet18': ('resnet18', 'ResNet18_Weights'),
    'resnet34': ('resnet34', 'ResNet34_Weights'),
    'resnet50': ('resnet50', 'ResNet50_Weights'),
    'mobilenetv2': ('mobilenet_v2', 'MobileNet_V2_Weights'),
}


def build_model(arch: str, dataset: str = 'cifar10', pretrained: bool = False,
                num_classes: Optional[int] = None) -> nn.Module:
    """Instantiate ``arch`` for ``dataset``.

    ``dataset`` in {'cifar10', 'cifar100', 'imagenet'}. For ImageNet the
    torchvision checkpoints are used when ``pretrained`` is set.
    """
    arch = arch.lower().replace('-', '').replace('_', '')
    dataset = dataset.lower()
    n_cls = num_classes or {'cifar10': 10, 'cifar100': 100, 'imagenet': 1000}[dataset]

    if dataset.startswith('cifar'):
        if arch in _CIFAR_BUILDERS:
            return _CIFAR_BUILDERS[arch](n_cls)
        # ImageNet-style backbones fine-tuned on CIFAR (used in Fig. 4)
        return _torchvision(arch, pretrained=pretrained, num_classes=n_cls, adapt_cifar=True)

    return _torchvision(arch, pretrained=pretrained, num_classes=n_cls, adapt_cifar=False)


def _torchvision(arch: str, pretrained: bool, num_classes: int, adapt_cifar: bool):
    import torchvision.models as tvm
    if arch not in _IMAGENET_BUILDERS:
        raise ValueError(f'unknown architecture {arch!r}; '
                         f'known: {sorted(set(_CIFAR_BUILDERS) | set(_IMAGENET_BUILDERS))}')
    fn_name, weights_enum = _IMAGENET_BUILDERS[arch]
    weights = getattr(tvm, weights_enum).DEFAULT if pretrained else None
    model = getattr(tvm, fn_name)(weights=weights)

    if num_classes != 1000:
        model = _replace_classifier(model, num_classes)
    if adapt_cifar:
        model = _adapt_stem_for_cifar(model)
    return model


def _replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        else:
            last = model.classifier[-1]
            model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return model


def _adapt_stem_for_cifar(model: nn.Module) -> nn.Module:
    """32x32 inputs: 3x3 stride-1 stem, no max-pool (standard CIFAR adaptation)."""
    if hasattr(model, 'conv1') and isinstance(model.conv1, nn.Conv2d):
        old = model.conv1
        model.conv1 = nn.Conv2d(old.in_channels, old.out_channels, 3, 1, 1, bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    if hasattr(model, 'maxpool'):
        model.maxpool = nn.Identity()
    if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
        first = model.features[0]
        if isinstance(first, nn.Sequential) and isinstance(first[0], nn.Conv2d):
            c = first[0]
            first[0] = nn.Conv2d(c.in_channels, c.out_channels, 3, 1, 1, bias=False)
    return model


def load_checkpoint(model: nn.Module, path: str, strict: bool = True) -> nn.Module:
    state = torch.load(path, map_location='cpu', weights_only=False)
    state = state.get('model', state.get('state_dict', state))
    state = { k.replace('module.', ''): v for k, v in state.items() }
    model.load_state_dict(state, strict=strict)
    return model
