"""CIFAR backbones used in Tables 1 and 3: ResNet-20/56 and VGG-16."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# ResNet for CIFAR (He et al., 3 stages of n blocks; ResNet-20 -> n=3, 56 -> n=9)
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = x if self.downsample is None else self.downsample(x)
        return F.relu(out + shortcut)


class ResNetCifar(nn.Module):
    def __init__(self, n_blocks: int, num_classes: int = 10, width: int = 16):
        super().__init__()
        self.in_planes = width
        self.conv1 = nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(width, n_blocks, 1)
        self.layer2 = self._make_layer(width * 2, n_blocks, 2)
        self.layer3 = self._make_layer(width * 4, n_blocks, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width * 4, num_classes)
        self._init_weights()

    def _make_layer(self, planes, blocks, stride):
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def resnet20(num_classes: int = 10):
    return ResNetCifar(3, num_classes)


def resnet32(num_classes: int = 10):
    return ResNetCifar(5, num_classes)


def resnet56(num_classes: int = 10):
    return ResNetCifar(9, num_classes)


# --------------------------------------------------------------------------- #
# VGG-16 (BN) for CIFAR
# --------------------------------------------------------------------------- #
VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
             512, 512, 512, 'M', 512, 512, 512, 'M']


class VGGCifar(nn.Module):
    def __init__(self, cfg=VGG16_CFG, num_classes: int = 10, batch_norm: bool = True):
        super().__init__()
        layers, in_c = [], 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
                continue
            layers.append(nn.Conv2d(in_c, v, 3, padding=1, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            in_c = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


def vgg16(num_classes: int = 10):
    return VGGCifar(VGG16_CFG, num_classes)
