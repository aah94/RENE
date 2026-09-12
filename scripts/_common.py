"""Shared CLI plumbing for the experiment scripts."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rene.data import DataConfig, build_loaders, input_size_for      # noqa: E402
from rene.models import build_model, load_checkpoint                 # noqa: E402
from rene.search import LayerSelectionRule                           # noqa: E402
from rene.utils import get_logger, pick_device, set_seed             # noqa: E402


def add_common_args(p: argparse.ArgumentParser):
    p.add_argument('--arch', default='resnet20')
    p.add_argument('--dataset', default='cifar10',
                   choices=['cifar10', 'cifar100', 'imagenet'])
    p.add_argument('--data-root', default='./data')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--search-batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='./runs/exp')
    p.add_argument('--checkpoint', default=None,
                   help='path to the pre-trained (uncompressed) weights')
    p.add_argument('--pretrained', action='store_true',
                   help='use torchvision ImageNet weights')
    return p


def add_layer_rule_args(p: argparse.ArgumentParser):
    g = p.add_argument_group('layer selection')
    g.add_argument('--include-linear', action='store_true')
    g.add_argument('--include-1x1', action='store_true')
    g.add_argument('--include-grouped', action='store_true',
                   help='also decompose depthwise convs (MobileNetV2)')
    g.add_argument('--include-classifier', action='store_true')
    g.add_argument('--min-params', type=int, default=512)
    g.add_argument('--skip-names', nargs='*', default=[])
    return p


def rule_from_args(args) -> LayerSelectionRule:
    return LayerSelectionRule(
        include_linear=args.include_linear,
        include_1x1=args.include_1x1,
        include_grouped=args.include_grouped,
        include_classifier=args.include_classifier,
        min_params=args.min_params,
        skip_names=tuple(args.skip_names),
    )


def setup(args, logname: str = 'rene'):
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)
    log = get_logger(logname, os.path.join(args.out, f'{logname}.log')).info
    log(f'device={device} | args={vars(args)}')
    return device, log


def get_data(args):
    cfg = DataConfig(dataset=args.dataset, root=args.data_root,
                     batch_size=args.batch_size, search_batch_size=args.search_batch_size,
                     workers=args.workers, val_fraction=args.val_fraction,
                     seed=args.seed, image_size=args.image_size)
    return build_loaders(cfg)


def get_model(args, num_classes: int):
    model = build_model(args.arch, args.dataset, pretrained=args.pretrained,
                        num_classes=num_classes)
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint)
    return model


def in_size(args):
    return input_size_for(args.dataset, args.image_size)
