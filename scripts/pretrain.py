#!/usr/bin/env python
"""Step 0 -- train the uncompressed baselines (the ``Original`` rows).

Examples
--------
    python scripts/pretrain.py --arch resnet20 --dataset cifar10  --epochs 200
    python scripts/pretrain.py --arch vgg16    --dataset cifar10  --epochs 200
    python scripts/pretrain.py --arch resnet56 --dataset cifar100 --epochs 200

ImageNet baselines come from torchvision (``--pretrained``); no training needed.
"""

import argparse
import os

from _common import add_common_args, get_data, get_model, in_size, setup

from rene.finetune import TrainConfig, train_model
from rene.utils import compression_summary, count_flops, count_params, evaluate


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--scheduler', default='cosine', choices=['cosine', 'step', 'none'])
    p.add_argument('--eval-only', action='store_true')
    args = p.parse_args()

    device, log = setup(args, 'pretrain')
    train_loader, _, test_loader, n_cls = get_data(args)
    model = get_model(args, n_cls).to(device)

    log(f'{args.arch}: {count_params(model) / 1e6:.3f}M params, '
        f'{count_flops(model, in_size(args), device) / 1e6:.1f}M FLOPs')

    if args.eval_only:
        log(f'top1 = {evaluate(model, test_loader, device)["top1"]:.2f}')
        return

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                      scheduler=args.scheduler)
    ckpt = os.path.join(args.out, f'{args.arch}_{args.dataset}.pth')
    res = train_model(model, train_loader, test_loader, cfg, device, log, ckpt)
    log(f'best top1 = {res["top1"]:.2f} -> {ckpt}')


if __name__ == '__main__':
    main()
