#!/usr/bin/env python
"""Step 2 -- decompose at the searched ranks and fine-tune (Eq. 10-11).

Example
-------
    python scripts/finetune.py --arch resnet20 --dataset cifar10 \
        --checkpoint runs/pretrain/resnet20_cifar10.pth \
        --ranks runs/search_r20_cp/ranks.json \
        --epochs 100 --lr 1e-5 --lam 0.5 --out runs/ft_r20_cp
"""

import argparse
import json
import os

from _common import add_common_args, get_data, get_model, in_size, setup

from rene.finetune import FinetuneConfig, decompose_model, finetune_decomposed
from rene.utils import (compression_summary, evaluate, format_summary, load_json,
                        save_json)


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--ranks', required=True, help='ranks.json produced by search.py')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--lam', type=float, default=0.5, help='lambda of Eq. (11)')
    p.add_argument('--diss-reduction', default='mean', choices=['mean', 'sum'],
                   help="'sum' is literal Eq. (10); 'mean' keeps L_diss on the scale of L_ce")
    p.add_argument('--no-weight-term', action='store_true')
    p.add_argument('--no-output-term', action='store_true')
    p.add_argument('--kd-logits', action='store_true')
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--scheduler', default='cosine', choices=['cosine', 'step', 'none'])
    p.add_argument('--als-iters', type=int, default=100)
    args = p.parse_args()

    device, log = setup(args, 'finetune')
    train_loader, _, test_loader, n_cls = get_data(args)

    spec = load_json(args.ranks)
    ranks, decomp = spec['ranks'], spec['decomp']
    log(f'{len(ranks)} layers, decomposition={decomp}')

    teacher = get_model(args, n_cls)
    base_top1 = evaluate(teacher.to(device), test_loader, device)['top1']
    log(f'original top1 = {base_top1:.2f}')

    student = decompose_model(teacher.cpu(), ranks, decomp, als_iters=args.als_iters, verbose=True)
    summary = compression_summary(teacher, student, in_size(args), device='cpu')
    log(format_summary(summary))
    pre_ft = evaluate(student.to(device), test_loader, device)['top1']
    log(f'top1 before fine-tuning = {pre_ft:.2f}')

    cfg = FinetuneConfig(epochs=args.epochs, lr=args.lr, lam=args.lam,
                         diss_reduction=args.diss_reduction,
                         use_weight_term=not args.no_weight_term,
                         use_output_term=not args.no_output_term,
                         kd_logits=args.kd_logits, weight_decay=args.weight_decay,
                         scheduler=args.scheduler)
    ckpt = os.path.join(args.out, 'compressed.pth')
    res = finetune_decomposed(student, teacher, list(ranks.keys()), train_loader,
                              test_loader, cfg, device, log, ckpt)

    row = {'arch': args.arch, 'dataset': args.dataset, 'decomp': decomp,
           'top1_original': base_top1, 'top1_before_finetune': pre_ft,
           'top1': res['top1'], 'flops_reduction_pct': summary['flops_reduction_pct'],
           'compression_rate': summary['compression_rate'], 'ranks': ranks}
    save_json(row, os.path.join(args.out, 'result.json'))
    log(f"RESULT  Top-1 {row['top1']:.2f} (orig {base_top1:.2f})  "
        f"FLOPs down {row['flops_reduction_pct']:.2f}%  "
        f"Comp. Rate {row['compression_rate']:.2f}%")


if __name__ == '__main__':
    main()
