#!/usr/bin/env python
"""Figure 4 -- automatic vs. manual rank selection.

Fixes one rank across all layers so the decomposed model keeps
{1, 5, 10, 25, 50, 75}% of the original parameters, fine-tunes each for 20
epochs, and compares against the ranks found by RENE.

    python scripts/exp_manual_vs_auto.py --arch vgg16 --dataset cifar10 \
        --checkpoint runs/pretrain/vgg16_cifar10.pth \
        --ranks runs/search_vgg_tt/ranks.json --out runs/fig4_vgg_c10
"""

import argparse
import os

from _common import (add_common_args, add_layer_rule_args, get_data, get_model,
                     in_size, rule_from_args, setup)

from rene.finetune import (FinetuneConfig, decompose_model, decomposed_layer_names,
                           finetune_decomposed, ranks_for_param_budget)
from rene.utils import compression_summary, evaluate, load_json, save_json


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    add_layer_rule_args(p)
    p.add_argument('--decomp', default='tt', choices=['tt', 'cp', 'cp4'])
    p.add_argument('--budgets', type=float, nargs='*', default=[1, 5, 10, 25, 50, 75],
                   help='percent of the original parameter count')
    p.add_argument('--ranks', default=None, help='ranks.json from RENE, for the comparison point')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--lam', type=float, default=0.5)
    p.add_argument('--budget-mode', default='global', choices=['global', 'ratio'])
    args = p.parse_args()

    device, log = setup(args, 'fig4')
    train_loader, _, test_loader, n_cls = get_data(args)
    teacher = get_model(args, n_cls)
    base_top1 = evaluate(teacher.to(device), test_loader, device)['top1']
    log(f'uncompressed top1 = {base_top1:.2f}')

    rule = rule_from_args(args)
    names = decomposed_layer_names(teacher.cpu(), rule)
    ft = FinetuneConfig(epochs=args.epochs, lr=args.lr, lam=args.lam)
    rows = []

    for pct in args.budgets:
        ranks = ranks_for_param_budget(teacher, pct / 100.0, args.decomp, names,
                                       mode=args.budget_mode)
        student = decompose_model(teacher, ranks, args.decomp)
        s = compression_summary(teacher, student, in_size(args), device='cpu')
        res = finetune_decomposed(student, teacher, names, train_loader, test_loader,
                                  ft, device, log)
        rows.append({'setting': 'manual', 'target_pct': pct,
                     'params_pct': 100 * s['params_ratio'], 'top1': res['top1'],
                     'flops_reduction_pct': s['flops_reduction_pct'],
                     'rank': sorted(set(ranks.values()))})
        log(f"[manual {pct:>5.1f}%] params {100 * s['params_ratio']:.2f}% "
            f"top1 {res['top1']:.2f}")

    if args.ranks:
        spec = load_json(args.ranks)
        student = decompose_model(teacher, spec['ranks'], spec['decomp'])
        s = compression_summary(teacher, student, in_size(args), device='cpu')
        res = finetune_decomposed(student, teacher, list(spec['ranks'].keys()),
                                  train_loader, test_loader, ft, device, log)
        rows.append({'setting': 'RENE', 'target_pct': None,
                     'params_pct': 100 * s['params_ratio'], 'top1': res['top1'],
                     'flops_reduction_pct': s['flops_reduction_pct'],
                     'rank': spec['ranks']})
        log(f"[RENE] params {100 * s['params_ratio']:.2f}% top1 {res['top1']:.2f}")

    save_json({'uncompressed_top1': base_top1, 'arch': args.arch,
               'dataset': args.dataset, 'decomp': args.decomp, 'rows': rows},
              os.path.join(args.out, 'fig4.json'))
    log(f"-> {os.path.join(args.out, 'fig4.json')}")


if __name__ == '__main__':
    main()
