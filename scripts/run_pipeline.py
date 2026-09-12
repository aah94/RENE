#!/usr/bin/env python
"""End-to-end RENE: search -> decompose -> fine-tune, reproducing one table row.

Tables 1 and 2 of the paper:

    # ResNet-20 / CIFAR-10, CP        (Table 1)
    python scripts/run_pipeline.py --arch resnet20 --dataset cifar10 --decomp cp \
        --checkpoint runs/pretrain/resnet20_cifar10.pth --out runs/t1_r20_cp

    # VGG-16 / CIFAR-10, TT           (Table 1)
    python scripts/run_pipeline.py --arch vgg16 --dataset cifar10 --decomp tt \
        --checkpoint runs/pretrain/vgg16_cifar10.pth --out runs/t1_vgg_tt

    # ResNet-18 / ImageNet-1K, TT     (Table 2)
    python scripts/run_pipeline.py --arch resnet18 --dataset imagenet --pretrained \
        --decomp tt --rank-min 50 --rank-max 850 --step 100 --lr-w 1e-4 \
        --data-root /path/to/imagenet --out runs/t2_r18_tt
"""

import argparse
import json
import os

from _common import (add_common_args, add_layer_rule_args, get_data, get_model,
                     in_size, rule_from_args, setup)

from rene.finetune import (FinetuneConfig, decompose_model, finetune_decomposed)
from rene.search import RENESearch, SearchConfig
from rene.utils import (compression_summary, evaluate, format_summary, save_json)


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    add_layer_rule_args(p)
    g = p.add_argument_group('search')
    g.add_argument('--decomp', default='tt', choices=['tt', 'cp', 'cp4'])
    g.add_argument('--rank-min', type=int, default=10)
    g.add_argument('--rank-max', type=int, default=100)
    g.add_argument('--step', type=int, default=10)
    g.add_argument('--factor', type=int, default=10)
    g.add_argument('--iters', type=int, default=100)
    g.add_argument('--warmup-iters', type=int, default=20)
    g.add_argument('--lr-w', type=float, default=1e-3)
    g.add_argument('--lr-alpha', type=float, default=1e-3)
    g.add_argument('--gamma', type=float, default=0.4)
    g.add_argument('--beta', type=float, default=0.8)
    g.add_argument('--alpha-objective', default='val_ce', choices=['val_ce', 'recon'])
    f = p.add_argument_group('fine-tune')
    f.add_argument('--epochs', type=int, default=100)
    f.add_argument('--lr', type=float, default=1e-5)
    f.add_argument('--lam', type=float, default=0.5)
    f.add_argument('--diss-reduction', default='mean', choices=['mean', 'sum'])
    f.add_argument('--skip-finetune', action='store_true')
    args = p.parse_args()

    device, log = setup(args, 'pipeline')
    train_loader, val_loader, test_loader, n_cls = get_data(args)
    model = get_model(args, n_cls)
    base_top1 = evaluate(model.to(device), test_loader, device)['top1']
    log(f'original top1 = {base_top1:.2f}')

    # ---- step 1: rank search ------------------------------------------------
    cfg = SearchConfig(decomp=args.decomp, rank_min=args.rank_min, rank_max=args.rank_max,
                       step=args.step, factor=args.factor, iters=args.iters,
                       warmup_iters=args.warmup_iters, lr_w=args.lr_w,
                       lr_alpha=args.lr_alpha, gamma=args.gamma, beta=args.beta,
                       alpha_objective=args.alpha_objective, seed=args.seed)
    searcher = RENESearch(model.cpu(), cfg, rule=rule_from_args(args),
                          device=str(device), logger=log)
    ranks = searcher.run(val_loader)
    searcher.save(os.path.join(args.out, 'search.json'))
    log('ranks: ' + json.dumps(ranks))

    # ---- step 2: final decomposition ---------------------------------------
    teacher = get_model(args, n_cls)
    student = decompose_model(teacher, ranks, args.decomp, als_iters=100, verbose=True)
    summary = compression_summary(teacher, student, in_size(args), device='cpu')
    log(format_summary(summary))
    pre_ft = evaluate(student.to(device), test_loader, device)['top1']
    log(f'top1 before fine-tuning = {pre_ft:.2f}')

    # ---- step 3: fine-tuning ------------------------------------------------
    top1 = pre_ft
    if not args.skip_finetune:
        ft = FinetuneConfig(epochs=args.epochs, lr=args.lr, lam=args.lam,
                            diss_reduction=args.diss_reduction)
        res = finetune_decomposed(student, teacher, list(ranks.keys()), train_loader,
                                  test_loader, ft, device, log,
                                  os.path.join(args.out, 'compressed.pth'))
        top1 = res['top1']

    row = {'method': f'RENE({args.decomp.upper()})', 'arch': args.arch,
           'dataset': args.dataset, 'top1_original': base_top1,
           'top1_before_finetune': pre_ft, 'top1': top1,
           'flops_reduction_pct': summary['flops_reduction_pct'],
           'compression_rate': summary['compression_rate'],
           'search_seconds': searcher.search_seconds, 'ranks': ranks}
    save_json(row, os.path.join(args.out, 'result.json'))
    log(f"RESULT | {row['method']:12s} | Top-1 {top1:6.2f} (orig {base_top1:.2f}) "
        f"| FLOPs down {summary['flops_reduction_pct']:6.2f}% "
        f"| Comp. Rate {summary['compression_rate']:6.2f}% "
        f"| search {searcher.search_seconds:.0f}s")


if __name__ == '__main__':
    main()
