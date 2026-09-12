#!/usr/bin/env python
"""Step 1 -- RENE rank search (Algorithm 1).

Paper settings (Section 5.1)
----------------------------
CIFAR-10/100 : ranks {10..100},  s=10,  f=10, lr 1e-3  -> 2 search steps
ImageNet-1K  : ranks {50..850},  s=100, f=10, lr 1e-4  -> 3 search steps
gamma=0.4, beta=0.8, SGD + Nesterov momentum 0.9

Examples
--------
    python scripts/search.py --arch resnet20 --dataset cifar10 \
        --decomp cp --checkpoint runs/pretrain/resnet20_cifar10.pth \
        --out runs/search_r20_cp

    python scripts/search.py --arch resnet18 --dataset imagenet --pretrained \
        --decomp tt --rank-min 50 --rank-max 850 --step 100 --lr-w 1e-4 \
        --data-root /path/to/imagenet --out runs/search_r18_tt
"""

import argparse
import json
import os

from _common import (add_common_args, add_layer_rule_args, get_data, get_model,
                     in_size, rule_from_args, setup)

from rene.finetune import decompose_model
from rene.search import RENESearch, SearchConfig
from rene.utils import (compression_summary, count_params, format_summary,
                        save_json)


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    add_layer_rule_args(p)
    g = p.add_argument_group('search')
    g.add_argument('--decomp', default='tt', choices=['tt', 'cp', 'cp4'])
    g.add_argument('--rank-min', type=int, default=10)
    g.add_argument('--rank-max', type=int, default=100)
    g.add_argument('--step', type=int, default=10, help='initial step size s')
    g.add_argument('--factor', type=int, default=10, help='refinement factor f > 1')
    g.add_argument('--iters', type=int, default=100, help='T joint updates per step')
    g.add_argument('--warmup-iters', type=int, default=20)
    g.add_argument('--lr-w', type=float, default=1e-3)
    g.add_argument('--lr-alpha', type=float, default=1e-3)
    g.add_argument('--gamma', type=float, default=0.4)
    g.add_argument('--beta', type=float, default=0.8)
    g.add_argument('--alpha-objective', default='val_ce', choices=['val_ce', 'recon'])
    g.add_argument('--als-iters', type=int, default=30)
    args = p.parse_args()

    device, log = setup(args, 'search')
    _, val_loader, _, n_cls = get_data(args)
    model = get_model(args, n_cls)

    cfg = SearchConfig(
        decomp=args.decomp, rank_min=args.rank_min, rank_max=args.rank_max,
        step=args.step, factor=args.factor, iters=args.iters,
        warmup_iters=args.warmup_iters, lr_w=args.lr_w, lr_alpha=args.lr_alpha,
        gamma=args.gamma, beta=args.beta, alpha_objective=args.alpha_objective,
        als_iters=args.als_iters, seed=args.seed)

    searcher = RENESearch(model, cfg, rule=rule_from_args(args), device=str(device), logger=log)
    log(f'searching over {len(searcher.layer_names)} layers: {searcher.layer_names}')
    ranks = searcher.run(val_loader)

    # what the selected configuration actually costs
    compressed = decompose_model(model, ranks, args.decomp, als_iters=100)
    summary = compression_summary(model, compressed, in_size(args), device='cpu')
    log('selected ranks: ' + json.dumps(ranks, indent=2))
    log(format_summary(summary))

    searcher.save(os.path.join(args.out, 'search.json'),
                  extra={'summary': summary, 'arch': args.arch,
                         'dataset': args.dataset, 'decomp': args.decomp})
    save_json({'ranks': ranks, 'decomp': args.decomp, 'arch': args.arch,
               'dataset': args.dataset, 'summary': summary,
               'search_seconds': searcher.search_seconds},
              os.path.join(args.out, 'ranks.json'))
    log(f'-> {os.path.join(args.out, "ranks.json")}')


if __name__ == '__main__':
    main()
