#!/usr/bin/env python
"""Table 3 -- double compression: knowledge distillation followed by RENE.

CIFAR-100 : teacher ResNet-56  -> student ResNet-20
ImageNet-1K: teacher ResNet-34 -> student ResNet-18

Produces three rows: ``Distillation`` (KD only), ``RENE(Teacher)`` and
``RENE(Student)`` -- both compressed with RENE, the student additionally
distilled first. Compression figures are reported against the *teacher*.

    python scripts/exp_double_compression.py --dataset cifar100 \
        --teacher-arch resnet56 --student-arch resnet20 \
        --teacher-ckpt runs/pretrain/resnet56_cifar100.pth \
        --out runs/table3_c100
"""

import argparse
import copy
import os

from _common import (add_common_args, add_layer_rule_args, get_data, in_size,
                     rule_from_args, setup)

from rene.finetune import (FinetuneConfig, decompose_model, decomposed_layer_names,
                           finetune_decomposed, train_kd)
from rene.models import build_model, load_checkpoint
from rene.search import RENESearch, SearchConfig
from rene.utils import (compression_summary, count_flops, count_params, evaluate,
                        save_json)


def compress(model, tag, args, rule, val_loader, train_loader, test_loader, device, log):
    """Run the full RENE pipeline on one model, return (compressed_model, top1)."""
    cfg = SearchConfig(decomp=args.decomp, rank_min=args.rank_min, rank_max=args.rank_max,
                       step=args.step, factor=args.factor, iters=args.iters,
                       warmup_iters=args.warmup_iters, lr_w=args.lr_w,
                       gamma=args.gamma, beta=args.beta, seed=args.seed)
    searcher = RENESearch(model, cfg, rule=rule, device=str(device), logger=log)
    ranks = searcher.run(val_loader)
    log(f'[{tag}] ranks: {ranks}')

    teacher = copy.deepcopy(model)
    student = decompose_model(teacher, ranks, args.decomp, als_iters=100)
    ft = FinetuneConfig(epochs=args.epochs, lr=args.lr, lam=args.lam)
    res = finetune_decomposed(student, teacher, list(ranks.keys()), train_loader,
                              test_loader, ft, device, log,
                              os.path.join(args.out, f'{tag}.pth'))
    return student, res['top1'], ranks


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    add_layer_rule_args(p)
    p.add_argument('--teacher-arch', default='resnet56')
    p.add_argument('--student-arch', default='resnet20')
    p.add_argument('--teacher-ckpt', default=None)
    p.add_argument('--student-ckpt', default=None, help='skip KD if the distilled student exists')
    p.add_argument('--decomp', default='tt', choices=['tt', 'cp', 'cp4'])
    p.add_argument('--rank-min', type=int, default=10)
    p.add_argument('--rank-max', type=int, default=100)
    p.add_argument('--step', type=int, default=10)
    p.add_argument('--factor', type=int, default=10)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--warmup-iters', type=int, default=20)
    p.add_argument('--lr-w', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.4)
    p.add_argument('--beta', type=float, default=0.8)
    p.add_argument('--kd-epochs', type=int, default=100)
    p.add_argument('--kd-lr', type=float, default=0.05)
    p.add_argument('--epochs', type=int, default=50, help='RENE fine-tuning epochs')
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--lam', type=float, default=0.5)
    args = p.parse_args()

    device, log = setup(args, 'table3')
    train_loader, val_loader, test_loader, n_cls = get_data(args)
    rule = rule_from_args(args)
    isz = in_size(args)

    # --- teacher -------------------------------------------------------------
    teacher = build_model(args.teacher_arch, args.dataset,
                          pretrained=args.pretrained, num_classes=n_cls)
    if args.teacher_ckpt:
        teacher = load_checkpoint(teacher, args.teacher_ckpt)
    t_params, t_flops = count_params(teacher), count_flops(teacher, isz, 'cpu')
    t_top1 = evaluate(teacher.to(device), test_loader, device)['top1']
    log(f'teacher {args.teacher_arch}: top1={t_top1:.2f} '
        f'params={t_params / 1e6:.2f}M flops={t_flops / 1e6:.1f}M')

    # --- student: distillation (the ``Distillation`` row) --------------------
    student = build_model(args.student_arch, args.dataset, num_classes=n_cls)
    if args.student_ckpt:
        student = load_checkpoint(student, args.student_ckpt)
        s_top1 = evaluate(student.to(device), test_loader, device)['top1']
    else:
        s_top1 = train_kd(student, teacher, train_loader, test_loader, device,
                          epochs=args.kd_epochs, lr=args.kd_lr, log=log,
                          ckpt_path=os.path.join(args.out, 'kd_student.pth'))['top1']
    s_params, s_flops = count_params(student), count_flops(student, isz, 'cpu')
    log(f'distilled student {args.student_arch}: top1={s_top1:.2f}')

    rows = [{'method': 'Distillation', 'top1': s_top1,
             'flops_reduction_pct': 100 * (1 - s_flops / t_flops),
             'compression_rate': 100 * (1 - s_params / t_params)}]

    # --- RENE(Teacher) and RENE(Student), both measured against the teacher ---
    for tag, model in [('RENE(Teacher)', teacher.cpu()), ('RENE(Student)', student.cpu())]:
        comp, top1, ranks = compress(model, tag.replace('(', '_').replace(')', ''),
                                     args, rule, val_loader, train_loader,
                                     test_loader, device, log)
        c_params, c_flops = count_params(comp), count_flops(comp, isz, 'cpu')
        rows.append({'method': tag, 'top1': top1,
                     'flops_reduction_pct': 100 * (1 - c_flops / t_flops),
                     'compression_rate': 100 * (1 - c_params / t_params),
                     'ranks': ranks})
        log(f"[{tag}] top1={top1:.2f} FLOPs down {rows[-1]['flops_reduction_pct']:.2f}% "
            f"Comp. Rate {rows[-1]['compression_rate']:.2f}%")

    save_json({'teacher': {'arch': args.teacher_arch, 'top1': t_top1},
               'student': {'arch': args.student_arch, 'top1': s_top1},
               'rows': rows}, os.path.join(args.out, 'table3.json'))
    log('\n'.join(f"{r['method']:>14s} | Top-1 {r['top1']:6.2f} | "
                  f"FLOPs {r['flops_reduction_pct']:6.2f}% | "
                  f"Comp {r['compression_rate']:6.2f}%" for r in rows))


if __name__ == '__main__':
    main()
