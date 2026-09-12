"""Parameter, FLOPs and compression-rate metrics (the columns of Tables 1-3).

FLOPs are counted as multiply-accumulate operations of the conv / linear layers,
the convention used by the compression literature the paper compares against.
Because every factorised layer is built out of plain ``nn.Conv2d`` / ``nn.Linear``
submodules, the same hook-based counter works unchanged on decomposed models.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    ps = model.parameters()
    return sum(p.numel() for p in ps if (p.requires_grad or not trainable_only))


@torch.no_grad()
def count_flops(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 32, 32),
                device: Optional[torch.device] = None) -> int:
    """MACs of all Conv2d / Linear layers for one forward pass."""
    device = device or next(model.parameters()).device
    total = {'flops': 0}
    handles = []

    def conv_hook(module, inp, out):
        out_elems = out.numel() // out.shape[0]                    # per sample
        kernel = module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
        total['flops'] += out_elems * kernel

    def linear_hook(module, inp, out):
        total['flops'] += module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    model(torch.zeros(*input_size, device=device))
    for h in handles:
        h.remove()
    model.train(was_training)
    return total['flops']


def compression_summary(original: nn.Module, compressed: nn.Module,
                        input_size: Tuple[int, ...] = (1, 3, 32, 32),
                        device: Optional[torch.device] = None) -> Dict[str, float]:
    """Reproduce the ``FLOPs (down %)`` and ``Comp. Rate`` columns of the paper."""
    p0, p1 = count_params(original), count_params(compressed)
    f0 = count_flops(original, input_size, device)
    f1 = count_flops(compressed, input_size, device)
    return {
        'params_original': p0,
        'params_compressed': p1,
        'params_reduction_pct': 100.0 * (1 - p1 / p0),
        'compression_rate': 100.0 * (1 - p1 / p0),      # paper's "Comp. Rate"
        'flops_original': f0,
        'flops_compressed': f1,
        'flops_reduction_pct': 100.0 * (1 - f1 / f0),
        'params_ratio': p1 / p0,
        'flops_ratio': f1 / f0,
    }


def format_summary(s: Dict[str, float]) -> str:
    return (f"params {s['params_original']/1e6:.3f}M -> {s['params_compressed']/1e6:.3f}M "
            f"(Comp. Rate {s['compression_rate']:.2f}%)  |  "
            f"FLOPs {s['flops_original']/1e6:.1f}M -> {s['flops_compressed']/1e6:.1f}M "
            f"(down {s['flops_reduction_pct']:.2f}%)")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, topk: Tuple[int, ...] = (1,)) -> Dict[str, float]:
    """Top-k accuracy on ``loader`` (the TOP-1 column of the tables)."""
    model.eval()
    correct = {k: 0 for k in topk}
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        maxk = max(topk)
        _, pred = logits.topk(min(maxk, logits.shape[1]), 1, True, True)
        hit = pred.eq(y.view(-1, 1).expand_as(pred))
        for k in topk:
            correct[k] += hit[:, :k].any(dim=1).sum().item()
        n += y.numel()
    return {f'top{k}': 100.0 * correct[k] / max(n, 1) for k in topk}
