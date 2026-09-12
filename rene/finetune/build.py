"""Build the final decomposed network from a rank configuration (Section 4.4).

Given the optimal ranks ``R* = {r*_1, ..., r*_n}`` produced by Algorithm 1, this
replaces each selected dense layer by its factorised counterpart, initialised
from the pre-trained weights with TT-SVD / CP-ALS.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from ..decomposition.layers import build_factorized, max_rank_of
from ..search.supernet import (LayerSelectionRule, get_module, select_layers,
                               set_module)


def decompose_model(model: nn.Module, ranks: Dict[str, int], decomp: str,
                    als_iters: int = 100, inplace: bool = False,
                    verbose: bool = False) -> nn.Module:
    """Replace ``model``'s layers named in ``ranks`` with rank-``r*`` factorisations."""
    net = model if inplace else copy.deepcopy(model)
    net = net.cpu()                       # TT-SVD / CP-ALS run on CPU
    for name, rank in ranks.items():
        base = get_module(net, name)
        kw = {'als_iters': als_iters} if decomp.startswith('cp') else {}
        new = build_factorized(base, int(rank), decomp, **kw)
        set_module(net, name, new)
        if verbose:
            print(f'  {name}: rank {rank} '
                  f'({base.weight.numel()} -> {new.n_params()} params)')
    return net


def uniform_ranks(model: nn.Module, rank: int, decomp: str,
                  layer_names: Sequence[str]) -> Dict[str, int]:
    """The same rank for every layer, clipped to each layer's feasible maximum."""
    return {n: max(1, min(int(rank), max_rank_of(get_module(model, n), decomp)))
            for n in layer_names}


def ratio_ranks(model: nn.Module, ratio: float, decomp: str,
                layer_names: Sequence[str]) -> Dict[str, int]:
    """Per-layer rank set to ``ratio * max_rank_i``."""
    return {n: max(1, int(round(ratio * max_rank_of(get_module(model, n), decomp))))
            for n in layer_names}


def _param_count_with(model: nn.Module, ranks: Dict[str, int], decomp: str) -> int:
    """Parameter count of the decomposed model, without running any ALS/SVD."""
    from ..decomposition.cp import cp_conv_num_params, cp_linear_max_rank
    from ..decomposition.tt import tt_conv_num_params

    total = sum(p.numel() for p in model.parameters())
    for name, r in ranks.items():
        mod = get_module(model, name)
        dense = mod.weight.numel()
        if isinstance(mod, nn.Conv2d):
            r1 = max(1, min(r, mod.out_channels, mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1]))
            r2 = max(1, min(r, mod.in_channels, r1 * mod.kernel_size[0] * mod.kernel_size[1]))
            if decomp == 'tt':
                new = tt_conv_num_params(mod.out_channels, mod.in_channels,
                                         mod.kernel_size, r1, r2)
            elif decomp == 'cp':
                new = cp_conv_num_params(mod.out_channels, mod.in_channels,
                                         mod.kernel_size, r, 'cp3')
            else:
                new = cp_conv_num_params(mod.out_channels, mod.in_channels,
                                         mod.kernel_size, r, 'cp4')
        else:
            rr = max(1, min(r, mod.in_features, mod.out_features))
            new = rr * (mod.in_features + mod.out_features)
        total += new - dense
    return total


def ranks_for_param_budget(model: nn.Module, target_ratio: float, decomp: str,
                           layer_names: Sequence[str], mode: str = 'global',
                           lo: int = 1, hi: Optional[int] = None) -> Dict[str, int]:
    """Manual-rank baseline of Figure 4.

    Finds the single global rank (``mode='global'``) -- or the global fraction of
    each layer's maximum rank (``mode='ratio'``) -- whose decomposed model has
    ``target_ratio`` of the original parameter count.
    """
    dense_total = sum(p.numel() for p in model.parameters())
    hi = hi or max(max_rank_of(get_module(model, n), decomp) for n in layer_names)

    def build(x):
        return (uniform_ranks(model, int(round(x)), decomp, layer_names) if mode == 'global'
                else ratio_ranks(model, x, decomp, layer_names))

    def ratio_at(x):
        return _param_count_with(model, build(x), decomp) / dense_total

    lo_x, hi_x = (lo, hi) if mode == 'global' else (1e-4, 1.0)
    for _ in range(60):                                   # bisection
        mid = (lo_x + hi_x) / 2
        if ratio_at(mid) < target_ratio:
            lo_x = mid
        else:
            hi_x = mid
        if mode == 'global' and hi_x - lo_x <= 1:
            break
    return build(hi_x)


def decomposed_layer_names(model: nn.Module,
                           rule: Optional[LayerSelectionRule] = None) -> list:
    return select_layers(model, rule or LayerSelectionRule())
