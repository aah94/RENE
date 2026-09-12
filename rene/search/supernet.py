"""Turn a pre-trained network into a searchable "mixed rank" supernet."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ..decomposition.layers import max_rank_of
from .mixed import MixedRankLayer, make_mixed


@dataclass
class LayerSelectionRule:
    """Which layers of the network get decomposed.

    Defaults target the block convolutions: the 1x1 downsample shortcuts, the
    classifier and (via ``min_params``) the small CIFAR stem are left dense.
    This gives 18 searched layers on ResNet-20 and 13 on VGG-16. Note that the
    ImageNet stem (7x7, 3->64) is well above ``min_params`` and *is* selected, so
    ResNet-18 yields 17 layers. ``skip_names`` matches on substrings, so excluding
    the stem needs a rule that does not also match ``layer1.0.conv1``.
    """
    include_conv: bool = True
    include_linear: bool = False
    include_1x1: bool = False
    include_grouped: bool = False          # depthwise convs are already cheap
    include_classifier: bool = False
    min_params: int = 512                  # skip trivially small layers
    skip_names: Sequence[str] = field(default_factory=tuple)

    def accepts(self, name: str, module: nn.Module, is_last_linear: bool) -> bool:
        if any(s in name for s in self.skip_names):
            return False
        if isinstance(module, nn.Conv2d):
            if not self.include_conv:
                return False
            if module.groups != 1 and not self.include_grouped:
                return False
            if max(module.kernel_size) == 1 and not self.include_1x1:
                return False
            return module.weight.numel() >= self.min_params
        if isinstance(module, nn.Linear):
            if not self.include_linear:
                return False
            if is_last_linear and not self.include_classifier:
                return False
            return module.weight.numel() >= self.min_params
        return False


def select_layers(model: nn.Module, rule: LayerSelectionRule) -> List[str]:
    """Names of the modules that will be decomposed, in forward order."""
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    last_linear = linear_names[-1] if linear_names else None
    out = []
    for name, module in model.named_modules():
        if rule.accepts(name, module, is_last_linear=(name == last_linear)):
            out.append(name)
    return out


def get_module(model: nn.Module, name: str) -> nn.Module:
    mod = model
    for part in name.split('.'):
        mod = mod[int(part)] if part.isdigit() and isinstance(mod, (nn.Sequential, nn.ModuleList)) \
            else getattr(mod, part)
    return mod


def set_module(model: nn.Module, name: str, new: nn.Module) -> None:
    parts = name.split('.')
    parent = get_module(model, '.'.join(parts[:-1])) if len(parts) > 1 else model
    leaf = parts[-1]
    if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(leaf)] = new
    else:
        setattr(parent, leaf, new)


def arange_candidates(lo: int, hi: int, step: int) -> List[int]:
    """``{ r | r = lo + k*s, k in N, lo <= r <= hi }`` (Section 4.3)."""
    step = max(1, int(step))
    lo, hi = int(lo), int(hi)
    if hi < lo:
        lo, hi = hi, lo
    return list(range(lo, hi + 1, step))


class RankSupernet(nn.Module):
    """A frozen backbone whose target layers are replaced by mixed-rank layers."""

    def __init__(self, model: nn.Module, decomp: str,
                 rule: Optional[LayerSelectionRule] = None,
                 layer_names: Optional[Sequence[str]] = None,
                 candidates: Optional[Dict[str, Sequence[int]]] = None,
                 als_iters: int = 50):
        super().__init__()
        self.decomp = decomp
        self.rule = rule or LayerSelectionRule()
        self.model = model
        self.layer_names: List[str] = list(layer_names) if layer_names is not None \
            else select_layers(model, self.rule)

        # everything that is *not* being searched stays frozen
        for prm in self.model.parameters():
            prm.requires_grad_(False)

        self.max_ranks: Dict[str, int] = {
            n: max_rank_of(get_module(model, n), decomp) for n in self.layer_names}

        candidates = candidates or {}
        self.mixed: Dict[str, MixedRankLayer] = {}
        for name in self.layer_names:
            base = get_module(self.model, name)
            cand = candidates.get(name, [self.max_ranks[name]])
            layer = make_mixed(base, cand, decomp, als_iters=als_iters)
            set_module(self.model, name, layer)
            self.mixed[name] = layer

    # -- convenience ---------------------------------------------------------
    @property
    def mixed_layers(self) -> List[MixedRankLayer]:
        return [self.mixed[n] for n in self.layer_names]

    def forward(self, x):
        return self.model(x)

    def reset_candidates(self, candidates: Dict[str, Sequence[int]]):
        for name, cand in candidates.items():
            self.mixed[name].reset_candidates(cand)
        return self

    def weight_parameters(self):
        for layer in self.mixed_layers:
            yield from layer.weight_parameters()

    def alpha_parameters(self):
        for layer in self.mixed_layers:
            yield from layer.alpha_parameters()

    @torch.no_grad()
    def selected_ranks(self) -> Dict[str, int]:
        return {n: self.mixed[n].best_rank() for n in self.layer_names}

    @torch.no_grad()
    def rank_coefficients(self) -> Dict[str, Dict[int, float]]:
        return {n: dict(zip(self.mixed[n].candidates,
                            self.mixed[n].p.detach().cpu().tolist()))
                for n in self.layer_names}
