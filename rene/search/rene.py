"""Algorithm 1 -- Rank adapt tENsor dEcomposition (RENE).

Multi-step rank search (Section 4.3). Each outer step optimises the weights
(Eq. 8) and the rank logits (Eq. 9) over the current candidate set, picks
``rbar_i = argmax_r softmax(alpha_i)``, then contracts the search space around
``rbar_i`` and refines the sampling resolution::

    Lb_i <- rbar_i - s/2 ,  Ub_i <- rbar_i + s/2 ,  s <- floor(s / f)

The bounds use the *current* step size and the step is divided afterwards, which
reproduces the toy example of Figure 3: ``[100..800] step 100 -> rbar=200 ->
[150, 250] step 10 -> rbar=240 -> [235, 245] step 1``.

With the paper's settings this gives 2 search steps on CIFAR-10/100
(``{10..100}``, ``s=10``, ``f=10``) and 3 on ImageNet-1K (``{50..850}``,
``s=100``, ``f=10``).
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import total_alpha_loss, total_weight_loss
from .supernet import (LayerSelectionRule, RankSupernet, arange_candidates,
                       get_module, select_layers)


@dataclass
class SearchConfig:
    """Hyper-parameters of the search stage (Section 5.1 defaults)."""
    decomp: str = 'tt'                 # 'tt' | 'cp' | 'cp4'
    rank_min: int = 10                 # initial lower bound  (CIFAR: 10, ImageNet: 50)
    rank_max: int = 100                # initial upper bound  (CIFAR: 100, ImageNet: 850)
    step: int = 10                     # initial step size s  (CIFAR: 10, ImageNet: 100)
    factor: int = 10                   # refinement factor f > 1
    iters: int = 100                   # T: joint updates per search step
    warmup_iters: int = 20             # weight-only updates first (avoids collapse)
    lr_w: float = 1e-3                 # eta_w  (CIFAR: 1e-3, ImageNet: 1e-4)
    lr_alpha: float = 1e-3             # eta_alpha
    momentum: float = 0.9              # SGD with Nesterov momentum
    nesterov: bool = True
    gamma: float = 0.4                 # Eq. (5)
    beta: float = 0.8                  # Eq. (5)
    alpha_objective: str = 'val_ce'    # 'val_ce' (Eq. 7) | 'recon' (data-free)
    als_iters: int = 30                # CP-ALS iterations when (re)building branches
    weight_steps_per_iter: int = 1
    grad_clip: float = 0.0
    seed: int = 0
    log_every: int = 20


class RENESearch:
    """Runs Algorithm 1 on a pre-trained model."""

    def __init__(self, model: nn.Module, cfg: SearchConfig,
                 rule: Optional[LayerSelectionRule] = None,
                 device: str = 'cpu', logger: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.log = logger or print
        self.rule = rule or LayerSelectionRule()

        self.model = copy.deepcopy(model).to(self.device).eval()
        self.layer_names = select_layers(self.model, self.rule)
        if not self.layer_names:
            raise RuntimeError('layer selection rule matched no layer')

        # initial search space, identical for every layer, clipped per layer by
        # the mixed layer itself (a rank above min(C_out, C_in) is meaningless)
        init = arange_candidates(cfg.rank_min, cfg.rank_max, cfg.step)
        self.supernet = RankSupernet(
            self.model, cfg.decomp, rule=self.rule, layer_names=self.layer_names,
            candidates={n: init for n in self.layer_names}, als_iters=cfg.als_iters,
        ).to(self.device)

        self.history: List[Dict] = []
        self.search_seconds: float = 0.0

    # ------------------------------------------------------------------ #
    # optimisers (Eq. 8 / Eq. 9)
    # ------------------------------------------------------------------ #
    def _make_optimizers(self):
        cfg = self.cfg
        opt_w = torch.optim.SGD(list(self.supernet.weight_parameters()), lr=cfg.lr_w,
                                momentum=cfg.momentum, nesterov=cfg.nesterov)
        opt_a = torch.optim.SGD(list(self.supernet.alpha_parameters()), lr=cfg.lr_alpha,
                                momentum=cfg.momentum, nesterov=cfg.nesterov)
        return opt_w, opt_a

    # ------------------------------------------------------------------ #
    # inner optimisation of one search step
    # ------------------------------------------------------------------ #
    def _optimise(self, val_loader, step_idx: int):
        cfg = self.cfg
        layers = self.supernet.mixed_layers
        opt_w, opt_a = self._make_optimizers()
        val_iter = iter(val_loader) if val_loader is not None else None

        def next_batch():
            nonlocal val_iter
            try:
                return next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                return next(val_iter)

        total_iters = cfg.warmup_iters + cfg.iters
        for t in range(total_iters):
            # ---- weight update, Eq. (8): data-free ------------------------
            for _ in range(cfg.weight_steps_per_iter):
                opt_w.zero_grad(set_to_none=True)
                loss_w, ld, lr_ = total_weight_loss(layers, cfg.gamma, cfg.beta)
                loss_w.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.supernet.weight_parameters()), cfg.grad_clip)
                opt_w.step()

            # ---- rank-coefficient update, Eq. (9) -------------------------
            #  only after the warm-up, "to prevent convergence collapse"
            la = torch.tensor(float('nan'))
            if t >= cfg.warmup_iters:
                opt_a.zero_grad(set_to_none=True)
                if cfg.alpha_objective == 'val_ce' and val_iter is not None:
                    x, y = next_batch()
                    x, y = x.to(self.device), y.to(self.device)
                    obj = F.cross_entropy(self.supernet(x), y)
                else:  # data-free variant: reuse the decomposition error
                    obj = sum(l.decomposition_loss() for l in layers)
                loss_a, la, _ = total_alpha_loss(obj, layers, cfg.gamma, cfg.beta)
                loss_a.backward()
                opt_a.step()

            if cfg.log_every and (t + 1) % cfg.log_every == 0:
                self.log(f'  [step {step_idx}] it {t + 1:4d}/{total_iters} '
                         f'L_d={ld.item():.4e}  L_r={lr_.item():.4f}  '
                         f'L_val={la.item():.4f}  '
                         f'mean_rank={self._mean_expected_rank():.1f}')

    def _mean_expected_rank(self) -> float:
        vals = [l.expected_rank() for l in self.supernet.mixed_layers]
        return sum(vals) / len(vals)

    # ------------------------------------------------------------------ #
    # Algorithm 1
    # ------------------------------------------------------------------ #
    def run(self, val_loader=None) -> Dict[str, int]:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        if cfg.factor <= 1:
            raise ValueError('factor f must be > 1')

        s = int(cfg.step)
        bounds = {n: (cfg.rank_min, cfg.rank_max) for n in self.layer_names}
        selected: Dict[str, int] = {}
        t0 = time.time()
        step_idx = 0

        while s >= 1:
            spaces = {n: self.supernet.mixed[n].candidates for n in self.layer_names}
            sizes = sorted({len(v) for v in spaces.values()})
            self.log(f'[RENE] search step {step_idx}: s={s} '
                     f'| candidates/layer={sizes} '
                     f'| example {self.layer_names[0]} -> {spaces[self.layer_names[0]]}')

            self._optimise(val_loader, step_idx)

            # rbar_i = argmax_r softmax(alpha_i)   (Algorithm 1, l.11)
            selected = self.supernet.selected_ranks()
            self.history.append({
                'step': step_idx, 's': s,
                'candidates': {n: list(spaces[n]) for n in self.layer_names},
                'selected': dict(selected),
                'coefficients': self.supernet.rank_coefficients(),
            })
            self.log(f'[RENE] step {step_idx} selected: '
                     f'{ {k.split(".")[-1] if len(k) > 12 else k: v for k, v in list(selected.items())[:8]} }'
                     f'{" ..." if len(selected) > 8 else ""}')

            s_next = s // cfg.factor
            if s_next < 1:
                break

            # contract the space around rbar_i using the *current* step (Fig. 3)
            half = s / 2.0
            new_cand = {}
            for n in self.layer_names:
                lb = int(math.floor(selected[n] - half))
                ub = int(math.ceil(selected[n] + half))
                lb = max(1, lb)
                bounds[n] = (lb, ub)
                new_cand[n] = arange_candidates(lb, ub, s_next)
            s = s_next
            self.supernet.reset_candidates(new_cand)   # reinitialise weights + alphas
            step_idx += 1

        self.search_seconds = time.time() - t0
        self.log(f'[RENE] search finished in {self.search_seconds:.1f}s '
                 f'after {step_idx + 1} step(s)')
        self.selected_ranks = selected
        return selected

    # ------------------------------------------------------------------ #
    def save(self, path: str, extra: Optional[Dict] = None):
        payload = {
            'config': asdict(self.cfg),
            'layer_names': self.layer_names,
            'max_ranks': self.supernet.max_ranks,
            'selected_ranks': getattr(self, 'selected_ranks', {}),
            'search_seconds': self.search_seconds,
            'history': self.history,
        }
        if extra:
            payload.update(extra)
        with open(path, 'w') as fh:
            json.dump(payload, fh, indent=2)
        self.log(f'[RENE] wrote {path}')
        return payload
