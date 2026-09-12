"""Fine-tuning of the decomposed network (Section 4.4, Eq. 10-11).

The original model is the teacher, the decomposed model the student::

    L_diss = sum_i ||W_i - W_hat^{(r*)}_i||_F^2                    (weight term)
           + sum_x sum_i ||O_i(x) - D_i(x)||_F^2                   (output term)

    L_f    = L_ce + lambda * L_diss

``O_i`` and ``D_i`` are the outputs of layer ``i`` in the original and the
decomposed model; they are captured with forward hooks, so the pairing survives
the fact that one layer became a chain of three convolutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..search.supernet import get_module


@dataclass
class FinetuneConfig:
    """Section 5.1: lr 1e-5, lambda 0.5, SGD + Nesterov momentum 0.9."""
    epochs: int = 20
    lr: float = 1e-5
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 0.0
    lam: float = 0.5                    # lambda of Eq. (11)
    diss_reduction: str = 'mean'        # 'sum' is literal Eq. (10); 'mean' is scale-stable
    use_weight_term: bool = True
    use_output_term: bool = True
    kd_logits: bool = False             # optional KL on logits (off: not in Eq. 10)
    kd_temperature: float = 4.0
    kd_weight: float = 1.0
    scheduler: str = 'cosine'           # 'cosine' | 'none'
    grad_clip: float = 0.0
    label_smoothing: float = 0.0
    log_every: int = 100
    amp: bool = False


class LayerOutputCollector:
    """Captures the outputs of a set of named modules during a forward pass."""

    def __init__(self, model: nn.Module, names: Sequence[str]):
        self.model, self.names = model, list(names)
        self.outputs: Dict[str, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        def make_hook(name):
            def hook(_module, _inp, out):
                self.outputs[name] = out
            return hook
        for n in self.names:
            self._handles.append(get_module(self.model, n).register_forward_hook(make_hook(n)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self):
        self.outputs.clear()


class DistillationLoss(nn.Module):
    """Eq. (10) + Eq. (11)."""

    def __init__(self, teacher: nn.Module, student: nn.Module, layer_names: Sequence[str],
                 cfg: FinetuneConfig):
        super().__init__()
        self.cfg = cfg
        self.layer_names = list(layer_names)
        self.teacher = teacher
        self.student = student
        # frozen copies of the original weights W_i for the Frobenius term
        self.orig_weights = {n: get_module(teacher, n).weight.detach().clone()
                             for n in self.layer_names}
        self.t_collector = LayerOutputCollector(teacher, self.layer_names)
        self.s_collector = LayerOutputCollector(student, self.layer_names)

    def _reduce(self, sq_err: torch.Tensor, numel: int) -> torch.Tensor:
        return sq_err / numel if self.cfg.diss_reduction == 'mean' else sq_err

    def weight_term(self) -> torch.Tensor:
        """``sum_i ||W_i - W_hat_i||_F^2``."""
        dev = next(self.student.parameters()).device
        total = torch.zeros((), device=dev)
        for name in self.layer_names:
            w_hat = get_module(self.student, name).reconstruct()
            w_orig = self.orig_weights[name].to(dev)
            total = total + self._reduce(torch.sum((w_orig - w_hat) ** 2), w_orig.numel())
        return total

    def output_term(self) -> torch.Tensor:
        """``sum_x sum_i ||O_i(x) - D_i(x)||_F^2`` for the current batch."""
        dev = next(self.student.parameters()).device
        total = torch.zeros((), device=dev)
        for name in self.layer_names:
            o_i = self.t_collector.outputs[name].detach()
            d_i = self.s_collector.outputs[name]
            total = total + self._reduce(torch.sum((o_i - d_i) ** 2), d_i.numel())
        return total

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Runs both models on ``x`` and returns ``(L_f, parts)``."""
        cfg = self.cfg
        with torch.no_grad():
            self.t_collector.clear()
            t_logits = self.teacher(x)
        self.s_collector.clear()
        s_logits = self.student(x)

        ce = F.cross_entropy(s_logits, y, label_smoothing=cfg.label_smoothing)
        diss = torch.zeros((), device=s_logits.device)
        if cfg.use_weight_term:
            diss = diss + self.weight_term()
        if cfg.use_output_term:
            diss = diss + self.output_term()

        loss = ce + cfg.lam * diss
        parts = {'ce': ce.detach(), 'diss': diss.detach()}

        if cfg.kd_logits:
            T = cfg.kd_temperature
            kd = F.kl_div(F.log_softmax(s_logits / T, dim=1),
                          F.log_softmax(t_logits / T, dim=1),
                          reduction='batchmean', log_target=True) * (T * T)
            loss = loss + cfg.kd_weight * kd
            parts['kd'] = kd.detach()
        return loss, parts

    def __enter__(self):
        self.t_collector.__enter__()
        self.s_collector.__enter__()
        return self

    def __exit__(self, *exc):
        self.t_collector.__exit__(*exc)
        self.s_collector.__exit__(*exc)
