"""The composite compression losses of Section 4.2.

Rank loss (Eq. 5)::

    L_r(R) = gamma * sum_i ( sum_{r in R_i} p^{(r)}_i * r / max R_i ) ** beta

Total weight loss (Eq. 6) -- used to update the decomposition weights::

    L_Tw = [ sum_i ||W_i - sum_r p^{(r)}_i W_hat^{(r)}_i||_F^2 ] * L_r(R)

Total parameter loss (Eq. 7) -- used to update the rank logits::

    L_Talpha = L_val * L_r(R)

Both are *multiplicative*: "this scale-invariant feature balances both terms
without requiring separate trade-off hyperparameters".
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from .mixed import MixedRankLayer


def rank_loss(layers: Sequence[MixedRankLayer], gamma: float, beta: float) -> torch.Tensor:
    """Eq. (5)."""
    total = None
    for layer in layers:
        term = layer.rank_term(beta)
        total = term if total is None else total + term
    return gamma * total


def decomposition_loss(layers: Sequence[MixedRankLayer]) -> torch.Tensor:
    """``sum_i ||W_i - sum_r p^{(r)}_i W_hat^{(r)}_i||_F^2`` (first term of Eq. 6)."""
    total = None
    for layer in layers:
        term = layer.decomposition_loss()
        total = term if total is None else total + term
    return total


def total_weight_loss(layers: Sequence[MixedRankLayer], gamma: float, beta: float):
    """Eq. (6). Returns ``(loss, decomposition_part, rank_part)``."""
    ld = decomposition_loss(layers)
    lr = rank_loss(layers, gamma, beta)
    return ld * lr, ld.detach(), lr.detach()


def total_alpha_loss(val_loss: torch.Tensor, layers: Sequence[MixedRankLayer],
                     gamma: float, beta: float):
    """Eq. (7). Returns ``(loss, val_part, rank_part)``."""
    lr = rank_loss(layers, gamma, beta)
    return val_loss * lr, val_loss.detach(), lr.detach()


# --------------------------------------------------------------------------- #
# Fine-tuning losses (Section 4.4)
# --------------------------------------------------------------------------- #
def distillation_loss(weight_pairs: Iterable, output_pairs: Iterable) -> torch.Tensor:
    """Eq. (10).

    ``L_diss = sum_i ||W_i - W_hat^{(r*)}_i||_F^2 + sum_x sum_i ||O_i(x) - D_i(x)||_F^2``

    Args:
        weight_pairs: iterable of ``(W_i, W_hat_i)`` tensors.
        output_pairs: iterable of ``(O_i(x), D_i(x))`` activation tensors.
    """
    total = None

    def _acc(t, acc):
        return t if acc is None else acc + t

    for w_orig, w_hat in weight_pairs:
        total = _acc(torch.sum((w_orig - w_hat) ** 2), total)
    for o_i, d_i in output_pairs:
        if o_i.shape != d_i.shape:
            raise ValueError(f'teacher/student activation mismatch {o_i.shape} vs {d_i.shape}')
        total = _acc(torch.sum((o_i - d_i) ** 2), total)
    if total is None:
        return torch.zeros((), requires_grad=True)
    return total


def finetune_loss(ce: torch.Tensor, diss: torch.Tensor, lam: float) -> torch.Tensor:
    """Eq. (11): ``L_f = L_ce + lambda * L_diss``."""
    return ce + lam * diss
