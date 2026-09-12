"""Tensor-Train (TT) decomposition of convolutional and linear layers.

Implements Eq. (1) and Eq. (3) of the paper.

For a convolution weight ``W in R^{C_out x C_in x k x k}`` we view the tensor in
the TT-friendly ordering ``T in R^{C_out x (k*k) x C_in}`` with

    T[t, s, c] = W[t, c, i1, i2],   s = i1 * k + i2

and factorise it as a 3-core tensor train (Eq. 1 with N = 3)::

    T[t, s, c] = sum_{r1, r2} G_t[t, r1] G_y[r1, s, r2] G_s[r2, c]

which turns a single convolution into the three-convolution chain of Eq. (3)::

    X --(1x1 conv, G_s)--> R2 --(kxk conv, G_y)--> R1 --(1x1 conv, G_t)--> C_out

Following the paper ("we assume that the two TT ranks are equal") the default
setting ties ``R1 = R2 = r``, but the code keeps both ranks explicit so the
assumption can be lifted.
"""

from __future__ import annotations

from typing import Tuple

import torch


# --------------------------------------------------------------------------- #
# Rank feasibility
# --------------------------------------------------------------------------- #
def tt_conv_max_rank(out_channels: int, in_channels: int, kernel_size: Tuple[int, int]) -> int:
    """Largest TT rank that is still meaningful for a conv weight.

    Rank ``R1`` is bounded by the first unfolding ``(C_out, k*k*C_in)`` and rank
    ``R2`` by the second one ``(R1*k*k, C_in)``. With tied ranks the binding
    constraint is ``min(C_out, C_in, R1*k*k)`` which we relax to the simple
    ``min(C_out, C_in)`` bound used in the experiments.
    """
    kh, kw = kernel_size
    return max(1, min(out_channels, in_channels, out_channels * kh * kw, in_channels * kh * kw))


def tt_conv_num_params(out_channels: int, in_channels: int, kernel_size: Tuple[int, int],
                       r1: int, r2: int, bias: bool = False) -> int:
    """Parameter count of the TT-decomposed convolution (Eq. 3)."""
    kh, kw = kernel_size
    n = out_channels * r1 + r1 * kh * kw * r2 + r2 * in_channels
    return n + (out_channels if bias else 0)


def tt_linear_max_rank(out_features: int, in_features: int) -> int:
    return max(1, min(out_features, in_features))


# --------------------------------------------------------------------------- #
# TT-SVD initialisation
# --------------------------------------------------------------------------- #
def _truncated_svd(mat: torch.Tensor, rank: int):
    """Rank-truncated SVD returning ``(U, S, Vh)`` with at most ``rank`` columns."""
    # torch.linalg.svd on CPU/GPU; full_matrices=False keeps memory bounded.
    U, S, Vh = torch.linalg.svd(mat.double(), full_matrices=False)
    rank = min(rank, S.numel())
    return U[:, :rank], S[:rank], Vh[:rank, :]


def tt_svd_conv(weight: torch.Tensor, r1: int, r2: int):
    """TT-SVD initialisation of a conv weight.

    Args:
        weight: ``(C_out, C_in, kh, kw)`` tensor.
        r1, r2: the two TT ranks.

    Returns:
        ``(G_t, G_y, G_s)`` with shapes ``(C_out, r1)``, ``(r1, r2, kh, kw)``
        and ``(r2, C_in)``. ``G_y`` is already laid out as a conv kernel
        (``out=r1``, ``in=r2``).
    """
    c_out, c_in, kh, kw = weight.shape
    dtype, device = weight.dtype, weight.device

    r1 = max(1, min(r1, c_out, c_in * kh * kw))
    r2 = max(1, min(r2, c_in, r1 * kh * kw))

    # T[t, s, c] with s = (i1, i2)
    T = weight.permute(0, 2, 3, 1).reshape(c_out, kh * kw, c_in)

    # --- first unfolding: (C_out) x (k*k * C_in) -----------------------------
    U, S, Vh = _truncated_svd(T.reshape(c_out, kh * kw * c_in), r1)
    r1_eff = S.numel()
    G_t = U                                            # (C_out, r1)
    M = torch.diag(S) @ Vh                             # (r1, k*k*C_in)

    # --- second unfolding: (r1 * k*k) x (C_in) -------------------------------
    r2 = max(1, min(r2, c_in, r1_eff * kh * kw))
    U2, S2, Vh2 = _truncated_svd(M.reshape(r1_eff * kh * kw, c_in), r2)
    r2_eff = S2.numel()
    G_y = U2.reshape(r1_eff, kh, kw, r2_eff)           # (r1, kh, kw, r2)
    G_s = torch.diag(S2) @ Vh2                         # (r2, C_in)

    G_t = G_t.to(dtype=dtype, device=device).contiguous()
    G_y = G_y.permute(0, 3, 1, 2).to(dtype=dtype, device=device).contiguous()  # (r1, r2, kh, kw)
    G_s = G_s.to(dtype=dtype, device=device).contiguous()
    return G_t, G_y, G_s


def tt_reconstruct_conv(G_t: torch.Tensor, G_y: torch.Tensor, G_s: torch.Tensor) -> torch.Tensor:
    """Rebuild ``W in R^{C_out x C_in x kh x kw}`` from the three TT cores."""
    r1, r2, kh, kw = G_y.shape
    # (C_out, r1) x (r1, r2*kh*kw) -> (C_out, r2, kh, kw)
    tmp = G_t @ G_y.reshape(r1, r2 * kh * kw)
    tmp = tmp.reshape(-1, r2, kh, kw)
    # contract r2 with G_s: (C_out, r2, kh, kw) x (r2, C_in) -> (C_out, C_in, kh, kw)
    w = torch.einsum('orhw,ri->oihw', tmp, G_s)
    return w.contiguous()


def tt_svd_linear(weight: torch.Tensor, rank: int):
    """Rank-``r`` SVD factorisation of a linear weight ``(out, in)``.

    Returns ``(U, V)`` with ``W ~ U @ V``, shapes ``(out, r)`` and ``(r, in)``.
    """
    out_f, in_f = weight.shape
    rank = max(1, min(rank, out_f, in_f))
    U, S, Vh = _truncated_svd(weight, rank)
    sqrt_s = torch.diag(torch.sqrt(S))
    A = (U @ sqrt_s).to(weight.dtype).to(weight.device).contiguous()
    B = (sqrt_s @ Vh).to(weight.dtype).to(weight.device).contiguous()
    return A, B
