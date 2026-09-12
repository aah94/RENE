"""Canonical Polyadic (CP) decomposition of convolutional and linear layers.

CP expresses a tensor as a sum of rank-one tensors. For a conv weight
``W in R^{C_out x C_in x kh x kw}`` we use the standard CNN-friendly 3-way form
(``CP3``), obtained by folding the spatial modes together:

    T[t, s, c] = sum_{r} A[t, r] B[s, r] C[c, r],   s = i1 * kw + i2

which maps onto three cheap convolutions

    X --(1x1 conv, C)--> R --(kxk depthwise conv, B)--> R --(1x1 conv, A)--> C_out

and the fully separable 4-way form (``CP4``)

    W[t, c, i1, i2] = sum_r A[t, r] H[i1, r] V[i2, r] C[c, r]

which splits the spatial kernel into a (kh x 1) and a (1 x kw) depthwise conv.

Factors are initialised with alternating least squares (ALS); the search stage
then refines them by gradient descent on the decomposition loss (Eq. 6).
"""

from __future__ import annotations

from typing import List, Tuple

import torch


# --------------------------------------------------------------------------- #
# Rank feasibility / cost
# --------------------------------------------------------------------------- #
def cp_conv_max_rank(out_channels: int, in_channels: int, kernel_size: Tuple[int, int]) -> int:
    """A CP rank above this cannot reduce the parameter count."""
    kh, kw = kernel_size
    dense = out_channels * in_channels * kh * kw
    per_rank = out_channels + in_channels + kh * kw          # CP3 cost per rank
    return max(1, min(dense // max(per_rank, 1), out_channels * in_channels))


def cp_conv_num_params(out_channels: int, in_channels: int, kernel_size: Tuple[int, int],
                       rank: int, mode: str = 'cp3', bias: bool = False) -> int:
    kh, kw = kernel_size
    if mode == 'cp3':
        n = rank * (out_channels + in_channels + kh * kw)
    else:  # cp4
        n = rank * (out_channels + in_channels + kh + kw)
    return n + (out_channels if bias else 0)


def cp_linear_max_rank(out_features: int, in_features: int) -> int:
    return max(1, min(out_features, in_features))


# --------------------------------------------------------------------------- #
# ALS
# --------------------------------------------------------------------------- #
def _unfold(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    """Mode-``mode`` unfolding: ``(I_mode, prod(other dims))``."""
    return torch.movedim(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def _khatri_rao(matrices: List[torch.Tensor]) -> torch.Tensor:
    """Column-wise Kronecker product of matrices sharing the number of columns."""
    rank = matrices[0].shape[1]
    out = matrices[0]
    for m in matrices[1:]:
        out = (out[:, None, :] * m[None, :, :]).reshape(-1, rank)
    return out


def parafac_als(tensor: torch.Tensor, rank: int, n_iter: int = 100, tol: float = 1e-8,
                init: str = 'svd', seed: int = 0) -> List[torch.Tensor]:
    """Plain CP-ALS. Returns one factor matrix ``(I_k, rank)`` per mode.

    Self-contained (no tensorly dependency) so the repo stays torch-only.
    """
    tensor = tensor.double()
    n_modes = tensor.dim()
    device = tensor.device
    gen = torch.Generator(device='cpu').manual_seed(seed)

    factors: List[torch.Tensor] = []
    for mode in range(n_modes):
        dim = tensor.shape[mode]
        if init == 'svd' and dim > 1:
            unfolded = _unfold(tensor, mode)
            U, _, _ = torch.linalg.svd(unfolded, full_matrices=False)
            if U.shape[1] < rank:                       # pad with random columns
                pad = torch.randn(dim, rank - U.shape[1], generator=gen).to(device).double()
                U = torch.cat([U, pad], dim=1)
            factors.append(U[:, :rank].contiguous())
        else:
            factors.append(torch.randn(dim, rank, generator=gen).to(device).double())

    norm_t = torch.norm(tensor)
    prev_err = None
    eye = torch.eye(rank, dtype=tensor.dtype, device=device)

    for _ in range(n_iter):
        for mode in range(n_modes):
            others = [factors[m] for m in range(n_modes) if m != mode]
            # Gram of the Khatri-Rao product without forming it:
            # (KR^T KR) = Hadamard product of the per-factor Grams.
            # NB: the accumulator must start at ones, not at the identity.
            gram = torch.ones(rank, rank, dtype=tensor.dtype, device=device)
            for m in others:
                gram = gram * (m.T @ m)
            kr = _khatri_rao(others)                    # (prod other dims, rank)
            rhs = _unfold(tensor, mode) @ kr            # (I_mode, rank)
            factors[mode] = torch.linalg.solve(
                gram + 1e-9 * eye, rhs.T).T.contiguous()

        approx = cp_reconstruct(factors)
        err = torch.norm(tensor - approx) / (norm_t + 1e-12)
        if prev_err is not None and abs(prev_err - err.item()) < tol:
            break
        prev_err = err.item()

    return factors


def cp_reconstruct(factors: List[torch.Tensor]) -> torch.Tensor:
    """Rebuild a tensor from CP factor matrices (all with ``rank`` columns).

    ``out[i_1, ..., i_N] = sum_r prod_k factors[k][i_k, r]``.
    """
    out = factors[0]                                    # (I_1, rank)
    for f in factors[1:]:
        # (..., 1, rank) * (I_k, rank) -> (..., I_k, rank)
        out = out.unsqueeze(-2) * f
    return out.sum(dim=-1)


# --------------------------------------------------------------------------- #
# Conv / linear specialisations
# --------------------------------------------------------------------------- #
def cp_als_conv3(weight: torch.Tensor, rank: int, n_iter: int = 100, seed: int = 0):
    """CP3 factors of a conv weight.

    Returns ``(K_t, K_d, K_s)`` shaped ``(C_out, r)``, ``(r, 1, kh, kw)``
    (depthwise kernel) and ``(r, C_in)``.
    """
    c_out, c_in, kh, kw = weight.shape
    T = weight.permute(0, 2, 3, 1).reshape(c_out, kh * kw, c_in)
    A, B, C = parafac_als(T, rank, n_iter=n_iter, seed=seed)

    K_t = A.to(weight.dtype).to(weight.device).contiguous()              # (C_out, r)
    K_d = B.T.reshape(rank, 1, kh, kw).to(weight.dtype).to(weight.device).contiguous()
    K_s = C.T.to(weight.dtype).to(weight.device).contiguous()            # (r, C_in)
    return K_t, K_d, K_s


def cp3_reconstruct_conv(K_t: torch.Tensor, K_d: torch.Tensor, K_s: torch.Tensor) -> torch.Tensor:
    """Rebuild ``W in R^{C_out x C_in x kh x kw}`` from CP3 factors."""
    rank, _, kh, kw = K_d.shape
    B = K_d.reshape(rank, kh * kw)                                        # (r, k*k)
    # W[o, i, s] = sum_r K_t[o, r] B[r, s] K_s[r, i],  s = (i1, i2)
    w = torch.einsum('or,rs,ri->ois', K_t, B, K_s)
    return w.reshape(K_t.shape[0], K_s.shape[1], kh, kw).contiguous()


def cp_als_conv4(weight: torch.Tensor, rank: int, n_iter: int = 100, seed: int = 0):
    """CP4 factors: ``(K_t, K_h, K_w, K_s)`` for the fully separable form."""
    c_out, c_in, kh, kw = weight.shape
    T = weight.permute(0, 2, 3, 1)                                        # (C_out, kh, kw, C_in)
    A, H, V, C = parafac_als(T, rank, n_iter=n_iter, seed=seed)
    to = lambda x: x.to(weight.dtype).to(weight.device).contiguous()
    K_t = to(A)                                          # (C_out, r)
    K_h = to(H.T.reshape(rank, 1, kh, 1))                # depthwise (kh, 1)
    K_w = to(V.T.reshape(rank, 1, 1, kw))                # depthwise (1, kw)
    K_s = to(C.T)                                        # (r, C_in)
    return K_t, K_h, K_w, K_s


def cp4_reconstruct_conv(K_t, K_h, K_w, K_s) -> torch.Tensor:
    rank = K_t.shape[1]
    kh, kw = K_h.shape[2], K_w.shape[3]
    H = K_h.reshape(rank, kh)
    V = K_w.reshape(rank, kw)
    w = torch.einsum('or,rh,rw,ri->oihw', K_t, H, V, K_s)
    return w.contiguous()


def cp_als_linear(weight: torch.Tensor, rank: int):
    """Rank-``r`` factorisation of a matrix (CP on a 2-way tensor == SVD)."""
    out_f, in_f = weight.shape
    rank = max(1, min(rank, out_f, in_f))
    U, S, Vh = torch.linalg.svd(weight.double(), full_matrices=False)
    sqrt_s = torch.diag(torch.sqrt(S[:rank]))
    A = (U[:, :rank] @ sqrt_s).to(weight.dtype).to(weight.device).contiguous()
    B = (sqrt_s @ Vh[:rank, :]).to(weight.dtype).to(weight.device).contiguous()
    return A, B
