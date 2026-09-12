"""Correctness of the TT / CP factorisations."""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rene.decomposition import build_factorized, max_rank_of
from rene.decomposition.cp import cp_reconstruct, parafac_als
from rene.decomposition.tt import tt_reconstruct_conv, tt_svd_conv


def test_cp_als_recovers_exact_low_rank_tensor():
    torch.manual_seed(0)
    factors = [torch.randn(12, 3).double(), torch.randn(9, 3).double(),
               torch.randn(8, 3).double()]
    tensor = cp_reconstruct(factors)
    est = parafac_als(tensor, 3, n_iter=300)
    rel = (cp_reconstruct(est) - tensor).norm() / tensor.norm()
    assert rel < 1e-6, f'CP-ALS failed to recover an exact rank-3 tensor (rel={rel})'


def test_cp_als_error_decreases_with_rank():
    torch.manual_seed(0)
    tensor = torch.randn(16, 9, 8).double()
    errs = []
    for r in (2, 4, 8, 16):
        est = parafac_als(tensor, r, n_iter=200)
        errs.append(((cp_reconstruct(est) - tensor).norm() / tensor.norm()).item())
    assert all(a >= b - 1e-6 for a, b in zip(errs, errs[1:])), errs


def test_tt_svd_is_exact_at_full_rank():
    torch.manual_seed(0)
    w = torch.randn(8, 4, 3, 3).double()
    # r1 bounded by C_out, r2 by C_in -> full TT rank reconstructs exactly
    g_t, g_y, g_s = tt_svd_conv(w, 8, 4)
    rel = (tt_reconstruct_conv(g_t, g_y, g_s) - w).norm() / w.norm()
    assert rel < 1e-8, rel


def test_tt_error_decreases_with_rank():
    torch.manual_seed(0)
    w = torch.randn(32, 16, 3, 3)
    errs = []
    for r in (2, 4, 8, 16):
        g = tt_svd_conv(w, r, r)
        errs.append(((tt_reconstruct_conv(*g) - w).norm() / w.norm()).item())
    assert all(a >= b - 1e-6 for a, b in zip(errs, errs[1:])), errs


@pytest.mark.parametrize('decomp', ['tt', 'cp', 'cp4'])
@pytest.mark.parametrize('stride,padding', [(1, 1), (2, 1)])
def test_factorized_forward_equals_conv_with_reconstructed_weight(decomp, stride, padding):
    """The layer chain must be exactly a convolution with the rebuilt weight."""
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 24, 3, stride=stride, padding=padding, bias=False)
    layer = build_factorized(conv, 8, decomp)
    x = torch.randn(2, 16, 12, 12)
    ref = F.conv2d(x, layer.reconstruct(), None, stride, padding)
    assert torch.allclose(layer(x), ref, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize('decomp', ['tt', 'cp', 'cp4'])
def test_factorized_reduces_parameters(decomp):
    conv = nn.Conv2d(64, 128, 3, padding=1, bias=False)
    layer = build_factorized(conv, 16, decomp)
    assert layer.n_params() < conv.weight.numel()


@pytest.mark.parametrize('decomp', ['tt', 'cp'])
def test_linear_factorization(decomp):
    torch.manual_seed(0)
    lin = nn.Linear(64, 32)
    layer = build_factorized(lin, 16, decomp)
    x = torch.randn(4, 64)
    assert torch.allclose(layer(x), F.linear(x, layer.reconstruct(), lin.bias), atol=1e-4)


def test_max_rank_is_respected():
    conv = nn.Conv2d(16, 8, 3, padding=1, bias=False)
    layer = build_factorized(conv, 10_000, 'tt')
    assert layer.rank <= max_rank_of(conv, 'tt')
