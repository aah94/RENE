"""End-to-end wiring: decompose a model, measure it, and run the losses."""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rene.finetune import (FinetuneConfig, decompose_model, decomposed_layer_names,
                           ranks_for_param_budget, uniform_ranks)
from rene.finetune.distill import DistillationLoss
from rene.models import resnet20, vgg16
from rene.utils import compression_summary, count_flops, count_params


@pytest.mark.parametrize('decomp', ['tt', 'cp'])
def test_decompose_model_runs_and_compresses(decomp):
    model = resnet20()
    names = decomposed_layer_names(model)
    ranks = uniform_ranks(model, 8, decomp, names)
    small = decompose_model(model, ranks, decomp, als_iters=10)
    out = small(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)
    assert count_params(small) < count_params(model)
    assert count_flops(small, (1, 3, 32, 32), 'cpu') < count_flops(model, (1, 3, 32, 32), 'cpu')


def test_compression_summary_fields():
    model = resnet20()
    names = decomposed_layer_names(model)
    small = decompose_model(model, uniform_ranks(model, 6, 'tt', names), 'tt', als_iters=5)
    s = compression_summary(model, small, (1, 3, 32, 32), device='cpu')
    assert 0 < s['compression_rate'] < 100
    assert 0 < s['flops_reduction_pct'] < 100


def test_param_budget_bisection_hits_the_target():
    model = resnet20()
    names = decomposed_layer_names(model)
    for target in (0.25, 0.5):
        ranks = ranks_for_param_budget(model, target, 'tt', names)
        small = decompose_model(model, ranks, 'tt', als_iters=5)
        got = count_params(small) / count_params(model)
        assert abs(got - target) < 0.15, (target, got)


def test_distillation_loss_is_zero_for_an_identical_pair():
    """A model distilled against itself has no weight and no output discrepancy."""
    import copy
    model = resnet20().eval()
    names = decomposed_layer_names(model)

    class Wrap(nn.Module):
        """Student whose 'factorised' layers reproduce the teacher exactly."""
        pass

    student = copy.deepcopy(model)
    for n in names:
        mod = dict(student.named_modules())[n]
        mod.reconstruct = (lambda m: (lambda: m.weight))(mod)   # identity factorisation

    cfg = FinetuneConfig(lam=0.5, diss_reduction='sum')
    crit = DistillationLoss(model, student, names, cfg)
    x, y = torch.randn(2, 3, 32, 32), torch.tensor([1, 2])
    with crit:
        loss, parts = crit(x, y)
    assert parts['diss'].item() == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(loss)


def test_distillation_loss_is_positive_for_a_decomposed_student():
    model = resnet20().eval()
    names = decomposed_layer_names(model)
    student = decompose_model(model, uniform_ranks(model, 4, 'tt', names), 'tt', als_iters=5)
    crit = DistillationLoss(model, student, names, FinetuneConfig())
    x, y = torch.randn(2, 3, 32, 32), torch.tensor([1, 2])
    with crit:
        loss, parts = crit(x, y)
    assert parts['diss'].item() > 0
    loss.backward()
    assert any(p.grad is not None for p in student.parameters())
