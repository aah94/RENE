"""Search-stage behaviour: mixed layers, losses, and Algorithm 1."""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rene.models import resnet20
from rene.search import (LayerSelectionRule, RENESearch, SearchConfig, make_mixed,
                         rank_loss, select_layers, total_weight_loss)
from rene.search.supernet import arange_candidates


def test_candidate_grid_matches_paper_search_spaces():
    assert arange_candidates(10, 100, 10) == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert len(arange_candidates(50, 850, 100)) == 9        # ImageNet, step 1
    assert arange_candidates(195, 205, 1)[0] == 195


def test_mixed_layer_softmax_and_selection():
    conv = nn.Conv2d(16, 32, 3, padding=1, bias=False)
    m = make_mixed(conv, [4, 8, 12], 'tt')
    assert torch.allclose(m.p.sum(), torch.tensor(1.0))
    assert m.candidates == [4, 8, 12]
    with torch.no_grad():
        m.alpha.copy_(torch.tensor([0.0, 5.0, 0.0]))
    assert m.best_rank() == 8


def test_mixed_layer_gradients_reach_branches_and_alpha():
    conv = nn.Conv2d(8, 16, 3, padding=1, bias=False)
    m = make_mixed(conv, [2, 4], 'tt')
    loss, _, _ = total_weight_loss([m], gamma=0.4, beta=0.8)
    loss.backward()
    assert m.alpha.grad is not None and torch.isfinite(m.alpha.grad).all()
    assert m.branches[0].first.weight.grad is not None


def test_rank_loss_prefers_small_ranks():
    """L_r must be lower when the mass sits on the small candidate."""
    conv = nn.Conv2d(16, 32, 3, padding=1, bias=False)
    m = make_mixed(conv, [4, 32], 'tt')
    with torch.no_grad():
        m.alpha.copy_(torch.tensor([10.0, -10.0]))
    low = rank_loss([m], 0.4, 0.8).item()
    with torch.no_grad():
        m.alpha.copy_(torch.tensor([-10.0, 10.0]))
    high = rank_loss([m], 0.4, 0.8).item()
    assert low < high


def test_reset_candidates_rebuilds_branches_and_alpha():
    conv = nn.Conv2d(16, 32, 3, padding=1, bias=False)
    m = make_mixed(conv, [4, 8, 12], 'tt')
    m.reset_candidates([10, 11, 12])
    assert m.candidates == [10, 11, 12]
    assert m.alpha.shape[0] == 3
    assert len(m.branches) == 3
    assert torch.allclose(m.alpha, torch.zeros(3))


def test_layer_selection_defaults_skip_shortcuts_and_classifier():
    model = resnet20()
    names = select_layers(model, LayerSelectionRule())
    # the 18 block convolutions only: the 432-param CIFAR stem falls below
    # min_params, and shortcuts / classifier are excluded by the rule
    assert len(names) == 18
    assert 'conv1' not in names                  # stem stays dense
    assert not any('downsample' in n for n in names)   # 1x1 shortcuts are not
    assert 'fc' not in names                     # classifier is not


def test_supernet_freezes_the_backbone():
    model = resnet20()
    cfg = SearchConfig(rank_min=4, rank_max=8, step=4, iters=0, warmup_iters=0)
    s = RENESearch(model, cfg)
    for name, p in s.supernet.model.named_parameters():
        # the branch weights (Eq. 8) and the rank logits (Eq. 9) are the only
        # trainable tensors; the backbone around them stays frozen
        if 'branches' not in name and not name.endswith('alpha'):
            assert not p.requires_grad, name


def _schedule(step, factor):
    """The (s, n_candidates) trajectory Algorithm 1 walks through."""
    cfg = SearchConfig(rank_min=10, rank_max=100, step=step, factor=factor,
                       iters=0, warmup_iters=0, log_every=0)
    s = RENESearch(resnet20(), cfg)
    s.run(val_loader=None)
    return [h['s'] for h in s.history]


def test_search_step_count_matches_paper_cifar():
    """CIFAR-10/100: {10..100}, s=10, f=10 -> 2 search steps."""
    assert _schedule(10, 10) == [10, 1]


def test_search_returns_one_rank_per_selected_layer():
    cfg = SearchConfig(rank_min=8, rank_max=24, step=8, factor=8,
                       iters=1, warmup_iters=1, log_every=0, alpha_objective='recon')
    s = RENESearch(resnet20(), cfg)
    ranks = s.run(val_loader=None)
    assert set(ranks) == set(s.layer_names)
    assert all(isinstance(v, int) and v >= 1 for v in ranks.values())
