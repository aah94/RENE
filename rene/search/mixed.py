"""Continuous relaxation of the rank choice (Section 4.1).

For layer ``i`` we keep one decomposition ``W_hat^{(r)}_i`` per candidate rank
``r in R_i`` together with a learnable logit ``alpha^{(r)}_i``. The rank
coefficient is

    p^{(r)}_i = softmax(alpha_i)[r]

and the layer behaves, in the forward pass, like a dense layer whose weight is
the rank-coefficient-weighted mixture

    W_mix_i = sum_{r in R_i} p^{(r)}_i * W_hat^{(r)}_i .

Because convolution is linear in the weight, mixing the *weights* is exactly
equivalent to mixing the outputs of the individual branches, but costs a single
convolution instead of ``|R_i|`` of them.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decomposition.layers import FACTORIZED, build_factorized, max_rank_of


def _cpu_clone(module: nn.Module) -> nn.Module:
    """Detached float32 CPU copy of a dense layer, used only as a template."""
    import copy
    clone = copy.deepcopy(module).to(device='cpu', dtype=torch.float32)
    for prm in clone.parameters():
        prm.requires_grad_(False)
    return clone


class MixedRankLayer(nn.Module):
    """Base class holding the candidate branches and the rank logits."""

    def __init__(self, base: nn.Module, candidates: Sequence[int], decomp: str,
                 als_iters: int = 50):
        super().__init__()
        self.decomp = decomp
        self.als_iters = als_iters
        # frozen copy of the pre-trained weight W_i (target of the decomposition loss)
        self.register_buffer('w_orig', base.weight.data.clone())
        self.has_bias = base.bias is not None
        if self.has_bias:
            self.register_buffer('bias_orig', base.bias.data.clone())
        # Keep a CPU copy of the dense layer *outside* the module tree: it is only
        # a template for (re)building branches, and must not be registered as a
        # submodule or its parameters would be trained and counted twice.
        object.__setattr__(self, '_base_meta', _cpu_clone(base))
        self.max_feasible = max_rank_of(base, decomp)
        self.branches = nn.ModuleList()
        self.candidates: List[int] = []
        self.alpha = nn.Parameter(torch.zeros(1))
        self.reset_candidates(candidates)

    # -- candidate management ------------------------------------------------
    def sanitize(self, candidates: Sequence[int]) -> List[int]:
        """Clip to the feasible range and drop duplicates, keeping order."""
        seen, out = set(), []
        for r in candidates:
            r = int(max(1, min(int(r), self.max_feasible)))
            if r not in seen:
                seen.add(r)
                out.append(r)
        return sorted(out)

    def reset_candidates(self, candidates: Sequence[int]):
        """(Re)build the branches for a refined search space (Algorithm 1, l.14).

        The paper reinitialises weights *and* rank coefficients before each
        outer iteration, which is what this does.
        """
        self.candidates = self.sanitize(candidates)
        device, dtype = self.w_orig.device, self.w_orig.dtype
        kw = {'als_iters': self.als_iters} if self.decomp.startswith('cp') else {}
        # TT-SVD / CP-ALS run on CPU (accelerator linalg coverage is patchy),
        # then the initialised branches are moved to the working device.
        self.branches = nn.ModuleList([
            build_factorized(self._base_meta, r, self.decomp, **kw)
            for r in self.candidates
        ]).to(device=device, dtype=dtype)
        self.alpha = nn.Parameter(torch.zeros(len(self.candidates), device=device))
        return self

    # -- rank coefficients ---------------------------------------------------
    @property
    def p(self) -> torch.Tensor:
        """Rank coefficients ``p^{(r)}_i = softmax(alpha^{(r)}_i)``."""
        return F.softmax(self.alpha, dim=0)

    def mixed_weight(self) -> torch.Tensor:
        """``sum_r p^{(r)} W_hat^{(r)}``."""
        p = self.p
        out = None
        for coeff, branch in zip(p, self.branches):
            w = branch.reconstruct()
            out = coeff * w if out is None else out + coeff * w
        return out

    # -- losses --------------------------------------------------------------
    def decomposition_loss(self) -> torch.Tensor:
        """``||W_i - sum_r p^{(r)}_i W_hat^{(r)}_i||_F^2`` (first term of Eq. 6)."""
        return torch.sum((self.w_orig - self.mixed_weight()) ** 2)

    def rank_term(self, beta: float) -> torch.Tensor:
        """``(sum_r p^{(r)}_i * r / max R_i) ** beta`` (inner term of Eq. 5)."""
        ranks = torch.tensor([float(r) for r in self.candidates],
                             device=self.alpha.device, dtype=self.alpha.dtype)
        expected = torch.sum(self.p * ranks / max(self.candidates))
        return expected.clamp_min(1e-12) ** beta

    # -- selection -----------------------------------------------------------
    @torch.no_grad()
    def best_rank(self) -> int:
        """``argmax_r softmax(alpha^{(r)}_i)`` (Algorithm 1, l.11)."""
        return int(self.candidates[int(torch.argmax(self.p).item())])

    @torch.no_grad()
    def expected_rank(self) -> float:
        ranks = torch.tensor([float(r) for r in self.candidates], device=self.alpha.device)
        return float(torch.sum(self.p * ranks))

    def weight_parameters(self):
        return self.branches.parameters()

    def alpha_parameters(self):
        return [self.alpha]

    def extra_repr(self):
        return f'decomp={self.decomp}, candidates={self.candidates}'


class MixedRankConv2d(MixedRankLayer):
    def __init__(self, conv: nn.Conv2d, candidates: Sequence[int], decomp: str,
                 als_iters: int = 50):
        super().__init__(conv, candidates, decomp, als_iters)
        self.stride, self.padding = conv.stride, conv.padding
        self.dilation, self.groups = conv.dilation, conv.groups

    def forward(self, x):
        bias = self.bias_orig if self.has_bias else None
        return F.conv2d(x, self.mixed_weight(), bias, self.stride,
                        self.padding, self.dilation, self.groups)


class MixedRankLinear(MixedRankLayer):
    def __init__(self, lin: nn.Linear, candidates: Sequence[int], decomp: str,
                 als_iters: int = 50):
        super().__init__(lin, candidates, decomp, als_iters)

    def forward(self, x):
        bias = self.bias_orig if self.has_bias else None
        return F.linear(x, self.mixed_weight(), bias)


def make_mixed(module: nn.Module, candidates: Sequence[int], decomp: str,
               als_iters: int = 50) -> MixedRankLayer:
    if isinstance(module, nn.Conv2d):
        return MixedRankConv2d(module, candidates, decomp, als_iters)
    if isinstance(module, nn.Linear):
        return MixedRankLinear(module, candidates, decomp, als_iters)
    raise TypeError(f'cannot build a mixed-rank layer from {type(module).__name__}')
