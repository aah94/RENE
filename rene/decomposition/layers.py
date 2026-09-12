"""Factorised drop-in replacements for ``nn.Conv2d`` / ``nn.Linear``.

Every layer exposes the same three-part interface used by the rest of RENE:

``reconstruct()``   rebuild the dense weight ``W_hat^{(r)}`` (used by the
                    decomposition loss of Eq. 6 and the Frobenius part of
                    ``L_diss`` in Eq. 10);
``from_layer()``    build + initialise from a pre-trained dense layer;
``n_params()``      parameter count, for the compression-rate metric.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cp import (cp3_reconstruct_conv, cp4_reconstruct_conv, cp_als_conv3,
                 cp_als_conv4, cp_als_linear, cp_conv_max_rank, cp_linear_max_rank)
from .tt import (tt_conv_max_rank, tt_linear_max_rank, tt_reconstruct_conv,
                 tt_svd_conv, tt_svd_linear)


class FactorizedLayer(nn.Module):
    """Common base: stores the geometry of the layer it replaces."""

    def reconstruct(self) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# Tensor Train
# --------------------------------------------------------------------------- #
class TTConv2d(FactorizedLayer):
    """Conv2d decomposed as the three-convolution TT chain of Eq. (3).

    ``X -> [1x1: C_in -> r2] -> [kxk: r2 -> r1] -> [1x1: r1 -> C_out]``
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 r1: int, r2: int, stride=1, padding=0, dilation=1, bias: bool = True):
        super().__init__()
        kh, kw = kernel_size
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = (kh, kw)
        self.stride, self.padding, self.dilation = stride, padding, dilation
        self.r1 = max(1, min(r1, out_channels, in_channels * kh * kw))
        self.r2 = max(1, min(r2, in_channels, self.r1 * kh * kw))

        # G_s: 1x1 conv, C_in -> r2
        self.first = nn.Conv2d(in_channels, self.r2, 1, bias=False)
        # G_y: kxk conv, r2 -> r1 (carries stride / padding / dilation)
        self.core = nn.Conv2d(self.r2, self.r1, (kh, kw), stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        # G_t: 1x1 conv, r1 -> C_out
        self.last = nn.Conv2d(self.r1, out_channels, 1, bias=bias)

    @property
    def rank(self) -> int:
        return self.r1

    def forward(self, x):
        return self.last(self.core(self.first(x)))

    def reconstruct(self) -> torch.Tensor:
        G_s = self.first.weight.reshape(self.r2, self.in_channels)
        G_y = self.core.weight                                  # (r1, r2, kh, kw)
        G_t = self.last.weight.reshape(self.out_channels, self.r1)
        return tt_reconstruct_conv(G_t, G_y, G_s)

    @torch.no_grad()
    def load_from_dense(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        G_t, G_y, G_s = tt_svd_conv(weight, self.r1, self.r2)
        # TT-SVD may return smaller effective ranks; pad into the allocated cores
        self.first.weight.zero_()
        self.core.weight.zero_()
        self.last.weight.zero_()
        e1, e2 = G_t.shape[1], G_s.shape[0]
        self.first.weight[:e2].copy_(G_s.reshape(e2, self.in_channels, 1, 1))
        self.core.weight[:e1, :e2].copy_(G_y)
        self.last.weight[:, :e1].copy_(G_t.reshape(self.out_channels, e1, 1, 1))
        if bias is not None and self.last.bias is not None:
            self.last.bias.copy_(bias)
        return self

    @classmethod
    def from_layer(cls, conv: nn.Conv2d, r1: int, r2: int | None = None):
        r2 = r1 if r2 is None else r2
        layer = cls(conv.in_channels, conv.out_channels, conv.kernel_size, r1, r2,
                    stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                    bias=conv.bias is not None)
        layer = layer.to(conv.weight.device, conv.weight.dtype)
        return layer.load_from_dense(conv.weight.data, None if conv.bias is None else conv.bias.data)

    @staticmethod
    def max_rank(conv: nn.Conv2d) -> int:
        return tt_conv_max_rank(conv.out_channels, conv.in_channels, conv.kernel_size)

    def extra_repr(self):
        return (f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, '
                f'r1={self.r1}, r2={self.r2}')


class TTLinear(FactorizedLayer):
    """Linear layer factorised as ``W ~ A @ B`` with inner rank ``r``."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.r = max(1, min(rank, in_features, out_features))
        self.first = nn.Linear(in_features, self.r, bias=False)
        self.last = nn.Linear(self.r, out_features, bias=bias)

    @property
    def rank(self) -> int:
        return self.r

    def forward(self, x):
        return self.last(self.first(x))

    def reconstruct(self) -> torch.Tensor:
        return self.last.weight @ self.first.weight

    @torch.no_grad()
    def load_from_dense(self, weight, bias=None):
        A, B = tt_svd_linear(weight, self.r)
        self.first.weight.zero_(); self.last.weight.zero_()
        self.first.weight[:B.shape[0]].copy_(B)
        self.last.weight[:, :A.shape[1]].copy_(A)
        if bias is not None and self.last.bias is not None:
            self.last.bias.copy_(bias)
        return self

    @classmethod
    def from_layer(cls, lin: nn.Linear, rank: int, *_):
        layer = cls(lin.in_features, lin.out_features, rank, bias=lin.bias is not None)
        layer = layer.to(lin.weight.device, lin.weight.dtype)
        return layer.load_from_dense(lin.weight.data, None if lin.bias is None else lin.bias.data)

    @staticmethod
    def max_rank(lin: nn.Linear) -> int:
        return tt_linear_max_rank(lin.out_features, lin.in_features)


# --------------------------------------------------------------------------- #
# CP
# --------------------------------------------------------------------------- #
class CPConv2d(FactorizedLayer):
    """Conv2d as a CP3 chain: 1x1 -> depthwise kxk -> 1x1."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 rank: int, stride=1, padding=0, dilation=1, bias: bool = True):
        super().__init__()
        kh, kw = kernel_size
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = (kh, kw)
        self.r = max(1, rank)
        self.first = nn.Conv2d(in_channels, self.r, 1, bias=False)              # K_s
        self.depth = nn.Conv2d(self.r, self.r, (kh, kw), stride=stride, padding=padding,
                               dilation=dilation, groups=self.r, bias=False)    # K_d
        self.last = nn.Conv2d(self.r, out_channels, 1, bias=bias)               # K_t

    @property
    def rank(self) -> int:
        return self.r

    def forward(self, x):
        return self.last(self.depth(self.first(x)))

    def reconstruct(self) -> torch.Tensor:
        K_s = self.first.weight.reshape(self.r, self.in_channels)
        K_d = self.depth.weight                                   # (r, 1, kh, kw)
        K_t = self.last.weight.reshape(self.out_channels, self.r)
        return cp3_reconstruct_conv(K_t, K_d, K_s)

    @torch.no_grad()
    def load_from_dense(self, weight, bias=None, als_iters: int = 100, seed: int = 0):
        K_t, K_d, K_s = cp_als_conv3(weight, self.r, n_iter=als_iters, seed=seed)
        self.first.weight.copy_(K_s.reshape(self.r, self.in_channels, 1, 1))
        self.depth.weight.copy_(K_d)
        self.last.weight.copy_(K_t.reshape(self.out_channels, self.r, 1, 1))
        if bias is not None and self.last.bias is not None:
            self.last.bias.copy_(bias)
        return self

    @classmethod
    def from_layer(cls, conv: nn.Conv2d, rank: int, *_, als_iters: int = 100, seed: int = 0):
        layer = cls(conv.in_channels, conv.out_channels, conv.kernel_size, rank,
                    stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                    bias=conv.bias is not None)
        layer = layer.to(conv.weight.device, conv.weight.dtype)
        return layer.load_from_dense(conv.weight.data,
                                     None if conv.bias is None else conv.bias.data,
                                     als_iters=als_iters, seed=seed)

    @staticmethod
    def max_rank(conv: nn.Conv2d) -> int:
        return cp_conv_max_rank(conv.out_channels, conv.in_channels, conv.kernel_size)

    def extra_repr(self):
        return (f'{self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, rank={self.r}')


class CP4Conv2d(FactorizedLayer):
    """Fully separable CP4 chain: 1x1 -> depthwise (kh,1) -> depthwise (1,kw) -> 1x1."""

    def __init__(self, in_channels, out_channels, kernel_size, rank,
                 stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        kh, kw = kernel_size
        sh, sw = (stride, stride) if isinstance(stride, int) else stride
        ph, pw = (padding, padding) if isinstance(padding, int) else padding
        dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = (kh, kw)
        self.r = max(1, rank)
        self.first = nn.Conv2d(in_channels, self.r, 1, bias=False)
        self.vert = nn.Conv2d(self.r, self.r, (kh, 1), stride=(sh, 1), padding=(ph, 0),
                              dilation=(dh, 1), groups=self.r, bias=False)
        self.horz = nn.Conv2d(self.r, self.r, (1, kw), stride=(1, sw), padding=(0, pw),
                              dilation=(1, dw), groups=self.r, bias=False)
        self.last = nn.Conv2d(self.r, out_channels, 1, bias=bias)

    @property
    def rank(self) -> int:
        return self.r

    def forward(self, x):
        return self.last(self.horz(self.vert(self.first(x))))

    def reconstruct(self) -> torch.Tensor:
        K_s = self.first.weight.reshape(self.r, self.in_channels)
        K_t = self.last.weight.reshape(self.out_channels, self.r)
        return cp4_reconstruct_conv(K_t, self.vert.weight, self.horz.weight, K_s)

    @torch.no_grad()
    def load_from_dense(self, weight, bias=None, als_iters: int = 100, seed: int = 0):
        K_t, K_h, K_w, K_s = cp_als_conv4(weight, self.r, n_iter=als_iters, seed=seed)
        self.first.weight.copy_(K_s.reshape(self.r, self.in_channels, 1, 1))
        self.vert.weight.copy_(K_h)
        self.horz.weight.copy_(K_w)
        self.last.weight.copy_(K_t.reshape(self.out_channels, self.r, 1, 1))
        if bias is not None and self.last.bias is not None:
            self.last.bias.copy_(bias)
        return self

    @classmethod
    def from_layer(cls, conv, rank, *_, als_iters: int = 100, seed: int = 0):
        layer = cls(conv.in_channels, conv.out_channels, conv.kernel_size, rank,
                    stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                    bias=conv.bias is not None)
        layer = layer.to(conv.weight.device, conv.weight.dtype)
        return layer.load_from_dense(conv.weight.data,
                                     None if conv.bias is None else conv.bias.data,
                                     als_iters=als_iters, seed=seed)

    @staticmethod
    def max_rank(conv: nn.Conv2d) -> int:
        return cp_conv_max_rank(conv.out_channels, conv.in_channels, conv.kernel_size)


class CPLinear(TTLinear):
    """CP on a 2-way tensor coincides with a rank-``r`` matrix factorisation."""

    @torch.no_grad()
    def load_from_dense(self, weight, bias=None, **kw):
        A, B = cp_als_linear(weight, self.r)
        self.first.weight.zero_(); self.last.weight.zero_()
        self.first.weight[:B.shape[0]].copy_(B)
        self.last.weight[:, :A.shape[1]].copy_(A)
        if bias is not None and self.last.bias is not None:
            self.last.bias.copy_(bias)
        return self

    @staticmethod
    def max_rank(lin: nn.Linear) -> int:
        return cp_linear_max_rank(lin.out_features, lin.in_features)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
FACTORIZED = {
    'tt':  {'conv': TTConv2d, 'linear': TTLinear},
    'cp':  {'conv': CPConv2d, 'linear': CPLinear},
    'cp4': {'conv': CP4Conv2d, 'linear': CPLinear},
}


def build_factorized(module: nn.Module, rank: int, decomp: str, **kwargs) -> FactorizedLayer:
    """Factorise ``module`` at ``rank`` using ``decomp`` in {'tt', 'cp', 'cp4'}."""
    if decomp not in FACTORIZED:
        raise ValueError(f'unknown decomposition {decomp!r}, expected one of {list(FACTORIZED)}')
    kind = 'conv' if isinstance(module, nn.Conv2d) else 'linear'
    cls = FACTORIZED[decomp][kind]
    return cls.from_layer(module, rank, **kwargs)


def max_rank_of(module: nn.Module, decomp: str) -> int:
    kind = 'conv' if isinstance(module, nn.Conv2d) else 'linear'
    return FACTORIZED[decomp][kind].max_rank(module)
