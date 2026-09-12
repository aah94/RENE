"""RENE -- Rank adapt tENsor dEcomposition.

Reference implementation of "Unified Framework for Pre-trained Neural Network
Compression via Decomposition and Optimized Rank Selection"
(Aghababaei-Harandi & Amini, arXiv:2409.03555).

Pipeline
--------
1. ``rene.decomposition``  TT / CP factorisations of conv and linear layers.
2. ``rene.search``         continuous rank relaxation + Algorithm 1.
3. ``rene.finetune``       final decomposition and L_f fine-tuning (Eq. 11).
"""

__version__ = '0.1.0'

from .decomposition import build_factorized, max_rank_of
from .finetune import decompose_model, FinetuneConfig, finetune_decomposed
from .search import LayerSelectionRule, RENESearch, SearchConfig
from .utils import compression_summary, evaluate, pick_device, set_seed

__all__ = ['RENESearch', 'SearchConfig', 'LayerSelectionRule', 'decompose_model',
           'FinetuneConfig', 'finetune_decomposed', 'build_factorized',
           'max_rank_of', 'compression_summary', 'evaluate', 'pick_device', 'set_seed']
