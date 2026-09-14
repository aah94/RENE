# RENE — Rank adapt tENsor dEcomposition

PyTorch reimplementation of **"Unified Framework for Pre-trained Neural Network
Compression via Decomposition and Optimized Rank Selection"**
(Aghababaei-Harandi & Amini, [arXiv:2409.03555](https://arxiv.org/abs/2409.03555)).

RENE compresses a *pre-trained* network by decomposing its weight tensors (TT or
CP) while **searching the per-layer rank** in a continuous space, using a
composite compression loss under a rank constraint. A fine-tuning step with a
layer-wise distillation loss then recovers the original accuracy.

## Updates

- **September 2026** — Initial release: CP/TT decomposition, automatic rank search
  (Algorithm 1) and fine-tuning, with scripts for the paper's experiments.
- **In progress** — RENE is being extended into a general framework for model
  compression, adding **knowledge distillation performed jointly with the rank
  search** and **pruning**. These will be released in upcoming updates.

---

## Where each part of the paper lives

| Paper | Code |
|---|---|
| Eq. (1), (3) — TT decomposition of a conv layer | [`rene/decomposition/tt.py`](rene/decomposition/tt.py) |
| CP decomposition (sum of rank-one tensors) | [`rene/decomposition/cp.py`](rene/decomposition/cp.py) |
| Factorised conv/linear layers | [`rene/decomposition/layers.py`](rene/decomposition/layers.py) |
| §4.1 — rank coefficients `p = softmax(α)` | [`rene/search/mixed.py`](rene/search/mixed.py) |
| Eq. (5), (6), (7) — rank / weight / α losses | [`rene/search/losses.py`](rene/search/losses.py) |
| §4.3 + **Algorithm 1** — multi-step rank search | [`rene/search/rene.py`](rene/search/rene.py) |
| Eq. (10), (11) — `L_diss`, `L_f` fine-tuning | [`rene/finetune/distill.py`](rene/finetune/distill.py) |
| Tables 1–3 metrics (Top-1, FLOPs↓, Comp. Rate) | [`rene/utils/metrics.py`](rene/utils/metrics.py) |

## How the method works

**1. Decomposition.** A conv weight `W ∈ R^{C_out×C_in×k×k}` is viewed as
`T ∈ R^{C_out×(k·k)×C_in}` and factorised. TT (Eq. 3) turns one convolution into
a chain of three:

```
X ──1×1 (C_in→r₂)──▶ ──k×k (r₂→r₁)──▶ ──1×1 (r₁→C_out)──▶ Y
        G_s                 G_y                G_t
```

CP3 gives `1×1 → depthwise k×k → 1×1`. Both ranks are tied (`r₁ = r₂ = r`), as in
the paper. Because convolution is **linear in the weight**, the chain is exactly a
convolution with the reconstructed weight — that identity is what makes the
search below valid, and it is asserted in the test suite.

**2. Continuous rank relaxation (§4.1).** Each layer `i` keeps one decomposition
per candidate rank `r ∈ R_i` plus a learnable logit `α_i^{(r)}`, giving
`p_i^{(r)} = softmax(α_i)`. The layer forwards with the mixed weight
`W_mix = Σ_r p^{(r)} Ŵ^{(r)}` — one convolution, not `|R_i|` of them.

**3. Composite losses (§4.2).** Both are *multiplicative*, so no trade-off
hyper-parameter is needed:

```
L_r      = γ Σ_i ( Σ_r p_i^(r) · r / max R_i )^β                (Eq. 5)
L_Tw     = [ Σ_i ‖W_i − Σ_r p_i^(r) Ŵ_i^(r)‖²_F ] × L_r         (Eq. 6)  → updates weights
L_Tα     = L_val × L_r                                          (Eq. 7)  → updates α
```

`L_Tw` needs **no data at all**; only the α-step touches a held-out validation
split. That is the paper's "no training data in the search step" claim, and it is
why the search is fast.

**4. Multi-step search (§4.3, Algorithm 1).** Start coarse, then contract around
the winner and refine — reproducing Figure 3 exactly:

```
[100 … 800] s=100  →  r̄=200  →  [150, 250] s=10  →  r̄=240  →  [235, 245] s=1  →  r̄*
```

Bounds use the *current* step (`r̄ ± s/2`), then `s ← ⌊s/f⌋`. Weights and α are
reinitialised for each refined space. With the paper's settings this is **2 steps
on CIFAR** (`{10…100}`, s=10, f=10) and **3 on ImageNet** (`{50…850}`, s=100, f=10).

**5. Fine-tuning (§4.4).** The original model is the teacher, the decomposed one
the student:

```
L_diss = Σ_i ‖W_i − Ŵ_i^(r*)‖²_F  +  Σ_x Σ_i ‖O_i(x) − D_i(x)‖²_F     (Eq. 10)
L_f    = L_ce + λ · L_diss                                            (Eq. 11)
```

Layer outputs `O_i` / `D_i` are captured with forward hooks, so the pairing
survives one layer becoming a chain of three.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests -q          # verifies the decompositions and the search
```

## Quick start

```bash
# 0. baseline (ImageNet models come pre-trained from torchvision instead)
python scripts/pretrain.py --arch resnet20 --dataset cifar10 --epochs 200 \
    --out runs/pretrain

# 1. rank search (Algorithm 1)
python scripts/search.py --arch resnet20 --dataset cifar10 --decomp cp \
    --checkpoint runs/pretrain/resnet20_cifar10.pth --out runs/search_r20_cp

# 2. decompose at the searched ranks + fine-tune (Eq. 11)
python scripts/finetune.py --arch resnet20 --dataset cifar10 \
    --checkpoint runs/pretrain/resnet20_cifar10.pth \
    --ranks runs/search_r20_cp/ranks.json --epochs 100 --out runs/ft_r20_cp

# or all three at once
python scripts/run_pipeline.py --arch resnet20 --dataset cifar10 --decomp cp \
    --checkpoint runs/pretrain/resnet20_cifar10.pth --out runs/t1_r20_cp
```

## Reproducing the paper's experiments

### Table 1 — CIFAR-10 (search space `{10…100}`, s=10, f=10, lr 1e-3)

```bash
for d in cp tt; do
  python scripts/run_pipeline.py --arch resnet20 --dataset cifar10 --decomp $d \
      --checkpoint runs/pretrain/resnet20_cifar10.pth --out runs/t1_r20_$d
  python scripts/run_pipeline.py --arch vgg16 --dataset cifar10 --decomp $d \
      --checkpoint runs/pretrain/vgg16_cifar10.pth --out runs/t1_vgg_$d
done
```

| Model | Method | Top-1 | FLOPs ↓% | Comp. Rate |
|---|---|---|---|---|
| ResNet-20 | Original | 91.25 | – | – |
| ResNet-20 | RENE(CP) | 90.82 | 73.44 | 77.62 |
| ResNet-20 | RENE(TT) | 91.40 | 70.40 | 72.28 |
| VGG-16 | Original | 92.78 | – | – |
| VGG-16 | RENE(CP) | 92.51 | 86.23 | 98.60 |
| VGG-16 | RENE(TT) | 93.20 | 86.10 | 95.51 |

### Table 2 — ImageNet-1K (search space `{50…850}`, s=100, f=10, lr 1e-4)

```bash
python scripts/run_pipeline.py --arch resnet18 --dataset imagenet --pretrained \
    --decomp tt --rank-min 50 --rank-max 850 --step 100 --lr-w 1e-4 \
    --data-root /path/to/imagenet --epochs 30 --out runs/t2_r18_tt

# MobileNetV2 is dominated by 1x1 convs, so they must be in the search space
python scripts/run_pipeline.py --arch mobilenetv2 --dataset imagenet --pretrained \
    --decomp tt --include-1x1 --rank-min 50 --rank-max 850 --step 100 --lr-w 1e-4 \
    --data-root /path/to/imagenet --epochs 30 --out runs/t2_mbv2_tt
```

Paper: ResNet-18 RENE(TT) 70.88 / −68.9% FLOPs / 67.1% comp.;
MobileNetV2 RENE(TT) 70.12 / −26.7% / 42.34%.

### Figure 4 — automatic vs. manual rank selection

```bash
python scripts/exp_manual_vs_auto.py --arch vgg16 --dataset cifar10 --decomp tt \
    --checkpoint runs/pretrain/vgg16_cifar10.pth \
    --ranks runs/search_vgg_tt/ranks.json --epochs 20 --out runs/fig4_vgg_c10
```

Sweeps a single global rank hitting {1, 5, 10, 25, 50, 75}% of the original
parameters and compares against the searched ranks.

### Figure 5 — rank distribution, CP vs. TT

```bash
python scripts/exp_rank_distribution.py \
    --ranks runs/t2_r18_cp/ranks.json runs/t2_r18_tt/ranks.json \
    --labels "CP Ranks" "TT Ranks" --out runs/fig5
```

### Table 3 — double compression (KD + RENE)

```bash
python scripts/exp_double_compression.py --dataset cifar100 \
    --teacher-arch resnet56 --student-arch resnet20 \
    --teacher-ckpt runs/pretrain/resnet56_cifar100.pth --out runs/table3_c100
```

---

## Which layers get decomposed

`LayerSelectionRule` defaults select the **block convolutions**; the 1×1 downsample
shortcuts and the classifier stay dense, and the small CIFAR stem (432 params) falls
below `min_params`. That gives **18** searched layers on ResNet-20 and 13 on VGG-16.
The ImageNet stem (7×7, 3→64) is far above `min_params`, so ResNet-18 gives 17.
Override with `--include-1x1`, `--include-linear`, `--include-grouped`,
`--include-classifier`, `--skip-names`.

## Implementation notes

* **Rank feasibility.** Candidates are clipped per layer to what the layer can
  actually support (`min(C_out, C_in)` for TT), so a global search space like
  `{50…850}` is legal for every layer.
* **Search-step schedule.** `while s ≥ 1`, with bounds from the *current* step and
  `s ← ⌊s/f⌋` afterwards — the reading that reproduces Figure 3 and the paper's
  stated step counts. (Algorithm 1's listing divides `s` one line early, which
  would give `[195, 205]` instead of the figure's `[150, 250]`.)
* **Warm-up.** §5.1: "we initially update only the weights for several iterations
  before jointly updating both" — `--warmup-iters`, default 20.
* **`L_diss` scale.** Eq. (10) sums squared errors over every activation element,
  which is orders of magnitude above `L_ce`; with λ=0.5 that swamps the CE term.
  `--diss-reduction mean` (default) normalises each term by its element count and
  keeps the two on a comparable scale; `--diss-reduction sum` is the literal
  equation. Try `sum` if you want the letter of the paper.
* **CP-ALS** is self-contained (no `tensorly` dependency) and initialised by SVD.
  It recovers an exact rank-3 tensor to machine precision — see the tests.
* **Devices.** TT-SVD / CP-ALS run on CPU (accelerator `linalg` coverage is
  patchy) and the initialised branches are moved to the working device.

## Status

The decomposition and search components are verified by the test suite: TT-SVD is
exact at full rank, CP-ALS recovers exact low-rank tensors, the factorised chain
matches a dense convolution with the reconstructed weight, and the FLOPs/param
counters agree with published values (ResNet-20 40.8M / 0.27M, VGG-16 313M / 14.7M,
ResNet-18 1.81G / 11.69M).

The full pipeline runs end to end on GPU for CIFAR-10 / ResNet-20. Two practical
notes from those runs:

* **Use gradient clipping with CP.** `SearchConfig.grad_clip` defaults to `0` and
  CP-ALS initialisation produces large early gradients, which can diverge to `NaN`
  within a few iterations; a `NaN` softmax then makes `argmax` return the *minimum*
  rank for every layer. Setting `grad_clip=5` keeps the search stable.
* **The fine-tuning learning rate depends on the compression level.** At ~76%
  parameter reduction `lr=1e-5` works well and `1e-2` destroys the model; at ~95%
  reduction the decomposed net starts near chance and the larger rate is better.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{aghababaeiharandi2024rene,
  title   = {Unified Framework for Pre-trained Neural Network Compression via
             Decomposition and Optimized Rank Selection},
  author  = {Aghababaei-Harandi, Ali and Amini, Massih-Reza},
  journal = {arXiv preprint arXiv:2409.03555},
  year    = {2024},
  url     = {https://arxiv.org/abs/2409.03555}
}
```
