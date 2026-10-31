# Don't Read Everything: A Curvature-Conditioned Query for Linear Attention

[![EMNLP 2026](https://img.shields.io/badge/EMNLP%202026-Accepted-brightgreen)](https://arxiv.org/abs/2606.01294)
[![arXiv](https://img.shields.io/badge/arXiv-PDF-red?logo=arxiv)](https://arxiv.org/abs/2606.01294)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Paper-orange)](https://huggingface.co/papers/2606.01294)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**🎉 Accepted to EMNLP 2026!**

This is the official repository for **[Don't Read Everything: A Curvature-Conditioned Query for Linear Attention](https://arxiv.org/abs/2606.01294)**.

The repo provides **CCQ**, a read-side cleaning operator for linear attention, instantiated on two backbones and registered as HuggingFace models:

- **CCQ-GDN** (`model_type: ccq_gdn`): Curvature-Conditioned Query on a **Gated DeltaNet** backbone
- **CCQ-GLA** (`model_type: ccq_gla`): Curvature-Conditioned Query on a **Gated Linear Attention** backbone

## Introduction

Linear attention reduces the quadratic cost of softmax attention by maintaining a recurrent fast-weight state, but it consistently lags on in-context retrieval and long-context tasks. Existing remedies act on the **write side** of memory through gating, delta updates, or kernel feature maps — the read step is left unchanged: every past key contributes additively to the output, so useful targets are diluted by the bulk of stored vectors.

**CCQ** borrows one specific piece of softmax's machinery — the curvature of its log-partition function — and grafts it onto the read path of linear attention. The query is contracted along the high-variance directions of the running key covariance Σₜ (the Hessian of the softmax log-partition at q = 0) before it touches the linear-attention state:

```
q_clean_t = (I − λ_t Σ_t) q_t,      Σ_t = C̄_t − μ_t μ_tᵀ,      λ_t = σ(W_λ q_t + b_λ)
```

The write rule of the backbone is untouched, so CCQ slots onto any linear-attention variant at the cost of one extra d_k×d_k running statistic and one d_k vector per head.

## News
- [2026-10-31] Code update: Transformers 5.x and flash-linear-attention 0.5.x compatibility (generation cache fix), fused cleaning kernels (`attn_mode: "fused_chunk"`) that cut the cleaning overhead by roughly a quarter to two fifths in kernel time, and reproduced paper numbers. See the update section below.
- **[2026-10-28] 🏆 CCQ received the Best Paper Award at EMNLP 2026!**
- **[2026-08-30] 🎉 Our paper has been accepted to EMNLP 2026!**
- [2026-06] Code released: model implementations, paper configs, and fused Triton kernels.
- [2026-06] Paper available on [arXiv](https://arxiv.org/abs/2606.01294).

## Installation

Requires Python ≥ 3.12, PyTorch ≥ 2.5 **with CUDA**,
[Triton](https://github.com/triton-lang/triton), `transformers` (4.45 – 5.x), and
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
(`fla` ≥ 0.4.2, including 0.5.x), which supplies the backbone modules and shared kernel
utilities.

```bash
pip install -e .          # from this repo's root — pulls all dependencies
```

Tested stacks:

| Python | torch | triton | transformers | flash-linear-attention |
|---|---|---|---|---|
| 3.12 | 2.10 | 3.6 | 4.57 | 0.4.2 |
| 3.12 | 2.13 | 3.7.1 | 5.14.1 | 0.5.2 |

Notes:
- On Hopper (H100/H200) use triton ≥ 3.7.1; triton 3.6 rejects the gated `chunk_bwd_dqkwg`
  kernel used by the backbone.
- Transformers 5 and fla 0.5 need no extra steps; the package carries its own generation and
  serialization compatibility code (see the 2026-10-31 update below).
- To install into an existing environment without touching its torch/fla stack:
  `pip install --no-deps -e .`

## Usage

Importing `ccq` registers both model families with `transformers`. After
that, every framework that goes through `AutoConfig` / `AutoModelForCausalLM`
can build, train, and load CCQ models with no further glue:

```python
import ccq  # registers ccq_gdn / ccq_gla with HuggingFace

from transformers import AutoConfig, AutoModelForCausalLM

# build from one of the paper configs
config = AutoConfig.from_pretrained("configs/ccq_gdn_500M.json")
model = AutoModelForCausalLM.from_config(config)

# or load a trained checkpoint directory
model = AutoModelForCausalLM.from_pretrained("/path/to/checkpoint")
```

Generation works through the standard HF API (the recurrent state, the
running key second moment Σkkᵀ, and the running key sum are all carried in
the cache, so memory is O(1) in sequence length):

```python
out = model.generate(input_ids, max_new_tokens=64)
```

### Configs

The configurations used in the paper are in [`configs/`](configs/). The
`ccq_*` configs have `enable_ccq: true`; the `gdn_*` / `gla_*` configs are the
architecturally identical no-CCQ ablation baselines; `mamba2_*` and
`transformer_*` are the external baselines (these use fla's stock model
classes — add `import fla` to register them):

| config | params | model class |
|---|---:|---|
| `ccq_gdn_500M.json` / `gdn_500M.json` | 512.5M | CCQ-GDN (CCQ ✓ / ✗) |
| `ccq_gdn_1B.json` / `gdn_1B.json` | 1387.5M | CCQ-GDN (CCQ ✓ / ✗) |
| `ccq_gla_500M.json` / `gla_500M.json` | 513.1M | CCQ-GLA (CCQ ✓ / ✗) |
| `ccq_gla_1B.json` / `gla_1B.json` | 1388.3M | CCQ-GLA (CCQ ✓ / ✗) |
| `mamba2_500M.json` / `mamba2_1B.json` | 510.1M / 1371.8M | Mamba2 (fla) |
| `transformer_500M.json` / `transformer_1B.json` | 515.2M / 1364.3M | Transformer (fla) |

Because `enable_ccq` only toggles the read-side cleaning, the CCQ and
`gdn_*`/`gla_*` baseline variants share parameter counts and code paths —
flipping the flag is the entire ablation.

## Training

This repo intentionally ships **only the model implementation** — no trainer.
Because the models register with HuggingFace on import, you can pretrain them
with any framework that supports HF model classes: add `import ccq` near the
top of your training entry point and point it at one of the configs.

The experiments in the paper were trained with
[**flame**](https://github.com/fla-org/flame), the torchtitan-based training
framework of the `fla` ecosystem. A minimal flame invocation looks like:

```bash
# inside flame, after `import ccq` is reachable on PYTHONPATH
bash train.sh \
  --job.config_file flame/models/fla.toml \
  --model.config /path/to/ccq/configs/ccq_gdn_500M.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --training.dataset HuggingFaceFW/fineweb-edu \
  ...
```

See Appendix "Training Hyperparameters" of the paper for the full recipes.

## Repo structure

```
ccq/
├── gdn/                    # CCQ-GDN: config / layer / HF modeling
├── gla/                    # CCQ-GLA: config / layer / HF modeling
└── ops/
    ├── common/
    │   └── clean.py        # shared fused CCQ cleaning kernels (Triton):
    │                       #   chunk-parallel fwd (mean correction + cleaning),
    │                       #   chunk-parallel K-tiled bwd with exact gradients,
    │                       #   per-chunk prefix/suffix scan helpers
    ├── ccq_gdn/            # GDN orchestrator + recurrent decode path
    └── ccq_gla/            # GLA orchestrator + recurrent decode path
```

The Gated DeltaNet / GLA backbones are imported unmodified from
`flash-linear-attention`; this package adds only the CCQ read-side
cleaning. The cleaning operator runs as fused Triton kernels structured chunk-parallel
(one program per (chunk, batch·head)); the cross-chunk running key sum is the
only sequential dependency and is hoisted into a tiny per-chunk prefix
computed outside the kernel. Both training (chunkwise) and inference
(recurrent decode with cache handoff) paths share the same cleaning kernel
and are verified against a pure-PyTorch autograd reference.

## Update (2026-10-31): Transformers 5 / FLA 0.5 support and paper reproduction

### Compatibility

The code now runs unchanged on **Transformers 5.x** and **flash-linear-attention 0.5.x** as well
as on the versions the paper used (transformers 4.57, fla 0.4.2). Three fixes were needed; none
changes the model mathematics, weights or checkpoints:

1. **Generation / cache fix (important).** Transformers 5 passes a pre-allocated `DynamicCache`
   and no `cache_position` to the first decoding step. The fla ≤ 0.5.2 generation mixin slices the
   prompt from `len(cache)`, which is already positive for that empty cache, so `generate()` fed
   **only the last prompt token** and every greedy continuation was unconditional text (all
   needle-in-a-haystack scores were 0 while teacher-forced and manual cached decoding were
   correct). `ccq/generation.py` now slices by the number of tokens the recurrent cache has
   actually seen; both model classes use it. **If you evaluated generation tasks with
   Transformers 5 before this fix, re-run them.**
2. `_tied_weights_keys` is a target→source mapping on Transformers 5 and a list on 4.x
   (serialization metadata only; embeddings stay untied).
3. `USE_CUDA_GRAPH` is imported from `fla.utils` when present and defaults to `False` otherwise
   (fla 0.5 removed the public symbol; it only affects Triton autotuner capture).

Verified with torch 2.13, triton 3.7.1, transformers 5.14.1, fla 0.5.2 on H200: training with the
paper recipe, teacher-forced evaluation, `generate()` at batch 1 and batch 8 (identical outputs),
and the fused `attn_mode: "fused_chunk"` kernels.

### Reproducing the paper numbers

CCQ-GDN and GDN 500M re-trained and evaluated with the paper recipe (57,220 steps, 15B tokens) on the latest package versions: torch 2.13, triton 3.7.1, transformers 5.14.1, flash-linear-attention 0.5.2.

| 500M | LMB acc | PIQA | Hella | Wino | ARC-e | ARC-c | SIQA | BoolQ | Avg acc |
|---|---|---|---|---|---|---|---|---|---|
| GDN paper | 33.88 | 67.85 | 43.48 | 53.67 | 55.56 | 28.67 | 39.41 | 54.19 | 47.09 |
| GDN reproduction | 38.11 | 68.17 | 44.73 | 51.62 | 55.68 | 31.48 | 40.07 | 60.37 | 48.78 |
| CCQ-GDN paper | 38.19 | 68.28 | 44.61 | 53.35 | 56.14 | 30.55 | 40.23 | 55.96 | 48.41 |
| CCQ-GDN reproduction | 37.94 | 68.28 | 44.43 | 53.91 | 56.73 | 30.29 | 39.82 | 60.80 | 49.02 |

`Avg acc` is the mean of the eight accuracy columns.

| S-NIAH (acc %), 500M, paper / reproduction | 1K | 2K | 4K | 8K |
|---|---|---|---|---|
| task 1, GDN | 100 / 100.0 | 100 / 100.0 | 100 / 99.8 | 100 / 93.4 |
| task 1, CCQ-GDN | 100 / 100.0 | 100 / 100.0 | 100 / 100.0 | 99.6 / 100.0 |
| task 2, GDN | 100 / 100.0 | 100 / 100.0 | 84.6 / 87.2 | 24.8 / 40.0 |
| task 2, CCQ-GDN | 100 / 100.0 | 99.0 / 100.0 | 87.0 / 95.6 | 50.8 / 42.8 |
| task 3, GDN | 56.2 / 81.4 | 54.2 / 74.0 | 30.2 / 38.4 | – / – |
| task 3, CCQ-GDN | 67.6 / 82.0 | 53.4 / 73.0 | 33.6 / 38.6 | – / – |

## Fused cleaning kernels (`attn_mode: "fused_chunk"`)

`ccq.ops.common.fused_read` implements the covariance read `z_t = C_t q_t` and the
cleaning `q_t - lambda_t (z_t/pos_t - s_t m_t/pos_t^2)` as one forward and one backward
Triton kernel per chunk, instead of the generic `chunk_simple_gla(q, k, v=k)` read plus a
separate cleaning kernel and three generic backward kernels. The mathematics is unchanged;
`fused_chunk_ccq_gdn` / `fused_chunk_ccq_gla` have the same signatures and return values as
`chunk_ccq_gdn` / `chunk_ccq_gla`, and the layers select them with `attn_mode: "fused_chunk"`
(decode still uses the recurrent path). Both are verified against an FP64 reference, including
packed sequences, initial covariance/mean states, position offsets and the gradients flowing
into the final states. The fused backward keeps the covariance states in the activation dtype,
as the GDN backbone itself does for its own states. Measured at 32K tokens, 6 heads, K=256, V=512 (forward + backward, shared H200), the cleaning overhead over plain GDN drops from about 7.6 ms to 4.2–7.1 ms, i.e. 3–15% less kernel time for the memory operator; the training-step gain is a few percent.

## Citation

```bibtex
@inproceedings{le2026ccq,
  title     = {Don't Read Everything: A Curvature-Conditioned Query for Linear Attention},
  author    = {Le, Dong and Nguyen, Thong and Nguyen, Cong-Duy and Luu, Anh Tuan},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.01294},
}
```

## Acknowledgements

Built on [flash-linear-attention](https://github.com/fla-org/flash-linear-attention);
trained with [flame](https://github.com/fla-org/flame).
