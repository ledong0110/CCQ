# Don't Read Everything: A Curvature-Conditioned Query for Linear Attention

[![arXiv](https://img.shields.io/badge/arXiv-PDF-red?logo=arxiv)](https://arxiv.org/abs/2606.01294)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Paper-orange)](https://huggingface.co/papers/2606.01294)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

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
- [2026-06] Code released: model implementations, paper configs, and fused Triton kernels.
- [2026-06] Paper available on [arXiv](https://arxiv.org/abs/2606.01294).

## Installation

Requires Python ≥ 3.12, PyTorch ≥ 2.5 **with CUDA**,
[Triton](https://github.com/triton-lang/triton), `transformers` (< 5.0.0), and
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
(`fla` ≥ 0.4.2), which supplies the backbone modules and shared kernel
utilities.

```bash
pip install -e .          # from this repo's root — pulls all dependencies
```

Tested with: Python 3.12, torch 2.10 + CUDA 12.8, triton 3.6,
transformers 4.57, flash-linear-attention 0.4.2.

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

## Citation

```bibtex
@article{le2026ccq,
  title   = {Don't Read Everything: A Curvature-Conditioned Query for Linear Attention},
  author  = {Le, Dong and Nguyen, Thong and Nguyen, Cong-Duy and Luu, Anh-Tuan},
  journal = {arXiv preprint arXiv:2606.01294},
  year    = {2026},
}
```

## Acknowledgements

Built on [flash-linear-attention](https://github.com/fla-org/flash-linear-attention);
trained with [flame](https://github.com/fla-org/flame).
