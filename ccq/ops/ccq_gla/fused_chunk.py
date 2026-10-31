"""CCQ-GLA with the fused covariance read + cleaning (see ``ccq.ops.common.fused_read``).

Same mathematics and signature as :func:`chunk_ccq_gla`; only the cleaning
operator is implemented by one forward and one backward Triton kernel.
"""
from __future__ import annotations

import torch

from fla.modules.l2norm import l2norm
from fla.ops.gla.chunk import chunk_gla

from ccq.ops.common.fused_read import fused_ccq_clean_read


def fused_chunk_ccq_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lambda_h: torch.Tensor | float,
    g: torch.Tensor | None = None,
    scale: int | None = None,
    initial_state: torch.Tensor | None = None,
    initial_cov_state: torch.Tensor | None = None,
    initial_mean_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    position_offset: int = 0,
    subtract_mean: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    q = l2norm(q, eps=1e-6)
    k = l2norm(k, eps=1e-6)
    q_clean, final_cov_state, final_mean_state = fused_ccq_clean_read(
        q, k, lambda_h,
        position_offset=position_offset,
        initial_cov_state=initial_cov_state,
        initial_mean_state=initial_mean_state if subtract_mean else None,
        output_final_state=output_final_state,
        subtract_mean=subtract_mean,
        cu_seqlens=cu_seqlens,
    )
    o, final_state = chunk_gla(
        q=q_clean, k=k, v=v, g=g, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state, cu_seqlens=cu_seqlens,
    )
    return o, final_state, final_cov_state, final_mean_state
