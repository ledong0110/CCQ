"""`fused_recurrent_ccq_gdn` — recurrent-mode counterpart of `chunk_ccq_gdn`.

Same `enable_ccq` switch:
* True  -> fused CCQ cleaning + gated-delta-rule recurrent step, returns
           the two CCQ running sums (Sum_KK, Sum_K) alongside the
           backbone state.
* False -> identity wrapper around `fused_recurrent_gated_delta_rule`,
           returns ``None`` for the two CCQ sums.
"""
from __future__ import annotations

import torch

from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from ccq.ops.ccq_gdn.fused_clean import fused_ccq_clean


def fused_recurrent_ccq_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lambda_h: torch.Tensor | float,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    initial_cov_state: torch.Tensor | None = None,
    initial_mean_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
    position_offset: int = 0,
    subtract_mean: bool = True,
    enable_ccq: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if not enable_ccq:
        o, final_state = fused_recurrent_gated_delta_rule(
            q=q, k=k, v=v, g=g, gk=gk, gv=gv, beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            transpose_state_layout=transpose_state_layout,
        )
        return o, final_state, None, None

    # The recurrent path always wants the per-token loop; chunkwise would
    # be wrong for the cache-decode case. Force `use_chunk=False`.
    q_clean, k, final_cov_state, final_mean_state, use_qk_l2norm_in_kernel = fused_ccq_clean(
        q=q,
        k=k,
        lambda_h=lambda_h,
        initial_cov_state=initial_cov_state,
        initial_mean_state=initial_mean_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        position_offset=position_offset,
        subtract_mean=subtract_mean,
        use_chunk=False,
    )
    o, final_state = fused_recurrent_gated_delta_rule(
        q=q_clean, k=k, v=v, g=g, gk=gk, gv=gv, beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        transpose_state_layout=transpose_state_layout,
    )
    return o, final_state, final_cov_state, final_mean_state
