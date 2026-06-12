from __future__ import annotations

import torch

from fla.modules.l2norm import l2norm
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.cp import FLACPContext

from ccq.ops.common.clean import _CCQCleanQ, _ccq_compute_z_and_sum_kk


def chunk_ccq_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lambda_h: torch.Tensor | float,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    initial_cov_state: torch.Tensor | None = None,
    initial_mean_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
    position_offset: int = 0,
    subtract_mean: bool = True,
    enable_ccq: bool = True,
    **kwargs,
):
    
    if not enable_ccq:
        o, final_state = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
            transpose_state_layout=transpose_state_layout,
            **kwargs,
        )
        return o, final_state, None, None

    if use_qk_l2norm_in_kernel:
        # fla's triton l2norm (fp32 accumulator, autograd-friendly).
        q = l2norm(q, eps=1e-6)
        k = l2norm(k, eps=1e-6)
        use_qk_l2norm_in_kernel = False

    # 1) z = Sigma_t q_t  and  Sum_KK = sum k_j k_j^T  (via simple_gla, fla impl).
    z, final_cov_state = _ccq_compute_z_and_sum_kk(
        q=q, k=k,
        initial_state=initial_cov_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    if not output_final_state:
        final_cov_state = None

    # 2) Fused mean correction + cleaning + Sum_K accumulation, one triton pass.
    q_clean, final_mean_state = _CCQCleanQ.apply(
        q, k, z, lambda_h, position_offset, initial_mean_state,
        bool(output_final_state),
        bool(subtract_mean),
        cu_seqlens,
    )

    # 3) Backbone gated-delta-rule attention with cleaned q.
    o, final_state = chunk_gated_delta_rule(
        q=q_clean, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        cp_context=cp_context,
        transpose_state_layout=transpose_state_layout,
        **kwargs,
    )
    return o, final_state, final_cov_state, final_mean_state
