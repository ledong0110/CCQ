"""CCQ cleaning for the decode / short-sequence path (CCQ-GDN).

Computes the covariance read z via `chunk_simple_gla` (or the recurrent
`fused_recurrent_gla` for short T), then applies the shared fused
cleaning kernel `_CCQCleanQ` (ops.common.clean) and returns:

  * the cleaned query stream    ``q_clean = q - lambda * Sigma_t q_t``
  * the running second-moment   ``Sum_KK = sum_j k_j k_j^T``  (D, D)
  * the running first-moment    ``Sum_K  = sum_j k_j``        (D,)

Math reference (per b, h):
    C_t = sum_{j<=t} k_j k_j^T,   m_t = sum_{j<=t} k_j,   pos_t = t
    Sigma_t = C_t/t - (m_t/t)(m_t/t)^T
    z_t = C_t q_t
    q_clean_t = q_t - lambda_t * (z_t/t - m_t (m_t^T q_t)/t^2)

The backward (dq, dk, dz, dlambda) is the chunk-parallel K-tiled kernel
in ops.common.clean, verified against a pure-PyTorch autograd reference.
"""
from __future__ import annotations

import torch

from fla.modules.l2norm import l2norm
from fla.ops.simple_gla import chunk_simple_gla
from fla.ops.gla.fused_recurrent import fused_recurrent_gla
# Single source of truth for the cleaning op — same fused triton kernel
# (fwd + K-tiled bwd, verified against autograd ground truth) as the
# chunk training path. The previous triton_mean_correction-based path
# had a broken dk backward and has been retired.
from ccq.ops.common.clean import _CCQCleanQ


def fused_ccq_clean(
    q: torch.Tensor,
    k: torch.Tensor,
    lambda_h: torch.Tensor | float,
    *,
    initial_cov_state: torch.Tensor | None = None,
    initial_mean_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    position_offset: int = 0,
    subtract_mean: bool = True,
    use_chunk: bool = True,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]:
    """Forward + backward of the CCQ cleaning operator (decode/short-seq path).

    Returns
    -------
    q_clean, k, final_cov_state (= Sum_KK), final_mean_state (= Sum_K), use_qk_l2norm_in_kernel
    """
    if use_qk_l2norm_in_kernel:
        q = l2norm(q, eps=1e-6)
        k = l2norm(k, eps=1e-6)
        use_qk_l2norm_in_kernel = False

    T = q.shape[1]

    # z_t = (Sum_{j<=t} k_j k_j^T) q_t  and the final Sum_KK state.
    if use_chunk and T > chunk_size:
        z, final_cov_state = chunk_simple_gla(
            q=q, k=k, v=k, g=None, scale=1.0,
            initial_state=initial_cov_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
    else:
        z, final_cov_state = fused_recurrent_gla(
            q=q, k=k, v=k, scale=1.0,
            initial_state=initial_cov_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

    # Fused mean correction + cleaning (+ Sum_K state) in one triton kernel.
    q_clean, final_mean_state = _CCQCleanQ.apply(
        q, k, z, lambda_h, position_offset, initial_mean_state,
        bool(output_final_state), bool(subtract_mean), cu_seqlens,
    )
    return q_clean, k, final_cov_state, final_mean_state, use_qk_l2norm_in_kernel
