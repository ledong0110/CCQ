import torch

from fla.modules.l2norm import l2norm
from fla.ops.gla.fused_recurrent import fused_recurrent_gla
# Shared cleaning kernel with the chunk training path
from ccq.ops.common.clean import _CCQCleanQ


def fused_recurrent_ccq_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lambda_h: torch.Tensor | float,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    scale: int | None = None,
    initial_state: torch.Tensor | None = None,
    initial_cov_state: torch.Tensor | None = None,
    initial_mean_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    position_offset: int = 0,
    subtract_mean: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    q = l2norm(q, eps=1e-6)
    k = l2norm(k, eps=1e-6)
    # z_t = (Sum_{j<=t} k_j k_j^T) q_t via the recurrent covariance read.
    z, final_cov_state = fused_recurrent_gla(
        q=q, k=k, v=k, scale=1.0,
        initial_state=initial_cov_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    q, final_mean_state = _CCQCleanQ.apply(
        q, k, z, lambda_h, position_offset, initial_mean_state,
        bool(output_final_state), bool(subtract_mean), cu_seqlens,
    )
    o, final_state = fused_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=gk,
        gv=gv,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state, final_cov_state, final_mean_state
