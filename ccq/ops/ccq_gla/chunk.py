import torch

from fla.ops.gla.chunk import chunk_gla
from fla.modules.l2norm import l2norm

# Reuse the same fused mean-correction-+-cleaning kernel + chunk-parallel
# bwd that ships with CCQ-GDN. The math is identical (q' = q - λ*(z/pos - μμᵀq))
# regardless of whether z came from a Gated DeltaNet or GLA backbone.
# _ccq_compute_z_and_sum_kk wraps chunk_simple_gla with a recurrent
# fallback for short sequences (T <= 64, e.g. cache decode).
from ccq.ops.common.clean import (
    _CCQCleanQ,
    _ccq_compute_z_and_sum_kk,
)


def chunk_ccq_gla(
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
    
    # 1) l2norm via fla triton (replaces F.normalize)
    q = l2norm(q, eps=1e-6)
    k = l2norm(k, eps=1e-6)

    # 2) z = (Sum kk^T) q  and  final Sum kk^T  (chunk_simple_gla, with a
    #    recurrent fallback for short sequences — same helper as CCQ-GDN)
    z, final_cov_state = _ccq_compute_z_and_sum_kk(
        q=q, k=k,
        initial_state=initial_cov_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    # 3) fused mean correction + cleaning (single triton kernel with tiled bwd)
    q, final_mean_state = _CCQCleanQ.apply(
        q, k, z, lambda_h, position_offset, initial_mean_state,
        output_final_state, bool(subtract_mean), cu_seqlens,
    )

    # 4) GLA output
    o, final_state = chunk_gla(
        q=q, k=k, v=v, g=g, scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state, final_cov_state, final_mean_state
