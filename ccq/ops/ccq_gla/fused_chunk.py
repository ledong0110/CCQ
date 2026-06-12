import torch

from ccq.ops.ccq_gla.chunk import chunk_ccq_gla


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
    # Note: q, k are L2-normalised inside chunk_ccq_gla;
    # we deliberately do *not* normalise here to avoid a double pass.
    return chunk_ccq_gla(
        q=q,
        k=k,
        v=v,
        lambda_h=lambda_h,
        g=g,
        scale=scale,
        initial_state=initial_state,
        initial_cov_state=initial_cov_state,
        initial_mean_state=initial_mean_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        position_offset=position_offset,
        subtract_mean=subtract_mean,
    )
