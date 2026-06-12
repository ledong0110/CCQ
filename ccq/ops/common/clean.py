"""Shared CCQ cleaning kernels — used by both the GDN and GLA flavours.

Contains the chunk-parallel fused mean-correction + cleaning forward,
the K-tiled chunk-parallel backward (exact gradients, verified against
a pure-PyTorch autograd reference), the per-chunk prefix/suffix scan
helpers, and the covariance read z = (Σ k kᵀ) q.
"""

import torch
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm
from fla.ops.simple_gla import chunk_simple_gla
from fla.ops.gla.fused_recurrent import fused_recurrent_gla
from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import (
    USE_CUDA_GRAPH,
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    check_shared_mem,
    input_guard,
)

# ---------------------------------------------------------------------
# Chunk-PARALLEL cleaning kernels.
#
# The cleaning math has exactly one cross-chunk dependency: the running
# key sum cum_k at each chunk boundary (a (K,) vector per chunk). We
# hoist it out of the kernel: a torch-side fp32 cumsum + gather produces
# the EXCLUSIVE per-chunk prefix (already including any initial state),
# and every (chunk, batch·head) program then runs independently —
# grid (NT, B*H) instead of (B, H). At B=1, T=4096, H=6 this is 384
# programs instead of 6 (64x more parallelism); the old serial-over-T
# kernels ran at ~2% of HBM bandwidth on H200.
# ---------------------------------------------------------------------

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T', 'NT'])
def _chunk_block_sum_kernel(
    x_ptr, out_ptr,
    cu_seqlens, chunk_indices,
    T, NT,
    stride_xb, stride_xt, stride_xh,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """out[chunk] = sum of x over the chunk's rows (fp32). Grid (NT, B*H)."""
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_tl = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T_local = eos - bos
        t_start = i_tl * BT
        row = i_t
        base = x_ptr + bos * stride_xt + i_h * stride_xh
    else:
        T_local = T
        t_start = i_t * BT
        row = i_b * NT + i_t
        base = x_ptr + i_b * stride_xb + i_h * stride_xh
    offs_k = tl.arange(0, BK)
    mask_k = offs_k < K
    offs_t = t_start + tl.arange(0, BT)
    mask_t = offs_t < T_local
    x = tl.load(base + offs_t[:, None] * stride_xt + offs_k[None, :],
                mask=mask_t[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
    s = tl.sum(tl.where(mask_t[:, None], x, 0.0), axis=0)
    tl.store(out_ptr + (row * H + i_h) * K + offs_k, s, mask=mask_k)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T', 'NT'])
def _suffix_apply_kernel(
    src_ptr, carry_ptr, dst_ptr,
    cu_seqlens, chunk_indices,
    T, NT,
    stride_sb, stride_st, stride_sh,
    stride_db, stride_dt, stride_dh,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """dst[t] = within-chunk suffix of src + carry[chunk]. Grid (NT, B*H).

    carry[chunk] = sum of src over all LATER chunks in the same segment
    (precomputed torch-side from per-chunk totals)."""
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_tl = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T_local = eos - bos
        t_start = i_tl * BT
        row = i_t
        s_base = src_ptr + bos * stride_st + i_h * stride_sh
        d_base = dst_ptr + bos * stride_dt + i_h * stride_dh
    else:
        T_local = T
        t_start = i_t * BT
        row = i_b * NT + i_t
        s_base = src_ptr + i_b * stride_sb + i_h * stride_sh
        d_base = dst_ptr + i_b * stride_db + i_h * stride_dh
    offs_k = tl.arange(0, BK)
    mask_k = offs_k < K
    offs_t = t_start + tl.arange(0, BT)
    mask_t = offs_t < T_local
    td = mask_t[:, None] & mask_k[None, :]
    x = tl.load(s_base + offs_t[:, None] * stride_st + offs_k[None, :], mask=td, other=0.0).to(tl.float32)
    x = tl.where(mask_t[:, None], x, 0.0)
    total = tl.sum(x, axis=0)
    suffix = total[None, :] - tl.cumsum(x, axis=0) + x
    carry = tl.load(carry_ptr + (row * H + i_h) * K + offs_k, mask=mask_k, other=0.0)
    tl.store(d_base + offs_t[:, None] * stride_dt + offs_k[None, :],
             (suffix + carry[None, :]).to(dst_ptr.dtype.element_ty), mask=td)


def _chunk_block_sums(x, BT, cu_seqlens, chunk_indices, NT):
    """Per-chunk fp32 sums of x over time: (NTG, H, K)."""
    B, T, H, K = x.shape
    NTG = B * NT if cu_seqlens is None else len(chunk_indices)
    out = torch.empty(NTG, H, K, device=x.device, dtype=torch.float32)
    _chunk_block_sum_kernel[(NT, B * H)](
        x, out, cu_seqlens, chunk_indices,
        T, NT if cu_seqlens is None else 0,
        x.stride(0), x.stride(1), x.stride(2),
        H=H, K=K, BT=BT, BK=triton.next_power_of_2(K),
    )
    return out


def _cumk_chunk_prefix(k, BT, cu_seqlens, chunk_indices, initial_state_K, need_final):
    """Exclusive per-chunk prefix of the running key sum, fp32.

    Returns (prefix, final):
      prefix: (NTG, H, K) fp32 — sum of k over all positions before each
              chunk's start (within its segment), plus initial_state_K.
      final : (B_or_Nseg, H, K) fp32 final running key sum, or None.
    Only tiny (NTG, H, K) tensors are scanned torch-side; the per-chunk
    reduction over T runs as a parallel triton kernel.
    """
    B, T, H, K = k.shape
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    sums = _chunk_block_sums(k, BT, cu_seqlens, chunk_indices, NT)   # (NTG, H, K)
    if cu_seqlens is None:
        sv = sums.view(B, NT, H, K)
        prefix = sv.cumsum(1) - sv                                   # exclusive
        if initial_state_K is not None:
            prefix = prefix + initial_state_K.float()[:, None]
        final = None
        if need_final:
            final = sv.sum(1)
            if initial_state_K is not None:
                final = final + initial_state_K.float()
        return prefix.reshape(B * NT, H, K), final
    # varlen: exclusive cumsum within each segment's chunk range.
    cs = sums.cumsum(0)
    ex = cs - sums                                                   # global exclusive
    co = prepare_chunk_offsets(cu_seqlens, BT).long()                # (Nseg+1,) chunk-count offsets
    i_n = chunk_indices[:, 0].long()
    prefix = ex - ex.index_select(0, co[i_n])
    if initial_state_K is not None:
        prefix = prefix + initial_state_K.float()[i_n]
    final = None
    if need_final:
        last = co[1:] - 1
        final = cs.index_select(0, last) - ex.index_select(0, co[:-1])
        if initial_state_K is not None:
            final = final + initial_state_K.float()
    return prefix.contiguous(), final


def _rev_cumsum_time(x, BT, cu_seqlens, chunk_indices, out_dtype):
    """Segment-aware suffix sum along time: out[t] = sum_{s>=t, same seg} x[s].

    Per-chunk totals (parallel kernel) -> tiny torch suffix scan over
    chunk totals -> parallel apply kernel. x is fp32 (B, T, H, K)."""
    B, T, H, K = x.shape
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    totals = _chunk_block_sums(x, BT, cu_seqlens, chunk_indices, NT)  # (NTG, H, K)
    if cu_seqlens is None:
        tv = totals.view(B, NT, H, K)
        inc = tv.cumsum(1)
        carry = (inc[:, -1:] - inc).reshape(B * NT, H, K)             # later chunks only
    else:
        cs = totals.cumsum(0)
        co = prepare_chunk_offsets(cu_seqlens, BT).long()
        i_n = chunk_indices[:, 0].long()
        seg_last = co[1:] - 1
        carry = cs.index_select(0, seg_last[i_n]) - cs                # later chunks in segment
    out = torch.empty(B, T, H, K, device=x.device, dtype=out_dtype)
    _suffix_apply_kernel[(NT, B * H)](
        x, carry.contiguous(), out,
        cu_seqlens, chunk_indices,
        T, NT if cu_seqlens is None else 0,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        H=H, K=K, BT=BT, BK=triton.next_power_of_2(K),
    )
    return out


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['K', 'BT', 'SUBTRACT_MEAN', 'IS_VARLEN'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'NT', 'position_offset'])
def _ccq_clean_q_fwd_kernel(
    q_ptr, k_ptr, z_ptr, lam_ptr, out_ptr,
    prefix_ptr,
    cu_seqlens, chunk_indices,
    T, NT, position_offset,
    stride_qb, stride_qt, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_zb, stride_zt, stride_zh,
    stride_lb, stride_lt, stride_lh,
    stride_ob, stride_ot, stride_oh,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Fused mean-correction + cleaning, one program per (chunk, batch*head).

    The cross-chunk running key sum arrives precomputed via prefix_ptr
    (fp32, exclusive at chunk start, includes any initial state), so
    programs have no sequential dependency.
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_tl = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T_local = eos - bos
        t_start = i_tl * BT
        prefix_row = i_t
        q_base = q_ptr + bos * stride_qt + i_h * stride_qh
        k_base = k_ptr + bos * stride_kt + i_h * stride_kh
        z_base = z_ptr + bos * stride_zt + i_h * stride_zh
        o_base = out_ptr + bos * stride_ot + i_h * stride_oh
        lam_base = lam_ptr + bos * stride_lt + i_h * stride_lh
    else:
        T_local = T
        t_start = i_t * BT
        prefix_row = i_b * NT + i_t
        q_base = q_ptr + i_b * stride_qb + i_h * stride_qh
        k_base = k_ptr + i_b * stride_kb + i_h * stride_kh
        z_base = z_ptr + i_b * stride_zb + i_h * stride_zh
        o_base = out_ptr + i_b * stride_ob + i_h * stride_oh
        lam_base = lam_ptr + i_b * stride_lb + i_h * stride_lh

    offs_k = tl.arange(0, BK)
    mask_k = offs_k < K
    offs_t = t_start + tl.arange(0, BT)
    mask_t = offs_t < T_local
    td_mask = mask_t[:, None] & mask_k[None, :]

    q_chunk = tl.load(q_base + offs_t[:, None] * stride_qt + offs_k[None, :], mask=td_mask, other=0.0).to(tl.float32)
    z_chunk = tl.load(z_base + offs_t[:, None] * stride_zt + offs_k[None, :], mask=td_mask, other=0.0).to(tl.float32)
    lam_chunk = tl.load(lam_base + offs_t * stride_lt, mask=mask_t, other=0.0).to(tl.float32)

    # position is segment-RELATIVE so the mean correction is correct per doc.
    pos = (offs_t + 1 + position_offset).to(tl.float32)
    pos = tl.where(pos < 1.0, 1.0, pos)
    inv_pos = 1.0 / pos

    correction = z_chunk * inv_pos[:, None]
    if SUBTRACT_MEAN:
        k_chunk = tl.load(k_base + offs_t[:, None] * stride_kt + offs_k[None, :], mask=td_mask, other=0.0).to(tl.float32)
        state_k = tl.load(prefix_ptr + (prefix_row * H + i_h) * K + offs_k, mask=mask_k, other=0.0)
        cum_k = state_k[None, :] + tl.cumsum(k_chunk, axis=0)
        inner = tl.sum(cum_k * q_chunk, axis=1)
        mean_corr = (inner * inv_pos * inv_pos)[:, None] * cum_k
        correction = correction - mean_corr

    q_clean = q_chunk - lam_chunk[:, None] * correction
    tl.store(
        o_base + offs_t[:, None] * stride_ot + offs_k[None, :],
        q_clean.to(out_ptr.dtype.element_ty),
        mask=td_mask,
    )


def _reshape_lambda_3d(lam, q):
    """Coerce lambda to shape (B, T, H), broadcasting as needed."""
    B, T, H = q.shape[:3]
    if not isinstance(lam, torch.Tensor):
        return torch.full((B, T, H), float(lam), device=q.device, dtype=q.dtype)
    if lam.ndim == 0:
        return lam.expand(B, T, H).contiguous().to(q.dtype)
    if lam.ndim == 1 and lam.shape[0] == H:
        return lam.view(1, 1, H).expand(B, T, H).contiguous().to(q.dtype)
    if lam.ndim == 4 and lam.shape[-1] == 1:
        return lam.squeeze(-1).contiguous().to(q.dtype)
    return lam.to(q.dtype).contiguous()


def _reshape_lambda_4d(lam, q):
    """Coerce lambda to shape (B, T, H, 1) for broadcasting against (B,T,H,D)."""
    if not isinstance(lam, torch.Tensor):
        return float(lam)
    if lam.ndim == 0:
        return lam
    if lam.ndim == 1:
        return lam.view(*([1] * (q.ndim - 2)), -1, 1)
    if lam.ndim == q.ndim - 1:
        return lam.unsqueeze(-1)
    return lam


def _ccq_clean_q_triton_fwd(
    q, k, z, lambda_h, position_offset, initial_state_K,
    output_final_state, subtract_mean, cu_seqlens=None,
):
    B, T, H, K = q.shape
    BK = triton.next_power_of_2(K)
    BT = 64

    q_c = q.contiguous()
    k_c = k.contiguous()
    z_c = z.contiguous()
    lam_3d = _reshape_lambda_3d(lambda_h, q)
    out = torch.empty_like(q_c)

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Per-chunk exclusive cum-k prefix (torch-side; includes initial state)
    # + the final running key sum if requested.
    need_final = bool(output_final_state and subtract_mean)
    if subtract_mean or need_final:
        prefix, final_state_K = _cumk_chunk_prefix(
            k_c, BT, cu_seqlens, chunk_indices, initial_state_K, need_final,
        )
    else:
        prefix, final_state_K = torch.empty(0, device=q.device, dtype=torch.float32), None

    _ccq_clean_q_fwd_kernel[(NT, B * H)](
        q_c, k_c, z_c, lam_3d, out,
        prefix,
        cu_seqlens, chunk_indices,
        T, NT if cu_seqlens is None else 0, position_offset,
        q_c.stride(0), q_c.stride(1), q_c.stride(2),
        k_c.stride(0), k_c.stride(1), k_c.stride(2),
        z_c.stride(0), z_c.stride(1), z_c.stride(2),
        lam_3d.stride(0), lam_3d.stride(1), lam_3d.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        H=H,
        K=K,
        BT=BT,
        BK=BK,
        SUBTRACT_MEAN=bool(subtract_mean),
    )
    return out, final_state_K


# ---------------------------------------------------------------------
# Triton backward kernels for _CCQCleanQ — pure-triton, K-tiled.
#
# The bwd has K-wide reductions (s_t = <cum_k_t, q_t>, d_s_t = <cum_k_t,
# dq_clean_t>, dqc_z_t = <dq_clean_t, z_t>) that block trivial K
# tiling. Solution: two passes over K per chunk —
#   Pass 1: loop K tiles, accumulate the three per-token scalars
#           (BT scalars each) and the per-tile cum_k state.
#   Pass 2: loop K tiles, use the now-known scalars to emit dq, dz,
#           dcum_k per tile (no cross-tile dependency).
# The per-K-tile cum_k at chunk start arrives precomputed (prefix_ptr,
# torch-side cumsum+gather), so programs are independent across chunks:
# grid (NT, B*H). After the kernel, a torch-side segment-aware reverse
# cumsum turns dcum_k -> dk.
# BK_TILE = 64 fixed (matches the GDN K-block tiling); num_warps and
# num_stages are autotuned.
# ---------------------------------------------------------------------

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([3, 2] if check_shared_mem('ampere') else [2, 1])
    ],
    key=['K', 'BT', 'SUBTRACT_MEAN', 'IS_VARLEN'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'NT', 'position_offset'])
def _ccq_clean_q_bwd_fwd_pass_kernel(
    q_ptr, k_ptr, z_ptr, lam_ptr,
    dq_clean_ptr,
    prefix_ptr,
    dq_ptr, dz_ptr, dlam_ptr, dcum_k_ptr,
    cu_seqlens, chunk_indices,
    T, NT, position_offset,
    stride_qb, stride_qt, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_zb, stride_zt, stride_zh,
    stride_lb, stride_lt, stride_lh,
    stride_dcb, stride_dct, stride_dch,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Chunk-parallel bwd emit. One program per (chunk, batch*head).

    For each token t with cum_k_t = prefix(chunk) + intra-chunk cumsum:
        s_t   = cum_k_t . q_t
        d_s_t = cum_k_t . dq_clean_t
        f_t   = lambda_t / pos_t^2
        dq_t  = dq_clean_t + f_t * cum_k_t * d_s_t
        dz_t  = -lambda_t / pos_t * dq_clean_t
        dl_t  = -<dq_clean_t, z_t/pos_t - cum_k_t s_t/pos_t^2>
        dcum_k_t = f_t * (s_t * dq_clean_t + d_s_t * q_t)
    dk is then a (torch-side) reverse cumsum of dcum_k within segments.
    Two passes over K tiles per chunk: pass 1 accumulates the per-token
    scalars; pass 2 emits per tile.
    """
    BK_TILE: tl.constexpr = 64
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_tl = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T_local = eos - bos
        t_start = i_tl * BT
        prefix_row = i_t
        q_base    = q_ptr        + bos * stride_qt + i_h * stride_qh
        k_base    = k_ptr        + bos * stride_kt + i_h * stride_kh
        z_base    = z_ptr        + bos * stride_zt + i_h * stride_zh
        dqc_base  = dq_clean_ptr + bos * stride_qt + i_h * stride_qh
        dq_base   = dq_ptr       + bos * stride_qt + i_h * stride_qh
        dz_base   = dz_ptr       + bos * stride_qt + i_h * stride_qh
        lam_base  = lam_ptr      + bos * stride_lt + i_h * stride_lh
        dlam_base = dlam_ptr     + bos * stride_lt + i_h * stride_lh
        dcum_base = dcum_k_ptr   + bos * stride_dct + i_h * stride_dch
    else:
        T_local = T
        t_start = i_t * BT
        prefix_row = i_b * NT + i_t
        q_base    = q_ptr        + i_b * stride_qb + i_h * stride_qh
        k_base    = k_ptr        + i_b * stride_kb + i_h * stride_kh
        z_base    = z_ptr        + i_b * stride_zb + i_h * stride_zh
        dqc_base  = dq_clean_ptr + i_b * stride_qb + i_h * stride_qh
        dq_base   = dq_ptr       + i_b * stride_qb + i_h * stride_qh
        dz_base   = dz_ptr       + i_b * stride_qb + i_h * stride_qh
        lam_base  = lam_ptr      + i_b * stride_lb + i_h * stride_lh
        dlam_base = dlam_ptr     + i_b * stride_lb + i_h * stride_lh
        dcum_base = dcum_k_ptr   + i_b * stride_dcb + i_h * stride_dch

    offs_t = t_start + tl.arange(0, BT)
    mask_t = offs_t < T_local

    pos = (offs_t + 1 + position_offset).to(tl.float32)
    pos = tl.where(pos < 1.0, 1.0, pos)
    inv_pos = 1.0 / pos
    inv_pos_sq = inv_pos * inv_pos

    lam_chunk = tl.load(lam_base + offs_t * stride_lt, mask=mask_t, other=0.0).to(tl.float32)

    # ---- per-K-tile cum_k prefix at this chunk's start (fp32, incl. init) ----
    pk_base = prefix_ptr + (prefix_row * H + i_h) * K
    ok_1 = tl.arange(0, BK_TILE)
    mk_1 = ok_1 < K
    td_1 = mask_t[:, None] & mk_1[None, :]
    if SUBTRACT_MEAN:
        cum_k_state_1 = tl.load(pk_base + ok_1, mask=mk_1, other=0.0)
    if K > 64:
        ok_2 = 64 + tl.arange(0, BK_TILE)
        mk_2 = ok_2 < K
        td_2 = mask_t[:, None] & mk_2[None, :]
        if SUBTRACT_MEAN:
            cum_k_state_2 = tl.load(pk_base + ok_2, mask=mk_2, other=0.0)
    if K > 128:
        ok_3 = 128 + tl.arange(0, BK_TILE)
        mk_3 = ok_3 < K
        td_3 = mask_t[:, None] & mk_3[None, :]
        if SUBTRACT_MEAN:
            cum_k_state_3 = tl.load(pk_base + ok_3, mask=mk_3, other=0.0)
    if K > 192:
        ok_4 = 192 + tl.arange(0, BK_TILE)
        mk_4 = ok_4 < K
        td_4 = mask_t[:, None] & mk_4[None, :]
        if SUBTRACT_MEAN:
            cum_k_state_4 = tl.load(pk_base + ok_4, mask=mk_4, other=0.0)

    # ============= Pass 1: K-wide reductions =============
    s_t     = tl.zeros((BT,), dtype=tl.float32)
    d_s_t   = tl.zeros((BT,), dtype=tl.float32)
    dqc_z_t = tl.zeros((BT,), dtype=tl.float32)

    q1_p1   = tl.load(q_base   + offs_t[:, None] * stride_qt + ok_1[None, :], mask=td_1, other=0.0).to(tl.float32)
    dqc1_p1 = tl.load(dqc_base + offs_t[:, None] * stride_qt + ok_1[None, :], mask=td_1, other=0.0).to(tl.float32)
    z1_p1   = tl.load(z_base   + offs_t[:, None] * stride_zt + ok_1[None, :], mask=td_1, other=0.0).to(tl.float32)
    dqc_z_t += tl.sum(dqc1_p1 * z1_p1, axis=1)
    if SUBTRACT_MEAN:
        k1_p1 = tl.load(k_base + offs_t[:, None] * stride_kt + ok_1[None, :], mask=td_1, other=0.0).to(tl.float32)
        ck1 = cum_k_state_1[None, :] + tl.cumsum(k1_p1, axis=0)
        s_t   += tl.sum(ck1 * q1_p1,   axis=1)
        d_s_t += tl.sum(ck1 * dqc1_p1, axis=1)

    if K > 64:
        q2_p1   = tl.load(q_base   + offs_t[:, None] * stride_qt + ok_2[None, :], mask=td_2, other=0.0).to(tl.float32)
        dqc2_p1 = tl.load(dqc_base + offs_t[:, None] * stride_qt + ok_2[None, :], mask=td_2, other=0.0).to(tl.float32)
        z2_p1   = tl.load(z_base   + offs_t[:, None] * stride_zt + ok_2[None, :], mask=td_2, other=0.0).to(tl.float32)
        dqc_z_t += tl.sum(dqc2_p1 * z2_p1, axis=1)
        if SUBTRACT_MEAN:
            k2_p1 = tl.load(k_base + offs_t[:, None] * stride_kt + ok_2[None, :], mask=td_2, other=0.0).to(tl.float32)
            ck2 = cum_k_state_2[None, :] + tl.cumsum(k2_p1, axis=0)
            s_t   += tl.sum(ck2 * q2_p1,   axis=1)
            d_s_t += tl.sum(ck2 * dqc2_p1, axis=1)

    if K > 128:
        q3_p1   = tl.load(q_base   + offs_t[:, None] * stride_qt + ok_3[None, :], mask=td_3, other=0.0).to(tl.float32)
        dqc3_p1 = tl.load(dqc_base + offs_t[:, None] * stride_qt + ok_3[None, :], mask=td_3, other=0.0).to(tl.float32)
        z3_p1   = tl.load(z_base   + offs_t[:, None] * stride_zt + ok_3[None, :], mask=td_3, other=0.0).to(tl.float32)
        dqc_z_t += tl.sum(dqc3_p1 * z3_p1, axis=1)
        if SUBTRACT_MEAN:
            k3_p1 = tl.load(k_base + offs_t[:, None] * stride_kt + ok_3[None, :], mask=td_3, other=0.0).to(tl.float32)
            ck3 = cum_k_state_3[None, :] + tl.cumsum(k3_p1, axis=0)
            s_t   += tl.sum(ck3 * q3_p1,   axis=1)
            d_s_t += tl.sum(ck3 * dqc3_p1, axis=1)

    if K > 192:
        q4_p1   = tl.load(q_base   + offs_t[:, None] * stride_qt + ok_4[None, :], mask=td_4, other=0.0).to(tl.float32)
        dqc4_p1 = tl.load(dqc_base + offs_t[:, None] * stride_qt + ok_4[None, :], mask=td_4, other=0.0).to(tl.float32)
        z4_p1   = tl.load(z_base   + offs_t[:, None] * stride_zt + ok_4[None, :], mask=td_4, other=0.0).to(tl.float32)
        dqc_z_t += tl.sum(dqc4_p1 * z4_p1, axis=1)
        if SUBTRACT_MEAN:
            k4_p1 = tl.load(k_base + offs_t[:, None] * stride_kt + ok_4[None, :], mask=td_4, other=0.0).to(tl.float32)
            ck4 = cum_k_state_4[None, :] + tl.cumsum(k4_p1, axis=0)
            s_t   += tl.sum(ck4 * q4_p1,   axis=1)
            d_s_t += tl.sum(ck4 * dqc4_p1, axis=1)

    # ---- dlam (per-token reduction, complete now that pass 1 is done) ----
    if SUBTRACT_MEAN:
        dlam_chunk = -dqc_z_t * inv_pos + s_t * d_s_t * inv_pos_sq
    else:
        dlam_chunk = -dqc_z_t * inv_pos
    tl.store(dlam_base + offs_t * stride_lt, dlam_chunk.to(dlam_ptr.dtype.element_ty), mask=mask_t)

    f_t = lam_chunk * inv_pos_sq
    lam_invpos = lam_chunk * inv_pos

    # ============= Pass 2: per-tile emit dq, dz, dcum_k =============
    # Re-uses pass-1 register tiles where alive; reloads hit L2.
    if SUBTRACT_MEAN:
        ck1 = cum_k_state_1[None, :] + tl.cumsum(k1_p1, axis=0)
        dq1 = dqc1_p1 + (f_t * d_s_t)[:, None] * ck1
        dcum1 = (f_t * s_t)[:, None] * dqc1_p1 + (f_t * d_s_t)[:, None] * q1_p1
        tl.store(dcum_base + offs_t[:, None] * stride_dct + ok_1[None, :],
                 dcum1.to(dcum_k_ptr.dtype.element_ty), mask=td_1)
    else:
        dq1 = dqc1_p1
    dz1 = -lam_invpos[:, None] * dqc1_p1
    tl.store(dq_base + offs_t[:, None] * stride_qt + ok_1[None, :], dq1.to(dq_ptr.dtype.element_ty), mask=td_1)
    tl.store(dz_base + offs_t[:, None] * stride_qt + ok_1[None, :], dz1.to(dz_ptr.dtype.element_ty), mask=td_1)

    if K > 64:
        if SUBTRACT_MEAN:
            ck2 = cum_k_state_2[None, :] + tl.cumsum(k2_p1, axis=0)
            dq2 = dqc2_p1 + (f_t * d_s_t)[:, None] * ck2
            dcum2 = (f_t * s_t)[:, None] * dqc2_p1 + (f_t * d_s_t)[:, None] * q2_p1
            tl.store(dcum_base + offs_t[:, None] * stride_dct + ok_2[None, :],
                     dcum2.to(dcum_k_ptr.dtype.element_ty), mask=td_2)
        else:
            dq2 = dqc2_p1
        dz2 = -lam_invpos[:, None] * dqc2_p1
        tl.store(dq_base + offs_t[:, None] * stride_qt + ok_2[None, :], dq2.to(dq_ptr.dtype.element_ty), mask=td_2)
        tl.store(dz_base + offs_t[:, None] * stride_qt + ok_2[None, :], dz2.to(dz_ptr.dtype.element_ty), mask=td_2)

    if K > 128:
        if SUBTRACT_MEAN:
            ck3 = cum_k_state_3[None, :] + tl.cumsum(k3_p1, axis=0)
            dq3 = dqc3_p1 + (f_t * d_s_t)[:, None] * ck3
            dcum3 = (f_t * s_t)[:, None] * dqc3_p1 + (f_t * d_s_t)[:, None] * q3_p1
            tl.store(dcum_base + offs_t[:, None] * stride_dct + ok_3[None, :],
                     dcum3.to(dcum_k_ptr.dtype.element_ty), mask=td_3)
        else:
            dq3 = dqc3_p1
        dz3 = -lam_invpos[:, None] * dqc3_p1
        tl.store(dq_base + offs_t[:, None] * stride_qt + ok_3[None, :], dq3.to(dq_ptr.dtype.element_ty), mask=td_3)
        tl.store(dz_base + offs_t[:, None] * stride_qt + ok_3[None, :], dz3.to(dz_ptr.dtype.element_ty), mask=td_3)

    if K > 192:
        if SUBTRACT_MEAN:
            ck4 = cum_k_state_4[None, :] + tl.cumsum(k4_p1, axis=0)
            dq4 = dqc4_p1 + (f_t * d_s_t)[:, None] * ck4
            dcum4 = (f_t * s_t)[:, None] * dqc4_p1 + (f_t * d_s_t)[:, None] * q4_p1
            tl.store(dcum_base + offs_t[:, None] * stride_dct + ok_4[None, :],
                     dcum4.to(dcum_k_ptr.dtype.element_ty), mask=td_4)
        else:
            dq4 = dqc4_p1
        dz4 = -lam_invpos[:, None] * dqc4_p1
        tl.store(dq_base + offs_t[:, None] * stride_qt + ok_4[None, :], dq4.to(dq_ptr.dtype.element_ty), mask=td_4)
        tl.store(dz_base + offs_t[:, None] * stride_qt + ok_4[None, :], dz4.to(dz_ptr.dtype.element_ty), mask=td_4)


def _ccq_clean_q_bwd_triton(
    q, k, z, lambda_h, dq_clean,
    position_offset, initial_state_K,
    subtract_mean, cu_seqlens=None,
):
    """Chunk-parallel triton bwd. Returns (dq, dk, dz, dlam_tensor_or_None)."""
    B, T, H, K = q.shape
    BT = 64

    q_c   = q.contiguous()
    k_c   = k.contiguous()
    z_c   = z.contiguous()
    dqc_c = dq_clean.contiguous()
    lam_3d = _reshape_lambda_3d(lambda_h, q)

    dq = torch.empty_like(q_c)
    dz = torch.empty_like(z_c)
    dlam = torch.empty_like(lam_3d)
    # fp32 dcum_k so the torch-side reverse cumsum accumulates in fp32
    # (the old serial kernel carried fp32 across chunks; keep that accuracy).
    dcum_k = (torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
              if subtract_mean else torch.empty(0, device=q.device))

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if subtract_mean:
        prefix, _ = _cumk_chunk_prefix(k_c, BT, cu_seqlens, chunk_indices, initial_state_K, False)
    else:
        prefix = torch.empty(0, device=q.device, dtype=torch.float32)

    _ccq_clean_q_bwd_fwd_pass_kernel[(NT, B * H)](
        q_c, k_c, z_c, lam_3d,
        dqc_c,
        prefix,
        dq, dz, dlam, dcum_k,
        cu_seqlens, chunk_indices,
        T, NT if cu_seqlens is None else 0, position_offset,
        q_c.stride(0), q_c.stride(1), q_c.stride(2),
        k_c.stride(0), k_c.stride(1), k_c.stride(2),
        z_c.stride(0), z_c.stride(1), z_c.stride(2),
        lam_3d.stride(0), lam_3d.stride(1), lam_3d.stride(2),
        dcum_k.stride(0) if subtract_mean else 0,
        dcum_k.stride(1) if subtract_mean else 0,
        dcum_k.stride(2) if subtract_mean else 0,
        H=H, K=K, BT=BT,
        SUBTRACT_MEAN=bool(subtract_mean),
    )

    if subtract_mean:
        # dk_j = sum_{t >= j, same segment} dcum_k_t — chunk-parallel suffix.
        dk = _rev_cumsum_time(dcum_k, BT, cu_seqlens, chunk_indices, k_c.dtype)
    else:
        dk = torch.zeros_like(k_c)

    return dq, dk, dz, dlam


class _CCQCleanQ(torch.autograd.Function):
    """Forward AND backward in pure triton — no torch-recomputation."""
    @staticmethod
    @autocast_custom_fwd
    @input_guard
    def forward(ctx, q, k, z, lambda_h, position_offset, initial_state_K,
                output_final_state, subtract_mean, cu_seqlens):
        q_clean, final_state_K = _ccq_clean_q_triton_fwd(
            q, k, z, lambda_h, position_offset, initial_state_K,
            output_final_state, subtract_mean, cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(
            q, k, z,
            lambda_h if isinstance(lambda_h, torch.Tensor) else torch.empty(0, device=q.device),
            initial_state_K if initial_state_K is not None else torch.empty(0, device=q.device),
            cu_seqlens if cu_seqlens is not None else torch.empty(0, device=q.device, dtype=torch.long),
        )
        ctx.lambda_is_tensor = isinstance(lambda_h, torch.Tensor)
        ctx.lambda_scalar = None if ctx.lambda_is_tensor else float(lambda_h)
        ctx.has_init_K = initial_state_K is not None
        ctx.has_cu_seqlens = cu_seqlens is not None
        ctx.position_offset = position_offset
        ctx.subtract_mean = subtract_mean
        ctx.output_final_state = output_final_state
        return q_clean, final_state_K

    @staticmethod
    @autocast_custom_bwd
    @input_guard
    def backward(ctx, dq_clean, d_final_state_K):
        q, k, z, lam_t, init_K_t, cu_t = ctx.saved_tensors
        init_K = init_K_t if ctx.has_init_K else None
        cu_seqlens = cu_t if ctx.has_cu_seqlens else None
        lam_for_bwd = lam_t if ctx.lambda_is_tensor else ctx.lambda_scalar

        dq, dk, dz, dlam_3d = _ccq_clean_q_bwd_triton(
            q, k, z, lam_for_bwd, dq_clean,
            ctx.position_offset, init_K, ctx.subtract_mean,
            cu_seqlens=cu_seqlens,
        )

        dq = dq if q.requires_grad else None
        dk = dk if k.requires_grad else None
        dz = dz if z.requires_grad else None
        if ctx.lambda_is_tensor:
            # Reshape dlam_3d back to the original lambda_h shape.
            if lam_t.ndim == 0:
                dl = dlam_3d.sum() if lam_t.requires_grad else None
            elif lam_t.ndim == 1:                          # per-head (H,)
                dl = dlam_3d.sum(dim=(0, 1)) if lam_t.requires_grad else None
            elif lam_t.ndim == 4 and lam_t.shape[-1] == 1: # (B, T, H, 1)
                dl = dlam_3d.unsqueeze(-1) if lam_t.requires_grad else None
            else:
                dl = dlam_3d if lam_t.requires_grad else None
        else:
            dl = None
        # 9 inputs to forward → 9 grad outputs
        return dq, dk, dz, dl, None, None, None, None, None


def _ccq_compute_z_and_sum_kk(q, k, output_final_state, cu_seqlens,
                              initial_state=None, use_chunk=True, chunk_size=64):
    """Compute z = (sum_{j<=t} k_j k_j^T) q_t and the final Sum_KK = sum_j k_j k_j^T.

    Mathematically `chunk_simple_gla(q, k, v=k, g=None, scale=1.0)`; we keep
    a small fallback to `fused_recurrent_gla` for very short sequences /
    cache decode. `initial_state` carries the Sum_KK from previous calls.
    """
    if use_chunk and q.shape[1] > chunk_size:
        return chunk_simple_gla(
            q=q, k=k, v=k, g=None, scale=1.0,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
    return fused_recurrent_gla(
        q=q, k=k, v=k, scale=1.0,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
