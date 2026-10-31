"""Fused CCQ covariance read + cleaning (forward and backward in one Triton pass each).

The stock path materialises z = C_t q_t with a generic linear-attention read
(``chunk_simple_gla`` with v = k), cleans q in a second kernel, and in the
backward runs three generic kernels (dq/dk, dv and the cleaning backward)
that each rebuild the same intra-chunk matrices. With v = k the two gradient
roles of k coincide and C_t is symmetric, so a single chunk-parallel kernel
can produce q_clean (forward) and dq, dk, dlambda (backward) directly from
q, k, the per-chunk covariance states and the running key sum.

Per document (positions 1-based, pos_t = t + position_offset):

    C_t = sum_{j<=t} k_j k_j^T      m_t = sum_{j<=t} k_j
    z_t = C_t q_t                   s_t = m_t . q_t
    corr_t = z_t / pos_t - s_t m_t / pos_t^2        (mean term optional)
    q_clean_t = q_t - lambda_t corr_t

Backward, given dqc_t = dL/dq_clean_t:

    dlambda_t = -dqc_t . corr_t
    dz_t  = -lambda_t dqc_t / pos_t
    ds_t  =  lambda_t (dqc_t . m_t) / pos_t^2
    dm_t  =  lambda_t s_t dqc_t / pos_t^2 + ds_t q_t
    dq_t  =  dqc_t + ds_t m_t + C_t dz_t
    dk_j  =  sum_{t>=j} [ q_t (k_j . dz_t) + dz_t (k_j . q_t) + dm_t ]

The chunk recurrences for C (forward) and dC (reverse) reuse fla's
``chunk_fwd_h`` / ``chunk_bwd_dh``; everything else is the two kernels below.
"""

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard

from ccq.ops.common.clean import _chunk_block_sums, _reshape_lambda_3d

BT_FUSED = 64
STATES_IN_FP32 = False


def _scan_chunks(x, dim):
    """Inclusive cumsum along ``dim`` using the fast innermost-dimension scan."""
    return x.movedim(dim, -1).cumsum(-1).movedim(-1, dim)


def _cumk_chunk_prefix(k, BT, cu_seqlens, chunk_indices, initial_state_K, need_final):
    """Exclusive per-chunk prefix of the running key sum (fp32, includes the initial state) and the final sum.

    Same contract as the stock helper in ``clean.py``; the chunk-level scans run
    along the innermost dimension, which is several times faster in torch.
    """
    B, T, H, K = k.shape
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    sums = _chunk_block_sums(k, BT, cu_seqlens, chunk_indices, NT)   # (NTG, H, K)
    if cu_seqlens is None:
        sv = sums.view(B, NT, H, K)
        inc = _scan_chunks(sv, 1)
        prefix = inc - sv
        if initial_state_K is not None:
            prefix = prefix + initial_state_K.float()[:, None]
        final = None
        if need_final:
            final = inc[:, -1]
            if initial_state_K is not None:
                final = final + initial_state_K.float()
        return prefix.reshape(B * NT, H, K).contiguous(), final
    cs = _scan_chunks(sums, 0)
    ex = cs - sums
    co = prepare_chunk_offsets(cu_seqlens, BT).long()
    i_n = chunk_indices[:, 0].long()
    prefix = ex - ex.index_select(0, co[i_n])
    if initial_state_K is not None:
        prefix = prefix + initial_state_K.float()[i_n]
    final = None
    if need_final:
        final = cs.index_select(0, co[1:] - 1) - ex.index_select(0, co[:-1])
        if initial_state_K is not None:
            final = final + initial_state_K.float()
    return prefix.contiguous(), final


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3]
    ],
    key=['H', 'K', 'BT', 'BK', 'SUBTRACT_MEAN', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'NT', 'position_offset'])
def _ccq_fused_read_fwd_kernel(
    q, k, h, lam, prefix, qc,
    cu_seqlens, chunk_indices,
    T, NT, position_offset,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """One program per (chunk, batch*head): q_clean = q - lam * (z / pos - s m / pos^2)."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_tg = i_b * NT + i_t
        bos = i_b * T
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    qc += (bos * H + i_h) * K
    lam += bos * H + i_h
    h += (i_tg * H + i_h) * K * K
    prefix += (i_tg * H + i_h) * K

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    pos = (o_t + 1 + position_offset).to(tl.float32)
    pos = tl.where(pos < 1.0, 1.0, pos)
    inv_pos = 1.0 / pos
    b_lam = tl.load(lam + o_t * H, mask=m_t, other=0.0).to(tl.float32)

    # pass 1: intra-chunk q k^T and the per-token scalar s_t = m_t . q_t
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    b_s = tl.zeros([BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        b_q = tl.load(q + o_t[:, None] * H * K + o_k[None, :], mask=m_tk, other=0.0)
        b_k = tl.load(k + o_t[:, None] * H * K + o_k[None, :], mask=m_tk, other=0.0)
        b_A += tl.dot(b_q, tl.trans(b_k))
        if SUBTRACT_MEAN:
            b_pre = tl.load(prefix + o_k, mask=m_k, other=0.0)
            b_cum = b_pre[None, :] + tl.cumsum(b_k.to(tl.float32), axis=0)
            b_s += tl.sum(b_cum * b_q.to(tl.float32), axis=1)
    m_A = (o_t[:, None] >= o_t[None, :]) & m_t[:, None] & m_t[None, :]
    b_A = tl.where(m_A, b_A, 0.0).to(q.dtype.element_ty)
    b_sp = b_s * inv_pos * inv_pos

    # pass 2: per output tile, z = q C_prev + (A o M) k, then clean
    for i_v in range(tl.cdiv(K, BK)):
        o_v = i_v * BK + tl.arange(0, BK)
        m_v = o_v < K
        m_tv = m_t[:, None] & m_v[None, :]
        b_z = tl.zeros([BT, BK], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_q = tl.load(q + o_t[:, None] * H * K + o_k[None, :], mask=m_t[:, None] & m_k[None, :], other=0.0)
            b_h = tl.load(h + o_k[:, None] * K + o_v[None, :], mask=m_k[:, None] & m_v[None, :], other=0.0)
            b_z += tl.dot(b_q, b_h.to(b_q.dtype))
        b_kv = tl.load(k + o_t[:, None] * H * K + o_v[None, :], mask=m_tv, other=0.0)
        b_z += tl.dot(b_A, b_kv)
        b_qv = tl.load(q + o_t[:, None] * H * K + o_v[None, :], mask=m_tv, other=0.0).to(tl.float32)
        b_corr = b_z * inv_pos[:, None]
        if SUBTRACT_MEAN:
            b_pre = tl.load(prefix + o_v, mask=m_v, other=0.0)
            b_cum = b_pre[None, :] + tl.cumsum(b_kv.to(tl.float32), axis=0)
            b_corr -= b_sp[:, None] * b_cum
        b_out = b_qv - b_lam[:, None] * b_corr
        tl.store(qc + o_t[:, None] * H * K + o_v[None, :], b_out.to(qc.dtype.element_ty), mask=m_tv)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['H', 'K', 'BT', 'BK', 'SUBTRACT_MEAN', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'NT', 'position_offset', 'dlam_stride_v'])
def _ccq_fused_read_bwd_kernel(
    q, k, h, dh, lam, prefix, dqc,
    dq, dk, dlam, dm_tot,
    cu_seqlens, chunk_indices,
    T, NT, position_offset, dlam_stride_v,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """One program per (output tile, chunk, batch*head): dq, dk (fp32, with the in-chunk suffix of dm),
    a partial of dlambda (summed over tiles outside), and the chunk sums of dm.

    ``h`` holds the covariance state at the chunk start and ``dh`` the gradient
    w.r.t. the state after the chunk (fla's reverse-scan convention).
    """
    i_v, i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_tg = i_b * NT + i_t
        bos = i_b * T
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    dqc += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    lam += bos * H + i_h
    dlam += i_v * dlam_stride_v + bos * H + i_h
    h += (i_tg * H + i_h) * K * K
    dh += (i_tg * H + i_h) * K * K
    prefix += (i_tg * H + i_h) * K
    dm_tot += (i_tg * H + i_h) * K

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    pos = (o_t + 1 + position_offset).to(tl.float32)
    pos = tl.where(pos < 1.0, 1.0, pos)
    inv_pos = 1.0 / pos
    b_lam = tl.load(lam + o_t * H, mask=m_t, other=0.0).to(tl.float32)
    b_lp = b_lam * inv_pos  # dz_t = -b_lp * dqc_t
    o_v = i_v * BK + tl.arange(0, BK)
    m_v = o_v < K
    m_tv = m_t[:, None] & m_v[None, :]

    # pass 1 over K tiles: A = q k^T, B = dz k^T, s = m . q, w = m . dqc, and the state products for this tile
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    b_B = tl.zeros([BT, BT], dtype=tl.float32)
    b_s = tl.zeros([BT], dtype=tl.float32)
    b_w = tl.zeros([BT], dtype=tl.float32)
    b_z = tl.zeros([BT, BK], dtype=tl.float32)    # q C_prev          (this tile)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)   # dz C_prev         (this tile)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)   # k (dC + dC^T)     (this tile)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        m_kv = m_k[:, None] & m_v[None, :]
        b_q = tl.load(q + o_t[:, None] * H * K + o_k[None, :], mask=m_tk, other=0.0)
        b_k = tl.load(k + o_t[:, None] * H * K + o_k[None, :], mask=m_tk, other=0.0)
        b_g = tl.load(dqc + o_t[:, None] * H * K + o_k[None, :], mask=m_tk, other=0.0)
        b_dz = (-b_lp[:, None] * b_g.to(tl.float32)).to(b_k.dtype)
        b_A += tl.dot(b_q, tl.trans(b_k))
        b_B += tl.dot(b_dz, tl.trans(b_k))
        b_h = tl.load(h + o_k[:, None] * K + o_v[None, :], mask=m_kv, other=0.0).to(b_q.dtype)
        b_z += tl.dot(b_q, b_h)
        b_dq += tl.dot(b_dz, b_h)
        b_dh1 = tl.load(dh + o_k[:, None] * K + o_v[None, :], mask=m_kv, other=0.0)
        b_dh2 = tl.load(dh + o_v[:, None] * K + o_k[None, :], mask=m_v[:, None] & m_k[None, :], other=0.0)
        b_dk += tl.dot(b_k, (b_dh1 + tl.trans(b_dh2)).to(b_k.dtype))
        if SUBTRACT_MEAN:
            b_pre = tl.load(prefix + o_k, mask=m_k, other=0.0)
            b_cum = b_pre[None, :] + tl.cumsum(b_k.to(tl.float32), axis=0)
            b_s += tl.sum(b_cum * b_q.to(tl.float32), axis=1)
            b_w += tl.sum(b_cum * b_g.to(tl.float32), axis=1)
    m_A = (o_t[:, None] >= o_t[None, :]) & m_t[:, None] & m_t[None, :]
    b_A = tl.where(m_A, b_A, 0.0).to(q.dtype.element_ty)
    b_B = tl.where(m_A, b_B, 0.0).to(q.dtype.element_ty)
    b_ds = b_lam * b_w * inv_pos * inv_pos          # dL/ds_t
    b_dm_g = b_lam * b_s * inv_pos * inv_pos        # coefficient of dqc_t inside dm_t

    b_gv = tl.load(dqc + o_t[:, None] * H * K + o_v[None, :], mask=m_tv, other=0.0)
    b_qv = tl.load(q + o_t[:, None] * H * K + o_v[None, :], mask=m_tv, other=0.0)
    b_kv = tl.load(k + o_t[:, None] * H * K + o_v[None, :], mask=m_tv, other=0.0)
    b_gvf = b_gv.to(tl.float32)

    # dq tile: dqc + ds m + dz C_prev + (B o M) k ; z tile only feeds dlambda
    b_z += tl.dot(b_A, b_kv)
    b_dq += tl.dot(b_B, b_kv) + b_gvf
    b_u = tl.sum(b_gvf * b_z, axis=1)
    if SUBTRACT_MEAN:
        b_pre = tl.load(prefix + o_v, mask=m_v, other=0.0)
        b_cum = b_pre[None, :] + tl.cumsum(b_kv.to(tl.float32), axis=0)
        b_dq += b_ds[:, None] * b_cum
    tl.store(dq + o_t[:, None] * H * K + o_v[None, :], b_dq.to(dq.dtype.element_ty), mask=m_tv)

    # dk tile: (A o M)^T dz + (B o M)^T q + k (dC + dC^T) + suffix(dm)
    b_dzv = (-b_lp[:, None] * b_gvf).to(b_qv.dtype)
    b_dk += tl.dot(tl.trans(b_A), b_dzv) + tl.dot(tl.trans(b_B), b_qv)
    if SUBTRACT_MEAN:
        b_dm = b_dm_g[:, None] * b_gvf + b_ds[:, None] * b_qv.to(tl.float32)
        b_dm = tl.where(m_t[:, None], b_dm, 0.0)
        b_tot = tl.sum(b_dm, axis=0)
        b_dk += b_tot[None, :] - tl.cumsum(b_dm, axis=0) + b_dm
        tl.store(dm_tot + o_v, b_tot, mask=m_v)
    tl.store(dk + o_t[:, None] * H * K + o_v[None, :], b_dk, mask=m_tv)

    # dlambda partial for this tile; tile 0 also carries the mean term
    b_dl = -b_u * inv_pos
    if SUBTRACT_MEAN:
        b_dl += tl.where(i_v == 0, b_s * b_w * inv_pos * inv_pos, 0.0)
    tl.store(dlam + o_t * H, b_dl, mask=m_t)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T', 'NT'])
def _ccq_add_carry_kernel(
    src, carry, dst,
    cu_seqlens, chunk_indices,
    T, NT,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """dst[t] = src[t] + carry[chunk of t] (per head), casting to dst's dtype."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_tg = i_b * NT + i_t
        bos = i_b * T
    src += (bos * H + i_h) * K
    dst += (bos * H + i_h) * K
    carry += (i_tg * H + i_h) * K
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        b_x = tl.load(src + o_t[:, None] * H * K + o_k[None, :], mask=m_tk, other=0.0)
        b_c = tl.load(carry + o_k, mask=m_k, other=0.0)
        tl.store(dst + o_t[:, None] * H * K + o_k[None, :], (b_x + b_c[None, :]).to(dst.dtype.element_ty), mask=m_tk)


def _chunk_meta(T, B, cu_seqlens, BT):
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    return chunk_indices, NT, NT if cu_seqlens is None else 0


def _chunk_carry(dm_tot, B, NT, cu_seqlens, chunk_indices, BT, dfinal_mean):
    """Sum of dm over LATER chunks of the same segment (+ dfinal_mean), and the per-segment total."""
    NTG, H, K = dm_tot.shape
    if cu_seqlens is None:
        tv = dm_tot.view(B, NT, H, K)
        inc = _scan_chunks(tv, 1)
        total = inc[:, -1]
        carry = (total[:, None] - inc)
        if dfinal_mean is not None:
            carry = carry + dfinal_mean.float()[:, None]
        return carry.reshape(NTG, H, K).contiguous(), total
    cs = _scan_chunks(dm_tot, 0)
    co = prepare_chunk_offsets(cu_seqlens, BT).long()
    i_n = chunk_indices[:, 0].long()
    seg_last = co[1:] - 1
    seg_first = co[:-1]
    # a segment with zero chunks cannot occur (every document has >= 1 token)
    total = cs.index_select(0, seg_last) - cs.index_select(0, seg_first) + dm_tot.index_select(0, seg_first)
    carry = cs.index_select(0, seg_last[i_n]) - cs
    if dfinal_mean is not None:
        carry = carry + dfinal_mean.float()[i_n]
    return carry.contiguous(), total


def _fused_read_fwd(q, k, lam3, position_offset, initial_cov_state, initial_mean_state,
                    output_final_state, subtract_mean, cu_seqlens):
    B, T, H, K = q.shape
    BT = BT_FUSED
    chunk_indices, NT, NT_arg = _chunk_meta(T, B, cu_seqlens, BT)
    # covariance states at chunk starts (same kernel and precision choice as the stock simple_gla path)
    h, final_cov = chunk_fwd_h(
        k=k, v=k, h0=initial_cov_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_size=BT, states_in_fp32=False,
    )
    need_final = bool(output_final_state and subtract_mean)
    if subtract_mean or need_final:
        prefix, final_mean = _cumk_chunk_prefix(k, BT, cu_seqlens, chunk_indices, initial_mean_state, need_final)
    else:
        prefix, final_mean = torch.empty(0, device=q.device, dtype=torch.float32), None
    qc = torch.empty_like(q)
    BK = min(64, triton.next_power_of_2(K))
    _ccq_fused_read_fwd_kernel[(NT, B * H)](
        q, k, h, lam3, prefix, qc,
        cu_seqlens, chunk_indices,
        T, NT_arg, position_offset,
        H=H, K=K, BT=BT, BK=BK,
        SUBTRACT_MEAN=bool(subtract_mean),
    )
    return qc, final_cov, final_mean


def _fused_read_bwd(q, k, lam3, dqc, position_offset, initial_cov_state, initial_mean_state,
                    subtract_mean, cu_seqlens, dfinal_cov, dfinal_mean):
    B, T, H, K = q.shape
    BT = BT_FUSED
    chunk_indices, NT, NT_arg = _chunk_meta(T, B, cu_seqlens, BT)
    # Covariance states in the activation dtype, as the GDN backbone keeps its own states in
    # both directions; this halves the state traffic of the fused kernel.
    h, _ = chunk_fwd_h(
        k=k, v=k, h0=initial_cov_state, output_final_state=False,
        cu_seqlens=cu_seqlens, chunk_size=BT, states_in_fp32=STATES_IN_FP32,
    )
    # dz = -lam * dqc / pos feeds the reverse scan for dC; computed by a cheap elementwise pass
    pos = _positions(T, B, H, cu_seqlens, position_offset, q.device)
    dz = (-(lam3.float() / pos)[..., None] * dqc.float()).to(q.dtype)
    dh, dh0 = chunk_bwd_dh(
        q=q, k=k, v=k, do=dz, h0=initial_cov_state, dht=dfinal_cov, scale=1.0,
        cu_seqlens=cu_seqlens, chunk_size=BT, states_in_fp32=STATES_IN_FP32,
    )
    if subtract_mean:
        prefix, _ = _cumk_chunk_prefix(k, BT, cu_seqlens, chunk_indices, initial_mean_state, False)
    else:
        prefix = torch.empty(0, device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    dk32 = torch.empty(B, T, H, K, device=q.device, dtype=torch.float32)
    BK = min(64, triton.next_power_of_2(K))
    NV = triton.cdiv(K, BK)
    dlam_part = torch.empty(NV, B, T, H, device=q.device, dtype=torch.float32)
    NTG = B * NT if cu_seqlens is None else NT
    dm_tot = torch.zeros(NTG, H, K, device=q.device, dtype=torch.float32)
    _ccq_fused_read_bwd_kernel[(NV, NT, B * H)](
        q, k, h, dh, lam3, prefix, dqc,
        dq, dk32, dlam_part, dm_tot,
        cu_seqlens, chunk_indices,
        T, NT_arg, position_offset, B * T * H,
        H=H, K=K, BT=BT, BK=BK,
        SUBTRACT_MEAN=bool(subtract_mean),
    )
    dlam = dlam_part.sum(0)
    dmean0 = None
    if subtract_mean or dfinal_mean is not None:
        carry, total = _chunk_carry(dm_tot, B, NT, cu_seqlens, chunk_indices, BT, dfinal_mean)
        if initial_mean_state is not None:
            dmean0 = total
            if dfinal_mean is not None:
                dmean0 = dmean0 + dfinal_mean.float()
    else:
        carry = torch.zeros(NTG, H, K, device=q.device, dtype=torch.float32)
    dk = torch.empty_like(k)
    _ccq_add_carry_kernel[(NT, B * H)](
        dk32, carry, dk,
        cu_seqlens, chunk_indices,
        T, NT_arg,
        H=H, K=K, BT=BT, BK=min(64, triton.next_power_of_2(K)),
    )
    return dq, dk, dlam, dh0, dmean0


def _positions(T, B, H, cu_seqlens, position_offset, device):
    """Segment-relative 1-based positions (+ offset, clamped at 1) broadcast to [B, T, H] fp32."""
    if cu_seqlens is None:
        pos = torch.arange(1, T + 1, device=device, dtype=torch.float32) + position_offset
        return pos.clamp_min(1.0).view(1, T, 1).expand(B, T, H)
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    starts = torch.repeat_interleave(cu_seqlens[:-1].long(), lengths)
    pos = (torch.arange(T, device=device) - starts + 1).float() + position_offset
    return pos.clamp_min(1.0).view(1, T, 1).expand(B, T, H)


class FusedCCQCleanRead(torch.autograd.Function):
    """q_clean, final covariance and final key-sum states from (q, k, lambda) in one fused read."""

    @staticmethod
    @autocast_custom_fwd
    @input_guard
    def forward(ctx, q, k, lambda_h, position_offset, initial_cov_state, initial_mean_state,
                output_final_state, subtract_mean, cu_seqlens):
        lam3 = _reshape_lambda_3d(lambda_h, q)
        qc, final_cov, final_mean = _fused_read_fwd(
            q, k, lam3, position_offset, initial_cov_state, initial_mean_state,
            output_final_state, subtract_mean, cu_seqlens,
        )
        ctx.save_for_backward(q, k, lam3, initial_cov_state, initial_mean_state, cu_seqlens)
        ctx.lambda_h = lambda_h if isinstance(lambda_h, torch.Tensor) else None
        ctx.position_offset = position_offset
        ctx.subtract_mean = subtract_mean
        return qc, final_cov, final_mean

    @staticmethod
    @autocast_custom_bwd
    @input_guard
    def backward(ctx, dqc, dfinal_cov, dfinal_mean):
        q, k, lam3, h0, m0, cu_seqlens = ctx.saved_tensors
        dq, dk, dlam3, dh0, dm0 = _fused_read_bwd(
            q, k, lam3, dqc, ctx.position_offset, h0, m0, ctx.subtract_mean, cu_seqlens,
            dfinal_cov, dfinal_mean,
        )
        lam_t = ctx.lambda_h
        dl = None
        if lam_t is not None and lam_t.requires_grad:
            if lam_t.ndim == 0:
                dl = dlam3.sum()
            elif lam_t.ndim == 1:
                dl = dlam3.sum(dim=(0, 1))
            elif lam_t.ndim == 4 and lam_t.shape[-1] == 1:
                dl = dlam3.unsqueeze(-1)
            else:
                dl = dlam3
            dl = dl.to(lam_t.dtype)
        return dq, dk, dl, None, dh0, dm0, None, None, None


def fused_ccq_clean_read(
    q, k, lambda_h, *, position_offset=0, initial_cov_state=None, initial_mean_state=None,
    output_final_state=False, subtract_mean=True, cu_seqlens=None,
):
    """Fused replacement for ``_ccq_compute_z_and_sum_kk`` + ``_CCQCleanQ``.

    Returns (q_clean, final_cov_state, final_mean_state); the states are None unless requested.
    """
    qc, final_cov, final_mean = FusedCCQCleanRead.apply(
        q, k, lambda_h, int(position_offset), initial_cov_state, initial_mean_state,
        bool(output_final_state), bool(subtract_mean), cu_seqlens,
    )
    if not output_final_state:
        final_cov, final_mean = None, None
    return qc, final_cov, final_mean
