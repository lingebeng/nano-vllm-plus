"""
Flash Attention Triton Kernels for nano-vllm inference.

Extracted from Flash Attention v1 (flash_attn_triton.py by Tri Dao) and adapted for:
- Triton 3.x compatibility (removed trans_b, TMP buffer workaround)
- GQA (Grouped Query Attention) support
- Inference-only (no backward pass, no LSE output)
- Paged KV cache decode kernel
"""

import math

import torch

import triton
import triton.language as tl


# ============================================================
# Flash Attention Forward Kernel (Prefill)
# Adapted from FA v1's flash_attn_triton.py
# ============================================================

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    nheads, seqlen_q, seqlen_k, headdim,
    CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,
    GQA_RATIO: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    off_kv_h = off_h // GQA_RATIO

    # Causal offset for bottom-right aligned masking (seqlen_q <= seqlen_k)
    causal_offset = seqlen_k - seqlen_q

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # pointers to Q (uses off_h for Q heads)
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    # pointers to K, V (uses off_kv_h for KV heads - GQA)
    k_ptrs = K + off_b * stride_kb + off_kv_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_kv_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])

    # initialize accumulators
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # load q: stays in SRAM throughout
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0)

    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M + causal_offset, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- load k --
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen_k,
                            other=0.0)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn,
                            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                            other=0.0)

        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where((offs_m + causal_offset)[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))

        # -- online softmax --
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale previous accumulator
        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]

        # -- load v and update output --
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n)[:, None] < seqlen_k,
                            other=0.0)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn,
                            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                            other=0.0)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics --
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    # final rescale
    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]

    # write output
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(out_ptrs, acc_o,
                     mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))


def _flash_attn_forward(q, k, v, causal=True, softmax_scale=None):
    """
    q: (batch, seqlen_q, nheads, headdim)
    k: (batch, seqlen_k, nheads_kv, headdim)
    v: (batch, seqlen_k, nheads_kv, headdim)
    Returns: (batch, seqlen_q, nheads, headdim)
    """
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_kv, _ = k.shape
    assert d <= 128, 'FlashAttention only supports head dimensions up to 128'
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype in [torch.float16, torch.bfloat16]
    assert nheads % nheads_kv == 0
    gqa_ratio = nheads // nheads_kv
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q, k, v, o,
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        o.stride(0), o.stride(2), o.stride(1),
        nheads, seqlen_q, seqlen_k, d,
        seqlen_q // 32, seqlen_k // 32,  # cache keys for triton compilation
        gqa_ratio, causal, BLOCK_HEADDIM,
        BLOCK_M=BLOCK, BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k,
                           softmax_scale=None, causal=True, **kwargs):
    """
    Variable-length flash attention for prefill.

    q: (total_q, nheads, headdim)
    k: (total_k, nheads_kv, headdim)
    v: (total_k, nheads_kv, headdim)
    cu_seqlens_q: (batch + 1,) cumulative sequence lengths for Q
    cu_seqlens_k: (batch + 1,) cumulative sequence lengths for K
    Returns: (total_q, nheads, headdim)
    """
    batch_size = len(cu_seqlens_q) - 1
    nheads = q.shape[1]
    headdim = q.shape[2]
    output = torch.empty_like(q)

    for i in range(batch_size):
        q_start = cu_seqlens_q[i].item()
        q_end = cu_seqlens_q[i + 1].item()
        k_start = cu_seqlens_k[i].item()
        k_end = cu_seqlens_k[i + 1].item()

        # (1, seqlen_q, nheads, headdim)
        q_i = q[q_start:q_end].unsqueeze(0)
        k_i = k[k_start:k_end].unsqueeze(0)
        v_i = v[k_start:k_end].unsqueeze(0)

        o_i = _flash_attn_forward(q_i, k_i, v_i, causal=causal, softmax_scale=softmax_scale)
        output[q_start:q_end] = o_i.squeeze(0)

    return output


# ============================================================
# Paged Attention Decode Kernel
# ============================================================

@triton.jit
def _paged_attn_decode_kernel(
    Q, K_cache, V_cache, Out,
    block_table_ptr,
    cache_seqlens_ptr,
    softmax_scale,
    stride_qb, stride_qh,
    stride_cache_block, stride_cache_seq, stride_cache_head,
    stride_ob, stride_oh,
    stride_bt_b,
    nheads_kv,
    GQA_RATIO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head = head_idx // GQA_RATIO

    seqlen = tl.load(cache_seqlens_ptr + batch_idx)

    # load query vector: (BLOCK_HEADDIM,)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptr = Q + batch_idx * stride_qb + head_idx * stride_qh
    q_vec = tl.load(q_ptr + offs_d, mask=offs_d < BLOCK_HEADDIM, other=0.0).to(tl.float32)

    # accumulators for online softmax
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_HEADDIM], dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)

    # iterate over all KV tokens in tiles of BLOCK_N
    for start_n in range(0, seqlen, BLOCK_N):
        # determine which cache block this tile is in
        block_idx = start_n // BLOCK_SIZE
        offset_in_block = start_n % BLOCK_SIZE
        physical_block_id = tl.load(block_table_ptr + batch_idx * stride_bt_b + block_idx)

        # load K tile: (BLOCK_N, BLOCK_HEADDIM)
        k_base = K_cache + physical_block_id * stride_cache_block + kv_head * stride_cache_head
        k_ptrs = k_base + (offset_in_block + offs_n)[:, None] * stride_cache_seq + offs_d[None, :]
        mask_n = (start_n + offs_n) < seqlen
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # compute attention scores: (BLOCK_N,)
        qk = tl.sum(q_vec[None, :] * k, axis=1) * softmax_scale
        qk = tl.where(mask_n, qk, float("-inf"))

        # online softmax update
        m_new = tl.maximum(m_i, tl.max(qk, axis=0)[None])
        exp_qk = tl.exp(qk - m_new)
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(exp_qk, axis=0)[None]

        # load V tile: (BLOCK_N, BLOCK_HEADDIM)
        v_base = V_cache + physical_block_id * stride_cache_block + kv_head * stride_cache_head
        v_ptrs = v_base + (offset_in_block + offs_n)[:, None] * stride_cache_seq + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # rescale and accumulate
        acc_o = acc_o * tl.exp(m_i - m_new)
        acc_o += tl.sum(exp_qk[:, None] * v, axis=0)

        m_i = m_new
        l_i = l_new

    # final output: normalize by softmax denominator
    acc_o = acc_o / l_i

    # store output
    out_ptr = Out + batch_idx * stride_ob + head_idx * stride_oh
    tl.store(out_ptr + offs_d, acc_o.to(Out.dtype.element_ty), mask=offs_d < BLOCK_HEADDIM)


def flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=None,
                            block_table=None, softmax_scale=None, causal=True):
    """
    Paged attention for decode phase.

    q: (batch, 1, nheads, headdim)
    k_cache: (num_blocks, block_size, nheads_kv, headdim)
    v_cache: (num_blocks, block_size, nheads_kv, headdim)
    cache_seqlens: (batch,)
    block_table: (batch, max_num_blocks)
    Returns: (batch, 1, nheads, headdim)
    """
    batch, seqlen_q, nheads, headdim = q.shape
    assert seqlen_q == 1
    _, block_size, nheads_kv, _ = k_cache.shape
    assert nheads % nheads_kv == 0
    gqa_ratio = nheads // nheads_kv
    softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)

    # reshape q: (batch, 1, nheads, headdim) -> (batch, nheads, headdim) for kernel
    q_flat = q.squeeze(1)  # (batch, nheads, headdim)
    o_flat = torch.empty_like(q_flat)

    BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
    BLOCK_N = min(128, block_size)  # tile size for KV iteration, must divide block_size
    assert block_size % BLOCK_N == 0

    grid = (batch, nheads)
    _paged_attn_decode_kernel[grid](
        q_flat, k_cache, v_cache, o_flat,
        block_table,
        cache_seqlens,
        softmax_scale,
        q_flat.stride(0), q_flat.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        o_flat.stride(0), o_flat.stride(1),
        block_table.stride(0),
        nheads_kv,
        gqa_ratio,
        block_size,
        BLOCK_HEADDIM,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=1,
    )

    return o_flat.unsqueeze(1)  # (batch, 1, nheads, headdim)


# ============================================================
# Paged Attention Prefill Kernel (for verify / prefix-cached prefill)
# Reads K/V directly from paged cache via block tables,
# eliminating the need for gather_kv_from_cache.
# ============================================================

@triton.jit
def _paged_attn_prefill_kernel(
    Q, K_cache, V_cache, Out,
    block_table_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    softmax_scale,
    stride_qm, stride_qh,
    stride_cache_block, stride_cache_seq, stride_cache_head,
    stride_om, stride_oh,
    stride_bt_b,
    nheads_kv,
    NUM_Q_TILES: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (num_seqs * NUM_Q_TILES, nheads)
    program_idx = tl.program_id(0)
    seq_idx = program_idx // NUM_Q_TILES
    q_tile_idx = program_idx % NUM_Q_TILES
    head_idx = tl.program_id(1)
    kv_head = head_idx // GQA_RATIO

    # Per-sequence lengths from cu_seqlens
    q_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    q_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seqlen_q = q_end - q_start

    k_start = tl.load(cu_seqlens_k_ptr + seq_idx)
    k_end = tl.load(cu_seqlens_k_ptr + seq_idx + 1)
    seqlen_k = k_end - k_start

    # Bottom-right aligned causal offset
    causal_offset = seqlen_k - seqlen_q

    # Q tile offset within this sequence
    q_tile_start = q_tile_idx * BLOCK_Q

    # Load Q: (BLOCK_Q, BLOCK_HEADDIM)
    offs_q = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + (q_start + q_tile_start + offs_q[:, None]) * stride_qm + head_idx * stride_qh + offs_d[None, :]
    q_mask = (q_tile_start + offs_q[:, None]) < seqlen_q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Per-query online softmax accumulators
    m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_HEADDIM], dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)

    # Causal end: last K position this Q tile can see
    end_n = tl.minimum(seqlen_k, (q_tile_start + BLOCK_Q) + causal_offset)

    # Iterate over KV cache in tiles of BLOCK_N
    for start_n in range(0, end_n, BLOCK_N):
        block_idx = start_n // BLOCK_SIZE
        offset_in_block = start_n % BLOCK_SIZE
        physical_block = tl.load(block_table_ptr + seq_idx * stride_bt_b + block_idx)

        # Load K tile: (BLOCK_N, BLOCK_HEADDIM) from paged cache
        k_base = K_cache + physical_block * stride_cache_block + kv_head * stride_cache_head
        k_ptrs = k_base + (offset_in_block + offs_n)[:, None] * stride_cache_seq + offs_d[None, :]
        mask_n = (start_n + offs_n) < seqlen_k
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        # QK: (BLOCK_Q, BLOCK_N) — tl.dot produces float32
        qk = tl.zeros([BLOCK_Q, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= softmax_scale

        # Mask invalid K positions
        qk += tl.where(mask_n[None, :], 0, float("-inf"))
        # Causal mask
        qk += tl.where((q_tile_start + offs_q + causal_offset)[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        # Mask padded Q rows
        qk += tl.where((q_tile_start + offs_q)[:, None] < seqlen_q, 0, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_qk = tl.exp(qk - m_new[:, None])
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(exp_qk, axis=1)

        # Load V tile: (BLOCK_N, BLOCK_HEADDIM) from paged cache
        v_base = V_cache + physical_block * stride_cache_block + kv_head * stride_cache_head
        v_ptrs = v_base + (offset_in_block + offs_n)[:, None] * stride_cache_seq + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Rescale and accumulate
        acc = acc * tl.exp(m_i - m_new)[:, None]
        acc += tl.dot(exp_qk.to(v.dtype), v)

        m_i = m_new
        l_i = l_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output for valid Q positions
    out_ptrs = Out + (q_start + q_tile_start + offs_q[:, None]) * stride_om + head_idx * stride_oh + offs_d[None, :]
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=(q_tile_start + offs_q)[:, None] < seqlen_q)


def flash_attn_paged_prefill(q, k_cache, v_cache, block_table,
                              cu_seqlens_q, cu_seqlens_k,
                              max_seqlen_q, softmax_scale=None):
    """
    Prefill attention reading K/V directly from paged cache.

    q: (total_q, nheads, headdim)
    k_cache: (num_blocks, block_size, nheads_kv, headdim)
    v_cache: (num_blocks, block_size, nheads_kv, headdim)
    block_table: (num_seqs, max_blocks_per_seq)
    cu_seqlens_q: (num_seqs + 1,)
    cu_seqlens_k: (num_seqs + 1,)
    max_seqlen_q: int
    Returns: (total_q, nheads, headdim)
    """
    nheads = q.shape[1]
    headdim = q.shape[2]
    nheads_kv = k_cache.shape[2]
    block_size = k_cache.shape[1]
    num_seqs = block_table.shape[0]

    assert nheads % nheads_kv == 0
    gqa_ratio = nheads // nheads_kv
    softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)

    output = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
    BLOCK_Q = 64  # fixed tile size to stay within shared memory limits
    BLOCK_N = min(128, block_size)
    assert block_size % BLOCK_N == 0

    num_q_tiles = triton.cdiv(max_seqlen_q, BLOCK_Q)
    grid = (num_seqs * num_q_tiles, nheads)
    _paged_attn_prefill_kernel[grid](
        q, k_cache, v_cache, output,
        block_table,
        cu_seqlens_q, cu_seqlens_k,
        softmax_scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        output.stride(0), output.stride(1),
        block_table.stride(0),
        nheads_kv,
        num_q_tiles,
        gqa_ratio,
        block_size,
        BLOCK_HEADDIM,
        BLOCK_Q=BLOCK_Q,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=1,
    )
    return output


# ============================================================
# Prefix Cache Helper
# ============================================================

def gather_kv_from_cache(k_cache, v_cache, block_table, cu_seqlens_k, block_size):
    """
    Gather K/V from paged cache into contiguous tensors for prefix-cached prefill.

    k_cache: (num_blocks, block_size, nheads_kv, headdim)
    v_cache: (num_blocks, block_size, nheads_kv, headdim)
    block_table: (num_seqs, max_blocks)
    cu_seqlens_k: (num_seqs + 1,)
    Returns: (total_k, nheads_kv, headdim), (total_k, nheads_kv, headdim)
    """
    total_k = cu_seqlens_k[-1].item()
    num_kv_heads = k_cache.shape[2]
    headdim = k_cache.shape[3]
    num_seqs = block_table.shape[0]

    k_out = torch.empty(total_k, num_kv_heads, headdim, dtype=k_cache.dtype, device=k_cache.device)
    v_out = torch.empty(total_k, num_kv_heads, headdim, dtype=v_cache.dtype, device=v_cache.device)

    for i in range(num_seqs):
        start = cu_seqlens_k[i].item()
        seqlen = cu_seqlens_k[i + 1].item() - start
        num_blocks = (seqlen + block_size - 1) // block_size
        block_ids = block_table[i, :num_blocks]
        # gather all blocks at once via advanced indexing
        gathered_k = k_cache[block_ids].reshape(-1, num_kv_heads, headdim)[:seqlen]
        gathered_v = v_cache[block_ids].reshape(-1, num_kv_heads, headdim)[:seqlen]
        k_out[start:start + seqlen] = gathered_k
        v_out[start:start + seqlen] = gathered_v

    return k_out, v_out
