import os
import math

import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

from nanovllm.kernels.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache, gather_kv_from_cache
from nanovllm.utils.context import get_context

USE_NAIVE_ATTN = os.environ.get("NANO_NAIVE_ATTN", "0") == "1"


def naive_paged_attn_decode(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale, num_kv_heads):
    """
    Naive decode attention: gather KV from paged cache via block_table,
    pad to max_seqlen, then use batched PyTorch SDPA.
    q: (batch, num_heads, head_dim)
    k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
    cache_seqlens: (batch,)
    block_table: (batch, max_num_blocks)
    """
    batch, num_heads, head_dim = q.shape
    block_size = k_cache.shape[1]
    max_num_blocks = block_table.shape[1]
    max_seqlen = max_num_blocks * block_size
    gqa_ratio = num_heads // num_kv_heads

    # Gather all blocks for all sequences at once via advanced indexing
    k_gathered = k_cache[block_table]
    v_gathered = v_cache[block_table]
    k_gathered = k_gathered.reshape(batch, max_seqlen, num_kv_heads, head_dim)
    v_gathered = v_gathered.reshape(batch, max_seqlen, num_kv_heads, head_dim)
    k_gathered = k_gathered.permute(0, 2, 1, 3)
    v_gathered = v_gathered.permute(0, 2, 1, 3)
    if gqa_ratio > 1:
        k_gathered = k_gathered.repeat_interleave(gqa_ratio, dim=1)
        v_gathered = v_gathered.repeat_interleave(gqa_ratio, dim=1)

    pos = torch.arange(max_seqlen, device=q.device).unsqueeze(0)
    mask = pos < cache_seqlens.unsqueeze(1)
    attn_mask = mask.unsqueeze(1).unsqueeze(1)

    q_4d = q.unsqueeze(2)
    o = F.scaled_dot_product_attention(q_4d, k_gathered, v_gathered,
                                       attn_mask=attn_mask, scale=softmax_scale)
    return o.squeeze(2)


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        n = context.num_prefill_tokens
        outputs = []

        # --- Prefill portion ---
        if n > 0:
            pq, pk, pv = q[:n], k[:n], v[:n]
            if context.prefill_block_tables is not None:
                # Non-first chunks or prefix cache: gather cached KV from paged cache
                block_size = k_cache.shape[1]
                pk, pv = gather_kv_from_cache(k_cache, v_cache, context.prefill_block_tables,
                                              context.cu_seqlens_k, block_size)
            po = flash_attn_varlen_func(pq, pk, pv,
                                        max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                        max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                        softmax_scale=self.scale, causal=True)
            outputs.append(po)

        # --- Decode portion ---
        num_decode = q.shape[0] - n
        if num_decode > 0:
            dq = q[n:]
            if USE_NAIVE_ATTN:
                do = naive_paged_attn_decode(dq, k_cache, v_cache,
                                             cache_seqlens=context.context_lens, block_table=context.decode_block_tables,
                                             softmax_scale=self.scale, num_kv_heads=self.num_kv_heads)
            else:
                do = flash_attn_with_kvcache(dq.unsqueeze(1), k_cache, v_cache,
                                             cache_seqlens=context.context_lens, block_table=context.decode_block_tables,
                                             softmax_scale=self.scale, causal=True).squeeze(1)
            outputs.append(do)

        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=0)
