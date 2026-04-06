import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.rotary_embedding import RotaryEmbedding


class Eagle3Attention(nn.Module):

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 max_position, rope_theta):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        input_size = 2 * hidden_size
        self.q_proj = nn.Linear(input_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(input_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(input_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(head_dim, head_dim, max_position, rope_theta)

    def forward(self, hidden_states, positions, past_kv=None, kv_valid_lens=None):
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE using flattened positions (compatible with RotaryEmbedding.forward)
        q_flat = q.reshape(-1, self.num_heads, self.head_dim)
        k_flat = k.reshape(-1, self.num_kv_heads, self.head_dim)
        pos_flat = positions.reshape(-1)
        q_flat, k_flat = self.rotary_emb(pos_flat, q_flat, k_flat)
        q = q_flat.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_flat.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_past_kv = (k, v)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Build attention mask for padded KV caches
        attn_mask = None
        if kv_valid_lens is not None and past_kv is not None:
            total_kv_len = k.shape[2]
            # kv_valid_lens is the number of valid PAST entries per sequence
            # Valid positions: [0..kv_valid_lens[i]-1] (past) and [past_len..past_len+seq_len-1] (new)
            past_len = past_kv[0].shape[2]
            pos_range = torch.arange(total_kv_len, device=k.device)
            # Mask: valid past entries OR new entries (at end after padding)
            past_valid = pos_range.unsqueeze(0) < kv_valid_lens.unsqueeze(1)  # (bsz, total_kv_len)
            new_valid = pos_range.unsqueeze(0) >= past_len  # (1, total_kv_len)
            mask = past_valid | new_valid  # (bsz, total_kv_len)
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (bsz, 1, 1, total_kv_len)

        o = F.scaled_dot_product_attention(q, k, v,
            attn_mask=attn_mask,
            is_causal=(past_kv is None and seq_len > 1))
        o = o.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(o), new_past_kv


class Eagle3MLP(nn.Module):

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return self.down_proj(self.act_fn(gate_up))


class Eagle3DecoderLayer(nn.Module):

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 intermediate_size, max_position, rope_theta, rms_norm_eps,
                 norm_before_residual=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_before_residual = norm_before_residual
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Eagle3Attention(hidden_size, num_heads, num_kv_heads,
                                         head_dim, max_position, rope_theta)
        self.mlp = Eagle3MLP(hidden_size, intermediate_size)

    def forward(self, hidden_states, positions, past_kv=None, kv_valid_lens=None):
        embeds = hidden_states[:, :, :self.hidden_size]
        hidden = hidden_states[:, :, self.hidden_size:]

        if self.norm_before_residual:
            hidden = self.hidden_norm(hidden)
            residual = hidden
        else:
            residual = hidden
            hidden = self.hidden_norm(hidden)

        embeds = self.input_layernorm(embeds)
        attn_input = torch.cat([embeds, hidden], dim=-1)

        attn_output, new_past_kv = self.self_attn(attn_input, positions, past_kv, kv_valid_lens)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_past_kv


class Eagle3Speculator(nn.Module):

    def __init__(self, config):
        super().__init__()
        tc = config
        self.hidden_size = tc["hidden_size"]
        self.draft_vocab_size = config.get("draft_vocab_size", 32000)
        target_vocab_size = tc["vocab_size"]
        target_hidden_size = config.get("target_hidden_size") or self.hidden_size

        self.embed_tokens = nn.Embedding(target_vocab_size, self.hidden_size)
        self.fc = nn.Linear(3 * target_hidden_size, self.hidden_size, bias=False)
        self.layers = nn.ModuleList([Eagle3DecoderLayer(
            hidden_size=self.hidden_size,
            num_heads=tc["num_attention_heads"],
            num_kv_heads=tc["num_key_value_heads"],
            head_dim=tc.get("head_dim", self.hidden_size // tc["num_attention_heads"]),
            intermediate_size=tc["intermediate_size"],
            max_position=tc.get("max_position_embeddings", 40960),
            rope_theta=tc.get("rope_theta", 1000000),
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            norm_before_residual=config.get("norm_before_residual", True),
        )])
        self.norm = RMSNorm(self.hidden_size, eps=tc.get("rms_norm_eps", 1e-6))
        self.lm_head = nn.Linear(self.hidden_size, self.draft_vocab_size, bias=False)

        self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(target_vocab_size, dtype=torch.bool))

    def forward(self, input_ids, aux_hidden_states, positions, past_kv=None, kv_valid_lens=None):
        """
        Returns:
            hidden_states: (bsz, seq_len, H) pre-lm-head hidden states
            draft_logits: (bsz, seq_len, draft_vocab_size) raw draft logits
            new_past_kv: updated KV cache
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)
        if positions.dim() == 1:
            positions = positions.unsqueeze(1)
        if aux_hidden_states.dim() == 2:
            aux_hidden_states = aux_hidden_states.unsqueeze(1)

        embeds = self.embed_tokens(input_ids)

        if aux_hidden_states.shape[-1] == self.hidden_size:
            fused_hidden = aux_hidden_states
        else:
            fused_hidden = self.fc(aux_hidden_states)

        layer_input = torch.cat([embeds, fused_hidden], dim=-1)
        hidden_states, new_past_kv = self.layers[0](layer_input, positions, past_kv, kv_valid_lens)
        normed = self.norm(hidden_states)
        draft_logits = self.lm_head(normed)  # (bsz, seq_len, draft_vocab_size)

        return hidden_states, draft_logits, new_past_kv

    def greedy_sample(self, draft_logits):
        """Efficiently convert draft logits to target token IDs via argmax + d2t mapping.
        Avoids allocating a full target vocab tensor."""
        draft_ids = draft_logits.argmax(dim=-1)  # (bsz, seq_len)
        target_ids = draft_ids + self.d2t[draft_ids]
        return target_ids

    def compute_logits(self, hidden_states):
        draft_logits = self.lm_head(hidden_states)
        bsz, seq_len, _ = draft_logits.shape
        # d2t contains offsets: target_id = draft_id + d2t[draft_id]
        target_indices = torch.arange(self.draft_vocab_size, device=draft_logits.device) + self.d2t
        target_vocab_size = self.embed_tokens.num_embeddings
        mapped = draft_logits.new_full((bsz, seq_len, target_vocab_size), float("-inf"))
        mapped[:, :, target_indices] = draft_logits
        return mapped
