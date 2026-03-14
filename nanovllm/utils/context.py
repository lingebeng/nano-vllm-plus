from dataclasses import dataclass
import torch


@dataclass
class Context:
    num_prefill_tokens: int = 0
    # prefill fields
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    prefill_block_tables: torch.Tensor | None = None
    # decode fields
    context_lens: torch.Tensor | None = None
    decode_block_tables: torch.Tensor | None = None
    # common
    slot_mapping: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(**kwargs):
    global _CONTEXT
    _CONTEXT = Context(**kwargs)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
