from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], list[int], list[Sequence]]:
        """Returns (prefill_seqs, chunk_sizes, decode_seqs)."""
        budget = self.max_num_batched_tokens
        num_seqs = 0
        prefill_seqs = []
        chunk_sizes = []
        decode_seqs = []

        # Step 1: schedule running sequences (decode + continuing prefill)
        new_running = deque()
        has_prefilling = not self.enable_chunked_prefill and any(seq.is_prefilling for seq in self.running)
        for seq in self.running:
            if num_seqs >= self.max_num_seqs or budget <= 0:
                new_running.append(seq)
                continue
            if seq.is_prefilling:
                chunk = min(seq.num_uncomputed_tokens, budget)
                if chunk == 0:
                    new_running.append(seq)
                    continue
                budget -= chunk
                prefill_seqs.append(seq)
                chunk_sizes.append(chunk)
            else:
                if has_prefilling:
                    new_running.append(seq)
                    continue
                if not self.block_manager.can_append(seq):
                    # try preemption
                    if new_running:
                        self.preempt(new_running.pop())
                    elif self.running:
                        # skip this seq, will be preempted
                        self.preempt(seq)
                        continue
                    else:
                        self.preempt(seq)
                        continue
                self.block_manager.may_append(seq)
                budget -= 1
                decode_seqs.append(seq)
            num_seqs += 1

        # Step 2: schedule new sequences from waiting
        while self.waiting and num_seqs < self.max_num_seqs and budget > 0:
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)
            seq.num_computed_tokens = seq.num_cached_tokens
            remaining = seq.num_uncomputed_tokens
            chunk = min(remaining, budget)
            if chunk == 0:
                break
            budget -= chunk
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            prefill_seqs.append(seq)
            chunk_sizes.append(chunk)
            num_seqs += 1

        # Rebuild running queue: decode + prefill + deferred
        self.running = deque(decode_seqs + prefill_seqs)
        self.running.extend(new_running)

        return prefill_seqs, chunk_sizes, decode_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.num_computed_tokens = 0
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, prefill_seqs: list[Sequence], chunk_sizes: list[int],
                    decode_seqs: list[Sequence], token_ids: list[int]):
        """Process results. token_ids covers completed-prefill seqs + decode seqs."""
        tid_idx = 0

        for seq, chunk in zip(prefill_seqs, chunk_sizes):
            seq.num_computed_tokens += chunk
            if not seq.is_prefilling:
                # prefill just completed, consume one sampled token
                seq.append_token(token_ids[tid_idx])
                tid_idx += 1
                if (not seq.ignore_eos and token_ids[tid_idx - 1] == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)

        for seq in decode_seqs:
            token_id = token_ids[tid_idx]
            tid_idx += 1
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def ensure_speculative_slots(self, decode_seqs: list[Sequence], num_tokens: int):
        """Pre-allocate blocks for speculative tokens. Returns seqs that can speculate."""
        ok_seqs = []
        for seq in decode_seqs:
            if self.block_manager.ensure_slots(seq, num_tokens):
                ok_seqs.append(seq)
        return ok_seqs

    def postprocess_speculative(self, decode_seqs: list[Sequence],
                                 accepted_tokens: list[list[int]]):
        """Process results from speculative decoding (variable tokens per seq).
        Blocks were pre-allocated via ensure_speculative_slots, so we skip may_append."""
        for seq, tokens in zip(decode_seqs, accepted_tokens):
            for token_id in tokens:
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                    break
            if not seq.is_finished:
                # Clean up unused pre-allocated blocks
                needed_blocks = seq.num_blocks
                while len(seq.block_table) > needed_blocks:
                    block_id = seq.block_table.pop()
                    block = self.block_manager.blocks[block_id]
                    block.ref_count -= 1
                    if block.ref_count == 0:
                        self.block_manager._deallocate_block(block_id)
                # Hash any full blocks that aren't hashed yet
                bm = self.block_manager
                for i in range(len(seq.block_table)):
                    block = bm.blocks[seq.block_table[i]]
                    is_full = (i < len(seq.block_table) - 1) or (len(seq) % bm.block_size == 0)
                    if is_full and block.hash == -1:
                        token_ids = seq.block(i)
                        prefix = bm.blocks[seq.block_table[i-1]].hash if i > 0 else -1
                        h = bm.compute_hash(token_ids, prefix)
                        block.update(h, token_ids)
                        bm.hash_to_block_id[h] = block.block_id
