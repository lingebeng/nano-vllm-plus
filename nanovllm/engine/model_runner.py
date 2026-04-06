import json
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()

        # Speculative decoding: load draft model
        self.draft_model = None
        self.aux_layer_ids = None
        self.draft_kv_cache = {}  # seq_id → (K, V) tensors for draft model
        if config.speculative_model:
            self._load_draft_model(config)

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = max(1, min(max_num_batched_tokens // seq_len, self.config.max_num_seqs))
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        # Warmup with prefill-only batch
        self.run(seqs, [seq_len] * num_seqs, [])
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_chunked_prefill(self, prefill_seqs: list[Sequence], chunk_sizes: list[int],
                                 decode_seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []

        # --- Prefill part ---
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        need_prefill_block_tables = False

        for seq, chunk_size in zip(prefill_seqs, chunk_sizes):
            start = seq.num_computed_tokens
            end = start + chunk_size
            input_ids.extend(seq.token_ids[start:end])
            positions.extend(list(range(start, end)))

            seqlen_q = chunk_size
            seqlen_k = end  # full context length after this chunk
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if seq.num_computed_tokens > 0:
                need_prefill_block_tables = True

            # slot mapping for prefill chunk
            if seq.block_table:
                start_block = start // self.block_size
                end_block = (end - 1) // self.block_size + 1
                for i in range(start_block, end_block):
                    block_start = seq.block_table[i] * self.block_size
                    token_start = max(start, i * self.block_size)
                    token_end = min(end, (i + 1) * self.block_size)
                    offset_start = token_start - i * self.block_size
                    offset_end = token_end - i * self.block_size
                    slot_mapping.extend(list(range(block_start + offset_start, block_start + offset_end)))

        num_prefill_tokens = len(input_ids)

        # --- Decode part ---
        context_lens = []
        for seq in decode_seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        # --- Build tensors ---
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        ctx_kwargs = dict(
            num_prefill_tokens=num_prefill_tokens,
            slot_mapping=slot_mapping,
        )

        if prefill_seqs:
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            ctx_kwargs.update(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )
            if need_prefill_block_tables or any(seq.num_cached_tokens > 0 for seq in prefill_seqs):
                ctx_kwargs["prefill_block_tables"] = self.prepare_block_tables(prefill_seqs)

        if decode_seqs:
            context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            ctx_kwargs.update(
                context_lens=context_lens,
                decode_block_tables=self.prepare_block_tables(decode_seqs),
            )

        set_context(**ctx_kwargs)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, has_prefill: bool):
        if has_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model(input_ids, positions)
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.decode_block_tables.size(1)] = context.decode_block_tables
            graph.replay()
            return graph_vars["outputs"][:bs]

    def run(self, prefill_seqs: list[Sequence], chunk_sizes: list[int],
            decode_seqs: list[Sequence]) -> list[int]:
        input_ids, positions = self.prepare_chunked_prefill(prefill_seqs, chunk_sizes, decode_seqs)
        has_prefill = len(prefill_seqs) > 0

        hidden_states = self.run_model(input_ids, positions, has_prefill)

        # Extract hidden states for positions that need sampling:
        # - last token of each prefill seq that completes prefill
        # - all decode tokens
        sample_indices = []
        sample_seqs = []
        offset = 0
        for seq, chunk in zip(prefill_seqs, chunk_sizes):
            if seq.num_computed_tokens + chunk >= seq.num_prompt_tokens:
                # This chunk completes prefill — sample from the last position
                sample_indices.append(offset + chunk - 1)
                sample_seqs.append(seq)
            offset += chunk

        # All decode tokens
        num_prefill_tokens = get_context().num_prefill_tokens
        for i, seq in enumerate(decode_seqs):
            sample_indices.append(num_prefill_tokens + i)
            sample_seqs.append(seq)

        if not sample_indices:
            reset_context()
            return []

        sample_indices = torch.tensor(sample_indices, dtype=torch.int64, device=hidden_states.device)
        sampled_hidden = hidden_states[sample_indices]
        logits = self.model.compute_logits(sampled_hidden)

        if self.rank == 0:
            temperatures = self.prepare_sample(sample_seqs)
            token_ids = self.sampler(logits, temperatures).tolist()
        else:
            token_ids = None

        reset_context()
        return token_ids

    # ---- Speculative decoding ----

    def _load_draft_model(self, config: Config):
        import os
        from nanovllm.models.eagle3 import Eagle3Speculator
        from safetensors import safe_open

        spec_path = config.speculative_model
        with open(os.path.join(spec_path, "config.json")) as f:
            spec_config = json.load(f)

        tc = spec_config["transformer_layer_config"]
        tc["vocab_size"] = config.hf_config.vocab_size
        self.draft_model = Eagle3Speculator({**spec_config, **tc,
                                              "draft_vocab_size": spec_config.get("draft_vocab_size", 32000),
                                              "target_hidden_size": spec_config.get("target_hidden_size"),
                                              "norm_before_residual": spec_config.get("norm_before_residual", True)})
        self.draft_model = self.draft_model.cuda()

        # Load weights
        for file in [os.path.join(spec_path, "model.safetensors")]:
            with safe_open(file, "pt", "cpu") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    if name in ("d2t", "t2d"):
                        self.draft_model.get_buffer(name).copy_(tensor)
                    else:
                        param = self.draft_model.get_parameter(name)
                        param.data.copy_(tensor.to(param.dtype))

        self.draft_model.eval()

        # Determine auxiliary layer indices (must match vLLM/speculators default)
        n = config.hf_config.num_hidden_layers
        self.aux_layer_ids = {2, n // 2, n - 3}
        self.num_speculative_tokens = config.num_speculative_tokens

    @torch.inference_mode()
    def run_speculative(self, decode_seqs: list[Sequence]) -> list[list[int]]:
        K = self.num_speculative_tokens
        N = len(decode_seqs)

        # === Phase 1: Target decode (1 token/seq) → logits + aux hidden states ===
        input_ids, positions = self.prepare_chunked_prefill([], [], decode_seqs)
        result = self.model(input_ids, positions, aux_layer_ids=self.aux_layer_ids)
        hidden_states, aux_hidden = result  # aux_hidden: (N, 3*H)

        target_logits = self.model.compute_logits(hidden_states)  # (N, vocab)
        t0_tokens = target_logits.argmax(dim=-1)  # (N,)
        reset_context()

        # === Phase 2: Draft K tokens with persistent KV cache (batched) ===
        all_draft_tokens = torch.empty(K, N, dtype=torch.long, device="cuda")

        # Build batched past_kv from per-sequence caches
        past_kv, kv_lengths = self._get_batched_draft_kv(decode_seqs)
        draft_hidden = None

        for k in range(K):
            if k == 0:
                d_input = t0_tokens  # (N,)
                d_aux = aux_hidden  # (N, 3*H)
            else:
                d_input = all_draft_tokens[k - 1]
                d_aux = draft_hidden  # (N, H)

            # Use contiguous positions based on draft KV cache length
            d_positions = kv_lengths + k

            draft_h, draft_logits_raw, past_kv = self.draft_model(
                d_input, d_aux, d_positions, past_kv, kv_lengths + k
            )
            draft_hidden = draft_h[:, -1, :]  # (N, H)
            all_draft_tokens[k] = self.draft_model.greedy_sample(draft_logits_raw[:, -1:, :])[:, 0]

        # Convert draft tokens for verify and accept
        draft_tokens_list = [all_draft_tokens[k].tolist() for k in range(K)]
        t0_list = t0_tokens.tolist()

        # === Phase 3: Target verify (K+1 tokens/seq) ===
        verify_logits = self._run_verify(decode_seqs, t0_list, draft_tokens_list)

        # === Phase 4: Vectorized accept/reject ===
        target_preds = verify_logits.argmax(dim=-1)  # (N, K+1)
        draft_tokens_t = all_draft_tokens.T  # (N, K)
        matches = (target_preds[:, :K] == draft_tokens_t)
        num_accepted = matches.cumprod(dim=1).sum(dim=1)  # (N,)

        target_preds_cpu = target_preds.tolist()
        num_accepted_cpu = num_accepted.tolist()

        accepted = []
        for i, seq in enumerate(decode_seqs):
            na = num_accepted_cpu[i]
            seq_tokens = [t0_list[i]]
            for k in range(na):
                seq_tokens.append(draft_tokens_list[k][i])
            if na < K:
                seq_tokens.append(target_preds_cpu[i][na])
            else:
                seq_tokens.append(target_preds_cpu[i][K])
            accepted.append(seq_tokens)

        # Update per-sequence draft KV caches (trim rejected entries)
        self._update_draft_kv(decode_seqs, past_kv, num_accepted_cpu)

        reset_context()
        return accepted

    def _get_batched_draft_kv(self, decode_seqs):
        """Build batched (K, V) from per-sequence draft KV caches, padding to max length.
        Returns (past_kv, kv_lengths) where kv_lengths is (N,) tensor."""
        N = len(decode_seqs)
        kv_list = [self.draft_kv_cache.get(seq.seq_id) for seq in decode_seqs]

        # Compute per-sequence KV lengths
        lens = [kv[0].shape[2] if kv is not None else 0 for kv in kv_list]
        kv_lengths = torch.tensor(lens, dtype=torch.long, device="cuda")

        # If no sequences have cached KV, return None
        if all(kv is None for kv in kv_list):
            return None, kv_lengths

        # Find max KV length
        max_len = max(lens)
        if max_len == 0:
            return None, kv_lengths

        # Get shapes from first non-None entry
        ref_kv = next(kv for kv in kv_list if kv is not None)
        num_kv_heads = ref_kv[0].shape[1]
        head_dim = ref_kv[0].shape[3]
        dtype = ref_kv[0].dtype

        # Stack into batched tensors with padding
        batched_k = torch.zeros(N, num_kv_heads, max_len, head_dim, dtype=dtype, device="cuda")
        batched_v = torch.zeros(N, num_kv_heads, max_len, head_dim, dtype=dtype, device="cuda")
        for i, kv in enumerate(kv_list):
            if kv is not None:
                L = kv[0].shape[2]
                batched_k[i, :, :L, :] = kv[0][0]
                batched_v[i, :, :L, :] = kv[1][0]

        return (batched_k, batched_v), kv_lengths

    def _update_draft_kv(self, decode_seqs, past_kv, num_accepted_cpu):
        """Store per-sequence draft KV caches after trimming rejected entries."""
        K = self.num_speculative_tokens
        if past_kv is None:
            return

        batched_k, batched_v = past_kv
        for i, seq in enumerate(decode_seqs):
            na = num_accepted_cpu[i]
            # We added K entries during drafting. Keep t0 + accepted drafts.
            to_remove = max(K - na - 1, 0)
            keep = batched_k.shape[2] - to_remove
            self.draft_kv_cache[seq.seq_id] = (
                batched_k[i:i+1, :, :keep, :].contiguous(),
                batched_v[i:i+1, :, :keep, :].contiguous(),
            )

    def _run_verify(self, decode_seqs: list[Sequence], t0_list: list[int],
                    draft_tokens_list: list[list[int]]) -> torch.Tensor:
        """Run target model verification on [t0, d1, ..., dK] for each seq.
        Returns logits of shape (N, K+1, vocab)."""
        K = len(draft_tokens_list)
        N = len(decode_seqs)
        num_verify = K + 1  # t0 + all K draft tokens

        input_ids = []
        positions_list = []
        slot_mapping = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = num_verify
        max_seqlen_k = 0

        for i, seq in enumerate(decode_seqs):
            # Tokens: [t0, d1, ..., dK]
            verify_tokens = [t0_list[i]] + [draft_tokens_list[k][i] for k in range(K)]
            input_ids.extend(verify_tokens)

            base_pos = len(seq)  # position after existing context
            pos_range = list(range(base_pos, base_pos + num_verify))
            positions_list.extend(pos_range)

            # Slot mapping for the num_verify new positions
            for p in range(num_verify):
                abs_pos = base_pos + p
                block_idx = abs_pos // self.block_size
                offset = abs_pos % self.block_size
                physical = seq.block_table[block_idx]
                slot_mapping.append(physical * self.block_size + offset)

            cu_seqlens_q.append(cu_seqlens_q[-1] + num_verify)
            seqlen_k = base_pos + num_verify
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)

        input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_t = torch.tensor(positions_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # Need block tables for reading from KV cache
        block_tables = self.prepare_block_tables(decode_seqs)

        set_context(
            num_prefill_tokens=N * num_verify,
            slot_mapping=slot_mapping_t,
            cu_seqlens_q=cu_seqlens_q_t,
            cu_seqlens_k=cu_seqlens_k_t,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            prefill_block_tables=block_tables,
        )

        hidden_states = self.model(input_ids_t, positions_t)
        logits = self.model.compute_logits(hidden_states)  # (N*num_verify, vocab)
        logits = logits.view(N, num_verify, -1)
        return logits
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(num_prefill_tokens=0, slot_mapping=slot_mapping[:bs],
                        context_lens=context_lens[:bs], decode_block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
