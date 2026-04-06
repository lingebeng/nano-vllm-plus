# EAGLE3 投机解码（Speculative Decoding）实现探究日志

## 一、项目背景与目标

在 nano-vllm-plus 推理引擎中实现基于 EAGLE3 的投机解码，使用 `RedHatAI/Qwen3-8B-speculator.eagle3`（1B 参数）作为 draft 模型，为 Qwen3-8B（8B 参数）目标模型加速解码。

**目标**: 在 decode 阶段通过"小模型猜测 + 大模型验证"的方式，将多次串行 decode 合并为一次并行 verify，从而提升吞吐量。

**EAGLE3 模型卡报告的基准**: mean accepted tokens = 2.13~2.48（K=3 时），对应约 1.5~2x 加速。

---

## 二、算法设计思路

### 2.1 核心算法：四阶段投机解码

每个投机解码 cycle（仅在 decode-only 批次时触发）：

1. **Target Decode**（1 token/seq）：正常 decode 前向传播 → 采样得到 `t0` + 从指定层提取辅助隐藏状态（aux hidden states）
2. **Draft**（K=3 次迭代）：draft 模型自回归生成 `d1, d2, d3`，利用 target 的辅助隐藏状态
3. **Target Verify**（K+1 tokens/seq）：将 `[t0, d1, d2, d3]` 作为 prefill 段喂入 target 模型，一次前向得到 K+1 个验证 logits
4. **Accept/Reject**：贪心接受——如果 `argmax(verify_logits[i]) == d_{i+1}` 则接受，否则用 target 的 argmax 作为修正 token

**结果**: 每个 cycle 序列推进 `1(t0) + accepted_count + 1(bonus/correction)` 个 token，最少 2 个，最多 K+2=5 个。

### 2.2 EAGLE3 Draft 模型架构

- `embed_tokens`: Embedding(151936, 4096) — 与 target 相同词表
- `fc`: Linear(3×4096=12288, 4096) — 融合 3 层辅助隐藏状态
- 1 个 `Eagle3DecoderLayer`：
  - 注意力输入维度 = 2×hidden_size = 8192（embedding + hidden state 拼接）
  - 32 个 query heads，8 个 KV heads，head_dim=128
  - MLP: 4096 → 12288 → 4096，SiLU 激活
- `lm_head`: Linear(4096, 32000) — draft 词表（32K，小于 target 的 152K）
- `d2t`: LongTensor(32000) — draft→target 词表偏移映射
- `norm_before_residual = True`

### 2.3 涉及的文件

| 文件 | 操作 | 用途 |
|------|------|------|
| `nanovllm/config.py` | 修改 | 增加 `speculative_model`, `num_speculative_tokens` 字段 |
| `nanovllm/models/eagle3.py` | **新建** | EAGLE3 draft 模型实现 |
| `nanovllm/models/qwen3.py` | 修改 | Target 模型提取辅助隐藏状态 |
| `nanovllm/engine/model_runner.py` | 修改 | 加载 draft 模型，`run_speculative` 核心逻辑 |
| `nanovllm/engine/llm_engine.py` | 修改 | 路由到投机解码路径 |
| `nanovllm/engine/scheduler.py` | 修改 | `postprocess_speculative` 处理变长 token 接受 |
| `nanovllm/engine/block_manager.py` | 修改 | `ensure_slots` 预分配 block |
| `bench/bench_speculative*.py` | **新建** | 性能基准测试 |

---

## 三、实现过程中遇到的 Bug 与修复

### Bug 1：辅助层索引错误（严重）

**现象**: 接受率极低（~3-6%），draft 模型几乎无法预测正确的 token。

**原因**: 初始实现使用 `{n//3-1, 2*n//3-1, n-1}` = `{11, 23, 35}` 提取辅助隐藏状态，但 vLLM/speculators 的默认值是 `{2, n//2, n-3}` = `{2, 18, 33}`。

draft 模型的 `fc` 层权重是在特定层索引上训练的。喂入错误层的隐藏状态等于给 draft 模型随机噪声输入。

**修复**:
```python
# 之前（错误）
self.aux_layer_ids = {n // 3 - 1, 2 * n // 3 - 1, n - 1}  # {11, 23, 35}

# 之后（正确）
self.aux_layer_ids = {2, n // 2, n - 3}  # {2, 18, 33}
```

**效果**: 修复后接受率仍然很低（~4%），说明还有其他 bug。

---

### Bug 2：Draft 模型"回声"问题（严重）

**现象**: Draft 模型在 K=3 步中生成完全相同的 token（如 `[d1, d1, d1]`）。

**原因**: 所有 K 步都传入了相同的 target aux_hidden（3×H 维），导致：
- `fc` 层输出在每步完全相同
- 当相同 token + 相同 aux_hidden 被喂入时，V 值完全一致
- 注意力输出 = V（无论 KV cache 内容如何），因此每步输出相同

**修复**: 第 0 步使用 target 的 aux_hidden（3×H），后续步使用 draft 模型自身的隐藏状态输出（H）：
```python
for k in range(K):
    if k == 0:
        d_aux = aux_hidden      # (N, 3*H) — 来自 target 模型
    else:
        d_aux = draft_hidden     # (N, H) — 来自 draft 模型上一步输出
```

Eagle3Speculator 的 `forward()` 中已有分支处理两种输入维度：
```python
if aux_hidden_states.shape[-1] == self.hidden_size:
    fused_hidden = aux_hidden_states           # H → 直接使用
else:
    fused_hidden = self.fc(aux_hidden_states)   # 3*H → 通过 fc 压缩
```

**效果**: Draft 预测变得多样化，但接受率仍只有 ~4-5%。

---

### Bug 3：缺少持久化 Draft KV Cache（严重）

**现象**: 修复 Bug 1 和 2 后接受率仍然很低（~4-5%）。

**原因**: 每个投机 cycle 都将 `past_kv = None` 重置，draft 模型没有跨 cycle 的上下文记忆。它每次只能看到当前步的 token + aux_hidden，无法利用历史信息进行预测。

**分析**: vLLM 为 draft 模型维护持久化的 KV cache，使 draft 模型可以"记住"之前生成过的 token，就像普通自回归生成一样。

**修复（第一版：逐序列循环）**:
```python
self.draft_kv_cache = {}  # seq_id → (K, V) tensors

# 在 run_speculative 中，逐个序列处理 draft 步
for i, seq in enumerate(decode_seqs):
    past_kv = self.draft_kv_cache.get(seq.seq_id)
    for k in range(K):
        _, _, past_kv = self.draft_model(d_input[i:i+1], d_aux[i:i+1], ...)
    self.draft_kv_cache[seq.seq_id] = past_kv
```

**效果**:
- 接受率大幅提升：19-31%
- Batch=1 加速 1.14x
- **但 Batch=16 只有 0.65x**（比基线更慢！），因为 N×K=48 次逐序列 forward 的 kernel 启动开销巨大

---

### Bug 4：逐序列循环导致大批次性能崩溃（性能问题）

**现象**: Batch=16 时投机解码比基线慢 35%（0.65x）。

**原因**: 逐序列处理 draft 步意味着 N×K = 16×3 = 48 次独立的小矩阵乘法。GPU 无法充分利用并行性，kernel 启动开销占主导。

**修复思路**: 将 N 个序列的 draft KV cache 合并为 batch 化张量，一次 forward 处理所有序列。

**挑战**: 不同序列的 draft KV cache 长度可能不同（因为接受的 token 数不同），需要处理 padding 和注意力 mask。

**实现**:

1. **`_get_batched_draft_kv()`**: 将各序列的 KV cache stack 到一个张量中，短的用零填充
```python
def _get_batched_draft_kv(self, decode_seqs):
    # 返回 (past_kv, kv_lengths)
    # past_kv: (batched_k, batched_v)，shape = (N, heads, max_len, dim)
    # kv_lengths: (N,) 每个序列的实际 KV 长度
    batched_k = torch.zeros(N, num_kv_heads, max_len, head_dim, ...)
    for i, kv in enumerate(kv_list):
        if kv is not None:
            batched_k[i, :, :L, :] = kv[0][0]  # 前 L 个位置是有效数据
    return (batched_k, batched_v), kv_lengths
```

2. **`_update_draft_kv()`**: 接受后裁剪被拒绝的 KV 条目
```python
def _update_draft_kv(self, decode_seqs, past_kv, num_accepted_cpu):
    for i, seq in enumerate(decode_seqs):
        na = num_accepted_cpu[i]
        to_remove = max(K - na - 1, 0)  # 注意 max 防止负数！
        keep = batched_k.shape[2] - to_remove
        self.draft_kv_cache[seq.seq_id] = (
            batched_k[i:i+1, :, :keep, :].contiguous(),
            batched_v[i:i+1, :, :keep, :].contiguous(),
        )
```

---

### Bug 5：`_update_draft_kv` 中 `K - na - 1` 下溢（崩溃风险）

**现象**: 当所有 K=3 个 draft token 都被接受时（na=K），程序会崩溃。

**原因**: `keep = total_len - (K - na - 1)`，当 na=K=3 时，`K - na - 1 = -1`，导致 `keep = total_len + 1`，越界。

**分析**:
- K=3 步 draft 在 KV cache 中添加了 3 个条目（对应 t0, d1, d2 的输入）
- 当 na=K 时，所有条目都有效，不需要移除任何条目
- 公式 `K - na - 1` 在 na=K 时得到 -1，需要 clamp 到 0

**修复**:
```python
to_remove = max(K - na - 1, 0)  # 添加 max(..., 0) 防止负数
```

---

### Bug 6：Draft 模型 RoPE 位置不连续（精度问题）

**现象**: 接受率低于预期（~19-31% vs 模型卡报告的 ~71-83%）。

**原因**: Draft 步使用 target 模型的位置 `len(seq) + k`，但 draft KV cache 中的条目位置与之不连续。

举例：如果第 1 个 cycle 接受了 1 个 draft token，序列增长了 3 个 token（t0 + d1 + correction）。Draft KV cache 中只有 2 个条目（位置 0, 1 对应 t0 和 d1），但 correction token（位置 2）不在 draft KV 中。下一个 cycle 的 draft 步使用位置 3，与 KV cache 中位置 1 之间有断层（缺少位置 2）。

RoPE 依赖位置连续性来编码相对位置关系，位置断层会导致 draft 模型的注意力计算不准确。

**修复**: 使用基于 draft KV cache 长度的连续位置，而非 target 序列位置：
```python
# 之前（位置可能不连续）
d_positions = torch.tensor([len(seq) + k for seq in decode_seqs], ...)

# 之后（基于 draft KV 长度，保证连续）
d_positions = kv_lengths + k  # kv_lengths 是每个序列 draft KV 的实际长度
```

---

### Bug 7：零填充 KV 导致注意力权重稀释（精度问题）

**现象**: 批量化后的接受率可能下降。

**原因**: 当不同序列的 draft KV cache 长度不同时，短序列的 KV 用零填充。在 `F.scaled_dot_product_attention` 中：
- 零 K 向量的点积 = 0
- softmax(0) = exp(0)/Z = 非零权重
- 虽然零 V 的贡献为 0，但非零注意力权重会稀释分配给实际位置的权重
- 结果：注意力输出的幅度被缩小，影响 draft 预测质量

**修复**: 在 `Eagle3Attention` 中添加 boolean attention mask：
```python
def forward(self, hidden_states, positions, past_kv=None, kv_valid_lens=None):
    ...
    if kv_valid_lens is not None and past_kv is not None:
        total_kv_len = k.shape[2]
        past_len = past_kv[0].shape[2]
        pos_range = torch.arange(total_kv_len, device=k.device)
        # 有效位置 = 实际 past 条目 OR 当前步新条目
        past_valid = pos_range.unsqueeze(0) < kv_valid_lens.unsqueeze(1)
        new_valid = pos_range.unsqueeze(0) >= past_len
        mask = past_valid | new_valid
        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (bsz, 1, 1, total_kv_len)

    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, ...)
```

---

## 四、最终性能结果

### 真实文本场景（Qwen3-8B + EAGLE3, K=3, max_tokens=512）

| 批次大小 | 基线 (tok/s) | 投机 (tok/s) | 平均 token/cycle | 接受率 | 加速比 |
|---------|-------------|-------------|-----------------|--------|--------|
| 1       | 34.9        | 47.5        | 2.94            | 31%    | **1.36x** |
| 4       | 132.3       | 157.2       | 2.74            | 25%    | **1.19x** |
| 16      | 500.8       | 588.2       | 3.15            | 38%    | **1.17x** |

### BS=1 不同输出长度

| max_tokens | 基线 (tok/s) | 投机 (tok/s) | 加速比 |
|-----------|-------------|-------------|--------|
| 256       | 35.9        | 39.6        | **1.10x** |
| 1024      | 35.9        | 46.9        | **1.31x** |

### 进化过程

| 阶段 | Batch=1 | Batch=16 | 主要变化 |
|------|---------|----------|---------|
| 初始实现 | ~1.0x | ~1.0x | 辅助层索引错误，无持久 KV |
| 修复辅助层 + 回声 | ~1.0x | ~1.0x | 接受率 ~4-5%，太低无加速 |
| 添加持久 KV（逐序列） | 1.14x | 0.65x | 接受率 19-31%，但大批次严重退化 |
| 批量化 + 位置修复 + 注意力 mask | **1.36x** | **1.17x** | 所有批次正向加速 |

---

## 五、关键设计决策与思考

### 5.1 为什么需要持久化 Draft KV Cache？

Draft 模型本质上是自回归生成，需要"记住"之前生成过的 token。没有持久 KV cache 时，每个 cycle 的 draft 模型只看到当前 1 个 token + target 的辅助隐藏状态，信息严重不足，无法做出有意义的预测。

添加持久 KV cache 后，draft 模型可以回顾所有历史已接受的 token 上下文，接受率从 ~4% 跃升至 ~30%。

### 5.2 为什么不保留被拒绝的 KV 条目？

曾考虑过"保留所有 K 条目不裁剪"的方案来保证所有序列 KV 长度一致（避免 padding），但分析后放弃：

- 被拒绝的 draft token 对应错误的隐藏状态
- 随着 cycle 累积，错误条目越来越多（~2/3 是错误的）
- Draft 模型会在注意力中关注这些"幻觉上下文"，导致预测质量急剧下降

最终选择裁剪 + padding + 注意力 mask 的方案，代价是实现更复杂，但保证了 KV cache 的正确性。

### 5.3 为什么 Draft 位置要用连续编号？

RoPE 编码的是**相对位置**。如果 draft KV cache 中的位置是 `[0, 1, 5, 8]`（因为中间的 correction token 不在 cache 中），RoPE 会认为相邻条目之间间距很大，导致注意力权重计算不准确。

使用基于 draft KV 长度的连续编号 `[0, 1, 2, 3, ...]` 可以保证 RoPE 一致性。虽然这些位置与 target 模型的位置不同，但 draft 模型只是一个预测器，位置一致性比位置准确性更重要。

### 5.4 接受率与理论值的差距

我们实现的接受率（25-38%，avg 2.74-3.15 tokens/cycle）与模型卡报告（2.13-2.48 mean accepted tokens，对应 avg 4.13-4.48 tokens/cycle）仍有差距。主要原因：

1. **缺少 Draft Prefill**: 我们的 draft 模型在第一个 cycle 时 KV cache 是空的，对 prompt 没有任何上下文。vLLM 在 prefill 阶段就让 draft 模型处理完整 prompt，建立初始上下文。
2. **Correction Token 上下文缺失**: 每个 cycle 的 correction/bonus token 不在 draft KV cache 中，造成上下文"洞"。
3. **温度采样**: 我们使用 temperature=0.6 采样，但验证用 argmax 贪心比较，温度引入的随机性会降低匹配率。

---

## 六、代码结构总结

### 核心流程（`model_runner.py::run_speculative`）

```
┌─────────────────────────────────────────────┐
│ Phase 1: Target Decode                       │
│   input_ids, positions = prepare(decode_seqs)│
│   hidden, aux_hidden = model(input_ids, pos, │
│                               aux_layer_ids) │
│   t0 = argmax(logits)                        │
├─────────────────────────────────────────────┤
│ Phase 2: Draft K Tokens (Batched)            │
│   past_kv, kv_lengths = get_batched_kv()     │
│   for k in range(K):                         │
│     positions = kv_lengths + k  ← 连续位置    │
│     h, logits, past_kv = draft_model(         │
│       input, aux, positions, past_kv,         │
│       kv_valid_lens=kv_lengths+k)  ← 注意力mask│
│     draft_tokens[k] = greedy_sample(logits)   │
├─────────────────────────────────────────────┤
│ Phase 3: Target Verify                       │
│   verify_input = [t0, d1, ..., dK] per seq   │
│   verify_logits = model(verify_input)         │
├─────────────────────────────────────────────┤
│ Phase 4: Accept/Reject                       │
│   matches = (target_preds[:,:K] == drafts)    │
│   num_accepted = cumprod(matches).sum(dim=1)  │
│   → 每序列接受 2~K+2 个 token                 │
├─────────────────────────────────────────────┤
│ Phase 5: Update Draft KV Cache               │
│   裁剪被拒绝的条目，保留 t0 + accepted drafts  │
└─────────────────────────────────────────────┘
```

### Draft 模型数据流（`eagle3.py`）

```
input_ids ──→ embed_tokens ──→ embeds ─────┐
                                            ├──→ [embeds, hidden] ──→ Attention(2H→H) ──→ MLP ──→ hidden_states
aux_hidden ──→ fc(3H→H) or identity ──→ hidden ┘                         ↑
                                                                     RoPE + KV Cache
                                                                     + Attention Mask

hidden_states ──→ norm ──→ lm_head(H→32K) ──→ draft_logits
draft_logits ──→ argmax ──→ draft_id + d2t[draft_id] ──→ target_id
```

---

## 七、性能优化：从 1.36x 到 2.55x

### 7.1 优化前基线（第四节结果）

| 批次大小 | 基线 (tok/s) | 投机 (tok/s) | 平均 token/cycle | 加速比 |
|---------|-------------|-------------|-----------------|--------|
| 1       | 34.9        | 47.5        | 2.94            | 1.36x  |
| 4       | 132.3       | 157.2       | 2.74            | 1.19x  |
| 16      | 500.8       | 588.2       | 3.15            | 1.17x  |

### 7.2 瓶颈分析

通过 profiling 分析原始 cycle 各阶段耗时（BS=1）：

| 阶段 | 耗时 | 说明 |
|------|------|------|
| Phase 1: Target Decode | ~29ms | 目标模型前向（1 token）+ 采样 + 辅助隐藏状态提取 |
| Phase 2: Draft × 3 | ~5ms | Draft 模型 3 步自回归 |
| Phase 3: Target Verify | ~13ms | 目标模型前向（K+1=4 tokens）作为 prefill 段 |
| Phase 4: Accept/Reject + KV 更新 | ~15ms | 包括 CPU-GPU 同步、内存分配、KV 裁剪 |
| **合计** | **~62ms** | |

**关键发现**：Phase 1（Target Decode）占 cycle 时间的 47%，但它只产出 1 个 token（t0）。Phase 3 已经通过 verify 产出了相同信息——verify_logits[0] 的 argmax 就等于 t0。如果能复用上一 cycle 的 verify 结果，就可以完全跳过 Phase 1。

### 7.3 优化 1：Merged Mode（跳过 Phase 1）

**核心思想**：在 Phase 3 verify 完成后，除了得到 accept/reject 结果，还得到了 correction token 和对应位置的辅助隐藏状态。这些信息正好是下一 cycle Phase 1 的输出。

**实现**：

1. 在 `__init__` 中增加 `self._prev_correction = {}` 存储每个序列上一 cycle 的 correction 信息

2. 修改 `_run_verify()` 使其同时提取 verify 阶段的辅助隐藏状态（通过 forward hooks）：
```python
def _run_verify(self, decode_seqs, start_list, draft_tokens_list, base_offset=0):
    # ...
    # 使用 hooks 提取 aux_hidden
    aux_captures = []
    hooks = []
    for layer_idx in sorted(self.aux_layer_ids):
        def make_hook(idx):
            def hook_fn(module, input, output):
                h, r = output
                aux_captures.append((idx, (h + r).detach()))
            return hook_fn
        hooks.append(self.model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
    hidden_states = self.model(input_ids_t, positions_t)
    for h in hooks:
        h.remove()
    # ...
    return logits, verify_aux  # (N, K+1, vocab), (N, K+1, 3*H)
```

3. 修改 `run_speculative()` 主流程：
```python
def run_speculative(self, decode_seqs):
    merged = all(seq.seq_id in self._prev_correction for seq in decode_seqs)

    if merged:
        # 跳过 Phase 1，复用上一 cycle 的 correction
        start_tokens = [self._prev_correction[seq.seq_id][0] for ...]
        start_aux = [self._prev_correction[seq.seq_id][1] for ...]
    else:
        # 首次 cycle：正常执行 Phase 1
        ...

    # Phase 2: Draft（不变）
    # Phase 3: Verify（merged 时 base_offset=-1）
    base_offset = -1 if merged else 0
    verify_logits, verify_aux = self._run_verify(..., base_offset)

    # Phase 4: Accept/Reject
    for i, seq in enumerate(decode_seqs):
        na = num_accepted_cpu[i]
        if merged:
            seq_tokens = [draft_tokens_list[k][i] for k in range(na)]
        else:
            seq_tokens = [start_list[i]] + [draft_tokens_list[k][i] for k in range(na)]
        seq_tokens.append(correction)

    # 保存 correction 供下一 cycle 使用
    for i, seq in enumerate(decode_seqs):
        na = num_accepted_cpu[i]
        self._prev_correction[seq.seq_id] = (accepted[i][-1], verify_aux[i, na])
```

4. Merged 模式的位置计算：
```python
# 原始模式：verify 从 len(seq) 开始（Phase 1 已写入 KV cache）
# Merged 模式：verify 从 len(seq)-1 开始（correction 的 KV 还未写入）
base_pos = len(seq) + base_offset
```

**Merged 模式的 token 产出**：
- 原始模式：t0 + na + correction = na + 2 tokens/cycle
- Merged 模式：na + correction = na + 1 tokens/cycle（t0 不再单独产出，因为上一 cycle 的 correction 就是本 cycle 的"起始 token"，已被上一 cycle 计数）

**关键权衡**：虽然每 cycle 少产出 1 个 token，但 cycle 时间从 ~62ms 降至 ~37ms（节省 ~29ms Phase 1），净效果正向。

### 7.4 优化 2：Draft Prefill（提升接受率）

**问题**：原始实现中 draft 模型的 KV cache 从空开始，在第一个 decode cycle 时对 prompt 没有任何上下文记忆。

**实现**：在 target 模型 prefill 阶段，通过 forward hooks 提取辅助隐藏状态，然后将完整 prompt 通过 draft 模型建立初始 KV cache。

```python
# 在 run() 中，prefill 时提取 aux_hidden
def run(self, prefill_seqs, chunk_sizes, decode_seqs):
    need_draft_prefill = has_prefill and self.draft_model is not None
    if need_draft_prefill:
        # 使用 hooks 提取辅助隐藏状态
        aux_captures = []
        hooks = []
        for layer_idx in sorted(self.aux_layer_ids):
            hooks.append(self.model.model.layers[layer_idx].register_forward_hook(...))
        hidden_states = self.run_model(input_ids, positions, has_prefill)
        for h in hooks:
            h.remove()
        aux_hidden = torch.cat([c[1] for c in aux_captures], dim=-1)
    # ...
    if need_draft_prefill:
        self._accumulate_draft_prefill(prefill_seqs, chunk_sizes, aux_hidden)
```

支持 chunked prefill——逐 chunk 累积 aux_hidden，当 prefill 完成时合并并运行 draft prefill：

```python
def _accumulate_draft_prefill(self, prefill_seqs, chunk_sizes, aux_hidden):
    for seq, chunk in zip(prefill_seqs, chunk_sizes):
        self._prefill_aux_chunks[seq.seq_id].append(chunk_aux)
        if seq.num_computed_tokens + chunk >= seq.num_prompt_tokens:
            full_aux = torch.cat(self._prefill_aux_chunks.pop(seq.seq_id), dim=0)
            self._draft_prefill(seq, full_aux)

def _draft_prefill(self, seq, prompt_aux_hidden):
    prompt_ids = torch.tensor(seq.prompt_token_ids, ...).unsqueeze(0)
    positions = torch.arange(seq.num_prompt_tokens, ...).unsqueeze(0)
    aux = prompt_aux_hidden.unsqueeze(0)
    _, _, draft_kv = self.draft_model(prompt_ids, aux, positions, past_kv=None)
    self.draft_kv_cache[seq.seq_id] = (draft_kv[0], draft_kv[1])
```

### 7.5 优化 3：Forward Hooks 替代 aux_layer_ids 参数

**问题**：最初通过给 `Qwen3Model.forward()` 传入 `aux_layer_ids` 参数来提取辅助隐藏状态。但这改变了模型的输入/输出签名，导致 `torch.compile` 反复重编译（RMSNorm 看到不同 rank 的张量：有 aux_layer_ids 时返回 tuple，没有时返回 tensor）。

**修复**：使用 PyTorch forward hooks，在不改变模型签名的情况下捕获中间层输出：

```python
for layer_idx in sorted(self.aux_layer_ids):
    def make_hook(idx):
        def hook_fn(module, input, output):
            h, r = output  # Qwen3DecoderLayer 返回 (hidden_states, residual)
            aux_captures.append((idx, (h + r).detach()))
        return hook_fn
    hooks.append(self.model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
```

`h + r` 等价于 `hidden_states + residual`，即经过 input_layernorm 融合后的真实隐藏状态。与 `aux_layer_ids` 代码路径中的 `hidden_states + residual` 完全一致。

### 7.6 未采用的优化：Correction Token 回填

**尝试**：每 cycle 结束后，将 correction token 通过 draft 模型前向传播，填补 draft KV cache 中的上下文空洞。

**结果**：额外的 draft 前向传播（~2ms）+ KV 拷贝开销抵消了接受率的微小改善（~0.2 tokens/cycle）。

**分析**：在 merged 模式下，correction token 自然会在下一 cycle 的 Phase 2 第 0 步被处理（作为 `start_tokens` 输入），其 KV 条目会正常添加到 draft KV cache 中。因此 KV cache 中唯一的"空洞"是当前 cycle 的 correction，它会在下一 cycle 立即被填补。在这种情况下，显式回填的边际收益不足以覆盖其开销。

### 7.7 优化后性能结果

Qwen3-8B + EAGLE3, K=3, max_tokens=512, enforce_eager=True：

| 批次大小 | 基线 (tok/s) | 投机 (tok/s) | 平均 token/cycle | 加速比 |
|---------|-------------|-------------|-----------------|--------|
| 1       | 33.4        | 85.1        | 3.41            | **2.55x** |
| 4       | 123.1       | 160.1       | 2.67            | **1.30x** |
| 16      | 463.0       | 664.8       | 2.98            | **1.44x** |

### 7.8 优化后 Cycle 耗时分解（BS=1, Merged Mode）

| 阶段 | 耗时 | 说明 |
|------|------|------|
| Phase 1 (Merged 设置) | 0.1ms | 从 `_prev_correction` 读取 tensor |
| Phase 2 (Draft × 3) | 5.3ms | 3 步 draft 自回归 |
| Phase 3 (Verify) | 31.2ms | 目标模型前向（4 tokens prefill）+ hooks 提取 aux |
| Phase 4 (Accept + KV 更新) | 0.25ms | 向量化 accept/reject + KV 裁剪 |
| **合计** | **~37ms** | 比优化前 62ms 减少 40% |

### 7.9 进化总结

| 阶段 | BS=1 加速 | BS=16 加速 | 主要变化 |
|------|----------|----------|---------|
| 初始实现 | ~1.0x | ~1.0x | 辅助层索引错误，无持久 KV |
| 修复辅助层 + 回声 | ~1.0x | ~1.0x | 接受率 ~4-5%，太低无加速 |
| 持久 KV（逐序列） | 1.14x | 0.65x | 接受率 19-31%，大批次退化 |
| 批量化 + 位置修复 + 注意力 mask | 1.36x | 1.17x | 所有批次正向加速 |
| **Merged Mode + Draft Prefill + Hooks** | **2.55x** | **1.44x** | 跳过 Phase 1，draft 有 prompt 上下文 |

---

## 八、核心流程（优化后）

### 优化后流程（`model_runner.py::run_speculative`，Merged Mode）

```
┌─────────────────────────────────────────────┐
│ Phase 1: Merged Setup (0.1ms)               │
│   start_tokens = _prev_correction[seq][0]   │
│   start_aux = _prev_correction[seq][1]      │
│   (首次 cycle 回退到 target decode + hooks) │
├─────────────────────────────────────────────┤
│ Phase 2: Draft K Tokens (5.3ms)             │
│   past_kv, kv_lengths = get_batched_kv()    │
│   for k in range(K):                        │
│     positions = kv_lengths + k              │
│     h, logits, past_kv = draft_model(       │
│       input, aux, positions, past_kv,       │
│       kv_valid_lens=kv_lengths+k)           │
│     draft_tokens[k] = greedy_sample(logits) │
├─────────────────────────────────────────────┤
│ Phase 3: Target Verify (31ms)               │
│   verify_input = [start, d1, ..., dK]       │
│   base_pos = len(seq) - 1  (merged)         │
│   verify_logits, verify_aux = _run_verify(  │
│     ..., base_offset=-1)                    │
│   # hooks 同时提取 aux_hidden 供下一 cycle  │
├─────────────────────────────────────────────┤
│ Phase 4: Accept/Reject (0.25ms)             │
│   matches = (target_preds[:,:K] == drafts)  │
│   num_accepted = cumprod(matches).sum(dim=1)│
│   accepted = [d1..d_na, correction]         │
│   _prev_correction[seq] = (correction,      │
│                             verify_aux[na])  │
│   _update_draft_kv(trim rejected entries)   │
└─────────────────────────────────────────────┘
```

### Merged vs Original 模式对比

```
Original (首次 cycle):
  Phase 1 (29ms) → Phase 2 (5ms) → Phase 3 (13ms) → Phase 4 (15ms) = 62ms
  产出: t0 + na + correction = na + 2 tokens

Merged (后续 cycle):
  Phase 1 (0.1ms) → Phase 2 (5ms) → Phase 3 (31ms) → Phase 4 (0.25ms) = 37ms
  产出: na + correction = na + 1 tokens
  （correction 复用为下一 cycle 的起始 token）

注: Phase 3 在 merged 模式下比 original 慢（31ms vs 13ms），因为 verify 使用 hooks
提取 aux_hidden，而 original 的 Phase 3 不需要 hooks（aux 在 Phase 1 已提取）。
但总体仍快 40%，因为完全跳过了 Phase 1 的 target decode。
```
