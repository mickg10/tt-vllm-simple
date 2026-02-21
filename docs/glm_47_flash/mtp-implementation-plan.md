# MTP Implementation Plan: GLM-4.7-Flash on T3K

**Date**: 2026-02-14
**Status**: PLAN — awaiting implementation
**Sources**: kernel-optimization-research.md, algo-paths-to-30tps.md, mtp-code-analysis.md, mtp-papers-research.md

---

## Executive Summary

30 tok/s at bs=1 on Wormhole is NOT achievable (op dispatch overhead alone ~25ms vs 33ms target).
Realistic ceiling: **~12-16 tok/s** with MTP + all kernel optimizations.

The strategy is phased: quick wins first (each independently valuable), then MTP (biggest
single lever), then compound optimizations.

### Current Performance

| Metric | Value | Target |
|--------|-------|--------|
| Decode bs=1 | 7.0 tok/s (143ms ITL) | 30 tok/s (33ms) |
| Decode bs=32 agg | 208.3 tok/s | 150 tok/s **EXCEEDED** |
| Prefill 1k bs=1 | 205 tok/s | 1000 tok/s |

### Projected Path

| Phase | Expected ITL | tok/s | Cumulative Speedup |
|-------|-------------|-------|-------------------|
| Current baseline | 143ms | 7.0 | 1.0x |
| A: Quick wins (-5-8ms) | ~135ms | 7.4 | 1.06x |
| B: MTP K=1 (1.8x effective) | 135ms / 1.8 = 75ms eff | 13.3 | 1.9x |
| C: Multi-step MTP K=3-5 | 145ms / 2.5 = 58ms eff | ~17 | 2.4x |
| D: Async CCL (-7ms) | 128ms / 2.5 = 51ms eff | ~20 | 2.8x |

---

## Phase A: Quick Wins (0-3 days)

Low-effort changes that each shave 1-5ms. Independent of each other — can test sequentially.

### A1: BFP4 Expert Weights (10 min, env var only)

**What**: Change expert weight dtype from BF8 to BFP4.
**How**: Set `GLM4_MOE_LITE_EXPERTS_TT_DTYPE=bfloat4_b` in `.env.glm47`.
**Expected**: 0.4-1ms at bs=1 (experts mostly skipped), 5ms at bs=32 (all experts read).
**Risk**: Quality degradation — must validate coherence. DeepSeek V3 uses BFP4 for gate+down successfully.
**Rollback**: Change env var back.
**Validation**: bench_decode.py bs=1 + bs=32, plus coherence check (200 tok generation).

### A2: BFP4 Shared MLP Weights (10 min, env var only)

**What**: Change dense/shared MLP dtype from BF16 to BFP4.
**How**: Set `GLM4_MOE_LITE_DENSE_TT_DTYPE=bfloat4_b` in `.env.glm47`.
**Expected**: 0.5-1ms at bs=1 (15MB → 3.75MB per layer per device).
**Risk**: Medium — shared MLP is on the critical path. DeepSeek V3 uses BFP4 for all MLP weights.
**Rollback**: Change env var back.
**Dependencies**: Delete weight cache to force reconversion: `rm -rf /home/ttuser/.cache/tt_metal_cache/glm*`

### A3: Pre-TILE Router Bias (30 min code change)

**What**: Pre-convert router bias to TILE layout during init instead of per-step.
**Where**: `decoder_layer_tt.py` — the `to_layout(ROW_MAJOR → TILE)` + `repeat` + `add` sequence in the MoE router (ops 36-38 in kernel research).
**Expected**: ~3ms (eliminates 3 ops × 46 MoE layers = 138 ops, at ~8µs each = 1.1ms + layout conversion overhead).
**How**: During model init, convert bias tensor to TILE layout and store. Remove per-step conversion.
**Risk**: Low — pure performance, no functional change.

### A4: nlp_concat_heads_decode (1 hr code change)

**What**: Replace permute+reshape+permute sequence in attention output with `ttnn.experimental.nlp_concat_heads_decode`.
**Where**: `decoder_layer_tt.py` — attention output path (ops 28-30 in kernel research).
**Expected**: ~2ms (eliminates 2 ops × 47 layers = 94 ops).
**How**: Use `ttnn.experimental.nlp_concat_heads_decode(attn_output)` instead of manual reshape.
**Risk**: Medium — need to verify API compatibility with MLA's unusual head dimensions (20 heads × 256 v_head_dim).
**Prerequisite**: Verify API exists in current tt-metal version and supports the tensor shapes.

---

## Phase B: MTP K=1 Implementation (1-2 weeks)

The single largest lever. Predicts 1 additional token per decode step.
At ~85-90% acceptance rate: **1.8x effective throughput** (7 → 12.6 tok/s).

### Architecture

```
Main model (layers 0-46) → hidden_state[t] + token[t+1]
                              ↓
MTP layer 47: enorm(embed(token[t+1])) ⊕ hnorm(hidden[t]) → eh_proj → decoder_layer → shared_head → logits[t+2]
                              ↓
If token[t+2] matches verified token: accept (skip one decode step)
If mismatch: reject, use verified token, discard draft
```

### Implementation Steps

#### B1: Load MTP Weights onto Device

**Where**: `tt-metal/models/demos/glm4_moe_lite/tt/weights.py`
**What**: Remove the `num_layers=47` filter for MTP-specific weights. Load layer 47 weights separately.
**Details**:
- Currently `load_glm_lazy_state_dict(num_layers=47)` filters out all `model.layers.47.*` keys
- Need a separate loading path: `load_glm_lazy_state_dict(num_layers=48)` then extract layer 47
- Layer 47 has 212 weight keys (same as a regular MoE layer + MTP-specific: embed_tokens, enorm, hnorm, eh_proj, shared_head)
- Embed_tokens and shared_head.head may be shared with main model (check weight tying)

#### B2: Implement TT MTP Forward Pass

**Where**: New file `tt-metal/models/demos/glm4_moe_lite/tt/mtp_layer_tt.py`
**What**: TT implementation of `Glm4MoeMultiTokenPredictorLayer.forward()`

```python
# Pseudocode for MTP forward on TT device
def mtp_forward(input_ids_next, hidden_states_last_layer, mtp_weights, device):
    # 1. Embed the predicted next token
    embeds = ttnn.embedding(input_ids_next, mtp_weights.embed_tokens)
    # 2. Mask position 0 (not needed by MTP)
    # 3. Normalize both inputs
    embeds_normed = rms_norm(embeds, mtp_weights.enorm)
    hidden_normed = rms_norm(hidden_states_last_layer, mtp_weights.hnorm)
    # 4. Concatenate [embeds, hidden] along last dim → [1, 1, 1, 4096]
    combined = ttnn.concat([embeds_normed, hidden_normed], dim=-1)
    # 5. Project back to hidden_size → [1, 1, 1, 2048]
    hidden = ttnn.linear(combined, mtp_weights.eh_proj)
    # 6. Run through full decoder layer (reuse existing decoder_layer_tt code!)
    hidden = decoder_layer_forward(hidden, mtp_weights.mtp_block, ...)
    # 7. Norm + LM head → logits
    hidden = rms_norm(hidden, mtp_weights.shared_head_norm)
    logits = ttnn.linear(hidden, mtp_weights.shared_head_head)
    return logits
```

**Key decisions**:
- The MTP decoder layer is identical to layers 1-46 (MoE). Reuse existing `TtDecoderLayer` class.
- MTP needs its own KV cache slot (separate from main model's 47 layers).
- MTP attention sees only the current position (no history needed for K=1).
  Actually: MTP attention CAN see prior positions. For K=1, the MTP layer maintains its own KV cache across decode steps. This is important for quality.

#### B3: Extract Hidden States from Main Model

**Where**: `tt-metal/models/demos/glm4_moe_lite/tt/model_tt.py`
**What**: After the last transformer layer (46), capture the hidden state BEFORE the final norm/LM head.

Currently the decode loop is:
```python
for layer_idx in range(self.num_layers_to_run):  # 0..46
    hidden = self.layers[layer_idx](hidden, ...)
hidden = self.norm(hidden)
logits = self.lm_head(hidden)
```

Need to capture `hidden` after the loop but before norm:
```python
for layer_idx in range(self.num_layers_to_run):
    hidden = self.layers[layer_idx](hidden, ...)
last_hidden = hidden  # <-- capture for MTP
hidden = self.norm(hidden)
logits = self.lm_head(hidden)
# MTP forward
mtp_logits = self.mtp_forward(next_token_ids, last_hidden)
```

#### B4: On-Device MTP Propose + Sample

**Where**: `model_tt.py` (extend decode method)
**What**: After main model produces token[t+1], run MTP to produce draft token[t+2].

Two approaches:
- **(a) Device-side propose** (preferred): Run MTP on device, argmax on device, return both tokens to host.
  Pro: No host-device round trip for MTP. Con: More complex device code.
- **(b) Host-side propose**: Return hidden states to host, run MTP on CPU/GPU.
  Pro: Simpler. Con: Host-device transfer latency kills the benefit.

**Recommendation**: (a) Device-side. The whole point is to amortize the per-step overhead.

#### B5: Integrate with vLLM TT Model Runner

**Where**: `vllm/vllm/worker/tt_model_runner.py`
**What**: Handle the case where the model returns 2 tokens per step instead of 1.

Options:
1. **Simple greedy approach** (recommended for K=1):
   - Model returns `(verified_token, draft_token, draft_logprobs)`
   - Runner feeds both tokens to scheduler
   - Next step, model verifies draft_token while producing new main token
   - If draft was correct: scheduler advances by 2 positions
   - If draft was wrong: discard, recompute from verified token

2. **Full spec decode integration** (vLLM v1 `EagleProposer`):
   - More complex, supports K>1, tree verification
   - Required for Phase C (multi-step MTP)
   - Overkill for K=1

**Recommendation**: Start with option 1 for K=1. Refactor to option 2 when implementing Phase C.

#### B6: KV Cache Management for MTP

**What**: MTP layer needs its own KV cache (separate from main model's 47 layers).
**Where**: `tt_model_runner.py` (cache allocation) and `model_tt.py` (cache management)
**Details**:
- Allocate 1 additional KV cache layer (total: 48 instead of 47)
- MTP KV cache uses same page table as main model
- When draft token is rejected, MTP KV cache must be rolled back (delete the rejected entry)
- For K=1, rollback is trivial (just decrement the sequence length counter)

### B Phase Risks

1. **Trace compatibility**: MTP forward must work within the traced decode path. If MTP execution breaks traces, may need to run MTP outside the trace (adding ~3ms of untraced overhead).
2. **KV cache rollback**: On rejection, must correctly invalidate the MTP cache entry. Incorrect rollback = attention corruption.
3. **Embedding sharing**: If `embed_tokens` is weight-tied with main model, must ensure no duplicate device memory.
4. **Warmup**: Need to capture MTP trace for each batch bucket (5 additional trace captures → ~90s extra warmup).

---

## Phase C: Multi-Step MTP K=3-5 (2-3 weeks, after Phase B)

Feed MTP output back as input to generate a chain of K draft tokens.

### Architecture

```
Main model → hidden[t] → MTP → draft[t+2] → MTP → draft[t+3] → ... → MTP → draft[t+K+1]
                                    ↑                    ↑
                              embed(draft[t+1])    embed(draft[t+2])
```

### Expected Performance

| K | Overhead (K/47) | E[tok/step] α=0.80 | Effective tok/s | Speedup |
|---|----------------|---------------------|-----------------|---------|
| 1 | 2.1% | 1.80 | 12.6 | 1.8x |
| 3 | 6.4% | 2.95 | ~17 | 2.4x |
| 5 | 10.6% | 3.69 | ~20 | 2.9x |

Note: Acceptance rate decays at each position because MTP was trained with teacher-forced
inputs, not its own predictions. Without FastMTP-style fine-tuning, expect α ≈ [0.87, 0.73,
0.60, 0.50, 0.42] for positions 1-5.

### Key Requirements

1. **Batched verification**: Pack K+1 tokens into a single forward pass as a "fake batch".
   On memory-bound T3K, verifying K+1 tokens costs ≈ same as 1 token (same weight reads).
   This is the key multiplier — MTP draft overhead (K × 3ms) amortized against single verify step.

2. **Full vLLM spec decode integration**: Need `EagleProposer`-compatible interface for TT.
   - Propose K draft tokens on device
   - Return draft tokens + draft logprobs to host
   - Host runs rejection sampling (trivial for greedy: just compare)
   - Accepted tokens advance, rejected tokens rolled back

3. **KV cache for multi-step**: MTP layer KV cache grows with K drafts. On rejection at
   position i, must roll back positions i through K.

### Implementation Approach

1. Extend Phase B's on-device MTP to loop K times
2. Each iteration: embed(draft[i]) + hidden_from_mtp → MTP forward → draft[i+1]
3. Collect all K draft tokens on device
4. Run single batched verification forward pass (bs=K+1)
5. Return verified token + accepted drafts to host

---

## Phase D: Async CCL (1-2 weeks, independent of B/C)

### What
Overlap `all_reduce` communication with subsequent compute. Currently all_reduce is
synchronous — each layer waits for the reduce to complete before starting the next matmul.

### Expected Savings
- 14ms total all_reduce time across 47 layers
- With async overlap: ~7ms recoverable (50% overlap with next layer's first matmul)

### Implementation
- Use tt-metal's async CCL primitives (if available in current version)
- Requires careful dependency management in traced execution
- May conflict with trace-based execution model

### Risk
- High complexity, may not be compatible with current trace infrastructure
- Should be attempted only after Phase B delivers results

---

## Phase Order & Dependencies

```
Phase A1 (BFP4 experts)     ─┐
Phase A2 (BFP4 shared MLP)  ─┤ Independent, test sequentially
Phase A3 (pre-TILE bias)    ─┤
Phase A4 (nlp_concat_heads) ─┘
          ↓
Phase B (MTP K=1)           ── THE BIG WIN
          ↓
Phase C (Multi-step MTP)    ── Builds on B
          ↓
Phase D (Async CCL)         ── Independent, can overlap with C
```

### Decision Points

- **After A1/A2**: If BFP4 causes quality degradation, skip and move to A3.
- **After B**: Measure actual acceptance rate. If α < 0.80, investigate why before Phase C.
  Consider: is the MTP layer producing coherent tokens? Check with temperature=0 greedy.
- **After C with K=3**: If multi-step acceptance decays too fast (α₃ < 0.50), cap at K=2-3
  and focus on Phase D instead.

---

## Files to Modify

### Phase A
- `docker_tt/dev/.env.glm47` — env var changes
- `tt-metal/models/demos/glm4_moe_lite/tt/decoder_layer_tt.py` — pre-TILE bias, nlp_concat_heads

### Phase B (new + modified)
- **NEW**: `tt-metal/models/demos/glm4_moe_lite/tt/mtp_layer_tt.py` — MTP forward pass
- `tt-metal/models/demos/glm4_moe_lite/tt/weights.py` — load layer 47 weights
- `tt-metal/models/demos/glm4_moe_lite/tt/model_tt.py` — hidden state capture, MTP integration
- `vllm/vllm/worker/tt_model_runner.py` — 2-token output handling, KV cache management
- `vllm/vllm/transformers_utils/configs/glm4_moe_lite.py` — expose num_nextn_predict_layers

### Phase C
- Same as Phase B, extended for K>1 loop
- `vllm/vllm/v1/spec_decode/` — TT-compatible proposer (or extend tt_model_runner)

### Phase D
- `tt-metal/models/demos/glm4_moe_lite/tt/decoder_layer_tt.py` — async all_reduce calls
- Potentially tt-metal core library changes

---

## Success Criteria

| Phase | Metric | Pass | Fail |
|-------|--------|------|------|
| A (combined) | Decode bs=1 ITL | < 138ms | > 140ms or quality degradation |
| B | MTP acceptance rate | > 0.80 | < 0.70 |
| B | Effective bs=1 tok/s | > 11 | < 9 (MTP overhead too high) |
| C (K=3) | Effective bs=1 tok/s | > 15 | < 12 (acceptance decay too fast) |
| D | Decode bs=1 ITL | < 128ms (pre-MTP) | No improvement |
| All combined | Effective bs=1 tok/s | > 14 | — |

---

## Revised Targets

Based on research, the 30 tok/s target is **not achievable on Wormhole**. Revised targets:

| Metric | Original Target | Revised Target | Rationale |
|--------|----------------|----------------|-----------|
| Decode bs=1 | 30 tok/s | **14-16 tok/s** | Hardware ceiling with MTP |
| Decode bs=32 agg | 150 tok/s | **208 tok/s** ✅ DONE | Already exceeded |
| Prefill 1k bs=1 | 1000 tok/s | **205 tok/s** | Separate optimization track |

To reach 30 tok/s would require either:
- Blackhole hardware (3x faster matmuls) + MTP
- Reducing op count from ~3155 to <1000 (aggressive kernel fusion in tt-metal runtime)
- Both of which are outside scope of this sprint
