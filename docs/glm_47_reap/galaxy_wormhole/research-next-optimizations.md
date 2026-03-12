# REAP-218B Next Optimization Opportunities (2026-03-11)

**Source**: Codex (gpt-5.4) research across 4 areas, verified against codebase.
**Baseline**: bs=1 ~39ms TPOT (vLLM metric), bs=32 ~167 tok/s decode-only.
**Current**: ~8,566 programs/decode step (89 MoE × 93 + 3 dense × 55 + misc).

## Priority Ranking

| Rank | Optimization | Expected bs=1 ITL savings | Effort | Risk |
|---|---|---|---|---|
| **1** | **DRAM Weight Prefetching** | 37-44ms (19-23%) | High | Med |
| **2** | **Fused partial QK RoPE kernel** | ~19ms (10%) | High | Med |
| **3** | **Q/K DRAM round-trip elimination + KV cache fusion** | ~7ms (3-4%) | Low | Low |
| **4** | **L1 interleaved MoE intermediates** | 2.8-3.7ms (1.5-2%) | Med | Med |
| **5** | **Fold QKV bias into linear** | ~1.4ms (0.7%) | Low | Low |

---

## 1. DRAM Weight Prefetching (HIGHEST IMPACT)

**What**: Port Llama3 Galaxy's `ttnn.dram_prefetcher()` infrastructure to REAP.
**Why**: bs=1 at ~15% DRAM utilization. Prefetching hides weight read latency behind compute.
**How**: Subdevice architecture (sender cores + worker cores), `ttnn.create_global_circular_buffer`,
  `set_sub_device_stall_group`, fixed tensor address registration.

**Reference implementation**:
- `models/demos/llama3_70b_galaxy/tt/prefetcher_common.py` — TtLlamaPrefetcherSetup
- `models/demos/llama3_70b_galaxy/tt/llama_model.py` — dram_prefetcher usage

**Phase 1 (easiest)**: Prefetch attention + shared expert weights (~21.2 MB/layer/device).
  These are unconditional (not routing-dependent).
**Phase 2 (harder)**: Routed expert weights via `sparse_matmul` `global_cb`/`sub_device_id` params.
  Challenge: next-layer top-8 routing not known in advance → must prefetch all 3 local experts.

**Expected**: 37-44ms savings at bs=1 if overlap is good. Less impactful at bs=32.
**Effort**: Multiple weeks — full trace architecture change.

---

## 2. Fused Partial QK RoPE Kernel

**What**: Replace 16 ops/layer (4 slice + 2 concat + 2×mul + 2×add for Q and K RoPE)
  with a single fused C++ kernel that does NeoX-style partial RoPE on both Q and K.
**Why**: Saves ~1,288 programs/decode (14 ops/layer × 92 layers).

**Blocker**: `ttnn.experimental.rotary_embedding_llama_fused_qk` assumes full-dim RoPE.
  GLM-4.7 uses partial RoPE (rotary_dim=64 of head_dim=128). Need a new
  `partial_rotary_embedding_fused_qk` kernel.

**Current ops per layer (2 calls × 8 ops each = 16)**:
  slice x_rot, slice x_pass, slice x1, slice x2, concat rearranged, mul×cos, mul×sin_neg, add, concat back

**Expected**: ~19ms savings (dispatch reduction + some bandwidth). Requires C++ kernel dev.

---

## 3. Q/K DRAM Round-Trip Elimination + KV Cache Fusion (LOW-HANGING FRUIT)

**What**: Keep Q/K in L1 through QK norm instead of DRAM→L1→DRAM→L1 round-trips.
  Plus use `paged_fused_update_cache(K,V)` instead of separate K/V cache updates.

**Where**: attention_tt.py lines 678-679 (`q = to_memory_config(q, DRAM)` before norm).
  tt_transformers already does L1 QK norm.

**Saves**: ~460 programs/decode (4 DRAM conversions/layer × 92 + 1 cache op/layer × 92).
**Expected**: ~7ms. Low risk, straightforward port from tt_transformers.

**Implementation details**:
- Q/K L1: Change lines 678-679 from `DRAM_MEMORY_CONFIG` to `L1_MEMORY_CONFIG`. Q is 24 KB, K is 2 KB — both trivially fit L1. RMSNorm forward() accepts interleaved input from any memory config.
- Fused KV: Replace two `paged_update_cache(keys, k)` + `paged_update_cache(values, v)` calls (lines 699-704) with single `paged_fused_update_cache(keys, k, values, v, ...)`. API exists in ttnn.experimental, used by tt_transformers and Llama3 Galaxy.

---

## 4. L1 Interleaved MoE Intermediates

**What**: Keep MoE intermediate tensors (w1_out, w3_out, x_ff) in L1 instead of DRAM.
**Constraint**: Must convert to DRAM before any CCL operation (all_reduce, reduce_scatter).
**EP_L1=1 is BROKEN** (global flag crosses CCL boundaries). Need per-op L1 placement.

**Safe targets**: w1_out (288 KB), w3_out (288 KB), x_ff (288 KB). All fit in 1.43 MB usable L1.
**Expected**: 2.8-3.7ms savings. Modest but safe if done per-op.

---

## 5. Fold QKV Bias into Linear

**What**: Replace separate `xqkv + bias` with `linear(..., bias=bias)`.
**Saves**: ~92 programs/decode (1 add/layer × 92).
**Expected**: ~1.4ms. Trivial change.

---

## Dead Ends (from this research)

- RMSNorm+Linear fusion: No existing kernel
- matmul_split for QKV: Already uses fused QKV weights
- addcmul for RoPE: Already implemented (saves 184 ops) but **suspected of causing crashes** — being reverted and tested
