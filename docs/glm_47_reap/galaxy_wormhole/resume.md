# GLM-4.7-REAP-218B Galaxy Wormhole — Resume State

**Date**: 2026-03-12 (Session 6: quick wins tested — Q/K L1 neutral, cos/sin pad impossible)
**Model**: cerebras/GLM-4.7-REAP-218B-A32B — 92 layers, 96 routed experts (top-8), GQA 96Q/8KV
**Hardware**: Galaxy Wormhole, 32 chips, MESH_DEVICE=TG, Mesh(8,4), TP=8, EP=32, DP=4
**REAP Current**: bs=1: **150-200ms TPOT** (~5-6.7 tok/s), bs=32: **348ms TPOT = 92 tok/s agg** — SEED-3 BF4
**GLM-4.7 Current**: bs=1: **4.1 tok/s**, bs=32: **104.0 tok/s agg** — all-BF4 experts, BF8 dense
**Container**: HEALTHY, single-bucket [32] trace, SEED-3 BF4, QK_L1=1

**MEASUREMENT NOTE**: All performance numbers from server-side TPOT histogram
(`vllm:time_per_output_token_seconds`), measured via delta method (before/after sum and count).
Previous "142.7 tok/s" was wall-time aggregate from bench_itl.py (inflated by measurement methodology).
Client-side wall-time includes prefill overhead and is NOT comparable across runs.

---

## Current Config

```
GLM4_MOE_REDUCE_IMPL=native                # axis-0 (TP) reduce
GLM4_MOE_REDUCE_IMPL_AXIS1=rs_ag_async     # axis-1 (DP) reduce — bypasses FABRIC_2D crash
GLM4_MOE_EP_REDUCE_DEVICE=1
GLM4_MOE_FUSE_SHARED_EP_REDUCE=0           # BROKEN: garbled output
GLM4_MOE_EP_L1=0                           # BROKEN: garbled
GLM4_MOE_DRAM_SHARD=0                      # TESTED: +2.1% (not significant)
GLM4_MOE_DENSE_TT_DTYPE=bf8                # +18.8% at bs=32
GLM4_MOE_EXPERTS_TT_DTYPE=bf8              # Base (overridden per-projection)
GLM4_MOE_EXPERTS_W1_DTYPE=bf4              # SEED-3: gate BF4
GLM4_MOE_EXPERTS_W2_DTYPE=bf4              # SEED-3: down BF4
GLM4_MOE_EXPERTS_W3_DTYPE=bf8              # SEED-3: up stays BF8
GLM4_MOE_ATTN_FIDELITY=hifi               # LoFi tested: no speedup
GLM4_MOE_MOE_SPARSE_FIDELITY=hifi         # LoFi tested: no speedup
MAX_NUM_SEQS=32
decode_trace_batch_buckets=[32]            # Single bucket: prevents trace recapture crash
trace_mode=decode_only
```

---

## Session 5: Trace Crash Fix + TPOT Verification (2026-03-12)

### Trace Recapture Crash — Root Cause + Fix
**Root cause**: DRAM fragmentation on trace recapture. Releasing a small (bs=1) trace
+ running prefill (allocates/frees buffers) fragments DRAM. The larger bs=32 trace
cannot find a contiguous region → crash.

**Crash pattern** (verified systematically):
| Direction | Result |
|-----------|--------|
| bs=1 → bs=32 | **CRASH** (2/2 reproducible) |
| bs=32 → bs=1 | OK |
| bs=32 first (no prior trace) | OK |
| bs=1 first (no prior trace) | OK |

**Fix**: `decode_trace_batch_buckets: [32]` — single bucket. bs=1 requests pad to 32
(replicate last entry, safe for KV writes). Same trace reused for all batch sizes.
No trace recapture ever occurs. **Verified**: bs=1→bs=32 passes cleanly.

**Trade-off**: bs=1 TPOT increases from 150-200ms to ~350ms (padded to 32-wide trace).
Acceptable for production where most time is spent at high batch utilization.

### addcmul Revert
Reverted `ttnn.addcmul` in `_apply_partial_rope_decode` back to `multiply+add`.
Confirmed addcmul was NOT causing crashes — the crash was trace recapture.
addcmul is performance-neutral; can be re-applied if desired.

### TPOT Verification (Server-Side Histogram)
Definitive steady-state measurement via vLLM metric delta (gen=100, before/after):
- **bs=32**: 348ms TPOT (99% of samples in 300-400ms bucket) → **92 tok/s aggregate**
- Consistent across 2 independent runs (346.9ms, 347.9ms)
- **bs=1** (no padding): 150-200ms TPOT (histogram bucket)

---

## Session 4: Trace Stability Fixes (2026-03-11)

Three trace stability fixes:
1. **trans_matrix_tt deallocation**: Removed shared RoPE tensor from cleanup tuple
2. **Batch bucket rounding**: Round batch UP to [1,4,8,16,32], pad inputs
3. **Release before capture**: Release existing traces before capturing new ones

Plus: embed_tt clone fix, prefill KV cache DP column fix.

---

## SEED-3: Selective BF4 — CONFIRMED OPTIMAL ✅

| Test | Config | bs=32 agg | PPL | vs BF8 |
|------|--------|-----------|-----|--------|
| 0 | All BF8 | 98.9 | 1.3659 | — |
| **1 (SEED-3)** | **w1/w2=BF4, w3=BF8** | **142.7*** | **1.2288** | **+44%, -10%** |
| 2 | All BF4 | 126.4 | 1.2884 | +28%, -5.7% |
| 3 | w1=BF4, w2=BF8, w3=BF4 | 128.3 | 1.4007 | +30%, +2.5% |

*142.7 is wall-time aggregate; server-side steady-state TPOT = 348ms = ~92 tok/s.
All tests measured consistently with same methodology, so relative comparisons are valid.

---

## SEED-4: GLM-4.7 Base (358B) — COMPLETED ✅

- 92 layers, 160 experts (top-8), EP=32 (5/device), all-BF4 experts
- **Key fix**: KV cache OOM — `get_num_available_blocks_tt()` defaulted to 131K tokens
- bs=1: 4.1 tok/s, bs=32: 104.0 tok/s agg (27% slower than REAP — 5 vs 3 experts/device)

---

## Session 6: Quick Win Testing (2026-03-12)

### P1: Q/K DRAM Round-Trip Elimination — NEUTRAL (+0.26%, noise)
A/B test with server-side TPOT histogram:
- **QK_L1=1**: 349.3ms TPOT (best of 3 runs)
- **QK_L1=0**: 348.4ms TPOT (best of 3 runs)
- **Delta: +0.9ms = +0.26%** — within measurement noise

Root cause of no improvement: In trace mode, program dispatches are baked in.
The actual DRAM traffic saved (Q=98KB + K=8KB per layer) is ~9μs — invisible at 348ms.
Shipped as default (harmless), but no performance impact.

### P3 Workaround: cos/sin Padding for Fused RoPE — IMPOSSIBLE
Architect proposed padding cos=1.0, sin=0.0 for non-rotary dims to use full-dim RoPE kernel.
**DISPROVEN**: NeoX-style `rotate_half` splits at `dim // 2`. Extending from rotary_dim=64 to
head_dim=128 shifts the split boundary from 32 to 64, mixing pass-through dims into rotary
computation. Only works with GPT-J interleaved rotation, NOT NeoX.

### P1b: Fused KV Cache — SKIPPED
Architect estimated ~0.4% (92 fewer dispatches). Given P1's result (368 fewer dispatches = 0%),
P1b would be even smaller. Not worth MEDIUM risk (core grid management).

### FP8 Dequant Bug Documented
Scale inversion bug in `_maybe_dequant_fp8()` at `layer_weights.py:73` — inverts `_scale`
but `_scale` is already the dequant multiplier. 1-line fix needed. Does NOT affect current
BF16 model deployment.

---

## Next Priorities (REVISED)

1. **DRAM weight prefetching** (HIGH effort, 19-23% — ONLY viable path for large gains)
   - Port Blackhole-only `ttnn.dram_prefetcher()` to Wormhole Galaxy
   - 3-5 days for Phase 1 (attention + shared expert weights)
   - sparse_matmul + global_cb for routed experts (untested, Phase 2)
2. **Native fused partial QK RoPE C++ kernel** (HIGH effort, ~3%)
   - cos/sin padding workaround IMPOSSIBLE (NeoX rotation)
   - Needs new C++ kernel with `rotary_dim` param + NeoX rotation support
   - 3-5 days
3. **FP8 dequant bug fix** (LOW effort, enables FP8 source model loading)
   - 1-line fix in `layer_weights.py:73`
   - No performance benefit for current deployment (BF16 source)

**Key lesson**: Software-only "quick wins" (dispatch elimination, small DRAM savings)
have effectively ZERO impact on DRAM-BW-bound trace mode execution. Only DRAM weight
prefetching can meaningfully improve throughput.

See `implementation-plan-session6.md` in plan/ for full architect analysis.

---

## Dead Ends (Do NOT Retry)

- EP_L1=1 → garbled output
- FUSE_SHARED_EP_REDUCE=1 → Phase 1: hang. Phase 2: garbled
- LoFi math fidelity → no speedup (DRAM-BW-bound)
- DRAM-sharded attn weights → +2.1% (not significant)
- Device-side sampling → +0% (hidden behind trace)
- Fused gate+up sparse matmul → -14.6% regression
- bs=64 → WORSE (MoE EP=32 doubles cost)
- Multi-bucket traces → bs=1→bs=32 crash (DRAM fragmentation). Use single bucket [32].
- Sparse matmul grid optimization → CRASH (non-full-width grid incompatible)
- SEED-5 (fused partial RoPE) → no API support
- SEED-6 (add-before-reduce) → garbled output
- **Q/K L1 interleaved** → +0.26% (noise). Dispatch/traffic elimination invisible in trace mode.
- **cos/sin padding for partial RoPE** → IMPOSSIBLE (NeoX rotation incompatible)
- **Fused KV cache** → skipped, expected <0.4% based on P1 result

## Key Files

| File | Purpose |
|------|---------|
| `tt-metal/models/demos/glm4_moe/tt/model_tt.py` | Model forward, trace capture/replay |
| `tt-metal/models/demos/glm4_moe/tt/attention_tt.py` | Attention + batch-bucketed init |
| `tt-metal/models/demos/glm4_moe/tt/decoder_layer_tt.py` | Decoder forward, EP reduce |
| `tt-metal/models/demos/glm4_moe/tt/moe_tt.py` | MoE routing, sparse matmul |
| `tt-metal/models/demos/glm4_moe/tt/layer_weights.py` | Weight loading, per-projection dtype |
| `docker_tt/dev/.env.glm47_reap` | All env flags |
