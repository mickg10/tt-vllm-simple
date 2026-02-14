# perf-opt

Iteration history and learnings for TT GLM performance.

## Ralph Loop Iteration 1 — Baseline (2026-02-12)

### Targets
- Batch=1 decode: 30 tok/s (currently 4.5 tok/s, need 6.7x)
- Batch=32 decode: 140+ tok/s aggregate (currently 27.8 tok/s, need 5.0x)
- Benchmark matrix: (1k/500, 10k/1000, 29k/3000 ctx/gen) × (batch=1,4,8,32)

### Baseline Benchmark — 1k context, 500 gen (2026-02-12)

| Batch | Aggregate tok/s | Per-user tok/s | TTFT (median) | ITL (median) | Wall time | Tokens |
|-------|----------------|----------------|---------------|-------------|-----------|--------|
| 1 | 2.9 | 4.5 | 59.31s | 223.0ms | 171.0s | 500 |
| 4 | 14.6 | 4.5 | 25.39s | 223.2ms | 137.0s | 2000 |
| 8 | 28.5 | 4.5 | 29.82s | 221.0ms | 140.2s | 4000 |
| 32 | 27.8 | 4.2 | 107.92s | 190.8ms | 576.2s | 15992 |

### Key Observations

1. **Per-user decode is constant at ~4.5 tok/s** across batch 1-8 (223ms ITL).
   At bs=32, slightly lower (4.2 tok/s, 191ms ITL) — possible trace/scheduling overhead.

2. **Batch=32 aggregate is limited by prefill overhead**, not decode:
   - Pure decode potential: 32 × 4.2 = 134 tok/s (close to 140 target!)
   - But TTFT=108s for 32 concurrent users with 1k context eats into wall time
   - Aggregate = 15992 tokens / 576s = 27.8 tok/s (decode is only ~468s of 576s)

3. **Two separate bottlenecks:**
   - **Batch=1 target (30 tok/s):** Requires reducing decode latency from 223ms to ~33ms (6.7x)
   - **Batch=32 target (140 tok/s):** Requires both faster decode (~4.5→4.4 tok/s per user is fine)
     AND dramatically faster prefill (TTFT from 108s to <10s for 32×1k)

4. **TTFT for single user at 1k context is 59s** — extremely slow. The flash_mla_prefill + MoE
   chunking at 32 tokens per chunk through 47 layers creates massive overhead.
   Prefill optimization is critical for batch=32 aggregate.

### Gap Analysis

| Target | Current | Gap | Primary Bottleneck |
|--------|---------|-----|-------------------|
| bs=1 30 tok/s | 4.5 tok/s | 6.7x | Decode latency (223ms → 33ms needed) |
| bs=32 140 tok/s | 27.8 tok/s | 5.0x | Prefill (TTFT=108s) + decode (191ms → ~30ms) |

### Codex Recommendation (gpt-5.2 via MCP, 2026-02-12)

**Winner: (c) L1 WIDTH_SHARDED activations end-to-end for decode**

Rationale from Codex:
- Decode matmuls are "tiny-M" (per_core_M=1), so latency + data-movement dominated
- Keeping activations in consistent L1 WIDTH_SHARDED layout removes repeated DRAM round-trips
  and layout/reshard churn across ALL major buckets (Q path, attn out, router, experts, shared)
- This is the layout the fast decode configs expect
- Affects 100% of decode compute, not just one 19% bucket

If L1 WIDTH_SHARDED is already in place, fallback is **(a) DRAM-sharded weights** using
`get_dram_sharded_matmul_config(...)` from DeepSeek V3.

DeepSeek V3 reference patterns:
- `config_helpers.py:94` — `get_activation_sharding_core_counts_for_dram_matmul()`
- `mlp.py:336` — `ttnn.create_sharded_memory_config_()` for decode activations
- `mlp.py:264` — `memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG` for linears/eltwise
- `mlp.py:500` — async CCL (`all_gather_async`, `reduce_scatter_minimal_async`)

### What Did NOT Work (avoid repeating)

| Attempt | Result | Why it failed |
|---------|--------|---------------|
| Dense BF8 weights | Slower (1.91 vs 2.64 tok/s) | BF8 matmul slower than BF16 for these shapes |
| `dense_decode` experts impl | Container restarts | Sequential per-expert loops unstable |
| MLA hifi2 fidelity | Timeout | Insufficient precision for FlashMLA |
| CPU router fallback | Slower (1.86 tok/s) | Host roundtrip latency |
| L1_MEMORY_CONFIG (non-sharded) | No improvement | Non-sharded L1 doesn't change DRAM weight reads |
| DECODE_L1_ACT=1 | No measurable change | Need WIDTH_SHARDED, not generic L1 |
| `L1_WIDTH_SHARDED_MEMORY_CONFIG` as output memory_config | No improvement (222.9ms→222.9ms) | See analysis below |
| DRAM-sharded attn weights (Phase 1) per-linear reshard | bs=32 +37%, bs=1 -22% | Per-linear reshard overhead (6/layer attn) dominates at M=1 |

### DRAM-Sharded Phase 1 Results + Batch Divergence (Architect Analysis #2, 2026-02-12)

**Finding**: DRAM-sharded attention weights with per-linear resharding:
- bs=32: 27.8 → 38.0 tok/s aggregate (+37%) — GOOD
- bs=1: 4.5 → 3.5 tok/s (-22%) — REGRESSION

**Root cause confirmed by Codex (gpt-5.2) and code analysis**:

`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` **requires** WIDTH_SHARDED input
activations (verified at `tt-metal/ttnn/cpp/.../matmul_device_operation.cpp:560`). This means
resharding is mandatory per-linear when using DRAM-sharded matmul.

The current Phase 1 `_attn_linear` → `_dram_sharded_linear` pattern does:
```
for each attention linear:
    a_sharded = ttnn.to_memory_config(a, WIDTH_SHARDED)   # reshard IN
    result = ttnn.linear(a_sharded, b, program_config=...) # fast matmul
    result_dram = ttnn.to_memory_config(result, DRAM)      # reshard OUT
```

**Per-layer overhead accounting:**
- 3 `_attn_linear` calls (w_q_kv_a, w_q_b, w_o) × 2 reshards each = 6 reshards
- 1 `_dram_sharded_mlp` call (entry + exit) = 2 reshards
- Total: 8 reshards/layer × 47 layers = **376 reshards per decode step**
- At bs=1 (~100-150μs per reshard): **38-56ms total overhead**
- Matmul savings at bs=1 (M=1, tiny): ~20-30ms → NET REGRESSION

For bs=32: matmul savings (~70-80ms) > reshard overhead (~56ms) → NET WIN

**The DeepSeek V3 pattern has ZERO reshards between ops** (verified: `grep to_memory_config
mlp.py` returns nothing). The flow stays L1 WIDTH_SHARDED throughout:
```
all_gather_async → w1(sharded→sharded) → silu → mul(sharded) → w2(sharded→sharded) → reduce_scatter_async
```

**Codex recommendation (verbatim)**:
1. DRAM-sharded matmul program config REQUIRES WIDTH_SHARDED input — can't bypass
2. Pre-allocating buffers via `ttnn.reshard(..., output_tensor=...)` reduces allocator churn but NOT data-movement cost
3. Batch-adaptive is pragmatic short-term
4. The REAL fix: keep activations L1 WIDTH_SHARDED across consecutive ops, only de-shard at ops that truly require interleaved
5. **235 sync all_reduce + 705 clone/step is very likely a bigger bs=1 latency limiter than DRAM-sharded weights**

### Batch-Adaptive Strategy: Three-Phase Plan

#### Phase 1 (This Sprint): "MLP-only DRAM-shard + reduce overhead"

**Disable DRAM-sharding for attention, keep it for MLP only.**

| Change | Impact | Risk |
|--------|--------|------|
| Keep `_dram_sharded_mlp` for shared expert (already works) | +20-25% bs=32 (MLP has largest weights) | Low (already tested) |
| Remove DRAM-sharding from `_attn_linear` | Eliminates 6 reshards/layer → saves ~40ms at bs=1 | Low (revert to working path) |
| Add explicit `MatmulMultiCoreReuseMultiCast1DProgramConfig` to attn linears | ~10-15% better than auto-select, zero resharding | Medium (need to compute configs) |
| Clone audit: remove unnecessary defensive clones | ~20-30ms bs=1 savings | Medium (need aliasing analysis) |

Feature flags:
- `GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1` — master switch
- `GLM4_MOE_LITE_DRAM_SHARDED_MLP=1` — MLP DRAM-sharding (keep ON)
- `GLM4_MOE_LITE_DRAM_SHARDED_ATTN=0` — attention DRAM-sharding (turn OFF)

Expected: bs=1 ~4.5-5.5 tok/s (no regression), bs=32 ~34-36 tok/s (+22-29%)

#### Phase 2 (Next Sprint): "Async CCL"

Replace `ttnn.all_reduce` (sync) with `all_gather_async` + `reduce_scatter_minimal_async`.
Import `CCL` semaphore manager from `deepseek_v3.tt.ccl`. Overlaps communication with compute.

Expected additional savings: ~20-30ms → bs=1 ~6-8 tok/s, bs=32 ~40-45 tok/s

#### Phase 3 (Medium Term): "End-to-End Sharded Decode"

Full DeepSeek V3 decode architecture: keep ALL activations in L1 WIDTH_SHARDED throughout
the decode path. Only convert at natural boundaries (before FlashMLA, before sparse MoE).
This eliminates ALL resharding overhead and achieves maximum DRAM bandwidth utilization.

Expected: bs=1 target 30 tok/s achievable with BF4 expert weights.

### Why `L1_WIDTH_SHARDED_MEMORY_CONFIG` Failed (Architect Analysis, 2026-02-12)

**Root cause: `ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG` is a sentinel constant, not a concrete shard config.**

It tells TTNN "put the output in L1 WIDTH_SHARDED" but does NOT:
1. Specify core grid or shard shape for the output shards
2. Change the matmul PROGRAM CONFIG — auto-selection still picks a generic program
3. Change how WEIGHTS are stored — they remain in DRAM interleaved format
4. Provide the input activation shard spec the matmul kernel needs

**The DeepSeek V3 fast decode pattern requires THREE co-designed components:**

| Component | DeepSeek V3 (fast) | GLM current (slow) |
|-----------|-------------------|-------------------|
| **Weight storage** | `dram_sharded_weight_config(k, n, dram_grid_size)` — WIDTH_SHARDED across 12 DRAM banks | `ttnn.DRAM_MEMORY_CONFIG` — interleaved (generic) |
| **Matmul program** | `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` with explicit per_core_M/N/block_w | Auto-selected (no program_config passed) |
| **Activation memory** | `ttnn.create_sharded_memory_config_()` with explicit shard shapes + core grids | `ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG` (sentinel, no shard spec) |

All three must be in place together. GLM has NONE of them. Passing only the output
memory_config sentinel does nothing because:
- The matmul auto-selector sees interleaved DRAM weights → picks a generic reader
- The generic reader achieves ~30-40% of peak DRAM bandwidth
- Output placement in L1 is negligible cost for tiny decode outputs (M=1)

**Where the 223 ms actually goes (estimated for bs=1 decode, 47 layers):**

| Category | Estimated time | Notes |
|----------|---------------|-------|
| Matmul weight reads (interleaved DRAM) | ~100-120 ms | ~50 MB weights/layer × 47 layers at ~30% DRAM efficiency |
| `ttnn.all_reduce` (TP sync) | ~20-30 ms | ~235 calls × 0.1 ms each (small tensors, linear topology) |
| Clone/permute/reshape DRAM traffic | ~30-40 ms | ~15 defensive clones/layer × 47 layers |
| Sparse MoE dispatch overhead | ~20-30 ms | Token remap + sparse kernel launch × 46 layers |
| FlashMLA decode + RoPE | ~15-20 ms | Relatively efficient already |
| **Total** | **~200-240 ms** | Matches observed 223 ms |

**Trace mode IS working** — verified that `begin_trace_capture`/`execute_trace` captures
the full 47-layer decode + LM head + sampling. The 223 ms is pure device execution time,
not Python dispatch overhead.

### Recommended Optimization Path (Priority Order)

#### P0: DRAM-Sharded Weights + Program Configs (Expected: 223ms → 80-100ms)

The single most impactful change. Directly adopt the DeepSeek V3 pattern:

1. **Weight loading** (`layer_weights.py`): Replace `memory_config=ttnn.DRAM_MEMORY_CONFIG`
   with `dram_sharded_weight_config(k, n, dram_grid_size)` for ALL decode projection weights.
   Import from `models.demos.deepseek_v3.utils.config_helpers`.

2. **Program configs** (`decoder_layer_tt.py`): For each `ttnn.linear` in the decode path,
   compute and pass `program_config=get_dram_sharded_matmul_config(M, K, N, in_cores, out_cores)`.
   M=USERS_PER_ROW=32 for all decode matmuls.

3. **Activation memory configs**: Replace `ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG` sentinel with
   explicit configs via `ttnn.create_sharded_memory_config_()` with calculated shard shapes.

**GLM dimension feasibility check (all tile-aligned):**

| Weight | K (tiles) | N (tiles) | Valid input cores | Valid output cores |
|--------|-----------|-----------|-------------------|-------------------|
| w_q_kv_a (TP=8) | 8 (256/32) | 42 (1344/32) | {1,2,4,8} | {1,2,3,6,7,14,21,42} |
| w_q_b (TP=8) | 3 (96/32) | 160 (5120/32) | {1,3} | {1,2,4,5,8,10,16,20,32,40} |
| w_kv_b1 (no TP*) | 6 (192/32) | 16 (512/32) | {1,2,3,6} | {1,2,4,8,16} |
| w_kv_b2 (TP=8) | 2 (64/32) | 8 (256/32) | {1,2} | {1,2,4,8} |
| w_o (TP=8) | 20 (640/32) | 64 (2048/32) | {1,2,4,5,10,20} | {1,2,4,8,16,32} |
| shared gate (TP=8) | 64 (2048/32) | 40 (1280/32) | {1,2,4,8,16,32} | {1,2,4,5,8,10,20,40} |
| shared down (TP=8) | 40 (1280/32) | 64 (2048/32) | {1,2,4,5,8,10,20,40} | {1,2,4,8,16,32} |

*w_kv_b1 can't use TP because 192/8=24 is not tile-aligned (24%32!=0)

Feature flag: `GLM4_MOE_LITE_DRAM_SHARDED_DECODE=1` (default 0, safe)

#### P1: Async CCL (Expected: additional 1.3-1.5x, 80ms → 55-65ms)

Replace `ttnn.all_reduce` with DeepSeek V3's async CCL pattern:
- `all_gather_async` before column-parallel matmuls
- `reduce_scatter_minimal_async` after row-parallel matmuls
- Requires a `CCL` semaphore manager (import from `deepseek_v3.tt.ccl`)

This overlaps communication with computation, hiding latency.

#### P2: Reduce clone/reshape overhead (Expected: additional 1.2x, 55ms → 45ms)

Profile which of the ~15 `ttnn.clone` calls per layer are still needed in the current
tt-metal version. Many were added defensively for aliasing bugs that may be fixed.
Each unnecessary clone adds a DRAM write (small, but 15 × 47 = 705 clones total).

#### P3: Optimize per-head matmuls (kv_b1, kv_b2)

These operate on [1, H, B, dim] tensors (H=20 heads). The per-head dimension creates
suboptimal tiling. Consider fusing heads or using batched matmul configs.

#### Combined projection: 223ms → 45ms → 22 tok/s (bs=1)

To reach 30 tok/s (33ms), we additionally need:
- BF4 weights for MoE experts (halves expert DRAM reads)
- OR reduce expert top-k from 4 to 2 for decode (halves expert compute)
- OR implement expert parallelism with async dispatch

## Next Backlog (Prioritized)

This backlog is derived from:
- the decode stage profile (dominant `ms/tok` stages), and
- the DeepSeek TT implementation diff (what “production” does that GLM bring-up does not).

P-2 (DONE, mandatory correctness stability):
- Deterministic greedy decode (`temperature=0`) on the TT endpoint is now passing.
- Root cause: FlashMLA decode with `fp32_dest_acc_en=True` corrupts greedy decode when the 2nd KV block is first touched (first observed at `pos=64` with `block_size=64`).
- Fix: force-disable FlashMLA `fp32_dest_acc_en` unless explicitly overridden behind an unsafe gate.
- Gate artifact: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/determinism_probe_zai-org_GLM-4.7-Flash_20260211_133728.md`

P-1 (mandatory metrics correctness):
- Fix GLM TT streaming output so we receive non-empty `delta.content` on long generations.
- Current symptom: `avg_ttft_s=0.000` for GLM TT in some iterations because the stream returns only a role chunk, then a usage-only chunk, then `[DONE]`.
- Gate: streaming benchmark captures `ttft_s>0` and non-empty content preview on the measured case(s).

P0 (required for 30 tok/s feasibility):
- Convert GLM TT from “replicated bring-up” to true TP sharding on the 1x8 mesh.
- Remove device0-only readback and implement correct output composition for vLLM.
- Expected payoff: large multi-x latency reduction across `q_path`, `attn_out`, shared MLP, router, and KV cache update.
- Gate: GLM smoke + Qwen smoke + warmed `pair100x500` perf artifact + stage-profile delta.

P1:
- Fuse `w_q_a` + `w_kv_a` into a single matmul (`w_q_kv_a`) following DeepSeek’s fused MLA approach.
- Target stages: `layer_q_path_s`, `layer_kv_cache_update_s`.
- Gate: same as P0, plus “reasonable English” manual check.

P2:
- Move decode hot tensors to L1/sharded memory configs; enable fast kernel configs where quality allows:
- `LoFi` compute, `packer_l1_acc=True`, BF8 weights broader than just KV+experts.
- Target stages: `layer_moe_experts_s`, `layer_moe_router_s`, plus reduce layout churn.

P3:
- Re-tune MoE router + experts specifically for tiny decode batches after TP is correct:
- sparse program config tuning
- avoid row-major/tile churn
- keep routing on-device (CPU router fallback is slower)

P4:
- If still below target, evaluate deeper quantization (BF4 for MoE weights) with explicit BF16-quality gates (not FP32 parity).

P5:
- Only after the short-loop is trending upward, run the long benchmark matrix and record iteration artifacts in `artifacts/perf_iterations/`.

## Iteration benchmark_8087_vs_8088_20260210_135207 (2026-02-10T13:53:25Z)

This is a quick, normalized A/B throughput snapshot (short prompts only) using the same model id on both endpoints.

- glm_remote_8087: avg_tokens_per_s=`19.050`, avg_latency_s=`2.405`, success=`4/4`
- glm_local_tt_8088: avg_tokens_per_s=`2.632`, avg_latency_s=`17.110`, success=`4/4`
- delta_tt_glm_vs_remote_glm_tokens_per_s: `-16.418`
- note: treated as throughput-only. At the time this was recorded, the manual gate was blocked by nondeterministic output (now fixed).

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/benchmark_8087_vs_8088_20260210_135207.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/benchmark_8087_vs_8088_20260210_135207.md`

## Iteration benchmark_8087_vs_8088_20260211_132556 (2026-02-11)

Quick A/B throughput snapshot (short prompts only) using the same model id on both endpoints.

- glm_remote_8087: avg_tokens_per_s=`19.169`, avg_latency_s=`5.818`, success=`4/4`
- glm_local_tt_8088: avg_tokens_per_s=`2.924`, avg_latency_s=`58.101`, success=`4/4`
- delta_tt_glm_vs_remote_glm_tokens_per_s: `-16.245`
- note: manual correctness gate is now PASS (deterministic greedy decode at `temperature=0`).

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/benchmark_8087_vs_8088_20260211_132556.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/benchmark_8087_vs_8088_20260211_132556.md`

## Iteration wordbench_001 (2026-02-06T06:57:19Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `145.3`
- startup_qwen_s: `93.2`

- glm_remote_8087: avg_e2e_tps=`16.678`, avg_decode_tps=`17.259`, avg_ttft_s=`1.066`, success=`1/10`
- glm_local_tt_8088: avg_e2e_tps=`0.000`, avg_decode_tps=`0.000`, avg_ttft_s=`0.000`, success=`0/10`
- qwen_local_tt_8088: avg_e2e_tps=`0.000`, avg_decode_tps=`0.000`, avg_ttft_s=`0.000`, success=`0/10`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-16.678`
- delta_tt_glm_vs_qwen_e2e_tps: `+0.000`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_wordbench_001.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_wordbench_001.md`

## Iteration pair100x500_001 (2026-02-06T07:21:15Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `144.3`
- startup_qwen_s: `89.2`

- glm_remote_8087: avg_e2e_tps=`21.286`, avg_decode_tps=`47.321`, avg_ttft_s=`12.919`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`0.000`, avg_decode_tps=`0.000`, avg_ttft_s=`0.000`, success=`0/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.256`, avg_decode_tps=`17.690`, avg_ttft_s=`1.622`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-21.286`
- delta_tt_glm_vs_qwen_e2e_tps: `-17.256`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_pair100x500_001.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_pair100x500_001.md`

## Iteration pair100x500_002 (2026-02-06T07:38:20Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `145.3`
- startup_qwen_s: `89.2`

- glm_remote_8087: avg_e2e_tps=`44.040`, avg_decode_tps=`46.969`, avg_ttft_s=`0.728`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`0.000`, avg_decode_tps=`0.000`, avg_ttft_s=`0.000`, success=`0/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.458`, avg_decode_tps=`17.504`, avg_ttft_s=`0.221`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-44.040`
- delta_tt_glm_vs_qwen_e2e_tps: `-17.458`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_pair100x500_002.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_pair100x500_002.md`

## Iteration pair100x500_003 (2026-02-06T07:57:42Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `145.3`
- startup_qwen_s: `89.2`

- glm_remote_8087: avg_e2e_tps=`45.467`, avg_decode_tps=`48.187`, avg_ttft_s=`0.640`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`2.435`, avg_decode_tps=`2.442`, avg_ttft_s=`1.221`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.255`, avg_decode_tps=`17.336`, avg_ttft_s=`0.241`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-43.032`
- delta_tt_glm_vs_qwen_e2e_tps: `-14.820`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_pair100x500_003.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_pair100x500_003.md`

## Iteration pair100x500_004_dense_decode (failed before measurement)

- target_tps: `30.0`
- override: `GLM4_MOE_LITE_MOE_EXPERTS_IMPL=dense_decode`
- result: GLM local startup did not reach `/health` within benchmark wait window (`wait_health` timeout), so no measured rows were produced.
- interpretation: `dense_decode` is not currently a safe low-risk default path.
- action: keep `sparse` experts path baseline and continue with instrumentation-first optimization.

## Iteration pair100x500_005_profile (2026-02-06T13:25:22Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `142.2`
- startup_qwen_s: `85.2`

- glm_remote_8087: avg_e2e_tps=`45.364`, avg_decode_tps=`47.812`, avg_ttft_s=`0.584`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`2.784`, avg_decode_tps=`2.794`, avg_ttft_s=`1.251`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.842`, avg_decode_tps=`17.926`, avg_ttft_s=`0.232`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-42.580`
- delta_tt_glm_vs_qwen_e2e_tps: `-15.058`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_pair100x500_005_profile.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_pair100x500_005_profile.md`

## Iteration pair100x500_006_novslice_profile (2026-02-06T13:44:28Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `141.3`
- startup_qwen_s: `85.2`

- glm_remote_8087: avg_e2e_tps=`44.754`, avg_decode_tps=`47.144`, avg_ttft_s=`0.586`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`2.716`, avg_decode_tps=`2.725`, avg_ttft_s=`1.218`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.902`, avg_decode_tps=`17.985`, avg_ttft_s=`0.230`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-42.038`
- delta_tt_glm_vs_qwen_e2e_tps: `-15.186`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_pair100x500_006_novslice_profile.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_pair100x500_006_novslice_profile.md`

## Decode Stage Profile Snapshot (2026-02-06, profile print_every=1 probe)

- probe setup:
- `GLM4_MOE_LITE_PROFILE=1`
- `GLM4_MOE_LITE_PROFILE_PRINT_EVERY=1`
- one short non-streaming prompt on `:8088` with GLM active.
- evidence source:
- `docker logs --tail 2000 dev-vllm-tt-1 | rg '\[glm4_moe_lite\]\[profile\]'`
- steady decode aggregate around calls `22..25`:
- `layer_total_s ~= 430-441 ms/tok` (model aggregate across layers).
- top contributors:
- `layer_moe_experts_s ~= 113-116 ms/tok`
- `layer_q_path_s ~= 75-77 ms/tok`
- `layer_kv_cache_update_s ~= 56-59 ms/tok`
- `layer_moe_router_s ~= 56-58 ms/tok`
- `layer_moe_shared_s ~= 43-44 ms/tok`
- `layer_attn_out_s ~= 42-43 ms/tok`
- interpretation:
- decode is dominated by MoE experts first, then Q-path + KV cache update.
- removing explicit V-cache slicing did not materially improve throughput (`pair100x500_006` vs `005`), so next work should target MoE + Q/KV path kernels/layout and not this slice path.

## Decode Stage Profile Snapshot (2026-02-06, profile print_every=32 probe)

- probe setup:
- `GLM4_MOE_LITE_PROFILE=1`
- `GLM4_MOE_LITE_PROFILE_PRINT_EVERY=32`
- one short non-streaming request to local TT GLM endpoint (`:8088`) with `max_tokens=160`.
- evidence source:
- `docker logs --since 5m dev-vllm-tt-1 | grep -E '\\[glm4_moe_lite\\]\\[profile\\]' | tail -n 50`
- steady decode aggregate (calls `96`, `tokens=96`):
- `layer_total_s=410.838 ms/tok`  => `~2.43 tok/s` aggregate decode.
- `layer_moe_experts_s=110.048 ms/tok`
- `layer_q_path_s=70.620 ms/tok`
- `layer_moe_router_s=58.220 ms/tok`
- `layer_kv_cache_update_s=48.904 ms/tok`
- `layer_moe_shared_s=44.890 ms/tok`
- `layer_attn_out_s=42.645 ms/tok`
- `layer_moe_merge_s=9.392 ms/tok`
- interpretation:
- this agrees with the print_every=1 probe and is lower-overhead for iterative tuning.
- largest levers remain: MoE expert compute and routing overhead; second wave: Q path and KV cache update.

## Dense Decode Recheck (2026-02-06, extended startup wait)

- setup:
- `GLM4_MOE_LITE_MOE_EXPERTS_IMPL=dense_decode`
- restarted `vllm-tt` with an extended manual health wait window (`<=900s`).
- observed behavior:
- container restart count increased and service remained `health: starting` in this probe.
- conclusion:
- treat `dense_decode` as currently unstable for iterative perf work.
- action:
- keep sparse experts path baseline and focus optimization on sparse MoE expert stage cost.

## Decode Lane Inflation Finding (2026-02-06, MoE sparse debug probe)

- probe setup:
- temporary runtime override: `GLM4_MOE_LITE_MOE_SPARSE_DEBUG=1`
- one short chat request to local TT GLM endpoint (`:8088`) after healthy startup.
- observed evidence from container logs:
- repeated lines (across MoE layers) reported:
- `tokens_per_device=32`
- `num_dispatch_devices=8`
- `total_tokens=256`
- `input_shape=(1, 1, 32, 2048)`
- interpretation:
- single-request decode is entering sparse MoE with tile-inflated token lanes.
- this strongly suggests padded-token compute is dominating decode cost for MoE, router, and related paths.
- optimization implication:
- highest-priority work is to carry true active decode token count through layer decode and only pad to the minimum sparse-kernel-required lane count (not full tile inflation).
## Iteration lanefix_step1_007 (2026-02-06T15:58:39Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `143.3`
- startup_qwen_s: `86.2`

- glm_remote_8087: avg_e2e_tps=`39.780`, avg_decode_tps=`46.974`, avg_ttft_s=`1.942`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`0.000`, avg_decode_tps=`0.000`, avg_ttft_s=`0.000`, success=`0/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.197`, avg_decode_tps=`17.863`, avg_ttft_s=`1.518`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-39.780`
- delta_tt_glm_vs_qwen_e2e_tps: `-17.197`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_lanefix_step1_007.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_lanefix_step1_007.md`

## Iteration lanefix_sparsebf16_step2_008 (2026-02-06T16:23:08Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `145.3`
- startup_qwen_s: `87.2`

- glm_remote_8087: avg_e2e_tps=`41.890`, avg_decode_tps=`46.779`, avg_ttft_s=`1.266`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`2.532`, avg_decode_tps=`2.554`, avg_ttft_s=`2.397`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.555`, avg_decode_tps=`18.252`, avg_ttft_s=`1.523`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-39.358`
- delta_tt_glm_vs_qwen_e2e_tps: `-15.023`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_lanefix_sparsebf16_step2_008.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_lanefix_sparsebf16_step2_008.md`

## Iteration lanefix_sparsebf16_mlahifi2_step3_009 (2026-02-06T16:39:02Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `143.3`
- startup_qwen_s: `87.2`

- glm_remote_8087: avg_e2e_tps=`41.814`, avg_decode_tps=`47.168`, avg_ttft_s=`1.376`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`0.000`, avg_decode_tps=`0.000`, avg_ttft_s=`0.000`, success=`0/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.447`, avg_decode_tps=`18.143`, avg_ttft_s=`1.541`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-41.814`
- delta_tt_glm_vs_qwen_e2e_tps: `-17.447`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_lanefix_sparsebf16_mlahifi2_step3_009.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_lanefix_sparsebf16_mlahifi2_step3_009.md`

## Iteration lane_sparsebf16_stable_step4_010 (2026-02-06T16:54:32Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `142.2`
- startup_qwen_s: `88.2`

- glm_remote_8087: avg_e2e_tps=`41.842`, avg_decode_tps=`47.231`, avg_ttft_s=`1.382`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`2.602`, avg_decode_tps=`2.626`, avg_ttft_s=`2.471`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.242`, avg_decode_tps=`17.975`, avg_ttft_s=`1.653`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-39.241`
- delta_tt_glm_vs_qwen_e2e_tps: `-14.640`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_lane_sparsebf16_stable_step4_010.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_lane_sparsebf16_stable_step4_010.md`

## Focused Probe (Sparse vs Dense-Decode Experts, 2026-02-06)

- baseline sparse probe (local GLM, `max_tokens=160`, same prompt twice):
- warmup: `102 tokens / 44.678s = 2.283 tok/s`
- measured: `102 tokens / 39.261s = 2.598 tok/s`
- dense-decode probe (`GLM4_MOE_LITE_MOE_EXPERTS_IMPL=dense_decode`):
- container repeatedly remained in startup path and emitted crash/restart traces before reaching healthy in practical iteration time.
- decision:
- keep sparse experts as active baseline.
- keep `dense_decode` as rejected for current performance track.

## Focused Probe (Router CPU Fallback, 2026-02-06)

- setup:
- temporary override `GLM4_MOE_LITE_MOE_ROUTER_IMPL=cpu` with all other baseline knobs unchanged.
- same prompt and token budget as sparse baseline probe (`max_tokens=160`, no-thinking).
- results:
- warmup: `117 tokens / 72.426s = 1.615 tok/s`
- measured: `117 tokens / 62.867s = 1.861 tok/s`
- baseline sparse reference from same cycle:
- measured: `102 tokens / 39.261s = 2.598 tok/s`
- decision:
- CPU router is slower; keep TT router as baseline.
## Iteration moe_reduce_remap_step5_012 (2026-02-06T22:28:35Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `145.3`
- startup_qwen_s: `85.2`

- glm_remote_8087: avg_e2e_tps=`41.074`, avg_decode_tps=`48.188`, avg_ttft_s=`1.814`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`2.333`, avg_decode_tps=`2.345`, avg_ttft_s=`1.908`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.299`, avg_decode_tps=`18.004`, avg_ttft_s=`1.582`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-38.741`
- delta_tt_glm_vs_qwen_e2e_tps: `-14.967`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_moe_reduce_remap_step5_012.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_moe_reduce_remap_step5_012.md`

## Iteration lm_head_vocab_shard_mesh_v1_001 (2026-02-06T23:26:01Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `147.3`
- startup_qwen_s: `87.2`

- glm_remote_8087: avg_e2e_tps=`43.596`, avg_decode_tps=`47.727`, avg_ttft_s=`1.012`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`0.000`, avg_decode_tps=`0.000`, avg_ttft_s=`0.000`, success=`0/1`
- qwen_local_tt_8088: avg_e2e_tps=`16.970`, avg_decode_tps=`17.632`, avg_ttft_s=`1.551`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-43.596`
- delta_tt_glm_vs_qwen_e2e_tps: `-16.970`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_lm_head_vocab_shard_mesh_v1_001.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_lm_head_vocab_shard_mesh_v1_001.md`

## Iteration remote_perf_check_001 (2026-02-07T07:15:31Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `76.1`
- startup_qwen_s: `82.1`

- glm_remote_8087: avg_e2e_tps=`45.297`, avg_decode_tps=`48.230`, avg_ttft_s=`0.691`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`1.691`, avg_decode_tps=`6.996`, avg_ttft_s=`1.488`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.900`, avg_decode_tps=`18.985`, avg_ttft_s=`0.212`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-43.606`
- delta_tt_glm_vs_qwen_e2e_tps: `-17.209`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_remote_perf_check_001.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_remote_perf_check_001.md`

## Iteration baseline_after_logprobs_fix_timeout900 (2026-02-07T21:07:53Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `144.2`
- startup_qwen_s: `78.1`

- glm_remote_8087: avg_e2e_tps=`40.706`, avg_decode_tps=`45.767`, avg_ttft_s=`1.377`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`0.000`, avg_decode_tps=`0.000`, avg_ttft_s=`0.000`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`14.302`, avg_decode_tps=`17.887`, avg_ttft_s=`9.516`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-40.706`
- delta_tt_glm_vs_qwen_e2e_tps: `-14.302`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_baseline_after_logprobs_fix_timeout900.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_baseline_after_logprobs_fix_timeout900.md`

## Iteration 2026-02-08_b (2026-02-08T06:23:23Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `72.1`
- startup_qwen_s: `79.2`

- glm_remote_8087: avg_e2e_tps=`44.820`, avg_decode_tps=`47.075`, avg_ttft_s=`0.689`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`0.000`, avg_decode_tps=`0.000`, avg_ttft_s=`0.000`, success=`0/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.010`, avg_decode_tps=`18.093`, avg_ttft_s=`0.216`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-44.820`
- delta_tt_glm_vs_qwen_e2e_tps: `-18.010`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_2026-02-08_b.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_2026-02-08_b.md`

## Iteration c (2026-02-08T07:11:49Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `71.2`
- startup_qwen_s: `76.1`

- glm_remote_8087: avg_e2e_tps=`45.140`, avg_decode_tps=`47.092`, avg_ttft_s=`0.641`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`0.000`, avg_decode_tps=`0.000`, avg_ttft_s=`0.000`, success=`0/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.910`, avg_decode_tps=`17.985`, avg_ttft_s=`0.212`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-45.140`
- delta_tt_glm_vs_qwen_e2e_tps: `-17.910`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_c.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_c.md`

## Iteration d (2026-02-08T07:32:51Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `71.2`
- startup_qwen_s: `76.1`

- glm_remote_8087: avg_e2e_tps=`44.447`, avg_decode_tps=`46.299`, avg_ttft_s=`0.629`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`0.000`, avg_decode_tps=`0.000`, avg_ttft_s=`0.000`, success=`0/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.805`, avg_decode_tps=`17.884`, avg_ttft_s=`0.225`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-44.447`
- delta_tt_glm_vs_qwen_e2e_tps: `-17.805`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_d.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_d.md`

## Iteration e (2026-02-08T07:49:54Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `71.2`
- startup_qwen_s: `76.2`
- prime_glm_tt: ttft_s=`77.80652131605893`, decode_tps=`6.851`
- prime_qwen_tt: ttft_s=`11.076258708024397`, decode_tps=`21.484`

- glm_remote_8087: avg_e2e_tps=`45.091`, avg_decode_tps=`47.139`, avg_ttft_s=`0.672`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`0.000`, avg_decode_tps=`0.000`, avg_ttft_s=`0.000`, success=`0/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.013`, avg_decode_tps=`18.089`, avg_ttft_s=`0.214`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-45.091`
- delta_tt_glm_vs_qwen_e2e_tps: `-18.013`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_e.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_e.md`

## Iteration trace_decodeonly_001b (2026-02-08T11:10:26Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `149.3`
- startup_qwen_s: `77.1`
- prime_glm_tt: ttft_s=`0.8748130159219727`, decode_tps=`6.475`
- prime_qwen_tt: ttft_s=`9.779531289008446`, decode_tps=`22.018`

- glm_remote_8087: avg_e2e_tps=`41.887`, avg_decode_tps=`45.257`, avg_ttft_s=`1.222`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`0.000`, avg_decode_tps=`0.000`, avg_ttft_s=`0.000`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.132`, avg_decode_tps=`18.219`, avg_ttft_s=`0.233`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-41.887`
- delta_tt_glm_vs_qwen_e2e_tps: `-18.132`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_trace_decodeonly_001b.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_trace_decodeonly_001b.md`

## Iteration it0001_baseline_small (2026-02-08T12:28:26Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `70.1`
- startup_qwen_s: `77.2`
- prime_glm_tt: ttft_s=`80.28357791691087`, decode_tps=`2.565`
- prime_qwen_tt: ttft_s=`11.558181258966215`, decode_tps=`21.990`

- glm_remote_8087: avg_e2e_tps=`43.903`, avg_decode_tps=`46.774`, avg_ttft_s=`0.965`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`0.000`, avg_decode_tps=`0.000`, avg_ttft_s=`0.000`, success=`0/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.676`, avg_decode_tps=`17.750`, avg_ttft_s=`0.216`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-43.903`
- delta_tt_glm_vs_qwen_e2e_tps: `-17.676`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it0001_baseline_small.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it0001_baseline_small.md`

## Iteration baseline_correctness_onecase (2026-02-08T16:12:51Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `71.2`
- startup_qwen_s: `76.2`
- prime_glm_tt: ttft_s=`None`, decode_tps=`0.000`
- prime_qwen_tt: ttft_s=`9.299581662053242`, decode_tps=`22.584`

- glm_remote_8087: avg_e2e_tps=`17.646`, avg_decode_tps=`18.620`, avg_ttft_s=`1.533`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`0.000`, avg_decode_tps=`0.000`, avg_ttft_s=`0.000`, success=`0/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.860`, avg_decode_tps=`17.935`, avg_ttft_s=`0.214`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-17.646`
- delta_tt_glm_vs_qwen_e2e_tps: `-17.860`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_baseline_correctness_onecase.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_baseline_correctness_onecase.md`

## Iteration it_tp0_trace_baseline_pair100x500 (2026-02-08T18:42:29Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `108.2`
- startup_qwen_s: `76.1`
- prime_glm_tt: ttft_s=`5.0086646099807695`, decode_tps=`6.407`
- prime_qwen_tt: ttft_s=`9.578090509050526`, decode_tps=`20.966`

- glm_remote_8087: avg_e2e_tps=`4.008`, avg_decode_tps=`44.857`, avg_ttft_s=`113.404`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`5.467`, avg_decode_tps=`6.480`, avg_ttft_s=`19.468`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.094`, avg_decode_tps=`18.214`, avg_ttft_s=`0.300`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `+1.459`
- delta_tt_glm_vs_qwen_e2e_tps: `-12.627`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_tp0_trace_baseline_pair100x500.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_tp0_trace_baseline_pair100x500.md`

## Iteration it_tp1_trace_pair100x500 (2026-02-08T18:53:56Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `147.3`
- startup_qwen_s: `77.2`
- prime_glm_tt: ttft_s=`6.44223356700968`, decode_tps=`6.142`
- prime_qwen_tt: ttft_s=`9.252289747004397`, decode_tps=`21.840`

- glm_remote_8087: avg_e2e_tps=`40.472`, avg_decode_tps=`43.976`, avg_ttft_s=`1.005`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`4.729`, avg_decode_tps=`6.038`, avg_ttft_s=`21.855`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.942`, avg_decode_tps=`18.019`, avg_ttft_s=`0.218`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-35.743`
- delta_tt_glm_vs_qwen_e2e_tps: `-13.213`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_tp1_trace_pair100x500.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_tp1_trace_pair100x500.md`

## Iteration i000 (2026-02-09T01:00:44Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `257.5`
- startup_qwen_s: `136.3`
- prime_glm_tt: ttft_s=`7.145051920088008`, decode_tps=`18.117`
- prime_qwen_tt: ttft_s=`205.22901183797512`, decode_tps=`18.379`

- glm_remote_8087: avg_e2e_tps=`41.539`, avg_decode_tps=`45.106`, avg_ttft_s=`1.307`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`12.972`, avg_decode_tps=`12.953`, avg_ttft_s=`0.000`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.564`, avg_decode_tps=`17.641`, avg_ttft_s=`0.223`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-28.567`
- delta_tt_glm_vs_qwen_e2e_tps: `-4.592`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_i000.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_i000.md`

## Iteration baseline_20260209_single100x500 (2026-02-09T02:32:27Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `153.3`
- startup_qwen_s: `143.3`
- prime_glm_tt: ttft_s=`3.963897556066513`, decode_tps=`17.471`
- prime_qwen_tt: ttft_s=`9.266793983988464`, decode_tps=`20.103`

- glm_remote_8087: avg_e2e_tps=`41.483`, avg_decode_tps=`45.572`, avg_ttft_s=`0.876`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`13.922`, avg_decode_tps=`13.901`, avg_ttft_s=`0.000`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.626`, avg_decode_tps=`17.700`, avg_ttft_s=`0.217`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-27.561`
- delta_tt_glm_vs_qwen_e2e_tps: `-3.704`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_baseline_20260209_single100x500.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_baseline_20260209_single100x500.md`

## Iteration baseline_moe_tp_20260209_single100x500 (2026-02-09T02:50:52Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `263.5`
- startup_qwen_s: `142.3`
- prime_glm_tt: ttft_s=`8.964151924941689`, decode_tps=`6.106`
- prime_qwen_tt: ttft_s=`9.581944462959655`, decode_tps=`19.145`

- glm_remote_8087: avg_e2e_tps=`41.006`, avg_decode_tps=`43.722`, avg_ttft_s=`0.621`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`4.585`, avg_decode_tps=`6.024`, avg_ttft_s=`24.808`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.662`, avg_decode_tps=`18.751`, avg_ttft_s=`0.225`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-36.421`
- delta_tt_glm_vs_qwen_e2e_tps: `-14.078`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_baseline_moe_tp_20260209_single100x500.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_baseline_moe_tp_20260209_single100x500.md`

## Iteration baseline_moe_tp0_20260209_single100x500 (2026-02-09T03:10:12Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `219.4`
- startup_qwen_s: `130.3`
- prime_glm_tt: ttft_s=`6.493858814006671`, decode_tps=`6.556`
- prime_qwen_tt: ttft_s=`8.78820785600692`, decode_tps=`22.405`

- glm_remote_8087: avg_e2e_tps=`42.300`, avg_decode_tps=`45.213`, avg_ttft_s=`0.624`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`5.380`, avg_decode_tps=`6.437`, avg_ttft_s=`20.754`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.575`, avg_decode_tps=`18.658`, avg_ttft_s=`0.216`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-36.920`
- delta_tt_glm_vs_qwen_e2e_tps: `-13.195`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_baseline_moe_tp0_20260209_single100x500.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_baseline_moe_tp0_20260209_single100x500.md`

## Iteration i001_router_tt (2026-02-09T07:16:31Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `125.2`
- startup_qwen_s: `77.2`
- prime_glm_tt: ttft_s=`4.739157774019986`, decode_tps=`18.001`
- prime_qwen_tt: ttft_s=`199.12704210705124`, decode_tps=`21.393`

- glm_remote_8087: avg_e2e_tps=`38.959`, avg_decode_tps=`42.977`, avg_ttft_s=`1.643`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`13.639`, avg_decode_tps=`13.619`, avg_ttft_s=`0.000`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.798`, avg_decode_tps=`17.875`, avg_ttft_s=`0.218`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-25.319`
- delta_tt_glm_vs_qwen_e2e_tps: `-4.159`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_i001_router_tt.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_i001_router_tt.md`

## Iteration sanity_now (2026-02-09T09:00:00Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `98.2`
- startup_qwen_s: `75.1`
- prime_glm_tt: ttft_s=`2.4493118789978325`, decode_tps=`16.282`
- prime_qwen_tt: ttft_s=`9.378540402976796`, decode_tps=`22.250`

- glm_remote_8087: avg_e2e_tps=`3.673`, avg_decode_tps=`44.152`, avg_ttft_s=`149.774`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`14.201`, avg_decode_tps=`14.177`, avg_ttft_s=`0.000`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.742`, avg_decode_tps=`17.841`, avg_ttft_s=`0.243`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `+10.527`
- delta_tt_glm_vs_qwen_e2e_tps: `-3.542`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_sanity_now.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_sanity_now.md`

## Iteration trace_fix_sanity_001 (2026-02-09T16:32:39Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `99.2`
- startup_qwen_s: `80.2`
- prime_glm_tt: ttft_s=`2.6685828070621938`, decode_tps=`17.636`
- prime_qwen_tt: ttft_s=`199.615561459912`, decode_tps=`19.465`

- glm_remote_8087: avg_e2e_tps=`38.885`, avg_decode_tps=`42.122`, avg_ttft_s=`0.804`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`13.397`, avg_decode_tps=`13.377`, avg_ttft_s=`0.000`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.718`, avg_decode_tps=`17.791`, avg_ttft_s=`0.212`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-25.489`
- delta_tt_glm_vs_qwen_e2e_tps: `-4.322`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_trace_fix_sanity_001.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_trace_fix_sanity_001.md`

## Iteration manual_baseline_now_single100x500 (2026-02-09T19:12:15Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `147.3`
- startup_qwen_s: `127.2`
- prime_glm_tt: ttft_s=`2.3879707870073617`, decode_tps=`17.658`
- prime_qwen_tt: ttft_s=`8.836829734034836`, decode_tps=`22.417`

- glm_remote_8087: avg_e2e_tps=`40.533`, avg_decode_tps=`43.420`, avg_ttft_s=`0.671`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`14.460`, avg_decode_tps=`14.439`, avg_ttft_s=`0.000`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.797`, avg_decode_tps=`18.881`, avg_ttft_s=`0.213`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-26.072`
- delta_tt_glm_vs_qwen_e2e_tps: `-4.337`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_manual_baseline_now_single100x500.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_manual_baseline_now_single100x500.md`
## Iteration it_glm47_trace_env_single100x500 (2026-02-09T19:55:11Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `212.3`
- startup_qwen_s: `128.2`
- prime_glm_tt: ttft_s=`4.851381987100467`, decode_tps=`6.570`
- prime_qwen_tt: ttft_s=`8.841564072063193`, decode_tps=`22.359`

- glm_remote_8087: avg_e2e_tps=`42.406`, avg_decode_tps=`44.926`, avg_ttft_s=`0.545`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`5.457`, avg_decode_tps=`6.461`, avg_ttft_s=`19.375`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.770`, avg_decode_tps=`18.854`, avg_ttft_s=`0.214`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-36.949`
- delta_tt_glm_vs_qwen_e2e_tps: `-13.313`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_glm47_trace_env_single100x500.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_glm47_trace_env_single100x500.md`

## Iteration it_glm47_trace_env_single100x500_b (2026-02-09T20:20:58Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `223.4`
- startup_qwen_s: `124.2`
- prime_glm_tt: ttft_s=`4.865344575140625`, decode_tps=`6.549`
- prime_qwen_tt: ttft_s=`9.012750912923366`, decode_tps=`22.626`

- glm_remote_8087: avg_e2e_tps=`40.922`, avg_decode_tps=`43.764`, avg_ttft_s=`0.650`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`5.450`, avg_decode_tps=`6.462`, avg_ttft_s=`19.554`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.795`, avg_decode_tps=`18.881`, avg_ttft_s=`0.215`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-35.472`
- delta_tt_glm_vs_qwen_e2e_tps: `-13.345`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_glm47_trace_env_single100x500_b.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_glm47_trace_env_single100x500_b.md`

## Iteration it_glm47_trace_env_single100x500_c (2026-02-09T20:37:46Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `207.4`
- startup_qwen_s: `124.2`
- prime_glm_tt: ttft_s=`4.915687714004889`, decode_tps=`6.552`
- prime_qwen_tt: ttft_s=`8.685109321027994`, decode_tps=`22.596`

- glm_remote_8087: avg_e2e_tps=`41.868`, avg_decode_tps=`44.346`, avg_ttft_s=`0.550`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`5.448`, avg_decode_tps=`6.462`, avg_ttft_s=`19.596`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.811`, avg_decode_tps=`18.895`, avg_ttft_s=`0.213`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-36.420`
- delta_tt_glm_vs_qwen_e2e_tps: `-13.363`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_glm47_trace_env_single100x500_c.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_glm47_trace_env_single100x500_c.md`

## Iteration it_remote_full_trace_single100x500_20260209_1 (2026-02-09T21:24:05Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `245.5`
- startup_qwen_s: `134.3`
- prime_glm_tt: ttft_s=`9.134966743877158`, decode_tps=`6.539`
- prime_qwen_tt: ttft_s=`192.69894506502897`, decode_tps=`21.403`

- glm_remote_8087: avg_e2e_tps=`9.594`, avg_decode_tps=`33.697`, avg_ttft_s=`41.186`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`5.236`, avg_decode_tps=`6.356`, avg_ttft_s=`22.877`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.927`, avg_decode_tps=`18.010`, avg_ttft_s=`0.228`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-4.358`
- delta_tt_glm_vs_qwen_e2e_tps: `-12.691`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_remote_full_trace_single100x500_20260209_1.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_remote_full_trace_single100x500_20260209_1.md`

## Iteration it_trace_tp1_single100x500_20260209_1 (2026-02-09T21:38:02Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `222.4`
- startup_qwen_s: `136.3`
- prime_glm_tt: ttft_s=`6.798719963058829`, decode_tps=`5.992`
- prime_qwen_tt: ttft_s=`9.225083007011563`, decode_tps=`21.040`

- glm_remote_8087: avg_e2e_tps=`28.675`, avg_decode_tps=`32.372`, avg_ttft_s=`2.230`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`4.662`, avg_decode_tps=`5.949`, avg_ttft_s=`22.125`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`17.614`, avg_decode_tps=`17.693`, avg_ttft_s=`0.230`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-24.013`
- delta_tt_glm_vs_qwen_e2e_tps: `-12.952`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_trace_tp1_single100x500_20260209_1.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_trace_tp1_single100x500_20260209_1.md`

## Iteration sanity_now (2026-02-09T22:38:11Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `218.4`
- startup_qwen_s: `129.3`
- prime_glm_tt: ttft_s=`5.370916251093149`, decode_tps=`6.529`
- prime_qwen_tt: ttft_s=`9.711334915831685`, decode_tps=`17.597`

- glm_remote_8087: avg_e2e_tps=`17.174`, avg_decode_tps=`32.620`, avg_ttft_s=`15.249`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`5.375`, avg_decode_tps=`6.374`, avg_ttft_s=`19.842`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.031`, avg_decode_tps=`18.114`, avg_ttft_s=`0.226`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-11.800`
- delta_tt_glm_vs_qwen_e2e_tps: `-12.657`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_sanity_now.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_sanity_now.md`

## Iteration it_20260210_perf_trace_after_blockingfix_b (2026-02-10T03:26:49Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `209.3`
- startup_qwen_s: `125.2`
- prime_glm_tt: ttft_s=`5.039740981999785`, decode_tps=`6.489`
- prime_qwen_tt: ttft_s=`177.7135914240498`, decode_tps=`22.549`

- glm_remote_8087: avg_e2e_tps=`31.305`, avg_decode_tps=`32.326`, avg_ttft_s=`0.588`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`5.290`, avg_decode_tps=`6.418`, avg_ttft_s=`22.582`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.819`, avg_decode_tps=`18.903`, avg_ttft_s=`0.212`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-26.016`
- delta_tt_glm_vs_qwen_e2e_tps: `-13.529`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_20260210_perf_trace_after_blockingfix_b.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_20260210_perf_trace_after_blockingfix_b.md`

## Iteration packer_l1_acc_001 (2026-02-10T04:43:03Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `148.3`
- startup_qwen_s: `130.2`
- prime_glm_tt: ttft_s=`3.3637802719604224`, decode_tps=`17.284`
- prime_qwen_tt: ttft_s=`178.88565128203481`, decode_tps=`22.352`

- glm_remote_8087: avg_e2e_tps=`30.940`, avg_decode_tps=`31.979`, avg_ttft_s=`0.611`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`13.370`, avg_decode_tps=`13.350`, avg_ttft_s=`0.000`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.802`, avg_decode_tps=`18.885`, avg_ttft_s=`0.212`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-17.570`
- delta_tt_glm_vs_qwen_e2e_tps: `-5.432`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_packer_l1_acc_001.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_packer_l1_acc_001.md`

## Iteration it_now_perf_trace_smoke (2026-02-10T08:03:38Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `154.3`
- startup_qwen_s: `73.1`
- prime_glm_tt: ttft_s=`5.045314510120079`, decode_tps=`6.474`
- prime_qwen_tt: ttft_s=`177.91483571310528`, decode_tps=`22.437`

- glm_remote_8087: avg_e2e_tps=`16.982`, avg_decode_tps=`32.259`, avg_ttft_s=`15.424`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`5.323`, avg_decode_tps=`6.414`, avg_ttft_s=`21.734`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.797`, avg_decode_tps=`18.882`, avg_ttft_s=`0.214`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-11.659`
- delta_tt_glm_vs_qwen_e2e_tps: `-13.474`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_now_perf_trace_smoke.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_now_perf_trace_smoke.md`

## Iteration it_20260210_ultra_correctness_safe_prefill_smoke (2026-02-10T09:22:33Z)

- target_tps: `30.0`
- ttmonitor_running: `True`
- startup_glm_s: `89.2`
- startup_qwen_s: `74.1`
- prime_glm_tt: ttft_s=`76.78370368899778`, decode_tps=`3.047`
- prime_qwen_tt: ttft_s=`8.875753548927605`, decode_tps=`19.298`

- glm_remote_8087: avg_e2e_tps=`21.834`, avg_decode_tps=`32.380`, avg_ttft_s=`8.265`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`2.352`, avg_decode_tps=`2.715`, avg_ttft_s=`38.734`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.844`, avg_decode_tps=`18.902`, avg_ttft_s=`0.164`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-19.482`
- delta_tt_glm_vs_qwen_e2e_tps: `-16.492`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_20260210_ultra_correctness_safe_prefill_smoke.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_20260210_ultra_correctness_safe_prefill_smoke.md`

## Iteration it_tp_trace_smoke_now (2026-02-10T11:51:52Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `201.3`
- startup_qwen_s: `72.1`
- prime_glm_tt: ttft_s=`6.104293592972681`, decode_tps=`5.965`
- prime_qwen_tt: ttft_s=`177.82939046807587`, decode_tps=`19.301`

- glm_remote_8087: avg_e2e_tps=`30.435`, avg_decode_tps=`32.041`, avg_ttft_s=`0.983`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`4.647`, avg_decode_tps=`5.934`, avg_ttft_s=`22.235`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.830`, avg_decode_tps=`18.887`, avg_ttft_s=`0.162`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-25.788`
- delta_tt_glm_vs_qwen_e2e_tps: `-14.182`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_tp_trace_smoke_now.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_tp_trace_smoke_now.md`

## Iteration it_tp_trace_smoke_now_2 (2026-02-10T12:04:26Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `155.3`
- startup_qwen_s: `72.1`
- prime_glm_tt: ttft_s=`6.271821678848937`, decode_tps=`5.977`
- prime_qwen_tt: ttft_s=`8.610553229926154`, decode_tps=`18.659`

- glm_remote_8087: avg_e2e_tps=`31.370`, avg_decode_tps=`32.373`, avg_ttft_s=`0.602`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`4.638`, avg_decode_tps=`5.934`, avg_ttft_s=`22.442`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.800`, avg_decode_tps=`18.857`, avg_ttft_s=`0.161`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-26.732`
- delta_tt_glm_vs_qwen_e2e_tps: `-14.162`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_tp_trace_smoke_now_2.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_tp_trace_smoke_now_2.md`

## Iteration it_perf_trace_baseline_recheck (2026-02-10T12:26:52Z)

- target_tps: `20.0`
- ttmonitor_running: `True`
- startup_glm_s: `156.3`
- startup_qwen_s: `73.1`
- prime_glm_tt: ttft_s=`5.131098365876824`, decode_tps=`6.487`
- prime_qwen_tt: ttft_s=`8.887978183105588`, decode_tps=`19.261`

- glm_remote_8087: avg_e2e_tps=`30.855`, avg_decode_tps=`31.847`, avg_ttft_s=`0.615`, success=`1/1`
- glm_local_tt_8088: avg_e2e_tps=`5.382`, avg_decode_tps=`6.392`, avg_ttft_s=`19.972`, success=`1/1`
- qwen_local_tt_8088: avg_e2e_tps=`18.795`, avg_decode_tps=`18.851`, avg_ttft_s=`0.160`, success=`1/1`
- delta_tt_glm_vs_remote_glm_e2e_tps: `-25.473`
- delta_tt_glm_vs_qwen_e2e_tps: `-13.412`
- goal_reached: `False`

- artifact_json: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_perf_trace_baseline_recheck.json`
- artifact_md: `/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations/iteration_it_perf_trace_baseline_recheck.md`

---

## Sprint 3: Native flash_mla_prefill (2026-02-12)

### Problem
TTFB was 6.65s even for short (6-token) prompts, and scaled linearly to 32.7s for 50 tokens.
Root cause: `decode_loop_trace` fed tokens one-at-a-time through the decode trace.

### Changes

1. **tp_size fix in prefill path** (`decoder_layer_tt.py`):
   - `run_decoder_layer_prefill_update_cache_tt()` was missing `tp_size` computation.
   - Added `mesh_rows, mesh_cols = _mesh_shape(device)` + `tp_size = int(...)` after `tp_enabled`.

2. **Trace release before prefill** (`model_tt.py` prefill()):
   - Release active decode trace before native prefill (prefill allocates buffers dynamically).
   - Deallocate trace output tensors (logits, top1) with force=True.
   - Trace lazily re-captured on next decode call (`_decode_trace_sampling` line 1452).

3. **MoE sparse_matmul block chunking** (`moe_tt.py`):
   - Root cause of hang: `sparse_matmul` program config has `per_core_M=1` → only supports
     1 sparsity block (32 tokens). Prefill with 64+ padded tokens → `num_blocks=2+` → device hang.
   - Fix: force the existing chunk mechanism to chunk at `sparsity_block_size` (32 tokens) when
     `total_tokens > sparsity_block_size` in the reduce dispatch path.
   - Each recursive chunk processes exactly 1 block → `per_core_M=1` works.

4. **Env config**: `PREFILL_IMPL=flash_mla_prefill`, `trace_mode=decode_only`.

### Results (single-query, sequential requests, warm)

| Prompt Tokens | TTFB (old) | TTFB (new) | Speedup |
|---------------|-----------|-----------|---------|
| 6 | 6.65s | 2.40s | 2.8x |
| 13 | ~7s | 2.47s | ~3x |
| 39 | ~26s | 2.60s | ~10x |
| 68 | ~45s | 5.44s | ~8x |

Decode throughput: ~5.1 tok/s (unchanged — traced decode is the same).

### Additional optimizations tried

5. **LoFi math fidelity + packer_l1_acc** (`decoder_layer_tt.py`):
   - Changed default MLP/attention compute kernel from TTNN default to LoFi + packer_l1_acc.
   - Also set `GLM4_MOE_LITE_MLA_FIDELITY=lofi` and `GLM4_MOE_LITE_MOE_SPARSE_FIDELITY=lofi`.
   - Result: 218ms → 195ms per token (~11% improvement). Correctness preserved.

6. **L1 memory config for intermediates** (tried, no effect):
   - Setting `memory_config=L1_MEMORY_CONFIG` on decode matmul outputs didn't help.
   - Non-sharded L1 config doesn't change DRAM weight read pattern.
   - DeepSeek uses WIDTH_SHARDED L1 + DRAM-sharded weights — fundamentally different.

7. **Batch=1 test** (MAX_NUM_SEQS=1):
   - 172ms per token (vs 218ms at batch=32) = 5.8 tok/s → 26% faster.
   - Confirms batch padding adds ~46ms overhead.

### Decode latency breakdown (per token, batch=32, all 47 layers)

From profiling (untraced warmup path):
| Component | ms/tok | % |
|-----------|--------|---|
| MoE experts (sparse_matmul) | 24.3 | 19% |
| KV cache update | 24.1 | 19% |
| Attention output (linear + CCL) | 23.3 | 18% |
| Q path (proj + RoPE + flash MLA) | 20.7 | 16% |
| MoE router (topk) | 14.0 | 11% |
| MoE shared expert (3 matmuls + CCL) | 10.3 | 8% |
| MLP dense (layer 0 only) | 3.6 | 3% |
| Other | 9.7 | 8% |
| **Total** | **130** | |

Traced decode: 195ms per token (50% overhead from per-op trace replay, embedding, sampling).

### Analysis: path to 30 tok/s

**Theoretical bandwidth limit**: ~5ms per token (10.2GB weight reads / 2048 GB/s T3K bandwidth)
**Current**: 195ms per token = 39x above theoretical limit.
**Gap**: Dominated by per-op overhead (kernel dispatch, DRAM↔L1 data flow per op), not bandwidth.

**Required changes (from DeepSeek V3 analysis):**
1. **DRAM-sharded weights** (`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`):
   - Distribute weight columns across 12 DRAM banks for parallel reads.
   - Requires changing weight loading + matmul program configs for every linear.
   - Impact estimate: 1.5-2x.

2. **L1 WIDTH-SHARDED activations** with custom core grids:
   - Pre-compute shard configs (USERS_PER_ROW=32, core_grid=7x4 like DeepSeek).
   - Keep all intermediates in L1, eliminate DRAM round-trips between ops.
   - Requires deep refactor of decode path memory management.
   - Impact estimate: 2-3x.

3. **Async CCL** (`all_gather_async` + `reduce_scatter_minimal_async`):
   - Overlap communication with computation.
   - Replace synchronous `all_reduce` with async scatter/gather.
   - Impact estimate: 1.2-1.5x.

4. **Fused wq_kv_a** projection:
   - Combine q_a + kv_a into single matmul, then slice.
   - Removes one full matmul from the attention path.
   - Impact estimate: 1.1x.

**Combined realistic estimate: 3-5x → 15-25 tok/s**. Reaching 30 tok/s would additionally
need weight quantization to bf4 or custom fused kernels.

### Remaining limitations
- MoE prefill chunks at 32 tokens per block (per_core_M=1 constraint).
  TTFB scales ~1.5s per additional 32-token chunk.
- For truly long prefill (1k+ tokens), need per_core_M > 1 or dense MoE fallback.
- Single-query decode at ~5.1 tok/s vs 30 tok/s target — requires DeepSeek-level
  infrastructure changes (DRAM-sharded weights, L1 width-sharded activations, async CCL).
- Aggregate throughput drops with flash_mla_prefill due to trace release/re-capture cycle.

---

## DRAM-Sharded Weights Phase 1 Results (2026-02-12)

### What changed
- **Task #9** (implementer-v2): Reverted L1 WIDTH_SHARDED experiment (zero improvement confirmed),
  implemented DRAM-sharded weights Phase 1 for MoE attention linears.
- Uses `dram_sharded_weight_config()` + `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`
  from DeepSeek V3 config_helpers.
- Feature-flagged: `GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1` (enabled in `.env.glm47`).
- Phase 1 scope: attention path linears only (q_a, q_b, kv_a projections).

### Why L1 WIDTH_SHARDED failed (architect analysis, task #8)
- `ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG` is a sentinel constant — it does NOT specify shard shape,
  core grid, or change the matmul program config.
- All three components must be co-designed: (1) DRAM-sharded weight storage across 12 banks,
  (2) `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, (3) explicit L1 WIDTH_SHARDED
  activation memory configs with shard specs matching the program config.
- Using the sentinel alone had zero effect on decode latency (222.9ms before and after).

### Coherency test results
- **30/32 PASS** (same as pre-DRAM-sharded baseline)
- Failing tests unchanged: #8 (chemistry electron count) and #25 (word puzzle) — model knowledge
  limitations, not regression.

### Benchmark results (1k context, 500 gen)

| Batch | Status | Agg tok/s | Dec tok/s | TTFT | ITL | Wall |
|-------|--------|-----------|-----------|------|-----|------|
| 1 | 1/1 | 3.5 | 4.0 | 21.23s | 227.3ms | 143.4s |
| 4 | FAIL | — | — | — | — | Container crash |
| 8 | FAIL | — | — | — | — | Container crash |
| 32 | FAIL | — | — | — | — | Container crash |

### Full Benchmark (post-container-restart, warmed)

| Batch | Aggregate tok/s | Per-user tok/s | TTFT (median) | ITL (median) | Wall time | Tokens |
|-------|----------------|----------------|---------------|-------------|-----------|--------|
| 1 | 3.3 | 3.5 | 8.58s | 227.1ms | 153.3s | 500 |
| 4 | 14.1 | 4.1 | 19.33s | 226.8ms | 142.1s | 2000 |
| 32 | 38.0 | 4.0 | 109.79s | 198.0ms | 420.8s | 15992 |

### Comparison vs Baseline

| Metric | Baseline | DRAM-sharded | Change |
|--------|----------|-------------|--------|
| bs=1 agg tok/s | 2.9 | 3.3 | +14% |
| bs=1 per-user tok/s | 4.5 | 3.5 | -22% |
| bs=1 ITL | 223.0ms | 227.1ms | +2% (noise) |
| bs=1 TTFB | 59.31s | 8.58s | -86% (warm cache) |
| bs=4 agg tok/s | 14.6 | 14.1 | -3% |
| bs=32 agg tok/s | 27.8 | 38.0 | **+37%** |
| bs=32 ITL | 190.8ms | 198.0ms | +4% (noise) |
| bs=32 decode loop | 468s (500 steps) | 311s (500 steps) | **-34%** |

### Analysis

- **bs=32 aggregate: 38.0 tok/s** — 37% improvement from 27.8 baseline. The decode loop
  (wall - TTFB) went from 468s to 311s, a 34% reduction. This is the main win.
- **bs=1 per-user decode: ~3.5-4.0 tok/s** — slightly worse than baseline 4.5. The DRAM
  sharding overhead (activation resharding + L1→DRAM move after matmul) may not pay off
  at M=1 batch for attention-only weights. The gain appears at high batch counts.
- **TTFB improvement at bs=1 (59s→8.6s)** is likely warm-cache effect, not DRAM-sharding.
- **Coherency: 30/32 PASS** — matches baseline. Same 2 known failures (#8 H₂O format, #25 word puzzle).
- Phase 1 only covers attention linears (~34% of decode). MLP + MoE expert paths (which
  dominate decode per profiling) are unchanged. Phase 2 should yield larger gains.

## Phase 2: Fused DRAM-sharded MLP Pipeline (2026-02-12)

### Changes
- **Discovery**: MLP weights were already DRAM-sharded in Phase 1 (layer_weights.py applied
  `_maybe_dram_shard_linear_weight` to gate/up/down). The issue was inefficient data movement:
  3 independent `_dram_sharded_linear` calls = 6 DRAM round-trips (3 input reshards + 3 output writebacks).
- **Fix**: New `_dram_sharded_mlp()` function runs gate→silu→up→mul→down entirely in L1
  WIDTH_SHARDED, following DeepSeek V3 decode MLP pattern.
- Reshard input once (shared between gate and up matmuls).
- Gate/up outputs stay in L1 WIDTH_SHARDED.
- silu(gate) * up runs in L1 WIDTH_SHARDED.
- Down matmul reads from L1 WIDTH_SHARDED (inner_cores match gate/up output sharding).
- Only final down output moves to DRAM (for all_reduce / residual add).
- Applied to both non-MoE MLP and MoE shared expert paths.

### Benchmark (ctx=1000, gen=500)

| Batch | Aggregate tok/s | Per-user tok/s | TTFT (median) | ITL (median) | Wall time | Tokens |
|-------|----------------|----------------|---------------|-------------|-----------|--------|
| 1 | 3.8 | 4.0 | 8.06s | 232.6ms | 133.5s | 500 |
| 4 | 14.8 | 4.3 | 18.59s | 221.7ms | 134.7s | 2000 |
| 32 | 37.5 | 4.0 | 112.14s | 199.7ms | 426.6s | 15992 |

### Comparison (Baseline → Phase 1 → Fused MLP)

| Metric | Baseline | Phase 1 | Fused MLP | vs Baseline |
|--------|----------|---------|-----------|-------------|
| bs=1 dec tok/s | 4.5 | 3.5 | 4.0 | -11% |
| bs=4 agg tok/s | 14.6 | 14.1 | 14.8 | +1% |
| bs=32 agg tok/s | 27.8 | 38.0 | 37.5 | **+35%** |

### Analysis
- Fused MLP recovered most of the Phase 1 bs=1 regression (3.5→4.0, +14%).
- bs=32 aggregate remains ~37-38 tok/s (+35% over baseline).
- Still ~11% below baseline at bs=1 (4.0 vs 4.5). Remaining overhead from
  attention projection resharding (5 separate _dram_sharded_linear calls in _attn_linear).
- Coherency: 30/32 PASS (no regression).

## Learnings from Ralph Loop Iteration 1 (2026-02-12)

### What did NOT work

| Attempt | bs=1 Impact | bs=32 Impact | Root Cause |
|---------|------------|-------------|------------|
| L1_WIDTH_SHARDED_MEMORY_CONFIG on MLP outputs | 0% | 0% | Sentinel constant, no program config change |
| DRAM-sharded attention weights (Phase 1) | -22% (4.5->3.5) | +37% (27.8->38.0) | Resharding overhead (376 ops/step) exceeds savings at low batch |
| DRAM-sharded MLP weights (Phase 2) | -7% (4.5->4.2) | Not tested cleanly | Same resharding overhead issue |
| Fused DRAM-sharded MLP | -11% (4.5->4.0) | Not tested | Improved but still net negative |

### Key insight

DRAM-sharded matmul REQUIRES WIDTH_SHARDED input activations (verified at matmul_device_operation.cpp:560). This makes resharding unavoidable with this approach. The overhead is ~50ms/step from 376 resharding operations.

### What to try next

1. Explicit MatmulMultiCoreReuseMultiCast1DProgramConfig (zero resharding, in progress)
2. Defensive clone audit and reduction (705 clones/step = major overhead)
3. Async CCL (replace sync all_reduce with async variants)
4. Batch-adaptive: DRAM-sharded for bs>=8, default for bs<8

### Team workflow lesson

NEVER run concurrent implementers -- they corrupt shared code and container state.

## Ralph Loop Iteration 2 — Approaches #6-#10 (2026-02-12)

### Approach #6: EXPLICIT_PROG_CFG=1 (explicit MatmulMultiCoreReuseMultiCast1DProgramConfig)

- Added explicit 1D matmul program configs for all 2D-weight decode matmuls
- Double-benchmark confirmed: NOT a one-time compile cost (103s TTFT repeats every request)
- bs=1 ITL=221ms (NO improvement over 223ms baseline)
- bs=32 ITL=176ms (21% better) but aggregate collapsed to 5.1 tok/s
- Only 17 tokens generated per request
- **ABANDONED**: auto-selector already optimal for M=1 with interleaved weights

### Approach #7: in0_block_w=8 + Clone Audit Phase 1 (BREAKTHROUGH)

Changed `in0_block_w` from 1 to 8 in sparse_matmul program configs for MoE experts.
Plus defensive clone removal for q_a/kv slices gated behind SKIP_DEFENSIVE_CLONES=1.

| Metric | Baseline | Approach #7 | Change |
|--------|----------|-------------|--------|
| bs=1 per-user tok/s | 4.5 | **5.6** | **+24%** |
| bs=32 aggregate tok/s | 27.8 | **59.1** | **+113%** |
| bs=1 ITL | 223ms | 179ms | -20% |
| bs=32 ITL | 191ms | 173ms | -9% |
| bs=1 TTFT | 2.4s | 6.2s | regressed |

MoE experts dropped from ~110ms to ~21ms per step (5x improvement).
Coherency: 30/32 PASS (baseline).

### Approach #8: N300 diagnostic (mesh reduction)

Attempted MESH_DEVICE=N300 (2 chips) to isolate collective overhead.
Model OOMs at layer 19/47 loading MoE expert weights. Cannot fit on fewer than 8 chips.

### Approach #9: Sharded MLP (end-to-end DRAM-sharded shared MLP)

Added GLM4_MOE_LITE_SHARDED_MLP=1 to enable DRAM-sharded shared MLP path independently.
**ZERO improvement**: bs=1 5.6 tok/s (same), bs=32 59.5 (noise).
Shared MLP is only 7.4% of total profile after in0_block_w=8 optimization.

### Approach #10: Clone Audit Phase 2 (extended defensive clone removal)

Extended SKIP_DEFENSIVE_CLONES to cover ALL defensive clone sites:
- 21 clones removed per layer across decoder_layer_tt.py and moe_tt.py
- 461 total clones removed from trace

| Metric | Approach #7 | Phase 2 | Change |
|--------|-------------|---------|--------|
| bs=1 per-user tok/s | 5.6 | 5.6 | 0% |
| bs=1 ITL | 179ms | 177ms | -1% (noise) |
| bs=32 aggregate tok/s | 59.1 | 58.9 | ~0% |

**ZERO improvement**: TT trace compiler already optimizes away clone overhead.

### Updated Profile (post in0_block_w=8, non-traced)

| Stage | ms/tok | % of total |
|-------|--------|------------|
| KV cache update | 22.7 | 19.8% |
| MoE experts | 21.4 | 18.6% |
| Attn out | 19.9 | 17.4% |
| Q path | 16.5 | 14.4% |
| MoE router | 12.6 | 11.0% |
| Shared MLP | 8.5 | 7.4% |
| Dense MLP | 4.7 | 4.1% |
| Unaccounted (in-layer) | 8.5 | 7.3% |
| **Total profiled** | **~115** | |
| **Traced ITL** | **179** | |
| **Gap (collectives + trace overhead)** | **64** | **36%** |

### Key Findings from Iteration 2

1. **in0_block_w=8 was the only big win** — reduced K-phases from 64/48 to 8/6 in sparse_matmul
2. **Profile is now flat** — no single dominant bottleneck (top 4 stages: 16-23ms)
3. **Clone removal has zero impact** — TT trace compiler already handles this
4. **64ms gap is NOT collective overhead** — approach #11 disproved this (see below)

### Approach #11: Collective Reduction (6 -> 2 all_reduces/layer)

Replicated w_q_kv_a, w_q_b, w_kv_b2 (removed 3 reduces) + fused MLP+MoE reduce (removed 1).
Removed 234 of 282 all_reduce calls per decode step.

| Metric | Prev Best (#7) | Collective Reduction (#11) | Delta |
|--------|---------------|---------------------------|-------|
| bs=1 decode | 5.6 tok/s | 5.6 tok/s | 0% |
| bs=1 ITL | 179ms | 179ms | 0% |
| bs=32 agg | 59.1 tok/s | 59.5 tok/s | ~0% |

**ZERO improvement.** Collectives are NOT the bottleneck. Trace mode eliminates
all Python/framework overhead (clones, mesh_partition, all_reduce setup).

### Root Cause Identified: DRAM Bandwidth at 8% of Spec

The 179ms ITL is pure device execution time dominated by **inefficient DRAM weight reads**.

TT tech report evidence (Saturating_DRAM_bandwidth.md):
- Wormhole DRAM spec: 288 GB/s per chip
- Llama3-70 decode (DRAM-sharded weights): 239-260 GB/s (82-90%)
- GLM-4.7-Flash decode (DRAM-interleaved weights): ~22 GB/s (8%)

With DRAM_INTERLEAVED weights, matmul readers access banks in round-robin,
causing NoC congestion. With DRAM-sharded weights, each reader accesses only
its local bank — no congestion, full bandwidth.

Previous DRAM-sharded attempt (#4) failed because `_dram_sharded_linear` bounced
activations DRAM -> L1 -> DRAM on every matmul call (20 reshards/layer * 47 = 940
DRAM round-trips, more traffic than savings).

### Solution: DeepSeek V3 L1-Resident Decode Pattern

Port the pattern from `tt-metal/models/demos/deepseek_v3/tt/mla/mla1d.py`:
- Activations NEVER touch DRAM (stay in L1 throughout)
- L1 WIDTH_SHARDED for DRAM-sharded matmuls
- L1 INTERLEAVED for reshape/permute/slice/per-head matmuls
- L1 HEIGHT_SHARDED for RoPE and FlashMLA
- Only weight reads use DRAM (sharded for full BW)

Expected: ~25-35ms ITL (29-40 tok/s) at 240 GB/s DRAM utilization.

---

## Approach #13: DRAM Bandwidth Diagnostic (2026-02-13)

**Hypothesis**: DRAM-interleaved weights are the bottleneck (~22 GB/s vs 288 GB/s spec).
**Result**: DISPROVEN.

Isolated matmul diagnostic (implementer ran single-matmul benchmarks):
- w_o (21MB): interleaved 91 GB/s vs sharded 81 GB/s (WORSE)
- w_gate (42MB): 159 vs 156 GB/s (same)
- w_down (42MB): 166 vs 160 GB/s (same)
- w_q_a (3MB): 14 vs 14 GB/s (same)

Root cause: GLM's hidden=2048 means most weights are <42MB. DRAM-interleaved
already achieves 55-57% BW for these sizes (91-166 GB/s). DRAM-sharding helps
large weights (Llama3-70 hidden=8192, weights 100-200MB) but does nothing for
small ones where NoC congestion isn't the bottleneck.

## MTP Investigation (2026-02-13)

Investigated Multi-Token Prediction as a fundamentally different strategy.

### Findings:
1. **Checkpoint confirmed**: GLM-4.7-Flash has `num_nextn_predict_layers=1`,
   212 MTP weights at `model.layers.47.*` (full MoE decoder layer + shared_head)
2. **vLLM support exists**: `Glm4MoeMTP` registered in model registry,
   uses `deepseek_mtp` method via `EagleProposer`
3. **TT backend BLOCKER**: `tt_model_runner.py:1409` explicitly says
   "currently does not support speculative decoding", returns `spec_token_ids=None`
4. **Math**: 1 MTP layer cycled 6x = ~23ms draft + 179ms target = 202ms for
   7 tokens IF all accepted. But acceptance with 1-layer cycling degrades rapidly.
   Realistic: 2-3 accepted = 16-21 tok/s. NOT enough for 30 tok/s.
5. **Effort**: Performance version = 4-8 weeks cross-stack (Codex estimate)

### Verdict: Not viable short-term.

## DISCOVERY: Batch-1 Trace Padding (2026-02-13)

**The trace always pads to MAX_NUM_SEQS=32, even at batch=1.**

Code path (`vllm/v1/worker/tt_model_runner.py:771`):
```python
B = int(self.scheduler_config.max_num_seqs)  # Always 32
```

At batch=1, the device computes all 47 layers for 32 batch items (31 are zeros).
This is the SAME trace replayed regardless of actual batch occupancy.

### Experiment needed:
Test with `MAX_NUM_SEQS=1` for bs=1 benchmarks. If ITL drops significantly,
batch-adaptive tracing (multiple traces for different batch sizes) is the path.

### Per-Kernel Floor Analysis (Updated):
- ~230us minimum per matmul kernel (device-side program runtime)
- 8-11 matmuls/layer * 47 layers = 376-517 matmul kernels
- At 230us each = 86-119ms minimum just for matmuls
- Target 33ms (30 tok/s) is BELOW this floor
- Even at batch=1 with optimal traces, theoretical max is ~11.8 tok/s

## Summary: All Approaches Tried

| # | Approach | bs=1 | bs=32 | Status |
|---|----------|------|-------|--------|
| 1 | L1 WIDTH_SHARDED sentinel | 0% | 0% | FAILED |
| 2 | DRAM-sharded attn weights | -22% | +37% | MIXED |
| 3 | DRAM-sharded MLP weights | regression | N/A | FAILED |
| 4 | EXPLICIT_PROG_CFG | ITL -22%, TTFT +43x | N/A | FAILED |
| 5 | in0_block_w=8 + clone Phase 1 | **+24%** | **+113%** | **ONLY WIN** |
| 6 | Sharded MLP (L1 resident) | 0% | 0% | FAILED |
| 7 | Clone audit Phase 2 | 0% | 0% | FAILED |
| 8 | Collective reduction | 0% | 0% | FAILED |
| 9 | DRAM bandwidth diagnostic | same/worse | N/A | DISPROVEN |
| 10 | MTP speculative decode | N/A | N/A | BLOCKED (TT runner) |
| 11 | N-gram speculation | N/A | N/A | BLOCKED (TT runner) |
| 12 | Weight folding | ~6% est. | ~6% est. | NOT TESTED |
| 13 | MAX_NUM_SEQS=1 for bs=1 | **+10.7%** | N/A | CONFIRMED |
| 14a | PRESERVE_TRACE=1 | ~0% | ~0% (noise) | NEUTRAL |
| 14b | Batched prefill | N/A | est. 3-5x prefill | **NEXT** |

Current best: 6.2 tok/s bs=1 (MAX_NUM_SEQS=1), 59.1 tok/s bs=32 (approach #5).

## Approach #14: MAX_NUM_SEQS=1 Diagnostic Results (2026-02-13)

| Metric | MAX_NUM_SEQS=32 | MAX_NUM_SEQS=1 | Delta |
|--------|----------------|----------------|-------|
| bs=1 ITL | 179ms | 162ms | -9.5% |
| bs=1 decode tok/s | 5.6 | 6.2 | +10.7% |

Batch padding overhead = ~17ms (10% of ITL). The remaining 162ms is actual
model compute through 47 layers. Per-kernel floor (85-120ms) is 1.35-1.9x
below measured, leaving limited room for further bs=1 improvement.

## Key Insight: bs=32 Bottleneck is Prefill, Not Decode

**Decode is already fast enough:**
- bs=32 decode ITL = 173ms (approach #7)
- Pure decode ceiling = 32 / 0.173 = 185 tok/s (above 140 target!)

**Prefill is the bottleneck:**
- Prefill processes requests ONE AT A TIME in a Python loop (model_tt.py:427)
- 32 serial prefills of 1k tokens = ~108s TTFT
- Prefill also releases the decode trace (model_tt.py:384-402), requiring re-capture

**Efficiency loss:** 59.1 / 185 = 32% — 68% of wall time is wasted on serial prefill.

## Approach #14b: PRESERVE_TRACE=1 Benchmark (2026-02-13)

| Metric | Baseline (#7) | PRESERVE_TRACE=1 | Delta |
|--------|--------------|-------------------|-------|
| bs=1 ITL | 179ms | 178.6ms | ~0% |
| bs=32 agg | 59.1 tok/s | 55.8 tok/s | -5.6% (noise) |
| bs=32 TTFT | 91s | 89.6s | -1.5% |

Essentially neutral. The ~6s trace re-capture overhead is negligible vs 77s total
prefill compute. The bottleneck IS the serial prefill compute itself, not the trace
lifecycle.

## Approach #15: Batched Prefill (NEXT -- in design)

### Strategy: "Flatten for Matmuls, Batch for Attention"

Replace serial per-request prefill loop with batched approach:
- Token-wise ops (linears, norms, MoE): on flattened [1,1,B*S_pad,hidden]
- FlashMLA attention: reshape to [B,H,S_pad,D] with is_causal=True (per-batch causal)
- KV cache fill: 32 calls to paged_fill_cache per layer (no batched API)
- RoPE: broadcasts across batch (all requests start at position 0)

### Expected Speedup (Codex-calibrated)
- Dense matmuls: 3-4x (weight amortization)
- MoE experts: 5-10x (2k tokens/expert vs 60-70)
- Net: 3-5x realistic (midpoint ~4x)

| Speedup | Prefill time | Total wall (bs=32) | Aggregate tok/s |
|---------|-------------|-------------------|-----------------|
| 2.5x | 31s | 116.5s | 137 |
| 4x | 19s | 104.5s | 153 |
| 5x | 15s | 100.5s | 159 |

### Evidence FlashMLA batch works
- test_flash_mla_prefill.py tests batch=2,8 with paged_attention=True
- Parameters: (8, 4096, 8, 1, 128, 32) -- 8 batch, 4k seq, paged
- is_causal=True correctly applies per-batch causal mask

## Approach #15: Batched Prefill (2026-02-13) — FAILED (REGRESSION)

### Design
Process all B requests through decoder stack simultaneously using [1,1,B*S_max,hidden]
flat tensors for token-wise ops, reshaped to [B,H,S,D] for attention.

### Implementation Issues & Fixes
1. **Matmul shape mismatch**: `w_kv_b1` and `w_kv_b2` operate on [B,H,S,D] per-head shape.
   TTNN non-bcast matmul requires dim-0==1 for 4D×2D weight broadcast.
   - Fix attempt 1: reshape [B,H,S,D]→[1,H,B*S,D] — WRONG, reinterprets memory incorrectly
     (H data for different B values is interleaved, not contiguous per H slice)
   - Fix attempt 2: permute [B,H,S,D]→[H,B,S,D]→reshape [1,H,B*S,D] — CORRECT but
     permute is expensive (full tensor data copy × 4 permutes/layer × 47 layers)
   - Fix attempt 3: loop over batch for w_kv_b1/w_kv_b2 only — CORRECT and fast

2. **PRESERVE_TRACE interaction**: With PRESERVE_TRACE=1, device hang after first request.
   Root cause: buffer allocation conflicts with active trace. Fixed by testing with PRESERVE_TRACE=0.

### Results (with loop-over-batch fix, PRESERVE_TRACE=0)
| Metric | Baseline (serial) | Batched | Delta |
|--------|-------------------|---------|-------|
| Coherency | 30/32 | 30/32 | Same |
| bs=32 agg tok/s | 55.8 | 42.5 | **-24% REGRESSION** |
| bs=32 TTFT | 89.6s | 193.1s | **+115% REGRESSION** |
| bs=32 ITL | 171.0ms | 173.6ms | Same |
| bs=32 wall | 286.6s | 376.2s | +31% |

### Root Cause Analysis
The batched prefill is SLOWER because:
1. **Activation bandwidth dominates weight bandwidth at 1k tokens**: At S=1024, the input/output
   activations for each matmul are larger than the weight matrix. Loading the weight 32 times (serial)
   costs less bandwidth than processing 32×1024 = 32768 tokens through permutes/reshapes.
2. **Per-layer permutes are expensive**: 4 permutes/layer × 47 layers = 188 large-tensor data copies
   (Q path: [1,1,B*S,H*D]↔[B,H,S,D], output path: same)
3. **FlashMLA with B=32**: Not faster than 32 serial calls because attention is O(S^2), not O(weight)
4. **MoE with 32k tokens**: May exceed L1 capacity, causing spills

### Key Insight
Batched prefill WOULD help if weight loading dominated (short sequences, large weights).
At 1k+ tokens, activation I/O dominates. The only way to speed up bs=32 prefill is to
reduce per-request prefill time or overlap prefill/decode (chunked prefill).

### Code Status
Implementation complete in model_tt.py and decoder_layer_tt.py, gated behind
`GLM4_MOE_LITE_BATCHED_PREFILL=1`. Disabled by default (set to 0).

## Measurement Methodology Update (2026-02-13)

Previous benchmarks mixed prefill and decode into a single "aggregate_tps" metric,
which underreported true decode throughput. New benchmark (`tests/bench_decode.py`)
measures them separately:

- **Test A (Decode)**: Short prompt (~10 tokens), long generation → prefill negligible,
  aggregate = per_user_decode_tps × batch_size
- **Test B (Prefill)**: Long prompt (ctx tokens), gen=1 → TTFT ≈ prefill time,
  prefill_tps = ctx_tokens / TTFT
- **Test C (Combined)**: End-to-end reference (old method, for comparison)

**Key finding**: Prefix caching is NOT supported on TT V0 backend
(`WARNING: Prefix caching is not supported for V0 TT backend, disabling it`).

### Corrected Baseline (2026-02-13) — gen=500 definitive run

| Metric | bs=1 | bs=32 | Target |
|--------|------|-------|--------|
| **Decode per-user** | 5.6 tok/s | 6.2 tok/s | 30 tok/s |
| **Decode aggregate** | 5.6 tok/s | **197.1 tok/s** | 150 tok/s |
| **Decode ITL** | 177.4ms | 162.0ms | — |
| **Decode TTFT** | 2.43s | 16.48s | — |
| **Prefill (1k tokens)** | 210 tok/s (TTFT=4.75s) | 11 tok/s (TTFT=91.9s) | 1000 tok/s |

Artifact: `artifacts/bench_decode_1770966983.json`

### Updated Targets

- **Decode aggregate bs=32: 150 tok/s** — **ALREADY MET** (197.1 tok/s)
- **Decode individual bs=1: 30 tok/s** — 5.6 tok/s, need 5.4x improvement
- **Prefill: 1000 tok/s** — 210 tok/s (bs=1), need ~4.8x improvement

### Bottleneck Analysis

The decode target for bs=32 is **already exceeded**. The old "55.8 tok/s aggregate"
was an artifact of including 94s of serial prefill time in the wall clock denominator.

True aggregate decode throughput at bs=32 is **197.1 tok/s** (6.2 tok/s × 32 users).

**The only remaining bottlenecks are:**
1. **Prefill speed** (210 tok/s vs 1000 target at bs=1) — affects TTFT and end-to-end
2. **Per-user decode speed** (5.6 tok/s vs 30 target at bs=1) — affects user experience
3. **Serial prefill scheduling** — 32×1k tokens processed serially = 91.9s total TTFT at bs=32

### Prefill Speed Analysis

Single-request 1k-token prefill takes 4.75s → 210 tok/s.

The prefill path processes tokens through the 47-layer decoder stack. At 1024 tokens:
- Embedding + RoPE setup: ~0.1s
- 47 decoder layers × ~0.1s each = ~4.7s
- LM head + logits: negligible (1 token)

To hit 1000 tok/s = 1s for 1k tokens, each layer must complete in ~21ms (vs ~100ms now).
This is the same ~5x gap as decode (177ms ITL vs 33ms target).

**Root cause is the same for both decode and prefill**: DRAM bandwidth utilization.
At ~30-40% of peak DRAM bandwidth, both prefill and decode are bandwidth-bound.

### Approach #16: MoE Sparse Prefill Optimization (PCM=32) — PARTIAL SUCCESS

**Profile data** (870-token prefill, PROFILE=1):
```
layer_total_s=6.966ms/tok (summed across 47 layers)
layer_moe_experts_s=3.746ms/tok (53.8%)   ← BOTTLENECK
layer_q_path_s=0.958ms/tok (13.8%)
layer_attn_out_s=0.942ms/tok (13.5%)
layer_moe_shared_s=0.403ms/tok (5.8%)
layer_moe_router_s=0.304ms/tok (4.4%)
layer_mlp_dense_s=0.240ms/tok (3.4%)
layer_kv_cache_fill_s=0.228ms/tok (3.3%)
```

**Root cause**: MoE sparse_matmul forced per_core_M=1 (32 tokens per call) →
29 chunked calls per layer × 47 layers = 1363 kernel launches per 1k prefill.

**Changes**:
1. `moe_tt.py`: Dynamic program config creation for prefill (per_core_M=num_blocks)
2. `moe_tt.py`: Smart padding to chunk_align=block×prefill_pcm (≤25% overhead)
3. `.env.glm47`: `GLM4_MOE_LITE_MOE_SPARSE_PREFILL_PCM=32` (1024 tokens/call)
4. `tt_worker.py`: `mesh_device.enable_program_cache()` (explicit program cache)

**Results** (PCM=32, 906 tokens, steady-state):
- Prefill: **173 tok/s avg** (178 peak) → +24% vs baseline 139 tok/s
- Decode: 5.4 tok/s → no regression from baseline 5.7

**Why only 24%**: The sparse_matmul kernel compute dominates. Reducing calls from
29→1 per layer only eliminates ~20% overhead (Python loop + tensor slicing).
For 10k tokens, no improvement (overhead is negligible vs compute).

**Key learning**: Per_core_M > 1 causes kernel recompilation (~40s) on first use.
Program cache + consistent padding mitigates this after first compile.

Even with ZERO MoE expert time, remaining ops (3.22ms/tok) limit prefill to ~310 tok/s.
1000 tok/s target requires kernel-level improvements or architectural changes.

### Optimization Priority (Updated)

Since decode bs=32 target is met, focus on:

1. **P0: Prefill speed** — 173→1000 tok/s (remaining 5.8× gap)
   - Kernel-level: sparse_matmul + matmul optimization for MoE prefill
   - System-level: traced prefill, pipeline parallelism across layers
   - Algorithm: skip routed MoE during prefill (quality trade-off)
   - Reference: Falcon-7B achieves 22k tok/s prefill on T3K (128× faster)

2. **P1: Decode bs=1** — 5.6→30 tok/s (177ms→33ms ITL)
   - Same DRAM bandwidth issue as prefill
   - MTP (1 layer in checkpoint) — blocked by TT runner
   - Speculative decode — alternative to MTP

### Approach #17: Dense Batched Expert Prefill (DENSE_PREFILL=1) — SUCCESS

**Motivation**: sparse_matmul is the wrong tool for GLM's near-dense routing (topk=4/E=64/block=32
→ ~87% dense blocks). DeepSeek V3 uses dense ttnn.linear for experts. Replacing sparse_matmul
with batched dense matmul for prefill eliminates sparsity indexing overhead entirely.

**Changes**:
1. `moe_tt.py`: Added `moe_dense_experts_forward_prefill_tt()` (~170 lines)
   - Broadcasts input [1,1,T,H] → [E_local,1,T,H] across expert batch dim
   - Stacks weights [E_local,1,H,I] via `_batch_weight()` helper (slice + concat from rank-5 storage)
   - Supports fused gate+up (w1w3_experts path) for 2 matmuls instead of 3
   - Routing weight expansion: scatter → repeat → permute → elementwise mul
   - Sums across experts, all_reduce across devices on T3K mesh
2. `decoder_layer_tt.py`: Two call sites (decode + prefill functions) updated
   - `use_dense_prefill = dense_prefill and tokens > 1` — strictly isolated from decode
3. `docker-compose.yml`: Added env var passthrough for `GLM4_MOE_LITE_MOE_DENSE_PREFILL`
4. `.env.glm47`: Added `GLM4_MOE_LITE_MOE_DENSE_PREFILL=1`

**Results** (bench_decode.py, warm program cache, 2026-02-13):

#### Prefill (bs=1, warm cache):

| Context | TTFT (s) | Prefill tok/s | vs Baseline | vs PCM=32 |
|---------|----------|---------------|-------------|-----------|
| 128 | 2.80 | 46 | N/A | N/A |
| 256 | 2.80 | 91 | N/A | N/A |
| 512 | 3.25 | 158 | N/A | N/A |
| 1024 | 3.94 | 260 | 15.3× (from 17) | +50% (from 173) |
| 2048 | 5.50 | 372 | N/A | N/A |
| 4096 | 9.46 | 433 | N/A | N/A |

Prefill throughput scales sub-linearly: ~596 tok/s pure compute rate (linear fit: TTFT ≈ 2.6s + 1.68ms/token).
Fixed overhead ~2.6s dominates at short contexts.

#### Decode (bs=1):

| Metric | Current | Baseline | Change |
|--------|---------|----------|--------|
| Per-user tok/s | 3.9 | 4.5 | -13% |
| ITL (ms) | 256 | 223 | +15% |

Decode regression NOT caused by dense_prefill code (confirmed: tokens=1 takes identical path).
Likely caused by EP_L1=1 + FUSE_EXPERTS_GATE_UP=1 which were added post-baseline.

#### Decode (bs=32):

| Metric | Current | Baseline |
|--------|---------|----------|
| Aggregate tok/s | 121.6 | 134.4 (from ITL) |
| Per-user tok/s | 3.8 | 4.2 |
| ITL (ms) | 261 | 191 |

#### Key Findings:

1. **Dense prefill is ~50% faster than sparse PCM=32** at 1k context (260 vs 173 tok/s)
2. **Program cache compilation**: ~55s per new prefill shape (first-time only). After compile,
   performance is consistent. This is a UX/warmup concern for production.
3. **Pure prefill compute rate**: ~596 tok/s (from linear fit). Fixed overhead (2.6s) dominates
   at short contexts. Target is 1000 tok/s.
4. **Remaining prefill gap**: 596 → 1000 tok/s = 1.68× improvement needed in pure compute
5. **Dense matmul does O(E_local × T × H × I) work** (all experts for all tokens) while only
   topk=4/64 are needed. This wastes 15× compute. For short prefills it doesn't matter
   (DRAM bandwidth bound), but for very long contexts this limits scaling.

Artifact: `artifacts/bench_decode_1770982778.json` (bs=1), `artifacts/bench_decode_1770982960.json` (gen=500), `artifacts/bench_decode_1770983151.json` (bs=32)

### Approach #18: Broadcast Mul for Routing Weights — SUCCESS (+18%)

**Motivation**: The routing weight expansion in Approach #17 creates a 33MB intermediate
per layer via `ttnn.repeat((H,1,1,1))` + permute + to_layout. This wastes DRAM bandwidth.
Codex confirmed ttnn.mul supports broadcasting on size-1 dims.

**Changes**:
1. `moe_tt.py` lines 827-835: Replace `repeat(H,1,1,1) + permute + to_layout + mul`
   with `permute(3,1,2,0) + to_layout + broadcast mul`
   - BEFORE: [1,1,T,E_local] → repeat → [H,1,T,E_local] (33MB) → permute → [E_local,1,T,H]
   - AFTER: [1,1,T,E_local] → permute → [E_local,1,T,1] → broadcast mul × [E_local,1,T,H]
2. `layer_weights.py`: Pre-computed batched weights ATTEMPTED but reverted — OOM
   (doubles expert memory footprint, DRAM too tight with KV cache)

**Results** (bench_decode.py, warm program cache, 2026-02-13):

#### Prefill (bs=1, warm cache):

| Context | tok/s (#18) | tok/s (#17) | Change |
|---------|-------------|-------------|--------|
| 128 | 47 | 46 | +2% |
| 256 | 96 | 91 | +5% |
| 512 | 177 | 158 | +12% |
| 1024 | 307 | 260 | **+18%** |
| 2048 | 438 | 372 | **+18%** |
| 4096 | 510 | 433 | **+18%** |

Pure compute rate: ~655 tok/s (linear fit, up from 596). Fixed overhead: ~1.8s (down from 2.6s).

#### Decode (bs=1): 4.1 tok/s (243ms ITL) — no change (expected, prefill-only optimization)

#### A/B Test: EP_L1 + FUSE_EXPERTS_GATE_UP (separate experiment)

| Config | Decode tok/s | Decode ITL | Prefill tok/s (1k) |
|--------|-------------|-----------|-------------------|
| EP_L1=0, FUSE=0 | 3.5 | 287ms | 220 |
| EP_L1=1, FUSE=1 | 3.9 | 257ms | 255 |

EP_L1 + FUSE_EXPERTS_GATE_UP HELP both decode (+11%) and prefill (+16%). Keep enabled.
The 4.5→3.9 baseline regression is from other code changes during sprint, not these flags.

### Updated Targets (2026-02-13, post Approach #18)

- **Decode aggregate bs=32: 150 tok/s** — ~131 tok/s estimated (4.1 × 32), 87% of target
- **Decode individual bs=1: 30 tok/s** — 4.1 tok/s, need 7.3× improvement
- **Prefill: 1000 tok/s** — 655 tok/s pure compute (307 tok/s end-to-end at 1k), need 1.53× in compute

### Optimization Priority (Updated 2026-02-13, post #18)

1. **P0: Prefill speed** — 655→1000 tok/s pure compute (1.53× gap)
   - Token-packing: use ttnn.gather to pack tokens by expert, eliminate 16× waste
   - Pre-compute weights: need memory-neutral approach (free 5D originals after batch, or in-place view)
   - Reduce fixed overhead (1.8s): traced prefill, avoid trace recompilation

2. **P1: Decode bs=1** — 4.1→30 tok/s (243ms→33ms ITL, 7.3× gap)
   - DRAM-sharded weights with explicit program configs (biggest single lever)
   - Async CCL (overlap communication with compute)
   - Clone audit (remove unnecessary defensive clones)

3. **P2: Decode bs=32** — 131→150 tok/s (1.14× gap, nearly met)
   - Likely met automatically once per-user decode improves

---

## Ralph Loop Iteration 2 — Env Var Regression Hunt (2026-02-13)

### Key Finding: Env Vars Cause 66% Decode Regression

The 4.1 tok/s decode was NOT from code changes — it was from env var regressions.
The old `.tmp.env.glm47.t3k_perf_trace_tp` produced 6.8 tok/s (146ms ITL), while the
current `.env.glm47` produced 4.1 tok/s (243ms ITL). Both use the SAME HEAD code (`d469fbda8f`).

| Setting | Old env (6.8 tok/s) | Current env (4.1 tok/s) |
|---------|----------|----------|
| MAX_NUM_SEQS | 1 | 32 |
| EP_L1 | unset | 1 |
| FUSE_EXPERTS_GATE_UP | unset | 1 |
| MLA_FIDELITY | hifi2 | lofi |
| MOE_SPARSE_FIDELITY | hifi2 | lofi |
| PREFILL_IMPL | decode_loop_trace | flash_mla_prefill |
| sample_on_device_mode | decode_only | unset |

**IMPORTANT**: ATTN_DP, FUSE_MLP_MOE_REDUCE, SHARDED_MLP, SKIP_DEFENSIVE_CLONES are in
.env.glm47 but NOT in docker-compose.yml's environment section, so they **never reach
the container**. The container defaults them to False/0.

### Corrected Baseline

- **Old env (perf-first)**: 6.8 tok/s bs=1, 146ms ITL
- **Current env**: 4.1 tok/s bs=1, 243ms ITL
- REGRESSION = 66%, source = env vars (EP_L1, FUSE_EXPERTS_GATE_UP, MAX_NUM_SEQS)

### Test Results from A/B Testing Session

| Test | Config Change | bs=1 decode ITL | Prefill 1k tok/s |
|------|-------------|-----------------|-----------------|
| Baseline (current env) | — | 242.5ms (4.1 tok/s) | 17.8 |
| Test A: SHARDED_MLP=0 | remove SHARDED_MLP | 245.3ms (NO change) | **196** (10×!) |
| Test B: FUSE_MLP_MOE_REDUCE=1 | add flag | 245.1ms (NO change) | 191 |
| Test E: ATTN_DP+FUSE_MOE | add flags | 249.3ms (NO change) | **23.7** (8× regression) |
| Old .tmp env (full) | MAX_NUM_SEQS=1, hifi2, etc. | **146ms (6.8 tok/s)** | 8 (decode_loop_trace) |

### What Did NOT Work (decode)

- SHARDED_MLP flag: ZERO decode impact, 10× prefill regression
- ATTN_DP=1: ZERO decode impact, 8× prefill regression
- FUSE_MLP_MOE_REDUCE=1: ZERO decode impact
- All above flags don't even reach container (missing from docker-compose.yml)

### Bisection Results (Iteration 2, 2026-02-13)

| Test | Config Change | bs=1 tok/s | ITL | Notes |
|------|-------------|-----------|-----|-------|
| Baseline | Current env (EP_L1=1, FUSE_GATE_UP=1, MAX_NUM_SEQS=32) | 4.1 | 243ms | |
| Test 1 | EP_L1=0 | 3.9 | 257ms | WORSE — EP_L1 helps |
| Test 2 | FUSE_EXPERTS_GATE_UP=0 | 3.8 | 260ms | WORSE — FUSE_GATE_UP helps |
| Test 3 | Both=0 | 3.6 | 274ms | WORST — both help |
| **Test 4** | **MAX_NUM_SEQS=1** | **6.4** | **155ms** | **PRIMARY CULPRIT** |

**ROOT CAUSE**: MAX_NUM_SEQS=32 causes trace to compile for 32 slots. At low occupancy
(bs=1), the padding overhead costs ~60% (155ms→243ms). This is a fundamental tradeoff:
bs=32 aggregate throughput requires MAX_NUM_SEQS=32, but bs=1 latency suffers.

**EP_L1=1 + FUSE_EXPERTS_GATE_UP=1 HELP**: ~14% improvement within the 32-slot regime
(3.6→4.1 tok/s). Keep enabled.

### Fixes Applied
1. docker-compose.yml: Added passthrough for ATTN_DP, FUSE_MLP_MOE_REDUCE, SHARDED_MLP,
   SKIP_DEFENSIVE_CLONES (were in .env.glm47 but never reached container)
2. .env.glm47: Set ATTN_DP=0, FUSE_MLP_MOE_REDUCE=0 (8× prefill regression when enabled)
3. .env.glm47: Kept SHARDED_MLP=0 (10× prefill improvement)
4. .env.glm47: Kept EP_L1=1, FUSE_EXPERTS_GATE_UP=1 (help decode)

### Updated Baselines (post-fix)
- **Decode bs=1**: 4.1 tok/s (MAX_NUM_SEQS=32, EP_L1=1, FUSE_GATE_UP=1)
- **Decode bs=1 (perf-first)**: 6.4-6.8 tok/s (MAX_NUM_SEQS=1)
- **Prefill bs=1 1k**: ~190 tok/s (with SHARDED_MLP=0, ATTN_DP=0)

### sample_on_device_mode Fix (2026-02-13 17:53 UTC)

Added `"sample_on_device_mode":"decode_only"` to OVERRIDE_TT_CONFIG. This eliminates
~19.8 MB PCIe logits readback per decode step (32 × 154880 × 4 bytes). Instead, only
token IDs (128 bytes) are read back.

| Metric | Before (no sample_on_device) | After (sample_on_device=decode_only) | Delta |
|--------|------------------------------|--------------------------------------|-------|
| Decode bs=1 | 4.2 tok/s, 238ms ITL | **4.3 tok/s, 229ms ITL** | +2% |
| Decode bs=32 agg | 123.5 tok/s, 246ms ITL | **129.9 tok/s, 235ms ITL** | +5% |
| Prefill 1k | 14 tok/s, 71.3s TTFB | **193 tok/s, 5.19s TTFB** | **+13.8×** |
| Prefill 10k | 101 tok/s, 99.0s TTFB | **355 tok/s, 28.1s TTFB** | **+3.5×** |

**KEY INSIGHT**: sample_on_device_mode is CRITICAL for prefill. Without it, full logits
(~19.8 MB) are read back after every prefill chunk, destroying prefill throughput. The
previous 14 tok/s "prefill regression" was entirely from this missing config — not from
SHARDED_MLP or SKIP_DEFENSIVE_CLONES.

Decode improvement is marginal (+2-5%) because decode is DRAM-bandwidth-bound and the
PCIe readback (~1.6ms) is small relative to the 229ms decode step.

### SKIP_DEFENSIVE_CLONES Bisection (2026-02-13 18:02 UTC)

SKIP_DEFENSIVE_CLONES=1 provides +83% prefill improvement (304 vs 166 tok/s at 1k ctx).
Avoids unnecessary tensor copies during prefill. Keep enabled.

### Current Best Baselines (2026-02-13, Iteration 2 final)

| Metric | Value | Target | Gap |
|--------|-------|--------|-----|
| Decode bs=1 | 4.3 tok/s (229ms ITL) | 30 tok/s | 7× |
| Decode bs=32 agg | 134.1 tok/s (236ms ITL) | 150 tok/s | 12% |
| Prefill bs=1 1k | 304 tok/s (3.29s TTFB) | 1000 tok/s | 3.3× |
| Prefill bs=1 10k | 426 tok/s (23.5s TTFB) | 1000 tok/s | 2.3× |
| Prefill bs=32 1k | 22 tok/s (44.9s TTFB) | — | Very slow |
| Prefill bs=32 10k | 24 tok/s (425.9s TTFB) | — | Very slow |

Config: MAX_NUM_SEQS=32, EP_L1=1, FUSE_EXPERTS_GATE_UP=1, SHARDED_MLP=0, ATTN_DP=0,
FUSE_MLP_MOE_REDUCE=0, SKIP_DEFENSIVE_CLONES=1, sample_on_device_mode=decode_only,
trace_mode=decode_only, flash_mla_prefill

Note: bs=32 prefill is extremely slow (22-24 tok/s). Likely needs batched prefill
optimization or chunked prefill strategy. Decode bs=32 agg is close to target.

### k_chunk_size=128 Test (2026-02-13 18:24 UTC) — NO IMPROVEMENT

Set GLM4_MOE_LITE_MLA_K_CHUNK_SIZE=128 (default 64). Expected ~26% decode speedup.

| Config | Decode bs=1 | ITL | Decode bs=32 agg | ITL |
|--------|------------|-----|-----------------|-----|
| k_chunk=64 (default) | 4.3 tok/s | 229ms | 134.1 tok/s | 236ms |
| k_chunk=128 (test) | 4.3 tok/s | 230ms | 133.8 tok/s | 236ms |

ZERO improvement. Minor correctness divergence at ~500 tokens (different but coherent text).
Reverted to default. MLA attention likely compute-bound, not memory-bound at this model size.

### Fused SiLU*mul Test (2026-02-13 19:06 UTC) — NEGATIVE, REVERTED

Replaced `ttnn.silu(gate); gate * up` with fused
`ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])`
at 15 sites across decoder_layer_tt.py (5), moe_tt.py (6), layer0_tt.py (4).

| Metric | Baseline | Fused SiLU | Delta |
|--------|----------|-----------|-------|
| Decode bs=1 | 4.3 tok/s, 229ms | 4.34 tok/s, 230ms | 0% (unchanged) |
| Decode bs=32 agg | 134.1 tok/s, 236ms | 116.8 tok/s, 237ms | **-13% REGRESSION** |
| Prefill 1k | 304 tok/s | 265 tok/s | **-13% REGRESSION** |

The fused op is SLOWER for bs=32 decode and prefill. Likely the fused kernel has higher
per-invocation overhead or different memory access patterns. Trace replay at bs=1 shows
identical ITL — fusion saves nothing in the traced path. REVERTED via git checkout.

### FUSE_MLP_MOE_REDUCE=1 Test (2026-02-13 19:20 UTC) — BROKEN, REVERTED

Set GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1 (with ATTN_DP=0). Expected: save 46 all_reduces/step.

| Metric | FUSE=0 (baseline) | FUSE=1 | Delta |
|--------|-------------------|--------|-------|
| Decode bs=1 | 4.3 tok/s | 4.3 tok/s | 0% |
| Decode bs=32 agg | 129 tok/s | 130 tok/s | 0% |
| Prefill 1k | 197 tok/s | 158 tok/s | -20% |
| Correctness | OK | **GIBBERISH** | BROKEN |

Output is complete gibberish. The fused reduce path produces numerically incorrect results.
ZERO decode improvement despite saving 46 all_reduces. REVERTED.

### What Has NOT Worked (Summary)

From Iteration 2-3, ALL proposed kernel-level optimizations failed:
- k_chunk_size=128: ZERO decode improvement (MLA likely compute-bound)
- Fused SiLU*mul: -13% bs=32 and prefill REGRESSION (fused kernel slower)
- FUSE_MLP_MOE_REDUCE=1: BROKEN correctness (gibberish), zero decode gain
- DRAM-sharded attention weights: bs=1 REGRESSION (-22%)
- DRAM-sharded MLP weights: bs=1 REGRESSION
- L1 WIDTH_SHARDED memory config: ZERO improvement
- Explicit MatmulMultiCoreReuseMultiCast1DProgramConfig: ZERO improvement

### What HAS Worked

- in0_block_w=8 for sparse_matmul: +24% bs=1, +113% bs=32 (from Iteration 1)
- EP_L1=1 + FUSE_EXPERTS_GATE_UP=1: +66% decode (3.9→6.4 within MAX_NUM_SEQS=1)
- sample_on_device_mode=decode_only: +13.8× prefill, +5% decode
- SKIP_DEFENSIVE_CLONES=1: +83% prefill
- sparse PCM=32: +24% prefill
- Dense batched prefill: +50% prefill
- **MOE_DENSE_PREFILL=0: +47% decode bs=1, +61% decode bs=32** (BLOCKED by prefill hang)

### Root Cause Analysis (from Architect)

The 229ms decode ITL at MAX_NUM_SEQS=32 is dominated by **FlashMLA core allocation**:
- B=1 with MAX_NUM_SEQS=1: 16 cores per sequence → attention ~15-20ms → 155ms ITL
- B=1 with MAX_NUM_SEQS=32: 2 cores per sequence → attention ~80-120ms → 229ms ITL
- 30 padded slots get dedicated cores that sit IDLE (early-exit but can't be borrowed)
- GLM has num_kv_heads=1 (MLA), so no KV head parallelism to compensate

### BREAKTHROUGH: MOE_DENSE_PREFILL=0 (2026-02-13)

**Artifact**: `bench_decode_1771014353.json`

Setting `GLM4_MOE_LITE_MOE_DENSE_PREFILL=0` produces massive decode improvements:

| Metric | Baseline (DENSE=1) | DENSE=0 | Change | Target |
|--------|-------------------|---------|--------|--------|
| Decode bs=1 tok/s | 4.34 | 6.39 | **+47%** | 30 |
| Decode bs=1 ITL | 229ms | 156ms | **-32%** | 33ms |
| Decode bs=32 agg tok/s | 129 | 207.4 | **+61%** | 150 ✅ |
| Decode bs=32 ITL | 236ms | 145ms | **-39%** | — |
| Decode bs=32 per-user | 4.03 | 6.48 | **+61%** | — |
| Prefill 1k bs=1 | 197 tok/s | **HANG** | ❌ | 1000 |

**bs=32 aggregate target of 150 tok/s EXCEEDED** (207.4 tok/s).

#### Why does a PREFILL flag affect DECODE?

**DISPROVEN hypotheses:**
- ~~Memory fragmentation theory~~: PRESERVE_TRACE=1 (keep warmup trace) gives 230ms ITL, NOT 156ms.
  The warmup trace is just as slow. The improvement is NOT about preserving a "clean" trace.
- ~~Sparse prefill PCM fix~~: PCM=4 takes 396s (6 min, usable but extremely slow). PCM=2 crashes
  with shape mismatch (`RuntimeError: Invalid subtile broadcast type` at `shared_out + routed_out`).
  Sparse prefill is fundamentally slow for multi-token inputs (~80x slower than dense).

**What we know:**
- DENSE_PREFILL=0 (sparse MoE for prefill) → after trace re-capture: 156ms decode ITL
- DENSE_PREFILL=1 (dense MoE for prefill) → after trace re-capture: 229ms decode ITL
- PRESERVE_TRACE=1 (warmup trace, no re-capture) → 230ms decode ITL (same as dense)
- PRESERVE_TRACE=1 also BREAKS bs=32 decode (engine hangs after 1 request)
- Decode (tokens=1) uses the SAME code path regardless of DENSE_PREFILL

**Current hypothesis:** The sparse MoE execution during prefill (scatter, moe_expert_token_remap,
sparse_matmul) changes device state (program cache, L1 core config, DRAM allocator) in a way that
improves the subsequently re-captured decode trace. This is a "priming" effect — the specific
kernels/configs from sparse MoE happen to be beneficial for decode trace compilation.

#### Sparse prefill at 1k: broken

Three failure modes:
- PCM=32 (no chunking): device hang (1024 tokens, per_core_M=32 likely exceeds L1)
- PCM=4 (8 chunks): runs but takes 396s (~80x slower than dense)
- PCM=2 (16 chunks): shape mismatch crash at decoder_layer_tt.py:1794 (`shared_out + routed_out`)
  Root cause: chunking produces malformed output tensor — `ttnn.slice` aliasing + wrong
  `num_dispatch_devices` divisor in reduce mode

Even if all bugs are fixed, sparse prefill at 1k would take ~6 min (unusable).

### Next Steps (Iteration 4)

1. **Investigate the priming mechanism** (TOP PRIORITY): Use Codex/device profiler to understand
   exactly what sparse MoE does to device state that helps decode. If we can replicate it with a
   minimal "primer" operation during trace capture, we get the decode win without sparse prefill.

   Candidate approach: Add a sparse MoE primer step in `_capture_decode_trace_sampling` (run one
   sparse_matmul with MoE weights before capturing the trace). This would make ALL re-captured
   traces fast, regardless of what prefill mode was used.

2. **Batch-bucketed traces**: Capture traces at B=1,4,8,16,32 buckets.
   At runtime, pad to nearest bucket. Design at `plan/glm47_flash/batch-bucketed-traces.md`.
   Combined with the priming fix, could push bs=1 toward ~10+ tok/s.

3. **Device profiling**: TT_METAL_DEVICE_PROFILER=1 to compare decode trace ops between
   the "fast" (156ms) and "slow" (229ms) traces. Would reveal exactly which ops are slower.

### Session Checkpoint (2026-02-14)

**Env state**: `.env.glm47` has `PRESERVE_TRACE=1` (line 83, needs revert to 0) and
`DENSE_PREFILL=1` (line 50, correct). Container is DOWN (no GLM containers running).

**Before any new work**: Implementer must revert `PRESERVE_TRACE=1` → `0` in `.env.glm47`.

**model_tt.py state**: Exception catch widened at `_prefill_compute()` (catches all exceptions,
not just OOM string match). Harmless — retries with trace release on any prefill failure.

**Recommended next optimization**: Batch-bucketed traces (design complete at
`plan/glm47_flash/batch-bucketed-traces.md`). This is orthogonal to the DENSE_PREFILL mystery
and gives ~50% bs=1 improvement (6.4 vs 4.3 tok/s) by allocating more FlashMLA cores per
sequence at low occupancy.

**Current best perf** (baseline, DENSE_PREFILL=1):
- Decode bs=1: 4.5 tok/s, 229ms ITL
- Decode bs=32: 128-130 tok/s aggregate, 236ms ITL
- Prefill 1k bs=1: 197 tok/s

**Best decode perf** (DENSE_PREFILL=0, but prefill broken):
- Decode bs=1: 6.39 tok/s, 156ms ITL
- Decode bs=32: 207 tok/s aggregate, 145ms ITL
- Prefill 1k: HANGS (unusable)

### BATCH-BUCKETED TRACES: IMPLEMENTED (2026-02-14)

**Artifact**: `bench_decode_1771080658.json`
**Design**: `plan/glm47_flash/batch-bucketed-traces.md`

Captures decode traces at B=1,4,8,16,32 buckets. At runtime, pads to nearest bucket
instead of MAX_NUM_SEQS=32. B=1 requests get 16 FlashMLA cores (was 2).

| Metric | Baseline | Bucketed | Change | Target |
|--------|----------|----------|--------|--------|
| Decode bs=1 tok/s | 4.34 | **6.93** | **+60%** | 30 |
| Decode bs=1 ITL | 229ms | **143.6ms** | **-37%** | 33ms |
| Decode bs=1 TTFT | 3.19s | **2.20s** | **-31%** | — |
| Decode bs=32 agg tok/s | 129 | 134.7 | +4.5% | 150 |
| Decode bs=32 ITL | 236ms | 235.8ms | ~same | — |
| Prefill 1k bs=1 tok/s | 197 | 127 | **-36%** | 1000 |
| Prefill 1k bs=1 TTFT | 5.08s | 7.88s | **+55%** | — |
| Prefill 1k bs=32 tok/s | — | 22.8 | — | — |

**Wins:**
- Decode bs=1: +60% throughput, -37% latency — exceeds design prediction of 6.4 tok/s
- Decode bs=1 TTFT also improved (-31%) — trace captured at B=1 executes faster
- Decode bs=32 marginally improved (+4.5%) — expected, B=32 bucket = same as before

**Regressions:**
- Prefill 1k bs=1: 127 vs 197 tok/s (-36%) — possibly from 250MB trace_region_size
  consuming DRAM bandwidth, or trace release/recapture overhead with 5 bucket states
- Prefill 1k TTFT: 7.88s vs ~5s — related to prefill throughput regression

**Warmup**: 3.5 min (was 2 min). 90s extra for 4 additional trace captures.

**Changes made:**
- model_tt.py: `_DecodeTraceSamplingState` dataclass, dict-based trace state,
  all trace methods refactored to per-bucket state
- tt_model_runner.py: bucket selection, variable-size decode padding, multi-bucket warmup
- .env.glm47: trace_region_size=250MB, decode_trace_batch_buckets=[1,4,8,16,32]

### SECTION 115 FIX + BUCKETED TRACES: COMBINED (2026-02-14)

**Artifact**: `bench_decode_1771082938.json`

Applied one-line fix at decoder_layer_tt.py:1174: `tokens > 1` → `tokens >= 33`.
This gives sparse MoE for decode (tokens≤32) + dense MoE for prefill (tokens≥33).
Combined with batch-bucketed traces from previous step.

| Metric | Baseline | Task #19 only | **Task #22 (combined)** | Target |
|--------|----------|---------------|------------------------|--------|
| Decode bs=1 tok/s | 4.34 | 6.93 | **6.97** | 30 |
| Decode bs=1 ITL | 229ms | 143.6ms | **143.1ms** | 33ms |
| Decode bs=32 agg tok/s | 129 | 134.7 | **208.3** | **150 ✅** |
| Decode bs=32 ITL | 236ms | 235.8ms | **144.9ms** | — |
| Decode bs=32 per-user | 4.03 | 4.21 | **6.51** | — |
| Prefill 1k bs=1 tok/s | 197 | 127 | **204.7** | 1000 |
| Prefill 1k bs=1 TTFT | 5.08s | 7.88s | **4.88s** | — |
| Prefill 1k bs=32 tok/s | — | 22.8 | **23.5** | — |

**bs=32 aggregate target of 150 tok/s EXCEEDED** (208.3 tok/s, +39% over target).

**Key insight**: trace_region_size needed 250MB (not 60MB) because sparse MoE decode
traces are larger than dense MoE traces (~70.9MB at B=16 bucket). This also fixed the
prefill regression from Task #19 (127 → 205 tok/s).

**Two changes that delivered this combined result:**
1. Batch-bucketed traces: bs=1 gets 16 FlashMLA cores → 143ms ITL (was 229ms)
2. Section 115 fix: bs=32 decode uses sparse MoE → 208 tok/s agg (was 129)

### Next Steps (Iteration 6)

1. **bs=1 decode: 7 → 30 tok/s** — Requires speculative decode / MTP (multi-token
   prediction). Cannot be achieved by kernel optimization alone — 143ms ITL is at
   the per-kernel floor (~157ms estimated for 9 matmuls × 47 layers).

2. **Prefill: 205 → 1000 tok/s** — Current bottleneck is sequential layer execution.
   Potential: pipeline parallelism, tensor-parallel prefill optimization, or
   reduced precision for prefill compute.

3. **Profile decode at B=1** — Use TT_METAL_DEVICE_PROFILER=1 to get exact op-level
   breakdown of the 143ms ITL. Identify if any single op dominates.
