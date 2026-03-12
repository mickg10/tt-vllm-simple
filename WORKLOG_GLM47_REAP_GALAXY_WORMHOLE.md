# GLM-4.7-REAP-218B on Galaxy Wormhole: Worklog

Canonical planning/docs live outside git at `/home/ttuser/src_docker/plan/glm47_reap/galaxy_wormhole/`.
Committed research reports at `docs/glm_47_reap/galaxy_wormhole/`.

## Run Command

```bash
cd /home/user/src_docker/ws/glm47_reap_268b_galaxy_wormhole/docker_tt
sg docker -c 'docker compose --env-file dev/.env.glm47_reap \
  -f dev/docker-compose.yml -f dev/docker-compose.galaxy.yml \
  up -d vllm-tt'
```

Use `--force-recreate` when env vars change (triggers weight reload, ~10-20 min).
**NEVER use `--force-recreate` casually** — weight cache rebuild takes ~60-66 min if dtypes change.

**Galaxy SSH**: `ssh -p 55211 user@38.97.6.6`

## Model

- **cerebras/GLM-4.7-REAP-218B-A32B** (268B total, 32B active)
- 92 MoE layers, 96 routed experts (top-8), GQA 96Q/8KV
- hidden_size=5120, moe_intermediate_size=5120, num_attention_heads=96, num_kv_heads=8

## Hardware & Parallelism

- Galaxy Wormhole: 32 chips, Mesh(8,4), 8x9=72 cores/chip, 12 DRAM channels/chip
- TP=8 (axis 0), EP=32 (all chips), DP=4 (axis 1)
- MESH_DEVICE=TG, FABRIC_2D

## Current Status (2026-03-12) — 92 tok/s steady-state aggregate, SEED-3 BF4

**All numbers from server-side TPOT histogram** (`vllm:time_per_output_token_seconds`),
measured via delta method (before/after sum and count). Client-side wall-time is unreliable.

| Batch | Steady-state TPOT | Aggregate tok/s | Config |
|-------|-------------------|-----------------|--------|
| bs=1  | 150-200ms         | ~5-6.7          | SEED-3: w1/w2 BF4, w3/attn/shared BF8 |
| bs=32 | **348ms**         | **~92**         | SEED-3: w1/w2 BF4, w3/attn/shared BF8 |

**Note**: Previous "142.7 tok/s" was wall-time aggregate from bench_itl.py, which is inflated
because it includes prefill-decode overlap in the denominator. Server-side TPOT histogram is
the ground truth for decode performance.

**Full BF4 re-test results (2026-03-10)**:

| Test | Config | bs=32 agg tok/s | PPL | vs BF8 baseline |
|------|--------|-----------------|-----|-----------------|
| Test 0 (BF8 baseline) | All BF8 | 98.9 | 1.3659 | — |
| **Test 1 (SEED-3)** | **w1/w2=BF4, w3=BF8** | **142.7** | **1.2288** | **+44.3% throughput, -10.0% PPL** |
| Test 2 (All BF4) | w1/w2/w3=BF4 | 126.4 | 1.2884 | +27.8%, -5.7% PPL |
| Test 3 (Protect w2) | w1=BF4, w2=BF8, w3=BF4 | 128.3 | 1.4007 | +29.7%, +2.5% PPL (WORSE) |

**SEED-3 is the optimal config**: best throughput AND best quality.
Theoretical sensitivity analysis predicted w2 most sensitive — empirically WRONG.
Protecting w3 (up projection, BF8) while quantizing w1/w2 (gate/down, BF4) is the sweet spot.

## Configuration

```
GLM4_MOE_DENSE_TT_DTYPE=bf8                # Shared expert + attention stays BF8
GLM4_MOE_EXPERTS_TT_DTYPE=bf8              # Base expert dtype (overridden per-projection)
GLM4_MOE_EXPERTS_W1_DTYPE=bf4              # Gate projection: BF4 (+44% SEED-3)
GLM4_MOE_EXPERTS_W2_DTYPE=bf4              # Down projection: BF4 (+44% SEED-3)
GLM4_MOE_EXPERTS_W3_DTYPE=bf8              # Up projection: stays BF8 (best quality)
GLM4_MOE_REDUCE_IMPL=native                # axis-0 (TP) all_reduce
GLM4_MOE_REDUCE_IMPL_AXIS1=rs_ag_async     # axis-1 (DP) — bypasses FABRIC_2D crash
GLM4_MOE_EP_REDUCE_DEVICE=1
GLM4_MOE_FUSE_SHARED_EP_REDUCE=0           # BROKEN: hang on TG
GLM4_MOE_EP_L1=0                           # BROKEN: garbled on TG
GLM4_MOE_DRAM_SHARD=0                      # +2.1% (not significant)
GLM4_MOE_ATTN_FIDELITY=hifi
GLM4_MOE_MOE_SPARSE_FIDELITY=hifi
GLM4_MOE_FUSE_EXPERTS_GATE_UP=0            # Disabled: -14.6% regression
MAX_NUM_SEQS=32
decode_trace_batch_buckets=[32]          # Single bucket: prevents trace recapture crash
trace_mode=decode_only
```

## Performance Progression

```
0.17 tok/s  → First working decode (eager mode, host-side everything)
1.2  tok/s  → Device RMSNorm + EP reduce
2.7  tok/s  → Full 32-way all_reduce + trace mode
3.5  tok/s  → Optimized host-side sampling
99.5 tok/s  → Batch>1 unblock (rs_ag_async) — 28.4x aggregate scaling
121.6 tok/s → Batch-bucketed attention init + L1 threshold fix (+22%)
126.4 tok/s → All-expert BF4 (+28% over prior BF8 baseline)
142.7 tok/s → SEED-3 selective BF4: w1/w2 BF4, w3 BF8 (+44% over BF8, best PPL)
~92  tok/s → Corrected measurement: server-side TPOT histogram (348ms/step at bs=32)
```

## History

### Phase 0: Initial Bringup (2026-02-28 to 2026-03-05)

Brought REAP-218B up from scratch on Galaxy Wormhole TG mesh.

- **0.17 tok/s**: First working decode. Eager mode, host-side everything.
- **1.2 tok/s**: Device RMSNorm, device EP reduce (2-step axis-0 then axis-1).
- **2.7 tok/s**: Full 32-way all_reduce (no cluster_axis), trace mode enabled.
- **3.5 tok/s**: Optimized host-side sampling, trace capture stabilized.

Key challenges:
- TG mesh ops (to_layout, slice, max, argmax) ALL HANG even outside trace.
  Forward pass ops (matmul, all_reduce, rms_norm, SDPA) work fine inside trace.
- Sampling MUST stay on host (`_host_argmax_from_trace_logits`, per-shard argmax).
- Device embedding removed to free 1.49 GB DRAM for trace buffer allocation.
- Trace re-captured each request (trace_region_size=0, prefill buffers overlap).

### Phase 1: V1 Quick Wins — ALL FAILED (2026-03-06 to 2026-03-07)

| Optimization | Result | Root Cause |
|---|---|---|
| EP_L1=1 (expert intermediates in L1) | GARBLED output | L1 memory config incompatible with TG mesh CCL |
| FUSE_SHARED_EP_REDUCE=1 | INFINITE HANG (device corruption) | CCL deadlock on 2D mesh |
| LoFi math fidelity (attn + experts) | -1.5% SLOWER | DRAM-BW-bound, not compute-bound |

**Critical finding**: REAP-218B at bs=1 is DRAM-bandwidth-bound. ~15% DRAM utilization.
~4.2 GB weight reads per decode per device. Compute reduction is useless.

### Phase 2: DRAM-BW Focus (2026-03-08 to 2026-03-09)

| Optimization | Result | Notes |
|---|---|---|
| V2 P1: BF8 dense weights | +2.4% at bs=1, **+18.8% at bs=32** | Compounds with batch amortization |
| V2 P2: Device-side sampling (topk) | +0% | Host sampling hidden behind device trace |
| V2 P3: DRAM-sharded attn weights | +2.1% (not significant) | Per-device matmuls too small after TP=8 |
| **V2 P6: Batch>1 unblock** | **3.5 → 99.5 tok/s (28.4x)** | rs_ag_async + 3 reshape fixes |

### Batch>1 Unblock (2026-03-09) — THE Breakthrough

**Root cause of batch>1 crash**: On TG FABRIC_2D, `ttnn.all_reduce(cluster_axis=1)` is
forced into a composite path (`all_gather + local_sum(dim=0)`) in `all_reduce_async.cpp`
that crashes for batch>=2. The natural `reduce_scatter(dim=3) + all_gather(dim=3)` path
works correctly.

**Fix** (5 changes):
1. `attention_tt.py:32,94` — `_REDUCE_IMPL_AXIS1` per-axis env var override
2. `attention_tt.py:618,713,729` — Three reshape fixes: two-shape `(logical, physical)`
   form for TG batch-slicing code path (was volume mismatch crash)
3. `.env.glm47_reap` — `GLM4_MOE_REDUCE_IMPL_AXIS1=rs_ag_async`, `MAX_NUM_SEQS=32`
4. `generator_vllm.py:255` — `warmup_batch = max_batch_size` (removed forced bs=1)
5. `docker-compose.yml` — `GLM4_MOE_REDUCE_IMPL_AXIS1` env passthrough

**Validated by**: Gemini 2.5 Pro (inline) + Codex 5.4 (background verification).
Both confirmed rs_ag_async safe for trace mode, negligible performance cost.

### Session 2: SEED Sprint (2026-03-09)

#### SEED-1: Batch-Bucketed Attention (+22% at bs=32)
- Pre-compute `_slice_mats`/`_user_sel_mats` dicts per batch bucket, select at runtime
- L1 fast path threshold raised from <=32 to <=64 in moe_tt.py
- B_phys calculations fixed to use `active_batch` and `logical_batch_after_slice`
- **Result**: 99.5 → 121.6 tok/s aggregate at bs=32

#### SEED-2: Fused Gate+Up Sparse Matmul — REJECTED
- **-14.6% regression at bs=1** — unnecessary `ttnn.clone` calls (porting bug from glm4_moe_lite)
  added 267 extra trace programs (+12-15ms)
- Even with clones removed, predicted break-even (slices offset dispatch savings)
- Code kept but disabled: `GLM4_MOE_FUSE_EXPERTS_GATE_UP=0`

#### SEED-3: BF4 Expert Weights — CONFIRMED VIABLE ✅
- **Original test (2026-03-10 morning)**: Selective BF4 (w1/w2=BF4, w3=BF8) appeared garbled.
  Flawed test: eyeball-only, no PPL, stale trace artifacts, untested code path.
- **Davor's proof**: DSv3/R1 with BFP4_b nearly lossless (256 experts × 58 layers, -0.15pp MMLU).
  Both models use routed_scaling_factor=2.5. TT BFP4_b has 1024x finer exponent granularity
  than DS's original FP8 (16:1 vs 16,384:1 data-to-scaler ratio).
- **Comprehensive re-test (2026-03-10)**: 4 configs tested with PPL + throughput + quality eval:

| Test | Config | bs=32 agg | PPL | vs BF8 |
|------|--------|-----------|-----|--------|
| 0 | All BF8 (baseline) | 98.9 | 1.3659 | — |
| **1 (SEED-3)** | **w1/w2=BF4, w3=BF8** | **142.7** | **1.2288** | **+44%, -10% PPL** |
| 2 | All BF4 | 126.4 | 1.2884 | +28%, -5.7% PPL |
| 3 | w1=BF4, w2=BF8, w3=BF4 | 128.3 | 1.4007 | +30%, +2.5% PPL |

- **SEED-3 (w1/w2=BF4, w3=BF8) is OPTIMAL**: best throughput AND best quality
- Theoretical per-projection sensitivity (w2>w1>w3) was WRONG empirically
- Protecting w3 (up) while quantizing w1 (gate) + w2 (down) gives best results
- **Deep analysis**: `docs/glm_47_reap/galaxy_wormhole/postmortem-bf4.md` (790 lines)

### bs=64 Analysis (2026-03-09) — NOT Beneficial

- bs=64 works functionally but produces 61.4 tok/s (WORSE than bs=32's 121.6)
- MoE EP=32 processes FULL batch (not DP-split) on all 32 devices
- bs=32 = 1 tile row = perfect amortization. bs=64 = 2 tile rows = doubles MoE cost
- Weight DRAM reads already well-amortized at bs=32 (only ~13% of ITL)
- **bs=32 is the architectural sweet spot**

### Session 3: SEED-4/5/6 + GLM-4.7 Bringup (2026-03-10 to 2026-03-11)

#### SEED-4: GLM-4.7 Base (358B) Bringup — COMPLETED
- `zai-org/GLM-4.7` — 92 layers, 160 experts (top-8), hidden=5120, EP=32 (5 experts/device)
- Config: all-BF4 experts, BF8 dense/attn/shared, KV=1024 BF16
- **Key fix**: KV cache block count — `get_num_available_blocks_tt()` defaulted to 131K tokens.
  Added GLM-4.7 match → 1024 tokens (48 blocks/layer = 138 MB KV vs 5.84 GB OOM).
- **Result**: bs=1: 4.1 tok/s, bs=32: 104.0 tok/s agg — 27% slower than REAP (5 vs 3 experts/device)

#### FP8 Dequant Bug Found (Affects GLM-4.7 + REAP FP8 Models)

Both FP8 models exist on HuggingFace:
- `cerebras/GLM-4.7-REAP-218B-A32B-FP8` (220 GB, E4M3 compressed-tensors)
- `zai-org/GLM-4.7-FP8` (362 GB, E4M3 compressed-tensors)

**BUG**: `_maybe_dequant_fp8()` in `layer_weights.py:73` inverts `_scale` keys, but
`_scale` in compressed-tensors convention is ALREADY the dequant multiplier. Both `_scale`
and `_scale_inv` should be passed directly to `dequantize_tensor()` without inversion.

```python
# BUGGY (current code):
inv_scale = (1.0 / state[scale_key].float().clamp(min=1e-12))

# CORRECT (1-line fix):
inv_scale = state[scale_key].float()
```

Proof: `compressed_tensors/quantization/lifecycle/forward.py:448` does `dequant = x_q * scale`,
and DSv3's `_scale_inv = 1/quant_scale` is also the dequant multiplier directly.

**Impact**: All FP8 weights loaded with inverted scales → garbage weights → garbled output.
Current WH Galaxy deployment uses BF16 model (`cerebras/GLM-4.7-REAP-218B-A32B`) and
bypasses all FP8 code, so NOT affected.

See `plan/glm47_reap/galaxy_wormhole/fp8-quant-analysis.md` for full analysis (Gemini+Codex verified).

#### GLM-4.7 Base (358B) vs REAP (268B) Comparison

| Model | Experts | Per-device | bs=1 | bs=32 agg |
|-------|---------|------------|------|-----------|
| GLM-4.7-REAP-218B | 96 routed (top-8) | 3 experts | ~5-6.7 tok/s | **92 tok/s** (TPOT-corrected) |
| GLM-4.7 Base (358B) | 160 routed (top-8) | 5 experts | **4.1 tok/s** | 104 tok/s (wall-time) |

REAP is faster at bs=1 because fewer experts = less DRAM weight reads per decode.
GLM-4.7 Base bs=32 appears higher because measured wall-time (not TPOT-corrected like REAP).

#### SEED-5: Fused Partial RoPE — CLOSED
- `ttnn.experimental.rotary_embedding_llama` does NOT support `partial_rotary_factor`
- No API parameter exists, kernel applies RoPE to ALL dimensions

#### SEED-6: Add-Before-Reduce CCL — REJECTED
- `FUSE_SHARED_EP_REDUCE=1` (the existing code IS SEED-6). rs_ag_async fixed the hang.
- **Garbled output**: "Paris. The capital of The capital is The capital of..."
- Correctness issue in fuse logic, not CCL deadlock

### Session 4: Trace Stability Fixes (2026-03-11)

Three trace stability fixes deployed:

1. **trans_matrix_tt deallocation bug**: `_release_all_decode_traces()` deallocated
   `self.rope["trans_matrix"]` (shared global). Fixed by removing from deallocation tuple.
2. **Batch bucket rounding**: Round batch UP to nearest bucket [1,4,8,16,32]. Pad inputs.
   Prevents trace recapture on every batch size change during continuous batching.
3. **Release before capture**: `_decode_trace()` calls `_release_all_decode_traces()` before
   capturing new trace. Ensures no active trace exists when allocating new buffers.

Also: embed_tt clone fix inside trace (decoder_layer_tt deallocates residual input),
prefill KV cache DP column fix (all 32 devices' caches filled, not just DP column 0).

### Session 5: Trace Crash Fix + TPOT Verification (2026-03-12)

**Crash root cause**: DRAM fragmentation on trace recapture. Specifically:
- bs=1→bs=32: **CRASH** (2/2 reproducible) — small trace release fragments DRAM
- bs=32→bs=1: OK — large→small allocation always succeeds
- First-trace capture (any size): OK — no prior fragmentation

**Fix**: `decode_trace_batch_buckets: [32]` — single bucket eliminates all trace recapture.
bs=1 requests pad to 32 (replicate last entry). Same trace reused for all batch sizes.
Trade-off: bs=1 TPOT increases from 150-200ms to ~350ms (padded to 32-wide trace).
**Verified**: bs=1→bs=32 sequence passes cleanly.

**TPOT verification**: Server-side TPOT histogram is definitive measurement.
Previous "142.7 tok/s" was wall-time aggregate (inflated). Actual steady-state:
- bs=32: 348ms TPOT = 92 tok/s aggregate (99% of samples in 300-400ms bucket)
- bs=1: 150-200ms TPOT (histogram bucket)

**Optimization research completed** (see `docs/glm_47_reap/galaxy_wormhole/research-next-optimizations.md`):
1. DRAM weight prefetching (19-23%, HIGH effort — only path for large gains)
2. Fused partial QK RoPE kernel (10%, HIGH effort)
3. Q/K DRAM round-trip elimination + fused KV cache (3-4%, LOW effort)

### Session 6: Quick Win Testing (2026-03-12)

#### P1: Q/K L1 — NEUTRAL (+0.26%, noise)
Changed `ttnn.to_memory_config(q/k, ttnn.DRAM_MEMORY_CONFIG)` → `ttnn.L1_MEMORY_CONFIG` at
attention_tt.py:678-679. Feature-flagged: `GLM4_MOE_QK_L1=1`.

A/B test with server-side TPOT histogram (3 runs each):
- **QK_L1=1**: 349.3ms TPOT (best steady-state)
- **QK_L1=0**: 348.4ms TPOT (best steady-state)
- **Delta: +0.9ms = +0.26%** — within measurement noise

Q=98KB + K=8KB per layer → ~9μs DRAM traffic saved. Invisible at 348ms. Program dispatch
elimination also invisible in trace mode (dispatches baked into trace).
Shipped as default (harmless). No performance impact.

#### P3 Workaround: cos/sin Padding — IMPOSSIBLE
Architect proposed padding cos=1.0, sin=0.0 for non-rotary dims to eliminate 8-10 ops/call
in `_apply_partial_rope_decode`. **DISPROVEN**: NeoX `rotate_half` splits at `dim // 2`.
Extending from rotary_dim=64 to head_dim=128 shifts split boundary from 32 to 64, mixing
pass-through dims into rotary computation. Only works with GPT-J interleaved rotation.

#### P1b: Fused KV Cache — SKIPPED
Expected ~0.4% (92 fewer dispatches). Given P1 showed 368 fewer dispatches = 0%,
P1b not worth the MEDIUM risk (core grid management). Skipped.

**Key lesson**: Software-only "quick wins" (dispatch elimination, small DRAM traffic savings)
have ZERO measurable impact on DRAM-BW-bound trace mode. Only DRAM weight prefetching
(overlapping weight reads with compute) can meaningfully reduce TPOT.

## Dead Ends (Do NOT Retry)

- **EP_L1=1** → garbled output (L1 incompatible with TG mesh CCL)
- **FUSE_SHARED_EP_REDUCE=1** → infinite hang, device corruption (CCL deadlock on 2D mesh)
- **LoFi math fidelity** → no speedup (DRAM-BW-bound, not compute-bound)
- **DRAM-sharded attn weights** → +2.1% not significant (per-device matmuls too small)
- **Device-side sampling** → +0% (host sampling hidden behind device trace)
- **Fused gate+up sparse matmul** → -14.6% (clone porting bug + break-even prediction)
- **bs=64** → works but 61.4 tok/s (WORSE, MoE EP=32 doubles cost)
- **DP specialization** → MoE EP=32 requires all 32 chips per layer
- **Async CCL overlap** → not real overlap in trace mode (single CQ)
- **FUSE_SHARED_EP_REDUCE=1** → Phase 1: hang. Phase 2: hang fixed by rs_ag_async but **garbled output**
- **SEED-6 (add-before-reduce)** → Same as FUSE_SHARED_EP_REDUCE. Garbled.
- **Sparse matmul grid optimization** → CRASH: non-full-width grid causes "Invalid subtile broadcast type"
- **Multi-bucket traces** → bs=1→bs=32 trace recapture crashes (DRAM fragmentation). Use single bucket [32].
- **Q/K L1 interleaved** → +0.26% (noise). Dispatch/DRAM savings invisible in trace mode.
- **cos/sin padding for partial RoPE** → IMPOSSIBLE (NeoX rotation shifts dim//2 boundary)
- **Fused KV cache** → skipped, expected <0.4% given P1 result

## Key Files

| File | Purpose |
|------|---------|
| `tt-metal/models/demos/glm4_moe/tt/attention_tt.py` | Attention + batch-bucketed init + `_simple_all_reduce` |
| `tt-metal/models/demos/glm4_moe/tt/decoder_layer_tt.py` | Decoder forward, EP reduce calls |
| `tt-metal/models/demos/glm4_moe/tt/moe_tt.py` | MoE routing, sparse matmul, L1 threshold, EP reduce |
| `tt-metal/models/demos/glm4_moe/tt/model_tt.py` | Model forward, trace capture/replay, DP batch sharding |
| `tt-metal/models/demos/glm4_moe/tt/generator_vllm.py` | vLLM interface, warmup, sampling |
| `tt-metal/models/demos/glm4_moe/tt/layer_weights.py` | Weight loading, per-projection dtype, BF4/BF8 support |
| `docker_tt/dev/.env.glm47_reap` | All env flags |
| `docker_tt/dev/docker-compose.yml` | Container definition, env passthrough |
| `docker_tt/dev/docker-compose.galaxy.yml` | Galaxy-specific overrides |
| `docker_tt/docs/glm_47_reap/galaxy_wormhole/` | Committed research reports |

## Next Phase (2026-03-12+)

### ~~Priority 1: Q/K L1 + Fused KV Cache~~ — TESTED, NEUTRAL/SKIPPED (Session 6)

### Priority 1 (REVISED): DRAM Weight Prefetching (HIGH effort, 19-23%)
- Port Llama3 Galaxy `ttnn.dram_prefetcher()` infrastructure
- Subdevice architecture (sender + worker cores), global circular buffer
- Phase 1: Prefetch attention + shared expert weights (~21.2 MB/layer/device, unconditional)
- Phase 2: Routed expert weights via sparse_matmul global_cb params
- **Only viable path for large gains** — DRAM-BW-bound at ~15% utilization

### Priority 2 (REVISED): Native Fused Partial QK RoPE C++ Kernel (HIGH effort, ~3%)
- cos/sin padding workaround IMPOSSIBLE (NeoX rotation, Session 6)
- Requires new C++ kernel with `rotary_dim` param + NeoX half-rotation support
- Replace 16 ops/layer with 2 (fused NeoX partial RoPE on Q+K)
- Saves ~1,288 programs/decode
- Requires new C++ kernel (ttnn.experimental.rotary_embedding_llama only does full-dim)

### Completed Seeds
- **SEED-0**: Batch>1 (rs_ag_async) — SHIPPED
- **SEED-1**: Batch-bucketed attention (+22%) — SHIPPED
- **SEED-3**: Selective BF4 (w1/w2=BF4, w3=BF8) — SHIPPED
- **SEED-4**: GLM-4.7 Base bringup — COMPLETED

### Closed Seeds
- **SEED-2**: Fused gate+up — REJECTED (-14.6%)
- **SEED-5**: Fused partial RoPE — CLOSED (no API support)
- **SEED-6**: Add-before-reduce CCL — REJECTED (garbled output)
