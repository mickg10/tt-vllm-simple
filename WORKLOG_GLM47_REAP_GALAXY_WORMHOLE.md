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

## Current Status (2026-03-10) — 126.4 tok/s aggregate, BF4 experts

| Batch | Per-user tok/s | Aggregate tok/s | ITL (ms) | Config |
|-------|---------------|-----------------|----------|--------|
| bs=1  | 4.7           | 4.7             | 177      | BF4 experts, BF8 dense |
| bs=32 | ~4.0          | 126.4           | 175      | BF4 experts, BF8 dense |

**Previous BF8 baseline** (for comparison):

| Batch | Per-user tok/s | Aggregate tok/s | ITL (ms) | Config |
|-------|---------------|-----------------|----------|--------|
| bs=1  | 4.1           | 4.1             | 185      | BF8 all |
| bs=32 | 3.8           | 121.6           | 191      | BF8 all |

**BF4 improvement**: +6.8% bs=1, **+27.8% bs=32**, PPL 5.7% better (1.29 vs 1.37).

## Configuration

```
GLM4_MOE_DENSE_TT_DTYPE=bf8                # Shared expert + attention stays BF8
GLM4_MOE_EXPERTS_TT_DTYPE=bf4              # ALL routed experts BF4 (+28% bs=32)
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
decode_trace_batch_buckets=[1,4,8,16,32]
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
126.4 tok/s → BF4 routed experts (+28% over prior BF8 baseline)
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
- **Re-test (2026-03-10)**: All-expert BF4 (w1/w2/w3 all BFP4_b), shared+attention BF8:
  - PPL: 1.2884 vs BF8 1.3659 — **5.7% BETTER**
  - bs=32: **+27.8%** throughput (126.4 vs 98.9 tok/s)
  - bs=1: **+6.8%** throughput (4.7 vs 4.4 tok/s)
  - Output: coherent, same quality as BF8
- **Deep analysis**: `docs/glm_47_reap/galaxy_wormhole/postmortem-bf4.md` (790 lines)
  covering BFP4_b format, per-projection sensitivity, Hadamard transforms, weight permutation,
  DSv3 comparison, scaler granularity, and full re-test execution plan.

### bs=64 Analysis (2026-03-09) — NOT Beneficial

- bs=64 works functionally but produces 61.4 tok/s (WORSE than bs=32's 121.6)
- MoE EP=32 processes FULL batch (not DP-split) on all 32 devices
- bs=32 = 1 tile row = perfect amortization. bs=64 = 2 tile rows = doubles MoE cost
- Weight DRAM reads already well-amortized at bs=32 (only ~13% of ITL)
- **bs=32 is the architectural sweet spot**

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
- **Weight prefetching** → no Wormhole support (Blackhole-only)

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

## Remaining Opportunities

1. **Blanket BF4 test**: All weights BF4 (experts + shared + attention). May give additional throughput.
2. **CCL library fix**: Proper fix for `all_reduce(cluster_axis=1)` on FABRIC_2D to eliminate rs_ag workaround.
3. **Dedicated trace region**: Currently re-capturing trace each request. Persistent trace region would eliminate recapture overhead.
4. **Router L1 (V2 P4)**: Code exists in moe_tt.py. Never tested in isolation. Est: 1-5ms.
5. **FP8 checkpoint source**: Test BF16→FP8→BFP4 conversion path for potentially even better quality.
