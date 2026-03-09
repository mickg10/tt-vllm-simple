# GLM-4.7-REAP-218B on Galaxy Wormhole: Worklog

Canonical planning/docs live outside git at `/home/ttuser/src_docker/plan/glm47_reap/galaxy_wormhole/`.

## Run Command

```bash
cd /home/user/src_docker/ws/glm47_reap_268b_galaxy_wormhole/docker_tt
sg docker -c 'docker compose --env-file dev/.env.glm47_reap \
  -f dev/docker-compose.yml -f dev/docker-compose.galaxy.yml \
  up -d vllm-tt'
```

Use `--force-recreate` when env vars change (triggers weight reload, ~10-20 min).

**Galaxy SSH**: `ssh -p 55211 user@38.97.6.6`

## Model

- **cerebras/GLM-4.7-REAP-218B-A32B** (268B total, 32B active)
- 92 MoE layers, 96 routed experts (top-8), GQA 96Q/8KV
- hidden_size=5120, num_attention_heads=96, num_kv_heads=8

## Hardware & Parallelism

- Galaxy Wormhole: 32 chips, Mesh(8,4), 8x9=72 cores/chip, 12 DRAM channels/chip
- TP=8 (axis 0), EP=32 (all chips), DP=4 (axis 1)
- MESH_DEVICE=TG, FABRIC_2D

## Current Status (2026-03-09) — 99.5 tok/s aggregate

| Batch | Per-user tok/s | Aggregate tok/s | ITL (ms) | Scaling |
|-------|---------------|-----------------|----------|---------|
| bs=1  | 4.1           | 4.1             | 188      | 1.0x    |
| bs=4  | 3.2           | 12.7            | 241      | 3.7x    |
| bs=8  | 3.1           | 24.8            | 235      | 7.3x    |
| bs=16 | 2.9           | 46.6            | 228      | 13.7x   |
| bs=32 | 3.1           | 99.5            | 187      | 28.4x   |

**Tile sweet spot**: bs=32 ITL = bs=1 ITL (both ~188ms). 32 rows = 1 tile height,
zero per-user overhead. Near-perfect scaling from bs=16 to bs=32.

## Configuration

```
GLM4_MOE_DENSE_TT_DTYPE=bf8                # +18.8% at bs=32 vs bf16
GLM4_MOE_EXPERTS_TT_DTYPE=bf8
GLM4_MOE_REDUCE_IMPL=native                # axis-0 (TP) all_reduce
GLM4_MOE_REDUCE_IMPL_AXIS1=rs_ag_async     # axis-1 (DP) — bypasses FABRIC_2D crash
GLM4_MOE_EP_REDUCE_DEVICE=1
GLM4_MOE_FUSE_SHARED_EP_REDUCE=0           # BROKEN: hang on TG
GLM4_MOE_EP_L1=0                           # BROKEN: garbled on TG
GLM4_MOE_DRAM_SHARD=0                      # +2.1% (not significant)
GLM4_MOE_ATTN_FIDELITY=hifi
GLM4_MOE_MOE_SPARSE_FIDELITY=hifi
MAX_NUM_SEQS=32
decode_trace_batch_buckets=[1,4,8,16,32]
trace_mode=decode_only
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

## Dead Ends (Do NOT Retry)

- **EP_L1=1** → garbled output (L1 incompatible with TG mesh CCL)
- **FUSE_SHARED_EP_REDUCE=1** → infinite hang, device corruption (CCL deadlock on 2D mesh)
- **LoFi math fidelity** → no speedup (DRAM-BW-bound, not compute-bound)
- **BF4** → garbled output (error accumulation over 92 layers)
- **DRAM-sharded attn weights** → +2.1% not significant (per-device matmuls too small)
- **Device-side sampling** → +0% (host sampling hidden behind device trace)

## Key Files

| File | Purpose |
|------|---------|
| `tt-metal/models/demos/glm4_moe/tt/attention_tt.py` | Attention + `_simple_all_reduce` |
| `tt-metal/models/demos/glm4_moe/tt/decoder_layer_tt.py` | Decoder forward, EP reduce calls |
| `tt-metal/models/demos/glm4_moe/tt/moe_tt.py` | MoE routing, sparse matmul, EP reduce |
| `tt-metal/models/demos/glm4_moe/tt/model_tt.py` | Model forward, trace capture/replay |
| `tt-metal/models/demos/glm4_moe/tt/generator_vllm.py` | vLLM interface, warmup, sampling |
| `tt-metal/models/demos/glm4_moe/tt/layer_weights.py` | Weight loading and conversion |
| `docker_tt/dev/.env.glm47_reap` | All env flags |
| `docker_tt/dev/docker-compose.yml` | Container definition, env passthrough |
| `docker_tt/dev/docker-compose.galaxy.yml` | Galaxy-specific overrides |

## Remaining Opportunities

1. **Router L1 (V2 P4)**: Code exists in moe_tt.py. Never tested in isolation. Est: 1-5ms.
2. **Fused gate+up sparse matmul (V2 P5)**: Concat w1+w3 for single sparse_matmul. Est: 3-8ms.
3. **BF8 + higher batch (bs=64?)**: May need multi-tile batch handling beyond 32.
4. **Dedicated trace region**: Currently re-capturing trace each request. Persistent trace
   region would eliminate recapture overhead.
5. **CCL library fix**: Proper fix for `all_reduce(cluster_axis=1)` on FABRIC_2D to eliminate
   the rs_ag workaround. See `plan/glm47_reap/galaxy_wormhole/batch-gt1-implementation-plan.md`.
