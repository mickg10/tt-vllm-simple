# GLM-4.7-REAP/355B Blackhole Galaxy — Execution Plan

**Created**: 2026-03-09
**Platform**: 32 Blackhole chips, Mesh(8,4), TP=4, DP=8, EP=32
**Models**: `zai-org/GLM-4.7-FP8` (355B, 160 experts) — primary; REAP-218B used for debugging only
**Workspace**: `ws/glm47_flash_blackhole_galaxy/` (shared with GLM-4.7-Flash)
**SSH**: `ssh -p 55212 mick@38.97.6.6`

---

## Architecture Reference

| Parameter | Value | Per Device (TP=4) |
|-----------|-------|-------------------|
| hidden_size | 5120 | 1280 |
| num_attention_heads (Q) | 96 | 24 |
| num_key_value_heads (KV) | 8 | 2 |
| head_dim | 128 | 128 |
| intermediate_size (dense) | 12288 | 3072 |
| moe_intermediate_size | 1536 | 1536 (EP-sharded, not TP-sharded) |
| n_routed_experts | 160 (full) / 96 (REAP) | 5 / 3 per device (EP=32) |
| num_experts_per_tok | 8 | 8 (top-8) |
| n_shared_experts | 1 | 1 (TP-sharded) |
| first_k_dense_replace | 3 | layers 0-2 use dense MLP |
| num_hidden_layers | 92 | 92 |
| partial_rotary_factor | 0.5 | RoPE on 64 of 128 dims |

**Key difference from WH Galaxy**: TP=8 (axis 0) becomes TP=4 (axis 1). DP=4 (axis 1) becomes DP=8 (axis 0).

---

## Parallelism Mapping (WH → BH)

| Parameter | WH Galaxy | BH Galaxy | Change |
|-----------|-----------|-----------|--------|
| TP | 8 (axis 0, rows) | 4 (axis 1, cols) | Swapped axis |
| DP | 4 (axis 1, cols) | 8 (axis 0, rows) | Swapped axis |
| EP | 32 (all devices) | 32 (all devices) | Same |
| Mesh | (8,4) | (8,4) | Same shape |
| Q heads/device | 12 | 24 | 2x |
| KV heads/device | 1 | 2 | 2x |
| Cores/device | 72 (8x9) | 130 (13x10) | 1.8x |
| DRAM/device | 12 GB | 32 GB | 2.7x |

---

## Execution Phases

### Phase 0: Pre-Flight
- Create BH workspace branch (`glm47_reap_blackhole` from `glm47_flash_blackhole`)
- Verify FP8 model on BH machine
- Rsync workflow: edit locally → rsync → restart container

**Status**: DONE

### Phase 1: Parallelism Adaptation (TP axis swap)
- `layer_weights.py`: `_tp_axis_and_size()` reads `GLM4_MOE_TP=4` env → returns (axis=1, size=4)
- `decoder_layer_tt.py`: Dynamic TP/DP axis for EP reduce
- `attention_tt.py`: `tp_size=4` in configuration, SDPA grid, core grid, reduce axes
- `moe_tt.py`: Dispatch axis follows TP axis
- `generator_vllm.py`: `_get_tp_size()` delegates to `_tp_axis_and_size()`
- `model_tt.py`: `configuration["tp_size"]` and `configuration["tp_axis"]`

**Status**: DONE

### Phase 2: FP8 Weight Loading
- FP8 E4M3 → BF16 (CPU dequant with per-row scale) → BF8 (device)
- Handle both `_scale` and `_scale_inv` key conventions
- Per-row `[R, 1]` scale format (not block-wise like DSv3)

**Status**: DONE (fix verified with coherent output)

### Phase 3: Attention Adaptation (GQA at TP=4)
- 96Q / 4 = 24 local heads (integer, OK)
- 8KV / 4 = 2 local KV heads (integer, OK)
- Partial RoPE (factor=0.5) unchanged
- QK norm (per-head, replicated) unchanged

**Status**: DONE

### Phase 4: MoE Adaptation
- 160 experts / 32 = 5 per device (clean division)
- Sparse matmul, routing, shared expert all dynamic on expert count
- Router gate weight: [160, 5120] loaded from config

**Status**: DONE

### Phase 5: Env File + Docker Configuration
- `.env.glm47_reap_blackhole`: TP=4, EP=32, trace_region_size=3GB
- `docker-compose.blackhole.yml`: GLM4_MOE_* env passthrough
- `docker-compose.galaxy.yml`: GLM4_MOE_* env passthrough + volume mounts

**Status**: DONE

### Phase 6: Trace Mode + Host Sampling
- Trace region: 3 GB (steady state ~2.15 GB, 40% headroom)
- Host-side sampling (device mesh ops hang on TG — may differ on BH)

**Status**: IN PROGRESS (trace region fixed, awaiting confirmation)

### Phase 7: Performance Optimization
- Grid fix: 12-13/130 cores → maximize full-row usage (P0)
- LoFi fidelity testing
- L1 activations testing
- Batch>1 testing

**Status**: NOT STARTED

---

## Code Changes Summary

6 Python files in `models/demos/glm4_moe/tt/`:

| File | Lines Changed | Purpose |
|------|--------------|---------|
| `layer_weights.py` | ~35 | TP axis env, FP8 dequant fix, cache key with expert count, timing logs |
| `model_tt.py` | ~85 | Program cache, compile/trace logging, sem reset, mesh-safe debug, env fixes |
| `generator_vllm.py` | ~5 | Model name in cache path |
| `ccl.py` | ~25 | `reset_global_semaphores()` method + documentation |
| `attention_tt.py` | ~10 | `noop` reduce impl for trace debugging |
| `moe_tt.py` | ~10 | `reduce_scatter` API update, dispatch axis |

Plus env/compose files in `docker_tt/dev/`.

---

## Rsync Workflow

```bash
# Model code:
rsync -avz --exclude='__pycache__' \
  /home/ttuser/src_docker/ws/glm47_flash_blackhole_galaxy/tt-metal/models/demos/glm4_moe/ \
  mick@38.97.6.6:/home/mick/ws/glm47_flash_blackhole_galaxy/tt-metal/models/demos/glm4_moe/ \
  -e "ssh -p 55212"

# Docker config:
rsync -avz --exclude='.git' --exclude='__pycache__' \
  /home/ttuser/src_docker/ws/glm47_flash_blackhole_galaxy/docker_tt/ \
  mick@38.97.6.6:/home/mick/ws/glm47_flash_blackhole_galaxy/docker_tt/ \
  -e "ssh -p 55212"
```

---

## Dependencies

```
Phase 0 (Pre-flight) → Phase 1 (Parallelism, CRITICAL)
                        ├→ Phase 2 (FP8, parallel)
                        └→ Phase 3 (Attention) → Phase 4 (MoE) → Phase 5 (Env)
                                                                   → Phase 6 (Trace)
                                                                     → Phase 7 (Perf)
```
