# GLM-4.7-355B on Blackhole Galaxy: Worklog

Canonical planning/docs live outside git at `/home/ttuser/src_docker/plan/glm47_reap_268b/blackhole_galaxy/`.
Committed research reports at `docs/glm_47_reap/galaxy_blackhole/`.

## Run Command

```bash
cd /home/mick/ws/glm47_flash_blackhole_galaxy/docker_tt
docker compose -p reap-bh --env-file dev/.env.glm47_reap_blackhole \
  -f dev/docker-compose.yml -f dev/docker-compose.galaxy.yml \
  -f dev/docker-compose.blackhole.yml \
  up -d vllm-tt
```

**Logs**:
```bash
docker compose -p reap-bh --env-file dev/.env.glm47_reap_blackhole \
  -f dev/docker-compose.yml -f dev/docker-compose.galaxy.yml \
  -f dev/docker-compose.blackhole.yml \
  logs -f vllm-tt
```

**Benchmark**:
```bash
python tests/bench_decode.py --url http://localhost:8088 --gen-tokens 50 \
  --only-batch 1 --skip-combined --prefill-contexts 0
```

**SSH**: `ssh -p 55212 mick@38.97.6.6`
**IPMI recovery**: `sudo ipmitool chassis power cycle` (only reliable device recovery after crashes)

## Model

- **zai-org/GLM-4.7-FP8** (355B total, 32B active per token) — Full model, NOT REAP
- 92 MoE layers (3 dense + 89 MoE), 160 routed experts (top-8), GQA 96Q/8KV
- hidden_size=5120, moe_intermediate_size=1536, head_dim=128
- FP8 weights (E4M3 compressed-tensors), 362 GB total
- **REAP-218B is BANNED on BH Galaxy** — full model preferred for quality, both fit in 32GB/device

## Hardware & Parallelism

- Blackhole Galaxy: 32 chips, Mesh(8,4), 13x10=130 cores/chip, 8 DRAM channels (32 GB GDDR7/chip)
- TP=4 (axis 1, columns), DP=8 (axis 0, rows), EP=32 (all chips)
- 5 experts/device (160/32), 24 Q heads/device, 2 KV heads/device
- Weight memory: 14.75 GB/device (46% of 32 GB DRAM)

## Current Status (2026-03-13) — Coherent Output, Pending Traced Benchmark

| Milestone | Status | Notes |
|-----------|--------|-------|
| 32 BH devices initialized | DONE | Mesh (8,4), FABRIC_1D |
| Weight loading from cache | DONE | ~17 min (vs 130 min cold build, ~11s/MoE layer) |
| Program cache | DONE | 1s compile warm-up (vs 3+ hours without) |
| Coherent output (eager) | DONE | "Paris. The capital of the United Kingdom is London..." |
| Trace mode (decode_only) | PENDING | Needed 3 GB trace region (had 50 MB), fixed in .env |
| Benchmark (bs=1) | PENDING | After trace mode confirmed |

## Configuration

```
HF_MODEL=/home/mick/models/GLM-4.7-FP8
MESH_DEVICE=(8,4)
ARCH_NAME=blackhole
GLM4_MOE_TP=4
GLM4_MOE_EP=32
GLM4_MOE_DENSE_TT_DTYPE=bf16
GLM4_MOE_EXPERTS_TT_DTYPE=bf8
GLM4_MOE_KV_CACHE_TT_DTYPE=bf16
GLM4_MOE_REDUCE_IMPL=rs_ag_async
GLM4_MOE_REDUCE_IMPL_AXIS1=rs_ag_async
trace_mode=decode_only
trace_region_size=3000000000 (3 GB)
```

## Performance Projections

| Scenario | tok/s (bs=1) | Notes |
|----------|-------------|-------|
| Conservative (TP=4 only) | ~5.0-5.1 | ~8ms penalty vs REAP from 67% more expert weights |
| With grid fix | ~5.9-6.5 | BH uses 12-13/130 cores (9-10%), fix gives 15-25% improvement |
| Theoretical ceiling | ~19 | BW-limited with DP replication |

## Memory Budget (per device)

| Component | Size |
|-----------|------|
| Expert weights (BF8, 89 layers, 5/device) | 10.50 GB |
| Attention weights (BF16/BF8, 92 layers) | 3.35 GB |
| Other weights (shared exp, norms, embed) | 0.90 GB |
| **Total weights** | **14.75 GB** |
| Runtime overhead | ~1.5-2.0 GB |
| **DRAM remaining for KV** | **~15.25 GB** |

## Key Bugs Found and Fixed (Sessions 11-13)

### Session 11: Trace + CCL Bug (REAP-218B)
- `reset_global_semaphores()` must NOT be called in `reset_sem_counters()` — causes async CCL desync during trace replay
- `DEBUG_TRACE_VERIFY=0` required in trace mode — debug checks incompatible with trace replay
- **Result**: REAP-218B 5.9 tok/s (151ms ITL) at bs=1 with trace + rs_ag_async

### Session 12: 355B-Specific Fixes
1. **Cache key collision**: `ep{N}_v1` hit REAP cache when loading 355B (different expert count). Fixed: `ep{N}_e{E}_v1`
2. **Program cache missing**: `enable_program_cache()` not called → 3+ hour compile. Fixed in `Glm4MoeTT.create()`
3. **Cross-model cache contamination**: Cache path didn't include model name. Fixed: `Path(model_id).name`
4. **Reduce scatter API**: `ttnn.experimental.reduce_scatter_minimal_async` → `ttnn.reduce_scatter`
5. **Env var parsing**: Empty string `""` treated as non-zero by `int()`. Fixed: `int(x or "0")`
6. **Debug readback crashes**: `get_device_tensors()` fails on some mesh implementations. Added try/except fallback.

### Session 13: Trace Region Too Small
- 355B trace needs ~2.5 GB but only 50 MB was configured
- `TT_FATAL: Creating trace buffers of size 2510405632B but only 50000000B allocated`
- **Fix**: `trace_region_size=3000000000` (3 GB) — 40% headroom over 2.15 GB steady state

## Key Code Changes (tt-metal)

| File | Changes |
|------|---------|
| `layer_weights.py` | Cache key with expert count, tilization timing logs, TP env fix |
| `model_tt.py` | Program cache, compile/trace logging, sem reset, mesh-safe debug, env fixes |
| `generator_vllm.py` | Model name in cache path |
| `ccl.py` | `reset_global_semaphores()` + documentation |
| `attention_tt.py` | `noop` reduce impl for trace debugging |
| `moe_tt.py` | `reduce_scatter` API update |

## Next Steps

1. **Confirm trace mode** with 3 GB trace region (bh54 container)
2. **Run definitive 355B benchmark** (bench_decode.py bs=1)
3. **Grid fix** — `_make_sparse_matmul_program_config` uses 12-13/130 cores on BH (9-10%). Fix could give 15-25% improvement.
4. **Investigate first-token garble** — prefill→trace transition
5. **Performance optimization** — lofi fidelity, L1 activations, batch>1

## Key SHAs

| Repo | Branch | Notes |
|------|--------|-------|
| tt-metal | `glm47_reap_blackhole` | Model code changes |
| docker_tt | `glm47_flash_blackhole` | Compose + env (shared with Flash BH) |

## Weight Loading Pipeline

```
Safetensors (FP8 E4M3) → torch.bfloat16 (CPU dequant with per-row scale) → ttnn.bfloat8_b (device)
```

- Per-row scales: `{key}_scale` shape `[R, 1]` (not `_scale_inv`, not block-wise)
- FP8 dequant fix applied in layer_weights.py (handles both `_scale` and `_scale_inv` conventions)
- Cold build: ~130 min (68 min tilization + device transfer)
- Warm cache: ~17 min (11s/layer from cached tensors)

## REAP-218B Comparison (WH Galaxy Baseline)

| Metric | 355B (BH, projected) | REAP-218B (WH, measured) |
|--------|---------------------|--------------------------|
| Experts | 160 (5/device) | 96 (3/device) |
| Weights/device | 14.75 GB | 10.50 GB |
| bs=1 tok/s | ~5.0-5.1 (projected) | 5.9 (trace + rs_ag_async) |
| Hardware | BH Galaxy (32 GB GDDR7) | WH Galaxy (12 GB HBM) |

## References

- Execution plan: `docs/glm_47_reap/galaxy_blackhole/execution-plan.md`
- Full 355B analysis: `docs/glm_47_reap/galaxy_blackhole/full-355b-analysis.md`
- Trace & perf research: `docs/glm_47_reap/galaxy_blackhole/trace-and-perf-research.md`
- Planning docs (outside git): `plan/glm47_reap_268b/blackhole_galaxy/`
