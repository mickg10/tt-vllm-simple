# GLM-4.7-REAP-218B Galaxy Wormhole — Resume State

**Date**: 2026-03-10 (BF4 confirmed viable)
**Model**: cerebras/GLM-4.7-REAP-218B-A32B — 92 layers, 96 routed experts (top-8), GQA 96Q/8KV
**Hardware**: Galaxy Wormhole, 32 chips, MESH_DEVICE=TG, Mesh(8,4), TP=8, EP=32, DP=4
**Current**: bs=1: **4.7 tok/s** (177ms ITL), bs=32: **126.4 tok/s agg** (175ms ITL) — **BF4 experts**, BF8 shared+attn, MAX_NUM_SEQS=32
**Container**: RUNNING and HEALTHY — BF4 routed experts, BF8 dense, rs_ag_async axis-1, batch 1..32

---

## V2 Phase Results

| Optimization | Result | Notes |
|---|---|---|
| V2 P1: BF8 dense weights | **+18.8% at bs=32** (99.5 vs 83.8 agg) | +2.4% at bs=1, compounds at batch |
| V2 P2: Device-side sampling (topk) | **+0%** | Host sampling fully hidden behind device trace |
| V2 P3: DRAM-sharded attn weights | **+2.1%** (5.32 tok/s, 188ms ITL) | Not significant — within noise |
| V2 P6: Batch>1 rs_ag unblock | **83.8 agg bs=32** (2.6/user, 191ms=bs=1!) | 3 reshape fixes + rs_ag_async axis-1 |

### V2 P3 DRAM-Sharded Weights — Details
- Implemented `_setup_dram_sharded_configs()` in attention_tt.py with DSv3 helpers
- Keeps both interleaved (prefill) and DRAM-sharded (decode) weight copies
- QKV: K=5120 N=1792, in=40 out=56 cores; O: K=1536 N=5120, in=48 out=40 cores
- Code works (prefill + trace capture + decode all pass) but no meaningful speedup
- Root cause: per-device matmul sizes too small for DRAM bank parallelism to help at bs=1
- Consistent with glm4_moe_lite finding: "DRAM-sharded matmuls are a dead end"
- **DRAM_SHARD=0 kept as baseline** (code left in place but disabled)

## Phase 1 Results (FAILED)

| Optimization | Result | Root Cause |
|---|---|---|
| EP_L1=1 | GARBLED output | L1 memory config incompatible with TG mesh CCL |
| FUSE_SHARED_EP_REDUCE=1 | INFINITE HANG (device corruption) | CCL deadlock on 2D mesh |
| LoFi (attn + experts) | -1.5% SLOWER | **DRAM-BW-bound**, not compute-bound |

**CRITICAL FINDING**: REAP-218B at bs=1 is DRAM-bandwidth-bound. ~15% DRAM utilization.
But improving DRAM utilization via weight sharding doesn't help because per-device matmuls
are too small after TP=8 sharding. The bottleneck is diffuse (spread across many small ops).

---

## Current Config

```
GLM4_MOE_REDUCE_IMPL=native                # axis-0 (TP) reduce
GLM4_MOE_REDUCE_IMPL_AXIS1=rs_ag_async     # axis-1 (DP) reduce — bypasses FABRIC_2D crash
GLM4_MOE_EP_REDUCE_DEVICE=1
GLM4_MOE_FUSE_SHARED_EP_REDUCE=0           # BROKEN: hang
GLM4_MOE_EP_L1=0                           # BROKEN: garbled
GLM4_MOE_DRAM_SHARD=0                      # TESTED: +2.1% (not significant)
GLM4_MOE_DENSE_TT_DTYPE=bf8                # +18.8% at bs=32
GLM4_MOE_EXPERTS_TT_DTYPE=bf8
GLM4_MOE_ATTN_FIDELITY=hifi                # LoFi tested: no speedup
GLM4_MOE_MOE_SPARSE_FIDELITY=hifi          # LoFi tested: no speedup
MAX_NUM_SEQS=32
decode_trace_batch_buckets=[1,4,8,16,32]
trace_mode=decode_only
```

---

## Session 2 Results (2026-03-09 — SEED-1/SEED-2 sprint)

### SEED-1: bs=64 Analysis + Batch-Bucketed Attention
- **Result**: bs=64 WORKS but architecturally limited (61.4 tok/s — WORSE than bs=32's 121.6)
- **Root cause**: MoE EP=32 processes full batch on all devices (not DP-split). bs=32 = 1 tile row optimal.
- **Side benefit**: Batch-bucketed attention init + L1 threshold fix → **+22% at bs=32** (99.5 → 121.6 tok/s)
- **Changes**: attention_tt.py (batch-bucketed `_slice_mats`/`_user_sel_mats` dicts, B_phys fixes),
  moe_tt.py (L1 threshold <=64)

### SEED-2: Fused Gate+Up Sparse Matmul
- **Result**: **-14.6% regression** at bs=1 — REJECTED
- **Root cause**: Unnecessary `ttnn.clone` calls (porting bug from glm4_moe_lite) added 267 extra trace programs
- Even with clones removed, predicted break-even (slices offset dispatch savings)
- **Code kept but disabled**: `GLM4_MOE_FUSE_EXPERTS_GATE_UP=0`

### Updated Benchmark (MAX_NUM_SEQS=32, gen=50)

| Batch | Per-user | Aggregate | ITL | vs Prior |
|-------|----------|-----------|-----|----------|
| bs=1  | 4.1      | 4.1       | 185ms | +17% (was 3.5) |
| bs=32 | 3.8      | **121.6** | 191ms | **+22%** (was 99.5) |
| bs=64 | 1.0      | 61.4      | ~320ms | WORSE (MoE doubles) |

---

## Next Steps — ACTIVE

### SEED-3: Selective BF4 (DSv3 Pattern — Experts Only)
**Status**: RESEARCH COMPLETE — needs implementation + quality validation
**Expected**: +23-35% aggregate if quality holds
**Design**: w1/w2 BF4, w3 BF8, attention/shared BF8
**Gap**: `_env_experts_dtype()` has no BF4 option; all 3 projections share `experts_dtype`.
Need per-projection env vars and dtype parser update.
See `research-seed.md` SEED-3 for full details.

### Code Changes On Galaxy (uncommitted)
- `attention_tt.py` — Batch-bucketed init (`_tg_init_batch_mats`, `_get_shard_cfgs`),
  B_phys fixes at lines 616/639/711/729, DRAM-sharded decode path (DRAM_SHARD=0)
- `layer_weights.py` — SEED-2 w1w3 weight loading (disabled FUSE=0)
- `moe_tt.py` — L1 threshold <=64, SEED-2 fused code (disabled FUSE=0)
- `model_tt.py` — Device-side topk sampling (always active)
- `docker-compose.yml` — FUSE env passthrough + prior passthrough vars
- `.env.glm47_reap` — MAX_NUM_SEQS=32, GLM4_MOE_FUSE_EXPERTS_GATE_UP=0
- `vllm/` — 3 files with try/except for sagemaker import fix

---

## Key Files
- Env: `docker_tt/dev/.env.glm47_reap`
- Model: `tt-metal/models/demos/glm4_moe/tt/model_tt.py`
- Decoder: `tt-metal/models/demos/glm4_moe/tt/decoder_layer_tt.py`
- MoE: `tt-metal/models/demos/glm4_moe/tt/moe_tt.py`
- Attention: `tt-metal/models/demos/glm4_moe/tt/attention_tt.py`
- Generator: `tt-metal/models/demos/glm4_moe/tt/generator_vllm.py`

### Workspace
- Local: `/home/ttuser/src_docker/ws/glm47_reap_268b_galaxy_wormhole/`
- Galaxy SSH: `ssh -p 55211 user@38.97.6.6`
- Galaxy code: `/home/user/src_docker/ws/glm47_reap_268b_galaxy_wormhole/`
- Start: `sg docker -c 'docker compose --env-file dev/.env.glm47_reap -f dev/docker-compose.yml -f dev/docker-compose.galaxy.yml up -d vllm-tt'`

### Research Docs
| File | Content |
|------|---------|
| `batch-gt1-implementation-plan.md` | **ACTIVE** — Batch>1 unblock via rs_ag |
| `execution-plan-v2.md` | DRAM-BW-focused plan with revised priorities |
| `execution-plan-v1.md` | Original plan (Phase 1 items FAILED) |

### Dead Ends (Do NOT Retry)
- EP_L1=1 → garbled output
- FUSE_SHARED_EP_REDUCE=1 → infinite hang, device corruption
- LoFi math fidelity → no speedup (DRAM-BW-bound)
- Blanket BF4 → garbled output (error accumulation over 92 layers)
- DRAM-sharded attn weights → +2.1% (not significant, per-device matmuls too small)
- Device-side sampling → +0% (host sampling hidden behind device trace)
- Fused gate+up sparse matmul → -14.6% (clone porting bug + break-even prediction)
- bs=64 → works but 61.4 tok/s (WORSE, MoE EP=32 doubles cost with no savings)
- See `research-seed.md` Rejected section for full dead-end list
