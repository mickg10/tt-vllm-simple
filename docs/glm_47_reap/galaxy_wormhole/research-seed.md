# Research Seed: GLM-4.7-REAP-218B on Galaxy Wormhole

**Purpose**: Confirmed-positive optimization opportunities ready for implementation.
Only add entries here when research achieves **complete confidence** in a positive result.
Each entry must have: root cause understood, fix designed, expected impact quantified, risk assessed.

**Referenced by**: `plan/glm47_flash/_global/team-structure.md` (mandatory reading for all team members)

---

## Active Seeds (Ready for Implementation)

### ~~SEED-1: Batch=64~~ — COMPLETED (bs=64 works but NOT beneficial)

**Status**: COMPLETED — bs=64 functional but architecturally limited by EP=32 tile sweet spot
**Actual Impact**: bs=32 is optimal. bs=64 = 61.4 tok/s (WORSE than 121.6 at bs=32)
**Side benefit**: Batch-bucketed attention init + L1 threshold fix → **+22% at bs=32** (99.5 → 121.6 tok/s)

**Implementation (2026-03-09)**:
1. `moe_tt.py`: L1 fast path threshold raised from `<=32` to `<=64` — router uses L1 for up to 64 tokens
2. `attention_tt.py`: Full batch-bucketed init — `_slice_mats`, `_user_sel_mats`, shard configs
   pre-computed per batch bucket with lazy creation via `_tg_init_batch_mats()`.
   Runtime forward selects correct pre-computed tensors by `active_batch`.
3. `attention_tt.py` lines 616/639/711/729: `B_phys` calculations fixed to use `active_batch`
   and `logical_batch_after_slice` instead of `self.max_batch_size`

**Why bs=64 is NOT beneficial**:
- MoE EP=32 processes FULL batch (not DP-split) on all 32 devices
- bs=32 = 1 tile row = perfect amortization with ~same ITL as bs=1
- bs=64 = 2 tile rows → doubles MoE compute, activation DRAM, CCL traffic
- Weight DRAM reads already well-amortized at bs=32 (only ~13% of ITL)
- Doubling batch gives NO weight read savings but doubles everything else
- **Benchmark**: bs=64 = 61.4 tok/s aggregate (HALF of bs=32's 121.6 tok/s)

**Final benchmark (MAX_NUM_SEQS=32, 2026-03-09)**:
- bs=1: **4.1 tok/s** (185ms ITL) — was 3.5 tok/s (+17%)
- bs=32: **121.6 tok/s aggregate** (191ms ITL) — was 99.5 tok/s (+22%)

**Code kept**: All batch-bucketed attention init code is kept (contributes to the +22% at bs=32).
MAX_NUM_SEQS stays at 32.

**Research sources**: `seed1-bs64-analysis.md` (architect, 2026-03-09),
`research-creative-optimizations.md` (creative researcher, 2026-03-09),
C++ kernel analysis (team lead, 2026-03-09).

---

### ~~SEED-2: Fused Gate+Up Sparse Matmul~~ — MOVED TO REJECTED

**See Rejected section below.** Implemented and benchmarked 2026-03-09: -14.6% regression at bs=1.
The slice+clone to split fused output costs ~13ms, outweighing ~1.3ms dispatch savings.

---

### SEED-3: Selective BF4 (w1/w2=BF4, w3=BF8) — CONFIRMED OPTIMAL ✅

**Status**: CONFIRMED OPTIMAL — Re-tested 2026-03-10 with 4 configs, rigorous PPL + throughput.
Davor showed BFP4_b is nearly lossless on DSv3/R1 (256 experts × 58 layers, -0.15pp MMLU).

**Original test (2026-03-10 morning)**: Selective BF4 appeared garbled.
Flawed test — eyeball-only, no PPL, potential stale trace/cache artifact.

**Comprehensive re-test (2026-03-10, 4 configs)**:

| Test | Config | bs=32 agg | PPL | vs BF8 |
|------|--------|-----------|-----|--------|
| 0 | All BF8 (baseline) | 98.9 | 1.3659 | — |
| **1 (SEED-3)** | **w1/w2=BF4, w3=BF8** | **142.7** | **1.2288** | **+44%, -10% PPL** |
| 2 | All BF4 (w1/w2/w3) | 126.4 | 1.2884 | +28%, -5.7% PPL |
| 3 | w1=BF4, w2=BF8, w3=BF4 | 128.3 | 1.4007 | +30%, +2.5% PPL (WORSE) |

**SEED-3 is OPTIMAL**: best throughput AND best quality of all tested configs.
Theoretical per-projection sensitivity (w2>w1>w3) was WRONG empirically.
Protecting w3 (up projection, BF8) while quantizing w1+w2 (gate+down, BF4) is the sweet spot.

**Config**:
```
GLM4_MOE_EXPERTS_W1_DTYPE=bf4    # gate
GLM4_MOE_EXPERTS_W2_DTYPE=bf4    # down
GLM4_MOE_EXPERTS_W3_DTYPE=bf8    # up (protected)
GLM4_MOE_DENSE_TT_DTYPE=bf8     # shared + attention
```
**Research sources**: Davor (DSv3/R1 BFP4 benchmarks), postmortem-bf4.md (execution plan + analysis)

---

### SEED-4: GLM-4.7 Base (358B) Bringup on Galaxy Wormhole — READY

**Status**: READY — zero code changes needed, FP8 dequant is ~50 LOC
**Expected**: ~174 tok/s agg bs=32, ~5.5 tok/s bs=1 (~135ms ITL) with SEED-3 BF4
**Research source**: `optimization-analysis-v1.md` (researcher, 2026-03-10, Codex-verified)

**Why it's faster than REAP**: 64 layers vs 92 (-30%), 160 experts vs 96 (+67%).
30% fewer layers dominates: net ~18% fewer total ops. Memory: 6.53 GB/device (20.4% of 32GB) — fits.

**Architecture (identical to REAP except layer/expert count)**:
- `zai-org/GLM-4.7` (BF16) or `zai-org/GLM-4.7-FP8` (FP8)
- 64 layers, 160 routed experts (top-8), 1 shared, hidden=5120, moe_intermediate=1536
- 96Q/8KV GQA, head_dim=128, partial_rotary_factor=0.5
- first_k_dense_replace=3, intermediate_size=12288

**Implementation plan**:
1. **BF16 source (zero code changes)**: Point HF_MODEL to `zai-org/GLM-4.7`, set same env vars.
   160%32=0 passes EP sharding. All config params auto-load from HF config.json.
2. **FP8 source (~50 LOC)**: Import DSv3's `dequantize_tensor()` from
   `models/demos/deepseek_v3/utils/dequantize.py`. Add FP8 detection + host-side dequant
   in `_linear_weight_tt()` (layer_weights.py:163) and `_experts_weight_tt()` (layer_weights.py:215).
   Path: FP8 → dequant to BF16 on host → `ttnn.as_tensor(dtype=bfloat4_b)` on device.
3. **Env config**: Copy `.env.glm47_reap` → `.env.glm47`, change HF_MODEL, keep all SEED-3 flags.
4. **Docker compose**: Add GLM4_MOE_EXPERTS_W1_DTYPE etc. passthrough (already done).
5. **Benchmark**: bs=1 and bs=32, gen=50, compare to REAP baseline.

**Effort**: 1-2 hours (BF16), half day (FP8 dequant)
**Risk**: Low — same code, same hardware, same model type. Only risk is weight cache rebuild time (~60 min for new model).

---

### SEED-5: Fused Partial RoPE — NEEDS VALIDATION

**Status**: NEEDS VALIDATION — 20 element-wise DRAM ops per layer for Q+K RoPE
**Expected**: 5-15% improvement (10-25ms saved from ~170ms ITL)
**Research source**: `optimization-analysis-v1.md` OPT-3 (researcher, 2026-03-10)

**Root cause**: `attention_tt.py:482-518` implements partial RoPE with 10 separate ops per tensor:
2 slices (rotary/pass), 2 slices (half-dims), neg, concat, 2 multiply, add, concat.
20 ops/layer × 89 MoE layers = 1,780 ops in DRAM interleaved memory.

**Fix options** (in order of preference):
1. **Use `ttnn.experimental.rotary_embedding_llama`** — may support partial rotary factor.
   Need to verify: NeoX-style rotation (not GPT-J), partial_rotary_factor=0.5, HEIGHT_SHARDED input.
2. **Custom fused kernel** — single op: slice rotary portion, apply rotation, concat with pass-through.
3. **Reduce ops without fusion** — pre-compute cos/sin for just the rotary dims, eliminate slices.

**Implementation plan**:
1. Check `ttnn.experimental.rotary_embedding_llama` API and constraints
2. Test with a single attention layer outside trace
3. If compatible: replace 10 ops with 1, benchmark
4. If not: try option 2 or 3

**Effort**: 2-4 hours (option 1), 1-2 days (option 2)
**Risk**: Medium — rotary embedding op may not support NeoX-style or partial factor.
**Transfers to GLM-4.7**: Yes, identical attention architecture.

---

### SEED-6: Add-Before-Reduce CCL Optimization — NEEDS VALIDATION

**Status**: NEEDS VALIDATION — eliminate 1 of 4 axis-0 all_reduces per MoE layer
**Expected**: ~4% improvement (~6.7ms saved from ~170ms ITL)
**Research source**: `optimization-analysis-v1.md` OPT-6 (researcher, 2026-03-10)

**Root cause**: Each MoE layer does 4 axis-0 all_reduces:
1. Attention O-proj TP reduce
2. Shared expert TP reduce
3. Routed expert TP reduce (before EP reduce)
4. (EP reduce uses rs_ag on axis-1)

#2 and #3 are both `all_reduce(axis=0)` of `[1,1,32,5120]` tensors.
If we `add(shared_partial, routed_partial)` FIRST, then do ONE all_reduce, we save 1 collective.

**Why this is NOT the same as FUSE_SHARED_EP_REDUCE=1** (which hangs):
- FUSE_SHARED_EP_REDUCE=1 fuses the ENTIRE reduce pipeline (including axis-1 rs_ag)
- This proposal ONLY merges the axis-0 TP reduce: `add → all_reduce(axis=0) → rs_ag(axis=1)`
- The add is a local op (no CCL), so it can't trigger the CCL deadlock

**Implementation** (`decoder_layer_tt.py:480-570`):
1. Add env var `GLM4_MOE_MERGE_TP_REDUCE=0` (default off)
2. When enabled: `ttnn.add(shared_partial, routed_partial)` → single `all_reduce(axis=0)` → `rs_ag(axis=1)`
3. Feature-flagged, safe to test without risk

**Effort**: 1-2 hours
**Risk**: Medium — need to verify the shared expert partial sum can be added before TP reduce
  (it should be, since both are TP-partial sums of the same hidden dimension).
**Transfers to GLM-4.7**: Yes, identical structure.

---

## Completed Seeds (Implemented)

### SEED-0: Batch>1 via rs_ag_async (V2 P6) — SHIPPED

**Implemented**: 2026-03-09 | **Result**: 3.5 → 99.5 tok/s aggregate (28.4x)
**Details**: See `batch-gt1-implementation-plan.md` and `WORKLOG_GLM47_REAP_GALAXY_WORMHOLE.md`.

---

## Rejected / Dead Ends (Do NOT Seed)

| Idea | Why Rejected |
|------|-------------|
| Fused gate+up sparse matmul | **-14.6% bs=1, -2.2% bs=32** — porting bug: unnecessary `ttnn.clone` calls added 267 extra trace programs (+12-15ms). Root cause: `research-fusion-regression.md`. Even with clone removed, predicted break-even (slices offset dispatch savings). Would need custom `split_silu_mul` C++ kernel for a real win. Code left in place (GLM4_MOE_FUSE_EXPERTS_GATE_UP=0 default). |
| ~~Selective BF4~~ | **REVERSED** — original test was flawed (stale cache/code bug). Re-test shows all-expert BF4 is VIABLE (+28% bs=32, PPL -5.7%). See SEED-3 above. |
| ~~Blanket BF4~~ | **REVERSED** — see SEED-3 above. All-expert BF4 confirmed coherent. |
| EP_L1=1 | Garbled — L1 incompatible with TG mesh CCL |
| FUSE_SHARED_EP_REDUCE=1 | Infinite hang — CCL deadlock on 2D mesh |
| LoFi math fidelity | No speedup — DRAM-BW-bound, not compute-bound |
| DRAM-sharded attn weights | +2.1% — not significant at any batch size |
| Device-side sampling | +0% — host sampling hidden behind device trace |
| DP specialization | MoE EP=32 requires all 32 chips per layer |
| Async CCL overlap | Not real overlap in trace mode (single CQ) |
| Weight prefetching | No Wormhole support (Blackhole-only) |
| Continuous batching tuning | Already optimal at bs=32 |

---

## Seed Lifecycle

```
RESEARCH → seed doc entry (status: CONFIRMED or NEEDS VALIDATION)
         → team lead reviews
         → implementer executes
         → benchmark validates
         → seed moves to "Completed" section with results
         → OR seed moves to "Rejected" if benchmark fails
```

**Rules**:
1. Only add seeds with COMPLETE CONFIDENCE in a positive result (or clear validation path)
2. Every seed must have: root cause, fix design, expected impact, effort estimate, risk
3. Research sources must be cited (which researcher, which doc, which date)
4. Seeds are model+device specific — each workstream has its own research-seed.md
5. Team lead updates this doc; researchers write to their own research docs
6. Implementer reads this doc to understand WHAT to implement and WHY
