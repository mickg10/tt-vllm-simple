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
