# Trace Region Sizing & 355B Performance Research

**Created**: 2026-03-13
**Hardware**: Blackhole Galaxy (32 BH chips, mesh 8x4, TP=4, DP=8)
**Models**: GLM-4.7-355B (160 experts) vs REAP-218B (96 experts)

---

## Trace Region Sizing

### How Trace Regions Work

The trace system records device dispatch commands during a "capture" phase, then replays
them efficiently. Key flow:

1. **`begin_trace_capture`**: Sets bypass mode — dispatch commands are written to a buffer
   instead of being sent to device
2. **During capture**: Each ttnn operation accumulates dispatch commands in `ordered_mesh_trace_md_`
3. **`end_trace_capture`**: Processes all metadata, computes total trace size, allocates DRAM buffer
4. **Buffer allocation** depends on `trace_region_size`:
   - `> 0` (dedicated): Reserved at TOP of each DRAM bank at device init. TT_FATAL if trace exceeds allocation.
   - `== 0` (dynamic): Allocated top-down in DRAM, sharing space with weights.

### What Determines Trace Size

For 92-layer GLM-4.7-355B decode at bs=1:
- ~2200-2500 programs total (12-15 per MoE layer, 8-10 per attention block)
- 500-2000 bytes per program dispatch (kernel refs, core grid, CB configs, args, semaphores)
- Trace stores per-range command sequences, takes MAX across device ranges

### Why Trace Grows Across Retries

Observed: 1.33 → 1.95 → 2.0 → 2.15 GB across retry captures.

Cause: **program cache warming**. First capture has compact dispatch (kernel refs from cache).
Re-captures may have cache misses (different `per_core_M` configs in sparse_matmul),
requiring full kernel binary uploads. Stabilizes at ~2.15 GB after all configs are cached.

### Sizing Recommendation

| Factor | Value |
|--------|-------|
| Observed steady-state trace size | ~2.15 GB |
| Current setting | 3 GB |
| Headroom | 40% |
| DRAM consumed per bank | 0.375 GB (3 GB / 8 banks) |
| DRAM remaining per bank | 3.625 GB |

**3 GB is adequate.** Going to 4 GB wastes 1 GB/chip. Alternative: `trace_region_size=0`
(dynamic mode) avoids pre-allocation, but requires no DRAM allocs/deallocs during capture.

---

## 355B vs REAP-218B Performance Projection

### Key Difference: Expert Count

| | REAP-218B | 355B |
|--|-----------|------|
| n_routed_experts | 96 | 160 (+67%) |
| experts_per_device | 3 | 5 (+67%) |
| Everything else | Same | Same |

### Impact on Performance

More experts per device means more DRAM weight reads per forward pass.
Sparse matmul reads ALL local expert weights (sparsity mask zeros out inactive ones).

REAP-218B baseline: 5.9 tok/s (170ms ITL) at bs=1.

Estimated ITL breakdown:
- Attention + RMSNorm: ~40% (~68ms) — **unchanged**
- MoE routing: ~5% (~8.5ms) — slightly larger gate weight
- Expert compute (sparse_matmul): ~35% (~60ms) — **scales with experts/device**
- Shared expert + all-reduce: ~15% (~25ms) — **unchanged**
- Embedding/LM head: ~5% (~8.5ms) — **unchanged**

355B expert compute: 60ms * 1.3-1.5x = **78-90ms** (not full 1.67x due to sparsity skip).

**355B projected**: ~195-200ms ITL → **~5.0-5.1 tok/s at bs=1**

---

## Grid Fix Opportunity (P0)

### The Bug

`_make_sparse_matmul_program_config()` in `moe_tt.py:93-127` selects a rectangular compute
grid. On BH (13x10=130 cores), the algorithm converges on pathologically small grids:

**Gate/up projections** (out_features=1536, n_tiles=48):
- Algorithm finds `per_core_N=4, num_blocks=12` (12 <= core_x=13, exit)
- **12 cores out of 130 (9.2% utilization)**

**Down projection** (out_features=5120, n_tiles=160):
- Algorithm converges on `per_core_N=13, num_blocks=13`
- **13 cores out of 130 (10% utilization)**

### Comparison with Wormhole

| Operation | WH (8x8=64) | BH (13x10=130) | BH is worse |
|-----------|-------------|-----------------|-------------|
| Gate/up | 48 cores (75%) | 12 cores (9.2%) | 4x worse |
| Down | 40 cores (62.5%) | 13 cores (10%) | 3x worse |

BH has 2x more cores but uses 3-4x fewer.

### Fix: Maximize Full-Row Core Usage

```python
max_full_rows = min(core_y, n_tiles // core_x)
if max_full_rows > 0:
    num_blocks = max_full_rows * core_x
    per_core_N = ceil(n_tiles / num_blocks)
else:
    num_blocks = n_tiles
    per_core_N = 1
```

| Operation | Before | After |
|-----------|--------|-------|
| Gate/up | 12 cores | 39 cores (3 rows) |
| Down | 13 cores | 130 cores (10 rows) |

### Projected Speedup

These are DRAM-bandwidth-bound. Adding cores increases effective bandwidth up to the
8 DRAM bank limit.

**Realistic estimate**: 1.5-2x speedup on MoE expert compute.
MoE experts are ~35% of ITL → **15-25% overall ITL reduction**.

- REAP: 170ms → 130-145ms → **6.9-7.7 tok/s** (vs 5.9 current)
- 355B: 200ms → 155-170ms → **5.9-6.5 tok/s** (vs projected 5.0-5.1)

### Implementation

- File: `moe_tt.py:93-127`
- Must verify sparse_matmul kernel handles wider core grids
- `MatmulMultiCoreReuseMultiCast1DProgramConfig` requires rectangular grid (full rows)
- Test with both bs=1 and bs=32 for trace compatibility
