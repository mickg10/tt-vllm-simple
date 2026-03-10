# GLM-4.7-REAP-218B Galaxy Wormhole: Optimization Analysis V1

**Date**: 2026-03-10
**Analyst**: Deep-dive researcher (Claude Opus 4.6), verified by Codex (GPT-5.4)
**Model**: GLM-4.7-REAP-218B (268B total, 32B active), 92 MoE layers, 96 routed experts (top-8), 1 shared expert
**Hardware**: Galaxy Wormhole, 32 chips, Mesh(8,4), TP=8, EP=32, DP=4
**Current perf**: 142.7 tok/s agg bs=32 (~4.5 tok/s/user, ~170ms ITL)
**Config**: SEED-3 BF4 (w1/w2=BF4, w3=BF8), trace_mode=decode_only, rs_ag_async axis-1, host sampling

---

## Table of Contents

1. [Task 1: Decode Forward Path End-to-End](#task-1-decode-forward-path-end-to-end)
2. [Task 2: Remaining Optimization Opportunities](#task-2-remaining-optimization-opportunities)
3. [Task 3: GLM-4.7-FP8 Feasibility](#task-3-glm-47-fp8-feasibility)

---

## Task 1: Decode Forward Path End-to-End

### 1.1 Trace Capture/Replay Overview

The decode path uses batch-bucketed traces (`model_tt.py:976`). For bs=32:

**Persistent input tensors (allocated before trace capture, `model_tt.py:1084-1119`):**

| Tensor | Shape | Layout | Memory | Mesh Mapper |
|--------|-------|--------|--------|-------------|
| `embed_tt` | [1,1,32,5120] | TILE | DRAM interleaved | Replicate |
| `positions_tt` | [8] (DP-sharded) | ROW_MAJOR | DRAM interleaved | ShardTensor2d(axis=1) |
| `cos_batch_tt` | [1,8,1,64] (DP-sharded) | TILE | DRAM interleaved | ShardTensor2d(axis=1) |
| `sin_batch_tt` | [1,8,1,64] (DP-sharded) | TILE | DRAM interleaved | ShardTensor2d(axis=1) |
| `page_table_tt` | [8,W] (DP-sharded) | ROW_MAJOR | DRAM interleaved | ShardTensor2d(axis=1) |
| `tokens_tt` | [32,1] | ROW_MAJOR | DRAM interleaved | Replicate |

**Note**: `tokens_tt` and `trans_matrix_tt` are allocated but NOT consumed by the traced graph (dead buffers).

**Trace replay loop (`model_tt.py:1022-1029`):**
1. `_update_trace_inputs()` -- 5x `copy_host_to_device_tensor` for embed, positions, cos, sin, page_table
2. `ccl.reset_sem_counters()` -- reset CCL semaphore round-robin
3. `synchronize_device()` -- drain async queue
4. `execute_trace()` -- replay captured trace (blocking=False)
5. `synchronize_device()` -- wait for completion
6. Host-side argmax from `logits_tt` (per-shard topk, ~256 bytes transfer)

**What is inside the trace (`model_tt.py:1183-1214`):**
- 92 decoder layer forwards
- Final sharded RMSNorm
- LM head linear (TP-sharded vocab)
- NO sampling ops on TG mesh (host-side after trace)

### 1.2 Per-Layer Decode Forward (MoE Layer, layers 3-91)

Reference: `decoder_layer_tt.py:211-297`

#### Phase 1: Pre-Attention RMSNorm

```
Op                    Input                    Output                  Notes
to_memory_config      DRAM/int [1,1,32,5120]   L1-W/sharded           8-core width shard
rms_norm              L1-W/sharded             L1-W/sharded           HiFi2, inplace=False
to_memory_config      L1-W/sharded             DRAM/int [1,1,32,5120]  Back to DRAM
(optional slice)      DRAM/int                 DRAM/int               If tile-padding restored
```
Total: 3-4 ops, 1 DRAM round-trip (DRAM -> L1-W -> compute -> L1-W -> DRAM)

#### Phase 2: GQA Attention (`attention_tt.py:558-783`)

Full sequence for bs=32 on TG (tg_batch_sliced=True, Bg=8 per DP group):

```
Step  Op                                Shape                           Mem In -> Out         Collective?
1     ttnn.linear (QKV)                 [1,1,32,5120]->[1,1,32,1792]    DRAM/int -> L1-W      No
2     add (QKV bias)                    [1,1,32,1792]+[1,1,1,1792]      L1-W + DRAM -> L1-W   No
3     to_memory_config                  [1,1,32,1792]                   L1-W -> DRAM/int      No
4     matmul (slice_mat)                [1,1,32,1792]->[1,1,8,1792]     DRAM -> L1/int        No
5     reshape                           logical [1,1,8,1792]            L1/int                No
6     nlp_create_qkv_heads_decode       [1,1,8,1792]->q[1,8,12,128]    L1/int -> L1-H        No
                                        +k[1,8,1,128]+v[1,8,1,128]
7     to_memory_config (q)              [1,8,12,128]                    L1-H -> DRAM/int      No
8     to_memory_config (k)              [1,8,1,128]                     L1-H -> DRAM/int      No
9     rms_norm (q_norm)                 [1,8,12,128]                    DRAM -> DRAM          No
10    rms_norm (k_norm)                 [1,8,1,128]                     DRAM -> DRAM          No
11-20 q partial RoPE (10 ops)           Various slices/concat/mul       DRAM -> DRAM          No
21-30 k partial RoPE (10 ops)           Various slices/concat/mul       DRAM -> DRAM          No
31    interleaved_to_sharded (q)        [1,8,12,128]                    DRAM -> L1-H          No
32    interleaved_to_sharded (k)        [1,8,1,128]                     DRAM -> L1-H          No
33    paged_update_cache (K)            k L1-H + keys DRAM              L1-H -> DRAM          No
34    paged_update_cache (V)            v L1-H + values DRAM            L1-H -> DRAM          No
35    paged_SDPA_decode                 q L1-H + KV cache DRAM -> DRAM  L1-H+DRAM -> DRAM     No
36    interleaved_to_sharded (attn)     [1,8,12,128]                    DRAM -> L1-H          No
37    nlp_concat_heads_decode           [1,8,12,128]->[1,1,8,1536]      L1-H -> L1-sharded    No
38    all_gather (axis=1, dim=2)        [1,1,8,1536]->[1,1,32,1536]     L1 -> DRAM            YES (4-way DP)
39    to_memory_config                  [1,1,32,1536]                   DRAM -> L1/int        No
40    matmul (user_sel_mat)             [1,1,32,1536]->[1,1,32,1536]    L1 -> L1-W            No
41    reshape                           [1,1,32,1536]                   L1-W                  No
42    matmul (o_proj)                   [1,1,32,1536]->[1,1,32,5120]    L1-W -> L1-W          No
43    all_reduce (axis=0, native)       [1,1,32,5120]                   L1-W -> DRAM/int      YES (8-way TP)
(optional reshape/slice for shape fix)
```

Total: ~43 ops, 2 collectives (all_gather axis-1, all_reduce axis-0), ~4 activation DRAM round-trips

#### Phase 3: Residual Add

```
ttnn.add                  DRAM/int + DRAM/int -> DRAM/int [1,1,32,5120]
```
1 op

#### Phase 4: Pre-MLP RMSNorm

Same as Phase 1: 3-4 ops, 1 DRAM round-trip

#### Phase 5: MoE Forward (`decoder_layer_tt.py:418-576`)

##### 5a. Shared Expert (`moe_tt.py:886-917`)

With FUSE_SHARED_EP_REDUCE=0, shared and routed run independently:

```
Step  Op                        Shape                         Mem              Collective?
1     linear (fused gate+up)    [1,1,32,5120]->[1,1,32,384]   DRAM -> DRAM     No
2     slice (gate half)         [1,1,32,384]->[1,1,32,192]    DRAM             No
3     slice (up half)           [1,1,32,384]->[1,1,32,192]    DRAM             No
4     mul (SiLU)                [1,1,32,192]                  DRAM -> DRAM     No
5     linear (down)             [1,1,32,192]->[1,1,32,5120]   DRAM -> DRAM     No
```

Note: shared expert uses fused gate+up weight (`w_mlp_gate_up`) for MoE layers (line 532-552 of layer_weights.py), so it's a single linear + slice instead of two separate linears. Per device after TP=8: intermediate=1536/8=192. The fused weight has 2*192=384 output dim.

**Shared TP reduce** (`decoder_layer_tt.py:527-532`):
```
all_reduce (axis=0, native)  DRAM/int [1,1,32,5120] -> DRAM/int   YES (8-way TP)
```

##### 5b. Router Top-K (`moe_tt.py:301-382`)

```
Step  Op                        Shape                   Mem           Collective?
1     linear (gate)             [1,1,32,5120]->[1,1,32,96]  DRAM->L1  No
2     sigmoid                   [1,1,32,96]             L1            No
3     add (e_score_bias)        [1,1,32,96]             L1            No
4     topk (k=8)                [1,1,32,96]->[1,1,32,8] L1            No
5     gather (scores)           [1,1,32,96]->[1,1,32,8] L1            No
6     sum (dim=3)               [1,1,32,8]->[1,1,32,1]  L1            No
7     add (epsilon)             [1,1,32,1]              L1            No
8     div (normalize)           [1,1,32,8]              L1            No
9     mul (scale 2.5)           [1,1,32,8]              L1            No
```
9 ops, ALL in L1 (use_l1=True because tokens<=64)

##### 5c. Sparse Expert Forward (`moe_tt.py:399-638`, skip_final_reduce=True)

```
Step  Op                        Shape                              Mem              Collective?
1     reshape (hidden)          [1,1,32,5120]                      DRAM             No
2-3   reshape (indices, weights) [1,1,32,8]                        L1               No
4-5   to_layout (idx,wt RM)    TILE->ROW_MAJOR                    L1               No
6     scatter                   [1,1,32,96] + idx/wt               DRAM/RM          No
7     moe_expert_token_remap   ->local_wt[1,1,32,3]+sparsity      DRAM/RM          No
8     reshape (expert_input)   [1,1,32,5120]                      DRAM             No
9     sparse_matmul (w1/gate)  [1,1,32,5120]->[1,3,32,1536]       DRAM             No
10    sparse_matmul (w3/up)    [1,1,32,5120]->[1,3,32,1536]       DRAM             No
11    mul (SiLU)               [1,3,32,1536]                      DRAM             No
12    reshape (x_ff)           [1,3,32,1536]                      DRAM             No
13    sparse_matmul (w2/down)  [1,3,32,1536]->[1,3,32,5120]       DRAM             No
14-15 reshape+permute (output) [3,1,32,5120]                      DRAM             No
16-18 layout+permute+layout(wt) [3,1,32,1]                        DRAM             No
19    mul (weighted)           [3,1,32,5120]                      DRAM             No
20    sum (dim=0)              [1,1,32,5120]                      DRAM             No
```
~20 ops, 0 collectives (skip_final_reduce=True)

##### 5d. EP Reduce (in decoder_layer_tt.py:525-570)

With FUSE_SHARED_EP_REDUCE=0, EP_REDUCE_DEVICE=1:

```
Routed EP reduce:
  all_reduce (axis=0, native)                DRAM -> DRAM       YES (8-way TP)
  reduce_scatter_minimal_async (axis=1,dim=3) DRAM -> DRAM       YES (4-way DP, reduce_scatter)
  all_gather_async (axis=1, dim=3)           DRAM -> DRAM       YES (4-way DP, all_gather)

Merge:
  add (shared + routed)                      DRAM -> DRAM       No
```

4 ops, 3 collectives

#### Phase 6: Final Residual Add

```
ttnn.add                  DRAM/int + DRAM/int -> DRAM/int [1,1,32,5120]
```
1 op

### 1.3 Summary: Per-Layer Op Counts

| Component | Ops | Collectives | DRAM Round-trips |
|-----------|-----|-------------|------------------|
| Pre-attn RMSNorm | 3-4 | 0 | 1 |
| GQA Attention | ~43 | 2 | 4 |
| Residual add #1 | 1 | 0 | 0 |
| Pre-MLP RMSNorm | 3-4 | 0 | 1 |
| Shared Expert | 5 | 0 | 0 |
| Shared TP Reduce | 1 | 1 | 0 |
| Router Top-K | 9 | 0 | 0 |
| Sparse Experts | ~20 | 0 | 0 |
| EP Reduce | 4 | 3 | 0 |
| Residual add #2 | 1 | 0 | 0 |
| **MoE Layer Total** | **~88** | **6** | **~6** |
| **Dense Layer Total** | **~56** | **3** | **~6** |

**Full trace (92 layers = 3 dense + 89 MoE + final norm + LM head):**
- Total ops: 3*56 + 89*88 + ~7 = ~8,007
- Total collectives: 3*3 + 89*6 = 543
- Collective breakdown per MoE layer:
  - 2 in attention (all_gather axis-1, all_reduce axis-0)
  - 1 shared expert TP reduce (all_reduce axis-0)
  - 3 EP reduce (all_reduce axis-0, reduce_scatter axis-1, all_gather axis-1)

### 1.4 Critical Path Analysis

For a single MoE decode layer at bs=32, the critical path is:

```
RMSNorm (DRAM->L1-W->DRAM)
  -> QKV linear (DRAM->L1-W)
    -> QKV bias add + DRAM write
      -> DP batch slice matmul
        -> QKV head split
          -> QK norm (DRAM round-trip)
            -> Partial RoPE (10 element-wise ops, ALL in DRAM)
              -> L1 shard + KV cache update
                -> Paged SDPA decode
                  -> L1 shard + concat heads
                    -> all_gather (DP, axis-1)
                      -> user_sel matmul + O projection
                        -> all_reduce (TP, axis-0)
                          -> Residual add
                            -> RMSNorm (DRAM->L1-W->DRAM)
                              -> [Shared expert: fused gate+up + SiLU + down]
                              -> [Router: gate + sigmoid + topk + gather + normalize]
                              -> [Sparse experts: 3x sparse_matmul + SiLU + sparse_matmul]
                                -> Shared TP reduce (all_reduce axis-0)
                                -> EP reduce (all_reduce axis-0 + rs_ag axis-1)
                                  -> Merge add
                                    -> Residual add
```

**Key observation**: The attention partial RoPE path performs 20 element-wise ops (10 for Q, 10 for K) entirely in DRAM interleaved memory. These are small tensors ([1,8,12,64] for Q slices) so each op is individually fast, but the cumulative DRAM bandwidth is non-trivial.

---

## Task 2: Remaining Optimization Opportunities

### OPT-1: Fuse Shared Expert Gate+Up into Single Linear (Already Done for MoE Layers)

**What**: For MoE layers (3-91), the shared expert already uses a fused `w_mlp_gate_up` weight (`layer_weights.py:532-552`), doing one linear + slice instead of two separate linears.

**Status**: Already implemented. For dense layers (0-2), gate and up are still separate (`layer_weights.py:516-531`). Since there are only 3 dense layers, fusing them would save ~6 ops total across the trace -- negligible impact.

**Impact**: N/A (already done for the 89 MoE layers that dominate).

### OPT-2: Eliminate Dead Trace Buffers

**What**: `tokens_tt` (model_tt.py:1100-1107) and `trans_matrix_tt` (stored in trace state, model_tt.py:1227) are allocated on device but never read by the traced graph. The decode path uses host-side embedding lookup (writing to `embed_tt`), and partial RoPE uses explicit cos/sin (not the trans_matrix).

**Where**: `model_tt.py:1100-1107` (tokens_tt allocation), `model_tt.py:1219-1233` (state storage)

**Expected impact**: Saves ~640 KB DRAM per device per trace bucket (tokens_tt: 32*4=128 bytes, trans_matrix: 32*32*2=2048 bytes -- negligible). More importantly, eliminates a `copy_host_to_device_tensor` call during replay if `tokens_tt` update is removed from `_update_trace_inputs`.

Actually, looking at `_update_trace_inputs` (line 1235-1309): tokens_tt is NOT updated there -- only embed_tt, positions_tt, cos/sin, and page_table_tt. So there is no wasted copy. The dead allocation wastes only DRAM.

**Impact**: Negligible (<0.1%). Cleanup only.
**Risk**: None.
**Transfers to GLM-4.7**: Yes, same pattern.

### OPT-3: Reduce Partial RoPE Element-Wise Ops

**What**: The partial RoPE implementation (`attention_tt.py:482-518`) uses 10 separate ops per tensor (Q and K): 2 slices for rotary/pass, 2 slices for half-dims, neg, concat, 2 multiply, add, concat. That's 20 ops per layer for Q+K.

A fused partial RoPE kernel or use of `ttnn.experimental.rotary_embedding_llama` could replace 10 ops with 1-2 ops. The current implementation works on small tensors in DRAM interleaved, so each individual op is fast (~15us dispatch + compute), but 20 * ~50us = ~1ms per layer, ~89ms across 89 MoE layers.

**Where**: `attention_tt.py:482-518` (decode), `attention_tt.py:520-556` (prefill)

**Expected impact**: If each RoPE op averages 50us (dispatch + small DRAM read/write), eliminating 16 of 20 ops saves ~800us/layer * 89 layers = ~71ms per decode step. That's ~71ms / ~170ms ITL = **~42% improvement at bs=32**. However, this estimate assumes no pipeline overlap. In trace mode, many of these small ops may overlap with prior DRAM writes. Realistic estimate: **5-15% improvement** (10-25ms saved).

**Risk**: Medium. Need to verify that `ttnn.experimental.rotary_embedding_llama` supports:
- Partial rotary factor (only rotate first 64 of 128 dims)
- NeoX-style rotation (cat(-x2, x1), not GPT-J style interleave)
- HEIGHT_SHARDED input (Q after head split)
- The current code explicitly handles NeoX-style, and the standard TT rotary embedding may assume GPT-J style.

**Transfers to GLM-4.7**: Yes, identical attention architecture (96Q/8KV, partial_rotary_factor=0.5).

### OPT-4: Reduce Attention Memory Config Transitions

**What**: The attention path has multiple DRAM <-> L1 transitions:
1. QKV linear outputs to L1-W, then `to_memory_config` back to DRAM (line 608)
2. After QK norm + RoPE (DRAM), `interleaved_to_sharded` to L1-H for SDPA (line 677-678)
3. SDPA output to DRAM, then `interleaved_to_sharded` to L1-H for concat_heads (line 707)

Each `to_memory_config`/`interleaved_to_sharded` is a DRAM bandwidth operation. Keeping data in L1 between QKV linear and head split could eliminate 2 transitions.

**Where**: `attention_tt.py:594-608` (QKV to DRAM), `attention_tt.py:665-678` (back to L1-H)

**Expected impact**: Each DRAM->L1 transition for [1,1,32,1792] costs ~1792*32*2 = 114 KB read. At ~300 GB/s DRAM BW, that's ~0.4us per device. Negligible. The bottleneck is not the data movement but the kernel launch overhead (even in trace, ~15us/op).

Eliminating 2 transitions saves ~2 ops * 89 layers * 15us = ~2.7ms. **~1.5% improvement**.

**Risk**: High. Changing memory configs affects program cache and may require L1 space management. QK norm currently requires DRAM interleaved input.
**Transfers to GLM-4.7**: Yes.

### OPT-5: Batch-Aware Router to L1 Pipeline

**What**: The router path (`moe_tt.py:301-382`) already uses L1 for all intermediate tensors when tokens<=64 (line 321). This is already optimal for decode. No further improvement possible here.

**Status**: Already optimized.

### OPT-6: Reduce CCL Collective Count

**What**: Each MoE layer has 6 collectives:
1. Attention all_gather (axis-1, 4-way DP batch gather)
2. Attention all_reduce (axis-0, 8-way TP O-proj reduce)
3. Shared expert all_reduce (axis-0, 8-way TP)
4. Routed expert all_reduce (axis-0, 8-way TP)
5. Routed expert reduce_scatter (axis-1, 4-way DP)
6. Routed expert all_gather (axis-1, 4-way DP)

Collectives #3 and #4 are both axis-0 all_reduce of [1,1,32,5120] tensors. If shared and routed outputs could be added before the TP reduce, only one all_reduce would be needed instead of two. This is exactly what `FUSE_SHARED_EP_REDUCE=1` does -- but it hangs on TG mesh.

**Where**: `decoder_layer_tt.py:480-524` (the fuse_reduce=True branch)

**Expected impact**: Eliminating 1 all_reduce per layer saves ~89 * (axis-0 all_reduce latency). A Ring all_reduce across 8 devices with ~114 KB payload takes ~50-100us. So ~89 * 75us = ~6.7ms = **~4% improvement**.

**Risk**: The fused path (`FUSE_SHARED_EP_REDUCE=1`) is confirmed to hang (`resume.md:49`). The hang is in the CCL layer, not in the fuse logic itself. A workaround could be:
- Instead of one fused all_reduce, do: `add(shared_partial, routed_partial)` FIRST (no collective), then a SINGLE axis-0 all_reduce, then axis-1 rs_ag_async. This avoids the specific pattern that triggers the hang (which may be related to the 1/DP scaling before the add).

**Transfers to GLM-4.7**: Yes, identical structure.

### OPT-7: Sparse Matmul Grid Utilization

**What**: `_make_sparse_matmul_program_config` (`moe_tt.py:93-127`) uses `core_x * core_y` = 72 cores on Galaxy WH. However, the rectangular grid constraint (lines 112-114) may reduce actual utilization.

For `out_features=1536` (gate/up): n_tiles = 1536/32 = 48, num_cores = 72, per_core_N = ceil(48/72) = 1, num_blocks = 48. 48 > 8 (core_x) and 48 % 8 == 0, so 48 cores used (6 rows * 8 cols). That's 48/72 = **67% grid utilization**.

For `out_features=5120` (down): n_tiles = 160, num_cores = 72, per_core_N = ceil(160/72) = 3, num_blocks = ceil(160/3) = 54. 54 > 8, 54 % 8 != 0. Increment per_core_N: 4 -> 40, 40 % 8 == 0. So 40 cores used (5 rows * 8 cols). That's 40/72 = **56% grid utilization**.

**Where**: `moe_tt.py:93-127`

**Expected impact**: Sparse matmuls dominate MoE compute. At bs=32 with 3 sparse matmuls per layer (w1, w3, w2), each executing 3 expert blocks, improving grid utilization from 56-67% to 100% would reduce compute time by ~35-44%. But sparse matmuls are DRAM-BW-bound (reading expert weights from DRAM), not compute-bound. The grid constraint means fewer cores read in parallel, which DOES affect DRAM bandwidth utilization.

Rough estimate: 3 sparse_matmuls * 89 layers, each taking ~300us (from REAP profiling). If 35% of that is wasted due to grid underutilization: 3 * 89 * 300 * 0.35 = ~28ms. **~16% improvement** if the grid constraint is fixed.

However, this requires changes to the `sparse_matmul` program config logic, which is shared infrastructure. The rectangular grid requirement is a hardware constraint of the multicast topology.

**Risk**: Medium-high. Changing sparse_matmul core allocation is deep infrastructure work.
**Transfers to GLM-4.7**: Yes, same issue (worse with 5 experts/device vs 3).

### OPT-8: Weight Pre-staging with Double Buffering

**What**: Expert weights are read from DRAM for each sparse_matmul. Since expert routing is known before expert compute starts, weights for the selected experts could be pre-staged to L1 while the router computes.

**Status**: This is effectively `EP_L1=1`, which is confirmed broken on TG mesh (garbled output). The L1 memory config is incompatible with CCL operations that follow.

**Where**: `moe_tt.py:260-261`

**Expected impact**: N/A -- proven broken.
**Risk**: N/A.

### OPT-9: Attention DP Batch Slicing Optimization

**What**: For bs=32 on TG, attention slices the batch into 4 DP groups of 8 using a `slice_mat` matmul (`attention_tt.py:629-634`). This is a [1,32,8,32] * [1,1,32,1792] matmul -- essentially a selection operation implemented as a matrix multiply.

The `user_sel_mat` matmul at the end (`attention_tt.py:733-739`) reverses this: gathering DP groups back to full batch.

Both matmuls use BF4 weights (line 373, 384) for the selection matrices. These are "identity-like" matrices that select rows -- BF4 precision is sufficient.

**Potential optimization**: Replace these matmuls with `ttnn.slice` + `ttnn.concat` operations if they have lower overhead. However, slice/concat on TG mesh may have the same or higher overhead due to mesh coordination.

**Expected impact**: ~2 matmuls * 89 layers * ~30us = ~5.3ms = **~3% improvement**. But only if a faster alternative exists.
**Risk**: High -- TG mesh slice/concat semantics are tricky.
**Transfers to GLM-4.7**: Yes.

### OPT-10: Reduce Number of Layers (GLM-4.7 Only)

**What**: GLM-4.7 has 64 layers vs REAP's 92. With identical per-layer structure, this alone gives:
- 64/92 = 0.696x the per-layer cost
- Expected ITL: 170ms * 0.696 = ~118ms
- Expected throughput at bs=32: 142.7 / 0.696 = ~205 tok/s aggregate

**Impact**: Built-in architectural advantage for GLM-4.7 over REAP.
**Transfers to GLM-4.7**: This IS GLM-4.7.

### Summary Table

| ID | Optimization | Expected Impact | Risk | Effort | Transfers? |
|----|-------------|-----------------|------|--------|------------|
| OPT-3 | Fused Partial RoPE | 5-15% | Medium | Medium | Yes |
| OPT-7 | Sparse Matmul Grid | up to 16% | High | High | Yes |
| OPT-6 | Reduce CCL Count | ~4% | Medium | Low | Yes |
| OPT-9 | DP Batch Slice Opt | ~3% | High | Medium | Yes |
| OPT-4 | Reduce Mem Transitions | ~1.5% | High | Medium | Yes |
| OPT-2 | Dead Buffer Cleanup | <0.1% | None | Low | Yes |

**Recommended priority**: OPT-3 (fused RoPE) > OPT-6 (CCL reduction) > OPT-7 (grid utilization)

---

## Task 3: GLM-4.7-FP8 Feasibility

### 3.1 Architecture Comparison

| Parameter | REAP-218B | GLM-4.7 (Full) | Delta |
|-----------|-----------|-----------------|-------|
| Total params | 268B | 358B | +34% |
| Active params | 32B | 32B | Same |
| num_hidden_layers | 92 | 64 | -30% |
| n_routed_experts | 96 | 160 | +67% |
| num_experts_per_tok | 8 | 8 | Same |
| first_k_dense_replace | 3 | 3 | Same |
| intermediate_size (dense) | 12288 | 12288 | Same |
| moe_intermediate_size | 1536 | 1536 | Same |
| hidden_size | 5120 | 5120 | Same |
| num_attention_heads | 96 | 96 | Same |
| num_key_value_heads | 8 | 8 | Same |
| head_dim | 128 | 128 | Same |
| partial_rotary_factor | 0.5 | 0.5 | Same |
| vocab_size | ~152K | ~152K | Same |

Key insight: Attention is IDENTICAL. MoE routing parameters are identical (top-8). Only the expert pool size and number of layers differ.

### 3.2 Memory Budget: 160 Experts with SEED-3 BF4

#### Per-Expert Weight Memory

All experts have identical dimensions: gate_proj [1536, 5120], up_proj [1536, 5120], down_proj [5120, 1536].

In TT tile format (32x32 tiles):

| Projection | Shape [in, out] | Tiles | BF4 bytes | BF8 bytes |
|-----------|-----------------|-------|-----------|-----------|
| w1 (gate) | [5120, 1536] | 160*48=7680 | 7680*576 = 4,423,680 | 7680*1088 = 8,355,840 |
| w2 (down) | [1536, 5120] | 48*160=7680 | 7680*576 = 4,423,680 | 7680*1088 = 8,355,840 |
| w3 (up) | [5120, 1536] | 160*48=7680 | -- | 7680*1088 = 8,355,840 |

SEED-3 config (w1=BF4, w2=BF4, w3=BF8):
- Per expert: 4,423,680 + 4,423,680 + 8,355,840 = **17,203,200 bytes = 16.4 MB**

Note: BF4 tile is 576 bytes (32*32*0.5 + overhead), BF8 tile is 1088 bytes (32*32*1 + overhead). The exact tile sizes include header and alignment overhead.

#### Per-Device Expert Memory (EP=32)

| Model | Experts | Per Device | Per Expert | Per Device/Layer |
|-------|---------|------------|------------|------------------|
| REAP | 96 | 3 | 16.4 MB | **49.2 MB** |
| GLM-4.7 | 160 | 5 | 16.4 MB | **82.0 MB** |

#### Total Expert Memory Per Device

| Model | MoE Layers | Per Layer | Total Expert DRAM |
|-------|------------|-----------|-------------------|
| REAP | 89 | 49.2 MB | **4.38 GB** |
| GLM-4.7 | 61 | 82.0 MB | **5.00 GB** |

#### Non-Expert Weight Memory Per Device

Attention weights (TP=8, per device):
- w_qkv: [5120, 1792] BF8 = 5120/32 * 1792/32 * 1088 = 160*56*1088 = 9.74 MB
- w_o: [1536, 5120] BF8 = 48*160*1088 = 8.35 MB
- QKV bias: [1, 1792] BF16 = negligible
- QK norm: [128] BF16 = negligible
- Layernorm: [5120] BF16 = negligible

Dense MLP (layers 0-2, TP=8):
- gate: [5120, 1536] BF8 = 8.35 MB
- up: [5120, 1536] BF8 = 8.35 MB
- down: [1536, 5120] BF8 = 8.35 MB

Shared expert MLP (TP=8, per device):
- fused gate+up: [5120, 384] BF8 = 160*12*1088 = 2.09 MB
- down: [192, 5120] BF8 = 6*160*1088 = 1.04 MB

Per layer non-expert: 9.74 + 8.35 + 2.09 + 1.04 = ~21.2 MB
LM head: ~vocab/8 * 5120 * BF8 = ~152K/8 * 5120 / 1024^2 = ~19K * 5120 / 1024^2 * 1088 tiles... approximately 100 MB
Embedding: host-only (not on device)

| Model | Layers | Attn+Shared/layer | Dense layers | LM Head | Total Non-Expert |
|-------|--------|-------------------|--------------|---------|------------------|
| REAP | 92 | 21.2 MB * 92 = 1.95 GB | 3 * 25 MB = 75 MB | ~100 MB | **~2.13 GB** |
| GLM-4.7 | 64 | 21.2 MB * 64 = 1.36 GB | 3 * 25 MB = 75 MB | ~100 MB | **~1.53 GB** |

#### Total Weight Memory Per Device

| Model | Expert DRAM | Non-Expert DRAM | Total | WH DRAM (32 GB) |
|-------|-------------|-----------------|-------|-----------------|
| REAP-218B | 4.38 GB | 2.13 GB | **6.51 GB** | 20.4% |
| GLM-4.7 | 5.00 GB | 1.53 GB | **6.53 GB** | 20.4% |

**Verdict**: GLM-4.7 with SEED-3 BF4 fits easily. Total weight memory is nearly identical to REAP despite 67% more experts, because GLM-4.7 has 30% fewer layers. KV cache and activation memory leave ample headroom.

### 3.3 Code Compatibility

The `glm4_moe` codebase is parameterized through `Glm4MoeHParams` (`config.py:10-116`), which reads all model dimensions from HF `config.json`. Most code paths use `hparams.n_routed_experts`, `hparams.num_hidden_layers`, etc. dynamically.

**Changes needed for 160 experts:**

| Component | File:Line | Change Needed? | Details |
|-----------|-----------|---------------|---------|
| Config loading | `config.py:66-90` | None | Auto from config.json |
| Expert weight loading | `layer_weights.py:609-697` | None | Loop uses `hparams.n_routed_experts` |
| Expert sharding | `layer_weights.py:236` | None | `160 % 32 == 0` passes check |
| MoE runtime | `moe_tt.py:171-294` | None | `experts_per_device = 160/32 = 5` auto |
| Sparse matmul config | `moe_tt.py:93-127` | None | Grid math adapts to expert count |
| Router gate weight | `layer_weights.py:575` | None | `w_gate` shape is `[5120, 160]` auto |
| Expert mapping | `moe_tt.py:195-209` | None | `torch.eye(32).repeat_interleave(5)` auto |
| Sparsity block | `moe_tt.py:245` | None | Block size 32 works for any expert count |
| Batch bucketing | `model_tt.py:1054-1234` | None | Batch-aware, not expert-aware |
| Attention | `attention_tt.py` | None | Completely expert-agnostic |
| Dense layers | `decoder_layer_tt.py:388-412` | None | `first_k_dense_replace=3` identical |

**Only actual code change**: None for 160 experts. The codebase is fully parameterized.

**Potential issue**: Weight loading time. 160 experts * 64 layers * 3 weights = 30,720 individual weight tensors to load. At REAP's loading rate (89 MoE layers * 96 * 3 = 25,632 weights in ~10 min), GLM-4.7 would be ~12 min (61 MoE layers * 160 * 3 = 29,280 weights). Acceptable with weight caching.

### 3.4 FP8 Weight Loading

The `glm4_moe` codebase has **NO FP8 dequantization support** (`layer_weights.py`). There is no `_dequant_weight()` function anywhere in `models/demos/glm4_moe/`.

The REAP model (`cerebras/GLM-4.7-REAP-218B-A32B`) ships BF16 weights, so no dequantization is needed. The per-projection dtype (BF4/BF8) is applied during `ttnn.as_tensor` conversion from BF16 torch tensors.

For FP8 source models (`zai-org/GLM-4.7-FP8`):

**What the FP8 model provides:**
- Weight tensors in `torch.float8_e4m3fn` format
- Per-block scale tensors named `{key}.weight_scale_inv` with shape `[ceil(R/block_h), ceil(C/block_w)]`
- Block size from `quantization_config["weight_block_size"]` in config.json (typically [128, 128])

**What needs to be added:**
1. **Dequantization function**: Convert FP8 + scale_inv -> BF16 on host before `ttnn.as_tensor`. Reference implementation exists in DSv3: `deepseek_v3/utils/dequantize.py:12-67` (`dequantize_tensor()`)
2. **Scale key lookup**: DSv3 uses `{key}.weight_scale_inv` (`mlp_dequant.py:47`). GLM-4.7-FP8 likely uses the same convention but should be verified from the safetensors index.
3. **Integration point**: In `_linear_weight_tt()` (`layer_weights.py:163`) and `_experts_weight_tt()` (`layer_weights.py:215`), add a dequant step before the existing `ttnn.as_tensor` call.

**Implementation sketch:**
```python
# In layer_weights.py, add before ttnn.as_tensor in _linear_weight_tt:
if weight_fp8 is not None and scale_inv is not None:
    from models.demos.deepseek_v3.utils.dequantize import dequantize_tensor
    block_shape = (1, block_h, block_w)  # from quantization_config
    torch_weight = dequantize_tensor(weight_fp8, scale_inv, block_shape)
else:
    torch_weight = torch_weight_out_in  # existing BF16 path
```

**Key difference from DSv3**: DSv3 dequantizes on host then converts to BF8/BF4 on device via `ttnn.as_tensor(dtype=ttnn.bfloat8_b)`. The same approach works for GLM-4.7: FP8 -> dequant to BF16 on host -> convert to BF4/BF8 in `ttnn.as_tensor`. No device-side FP8 support needed.

**Effort**: ~50 lines of code. Import DSv3's `dequantize_tensor`, add FP8 detection in `convert_decoder_layer_weights()` and `_experts_weight_tt()`.

### 3.5 Expected Performance: 160 vs 96 Experts

#### Per-Layer Analysis

**What changes with 160 experts (5 per device vs 3):**

1. **Router gate linear**: [5120, 160] instead of [5120, 96]. Cost scales with output dim: 160/96 = 1.67x. This is a small matmul (~1% of layer time), so +0.67% absolute.

2. **Topk**: k=8 from 160 candidates vs 96. Cost is O(n*k) where n is expert count: 160/96 = 1.67x. Also small (~1% of layer time).

3. **Sparse matmul**: The critical path. With 5 experts per device instead of 3:
   - Expert weight shapes are the same (1536x5120)
   - Number of local experts to compute increases: 5 vs 3
   - Sparsity block size unchanged (32)
   - Expert input reshaped to `[1, num_blocks, 32, 5120]` where num_blocks=1 (bs=32)
   - Expert output: `[num_blocks, 5, 32, 5120]` vs `[num_blocks, 3, 32, 5120]`
   - Each sparse_matmul reads 5 expert weights vs 3: **1.67x more DRAM reads**
   - Since sparse matmul is DRAM-BW-bound, expect ~1.67x cost

4. **Weight aggregation**: `mul(weighted)` and `sum(dim=0)` over 5 experts vs 3: 1.67x.

5. **Collectives**: Unchanged -- same tensor size [1,1,32,5120] regardless of expert count.

6. **Shared expert**: Unchanged -- same architecture.

7. **Attention**: Unchanged -- same GQA architecture.

#### Estimated Performance

The MoE block (shared + routed experts + router + EP reduce) is roughly 55% of layer time (based on GLM-4.7-Flash profiling patterns). Of that 55%, sparse matmul is ~40% of layer time (the dominant component).

With 64 layers instead of 92 and 1.67x sparse matmul cost:

| Component | REAP (89 MoE layers) | GLM-4.7 (61 MoE layers) | Ratio |
|-----------|---------------------|--------------------------|-------|
| Attention | 45% * 92 = 41.4 | 45% * 64 = 28.8 | 0.70x |
| Sparse MoE | 40% * 89 = 35.6 | 40% * 1.67 * 61 = 40.7 | 1.14x |
| Router+Shared | 10% * 89 = 8.9 | 10% * 1.1 * 61 = 6.7 | 0.75x |
| Collectives | 5% * 89 = 4.5 | 5% * 61 = 3.1 | 0.69x |
| Dense layers | 3 layers | 3 layers | 1.0x |
| Norm+LM head | 1x | 1x | 1.0x |

Normalized total: REAP = 100%, GLM-4.7 = ~82%

**Estimated GLM-4.7 performance with SEED-3 BF4 on Galaxy WH:**
- bs=32: 142.7 / 0.82 = **~174 tok/s aggregate**
- bs=1: 4.5 / 0.82 = **~5.5 tok/s** (~135ms ITL)

### 3.6 FP8 Feasibility Summary

| Aspect | Status | Effort |
|--------|--------|--------|
| Memory fits | YES (6.53 GB/device, 20.4% of 32 GB) | None |
| Code compatible | YES (fully parameterized) | None |
| Config changes | Point to GLM-4.7 config.json | 1 line |
| 160 experts | Works automatically (160%32=0) | None |
| FP8 dequant | NOT implemented, needs ~50 LOC | Low |
| Weight loading time | ~12 min (vs ~10 min for REAP) | Acceptable |
| Expected perf | ~174 tok/s bs=32, ~5.5 tok/s bs=1 | -- |

**Bottom line**: GLM-4.7 can run on the existing `glm4_moe` codebase with zero code changes if using a BF16 source model. For FP8 source, add host-side dequantization (~50 LOC, copy from DSv3). Expected performance is ~20% better than REAP thanks to 30% fewer layers partially offset by 67% more experts per layer.

---

## Appendix A: File Reference

| File | Path | Key Lines |
|------|------|-----------|
| Model runner | `tt-metal/models/demos/glm4_moe/tt/model_tt.py` | 976 (trace), 1054 (capture), 1235 (update inputs) |
| Decoder layer | `tt-metal/models/demos/glm4_moe/tt/decoder_layer_tt.py` | 211 (decode fwd), 418 (MoE fwd), 480 (fuse reduce) |
| Attention | `tt-metal/models/demos/glm4_moe/tt/attention_tt.py` | 558 (decode), 482 (partial RoPE), 629 (batch slice) |
| MoE | `tt-metal/models/demos/glm4_moe/tt/moe_tt.py` | 301 (topk), 399 (sparse fwd), 93 (grid config) |
| Layer weights | `tt-metal/models/demos/glm4_moe/tt/layer_weights.py` | 87 (dtype), 215 (experts), 323 (convert) |
| Config | `tt-metal/models/demos/glm4_moe/tt/config.py` | 10 (hparams), 66 (from_hf) |
| CCL | `tt-metal/models/demos/glm4_moe/tt/ccl.py` | 22 (semaphore mgmt), 205 (reset) |
| Env config | `docker_tt/dev/.env.glm47_reap` | Full current config |
| DSv3 dequant ref | `tt-metal/models/demos/deepseek_v3/utils/dequantize.py` | 12 (dequantize_tensor) |
| DSv3 FP8 loader ref | `tt-metal/models/demos/deepseek_v3/tt/mlp/mlp_dequant.py` | 17 (MLPDequant) |

## Appendix B: Proven Dead Ends (Do NOT Revisit)

These have been empirically tested and rejected:

| Optimization | Result | File Reference |
|-------------|--------|---------------|
| EP_L1=1 | Garbled output | TG mesh L1 incompatible with CCL |
| FUSE_SHARED_EP_REDUCE=1 | Hang/device corruption | CCL deadlock on 2D mesh |
| LoFi math fidelity | No speedup (-1.5%) | DRAM-BW-bound, not compute-bound |
| DRAM-sharded attn weights | +2.1% (not significant) | Per-device matmuls too small |
| Device-side sampling | +0% | Hidden behind trace replay |
| Fused gate+up sparse matmul | -14.6% regression | Clone overhead + break-even |
| bs=64 | 61.4 tok/s (WORSE) | MoE EP=32 doubles cost |
| Weight prefetching | N/A | Blackhole-only feature |
| Async CCL overlap | N/A | Single CQ in trace mode |

## Appendix C: Verified Findings (Codex Cross-Check)

The following findings were independently verified by Codex (GPT-5.4) against the codebase:

1. **Dead trace buffers**: `tokens_tt` and `trans_matrix_tt` are allocated but never consumed by the traced graph. Confirmed: `_update_trace_inputs` does NOT write to `tokens_tt`. (Codex verified model_tt.py:1235-1309)

2. **No FP8 dequant code**: Confirmed no `_dequant_weight`, `float8`, or `scale_inv` handling anywhere in `models/demos/glm4_moe/`. (Codex searched entire glm4_moe directory)

3. **160 experts works automatically**: `_experts_weight_tt` checks `160 % 32 == 0` (passes), `create_moe_runtime` computes `experts_per_device = 5`. All confirmed from layer_weights.py:236 and moe_tt.py:179-181.

4. **REAP config.json confirms**: `first_k_dense_replace=3` (NOT 1 as some docs claim), `intermediate_size=12288`, `n_routed_experts=96`, `num_hidden_layers=92`. (Codex read from HF cache)

5. **Per-layer collective count**: 6 collectives per MoE layer (2 attention + 1 shared TP + 3 EP reduce). Total: 3*3 + 89*6 = 543 collectives per decode step. (Codex traced through all forward paths)

6. **Shared expert uses fused gate+up**: For MoE layers (3-91), `w_mlp_gate_up` is created at layer_weights.py:532-552, used in `shared_expert_forward_tt` via the `w_gate_up is not None` branch at moe_tt.py:901-907. (Codex confirmed)
