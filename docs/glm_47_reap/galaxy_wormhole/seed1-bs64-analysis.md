# SEED-1: bs=64 Scope Analysis for GLM-4.7-REAP-218B on Galaxy Wormhole

**Date**: 2026-03-09
**Model**: GLM-4.7-REAP-218B (92 layers, 96Q/8KV GQA, 96 routed experts EP=32)
**Target**: Galaxy Wormhole Mesh(8,4), TP=8, EP=32, DP=4

## PRIMARY QUESTION: Does REAP call nlp_create_qkv_heads_decode?

**YES** -- `attention_tt.py:643`:
```python
q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv,
    num_heads=self.n_local_heads,  # 12
    num_kv_heads=self.n_local_kv_heads,  # 1
    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
)
```

However, **this is NOT a blocker** for bs=64 on TG Galaxy because of the DP=4 split:
- With bs=64, the attention DP-slices the batch: `logical_batch_after_slice = 64 // 4 = 16`
- `nlp_create_qkv_heads_decode` only sees batch=16 (padded to physical 32 tiles), well within the limit

---

## COMPLETE LIST OF bs=64 BLOCKERS

### BLOCKER 1: `nlp_create_qkv_heads_decode` num_users_supported=32
- **File**: `tt-metal/ttnn/.../nlp_create_qkv_heads_decode_device_operation.cpp:45-51`
- **Constraint**: `TT_FATAL(num_users <= 32, "Unsupported input shape")`
- **Impact on REAP bs=64**: **NOT a blocker** -- TG DP=4 means each device sees batch=16 after slicing
- **Would block**: Non-TG deployments or DP=1 configs trying bs>32
- **Fix type**: C++ kernel fix (change `num_users_supported` to 64, may need kernel tile-loop changes)

### BLOCKER 2: `nlp_concat_heads_decode` batch <= 32 validation
- **File**: `tt-metal/ttnn/.../nlp_concat_heads_decode_device_operation.cpp:39-40`
- **Constraint**: `TT_FATAL(input_shape[1] <= 32, "currently only support less than 32 users")`
  and `TT_FATAL(input_shape[2] == 32, "currently only support 32 padded heads")`
- **Impact on REAP bs=64**: **NOT a blocker** -- DP-sliced batch=16, padded heads=12 (padded to 32)
- **Fix type**: C++ kernel fix (generalize tile loop)

### BLOCKER 3: Attention `_DS_BATCH = 32` in DRAM-sharded configs
- **File**: `attention_tt.py:422`
- **Constraint**: `_DS_BATCH = 32` used for activation sharding and matmul program config M dimension
- **Impact on REAP bs=64**: **NOT a blocker IF DRAM_SHARD=0** (default). Blocker only if `GLM4_MOE_DRAM_SHARD=1`
- **Fix type**: Python-only (change `_DS_BATCH` to `batch_size_per_device_group` or dynamic)
- **Note**: With DP=4 slicing, per-group batch=16, tile-padded to 32 -- so _DS_BATCH=32 is actually correct. Only blocks if per-group batch > 32.

### BLOCKER 4: Physical reshape hardcoded to 32 in attention
- **File**: `attention_tt.py:636-640`
```python
xqkv = ttnn.reshape(
    xqkv,
    (1, 1, logical_batch_after_slice, int(_fqkv_shape[3])),
    (1, 1, 32, int(_fqkv_shape[3])),  # <-- hardcoded 32
)
```
- **Impact on REAP bs=64**: **NOT a blocker** -- `logical_batch_after_slice=16`, padded to 32 tiles. Correct.
- **Would block**: Per-group batch > 32 (i.e. total batch > 128 with DP=4)
- **Fix type**: Python-only (change `32` to `((logical_batch_after_slice + 31) // 32) * 32`)

### BLOCKER 5: MoE router L1 fast path `x.shape[2] <= 32`
- **File**: `moe_tt.py:307`
```python
use_l1 = int(x.shape[2]) <= 32  # decode mode only
```
- **Impact on REAP bs=64**: **Soft blocker** -- With bs=64, token dim is 64 (NOT DP-split for MoE).
  MoE receives the full batch (all tokens replicated to all 32 EP devices). So `x.shape[2] = 64 > 32`.
  This means L1 fast path is disabled, falling back to DRAM. Performance impact, NOT correctness.
- **Fix type**: Python-only (change threshold to 64 or remove)

### BLOCKER 6: SDPA decode core allocation
- **File**: `attention_tt.py:299-304` (SDPA program config) and `sdpa_decode_program_factory.cpp:202`
- **Constraint**: `TT_FATAL(num_cores_available >= B, ...)` where B = batch on this device
- **Impact on REAP bs=64**: **NOT a blocker** -- SDPA sees DP-sliced batch=16.
  SDPA grid is (8,8)=64 cores. 16 <= 64, plenty of cores for tree reduction.
- **Would block**: Per-device batch > 64 (i.e. total batch > 256 with DP=4)

### BLOCKER 7: `all_reduce(cluster_axis=1)` crash for batch>=2 (KNOWN ISSUE)
- **File**: Memory note: "V1 batch>1 BLOCKED: ttnn.all_reduce(cluster_axis=1) crashes for batch>=2"
- **Impact on REAP bs=64**: **POTENTIAL BLOCKER** -- but only if MoE uses `all_reduce(cluster_axis=1)`.
  Current MoE uses `GLM4_MOE_EP_REDUCE=full_ar` (no cluster_axis). Attention uses `cluster_axis=0` for TP reduce.
  The all_gather for DP reassembly uses `cluster_axis=1, dim=2` which is `all_gather` NOT `all_reduce`.
  **Verdict**: NOT a blocker for the current code paths.

### BLOCKER 8: Tile dimension question (M=1 tile vs M=2 tiles)
- **Impact**: With DP=4, per-device batch=16, tile-padded to 32 = 1 tile row. No change from bs=32.
  The MoE path sees full batch=64 = 2 tile rows. But MoE already handles variable token counts (prefill).
  `sparse_matmul` with `num_blocks = total_tokens // block` handles multi-block correctly.
- **Fix type**: None needed. Existing code handles arbitrary token counts.

### BLOCKER 9: Trace capture buffer sizing
- **File**: `model_tt.py:996-1021` (trace capture/replay)
- **Impact**: Trace captures fixed-size buffers for the given batch. bs=64 creates larger intermediate
  tensors (2x for non-DP-split paths). May exceed trace region if `trace_region_size` is set too small.
  The code already handles this gracefully with fallback to eager on `trace_region_size` errors.
- **Fix type**: Configuration (increase `trace_region_size` or set to 0 for auto)

### BLOCKER 10: KV cache page table sizing
- **File**: `generator_vllm.py:133-213` and `tt_worker.py:441`
- **Impact**: KV cache allocated per `num_blocks` from vLLM. Page table shape is `[batch, W]`.
  With bs=64, page table has 64 rows. Each DP group gets 16 rows (sharded).
  `paged_update_cache` and `paged_fill_cache` handle arbitrary batch via page_table.
- **Fix type**: None needed -- fully dynamic.

---

## SUMMARY TABLE

| # | Component | File | Fix Type | Blocks bs=64 on Galaxy TG? | Effort |
|---|-----------|------|----------|---------------------------|--------|
| 1 | nlp_create_qkv_heads_decode | C++ device op | C++ | NO (DP=4 gives 16/device) | N/A |
| 2 | nlp_concat_heads_decode | C++ device op | C++ | NO (DP=4 gives 16/device) | N/A |
| 3 | DRAM-sharded _DS_BATCH=32 | attention_tt.py:422 | Python | NO (default DRAM_SHARD=0) | Low |
| 4 | Physical reshape hardcoded 32 | attention_tt.py:639 | Python | NO (pad(16)=32) | Low |
| 5 | MoE L1 fast path <= 32 | moe_tt.py:307 | Python | SOFT (perf, not correctness) | Trivial |
| 6 | SDPA core allocation | sdpa_decode_program_factory.cpp | C++ | NO (16 < 64 cores) | N/A |
| 7 | all_reduce(axis=1) batch crash | CCL | C++ | NO (not used in bs=64 path) | N/A |
| 8 | Tile M=2 assumption | Various | None | NO (MoE handles multi-block) | None |
| 9 | Trace buffer sizing | model_tt.py | Config | SOFT (may need larger region) | Trivial |
| 10 | KV cache page table | generator_vllm.py | None | NO (fully dynamic) | None |

---

## KEY FINDING

**bs=64 on Galaxy TG (DP=4) requires NO C++ kernel changes.** The DP=4 split reduces per-device batch to 16, well within all existing kernel limits (32).

The only required changes are:
1. **MoE L1 threshold** (`moe_tt.py:307`): Change `<= 32` to `<= 64` -- Python, trivial, performance-only
2. **Trace region size**: May need tuning via config -- no code change

All other components handle bs=64 correctly through the existing DP sharding mechanism.

---

## ESTIMATED TOTAL EFFORT

- **Python changes**: ~2 lines (MoE threshold)
- **Configuration**: Update `--max-num-seqs=64` in vLLM launch, possibly `trace_region_size`
- **C++ changes**: NONE required
- **Testing**: Run benchmark matrix with bs=64 to validate correctness + measure throughput
- **Total**: ~30 minutes implementation + 1-2 hours testing

---

## RECOMMENDED IMPLEMENTATION ORDER

1. Set `--max-num-seqs=64` (or env equivalent) in vLLM launch config
2. Update `moe_tt.py:307` L1 threshold from 32 to 64
3. Run eager decode (no trace) with bs=64 to verify correctness
4. Enable trace mode and verify trace capture succeeds
5. Run full benchmark matrix at bs=64
6. If DRAM_SHARD=1 is desired, update `_DS_BATCH` to be dynamic (currently not needed)

---

## CAVEATS

1. **Memory**: bs=64 doubles activation memory vs bs=32. With 218B model (8.51 GB weights/device),
   there's ~3.49 GB free. Double batch = ~2x activation memory. May be tight for long contexts.
2. **MoE all_reduce**: `full_ar` all_reduce across all 32 devices with 2 tile rows (64 tokens)
   instead of 1. Should work but is untested at this size.
3. **Host-side sampling**: `_host_argmax_from_trace_logits` iterates over batch. 64 iterations
   instead of 32 -- trivial overhead but slightly slower host-side processing.
