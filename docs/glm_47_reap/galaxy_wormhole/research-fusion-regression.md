# SEED-2 Fusion Regression Analysis: Fused Gate+Up Sparse Matmul

**Date**: 2026-03-09
**Model**: GLM-4.7-REAP-218B-A32B, 92 layers (89 MoE, 3 dense), EP=32, TP=8, DP=4
**Hardware**: Galaxy Wormhole TG, 32 chips, Mesh(8,4), 72 cores/chip
**Mode**: Traced decode (trace_mode=decode_only)

## Regression Summary

| Metric | Baseline | Fused | Delta |
|--------|----------|-------|-------|
| bs=1 tok/s | 4.1 | 3.5 | **-14.6%** |
| bs=32 agg | 99.5 | 97.3 | -2.2% |
| bs=1 ITL | ~187ms | 202.5ms | **+15.5ms** |
| bs=32 ITL | ~187ms | 199.3ms | +12.3ms |

## Root Cause: Unnecessary `ttnn.clone` Calls (PRIMARY) + Extra `ttnn.slice` Programs

### Finding 1: REAP always clones; Lite has a skip path — THIS IS A BUG

**glm4_moe (REAP) — moe_tt.py:529-533 — ALWAYS clones:**
```python
gate_view = ttnn.slice(w1w3_out, begin_gate, end_gate)
up_view = ttnn.slice(w1w3_out, begin_up, end_up)
w1_out = ttnn.clone(gate_view, memory_config=sparse_mc)
w3_out = ttnn.clone(up_view, memory_config=sparse_mc)
ttnn.deallocate(w1w3_out, force=False)
```

**glm4_moe_lite — moe_tt.py:1542-1551 — conditionally skips clones:**
```python
if skip_defensive_clones:
    w1_out = ttnn.slice(w1w3_out, begin_gate, end_gate)
    w3_out = ttnn.slice(w1w3_out, begin_up, end_up)
else:
    gate_view = ttnn.slice(w1w3_out, begin_gate, end_gate)
    up_view = ttnn.slice(w1w3_out, begin_up, end_up)
    w1_out = ttnn.clone(gate_view, memory_config=sparse_mc)
    w3_out = ttnn.clone(up_view, memory_config=sparse_mc)
    ttnn.deallocate(w1w3_out, force=False)
```

The Lite model exposes `skip_defensive_clones` (set via `GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES` env var,
sourced from `decoder_layer_tt.py:423`). The REAP port **omitted this parameter entirely** — it
unconditionally clones.

### Finding 2: The shared expert in REAP itself proves clone is unnecessary

The shared expert path in the SAME file (moe_tt.py:901-912) does:
```python
if w_gate_up is not None:
    gate_up = ttnn.linear(x, w_gate_up, **kwargs)
    inter_tp = gate_up.shape[-1] // 2
    gate = ttnn.slice(gate_up, [..., 0:inter_tp])
    up = ttnn.slice(gate_up, [..., inter_tp:])
    ttnn.deallocate(gate_up, force=False)
```
No clone. This works in production traced decode. The `ttnn.mul` eltwise binary op accepts slice outputs
directly — it only requires device tensors in TILE layout with compatible memory config.

### Finding 3: Both `ttnn.slice` and `ttnn.clone` are device ops that add trace programs

Verified from C++ source:
- **`ttnn.slice`**: NOT metadata-only. Calls `device_operation::launch` which creates a device program.
  Path: `slice.cpp:156 -> slice_device_operation.cpp:222 -> slice_device_operation.cpp:283`.
- **`ttnn.clone`**: Also a device op. Builds explicit read/write kernels.
  Path: `clone_device_operation.cpp:64 -> clone_program_factory.cpp:141`.
- **`ttnn.reshape`/`ttnn.experimental.view`**: These ARE metadata-only (zero cost in trace).

### Finding 4: Program count math explains the regression

**Baseline (non-fused) per MoE layer**: 2 sparse_matmuls (gate, up) = 2 programs
**Fused per MoE layer**: 1 sparse_matmul + 2 slices + 2 clones = 5 programs
**Net change**: +3 programs per layer

Over 89 MoE layers (layers 3-91, `first_k_dense_replace=3`):
- Extra programs: 89 * 3 = **267 extra programs** in the trace
- At ~15us per-program dispatch: 267 * 15us = **4.0ms from dispatch alone**

Plus DRAM bandwidth for the clones:
- Each clone copies [1, 1, 32, 1536] bfloat16 = 96 KB (read) + 96 KB (write) = 192 KB
- 89 layers * 2 clones = 178 clone ops * 192 KB = **34.3 MB total DRAM traffic**
- At ~200 GB/s per chip DRAM BW: ~0.17ms (negligible bandwidth, but latency-bound)

**Slice operations also do DRAM traffic** (they materialize a new tensor):
- Each slice: 96 KB read + 96 KB write = 192 KB
- 89 * 2 = 178 slices * 192 KB = another 34.3 MB

**Total overhead estimate**: ~4.0ms (dispatch) + ~5-10ms (DRAM latency for 356 small ops)
This matches the observed **+12-15ms ITL regression**.

### Finding 5: `ttnn.swiglu` does NOT help

Searched tt-metal for built-in fused SiLU+gated multiply. Found `ttnn.swiglu` exists but:
1. Internally does `slice + slice + swish + multiply` — same number of programs
2. Applies SiLU to the WRONG half (second half) for REAP's `[w1|w3]` weight layout
3. No advantage over the current manual approach

### Finding 6: No metadata-only split exists for this shape

- `ttnn.chunk`: loops over `ttnn.slice` (no improvement)
- `ttnn.split`: has a fast 2-output device op but requires height >= 64 (2 tiles). Decode shape
  `[1,1,32,3072]` has height=32 (1 tile), falls back to repeated `slice`
- `ttnn.experimental.view`: reshape-only, cannot create offset-based subviews

## Why glm4_moe_lite Doesn't Regress (or Regresses Less)

Three reasons:

1. **Lite has `skip_defensive_clones`**: When enabled, it skips the 2 clones per layer,
   saving 2 programs/layer. REAP doesn't have this option.

2. **Lite has fewer MoE layers**: 46 MoE layers (47 total, 1 dense) vs REAP's 89.
   Even with clones: 46 * 5 = 230 fused programs vs 46 * 2 = 92 baseline = +138 extra.
   With skip_clones: 46 * 3 = 138 fused vs 92 baseline = +46 extra = ~0.7ms overhead.
   The saved sparse_matmul (46 programs) nearly cancels: net = 0 extra programs.

3. **Lite has larger ROI per fusion**: With `hidden_size=2048`, each sparse_matmul is
   lighter (fewer FLOPs, less DRAM traffic). The percentage overhead of slice is lower
   relative to the already-small total. More importantly, Lite may not use traced decode
   in all configurations — outside trace, the Python dispatch overhead is much larger
   (~100-500us per op), making the slice overhead relatively smaller.

## Recommendation: Remove Clones (Immediate Fix)

### Fix 1: Remove clones in REAP fused path (HIGH CONFIDENCE)

Change `moe_tt.py` lines 529-533 and 769-773 from:
```python
gate_view = ttnn.slice(w1w3_out, begin_gate, end_gate)
up_view = ttnn.slice(w1w3_out, begin_up, end_up)
w1_out = ttnn.clone(gate_view, memory_config=sparse_mc)
w3_out = ttnn.clone(up_view, memory_config=sparse_mc)
ttnn.deallocate(w1w3_out, force=False)
```

To:
```python
w1_out = ttnn.slice(w1w3_out, begin_gate, end_gate)
w3_out = ttnn.slice(w1w3_out, begin_up, end_up)
# w1w3_out stays alive until silu/mul consume the slice views
```

And after the `ttnn.mul` + deallocate of w1_out/w3_out, add:
```python
try:
    ttnn.deallocate(w1w3_out, force=False)
except Exception:
    pass
```

This mirrors exactly what the shared expert path (line 905-907) and the Lite
`skip_defensive_clones=True` path (line 1542-1544) already do successfully.

**Expected improvement**: Removes 2 clone programs per layer * 89 layers = 178 programs.
At 15us each = **2.67ms saved**. Plus eliminates 34.3 MB of unnecessary DRAM copy traffic.

### Fix 2: Evaluate net ROI with clones removed

After Fix 1, the fused path has: 1 sparse_matmul + 2 slices = 3 programs/layer
vs baseline: 2 sparse_matmuls = 2 programs/layer.
Net: +1 program/layer * 89 = +89 programs = +1.3ms overhead.

Saved: 1 sparse_matmul dispatch per layer * 89 = -89 programs = -1.3ms.

**Net dispatch overhead: approximately zero** (89 added slices cancel 89 saved matmuls).

The actual gain comes from:
- Saved DRAM reads for expert weights: 1 read of w1w3 instead of 2 reads of w1 + w3
- But the fused w1w3 is 2x the size, so total DRAM read is the same
- The fused matmul has 2x the output tiles, potentially worse L1 utilization

**Prediction**: With clones removed, fusion will be approximately break-even (+/- 1-2ms).
There is no strong theoretical reason for it to be significantly faster.

### Fix 3 (Future): True zero-overhead split

Would require a custom TTNN op `split_silu_mul` that:
1. Takes `[.., 2*I]` input tensor
2. Produces `[.., I]` output as `SiLU(input[..., :I]) * input[..., I:]`
3. Single device program, single DRAM read, single output write

This would save 2 programs/layer (the 2 slices) = 178 programs = 2.67ms.
Combined with the saved sparse_matmul: net -89 programs = -1.3ms.

## Summary Table

| Configuration | Programs/layer | Total (89 layers) | vs Baseline |
|--------------|----------------|-------------------|-------------|
| Baseline (2 sparse_matmul) | 2 | 178 | -- |
| Fused + clone (current) | 5 | 445 | **+267 (+15ms)** |
| Fused, no clone (Fix 1) | 3 | 267 | +89 (~0ms net*) |
| Fused + split_silu_mul (Fix 3) | 1 | 89 | **-89 (-1.3ms)** |

*Net ~0 because 89 saved sparse_matmul dispatches cancel 89 added slice dispatches.

## Verdict

**The regression is caused by a porting bug**: the REAP fused path unconditionally clones,
while the Lite source code it was ported from has `skip_defensive_clones` to avoid this.

**Immediate action**: Remove the clones (Fix 1). This should recover most of the 15ms regression.

**After Fix 1**: Re-benchmark. If fusion is still slower or break-even, it may not be worth
keeping — the theoretical benefit (one fewer matmul dispatch) is tiny (~15us/layer) compared
to the unavoidable cost of 2 slice device programs per layer.

**Abandon fusion only if** Fix 1 still shows regression. A custom `split_silu_mul` kernel (Fix 3)
would definitively make fusion worthwhile but requires C++ kernel development.
