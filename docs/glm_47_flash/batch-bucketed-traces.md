# Batch-Bucketed Traces: Design & Results

Created: 2026-02-13
Implemented: 2026-02-14
Status: **IMPLEMENTED** — committed to glm47_flash branch

## Summary

Capture decode traces at multiple batch buckets (B=1, 4, 8, 16, 32). At runtime, pad to
the nearest bucket instead of MAX_NUM_SEQS. This gives B=1 requests 16 FlashMLA cores
instead of 2.

## Measured Results (combined with Section 115 fix)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Decode bs=1 tok/s | 4.34 | **6.97** | **+60%** |
| Decode bs=1 ITL | 229ms | **143ms** | **-37%** |
| Decode bs=32 agg tok/s | 129 | **208.3** | **+61%** |
| Prefill 1k bs=1 tok/s | 197 | **205** | +4% |

Artifact: `plan/glm47_flash/artifacts/bench_decode_1771082938.json`

## Problem

With MAX_NUM_SEQS=32, the trace is captured at B=32 shapes. FlashMLA allocates 2 cores/seq
(64 total / 32 seqs). When only 1 user is active, 30 slots are padded — their cores sit idle.

Combined with the Section 115 bug (`tokens > 1` causing dense MoE in decode), this made
decode at bs=32 run 16x more MoE compute than necessary.

## Solution

Two changes:

1. **Batch-bucketed traces**: Capture traces at B=1,4,8,16,32. At runtime, pad to nearest
   bucket. B=1 requests get 16 FlashMLA cores.

2. **Section 115 fix**: Change `use_dense_prefill = tokens > 1` to `tokens >= 33` at
   decoder_layer_tt.py:1174. Decode (tokens≤32) uses sparse MoE (4 experts), prefill
   (tokens≥33) uses dense MoE (all 64 experts, stable).

## Files Changed

### model_tt.py
- Added `_DecodeTraceSamplingState` dataclass with all per-bucket trace fields
- Replaced 16 single-trace fields with `self._decode_trace_states: dict[int, State]`
- `_get_or_create_trace_state(batch, page_table_width)` — allocates per-bucket state
- All trace methods refactored to accept `state` parameter
- `_release_all_decode_traces()` iterates ALL bucket states

### tt_model_runner.py
- Parse `decode_trace_batch_buckets` from `override_tt_config`
- Compute `_decode_pad_target` — nearest bucket for decode, max_num_seqs for prefill
- Pad all decode tensors to bucket target
- `reset_batch` handles variable tensor sizes when bucket changes
- `warmup_model()` loops over all bucket sizes to pre-capture traces

### decoder_layer_tt.py
- Line 1174: `tokens > 1` → `tokens >= 33` (Section 115 fix)

### .env.glm47
- `trace_region_size=250000000` (250MB for 5 bucket traces — sparse MoE traces are ~70MB at B=16)
- `decode_trace_batch_buckets=[1,4,8,16,32]` in OVERRIDE_TT_CONFIG

## Memory Budget

- trace_region_size: 250MB (5 bucket traces, sparse MoE traces larger than dense)
- Per-device on T3K: 8 × 250MB = 2GB total trace region
- Warmup time: ~3.5 min (90s extra for 4 additional trace captures)

## Notes

- DeepSeek V3 does NOT implement batch-bucketed traces — this is new capability
- 60MB trace_region_size was insufficient (TT_FATAL at B=16 with sparse MoE trace ~70.9MB)
- First request to a new bucket has ~6s latency spike if not pre-warmed during startup

---

## V1 Engine Port (2026-02-17)

The V0 implementation above was specific to `model_tt.py` and V0's `tt_model_runner.py`. When we switched to VLLM_USE_V1=1, the V1 engine at `vllm/v1/worker/tt_model_runner.py` did NOT read the `decode_trace_batch_buckets` config — it always padded decode batches to `max_num_seqs=32`.

### V1 Changes (commit e7374bd)

Three changes to `vllm/v1/worker/tt_model_runner.py`:

1. **Constructor (lines 127-140)**: Read `decode_trace_batch_buckets` from `override_tt_config`. Auto-append `max_num_seqs` if not present. Store as sorted list.

2. **Decode padding (lines 519-526)**: Replace hardcoded `input_batch.max_num_reqs` with nearest-bucket lookup. For each decode step, find the smallest bucket >= current batch size.

3. **Warmup (lines 1559-1566)**: Loop over all bucket sizes instead of single warmup at `max_num_seqs`. Each bucket gets its own traced decode.

### V1 Results (gen=500, warm device)

| Metric | V1 no bucketing | V1 with bucketing | Delta |
|--------|----------------|-------------------|-------|
| bs=1 ITL | 150ms | 140.1ms | -9.9ms (6.6%) |
| bs=1 tok/s | 6.5 | 7.0 | +0.5 (7.7%) |
| bs=32 agg tok/s | 212.8 | 248.6 | +35.8 (16.8%) |

The bs=32 improvement was unexpected — per-bucket trace capture may produce better-optimized traces.
