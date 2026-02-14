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
