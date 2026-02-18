# GLM-4.7-Flash Performance Optimization — Summary

Last updated: 2026-02-18

For the full optimization log (5000+ lines), see: `plan/glm47_flash/perf-opt.md`

## Current Best Performance

| Batch | tok/s | ITL (ms) | Target | Status |
|-------|-------|----------|--------|--------|
| 1 | 7.0 | 140 | 30 tok/s | 4.3x gap remaining |
| 32 | 248.6 (aggregate) | — | 140 tok/s | **EXCEEDED by 77%** |

Engine: VLLM_USE_V1=1, trace_mode=decode_only, T3K (8 Wormhole chips)

## Key Optimizations Applied

| Optimization | Impact | Status |
|-------------|--------|--------|
| Batch-bucketed traces (V1) | +7.7% bs=1, +16.8% bs=32 | Committed |
| Sparse MoE decode (2-expert routing) | ~4x speedup vs dense | Committed |
| L1 decode activations | Reduces DRAM traffic | Committed |
| Tensor parallel (TP=8) | 8-way sharding | Committed |
| BF8 KV cache | 2x cache capacity | Committed |
| Fused QKV_A projection | 1 matmul instead of 3 | Committed |
| Fused KV branch kernel (C++) | ~2-5ms savings/iter | In progress |
| Section 115 fix (dense MoE threshold) | Prevents 16x MoE waste in decode | Committed |

## Software Optimization Ceiling

All Python/config-level optimizations are exhausted. The 140ms decode ITL at bs=1 is dominated by:
- 660 kernel dispatches × ~210μs fixed overhead = 106ms
- Remaining: DRAM reads for weights + compute

Reaching 30 tok/s (33ms ITL) requires C++ fused kernels to collapse multiple operations into single dispatches.

## Next Steps

1. **Wormhole RMSNorm kernel** — Custom C++ kernel replacing Blackhole-only LLK APIs (in progress)
2. **KV Cache Branch fusion** — Fuse DKV matmul + gather + RMSNorm + RoPE into 1 dispatch
3. **Pre-SDPA fusion** — Fuse Q-path projections
4. **Post-SDPA fusion** — Fuse output projection + residual + layernorm

## Architecture

- Model: zai-org/GLM-4.7-Flash (GLM-4, MoE, 47 layers, 64 experts per layer)
- Hardware: T3K (8 x Wormhole 8x8 cores, 12GB DRAM each)
- Config: `dev/.env.glm47` with all optimization flags
- Code: `tt-metal/models/demos/glm4_moe_lite/tt/`
