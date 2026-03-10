# BF4 Deep Analysis (Codex 5.4) for GLM-4.7-REAP-218B on Galaxy Wormhole

**Date**: 2026-03-09
**Researcher**: Claude Opus 4.6 (codex-bf4-researcher, READ-ONLY)
**Tool**: Codex GPT-5.4 deep analysis + local source verification
**Model**: GLM-4.7-REAP-218B-A32B (92 MoE layers, 96 experts top-8, hidden=5120, TP=8, EP=32)
**Hardware**: Galaxy Wormhole 32 chips, 12 GB DRAM/chip, Mesh(8,4)
**Current**: 99.5 tok/s aggregate bs=32 with BF8 all weights

---

## 1. BFP4_b Format in tt-metal (Verified from C++ Source)

### Format Definition
Source: `tt_metal/impl/data_format/bfloat4.cpp`, `tt_metal/impl/data_format/blockfloat_common.cpp`

- **Block Floating Point**: 1 shared 8-bit exponent per 16 elements
- **Per-element**: 1 sign bit + 3 mantissa bits = 4 bits/element
- **`_b` suffix**: Uses `ExpPrecision::B` (full 8-bit unbiased exponent), vs `_a` (5-bit rebias-to-15 path)
  - Confirmed at `jit_build/data_format.cpp:25`: `Bfp4 <-> Bfp4_b` mapping, line 43 treats `Bfp4_b` as `exp_b`

### Tile Size (Verified from `tt_backend_api_types.hpp:92`)
```
BFP4_b: (128 * 4) + (16 * 4) = 576 bytes per 32x32 tile (1024 elements)
BFP8_b: (256 * 4) + (16 * 4) = 1088 bytes per 32x32 tile
BF16:   1024 * 2              = 2048 bytes per 32x32 tile
```
- **BFP4 is 53% the size of BFP8** (576/1088 = 0.529), saving ~47% per weight tensor
- NOT exactly 2x smaller due to shared exponent overhead (64 bytes fixed per tile)

### Effective Precision
From `blockfloat_common.cpp:161,240`:
- `sign = data >> 3`, `man = data & 0x7`, `MANTISSA_BFP_WIDTH = 3` for Bfp4/Bfp4_b
- 3 mantissa bits means nonzero representable levels within a block are:
  `0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75` times the shared scale
- Values > ~2 exponent steps below the block max collapse toward zero
- **Rounding**: round-to-nearest-even (from `blockfloat_common.cpp:295`), NOT truncation
- **Denormals flushed to zero**: When mantissa shift count > exponent (verified in bfloat4.cpp unpack loop)

### Matmul Handling
- BFP4 weights are stored in DRAM in the packed tile format
- During matmul, the Tensix FPU unpacks BFP4 to full-width internal format before compute
- **No separate "BFP4 compute mode"** — all compute happens at full internal precision
- The math fidelity (HiFi4/HiFi2/LoFi) controls the accumulation precision, NOT the unpack precision
- BFP4 only saves DRAM bandwidth and DRAM storage, NOT compute

---

## 2. DeepSeek V3's Mixed Precision Strategy (Verified)

### Expert Weight Dtypes (from `models/demos/deepseek_v3/tt/experts.py:76`)
```python
dtype=ttnn.bfloat8_b if hf_name == "up_proj" else ttnn.bfloat4_b
```

| Weight | HF Name | TT Dtype | Rationale |
|--------|---------|----------|-----------|
| w1 (gate_proj) | gate_proj | **BFP4_b** | Gated by SiLU — errors damped by nonlinearity |
| w2 (down_proj) | down_proj | **BFP4_b** | Output projection, errors averaged across hidden dim |
| w3 (up_proj) | up_proj | **BFP8_b** | Element-wise multiply `SiLU(w1*x) * (w3*x)` — precision critical |

### Dense MLP (from `models/demos/deepseek_v3/tt/mlp/mlp.py:54`)
```python
WEIGHT_DTYPE = ttnn.bfloat4_b  # ALL dense MLP weights use BFP4
```

### LM Head (from `deepseek_v3/tt/lm_head.py:95`)
- Also uses `ttnn.bfloat4_b` — even the final vocabulary projection

### Math Fidelity with BF4
From `deepseek_v3/utils/config_helpers.py:28`: Expert kernels use **LoFi** config.
DSv3 fused-op tests accept **PCC ~0.97** for BF4-weighted MLP ops (test_ds_ff1_3.py:321, test_ds_ff2.py:427).

### Why DSv3 Tolerates BFP4
1. **Trained with FP8 block quantization awareness** — weight distributions already favorable for block formats
2. **Only 61 MoE layers** (vs REAP's 92) — less error accumulation
3. **Larger hidden dim (7168)** — more elements per projection, better statistical averaging
4. **Mixed precision strategy** — `up_proj` kept at BF8 to protect the critical multiplicative branch

---

## 3. Why GLM-4.7-Flash BF4 Failed (Context for REAP)

### Flash Config (verified from HF config.json)
- hidden=2048, moe_intermediate=1536, 64 experts, top-4, 47 MoE layers
- MLA attention (NOT standard GQA) — more precision-sensitive
- **Flash used BF4 for ALL weights (dense + experts)** — no mixed precision

### Key Differences from DSv3 Success
1. **Blanket BF4** vs DSv3's selective mixed precision
2. **Smaller hidden dim** (2048 vs 7168) — proportionally larger quantization errors
3. **Not trained for quantization** — no FP8-aware training like DSv3
4. **MLA attention** — absorbed latent attention is more precision-sensitive than standard GQA

### REAP vs Flash vs DSv3 Comparison
| Parameter | Flash | REAP-218B | DSv3 |
|-----------|-------|-----------|------|
| hidden_size | 2048 | 5120 | 7168 |
| num_experts | 64 | 96 | 256 |
| top_k | 4 | 8 | 8 |
| MoE layers | 47 | 92 | 61 |
| moe_intermediate | 1536 | 1536 | 2048 |
| Attention | MLA | GQA | MLA |
| BF4 result | GARBLED | GARBLED | WORKS (mixed) |

---

## 4. DRAM Bandwidth Math for REAP-218B BF4

### Codex 5.4 Analysis: Weight Memory Per Device

Using exact tile sizes and REAP-218B dimensions (hidden=5120, moe_intermediate=1536, 96 experts, EP=32, TP=8):

**Current BF8 baseline**: ~8.30 GiB/device total weight residency
- **Routed experts**: ~6.23 GiB/device (dominates — 75% of total)
- Attention + dense/shared MLP: ~2.07 GiB/device

### BF4 Savings Scenarios

| Strategy | Expert w1 | Expert w2 | Expert w3 | Weight/device | Savings | Throughput Ceiling |
|----------|-----------|-----------|-----------|---------------|---------|-------------------|
| **Current BF8 all** | BF8 | BF8 | BF8 | 8.30 GiB | baseline | 99.5 tok/s |
| **DSv3-like mixed** | BF4 | BF4 | BF8 | 6.34 GiB | 1.96 GiB (23.6%) | ~1.31x = ~130 tok/s |
| **All experts BF4** | BF4 | BF4 | BF4 | 5.37 GiB | 2.93 GiB (35.3%) | ~1.55x = ~154 tok/s |
| **All weights BF4** | BF4 | BF4 | BF4 | 4.44 GiB | 3.86 GiB (46.5%) | ~1.87x = ~186 tok/s |

**These are DRAM-bandwidth upper bounds** — the decode path is confirmed DRAM-bound.

The "throughput ceiling" assumes linear scaling with DRAM read reduction, which is an upper bound.
Real improvement will be less due to:
- CCL all_reduce latency (not BW-bound)
- Host sampling overhead (~15ms fixed)
- Non-weight DRAM reads (activations, KV cache)

---

## 5. What Must Change in REAP Code for BF4 Support

### Current State (from `models/demos/glm4_moe/tt/layer_weights.py`)

**`_env_experts_dtype()` (line 68-82)**: Only supports BF8, BF16, F32 — **NO BF4 option**
```python
if override in {"bf8", "bfloat8_b"}:
    return ttnn.bfloat8_b
if override in {"bf16", "bfloat16"}:
    return ttnn.bfloat16
if override in {"f32", "fp32", "float32"}:
    return ttnn.float32
raise ValueError(...)
```

**`_env_dense_dtype()` (line 85-99)**: Same — no BF4

**Expert weight loading (line 584-642)**: Uses single `experts_dtype` for ALL projections:
```python
experts_dtype = _env_experts_dtype()
# ...
w1_experts = _experts_weight_tt(..., dtype=experts_dtype)  # gate_proj
w3_experts = _experts_weight_tt(..., dtype=experts_dtype)  # up_proj
w2_experts = _experts_weight_tt(..., dtype=experts_dtype)  # down_proj
```

**Router gate (line 553-559)**: Hardcoded BF16 (GOOD — must stay BF16)

### Required Changes for DSv3-Style Experiment

1. **Add BF4 to `_env_experts_dtype()`**: Add `if override in {"bf4", "bfloat4_b"}: return ttnn.bfloat4_b`
2. **Add per-projection dtype support**: New env vars `GLM4_MOE_EXPERTS_W1_DTYPE`, `GLM4_MOE_EXPERTS_W3_DTYPE`, `GLM4_MOE_EXPERTS_W2_DTYPE` to override individual projections
3. **Keep router BF16**: Already hardcoded, no change needed
4. **Weight cache invalidation**: BF4 weights have different tile sizes, so cached BF8 weights will NOT work — must rebuild cache with `variants` suffix change

### Minimal Patch (add BF4 support without per-projection):
```python
# In _env_experts_dtype():
if override in {"bf4", "bfloat4_b"}:
    return ttnn.bfloat4_b
```
This would switch ALL expert projections (w1/w2/w3) to BF4 — riskier than DSv3 mixed approach.

---

## 6. Risk Assessment and Recommendations

### BF4 Risk Factors for REAP-218B

**Favorable (vs Flash)**:
- Larger hidden (5120 vs 2048) — 2.5x more elements per projection, better averaging
- Standard GQA (not MLA) — less precision-sensitive attention
- Top-8 routing (vs top-4 Flash) — more experts averaging dampens per-expert errors

**Unfavorable (vs DSv3)**:
- **92 MoE layers** (vs 61 DSv3) — 50% more layers for error accumulation
- **NOT trained with FP8 quantization awareness** — weight distributions likely have worse outliers
- **Smaller hidden** (5120 vs 7168) — less statistical averaging per projection
- **Already listed as "Dead End" in worklog** — previous blanket BF4 test → garbled output

**Critical risks**:
- Router gate precision: BF16 (safe, hardcoded)
- Attention QKV/O: BF4 is HIGH RISK — 96 Q-heads with GQA, softmax amplifies errors
- Shared expert: BF4 is MEDIUM RISK — runs on every token, errors are systematic
- Routed experts: BF4 is LOWER RISK — top-8 aggregation averages some error

### Codex 5.4 Recommended Experiment Order

**Experiment 1 (safest — DSv3 pattern)**:
- Attention: BF8 (no change)
- Router gate + correction bias: BF16 (no change)
- Shared expert: BF8 (no change)
- Routed expert w1 (gate_proj): **BF4**
- Routed expert w2 (down_proj): **BF4**
- Routed expert w3 (up_proj): **BF8** (keep — multiplicative branch)

Expected savings: ~1.96 GiB/device (23.6%)
Expected throughput ceiling: ~130 tok/s aggregate bs=32

**Experiment 2 (if Exp 1 clean)**:
- All routed expert projections: BF4
Expected savings: ~2.93 GiB/device (35.3%)

**NOT recommended**:
- BF4 for attention weights (Q/K/V/O)
- BF4 for shared expert weights
- BF4 for router gate
- Blanket BF4 for all weights (already failed)
- Changing math fidelity in same run as dtype change

### Math Fidelity Considerations
- Current: `GLM4_MOE_ATTN_FIDELITY=hifi` (from env), attention uses HiFi2 (from code)
- Current: `GLM4_MOE_MOE_SPARSE_FIDELITY=hifi` (from env)
- **DSv3 uses LoFi for BF4 expert matmuls** — this is safe because BF4 precision is already the bottleneck, not accumulation precision
- For REAP: keep HiFi for attention, could try LoFi for BF4 expert matmuls (secondary experiment)

---

## 7. Key Finding: Previous "Dead End" Was Blanket BF4

The worklog at `WORKLOG_GLM47_REAP_GALAXY_WORMHOLE.md:122` says:
```
- **BF4** → garbled output (error accumulation over 92 layers)
```

But this was **blanket BF4** (all weights including attention). A DSv3-style **selective BF4** (experts only, w3 kept at BF8) has NOT been tested and is a different experiment with significantly lower risk. The previous failure does not definitively rule out selective BF4.

---

## 8. Implementation Notes

### Code Changes Required (layer_weights.py)

The minimum viable change is 3 parts:

1. **Add BF4 to dtype parser** (2 lines each in `_env_experts_dtype` and `_env_dense_dtype`)
2. **Add per-projection expert dtype env vars** (new function + wire into weight loading)
3. **Update weight cache variant string** to include dtype info (prevent cache collisions)

### Weight Cache Rebuild
- BF4 tile size (576 bytes) differs from BF8 (1088 bytes)
- Cached weights at BF8 CANNOT be reinterpreted as BF4 — must regenerate
- The `experts_variant` string at line 589 (`f"ep{num_devices}_v1"`) should include dtype
- Expect ~10 min rebuild for 92 layers x 96 experts on Galaxy

### Testing Strategy
1. Short prompt (32 tokens) — check for coherent output
2. Medium prompt (256 tokens) — check for degradation over sequence length
3. Compare PCC of final logits: BF8 reference vs BF4 experiment
4. Run the standard benchmark matrix if output is coherent

---

## Sources

- `tt_metal/impl/data_format/bfloat4.cpp` — BFP4 pack/unpack
- `tt_metal/impl/data_format/blockfloat_common.cpp` — Shared BFP format, mantissa width
- `tt_metal/api/tt-metalium/tt_backend_api_types.hpp:29,57,92` — Format enum, tile sizes
- `tt_metal/api/tt-metalium/constants.hpp:20-21` — BFLOAT4_B_TILE_HW constant
- `models/demos/deepseek_v3/tt/experts.py:76` — DSv3 mixed BF4/BF8 strategy
- `models/demos/deepseek_v3/tt/mlp/mlp.py:54` — DSv3 dense MLP BF4
- `models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff2.py:427` — PCC ~0.97 for BF4
- `models/demos/glm4_moe/tt/layer_weights.py:68-99,584-642` — REAP dtype handling
- `models/demos/glm4_moe_lite/tt/layer_weights.py:77-78,99-100` — Flash BF4 support
- `docker_tt/WORKLOG_GLM47_REAP_GALAXY_WORMHOLE.md:122` — Previous BF4 dead end (blanket)
- `jit_build/data_format.cpp:25,43` — Bfp4_b exp_b path confirmation
- `tech_reports/data_formats/data_formats.md` — Format documentation
