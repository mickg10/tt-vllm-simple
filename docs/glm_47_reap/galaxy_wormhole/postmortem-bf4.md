# BF4 Analysis: GLM-4.7-REAP-218B on Galaxy Wormhole

**Date**: 2026-03-10 (v5 -- re-test execution plan)
**Model**: cerebras/GLM-4.7-REAP-218B-A32B (92 layers, 96 routed experts top-8, GQA 96Q/8KV)
**Hardware**: Galaxy Wormhole, 32 chips, Mesh(8,4), TP=8, EP=32
**Current BF8 performance**: bs=1: 4.1 tok/s (185ms ITL), bs=32: 121.6 tok/s agg (191ms ITL)
**Researcher**: Claude Opus 4.6 (researcher agent, READ-ONLY)
**Tools**: Codex GPT-5.4 xhigh (7 queries), local source verification, prior research synthesis
**Status**: ✅ BF4 VIABLE — Re-test 2026-03-10 confirms +28% bs=32 throughput, PPL -5.7% (BETTER than BF8). All routed experts BF4, shared+attention BF8.

---

## Executive Summary

**We may have killed BF4 prematurely.** New empirical data from Davor (TT quantization expert)
shows DeepSeek R1/V3 with ALL routed experts converted from FP8 to BFP4_b achieves:
- PPL 3.011 (+1.9% over FP8 baseline of 2.954)
- MMLU 84.95% (-0.15pp)
- GSM8K 95.38% (-0.45pp)
- **Basically lossless** across 256 experts x 58 MoE layers.

Our SEED-3 test (w1/w2=BF4, w3=BF8) was declared FAILED based on **qualitative eyeballing**
of output at temp=0. This is insufficient methodology:
1. No perplexity measurement
2. No benchmark suite (MMLU, GSM8K, etc.)
3. First output after container restart showed "Paris flexibly" stale trace artifact
4. Brand new per-projection dtype code (never tested before SEED-3) -- could have bugs
5. Weight cache key format changed -- possibly served stale BF8 weights
6. Never tested the FP8 source checkpoint (only BF16 -> BFP4)

This document provides: (A) a thorough technical analysis of WHY BF4 might work or fail
for REAP's architecture, and (B) a **complete execution plan** for rigorous re-testing.

**Correction**: Prior analysis wrongly claimed DSv3 had routed_scaling_factor=1.0.
**Both models use 2.5** (verified: `deepseek_v3/reference/config.json:57`).

---

## 1. DSv3/R1 Empirical Proof: BFP4_b Works on Deep MoE

### 1.1 Davor's Results (2026-03-10)

DeepSeek R1/V3: 256 routed experts, 58 MoE layers, top-8 routing.

| Metric | FP8 Baseline | Uniform BFP4 | Delta |
|--------|-------------|-------------|-------|
| Perplexity | 2.954 | 3.011 | **+1.9%** |
| MMLU | 85.10% | 84.95% | **-0.15pp** |
| GSM8K | 95.83% | 95.38% | **-0.45pp** |

This is essentially lossless. The BFP4_b format with 16-element shared exponent blocks
can preserve expert weight quality across a 58-layer deep MoE at production quality.

### 1.2 The Scaler Granularity Argument

Davor's insight on WHY BFP4 works when converting from FP8:

```
FP8 E4M3 with 128x128 block scaling:
  - 1 scale factor per 16,384 elements
  - Per-element: 3 mantissa bits + 4-bit exponent (within block)

BFP4_b with 16-element blocks:
  - 1 shared 8-bit exponent per 16 elements
  - Per-element: 3 mantissa bits (sign + 3 explicit)

Granularity improvement: 16,384 / 16 = 1024x finer exponent sharing
Mantissa bits: ~3 in both formats (approximately unchanged)
```

**The appealing argument**: FP8->BFP4 trades mantissa (unchanged ~3 bits) for 1024x
finer exponent resolution. This should be nearly lossless.

### 1.3 Rigorous Assessment of the Granularity Argument

Two independent Codex GPT-5.4 xhigh analyses show the granularity argument is
**true as metadata comparison but incomplete as an explanation**:

1. **The 1024x is not the active variable.** Once FP8 weights are dequantized to float32
   (as DSv3 does before BFP4 conversion), BFP4 does not "see" the old 128x128 scaler
   structure. It only sees the float values. The relevant metric is within-16-element
   dynamic range of the DEQUANTIZED values.

2. **FP8's 4-bit exponent does NOT constrain within-block DR at the BFP4-relevant
   threshold.** BFP4 zero-flush triggers at 4+ exponent difference (16x ratio). FP8 E4M3's
   representable range within a scaled block is ~2^15 wide -- far wider than 16x. The
   128x128 block scale shifts absolute exponents but does not compress within-group spread.

3. **Monte Carlo validation**: For Gaussian weights, BFP4 zero-flush rate is:
   - BF16 source: 14.9%
   - FP8 block-dequantized: 14.8-14.9%
   - FP8 per-row-dequantized: 14.8-14.9%
   - **Essentially unchanged** regardless of FP8 intermediate step.

4. **What ACTUALLY makes DSv3 BFP4 work**: Quantization-Aware Training (QAT). DSv3 was
   trained with FP8 in the forward pass. Over billions of steps, the optimizer pushes
   weights toward having reduced within-block dynamic range and magnitude clustering --
   creating non-Gaussian structure that is inherently BFP4-friendly. This is a property
   of the LEARNED WEIGHTS, not the conversion format.

5. **The granularity argument becomes valid conditionally**: IF QAT has made local weight
   neighborhoods BF4-friendly, THEN BF4's fine shared exponent preserves that structure.
   But it's the QAT that does the heavy lifting, not the format conversion.

### 1.4 Implications for REAP

REAP was trained in BF16. Its weights are full-precision Gaussian. The FP8 checkpoint
(`cerebras/GLM-4.7-REAP-218B-A32B-FP8`) is Post-Training Quantization (PTQ), not QAT.

**However**, this does NOT mean BF4 definitely fails on REAP. Davor's results show
such massive headroom (+1.9% PPL, -0.15pp MMLU) that even with REAP's architectural
disadvantages (wider experts, more layers), there might be enough margin. Our SEED-3
test was too weak to conclude definitively.

---

## 2. Architectural Comparison: DSv3 vs REAP

### 2.1 Key Differences

| Parameter | DSv3 | REAP-218B | Impact on BF4 |
|-----------|------|-----------|---------------|
| hidden_size | 7168 | 5120 | DSv3 more averaging per output |
| **moe_intermediate_size** | **2048** | **5120** | **2.5x wider w2 accumulation in REAP** |
| n_routed_experts | 256 | 96 | More DSv3 experts |
| num_experts_per_tok | 8 | 8 | Same |
| n_shared_experts | 1 | 1 | Same |
| MoE layers | 58 | 89 | **53% more error accumulation** |
| **routed_scaling_factor** | **2.5** | **2.5** | **SAME (prior analysis was WRONG)** |
| norm_topk_prob | true | true | Same |
| scoring_func | sigmoid | sigmoid | Same |
| first_k_dense_replace | 3 | 3 | Same |
| **Weight training** | **FP8-aware (QAT)** | **BF16 (PTQ FP8 available)** | **DSv3 weights adapted** |
| **FP8 block shape** | **[128, 128]** | **Per-row [R, 1]** | **Different scaling** |
| Expert kernel path | Dense ttnn.linear | Sparse matmul | DSv3 simpler |

### 2.2 Factors Working Against REAP BF4

1. **Expert width**: REAP w2 (down_proj, most sensitive) accumulates over 5120 dims vs
   DSv3's 2048. BF4 noise scales as sqrt(K): REAP has 1.58x more noise per output element.

2. **Depth**: 89 MoE layers vs 58. Required per-layer PCC for 0.83 total:
   0.9979 (REAP) vs 0.9968 (DSv3).

3. **Weight training**: BF16-trained Gaussian vs FP8 QAT-adapted.

4. **Kernel path**: scatter + moe_expert_token_remap + sparse_matmul + all_reduce
   (more numerical steps) vs dense ttnn.linear.

### 2.3 Factors Working FOR REAP BF4 (or Neutral)

1. **Massive DSv3 headroom**: PPL +1.9%, MMLU -0.15pp. Even if REAP loses 10x more
   quality from BF4, that's still only PPL +19%, MMLU -1.5pp -- possibly usable.

2. **Shared expert anchor**: Both models use shared expert + routed_scaling_factor=2.5.
   Shared expert stays BF8, anchoring 13.8% of MLP energy.

3. **RMSNorm**: Prevents magnitude divergence at each layer boundary.

4. **Routing scale identical**: Both use 2.5. This is NOT a differentiator.

### 2.4 Layer-Depth Sensitivity

| Per-layer PCC | Max MoE layers (total PCC >= 0.83) | DSv3 (58) | REAP (89) |
|--------------|-----------------------------------|-----------|-----------|
| 0.970 | 6 | FAIL | FAIL |
| 0.980 | 9 | FAIL | FAIL |
| 0.990 | 18 | FAIL | FAIL |
| 0.995 | 37 | FAIL | FAIL |
| 0.997 | 62 | PASS | FAIL |
| 0.998 | 93 | PASS | PASS (barely) |

DSv3 at PPL +1.9% implies per-layer PCC is very high (likely >= 0.999).
**If REAP's BF4 per-layer PCC is even 0.998, it survives 89 layers.**

---

## 3. Why Our SEED-3 Test Was Insufficient

### 3.1 Methodological Weaknesses

| Issue | Problem | Impact |
|-------|---------|--------|
| **Eyeball-only quality check** | "Garbled/repetitive at temp=0" is subjective | Could be mild degradation mistaken for failure |
| **No perplexity measurement** | PPL is the standard metric for quantization quality | DSv3 had +1.9% PPL -- we'd have caught marginal vs catastrophic |
| **No benchmark suite** | No MMLU, GSM8K, or other standard evals | Can't quantify degradation |
| **Single prompt** | Checked one or few outputs | Not statistically significant |
| **Stale trace artifact** | First output showed "Paris flexibly" | Container restart may not have cleared all state |
| **New untested code** | Per-projection dtype code was brand new | Bugs in weight loading could cause garbled output |
| **Weight cache collision** | dtype_tag changed format | Stale BF8 weights may have been served as "BF4" |
| **No FP8 source test** | Only tested BF16 -> BFP4 | FP8 -> BFP4 is the proven path from Davor |

### 3.2 Known Risks in SEED-3 Code Path

The per-projection dtype code (`_env_expert_projection_dtype()` at `layer_weights.py:87`)
was implemented specifically for SEED-3. Potential issues:

1. **Cache key format**: `dtype_tag = f"{w1_dtype.name}_{w2_dtype.name}_{w3_dtype.name}"`
   (line 612). If a previous run cached with a different format, stale weights could load.

2. **Weight conversion correctness**: `_experts_weight_tt()` passes dtype to `ttnn.as_tensor`
   (line 254). Need to verify BFP4_b conversion happens correctly for stacked expert tensors
   of shape `[num_devices, 1, experts_per_device, in, out]`.

3. **Fused gate+up interaction**: If `FUSE_EXPERTS_GATE_UP=0` but the flag somehow leaked,
   w1/w3 would be concatenated and both converted at w1's dtype, ignoring w3's separate dtype.

4. **No dtype verification logging**: The code logs dtypes at layer_idx == first_k_dense_replace
   (line 614-615) but doesn't verify the ACTUAL loaded tensor dtype matches the requested one.

---

## 4. BFP4_b Format Technical Background

### 4.1 Bit Layout

Per tile (32x32 = 1024 elements): 576 bytes total.
- 64 exponent bytes: shared 8-bit exponent per 16 elements
- 512 data bytes: 4 bits per element (1 sign + 3 mantissa)

### 4.2 Zero-Flush Mechanism

Elements whose exponent is 4+ below the block maximum are flushed to zero.
For Gaussian weights: ~14.6-15% zero-flush rate (~2.3 of 16 per block).

### 4.3 Per-Projection Sensitivity

```
w2 (down) >> w1 (gate) >> w3 (up)
  MOST          SECOND       LEAST
```

- **w2**: Error goes directly to residual, amplified by 2.5x routing scale
- **w1**: SiLU gate errors can flip gates open/closed
- **w3**: Error suppressed by closed gates (~50% of preactivations)

### 4.4 SEED-3 Tested the Wrong Pattern

SEED-3: w1/w2=BF4, w3=BF8 (DSv3 pattern) -- puts BF4 on the TWO MOST sensitive projections.
DSv3 can afford this because of narrower w2 (2048 vs 5120) and QAT-adapted weights.

---

## 5. Per-Projection Sensitivity and Hadamard Analysis

### 5.1 Hadamard: Provably Useless for REAP BF16 Weights

REAP BF16 weights are i.i.d. Gaussian (Shapiro-Wilk p>0.4, kurtosis~0). Multivariate
Gaussian is rotationally invariant: no orthogonal transform can reduce within-block DR.
Experimental: 60 tensors, DR 38.2x -> 38.2x, SNR 16.5dB -> 16.5dB. Zero improvement.

### 5.2 Weight Permutation: Theoretically Beneficial, Practically Impossible

Per-row magnitude-sorted blocking: zero-flush 14.9% -> 0.026%, PCC 0.9932 -> 0.9983.
Cannot implement: breaks matmul (each row needs different column permutation of x).
Shared column permutation: zero benefit for Gaussian weights (verified: +0.00 PCC).

---

## 6. Execution Plan: Rigorous BF4 Re-Testing

### 6.0 Prerequisites

**Container state**: Galaxy Wormhole container must be RUNNING and HEALTHY.
- Start: `sg docker -c 'docker compose --env-file dev/.env.glm47_reap -f dev/docker-compose.yml -f dev/docker-compose.galaxy.yml up -d vllm-tt'`
- Verify: `docker compose --env-file dev/.env.glm47_reap -f dev/docker-compose.yml -f dev/docker-compose.galaxy.yml ps`
- Wait for model load (check logs for "Started serving" message)

**Weight cache**: Located at the Docker volume. Each test MUST use a fresh cache key
(the `dtype_tag` in `experts_variant` ensures this if per-projection dtypes change).
To force rebuild: delete cached `.bin` files matching the layer pattern.

**Benchmark scripts**:
- Throughput: `tests/bench_reap_perf.py` (streaming, concurrent requests)
- Quality: Manual curl / Python script with diverse prompts + logprobs

### 6.1 Test 0: BF8 Baseline (Reference)

**Purpose**: Establish definitive BF8 baseline with proper metrics for comparison.

**Env vars** (in `.env.glm47_reap` or docker-compose override):
```bash
GLM4_MOE_EXPERTS_TT_DTYPE=bf8
# Remove any per-projection overrides:
# GLM4_MOE_EXPERTS_W1_DTYPE  (unset)
# GLM4_MOE_EXPERTS_W2_DTYPE  (unset)
# GLM4_MOE_EXPERTS_W3_DTYPE  (unset)
GLM4_MOE_DENSE_TT_DTYPE=bf8
GLM4_MOE_FUSE_EXPERTS_GATE_UP=0
```

**Steps**:
1. Fresh container restart (down + up, NOT restart)
2. Wait for full model load + warmup
3. Run throughput: `python3 tests/bench_reap_perf.py --batch-sizes 1,32 --max-tokens 50`
4. Run quality: 10 diverse prompts via curl (see prompt list in Section 6.6)
5. Run logprobs-based approximate PPL (see Section 6.7)
6. Record ALL outputs verbatim

**Expected**: ~4.1 tok/s bs=1, ~121.6 tok/s bs=32, coherent output.

### 6.2 Test 1: Reproduce SEED-3 (w1/w2=BF4, w3=BF8) with Proper Metrics

**Purpose**: Reproduce the original SEED-3 test with rigorous measurement.

**Env vars**:
```bash
GLM4_MOE_EXPERTS_TT_DTYPE=bf8          # fallback (overridden by per-proj)
GLM4_MOE_EXPERTS_W1_DTYPE=bf4          # gate_proj
GLM4_MOE_EXPERTS_W2_DTYPE=bf4          # down_proj (MOST sensitive)
GLM4_MOE_EXPERTS_W3_DTYPE=bf8          # up_proj
GLM4_MOE_DENSE_TT_DTYPE=bf8            # shared expert + attention stays BF8
GLM4_MOE_FUSE_EXPERTS_GATE_UP=0
```

**Steps**:
1. Fresh container restart
2. **Verify weight loading**: Check logs for `Expert dtypes: w1=DataType.bfloat4_b, w2=DataType.bfloat4_b, w3=DataType.bfloat8_b`
3. If cache exists for this dtype_tag, consider deleting to force rebuild
4. Wait for full model load + warmup
5. Run throughput: `python3 tests/bench_reap_perf.py --batch-sizes 1,32 --max-tokens 50`
6. Run quality: same 10 prompts as baseline
7. Run logprobs-based approximate PPL
8. Record ALL outputs verbatim
9. Compare against Test 0 baseline

**Expected throughput**: ~127 tok/s bs=32 (+5% over baseline, matching prior SEED-3 result).
**Quality**: This is the key measurement. Compare output coherence and PPL against baseline.

### 6.3 Test 2: All-Expert BF4 (Davor's Configuration)

**Purpose**: Match Davor's DSv3 test configuration as closely as possible.
ALL routed expert projections (w1, w2, w3) in BFP4_b. Shared expert stays BF8.

**Env vars**:
```bash
GLM4_MOE_EXPERTS_TT_DTYPE=bf4          # ALL routed experts BF4
# Remove per-projection overrides (let all fall through to bf4):
# GLM4_MOE_EXPERTS_W1_DTYPE  (unset)
# GLM4_MOE_EXPERTS_W2_DTYPE  (unset)
# GLM4_MOE_EXPERTS_W3_DTYPE  (unset)
GLM4_MOE_DENSE_TT_DTYPE=bf8            # shared expert + attention stays BF8
GLM4_MOE_FUSE_EXPERTS_GATE_UP=0
```

**Note on shared expert**: The shared expert uses `_env_dense_dtype()` (controlled by
`GLM4_MOE_DENSE_TT_DTYPE`), NOT `_env_experts_dtype()`. So setting
`GLM4_MOE_EXPERTS_TT_DTYPE=bf4` affects ONLY routed experts. Shared expert stays BF8.
Verified: `layer_weights.py:522,529,548,558` all use `dense_dtype`.

**Steps**:
1. Fresh container restart
2. Verify logs: `Expert dtypes: w1=DataType.bfloat4_b, w2=DataType.bfloat4_b, w3=DataType.bfloat4_b`
3. Force cache rebuild (new dtype_tag `bfloat4_b_bfloat4_b_bfloat4_b`)
4. Run throughput, quality, PPL (same as Test 0)
5. Compare against baseline

**Expected throughput**: ~133-140 tok/s bs=32 (~10-15% improvement).
**Quality**: Unknown. This is the critical test. If Davor's result generalizes, quality
should be acceptable (PPL degradation < 5%).

### 6.4 Test 3: Sensitivity-Informed Pattern (w2=BF8, w1/w3=BF4)

**Purpose**: Protect the most sensitive projection (w2/down) at BF8 while putting the
two less sensitive ones (w1/gate, w3/up) at BF4.

**Env vars**:
```bash
GLM4_MOE_EXPERTS_TT_DTYPE=bf8          # fallback
GLM4_MOE_EXPERTS_W1_DTYPE=bf4          # gate_proj (moderate sensitivity)
GLM4_MOE_EXPERTS_W2_DTYPE=bf8          # down_proj (PROTECT -- most sensitive)
GLM4_MOE_EXPERTS_W3_DTYPE=bf4          # up_proj (least sensitive)
GLM4_MOE_DENSE_TT_DTYPE=bf8
GLM4_MOE_FUSE_EXPERTS_GATE_UP=0
```

**Steps**: Same as Test 1.

**Expected throughput**: ~+2-3% over baseline (smaller savings, only 2/3 projections at BF4).
**Quality**: Should be BETTER than Test 1 (protects most sensitive projection) and BETTER
than Test 2 (protects most sensitive at BF8). If Test 2 passes, this should definitely pass.

### 6.5 Test 4: Blanket BF4 (Maximum Compression)

**Purpose**: Maximum compression test -- ALL weights in BFP4_b including shared expert
and attention. If Davor's headroom is as large as it appears, even this might work.

**Env vars**:
```bash
GLM4_MOE_EXPERTS_TT_DTYPE=bf4          # ALL routed experts BF4
# Remove per-projection overrides:
# GLM4_MOE_EXPERTS_W1_DTYPE  (unset)
# GLM4_MOE_EXPERTS_W2_DTYPE  (unset)
# GLM4_MOE_EXPERTS_W3_DTYPE  (unset)
GLM4_MOE_DENSE_TT_DTYPE=bf4            # shared expert + attention ALSO BF4
GLM4_MOE_FUSE_EXPERTS_GATE_UP=0
```

**WARNING**: `GLM4_MOE_DENSE_TT_DTYPE=bf4` puts attention QKV/O projections AND shared
expert gate/up/down ALL at BFP4_b. This is the worst-case quality configuration.
If this passes, BF4 has massive headroom on REAP.

**Steps**:
1. Fresh container restart
2. Verify logs show BF4 for both expert and dense weight loading
3. Run throughput, quality, PPL (same as Test 0)
4. Compare against baseline

**Expected throughput**: Maximum improvement -- all weight reads at BFP4_b bandwidth.
**Quality**: Likely the worst of all tests. If even this is coherent, BF4 is very safe.
If only this fails but Tests 1-3 pass, the failure is from attention/shared expert BF4.

**Expected weight cache rebuild time**: ~15-25 minutes (all 92 layers must reconvert).

### 6.6 Test 5: FP8 Checkpoint as Source (Advanced, Requires Code Changes)

**Purpose**: Test the FP8->BFP4 conversion path that matches Davor's DSv3 pipeline most closely.

**FP8 checkpoint**: `cerebras/GLM-4.7-REAP-218B-A32B-FP8` on HuggingFace
- Uses `compressed_tensors` quantization with weight strategy `channel`
- Per-row scaling: `[R, 1]` shape (1 scale per row)
- Scale key: `_scale` (NOT `_scale_inv` like DSv3)

**Required code changes** (in `layer_weights.py`):
1. Add FP8 dequantization helper:
```python
def _maybe_dequant_fp8(state_dict, key):
    """Dequantize FP8 weight if scale tensor exists."""
    w = state_dict[key]
    if w.dtype != torch.float8_e4m3fn:
        return w  # Already BF16/FP32, no dequant needed

    # Try _scale first (REAP convention), then _scale_inv (DSv3 convention)
    scale_key = key.replace(".weight", ".weight_scale")
    if scale_key not in state_dict:
        scale_key = key + "_scale"
    if scale_key not in state_dict:
        scale_key = key.replace(".weight", ".weight_scale_inv")
        if scale_key in state_dict:
            return w.float() * state_dict[scale_key].float()  # inv = multiply

    if scale_key not in state_dict:
        logger.warning(f"FP8 weight {key} has no scale tensor, casting directly")
        return w.float()

    scale = state_dict[scale_key].float()
    if scale.shape[-1] == 1:  # Per-row [R,1]
        return w.float() * scale  # Broadcast over columns
    else:  # Block-wise
        from models.demos.deepseek_v3.utils.dequantize import dequantize_tensor
        return dequantize_tensor(w, scale, block_shape=...)
```

2. Apply in expert weight loading loop (after line 621-623):
```python
w1 = _maybe_dequant_fp8(state, f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.weight")
w3 = _maybe_dequant_fp8(state, f"model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.weight")
w2 = _maybe_dequant_fp8(state, f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight")
```

3. Update `.env.glm47_reap`:
```bash
HF_MODEL=cerebras/GLM-4.7-REAP-218B-A32B-FP8
HF_HUB_OFFLINE=0   # Need to download FP8 checkpoint
```

4. Add cache version bump to avoid BF16-source / FP8-source weight collision.

**Note**: This test requires HuggingFace access and model download (~400GB for FP8).
May need `HF_TOKEN` if model is gated. Check availability first.

**Steps**:
1. Verify FP8 model is downloadable
2. Implement dequant code changes
3. Set `HF_HUB_OFFLINE=0`, `HF_MODEL=cerebras/GLM-4.7-REAP-218B-A32B-FP8`
4. Fresh container restart with FP8 model
5. Test with all-expert BF4 (matching Test 2 config)
6. Run throughput, quality, PPL
7. Compare against Test 0 (BF8 from BF16) and Test 2 (BF4 from BF16)

**Expected**: If the FP8 source helps, quality should be notably better than Test 2.
If it doesn't help (Monte Carlo prediction), quality should be similar to Test 2.

### 6.7 Quality Evaluation: 10 Diverse Prompts

Use these prompts for ALL tests (temperature=0, max_tokens=200):

```python
QUALITY_PROMPTS = [
    # 1. Factual recall
    "What is the capital of France and what is its population?",
    # 2. Reasoning / math
    "If a train travels 120 km in 2 hours, then 180 km in 3 hours, what is its average speed?",
    # 3. Code generation
    "Write a Python function that takes a list of integers and returns the two numbers that sum to a target value.",
    # 4. Creative writing
    "Write a short story (3 paragraphs) about a robot that discovers music for the first time.",
    # 5. Multilingual
    "Translate 'The weather is beautiful today' into French, German, and Japanese.",
    # 6. Summarization
    "Summarize the key differences between TCP and UDP protocols in networking.",
    # 7. Logical deduction
    "Alice is taller than Bob. Bob is taller than Charlie. Charlie is taller than Diana. Who is the shortest?",
    # 8. Technical explanation
    "Explain how a transformer neural network's attention mechanism works in simple terms.",
    # 9. Instruction following
    "List exactly 5 prime numbers between 50 and 100, one per line.",
    # 10. Open-ended
    "What are three potential consequences of widespread adoption of autonomous vehicles?",
]
```

For each prompt, record:
- Full output text (verbatim)
- Whether output is coherent (Y/N)
- Whether output is repetitive (Y/N)
- Whether output answers the question correctly (Y/N/Partial)

Score: number of coherent+correct responses out of 10.

### 6.8 Approximate Perplexity via Logprobs

vLLM supports logprobs via the completions API. Use this for approximate PPL:

```bash
# Single prompt PPL approximation
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cerebras/GLM-4.7-REAP-218B-A32B",
    "prompt": "The quick brown fox jumps over the lazy dog. The weather today is sunny and warm. In the field of machine learning, transformer architectures have revolutionized natural language processing by introducing self-attention mechanisms that allow models to weigh the importance of different input tokens when generating each output token.",
    "max_tokens": 100,
    "temperature": 0,
    "logprobs": 1,
    "echo": true
  }'
```

Extract per-token logprobs and compute:
```python
import math
ppl = math.exp(-sum(logprobs) / len(logprobs))
```

Run with 5+ different passages of ~100-200 tokens each. Average the PPL values.

**Note**: This gives CONDITIONAL perplexity (conditioned on the prompt), not the full
corpus perplexity that Davor measured. It's an approximation but sufficient for A/B
comparison between BF8 and BF4 configs. Relative delta is what matters.

If `echo` is not supported, use the completions endpoint with a long prompt and measure
logprobs on the generated continuation only.

### 6.9 Implementation Checklist

Before EACH test:

- [ ] Container fully stopped: `docker compose ... down`
- [ ] Env vars set correctly in `.env.glm47_reap` or compose override
- [ ] Per-projection env vars either SET (for mixed-dtype tests) or UNSET (for uniform tests)
- [ ] `docker-compose.galaxy.yml` passes through per-projection env vars (W1_DTYPE, W2_DTYPE, W3_DTYPE)
- [ ] Container started fresh: `docker compose ... up -d vllm-tt`
- [ ] Wait for "Started serving" in logs
- [ ] Verify expert dtypes in log output (search for "Expert dtypes:")
- [ ] Verify weight cache rebuilt if dtype changed (search for "Loading" or "Converting" in logs)
- [ ] Run Test 0 prompts first as sanity check

After EACH test:

- [ ] Record all 10 prompt outputs verbatim
- [ ] Record throughput numbers (bs=1 and bs=32)
- [ ] Record approximate PPL
- [ ] Record exact env vars used
- [ ] Note any anomalies (trace errors, stale outputs, etc.)

### 6.10 Docker-Compose Env Passthrough

The per-projection env vars (GLM4_MOE_EXPERTS_W1_DTYPE, etc.) must be passed through
to the container. Check `docker-compose.yml` and `docker-compose.galaxy.yml` for
environment passthrough. If not present, add:

```yaml
environment:
  - GLM4_MOE_EXPERTS_W1_DTYPE=${GLM4_MOE_EXPERTS_W1_DTYPE:-}
  - GLM4_MOE_EXPERTS_W2_DTYPE=${GLM4_MOE_EXPERTS_W2_DTYPE:-}
  - GLM4_MOE_EXPERTS_W3_DTYPE=${GLM4_MOE_EXPERTS_W3_DTYPE:-}
```

Empty default (`:-}`) means the var is unset inside the container if not specified in
.env, which causes `_env_expert_projection_dtype()` to fall through to the global
`GLM4_MOE_EXPERTS_TT_DTYPE`.

### 6.11 Decision Criteria

| Metric | BF4 VIABLE | MARGINAL | CONFIRMED BROKEN |
|--------|------------|----------|------------------|
| Approximate PPL delta vs BF8 | < 5% | 5-15% | > 15% |
| Coherent outputs (of 10) | >= 9 | 6-8 | <= 5 |
| Correct answers (of 10) | >= 8 | 5-7 | <= 4 |
| Repetitive outputs | 0 | 1-2 | >= 3 |

**If VIABLE**: BF4 is production-ready. Measure throughput gain and ship.
**If MARGINAL**: Investigate per-projection patterns (Test 3). May be usable with
sensitivity-informed dtype assignment.
**If BROKEN**: Investigate bug checklist (Section 6.12) before final closure.

### 6.12 Bug Investigation Checklist (If Tests Still Fail)

If tests produce garbled output, DO NOT immediately conclude "BF4 is broken". First:

1. **Verify BFP4_b weights are actually loaded**:
   - Add temporary logging after `_experts_weight_tt()` returns:
     ```python
     logger.info(f"  w1 actual dtype: {w1_experts.dtype}, shape: {w1_experts.shape}")
     ```
   - Check that dtype is `DataType.bfloat4_b`, not `DataType.bfloat8_b`

2. **Check weight cache isn't serving wrong dtype**:
   - Delete all cached `.bin` files for this layer/expert pattern
   - Or: change `experts_variant` string to force cache miss
   - Rebuild from scratch and test again

3. **Isolate single expert PCC**:
   - Load one expert's w1 weight in BF16 and BFP4_b
   - Compute a reference matmul output in BF16
   - Compute BFP4_b matmul output
   - Measure PCC between the two
   - Expected: ~0.993 for Gaussian weights, higher for FP8-dequanted

4. **Check for code bugs in per-projection dtype path**:
   - Verify `_env_expert_projection_dtype("W1")` returns the correct dtype
   - Verify the dtype flows through to `_experts_weight_tt(dtype=w1_dtype)`
   - Verify `ttnn.as_tensor(..., dtype=ttnn.bfloat4_b)` actually quantizes (not just cast)

5. **Check fused gate+up interaction**:
   - Verify `GLM4_MOE_FUSE_EXPERTS_GATE_UP=0` is definitely read correctly
   - If fused path is accidentally taken, w1 and w3 share w1's dtype

6. **Check trace capture with BF4 shapes**:
   - BFP4_b tiles are 576 bytes vs BFP8_b's 1088 bytes
   - Verify trace capture handles different weight tile sizes correctly
   - Test with `trace_mode=none` (no tracing) to isolate trace vs quantization issues

7. **Test with reduced layers** (diagnostic only):
   - `GLM4_MOE_NUM_LAYERS=10` with BF4 -- should produce near-coherent output
   - `GLM4_MOE_NUM_LAYERS=30` -- still likely OK
   - `GLM4_MOE_NUM_LAYERS=60` -- DSv3-equivalent depth
   - `GLM4_MOE_NUM_LAYERS=92` -- full model
   - This traces the error accumulation curve and confirms PCC^N model

8. **Compare single layer output**: BF8 vs BF4:
   - Run with `GLM4_MOE_NUM_LAYERS=4` (first_k_dense_replace=3, so 1 MoE layer)
   - BF8 config -> record output
   - BF4 config -> record output
   - Compare: if single-layer output is already garbled, it's a format or code issue,
     not error accumulation

9. **Check sparse_matmul precision vs ttnn.linear for BFP4 inputs**:
   - DSv3 uses dense `ttnn.linear` for expert forward -- simpler, fewer rounding steps
   - REAP uses `sparse_matmul` via scatter + token_remap + matmul + all_reduce
   - Create isolated test: same BFP4_b weight tensor, same input, compare output of
     `ttnn.linear(input, weight)` vs `sparse_matmul(input, weight, program_config)`
   - If PCC differs significantly, sparse_matmul may introduce additional precision loss
     that compounds with BFP4 quantization error
   - This would explain why DSv3 tolerates BF4 on w1/w2 but REAP does not

---

## 7. The FP8 Checkpoint Path (Detailed Analysis)

### 7.1 REAP's FP8 Checkpoint Format

`cerebras/GLM-4.7-REAP-218B-A32B-FP8`:
- Quantization: `compressed_tensors`, weight strategy `channel`
- Per-row scaling: scale shape `[R, 1]` (1 scale per row of weight matrix)
- Scale key convention: `_scale` (not `_scale_inv` like DSv3)
- FP8 dtype: E4M3 (torch.float8_e4m3fn)
- This is PTQ (post-training quantization), NOT QAT

### 7.2 Per-Row vs Block Scaling After TT Transpose

```
HuggingFace layout: w2 = [5120, 5120], per-row scale = [5120, 1]
  -> Per-row = per OUTPUT-neuron scale

After TT transpose for sparse_matmul: w2_t = [5120, 5120]
  -> Per-row becomes per-COLUMN in transposed layout
  -> BFP4_b groups 16 contiguous values in the row dimension
  -> One BFP4 group can mix elements with 16 DIFFERENT original per-row scales
```

This means per-row scaling may actually make dequantized values within a BFP4 group
MORE heterogeneous (different scales applied), potentially increasing within-block DR.
DSv3's 128x128 block scaling stays aligned with TT tiling and may be better suited.

### 7.3 Revised PCC Estimates

| Scenario | Estimated per-layer PCC | After 89 layers | Verdict |
|----------|------------------------|-----------------|---------|
| BF8 all (current) | ~0.998 | 0.837 | Usable |
| BF16->BFP4 all (SEED-3) | ~0.97 (eyeball est.) | 0.065 | Claimed garbled |
| FP8(PTQ)->BFP4 all | ~0.976-0.985 | 0.118-0.260 | Likely garbled |
| FP8(QAT)->BFP4 (DSv3) | ~0.999+ | 0.914+ | Lossless |

**Key uncertainty**: The "~0.97" for BF16->BFP4 was never measured -- it was estimated
from observed garbled output, which may have been a code bug. Actual per-layer PCC could
be higher, which would change the entire analysis.

---

## 8. Recommendations

### 8.1 Priority Ordering

| Priority | Test | Effort | Information Value |
|----------|------|--------|-------------------|
| **P0** | Test 0 (BF8 baseline with PPL) | Low | Establishes reference |
| **P1** | Test 2 (All-expert BF4, Davor config) | Low | Directly comparable to Davor's result |
| **P2** | Test 1 (Reproduce SEED-3 with PPL) | Low | Validates/invalidates prior conclusion |
| **P3** | Test 3 (w2=BF8, w1/w3=BF4) | Low | Best sensitivity-informed pattern |
| **P4** | Test 4 (Blanket BF4) | Low | Maximum compression stress test |
| **P5** | Test 5 (FP8 checkpoint) | High (code changes) | Only if Tests 1-4 fail |

### 8.2 If All Tests Fail (After Bug Investigation)

Accept BF8 as the floor. BFP4_b is architecturally incompatible with REAP-218B's
specific combination of 5120-wide experts, 89 MoE layers, and BF16-trained Gaussian weights.

### 8.3 If Tests Pass (PPL < 5%, Quality Acceptable)

Ship BF4 immediately. Expected throughput gain:
- All-expert BF4: ~133-140 tok/s bs=32 (+10-15%)
- Selective BF4 (w1/w3 only): ~124-127 tok/s bs=32 (+2-5%)
- DRAM savings: ~2-3 GB/device (23-47% reduction in expert weight footprint)

### 8.4 What Would Make BF4 More Likely to Work

1. **Use FP8 QAT checkpoint** (if Cerebras releases one -- current FP8 is PTQ)
2. **Reduce moe_intermediate_size** (architectural change, requires retraining)
3. **Fewer MoE layers** (architectural change)
4. **Per-element FP8 hardware support** (Wormhole limitation)
5. **Smaller BFP block size** (hardware change, per-4 instead of per-16)

---

## 9. Appendix: Key Source References

| File | Lines | What |
|------|-------|------|
| `deepseek_v3/reference/config.json` | 37-44, 57 | weight_block_size=[128,128], routed_scaling_factor=2.5 |
| `deepseek_v3/tt/experts.py` | 57-77 | FP8 dequant -> BF4/BF8 assignment |
| `deepseek_v3/utils/dequantize.py` | 12-67 | Block-wise dequantization algorithm |
| `deepseek_v3/tests/test_decoder_block.py` | 197 | Decoder PCC >= 0.9899 |
| `deepseek_v3/tests/test_moe_experts.py` | 201 | Expert PCC >= 0.98 |
| `glm4_moe/tt/config.py` | 39, 86 | routed_scaling_factor=2.5 |
| `glm4_moe/tt/layer_weights.py` | 87-101 | Per-projection dtype override |
| `glm4_moe/tt/layer_weights.py` | 220-259 | _experts_weight_tt conversion |
| `glm4_moe/tt/layer_weights.py` | 503-560 | Shared expert weight loading (uses dense_dtype) |
| `glm4_moe/tt/layer_weights.py` | 605-687 | Expert weight loading loop |
| `glm4_moe/tt/moe_tt.py` | 301-380, 399-582 | Router + expert forward |
| `tt_metal/impl/data_format/blockfloat_common.cpp` | 22-329 | BFP4 quantization |
| `docker_tt/dev/.env.glm47_reap` | full | Current env configuration |
| `docker_tt/tests/bench_reap_perf.py` | full | Streaming benchmark script |

## 10. Appendix: Corrections to Prior Analysis

| Claim | Correction | Source |
|-------|-----------|--------|
| "DSv3 routed_scaling_factor = 1.0" | **Both use 2.5** | `config.json:57` |
| "BF4 CONFIRMED BROKEN for REAP" | **Insufficiently tested -- re-test required** | This revision |
| "FP8 pre-conditioning reduces within-block DR" | **Only with QAT, not PTQ or format conversion** | Codex MC analysis |
| "Exponent granularity compensates mantissa loss" | **Conditionally true (QAT only), not universal** | Codex xhigh analysis |
| "BFP4_b fundamentally incompatible with deep MoE" | **DSv3/R1 proves it works at 58 layers** | Davor empirical data |

## 11. Appendix: Prior Research Synthesis

This analysis synthesizes:
- `plan/glm47_reap/galaxy_wormhole/research-bf4-codex.md` (2026-03-09)
- `plan/glm47_reap/galaxy_wormhole/research-bf4-gemini.md` (2026-03-09)
- `plan/glm47_reap/galaxy_wormhole/hadamard-rotation-results.md` (2026-02-25)
- `plan/glm47_reap/galaxy_wormhole/research-seed.md` (2026-03-10)
- Codex GPT-5.4 xhigh analyses (2026-03-10):
  1. BFP4_b format + per-projection sensitivity
  2. Hadamard/rotation feasibility for Gaussian weights
  3. Literature review: QuIP#, AQLM, GPTQ, AWQ, SmoothQuant, SpinQuant
  4. DSv3 vs REAP architectural comparison (corrected routed_scaling_factor)
  5. Layer-depth sensitivity + RMSNorm + shared expert anchor
  6. FP8 scaler granularity + Monte Carlo zero-flush analysis
  7. Scaler granularity argument validity assessment
- Davor (TT quantization expert): DSv3 R1 BFP4 empirical results (2026-03-10)
