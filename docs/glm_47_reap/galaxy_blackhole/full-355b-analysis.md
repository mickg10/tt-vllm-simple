# Full GLM-4.7 (355B) vs REAP-218B — Analysis & Decision

**Created**: 2026-03-09
**Decision**: Full GLM-4.7-355B is the PRIMARY model for BH Galaxy.
REAP-218B is BANNED — used only for debugging trace+CCL issues.

---

## Model Comparison

The ONLY difference between the two models is `n_routed_experts`: 160 vs 96.
REAP-218B is literally GLM-4.7 with 64 experts removed via REAP pruning.
All other architecture parameters are identical.

| Parameter | Full GLM-4.7 (355B) | REAP-218B |
|-----------|---------------------|-----------|
| HF model ID | `zai-org/GLM-4.7-FP8` | `cerebras/GLM-4.7-REAP-218B-A32B-FP8` |
| model_type | `glm4_moe` | `glm4_moe` |
| n_routed_experts | **160** | **96** |
| FP8 total size | 362 GB | 220 GB |
| Everything else | Same | Same |

---

## EP Sharding

| Model | Experts/Device (EP=32) | Division |
|-------|----------------------|----------|
| Full GLM-4.7 | 160 / 32 = **5** | Exact integer |
| REAP-218B | 96 / 32 = **3** | Exact integer |

Both divide evenly. No expert padding needed.

---

## Memory Budget (per BH device, 32 GB)

| Component | Full GLM-4.7 (BF8) | REAP-218B (BF8) |
|-----------|---------------------|-----------------|
| Expert weights (89 MoE layers) | **10.50 GB** | **6.30 GB** |
| Attention Q+O (92 layers, TP=4) | 2.87 GB | 2.87 GB |
| Attention K+V (92 layers, TP=4) | 0.48 GB | 0.48 GB |
| Shared expert (89 MoE layers) | 0.52 GB | 0.52 GB |
| Dense MLP (layers 0-2, TP=4) | 0.14 GB | 0.14 GB |
| Embedding + LM head | 0.10 GB | 0.10 GB |
| Router + norms | 0.14 GB | 0.09 GB |
| **Total weights** | **14.75 GB** | **10.50 GB** |
| **DRAM remaining** | **17.25 GB** | **21.50 GB** |
| **Fits?** | **YES (46%)** | **YES (33%)** |

Both fit easily. Full GLM-4.7 uses 46% of DRAM, leaving 17.25 GB for KV cache and runtime.

---

## Code Impact

**ZERO additional code changes** for the full model vs REAP. The codebase reads
`n_routed_experts` dynamically from config.json:

```python
# config.py
n_routed_experts=int(getattr(hf_config, "n_routed_experts"))

# layer_weights.py
num_experts_per_device = num_experts // max(1, num_devices)

# moe_tt.py (validation)
if num_experts % max(1, num_devices) != 0:
    raise ValueError(...)
```

160 % 32 = 0 — passes validation. Expert mapping matrix created dynamically.

The ONLY env file difference:
```bash
# Full 355B:
HF_MODEL=/home/mick/models/GLM-4.7-FP8

# REAP-218B:
# HF_MODEL=/home/mick/models/GLM-4.7-REAP-218B-A32B-FP8
```

---

## Performance Comparison

Full model reads ~67% more expert weight data per forward pass:
- Extra per-layer: (118 - 70.8) MB / 500 GB/s = ~0.094 ms
- Over 89 layers: ~8.4 ms extra ITL

| Scenario | Full GLM-4.7 | REAP-218B |
|----------|-------------|-----------|
| Conservative (TP=4) | ~5.5-7.5 tok/s | ~6-8 tok/s |
| With optimizations | ~9-11 tok/s | ~10-12 tok/s |
| Quality | **BEST** (unpruned) | Good (REAP pruned) |

---

## Decision Rationale

1. **Quality**: Full model is unpruned — 64 more experts means better coverage of the input space
2. **Memory**: Both fit easily in 32 GB/device (46% vs 33% utilization)
3. **Code**: Zero additional changes — same codebase, same execution plan
4. **Performance**: ~8ms ITL penalty from more expert weights — acceptable trade-off for quality
5. **Simplicity**: One model to maintain, one env file difference

---

## Sources

- Full GLM-4.7 FP8: https://huggingface.co/zai-org/GLM-4.7-FP8 (362 GB)
- REAP-218B FP8: https://huggingface.co/cerebras/GLM-4.7-REAP-218B-A32B-FP8 (220 GB)
