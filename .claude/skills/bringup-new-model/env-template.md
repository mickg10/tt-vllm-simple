# Environment File Template

When creating `dev/.env.<short_name>`, use this as the base and customize model-specific sections.

## Base (required for all models)

```env
# Workspace path (required)
WORKSPACE_PATH=/home/ttuser/src_docker/ws/<SHORT_NAME>

# Model identity
HF_MODEL=<HF_MODEL_ID>
HF_TOKEN=

# Device configuration
MESH_DEVICE=<DEVICE_TYPE>

# Build
BUILD_JOBS=16
SKIP_TT_METAL_BUILD=1

# vLLM serving defaults (conservative for bring-up)
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=1
VLLM_RPC_TIMEOUT=600000
VLLM_ENGINE_ITERATION_TIMEOUT_S=600
OVERRIDE_TT_CONFIG={"trace_mode":"none"}
```

## MoE models (add if model has routed experts)

```env
# MoE enable/disable (start with 0 for dense-only bring-up, flip to 1 for Phase 6)
<PREFIX>_ENABLE_MOE=0
# Sparse expert chunk size (reduce for DRAM-limited long prompts)
<PREFIX>_MOE_SPARSE_CHUNK_TOKENS=2048
# Sparse compute fidelity (hifi4 for correctness, hifi2 for speed experiments)
<PREFIX>_MOE_SPARSE_FIDELITY=hifi4
<PREFIX>_MOE_SPARSE_FP32_ACC=1
<PREFIX>_MOE_SPARSE_APPROX=0
<PREFIX>_MOE_SPARSE_DEBUG=0
```

## MLA models (add if model has kv_lora_rank)

```env
# MLA decode path
<PREFIX>_MLA_FIDELITY=hifi4
<PREFIX>_MLA_APPROX=0
# V-cache slicing (1 = legacy safe path, 0 = direct KVPE)
<PREFIX>_MLA_USE_V_CACHE_SLICE=1
```

## Performance tuning (add once correctness is stable)

```env
# TP sharding (0 = replicated bring-up, 1 = tensor-parallel)
<PREFIX>_TP=0
# Fused projections (0 = separate, 1 = fused)
<PREFIX>_FUSE_QKV_A=0
# Dense weight dtype (bf16 default, bf8 experimental)
<PREFIX>_DENSE_TT_DTYPE=bf16
```

## Profiling (add for performance work)

```env
<PREFIX>_PROFILE=0
<PREFIX>_PROFILE_LAYER=
<PREFIX>_PROFILE_PRINT_EVERY=32
```

## Partial model (add for fast iteration during early phases)

```env
# Run only first N layers (comment out for full model)
# <PREFIX>_NUM_LAYERS=2
# <PREFIX>_DEBUG_ALLOW_PARTIAL_LAYERS=1
```

## Convention

- `<PREFIX>` should be `<MODEL_TYPE_UPPER>` with dots replaced by underscores
  - Example: `glm4_moe_lite` -> `GLM4_MOE_LITE`
  - Example: `llama` -> `LLAMA`
- Every env var should be passed through in `dev/docker-compose.yml` under the `vllm-tt` service
- Every env var should be read via `os.environ.get(...)` in the tt-metal model code with sensible defaults
