---
name: bringup-new-model
description: Bring up a new HuggingFace model on Tenstorrent hardware. Scaffolds all three repos (tt-metal, vllm, docker_tt), sets up a reference endpoint for correctness comparison, and guides through the phased bring-up process.
argument-hint: "[hf_model_id]"
allowed-tools: Read, Grep, Glob, Bash, Write, Edit, AskUserQuestion, Task
---

You are guiding a developer through bringing up a new HuggingFace model on Tenstorrent (TT) hardware via the vLLM serving framework. This is a multi-repo, multi-phase process that requires careful scaffolding and validation at every step.

## Step 0: Gather Information

Before doing anything, you MUST collect the following from the user. Use AskUserQuestion for structured choices and ask follow-up questions as needed.

**Required information:**

1. **HF Model ID**: The HuggingFace model identifier (e.g. `zai-org/GLM-4.7-Flash`, `meta-llama/Llama-4-Scout-17B-16E`).
   - If `$ARGUMENTS` was provided, use that as the HF model ID.

2. **Reference endpoint**: An OpenAI-compatible API endpoint running this model on another machine (ideally NVIDIA GPU). Ask the user:
   - Base URL (e.g. `http://192.168.1.50:8000/v1`)
   - Model name at that endpoint (may differ from HF ID, e.g. the GGUF/quantized variant name)
   - Verify it is reachable: `curl -sf <base_url>/models`

3. **Device type**: Which TT device mesh to target.
   - Options: `N150`, `N300`, `T3K`, `TG`

4. **Short name**: A workspace identifier (lowercase, underscores ok). Derive a sensible default from the model name (e.g. `llama4_scout` for Llama-4-Scout).

5. **Model architecture details**: Read the model's `config.json` from the HF cache or download it. Extract:
   - `model_type` (e.g. `llama`, `deepseek_v3`, `glm4_moe_lite`)
   - `architectures` list (e.g. `["LlamaForCausalLM"]`)
   - Whether it uses MoE (check for `n_routed_experts`, `num_local_experts`, etc.)
   - Whether it uses MLA (check for `kv_lora_rank`, `qk_nope_head_dim`)
   - Key dims: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`, `vocab_size`
   - RoPE config: `rope_theta`, `rope_scaling`, `partial_rotary_factor`
   - Any special features: `tie_word_embeddings`, MTP layers, vision encoder, etc.

## Step 1: Verify Prerequisites

Before scaffolding, verify:

1. **Model weights are cached** (or download them):
   ```bash
   ls /home/ttuser/.cache/huggingface/hub/models--<org>--<model>/snapshots/
   ```
   If missing, offer to download: `huggingface-cli download <hf_model_id>`

2. **Reference endpoint is reachable**:
   ```bash
   curl -sf <ref_base_url>/models | python3 -m json.tool
   ```
   Record the exact model name returned.

3. **TT devices respond**:
   ```bash
   tt-smi -ls
   ```

4. **Qwen baseline is healthy** (mandatory regression gate):
   ```bash
   cd /home/ttuser/src_docker/ws/main/docker_tt
   ./tests/qwen32b_smoke.sh 8088 Qwen/Qwen3-32B
   ```

5. **Reference endpoint produces sane output** (save as golden reference):
   ```bash
   curl -s <ref_base_url>/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"<ref_model>","messages":[{"role":"user","content":"Return exactly one word: blue"}],"max_tokens":8,"temperature":0}' \
     | python3 -m json.tool
   ```

## Step 2: Create Workspace

```bash
cd /home/ttuser/src_docker/ws/main/docker_tt
make workspace-create NAME=<short_name>
```

This creates `/home/ttuser/src_docker/ws/<short_name>/` with worktrees for all three repos on a new `<short_name>` branch.

Set the workspace path variable for the rest of this process:
```
WORKSPACE=/home/ttuser/src_docker/ws/<short_name>
```

## Step 3: Scaffold tt-metal Model Directory

Create the model directory structure under `$WORKSPACE/tt-metal/models/demos/<short_name>/`:

```
<short_name>/
  tt/
    __init__.py
    config.py           # HParams dataclass parsed from HF config
    weights.py          # LazyStateDict + snapshot resolution
    generator_vllm.py   # vLLM TT model interface (THE critical file)
    model_tt.py         # TT model execution (prefill + decode)
    layer_weights.py    # Per-layer weight conversion to TT tensors
    tt_embedding.py     # Embedding on TT
  tests/
    __init__.py
    test_config.py      # Config parsing test
    test_weights.py     # Weight loading test
  scripts/
    reference_check.py  # Compare TT output vs reference endpoint
```

### 3a: `config.py` — Parse HF config into a typed dataclass

Read the model's `config.json` and create a `<Model>HParams` dataclass with:
- All architecture dimensions from the config
- Derived fields (e.g. `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim` for MLA models)
- A `from_hf_config(cfg)` classmethod
- A `validate()` method that checks invariants

### 3b: `weights.py` — Lazy weight loading

Follow the pattern from `models/demos/glm4_moe_lite/tt/weights.py` or `models/demos/deepseek_v3/tt/`:
- Parse `model.safetensors.index.json` to map keys to shard files
- Load tensors on-demand via `safetensors.safe_open`
- Never load the full state dict into RAM
- Resolve HF cache snapshot directories

### 3c: `generator_vllm.py` — vLLM interface (start as skeleton)

This is the file vLLM imports. It must implement:

```python
class <Model>ForCausalLM(nn.Module):
    model_capabilities = {"supports_prefix_caching": False}

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size,
                              max_seq_len, tt_data_parallel=1, optimizations=None):
        ...

    def prefill_forward(self, tokens, page_table, kv_cache, ...):
        ...  # Return logits [B, S, V]

    def decode_forward(self, tokens, position, page_table, kv_cache, ...):
        ...  # Return logits [B, 1, V]

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        ...  # Return List[List[ttnn.Tensor]]
```

For Phase 3 (skeleton), return placeholder zero logits with correct shapes. This proves the integration works end-to-end before tackling numerical correctness.

## Step 4: Scaffold vLLM Changes

In `$WORKSPACE/vllm/`:

### 4a: Config shim (if `model_type` is not in Transformers 4.x)

Create `vllm/transformers_utils/configs/<model_type>.py`:
- Inherit from `transformers.PretrainedConfig`
- Set `model_type = "<model_type>"`
- Parse all fields from `config.json`
- Handle `**kwargs` permissively

Register it:
- `vllm/transformers_utils/configs/__init__.py` — add import and export
- `vllm/transformers_utils/config.py` — add to `_CONFIG_REGISTRY`

### 4b: MLA registration (if applicable)

If the model uses MLA (`kv_lora_rank` is present), add the `model_type` to the MLA allowlist in `vllm/config.py:ModelConfig.is_deepseek_mla`.

### 4c: TT model registration

In `vllm/platforms/tt.py:register_tt_models()`:
```python
ModelRegistry.register_model(
    "TT<Model>ForCausalLM",
    "models.demos.<short_name>.tt.generator_vllm:<Model>ForCausalLM",
)
```

### 4d: KV cache sizing (if non-standard)

If the model needs special KV cache budget, add an `elif` branch in `vllm/worker/tt_worker.py` for the model_type.

### 4e: Add config test

Create `tests/config/test_<model_type>_config.py` that verifies:
- Config loads from local cache
- Head size is correct (especially for MLA models)
- `use_mla` flag is correct

## Step 5: Scaffold docker_tt Changes

In `$WORKSPACE/docker_tt/`:

### 5a: Entrypoint registration

Add to `entrypoint.sh`:
```bash
ModelRegistry.register_model('TT<Model>ForCausalLM', 'models.demos.<short_name>.tt.generator_vllm:<Model>ForCausalLM')
```

### 5b: Environment file

Create `dev/.env.<short_name>`:
```env
WORKSPACE_PATH=/home/ttuser/src_docker/ws/<short_name>
HF_MODEL=<hf_model_id>
HF_TOKEN=
MESH_DEVICE=<device_type>
BUILD_JOBS=16
SKIP_TT_METAL_BUILD=1
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=1
VLLM_RPC_TIMEOUT=600000
VLLM_ENGINE_ITERATION_TIMEOUT_S=600
OVERRIDE_TT_CONFIG={"trace_mode":"none"}
```

Add model-specific env vars as needed (MoE toggles, dtype knobs, profiling flags, etc.)

### 5c: Smoke test

Create `tests/<short_name>_smoke.sh`:
```bash
#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${1:-8088}
MODEL=${2:-<hf_model_id>}
exec "$SCRIPT_DIR/run_all.sh" "$PORT" "$MODEL"
```
Make it executable: `chmod +x tests/<short_name>_smoke.sh`

### 5d: Reference comparison benchmark

Create `scripts/benchmark_ref_vs_tt_<short_name>.py` — adapt from `scripts/benchmark_ref_vs_tt_glm.py` but parameterized:
- `--ref-base` defaults to the user's reference endpoint URL
- `--ref-model` defaults to the reference model name
- `--tt-base` defaults to `http://localhost:8088/v1`
- `--tt-model` defaults to the HF model ID
- Remove any model-specific flags (like `disable_thinking`) unless the new model needs them

### 5e: Docker-compose env var passthroughs

Add any model-specific environment variables to `dev/docker-compose.yml` under the `vllm-tt` service's `environment:` section, following the existing pattern.

## Step 6: Create Plan Directory

Create `/home/ttuser/src_docker/plan/<short_name>/` with:

### 6a: `overall.md` — Adapt from GLM plan structure

Include:
- Model architecture summary (from Step 0)
- Reference endpoint details
- Current baseline info (Qwen/known-good state)
- Phase map (same 7 phases)
- Hard constraints (Qwen non-regression, no CPU fallback, etc.)

### 6b: `checklist.md` — Quick-reference checklist

```markdown
# <Model> Bring-Up Checklist

## Before You Start Coding
1. Pick workspace: /home/ttuser/src_docker/ws/<short_name>
2. Confirm model cache is present
3. Confirm TT devices respond: tt-smi -ls
4. Confirm reference endpoint is reachable: curl -sf <ref_url>/models

## Before Any Merge
1. Run Qwen 32B non-regression: ./tests/qwen32b_smoke.sh 8088
2. If dependencies change: document why and rerun Qwen smoke

## Correctness Validation
1. Compare TT output vs reference endpoint for curated prompts
2. Token-by-token match for first 32 tokens (temperature=0, greedy)
3. If exact match fails, check logit PCC >= 0.98

## Performance Tracking
1. Run benchmark: scripts/benchmark_ref_vs_tt_<short_name>.py
2. Record tok/s for both reference and TT endpoints
3. Track delta across iterations
```

### 6c: `resume.md` — Living status document

Initialize with current state and immediate next steps.

## Step 7: Verify the Scaffold

Run these checks to confirm everything is wired correctly:

1. **Config loads**:
   ```bash
   cd $WORKSPACE/docker_tt
   docker compose --env-file dev/.env.<short_name> -f dev/docker-compose.yml run --rm vllm-tt \
     python3 -c "from vllm.transformers_utils.config import get_config; c = get_config('<hf_model_id>'); print(c.model_type, c.architectures)"
   ```

2. **Server starts** (even if forward is placeholder):
   ```bash
   docker compose --env-file dev/.env.<short_name> -f dev/docker-compose.yml up -d vllm-tt
   # Wait for health
   curl --retry 60 --retry-delay 2 --retry-all-errors -sf http://localhost:8088/health
   curl -sf http://localhost:8088/v1/models | python3 -m json.tool
   ```

3. **Reference endpoint is still up**:
   ```bash
   curl -sf <ref_base_url>/models
   ```

4. **Qwen non-regression**:
   ```bash
   docker compose --env-file dev/.env.qwen32b -f dev/docker-compose.yml up -d --force-recreate vllm-tt
   ./tests/qwen32b_smoke.sh 8088 Qwen/Qwen3-32B
   ```

5. **First A/B benchmark** (establishes baseline):
   ```bash
   python3 scripts/benchmark_ref_vs_tt_<short_name>.py \
     --ref-base <ref_base_url> --ref-model <ref_model> \
     --tt-base http://localhost:8088/v1 --tt-model <hf_model_id>
   ```

## Step 8: Commit the Scaffold

Create initial commits across all three repos. Framework improvements go on `main`/`dev`; model-specific code goes on the `<short_name>` branch.

Commit order:
1. **vllm** (config shim + registration) — smallest, most reviewable
2. **tt-metal** (model skeleton) — self-contained new directory
3. **docker_tt** (env + smoke + benchmark) — depends on the above

## Phase Guidance (After Scaffold)

After the scaffold is committed, proceed through phases. Each phase has gates:

| Phase | Focus | Gate |
|-------|-------|------|
| 1 | Baselines + guardrails | Qwen smoke passes, workspace works |
| 2 | vLLM enablement | Config loads, KV cache shape correct |
| 3 | Skeleton + integration | Server boots, /health returns OK |
| 4 | Layer 0 correctness | Embedding + attention + MLP match reference |
| 5 | KV cache + decode | Prefill+decode produces correct tokens |
| 6 | Full model | All layers + MoE/special, token-by-token match |
| 7 | Productionization | Perf target, stability soak, docs |

At every phase boundary:
1. Run `./tests/qwen32b_smoke.sh 8088` (non-regression)
2. Run `./tests/<short_name>_smoke.sh 8088` (model health)
3. Run `scripts/benchmark_ref_vs_tt_<short_name>.py` (A/B comparison vs reference)
4. Save artifacts under `plan/<short_name>/artifacts/`

## Key Patterns to Follow

- **Reuse existing TT primitives**: Look at `models/demos/deepseek_v3/tt/` for MoE/MLA patterns, `models/tt_transformers/` for standard attention
- **Lazy weight loading**: Never load the full state dict; use safetensors index
- **Cache converted TT tensors**: Include model snapshot hash + device type + dtype in cache key
- **Start replicated, then shard**: Begin with `ReplicateTensorToMesh` for correctness; convert to TP sharding for performance later
- **Gate everything by model_type**: All vLLM changes must be behind `model_type == "<model_type>"` guards
- **Reference endpoint is truth**: When in doubt about correctness, compare against the reference endpoint output
