# GLM-4.7-Flash On TT: Worklog (WIP)

This file is a lightweight, in-repo pointer and snapshot for the `glm47_flash` effort.
The canonical planning/docs live outside git under `/home/ttuser/src_docker/plan/glm47_flash/small_wormhole/`.

## Standard Dev Run

```bash
cd /home/ttuser/src_docker/ws/glm47_flash_small_wormhole/docker_tt
docker compose --env-file dev/.env.glm47 -f dev/docker-compose.yml \
  up -d --force-recreate vllm-tt
```

This is the **primary path** for GLM-4.7-Flash development. It uses the perf-trace-tp
config (tracing enabled, TP sharding, bf8 KV cache, batch-bucketed traces) which produces
~7 tok/s bs=1, ~208 tok/s aggregate bs=32 with coherent output.

**Correctness variant** (slower, ~3 tok/s, higher fidelity math):
```bash
docker compose --env-file dev/.env.glm47.correctness -f dev/docker-compose.yml \
  up -d --force-recreate vllm-tt
```

## Current Bring-Up Status (2026-02-11)

| Endpoint | URL | Description |
|----------|-----|-------------|
| TT vLLM | `http://localhost:8088/v1` | OpenAI-compatible API |
| Open WebUI | `http://localhost:3000` | Chat interface |
| TT Monitor | `http://localhost:9090` | Device monitoring |
| GPU Reference | `http://localhost:8087/v1` | Full model on NVIDIA GPU |

### Correctness
- Manual chat sanity: **PASS** (coherent English)
- Determinism at `temperature=0`: **PASS** (5 repeats; identical output)
- No KV-boundary corruption at `pos >= 64` (`--block-size=64`): **PASS**

### Performance (short prompts, end-to-end, 2026-02-14)
- Reference `:8087` (GPU): **~19 tok/s**
- TT `:8088` decode bs=1: **7.0 tok/s**, 143ms ITL (+56% from baseline)
- TT `:8088` decode bs=32: **208 tok/s** aggregate, 145ms ITL (**TARGET HIT >150**)
- TT `:8088` prefill 1k bs=1: **205 tok/s**, 4.9s TTFT
- TT `:8088` (correctness): **~3 tok/s**

### Key Optimizations (2026-02-14)
1. **Batch-bucketed traces**: Capture decode traces at B=1,4,8,16,32 instead of only B=32.
   Low-occupancy requests get more FlashMLA cores per sequence (bs=1: 16 cores vs 2).
   Implemented in model_tt.py (_DecodeTraceSamplingState) and tt_model_runner.py (bucket padding).
2. **Section 115 sparse MoE decode fix**: Changed `tokens > 1` to `tokens >= 33` in
   decoder_layer_tt.py so decode uses sparse MoE (fast) while prefill uses dense MoE (stable).

### Next Steps
- Target bs=1: 30 tok/s — needs MTP/speculative decode or deeper kernel optimization
- Target prefill: 1000 tok/s — needs pipeline parallelism or prefill-specific tracing
- Root cause of remaining gap: single-chip MoE latency, host-device round-trips in prefill

## Root Cause Fixed: KV-Boundary Gibberish
Symptom:
- Greedy decode corrupts exactly when the second paged-KV block is first touched (`pos == block_size`).

Cause:
- FlashMLA decode with `fp32_dest_acc_en=True` corrupts greedy decode at the first KV block boundary on this stack.

Fix:
- `tt-metal` forces FlashMLA `fp32_dest_acc_en=False` unless explicitly enabled via an unsafe escape hatch.

## Key SHAs (Worktrees)
- `docker_tt`: `cfe8358` (worklog pointer + from_source fork repo override + worktree-safe workspace base)
- `tt-metal`: `3b63e3cc34` (FlashMLA fp32 dest acc safety gate)
- `vllm`: `b2fbf06a6` (optional page_table boundary logging)

## Qwen32B Regression Gate
```bash
docker compose --env-file dev/.env.qwen32b -f dev/docker-compose.yml \
  up -d --force-recreate vllm-tt
```

## From-Source Bring-Up (Reproduce Current Code)
The `from_source/` variant now supports overriding repo URLs/refs via build args:
- `TT_METAL_REPO`, `TT_METAL_REF`
- `VLLM_REPO`, `VLLM_REF`

Example (requires branches pushed to your forks):
```bash
cd from_source
cat > .env <<'EOF'
TT_METAL_REPO=https://github.com/mickg10/tt-metal.git
TT_METAL_REF=glm47_flash
VLLM_REPO=https://github.com/mickg10/vllm.git
VLLM_REF=glm47_flash
HF_MODEL=zai-org/GLM-4.7-Flash
MESH_DEVICE=T3K
MAX_NUM_SEQS=1
OVERRIDE_TT_CONFIG={"trace_mode":"none","enable_model_warmup":false,"sample_on_device_mode":"decode_only"}
GLM_DEFAULT_DISABLE_THINKING=1
EOF
docker compose up -d --build
```

## References
- Plan/docs (canonical): `/home/ttuser/src_docker/plan/glm47_flash/small_wormhole/`
- Bring-up runbook: `/home/ttuser/src_docker/plan/glm47_flash/small_wormhole/resume.md`
- Porting playbook: `/home/ttuser/src_docker/plan/glm47_flash/small_wormhole/migrate_model_to_tt.md`
