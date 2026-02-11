# GLM-4.7-Flash On TT: Worklog (WIP)

This file is a lightweight, in-repo pointer and snapshot for the `glm47_flash` effort.
The canonical planning/docs live outside git under `/home/ttuser/src_docker/plan/glm47_flash/`.

## Current Bring-Up Status (2026-02-11)
- TT endpoint: `http://localhost:8088/v1` (OpenAI-style)
- Open WebUI: `http://localhost:3000`
- TT monitor: `http://localhost:9090`
- Reference endpoint: `http://localhost:8087/v1` (full model: `zai-org/GLM-4.7-Flash`)

Correctness:
- Manual chat sanity: PASS (coherent English).
- Determinism at `temperature=0`: PASS (5 repeats; identical output).
- No KV-boundary corruption at `pos >= 64` (`--block-size=64`): PASS.

Perf snapshot (short prompts, end-to-end):
- Reference `:8087`: ~19 tok/s
- TT `:8088`: ~3 tok/s

## Root Cause Fixed: KV-Boundary Gibberish
Symptom:
- Greedy decode corrupts exactly when the second paged-KV block is first touched (`pos == block_size`).

Cause:
- FlashMLA decode with `fp32_dest_acc_en=True` corrupts greedy decode at the first KV block boundary on this stack.

Fix:
- `tt-metal` forces FlashMLA `fp32_dest_acc_en=False` unless explicitly enabled via an unsafe escape hatch.

## Key SHAs (Worktrees)
- `docker_tt`: `38e6035` (from_source fork repo override + worktree-safe workspace base)
- `tt-metal`: `3b63e3cc34` (FlashMLA fp32 dest acc safety gate)
- `vllm`: `b2fbf06a6` (optional page_table boundary logging)

## Dev Bring-Up (Worktree)
- GLM:
  - `cd /home/ttuser/src_docker/ws/glm47_flash/docker_tt`
  - `docker compose --env-file dev/.env.glm47 -f dev/docker-compose.yml up -d --force-recreate vllm-tt`
- Qwen32B regression gate:
  - `docker compose --env-file dev/.env.qwen32b -f dev/docker-compose.yml up -d --force-recreate vllm-tt`

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
- Plan/docs (canonical): `/home/ttuser/src_docker/plan/glm47_flash/`
- Bring-up runbook: `/home/ttuser/src_docker/plan/glm47_flash/resume.md`
- Porting playbook: `/home/ttuser/src_docker/plan/glm47_flash/migrate_model_to_tt.md`
