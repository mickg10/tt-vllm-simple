#!/bin/bash
set -euo pipefail

# Ensure GLM-4.7-Flash weights are fully present in the local HF cache.
#
# Why this exists:
# - It's easy to end up with only a partial set of sharded safetensors locally.
# - Our tt-metal GLM bring-up can start without weights (placeholder logits), but
#   real inference requires all shards.
#
# Usage:
#   ./scripts/ensure_glm47_weights.sh
#   HF_TOKEN=... ./scripts/ensure_glm47_weights.sh

MODEL_ID="${MODEL_ID:-zai-org/GLM-4.7-Flash}"
# Known-good revision for weights. If the repo updates, you can override this.
REVISION="${REVISION:-a9308079ef95921451a690cd2d16cb572e564642}"
MAX_WORKERS="${MAX_WORKERS:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TT_METAL_DIR="${TT_METAL_DIR:-$WORKSPACE_ROOT/tt-metal}"

echo "[glm47] MODEL_ID=$MODEL_ID"
echo "[glm47] REVISION=$REVISION"
echo "[glm47] MAX_WORKERS=$MAX_WORKERS"
echo "[glm47] TT_METAL_DIR=$TT_METAL_DIR"

TOKEN_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  TOKEN_ARGS+=(--token "$HF_TOKEN")
fi

echo "[glm47] Downloading any missing safetensors shards into HF cache..."
uvx hf download "$MODEL_ID" \
  --revision "$REVISION" \
  --include "model-*.safetensors" \
  --max-workers "$MAX_WORKERS" \
  "${TOKEN_ARGS[@]}"

echo "[glm47] Verifying snapshot completeness..."
PYTHONPATH="$TT_METAL_DIR" python3 - <<'PY'
from models.demos.glm4_moe_lite.tt.weights import resolve_best_effort_snapshot_dir, find_missing_shards

snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
missing = find_missing_shards(snap)
print("snapshot:", snap)
if missing:
    raise SystemExit(f"missing {len(missing)} shards (example: {missing[0]})")
print("ok: all shards present")
PY

echo "[glm47] Done."
