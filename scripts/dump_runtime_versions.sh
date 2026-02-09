#!/bin/bash
# Dump runtime versions from a running vLLM container (best-effort).
# Usage: ./scripts/dump_runtime_versions.sh [variant]
#
# Variant defaults to `from_source` and expects a container named:
#   <variant>-vllm-tt-1
#
# This script is intended to produce a baseline snapshot for debugging and
# non-regression tracking (e.g., Qwen32B known-good state).
set -euo pipefail

VARIANT="${1:-from_source}"
CONTAINER="${VARIANT}-vllm-tt-1"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "ERROR: container '$CONTAINER' not found in 'docker ps'."
    echo "Hint: is the stack running? (e.g. 'make run-${VARIANT}')"
    exit 1
fi

echo "=== Runtime Snapshot ==="
echo "timestamp_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "variant: $VARIANT"
echo "container: $CONTAINER"

echo ""
echo "=== docker inspect (selected) ==="
docker inspect "$CONTAINER" --format 'image={{.Config.Image}}'
docker inspect "$CONTAINER" --format 'cmd={{json .Config.Cmd}}'

echo ""
echo "=== env (selected) ==="
docker exec "$CONTAINER" bash -lc 'printenv | egrep "^(HF_MODEL|HF_HOME|MESH_DEVICE|VLLM_TARGET_DEVICE|VLLM_RPC_TIMEOUT|MAX_MODEL_LEN|MAX_NUM_SEQS|TT_METAL_HOME|PYTHONPATH)=" || true'

echo ""
echo "=== git SHAs (best-effort) ==="
docker exec "$CONTAINER" bash -lc 'git -C /tt-metal rev-parse --short HEAD 2>/dev/null | sed "s/^/tt-metal: /" || echo "tt-metal: <no git>"'
docker exec "$CONTAINER" bash -lc 'git -C /vllm rev-parse --short HEAD 2>/dev/null | sed "s/^/vllm: /" || echo "vllm: <no git>"'

echo ""
echo "=== python packages ==="
docker exec -i "$CONTAINER" python - <<'PY'
import importlib.metadata as im
pkgs = ["vllm", "transformers", "torch"]
for p in pkgs:
    try:
        print(f"{p}: {im.version(p)}")
    except Exception as e:
        print(f"{p}: <missing> ({e})")
PY
