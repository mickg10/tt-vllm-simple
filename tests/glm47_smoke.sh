#!/bin/bash
# Convenience wrapper for a GLM-4.7-Flash E2E smoke check against an already-running server.
# Usage: ./glm47_smoke.sh [port] [model]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${1:-8088}
MODEL=${2:-zai-org/GLM-4.7-Flash}

"$SCRIPT_DIR/run_all.sh" "$PORT" "$MODEL"
"$SCRIPT_DIR/test_completions_logprobs.sh" "$PORT" "$MODEL"

echo "GLM smoke tests PASSED"

