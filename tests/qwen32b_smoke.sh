#!/bin/bash
# Convenience wrapper for the always-on non-regression backtest.
# Usage: ./qwen32b_smoke.sh [port] [model]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${1:-8088}
MODEL=${2:-Qwen/Qwen3-32B}

exec "$SCRIPT_DIR/run_all.sh" "$PORT" "$MODEL"

