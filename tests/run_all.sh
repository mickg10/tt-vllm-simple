#!/bin/bash
# Run all tests in sequence
# Usage: ./run_all.sh [port] [model]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${1:-8088}
MODEL=${2:-Qwen/Qwen3-0.6B}

echo "========================================"
echo "Running all tests"
echo "Port: $PORT"
echo "Model: $MODEL"
echo "========================================"

FAILED=0

echo ""
echo "--- Test 1: Health Check ---"
if "$SCRIPT_DIR/test_health.sh" "$PORT"; then
    echo ":: Health check passed"
else
    echo ":: Health check failed"
    FAILED=1
fi

echo ""
echo "--- Test 2: Model Listing ---"
if "$SCRIPT_DIR/test_models.sh" "$PORT" "$MODEL"; then
    echo ":: Model listing passed"
else
    echo ":: Model listing failed"
    FAILED=1
fi

echo ""
echo "--- Test 3: Chat Completion ---"
if "$SCRIPT_DIR/test_completion.sh" "$PORT" "$MODEL"; then
    echo ":: Chat completion passed"
else
    echo ":: Chat completion failed"
    FAILED=1
fi

echo ""
echo "========================================"
if [ $FAILED -eq 0 ]; then
    echo "All tests PASSED"
    exit 0
else
    echo "Some tests FAILED"
    exit 1
fi
