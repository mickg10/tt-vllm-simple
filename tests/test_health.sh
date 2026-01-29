#!/bin/bash
# Test that vLLM health endpoint responds
# Usage: ./test_health.sh [port]

set -e

PORT=${1:-8088}
echo "Testing health endpoint on port $PORT..."

response=$(curl -sf "http://localhost:${PORT}/health" 2>&1) || {
    echo "FAIL: Health check failed"
    exit 1
}

echo "PASS: Health check succeeded"
exit 0
