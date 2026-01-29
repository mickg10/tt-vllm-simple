#!/bin/bash
# Test that model listing works and returns expected model
# Usage: ./test_models.sh [port] [expected_model]

set -e

PORT=${1:-8088}
EXPECTED_MODEL=${2:-Qwen/Qwen3-0.6B}

echo "Testing model listing on port $PORT..."
echo "Expected model: $EXPECTED_MODEL"

response=$(curl -sf "http://localhost:${PORT}/v1/models") || {
    echo "FAIL: Could not fetch models"
    exit 1
}

# Check if expected model is in response
if echo "$response" | grep -q "$EXPECTED_MODEL"; then
    echo "PASS: Model '$EXPECTED_MODEL' found"
    echo "Response:"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    exit 0
else
    echo "FAIL: Model '$EXPECTED_MODEL' not found in response"
    echo "Response: $response"
    exit 1
fi
