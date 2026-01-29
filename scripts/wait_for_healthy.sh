#!/bin/bash
# Wait for vLLM to be healthy
# Usage: ./wait_for_healthy.sh [timeout_seconds] [port]

TIMEOUT=${1:-300}
PORT=${2:-8088}
INTERVAL=5

echo "Waiting for vLLM to be healthy on port $PORT (timeout: ${TIMEOUT}s)..."

elapsed=0
while [ $elapsed -lt $TIMEOUT ]; do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "vLLM is healthy after ${elapsed}s"
        exit 0
    fi
    sleep $INTERVAL
    elapsed=$((elapsed + INTERVAL))
    echo "  Waiting... (${elapsed}s elapsed)"
done

echo "ERROR: vLLM did not become healthy within ${TIMEOUT}s"
exit 1
