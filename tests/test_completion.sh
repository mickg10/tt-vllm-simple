#!/bin/bash
# Test that chat completion API works
# Usage: ./test_completion.sh [port] [model]

set -e

PORT=${1:-8088}
MODEL=${2:-Qwen/Qwen3-0.6B}

echo "Testing chat completion on port $PORT with model $MODEL..."

response=$(curl -sf "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in exactly 3 words.\"}],
        \"max_tokens\": 20,
        \"temperature\": 0
    }") || {
    echo "FAIL: Chat completion request failed"
    exit 1
}

# Check for a non-empty content in the first choice.
if printf '%s' "$response" | python3 -c '
import json, sys
obj = json.load(sys.stdin)
choices = obj.get("choices") or []
if not choices:
    raise SystemExit(2)
msg = (choices[0] or {}).get("message") or {}
content = msg.get("content")
if not isinstance(content, str) or not content.strip():
    raise SystemExit(3)
' 
then
    echo "PASS: Chat completion succeeded"
    echo "Response:"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    exit 0
else
    echo "FAIL: Invalid response (no choices)"
    echo "Response: $response"
    exit 1
fi
