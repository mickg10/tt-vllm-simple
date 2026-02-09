#!/bin/bash
# Test that /v1/completions supports logprobs without crashing the TT backend.
# Usage: ./test_completions_logprobs.sh [port] [model]

set -euo pipefail

PORT=${1:-8088}
MODEL=${2:-zai-org/GLM-4.7-Flash}

echo "Testing completions logprobs on port $PORT with model $MODEL..."

response=$(curl -sf "http://localhost:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"prompt\": \"Hello\",
        \"max_tokens\": 8,
        \"temperature\": 0,
        \"logprobs\": 2,
        \"stream\": false
    }") || {
    echo "FAIL: /v1/completions request failed"
    exit 1
}

if printf '%s' "$response" | python3 -c '
import json, sys
obj = json.load(sys.stdin)
choices = obj.get("choices") or []
if not choices:
    raise SystemExit("no choices")
choice0 = choices[0] or {}
text = choice0.get("text")
if not isinstance(text, str) or not text.strip():
    raise SystemExit("empty text")
lp = choice0.get("logprobs")
if not isinstance(lp, dict):
    raise SystemExit("missing logprobs")
tokens = lp.get("tokens")
token_logprobs = lp.get("token_logprobs")
if not isinstance(tokens, list) or not tokens:
    raise SystemExit("logprobs.tokens missing/empty")
if not isinstance(token_logprobs, list) or not token_logprobs:
    raise SystemExit("logprobs.token_logprobs missing/empty")
'
then
    echo "PASS: /v1/completions logprobs succeeded"
    exit 0
else
    echo "FAIL: Invalid /v1/completions logprobs response"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    exit 1
fi

