# Reference Endpoint Guide

The reference endpoint is the single source of truth for correctness during model bring-up. It runs the same model on different hardware (typically NVIDIA GPU) and serves an OpenAI-compatible API.

## Why a Reference Endpoint?

During TT bring-up, you need to answer "is my output correct?" at every phase. Comparing against a live reference endpoint is faster and more reliable than:
- Offline PyTorch reference scripts (which may have their own bugs)
- Comparing against logits files (which are fragile to serialize/deserialize)
- "Looks right to me" manual inspection

The reference endpoint also gives you a natural performance target and lets you run automated A/B benchmarks throughout the process.

## Setting Up the Reference

### Option A: vLLM on NVIDIA GPU (recommended)

On a separate machine with an NVIDIA GPU:

```bash
pip install vllm
vllm serve <hf_model_id> \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --dtype auto
```

### Option B: Any OpenAI-compatible server

Any server that implements the `/v1/chat/completions` endpoint works:
- [llama.cpp server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)
- [ollama](https://ollama.ai) (exposes OpenAI-compatible API)
- [SGLang](https://github.com/sgl-project/sglang)
- [TGI](https://github.com/huggingface/text-generation-inference)
- MLX (for Apple Silicon): `mlx_lm.server --model <model>`

### Option C: Cloud API

If the model is available via a cloud API (OpenAI, Together, Fireworks, etc.), you can use that. Just note:
- The model version/quantization may differ from your HF checkpoint
- Latency measurements won't be apples-to-apples
- Rate limits may slow down benchmarking

## Verifying the Reference

After setting up, verify:

```bash
# Check it's alive
curl -sf http://<host>:<port>/v1/models | python3 -m json.tool

# Check it produces output
curl -s http://<host>:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model_name>",
    "messages": [{"role": "user", "content": "Return exactly one word: blue"}],
    "max_tokens": 8,
    "temperature": 0
  }' | python3 -m json.tool

# Check streaming works
curl -s http://<host>:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model_name>",
    "messages": [{"role": "user", "content": "Count from 1 to 10"}],
    "max_tokens": 64,
    "temperature": 0,
    "stream": true
  }'
```

Record the model name exactly as returned by `/v1/models` — this is what you pass to `--ref-model` in benchmarks.

## Using the Reference During Bring-Up

### Phase 3-4 (Skeleton + Layer 0): Quick sanity checks

```bash
# Same prompt, same params, compare outputs
REF="http://<host>:<port>/v1"
TT="http://localhost:8088/v1"

for endpoint in "$REF" "$TT"; do
  curl -s "$endpoint/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"<model>","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16,"temperature":0}' \
    | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"
done
```

### Phase 5-6 (Decode + Full Model): Token-by-token comparison

Use the benchmark script with `temperature=0` to get deterministic outputs, then diff:

```bash
python3 scripts/benchmark_ref_vs_tt_<short_name>.py \
  --ref-base http://<host>:<port>/v1 \
  --ref-model <ref_model> \
  --tt-base http://localhost:8088/v1 \
  --tt-model <hf_model_id> \
  --temperature 0 \
  --max-tokens 32
```

### Phase 7 (Performance): Throughput comparison

```bash
python3 scripts/run_perf_iteration.py \
  --iteration-id <name>_001 \
  --suites single100x500 \
  --warmup-cases 40:80
```

## Network Considerations

- If the reference is on a different network, ensure the TT machine can reach it (firewall, VPN, etc.)
- For benchmarking, prefer a low-latency connection (same LAN) to avoid network variance dominating measurements
- If the reference is remote/high-latency, focus on output correctness (token matching) rather than latency comparisons
