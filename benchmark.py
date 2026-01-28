#!/usr/bin/env python3
"""Quick benchmark script for vLLM TT server."""

import argparse
import time
import requests
import statistics

BASE_URL = "http://localhost:8088"

PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "Write a haiku about programming.",
    "What's 2 + 2?",
    "List 3 prime numbers.",
]


def get_model_name(base_url: str) -> str:
    """Get the model name from vLLM."""
    resp = requests.get(f"{base_url}/v1/models", timeout=5)
    return resp.json()["data"][0]["id"]


def query_llm(base_url: str, model: str, prompt: str, max_tokens: int = 64) -> tuple[str, float, int]:
    """Query the LLM and return (response, latency_ms, tokens)."""
    start = time.perf_counter()

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
        timeout=120,
    )
    resp.raise_for_status()

    elapsed_ms = (time.perf_counter() - start) * 1000
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    tokens = data["usage"]["completion_tokens"]

    return content, elapsed_ms, tokens


def run_benchmark(base_url: str, model: str, iterations: int = 10, max_tokens: int = 64):
    """Run benchmark loop."""
    print(f"Running {iterations} iterations with max_tokens={max_tokens}")
    print("=" * 60)

    latencies = []
    tokens_per_sec = []

    for i in range(iterations):
        prompt = PROMPTS[i % len(PROMPTS)]
        try:
            response, latency_ms, tokens = query_llm(base_url, model, prompt, max_tokens)
            tps = (tokens / latency_ms) * 1000 if latency_ms > 0 else 0

            latencies.append(latency_ms)
            tokens_per_sec.append(tps)

            print(f"[{i+1:3d}/{iterations}] {latency_ms:7.1f}ms | {tokens:3d} tokens | {tps:5.1f} tok/s")
            print(f"    Q: {prompt[:50]}...")
            print(f"    A: {response[:80]}...")
            print()

        except Exception as e:
            print(f"[{i+1:3d}/{iterations}] ERROR: {e}")

    if latencies:
        print("=" * 60)
        print("SUMMARY")
        print(f"  Iterations:  {len(latencies)}")
        print(f"  Avg latency: {statistics.mean(latencies):.1f} ms")
        print(f"  Min latency: {min(latencies):.1f} ms")
        print(f"  Max latency: {max(latencies):.1f} ms")
        if len(latencies) > 1:
            print(f"  Std dev:     {statistics.stdev(latencies):.1f} ms")
        print(f"  Avg tok/s:   {statistics.mean(tokens_per_sec):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM TT server")
    parser.add_argument("-n", "--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("-t", "--max-tokens", type=int, default=64, help="Max tokens per response")
    parser.add_argument("--url", type=str, default=BASE_URL, help="vLLM base URL")
    args = parser.parse_args()

    # Quick connectivity check
    try:
        model = get_model_name(args.url)
        print(f"Connected to vLLM at {args.url}")
        print(f"Model: {model}")
        print()
    except Exception as e:
        print(f"Error connecting to vLLM: {e}")
        return 1

    run_benchmark(args.url, model, args.iterations, args.max_tokens)
    return 0


if __name__ == "__main__":
    exit(main())
