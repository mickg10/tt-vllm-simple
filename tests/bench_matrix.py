#!/usr/bin/env python3
"""Benchmark matrix: (context_len, gen_tokens) × batch_size.

Measures decode throughput via streaming SSE, reports per-user and aggregate stats.
Usage:
    python3 tests/bench_matrix.py [--url URL] [--model MODEL] [--timeout SECS]
"""
import argparse
import asyncio
import json
import sys
import time

import aiohttp


MATRIX = [
    # (context_tokens, gen_tokens)
    (1000, 500),
    (10000, 1000),
    (29000, 3000),
]
BATCHES = [1, 4, 8, 32]

# ~3.5 chars per token for English text with this tokenizer
CHARS_PER_TOKEN = 3.5


def make_prompt(target_tokens: int) -> str:
    """Generate a prompt that tokenizes to approximately target_tokens."""
    # Use a repeating pattern that tokenizes predictably.
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
    )
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    repeats = max(1, target_chars // len(base)) + 1
    prompt = (base * repeats)[:target_chars]
    return prompt


async def stream_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    request_id: int,
) -> dict:
    """Send a streaming chat completion and measure timing."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    t_start = time.monotonic()
    ttft = None
    token_times = []
    tokens_received = 0
    content_preview = ""
    error = None

    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {
                    "request_id": request_id,
                    "error": f"HTTP {resp.status}: {body[:200]}",
                    "tokens": 0,
                }
            async for line in resp.content:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    now = time.monotonic()
                    if ttft is None:
                        ttft = now - t_start
                    token_times.append(now)
                    tokens_received += 1
                    if len(content_preview) < 50:
                        content_preview += content
    except asyncio.TimeoutError:
        error = "timeout"
    except Exception as e:
        error = str(e)[:200]

    t_end = time.monotonic()
    wall_time = t_end - t_start

    # Compute inter-token latencies
    itls = []
    if len(token_times) > 1:
        for i in range(1, len(token_times)):
            itls.append(token_times[i] - token_times[i - 1])

    median_itl = sorted(itls)[len(itls) // 2] if itls else None
    decode_tps = (tokens_received - 1) / (token_times[-1] - token_times[0]) if len(token_times) > 1 else 0.0

    return {
        "request_id": request_id,
        "tokens": tokens_received,
        "ttft_s": ttft,
        "wall_s": wall_time,
        "median_itl_ms": median_itl * 1000 if median_itl else None,
        "decode_tps": decode_tps,
        "content_preview": content_preview[:40],
        "error": error,
    }


async def run_batch(
    url: str,
    model: str,
    ctx_tokens: int,
    gen_tokens: int,
    batch_size: int,
    timeout: float,
) -> dict:
    """Run batch_size concurrent streaming requests and aggregate results."""
    prompt = make_prompt(ctx_tokens)

    async with aiohttp.ClientSession() as session:
        t0 = time.monotonic()
        tasks = [
            stream_request(session, url, model, prompt, gen_tokens, timeout, i)
            for i in range(batch_size)
        ]
        results = await asyncio.gather(*tasks)
        wall_total = time.monotonic() - t0

    successes = [r for r in results if r.get("error") is None and r["tokens"] > 0]
    failures = [r for r in results if r not in successes]

    if not successes:
        return {
            "ctx_tokens": ctx_tokens,
            "gen_tokens": gen_tokens,
            "batch_size": batch_size,
            "success": 0,
            "total": batch_size,
            "wall_s": wall_total,
            "errors": [r.get("error", "no tokens") for r in failures],
        }

    ttfts = [r["ttft_s"] for r in successes if r["ttft_s"] is not None]
    decode_tps_list = [r["decode_tps"] for r in successes if r["decode_tps"] > 0]
    median_itls = [r["median_itl_ms"] for r in successes if r["median_itl_ms"] is not None]
    total_tokens = sum(r["tokens"] for r in successes)
    aggregate_tps = total_tokens / wall_total if wall_total > 0 else 0.0

    return {
        "ctx_tokens": ctx_tokens,
        "gen_tokens": gen_tokens,
        "batch_size": batch_size,
        "success": len(successes),
        "total": batch_size,
        "wall_s": round(wall_total, 2),
        "total_tokens": total_tokens,
        "aggregate_tps": round(aggregate_tps, 2),
        "avg_decode_tps": round(sum(decode_tps_list) / len(decode_tps_list), 2) if decode_tps_list else 0,
        "median_ttft_s": round(sorted(ttfts)[len(ttfts) // 2], 3) if ttfts else None,
        "median_itl_ms": round(sorted(median_itls)[len(median_itls) // 2], 1) if median_itls else None,
        "content_preview": successes[0].get("content_preview", "")[:30],
    }


def print_result(r: dict) -> None:
    """Print one benchmark result row."""
    status = f"{r['success']}/{r['total']}"
    ctx = r["ctx_tokens"]
    gen = r["gen_tokens"]
    bs = r["batch_size"]
    if r["success"] == 0:
        errs = r.get("errors", ["unknown"])
        print(f"  ctx={ctx:>5} gen={gen:>4} bs={bs:>2} | FAIL ({status}) errors: {errs[0][:60]}")
        return
    agg = r.get("aggregate_tps", 0)
    dec = r.get("avg_decode_tps", 0)
    ttft = r.get("median_ttft_s")
    itl = r.get("median_itl_ms")
    wall = r.get("wall_s", 0)
    tok = r.get("total_tokens", 0)
    ttft_str = f"{ttft:.2f}s" if ttft is not None else "N/A"
    itl_str = f"{itl:.1f}ms" if itl is not None else "N/A"
    print(
        f"  ctx={ctx:>5} gen={gen:>4} bs={bs:>2} | "
        f"{status} agg={agg:>7.1f}t/s dec={dec:>6.1f}t/s "
        f"ttft={ttft_str:>8} itl={itl_str:>8} wall={wall:>7.1f}s tok={tok:>6}"
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8088")
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--skip-long", action="store_true", help="Skip 29k context tests")
    parser.add_argument("--only-ctx", type=int, help="Only run this context size")
    parser.add_argument("--only-batch", type=int, help="Only run this batch size")
    args = parser.parse_args()

    # Check health first
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{args.url}/health", timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status != 200:
                    print(f"ERROR: health check failed with status {r.status}")
                    sys.exit(1)
        except Exception as e:
            print(f"ERROR: cannot reach {args.url}: {e}")
            sys.exit(1)

    print(f"Benchmark matrix: {args.url} model={args.model}")
    print(f"{'=' * 100}")

    all_results = []
    for ctx_tokens, gen_tokens in MATRIX:
        if args.only_ctx is not None and ctx_tokens != args.only_ctx:
            continue
        if args.skip_long and ctx_tokens >= 29000:
            print(f"\n--- Skipping ctx={ctx_tokens} gen={gen_tokens} (--skip-long) ---")
            continue
        print(f"\n--- ctx={ctx_tokens} gen={gen_tokens} ---")

        for batch_size in BATCHES:
            if args.only_batch is not None and batch_size != args.only_batch:
                continue

            # Estimate timeout: prefill + decode
            # Rough: prefill ~0.1s per 32 tokens, decode ~0.2s per token per user
            est_prefill = ctx_tokens * 0.1 / 32 * batch_size
            est_decode = gen_tokens * 0.25
            timeout = max(args.timeout, est_prefill + est_decode + 120)

            print(f"  Running ctx={ctx_tokens} gen={gen_tokens} bs={batch_size} (timeout={timeout:.0f}s)...", end="", flush=True)
            t0 = time.monotonic()
            result = await run_batch(args.url, args.model, ctx_tokens, gen_tokens, batch_size, timeout)
            elapsed = time.monotonic() - t0
            print(f" done ({elapsed:.1f}s)")
            print_result(result)
            all_results.append(result)

            # Flush results progressively
            sys.stdout.flush()

    # Summary table
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    print(f"{'ctx':>6} {'gen':>5} {'bs':>3} | {'status':>5} {'agg_tps':>9} {'dec_tps':>9} {'ttft':>9} {'itl':>9} {'wall':>8}")
    print(f"{'-' * 6} {'-' * 5} {'-' * 3} | {'-' * 5} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 8}")
    for r in all_results:
        status = f"{r['success']}/{r['total']}"
        agg = r.get("aggregate_tps", 0)
        dec = r.get("avg_decode_tps", 0)
        ttft = r.get("median_ttft_s")
        itl = r.get("median_itl_ms")
        wall = r.get("wall_s", 0)
        ttft_str = f"{ttft:.2f}s" if ttft is not None else "N/A"
        itl_str = f"{itl:.1f}ms" if itl is not None else "N/A"
        print(
            f"{r['ctx_tokens']:>6} {r['gen_tokens']:>5} {r['batch_size']:>3} | "
            f"{status:>5} {agg:>8.1f} {dec:>8.1f} {ttft_str:>9} {itl_str:>9} {wall:>7.1f}s"
        )

    # Write JSON output
    output_path = f"/home/ttuser/src_docker/plan/glm47_flash/artifacts/bench_matrix_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump({"timestamp": time.time(), "results": all_results}, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
