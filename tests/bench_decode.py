#!/usr/bin/env python3
"""Benchmark decode throughput and prefill speed SEPARATELY.

Since prefix caching is not supported on TT V0 backend, we isolate
decode from prefill by using different test configurations:

  Test A (Decode): Short prompt (~10 tokens), long generation.
    Prefill is negligible, so measured throughput ≈ pure decode.
    per_user_decode_tps from streaming ITL is the definitive metric.
    aggregate_decode_tps = per_user × batch_size.

  Test B (Prefill): Long prompt (ctx_tokens), generate 1 token.
    TTFT ≈ prefill time for ctx_tokens.
    prefill_tps = ctx_tokens / TTFT.

  Test C (Combined): Long prompt + long generation (for reference only).
    Shows end-to-end throughput including both prefill and decode.

Targets:
  - Decode aggregate bs=32: 150 tok/s
  - Decode individual bs=1: 30 tok/s
  - Prefill: 1000 tokens/sec

Usage:
    python3 tests/bench_decode.py [--url URL] [--model MODEL]
    python3 tests/bench_decode.py --only-batch 1 --gen-tokens 100
    python3 tests/bench_decode.py --skip-combined  # faster, skip Test C
"""
import argparse
import asyncio
import json
import sys
import time

import aiohttp


# Default context sizes to test prefill at
PREFILL_CONTEXTS = [1000, 10000]
# Default generation length for decode test
DECODE_GEN = 500
# Default batch sizes
BATCHES = [1, 32]
# ~3.5 chars per token for English text
CHARS_PER_TOKEN = 3.5
# Short prompt for decode-only test (prefill negligible)
SHORT_PROMPT = "Hello, what is the meaning of life? Please write a detailed essay."


def make_prompt(target_tokens: int) -> str:
    """Generate a prompt that tokenizes to approximately target_tokens."""
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
    )
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    repeats = max(1, target_chars // len(base)) + 1
    return (base * repeats)[:target_chars]


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

    itls = []
    if len(token_times) > 1:
        for i in range(1, len(token_times)):
            itls.append(token_times[i] - token_times[i - 1])

    median_itl = sorted(itls)[len(itls) // 2] if itls else None
    # decode_tps: tokens per second EXCLUDING the first token (pure decode)
    decode_tps = (
        (tokens_received - 1) / (token_times[-1] - token_times[0])
        if len(token_times) > 1
        else 0.0
    )

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


def aggregate_results(results: list, batch_size: int, wall_total: float) -> dict:
    """Aggregate individual request results into batch summary."""
    successes = [r for r in results if r.get("error") is None and r["tokens"] > 0]

    if not successes:
        return {
            "success": 0,
            "total": batch_size,
            "wall_s": round(wall_total, 2),
            "errors": [r.get("error", "no tokens") for r in results],
        }

    ttfts = [r["ttft_s"] for r in successes if r["ttft_s"] is not None]
    decode_tps_list = [r["decode_tps"] for r in successes if r["decode_tps"] > 0]
    median_itls = [r["median_itl_ms"] for r in successes if r["median_itl_ms"] is not None]
    total_tokens = sum(r["tokens"] for r in successes)

    # Per-user decode tps (from streaming ITL, excludes prefill)
    avg_decode_tps = (
        round(sum(decode_tps_list) / len(decode_tps_list), 2) if decode_tps_list else 0
    )
    # Aggregate decode tps = per_user × successful_batch_size (true decode throughput)
    aggregate_decode_tps = round(avg_decode_tps * len(successes), 2)
    # Wall-clock aggregate (includes prefill in denominator — for reference)
    wall_aggregate_tps = round(total_tokens / wall_total, 2) if wall_total > 0 else 0

    return {
        "success": len(successes),
        "total": batch_size,
        "wall_s": round(wall_total, 2),
        "total_tokens": total_tokens,
        "wall_aggregate_tps": wall_aggregate_tps,
        "aggregate_decode_tps": aggregate_decode_tps,
        "avg_decode_tps": avg_decode_tps,
        "median_ttft_s": round(sorted(ttfts)[len(ttfts) // 2], 3) if ttfts else None,
        "median_itl_ms": round(sorted(median_itls)[len(median_itls) // 2], 1)
        if median_itls
        else None,
        "content_preview": successes[0].get("content_preview", "")[:30],
    }


async def run_test_a_decode(
    url: str, model: str, gen_tokens: int, batch_size: int, timeout: float
) -> dict:
    """Test A: Pure decode measurement with short prompt."""
    async with aiohttp.ClientSession() as session:
        t0 = time.monotonic()
        tasks = [
            stream_request(session, url, model, SHORT_PROMPT, gen_tokens, timeout, i)
            for i in range(batch_size)
        ]
        results = await asyncio.gather(*tasks)
        wall = time.monotonic() - t0

    agg = aggregate_results(results, batch_size, wall)
    return {
        "test": "A_decode",
        "prompt_tokens": "~10",
        "gen_tokens": gen_tokens,
        "batch_size": batch_size,
        **agg,
    }


async def run_test_b_prefill(
    url: str, model: str, ctx_tokens: int, batch_size: int, timeout: float
) -> dict:
    """Test B: Prefill measurement with long prompt, gen=1."""
    prompt = make_prompt(ctx_tokens)
    async with aiohttp.ClientSession() as session:
        t0 = time.monotonic()
        tasks = [
            stream_request(session, url, model, prompt, 1, timeout, i)
            for i in range(batch_size)
        ]
        results = await asyncio.gather(*tasks)
        wall = time.monotonic() - t0

    agg = aggregate_results(results, batch_size, wall)

    # Prefill tps: ctx_tokens / TTFT (single-request prefill speed)
    ttft = agg.get("median_ttft_s")
    prefill_tps = round(ctx_tokens / ttft, 1) if ttft and ttft > 0 else None

    return {
        "test": "B_prefill",
        "ctx_tokens": ctx_tokens,
        "gen_tokens": 1,
        "batch_size": batch_size,
        "prefill_tps": prefill_tps,
        **agg,
    }


async def run_test_c_combined(
    url: str, model: str, ctx_tokens: int, gen_tokens: int, batch_size: int, timeout: float
) -> dict:
    """Test C: Combined prefill + decode (end-to-end reference)."""
    prompt = make_prompt(ctx_tokens)
    async with aiohttp.ClientSession() as session:
        t0 = time.monotonic()
        tasks = [
            stream_request(session, url, model, prompt, gen_tokens, timeout, i)
            for i in range(batch_size)
        ]
        results = await asyncio.gather(*tasks)
        wall = time.monotonic() - t0

    agg = aggregate_results(results, batch_size, wall)
    return {
        "test": "C_combined",
        "ctx_tokens": ctx_tokens,
        "gen_tokens": gen_tokens,
        "batch_size": batch_size,
        **agg,
    }


def print_decode_result(r: dict) -> None:
    """Print Test A result."""
    ok = r.get("success", 0)
    bs = r["batch_size"]
    gen = r["gen_tokens"]
    dec_user = r.get("avg_decode_tps", 0)
    dec_agg = r.get("aggregate_decode_tps", 0)
    itl = r.get("median_itl_ms")
    ttft = r.get("median_ttft_s")
    wall = r.get("wall_s", 0)

    target_agg = 150 if bs >= 32 else (30 if bs == 1 else None)
    mark = ""
    if target_agg:
        if bs == 1:
            # For bs=1, aggregate_decode_tps == per_user_decode_tps
            mark = " OK" if dec_user >= target_agg else f" MISS (target: {target_agg})"
        else:
            mark = " OK" if dec_agg >= target_agg else f" MISS (target: {target_agg})"

    itl_s = f"{itl:.1f}ms" if itl else "N/A"
    ttft_s = f"{ttft:.2f}s" if ttft else "N/A"
    print(
        f"  [A] DECODE  bs={bs:>2} gen={gen:>4} | "
        f"per_user={dec_user:>6.1f} t/s  agg={dec_agg:>7.1f} t/s{mark}  "
        f"itl={itl_s:>8}  ttft={ttft_s:>7}  wall={wall:.1f}s  {ok}/{bs}"
    )


def print_prefill_result(r: dict) -> None:
    """Print Test B result."""
    ok = r.get("success", 0)
    bs = r["batch_size"]
    ctx = r["ctx_tokens"]
    ttft = r.get("median_ttft_s")
    pfill = r.get("prefill_tps")
    wall = r.get("wall_s", 0)

    mark = ""
    if pfill:
        mark = " OK" if pfill >= 1000 else f" MISS (target: 1000)"

    ttft_s = f"{ttft:.2f}s" if ttft else "N/A"
    pfill_s = f"{pfill:.0f}" if pfill else "N/A"
    print(
        f"  [B] PREFILL bs={bs:>2} ctx={ctx:>5} | "
        f"prefill={pfill_s:>6} tok/s{mark}  ttft={ttft_s:>7}  "
        f"wall={wall:.1f}s  {ok}/{bs}"
    )


def print_combined_result(r: dict) -> None:
    """Print Test C result."""
    ok = r.get("success", 0)
    bs = r["batch_size"]
    ctx = r["ctx_tokens"]
    gen = r["gen_tokens"]
    dec_user = r.get("avg_decode_tps", 0)
    dec_agg = r.get("aggregate_decode_tps", 0)
    wall_agg = r.get("wall_aggregate_tps", 0)
    ttft = r.get("median_ttft_s")
    itl = r.get("median_itl_ms")
    wall = r.get("wall_s", 0)

    itl_s = f"{itl:.1f}ms" if itl else "N/A"
    ttft_s = f"{ttft:.2f}s" if ttft else "N/A"
    print(
        f"  [C] E2E     bs={bs:>2} ctx={ctx:>5} gen={gen:>4} | "
        f"wall_agg={wall_agg:>6.1f} t/s  decode_agg={dec_agg:>6.1f} t/s  "
        f"per_user={dec_user:>5.1f}  ttft={ttft_s}  itl={itl_s}  wall={wall:.1f}s  {ok}/{bs}"
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark decode and prefill separately"
    )
    parser.add_argument("--url", default="http://localhost:8088")
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--only-batch", type=int, help="Only run this batch size")
    parser.add_argument("--gen-tokens", type=int, help="Override decode gen tokens")
    parser.add_argument("--batches", type=str, help="Comma-separated batch sizes")
    parser.add_argument("--skip-combined", action="store_true", help="Skip Test C (faster)")
    parser.add_argument("--prefill-contexts", type=str, help="Comma-separated prefill context sizes")
    args = parser.parse_args()

    batches = BATCHES
    if args.batches:
        batches = [int(x) for x in args.batches.split(",")]
    gen_tokens = args.gen_tokens or DECODE_GEN

    prefill_contexts = PREFILL_CONTEXTS
    if args.prefill_contexts:
        prefill_contexts = [int(x) for x in args.prefill_contexts.split(",")]

    # Health check
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                f"{args.url}/health", timeout=aiohttp.ClientTimeout(total=10)
            ) as r:
                if r.status != 200:
                    print(f"ERROR: health check failed: {r.status}")
                    sys.exit(1)
        except Exception as e:
            print(f"ERROR: cannot reach {args.url}: {e}")
            sys.exit(1)

    print("Decode + Prefill Benchmark (separated measurement)")
    print(f"URL: {args.url}  Model: {args.model}")
    print(f"Targets: decode_agg_bs32 >= 150 t/s | decode_bs1 >= 30 t/s | prefill >= 1000 tok/s")
    print(f"{'=' * 100}")

    all_results = []

    # ---- Test A: Pure decode (short prompt) ----
    print(f"\n{'='*40} TEST A: DECODE {'='*40}")
    print(f"Short prompt (~10 tokens), gen={gen_tokens}")
    for bs in batches:
        if args.only_batch is not None and bs != args.only_batch:
            continue
        print(f"  Running bs={bs} gen={gen_tokens}...", end="", flush=True)
        r = await run_test_a_decode(args.url, args.model, gen_tokens, bs, args.timeout)
        print(" done")
        print_decode_result(r)
        all_results.append(r)
        sys.stdout.flush()

    # ---- Test B: Prefill (long prompt, gen=1) ----
    print(f"\n{'='*40} TEST B: PREFILL {'='*39}")
    for ctx in prefill_contexts:
        for bs in batches:
            if args.only_batch is not None and bs != args.only_batch:
                continue
            print(f"  Running bs={bs} ctx={ctx} gen=1...", end="", flush=True)
            r = await run_test_b_prefill(args.url, args.model, ctx, bs, args.timeout)
            print(" done")
            print_prefill_result(r)
            all_results.append(r)
            sys.stdout.flush()

    # ---- Test C: Combined (optional) ----
    if not args.skip_combined:
        print(f"\n{'='*40} TEST C: COMBINED {'='*38}")
        for ctx in prefill_contexts:
            for bs in batches:
                if args.only_batch is not None and bs != args.only_batch:
                    continue
                print(f"  Running bs={bs} ctx={ctx} gen={gen_tokens}...", end="", flush=True)
                r = await run_test_c_combined(
                    args.url, args.model, ctx, gen_tokens, bs, args.timeout
                )
                print(" done")
                print_combined_result(r)
                all_results.append(r)
                sys.stdout.flush()

    # ---- Summary ----
    print(f"\n{'=' * 100}")
    print("SUMMARY vs TARGETS")
    print(f"{'=' * 100}")

    decode_results = [r for r in all_results if r["test"] == "A_decode"]
    prefill_results = [r for r in all_results if r["test"] == "B_prefill"]

    if decode_results:
        print("\nDECODE (from Test A — short prompt, pure decode):")
        for r in decode_results:
            bs = r["batch_size"]
            dec_user = r.get("avg_decode_tps", 0)
            dec_agg = r.get("aggregate_decode_tps", 0)
            target = 150 if bs >= 32 else (30 if bs == 1 else None)
            val = dec_user if bs == 1 else dec_agg
            status = "OK" if target and val >= target else "MISS"
            target_s = f" (target: {target})" if target else ""
            print(f"  bs={bs:>2}: {val:>7.1f} t/s  [{status}]{target_s}")

    if prefill_results:
        print("\nPREFILL (from Test B — long prompt, gen=1):")
        for r in prefill_results:
            bs = r["batch_size"]
            ctx = r["ctx_tokens"]
            pfill = r.get("prefill_tps")
            ttft = r.get("median_ttft_s")
            status = "OK" if pfill and pfill >= 1000 else "MISS"
            pfill_s = f"{pfill:.0f}" if pfill else "N/A"
            ttft_s = f"{ttft:.2f}s" if ttft else "N/A"
            print(f"  bs={bs:>2} ctx={ctx:>5}: {pfill_s:>6} tok/s  [{status}] (target: 1000)  ttft={ttft_s}")

    # Save JSON
    ts = int(time.time())
    output_path = f"/home/ttuser/src_docker/plan/glm47_flash/small_wormhole/artifacts/bench_decode_{ts}.json"
    with open(output_path, "w") as f:
        json.dump({"timestamp": time.time(), "results": all_results}, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
