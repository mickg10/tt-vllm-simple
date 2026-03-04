#!/usr/bin/env python3
"""GLM-4.7-REAP-218B streaming benchmark — bs=1,4,8,32 performance matrix.

Measures TTFT, ITL (p50/p95/p99), per-sequence tok/s, and aggregate tok/s
using concurrent streaming requests via OpenAI async client.

Designed for Galaxy Wormhole (32 Wormhole chips, DP=4, TP=8, EP=32).
Tight DRAM (~143 KB/bank free at bs=1) — prompt/gen lengths kept conservative.

Usage:
    # Inside docker container or with port-forwarded vLLM:
    python3 tests/bench_reap_perf.py
    python3 tests/bench_reap_perf.py --batch-sizes 1,4 --max-tokens 50
    python3 tests/bench_reap_perf.py --prompt-lengths 100,500 --batch-sizes 1,4,8,32

    # From Galaxy host (port 8000):
    python3 tests/bench_reap_perf.py --endpoint http://localhost:8000/v1

Requirements:
    pip install openai  (or: pip install aiohttp)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "cerebras/GLM-4.7-REAP-218B-A32B"
DEFAULT_ENDPOINT = "http://localhost:8000/v1"

SEED_WORDS = [
    "river", "mountain", "signal", "archive", "lantern",
    "thread", "compass", "harbor", "forest", "clock",
    "memory", "bridge", "window", "machine", "storm",
    "garden", "copper", "engine", "library", "path",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class RequestMetrics:
    ok: bool
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    token_events: int = 0
    started_s: float = 0.0
    ended_s: float = 0.0
    ttft_s: Optional[float] = None
    itl_s: list[float] = field(default_factory=list)

    @property
    def latency_s(self) -> float:
        return max(0.0, self.ended_s - self.started_s)

    @property
    def completion_tokens_or_events(self) -> int:
        if self.completion_tokens is not None:
            return int(self.completion_tokens)
        return int(self.token_events)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _fmean(values: list[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _fmt_ms(seconds: Optional[float], digits: int = 1) -> str:
    if seconds is None:
        return "-"
    return f"{seconds * 1000:.{digits}f}"


def _fmt_float(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_prompt(target_tokens_approx: int, request_id: int = 0) -> str:
    """Build a prompt of approximately target_tokens_approx length."""
    prefix = (
        "You are running a performance benchmark. "
        "Read the context words and then respond with a single paragraph of plain text."
    )
    suffix = (
        "Task: Write one paragraph (no lists, no code blocks). "
        "Keep going until the server stops you due to max_tokens."
    )

    # Approximate: 1 word ~ 1.3 tokens for simple words
    filler_needed = max(0, int(target_tokens_approx) - 30)
    filler = " ".join(SEED_WORDS[i % len(SEED_WORDS)] for i in range(filler_needed))

    prompt = f"{prefix}\n\nContext: {filler}\n\n{suffix}"
    # Make each request unique to avoid prefix caching
    prompt += f"\n\n[benchmark_request_id={request_id}]"
    return prompt


# ---------------------------------------------------------------------------
# aiohttp streaming (no openai dependency needed)
# ---------------------------------------------------------------------------

async def stream_request_aiohttp(
    session,  # aiohttp.ClientSession
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
    request_id: int,
    ignore_eos: bool = True,
) -> RequestMetrics:
    """Send streaming /v1/chat/completions and measure token timing."""
    import aiohttp as _aiohttp

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    started_s = time.perf_counter()
    token_times: list[float] = []
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    try:
        timeout = _aiohttp.ClientTimeout(total=timeout_s, connect=30)
        async with session.post(
            url, json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                ended_s = time.perf_counter()
                return RequestMetrics(
                    ok=False, error=f"HTTP {resp.status}: {error_text[:200]}",
                    started_s=started_s, ended_s=ended_s,
                )

            async for line in resp.content:
                text = line.decode("utf-8").strip()
                if not text or not text.startswith("data: "):
                    continue
                data_str = text[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    now = time.perf_counter()

                    # Extract usage if present
                    usage = data.get("usage")
                    if usage:
                        if usage.get("prompt_tokens") is not None:
                            prompt_tokens = int(usage["prompt_tokens"])
                        if usage.get("completion_tokens") is not None:
                            completion_tokens = int(usage["completion_tokens"])

                    # Check for token content
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content") is not None:
                            token_times.append(now)
                except json.JSONDecodeError:
                    pass

    except asyncio.TimeoutError:
        ended_s = time.perf_counter()
        return RequestMetrics(
            ok=False, error=f"Timeout after {timeout_s}s",
            token_events=len(token_times),
            started_s=started_s, ended_s=ended_s,
            ttft_s=(token_times[0] - started_s) if token_times else None,
            itl_s=[token_times[i] - token_times[i - 1] for i in range(1, len(token_times))],
        )
    except Exception as e:
        ended_s = time.perf_counter()
        return RequestMetrics(
            ok=False, error=f"{type(e).__name__}: {e}",
            started_s=started_s, ended_s=ended_s,
        )

    ended_s = time.perf_counter()
    if not token_times:
        return RequestMetrics(
            ok=False, error="No tokens received",
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            started_s=started_s, ended_s=ended_s,
        )

    ttft_s = token_times[0] - started_s
    itl_s = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]

    return RequestMetrics(
        ok=True,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        token_events=len(token_times),
        started_s=started_s,
        ended_s=ended_s,
        ttft_s=ttft_s,
        itl_s=itl_s,
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def run_batch(
    endpoint: str,
    model: str,
    prompt_len: int,
    batch_size: int,
    max_tokens: int,
    timeout_s: float,
    ignore_eos: bool,
) -> tuple[dict[str, Any], list[RequestMetrics]]:
    """Run batch_size concurrent streaming requests."""
    import aiohttp

    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(batch_size):
            prompt = build_prompt(prompt_len, request_id=i)
            tasks.append(
                stream_request_aiohttp(
                    session, url, model, prompt, max_tokens,
                    timeout_s, i, ignore_eos,
                )
            )
        wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_end = time.perf_counter()

    wall_s = wall_end - wall_start
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    all_itls = [x for r in ok for x in r.itl_s]
    total_tokens = sum(r.completion_tokens_or_events for r in ok)

    seq_tps = []
    for r in ok:
        tokens = float(r.completion_tokens_or_events)
        dur = r.latency_s
        if dur > 0 and tokens > 0:
            seq_tps.append(tokens / dur)

    agg_tps = (total_tokens / wall_s) if wall_s > 0 and total_tokens > 0 else None

    summary = {
        "prompt_len": prompt_len,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "ok": len(ok),
        "total": len(results),
        "wall_s": wall_s,
        "ttft_mean_s": _fmean([float(x) for x in ttfts]) if ttfts else None,
        "itl_mean_s": _fmean([float(x) for x in all_itls]) if all_itls else None,
        "itl_p50_s": _percentile([float(x) for x in all_itls], 50) if all_itls else None,
        "itl_p95_s": _percentile([float(x) for x in all_itls], 95) if all_itls else None,
        "itl_p99_s": _percentile([float(x) for x in all_itls], 99) if all_itls else None,
        "seq_tps_mean": _fmean([float(x) for x in seq_tps]) if seq_tps else None,
        "agg_tps": agg_tps,
        "errors": [r.error for r in failed if r.error],
    }
    return summary, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> int:
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    prompt_lengths = [int(x) for x in args.prompt_lengths.split(",")]

    print(f"# GLM-4.7-REAP-218B Streaming Benchmark")
    print(f"")
    print(f"Timestamp: `{_now_utc()}`")
    print(f"Endpoint: `{args.endpoint}`")
    print(f"Model: `{args.model}`")
    print(f"Max tokens: `{args.max_tokens}`")
    print(f"Batch sizes: `{batch_sizes}`")
    print(f"Prompt lengths: `{prompt_lengths}`")
    print(f"Ignore EOS: `{args.ignore_eos}`")
    print()

    # Warmup
    if not args.no_warmup:
        print("## Warmup")
        print("Running 2 warmup requests...")
        for i in range(2):
            summary, _ = await run_batch(
                args.endpoint, args.model,
                prompt_len=50, batch_size=1,
                max_tokens=min(20, args.max_tokens),
                timeout_s=args.timeout, ignore_eos=args.ignore_eos,
            )
            status = "OK" if summary["ok"] > 0 else "FAILED"
            print(f"  Warmup {i+1}: {status}")
        print()

    # Main benchmark
    all_summaries: list[dict[str, Any]] = []

    for prompt_len in prompt_lengths:
        print(f"## Prompt ~{prompt_len} tokens, gen={args.max_tokens}")
        print()

        headers = [
            "BS", "OK", "TTFT(ms)", "ITL_mean(ms)", "ITL_p50(ms)",
            "ITL_p95(ms)", "ITL_p99(ms)", "Seq tok/s", "Agg tok/s", "Wall(s)",
        ]
        rows: list[list[str]] = []

        for bs in batch_sizes:
            print(f"  Running bs={bs}, prompt~{prompt_len}, gen={args.max_tokens}...", end="", flush=True)

            summary, results = await run_batch(
                args.endpoint, args.model,
                prompt_len=prompt_len, batch_size=bs,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout, ignore_eos=args.ignore_eos,
            )
            all_summaries.append(summary)

            errors = summary.get("errors", [])
            if errors:
                print(f" {len(errors)} errors")
                for e in errors[:2]:
                    print(f"    ERROR: {e[:120]}")
            else:
                print(f" OK ({summary['wall_s']:.1f}s)")

            rows.append([
                str(bs),
                f"{summary['ok']}/{summary['total']}",
                _fmt_ms(summary.get("ttft_mean_s")),
                _fmt_ms(summary.get("itl_mean_s")),
                _fmt_ms(summary.get("itl_p50_s")),
                _fmt_ms(summary.get("itl_p95_s")),
                _fmt_ms(summary.get("itl_p99_s")),
                _fmt_float(summary.get("seq_tps_mean")),
                _fmt_float(summary.get("agg_tps")),
                _fmt_float(summary.get("wall_s")),
            ])

            # Brief pause between runs
            await asyncio.sleep(2)

        print()
        print(_md_table(headers, rows))
        print()

    # Save JSON artifact
    artifact_dir = os.environ.get(
        "BENCH_ARTIFACT_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "plan", "glm47_reap_268b", "galaxy_wormhole", "artifacts"),
    )
    os.makedirs(artifact_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = os.path.join(artifact_dir, f"bench_reap_perf_{ts}.json")
    try:
        with open(artifact_path, "w") as f:
            json.dump({
                "timestamp": _now_utc(),
                "config": {
                    "model": args.model,
                    "endpoint": args.endpoint,
                    "max_tokens": args.max_tokens,
                    "batch_sizes": batch_sizes,
                    "prompt_lengths": prompt_lengths,
                    "ignore_eos": args.ignore_eos,
                },
                "results": all_summaries,
            }, f, indent=2, default=str)
        print(f"Artifact saved: {artifact_path}")
    except Exception as e:
        print(f"Warning: could not save artifact: {e}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="GLM-4.7-REAP-218B streaming benchmark (TTFT + ITL + throughput)"
    )
    parser.add_argument(
        "--endpoint", default=DEFAULT_ENDPOINT,
        help=f"OpenAI-compatible base URL (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-sizes", default="1,4,8,32",
        help="Comma-separated batch sizes (default: 1,4,8,32)",
    )
    parser.add_argument(
        "--prompt-lengths", default="100,500",
        help="Comma-separated approx prompt lengths in tokens (default: 100,500)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64,
        help="Max completion tokens per request (default: 64, keep low for tight DRAM)",
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0,
        help="Per-request timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--ignore-eos", action="store_true", default=True,
        help="Pass ignore_eos=true for consistent generation length (default: true)",
    )
    parser.add_argument(
        "--no-ignore-eos", dest="ignore_eos", action="store_false",
    )
    parser.add_argument(
        "--no-warmup", action="store_true", default=False,
        help="Skip warmup requests",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
