#!/usr/bin/env python3
"""Template: Benchmark reference endpoint vs TT endpoint with identical requests.

Copy this to scripts/benchmark_ref_vs_tt_<short_name>.py and customize the
defaults (--ref-base, --ref-model, --tt-model).

Usage:
    python3 scripts/benchmark_ref_vs_tt_<short_name>.py
    python3 scripts/benchmark_ref_vs_tt_<short_name>.py --ref-base http://192.168.1.50:8000/v1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = [
    "Return exactly one English word: blue",
    "Write five concise bullet points on why unit tests matter in software projects.",
    "Summarize in 3 sentences: paged KV cache for autoregressive LLM inference.",
    "Write a short Python function fibonacci(n) using an iterative approach.",
]


@dataclass
class EndpointConfig:
    key: str
    base_url: str
    model: str


def _chat_once(
    *,
    endpoint: EndpointConfig,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": endpoint.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if extra_body:
        payload.update(extra_body)

    req = urllib.request.Request(
        endpoint.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    latency_s = time.perf_counter() - t0

    obj = json.loads(raw)
    usage = obj.get("usage", {}) or {}
    choices = obj.get("choices", []) or []
    first = choices[0] if choices else {}
    msg = (first.get("message") if isinstance(first, dict) else {}) or {}
    content = msg.get("content") or ""
    completion_tokens = int(usage.get("completion_tokens") or 0)
    tps = (completion_tokens / latency_s) if latency_s > 0 else 0.0

    return {
        "latency_s": latency_s,
        "completion_tokens": completion_tokens,
        "tokens_per_s": tps,
        "finish_reason": first.get("finish_reason"),
        "content_len": len(content),
        "content_preview": content[:200],
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {"avg_latency_s": 0.0, "avg_tokens_per_s": 0.0}
    latencies = [float(r["latency_s"]) for r in results]
    tps = [float(r["tokens_per_s"]) for r in results]
    return {
        "avg_latency_s": statistics.fmean(latencies),
        "avg_tokens_per_s": statistics.fmean(tps),
        "success_count": len(results),
    }


def _run_endpoint(
    endpoint: EndpointConfig,
    prompts: list[str],
    repeats: int,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    for prompt in prompts:
        for rep in range(repeats):
            case = {"prompt": prompt, "repeat_index": rep}
            try:
                case["result"] = _chat_once(
                    endpoint=endpoint,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    extra_body=extra_body,
                )
                case["ok"] = True
            except Exception as e:
                case["ok"] = False
                case["error"] = f"{type(e).__name__}: {e}"
            per_case.append(case)

    ok_results = [c["result"] for c in per_case if c.get("ok")]
    summary = _aggregate(ok_results)
    summary["total_count"] = len(per_case)
    return {"endpoint": endpoint.base_url, "model": endpoint.model, "summary": summary, "results": per_case}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # --- CUSTOMIZE THESE DEFAULTS FOR YOUR MODEL ---
    parser.add_argument("--ref-base", default="http://localhost:8087/v1",
                        help="Reference endpoint base URL")
    parser.add_argument("--ref-model", default="CHANGE_ME",
                        help="Model name at reference endpoint")
    parser.add_argument("--tt-base", default="http://localhost:8088/v1",
                        help="TT endpoint base URL")
    parser.add_argument("--tt-model", default="CHANGE_ME",
                        help="Model name at TT endpoint (usually HF model ID)")
    # --- END CUSTOMIZE ---
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--artifact-dir",
                        default="/home/ttuser/src_docker/plan/CHANGE_ME/artifacts")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = artifact_dir / f"benchmark_ref_vs_tt_{timestamp}.json"

    ref = EndpointConfig("ref", args.ref_base, args.ref_model)
    tt = EndpointConfig("tt", args.tt_base, args.tt_model)

    print(f"Running reference ({ref.base_url})...")
    ref_res = _run_endpoint(ref, DEFAULT_PROMPTS, args.repeats, args.max_tokens, args.temperature, args.timeout_s)
    print(f"Running TT ({tt.base_url})...")
    tt_res = _run_endpoint(tt, DEFAULT_PROMPTS, args.repeats, args.max_tokens, args.temperature, args.timeout_s)

    ref_tps = ref_res["summary"]["avg_tokens_per_s"]
    tt_tps = tt_res["summary"]["avg_tokens_per_s"]

    report = {
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "config": {"max_tokens": args.max_tokens, "temperature": args.temperature, "repeats": args.repeats},
        "ref": ref_res,
        "tt": tt_res,
        "delta_tps": tt_tps - ref_tps,
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nRef avg tok/s: {ref_tps:.3f}")
    print(f"TT  avg tok/s: {tt_tps:.3f}")
    print(f"Delta tok/s:   {tt_tps - ref_tps:+.3f}")
    print(f"Artifact: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
