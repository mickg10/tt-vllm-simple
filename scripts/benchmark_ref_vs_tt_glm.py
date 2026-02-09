#!/usr/bin/env python3
"""Benchmark GLM reference endpoint vs TT endpoint with identical requests.

Default behavior:
- Benchmarks reference first (localhost:8087), then TT (localhost:8088).
- Uses deterministic settings and disables thinking by default via
  chat_template_kwargs so content/latency comparisons are meaningful.
- Emits timestamped JSON + Markdown artifacts.
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
    disable_thinking: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": endpoint.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

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
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    completion_tokens = int(usage.get("completion_tokens") or 0)
    tps = (completion_tokens / latency_s) if latency_s > 0 else 0.0

    return {
        "latency_s": latency_s,
        "completion_tokens": completion_tokens,
        "tokens_per_s": tps,
        "finish_reason": first.get("finish_reason"),
        "content_len": len(content),
        "reasoning_len": len(reasoning),
        "content_preview": content[:200],
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {
            "avg_latency_s": 0.0,
            "p50_latency_s": 0.0,
            "p95_latency_s": 0.0,
            "avg_completion_tokens": 0.0,
            "avg_tokens_per_s": 0.0,
        }

    latencies = [float(r["latency_s"]) for r in results]
    completion_tokens = [float(r["completion_tokens"]) for r in results]
    tokens_per_s = [float(r["tokens_per_s"]) for r in results]

    p50 = statistics.median(latencies)
    if len(latencies) == 1:
        p95 = latencies[0]
    else:
        idx = max(0, min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1)))))
        p95 = sorted(latencies)[idx]

    return {
        "avg_latency_s": statistics.fmean(latencies),
        "p50_latency_s": p50,
        "p95_latency_s": p95,
        "avg_completion_tokens": statistics.fmean(completion_tokens),
        "avg_tokens_per_s": statistics.fmean(tokens_per_s),
    }


def _run_endpoint(
    *,
    endpoint: EndpointConfig,
    prompts: list[str],
    repeats: int,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    disable_thinking: bool,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    for prompt in prompts:
        for rep in range(repeats):
            case = {
                "prompt": prompt,
                "repeat_index": rep,
            }
            try:
                case["result"] = _chat_once(
                    endpoint=endpoint,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    disable_thinking=disable_thinking,
                )
                case["ok"] = True
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                case["ok"] = False
                case["error"] = f"HTTPError {e.code}: {body[:500]}"
            except Exception as e:  # pragma: no cover
                case["ok"] = False
                case["error"] = f"{type(e).__name__}: {e}"
            per_case.append(case)

    ok_results = [c["result"] for c in per_case if c.get("ok")]
    summary = _aggregate(ok_results)
    summary["success_count"] = len(ok_results)
    summary["total_count"] = len(per_case)
    summary["success_rate"] = (len(ok_results) / len(per_case)) if per_case else 0.0

    return {
        "endpoint": endpoint.base_url,
        "model": endpoint.model,
        "summary": summary,
        "results": per_case,
    }


def _write_markdown(report: dict[str, Any], md_path: Path) -> None:
    ref = report["runs"]["ref_8087"]["summary"]
    tt = report["runs"]["tt_8088"]["summary"]
    delta = report["delta_tt_minus_ref"]
    lines = [
        f"# Benchmark 8087 vs 8088 ({report['timestamp_utc']})",
        "",
        "## Config",
        f"- disable_thinking: `{report['config']['disable_thinking']}`",
        f"- max_tokens: `{report['config']['max_tokens']}`",
        f"- temperature: `{report['config']['temperature']}`",
        f"- repeats: `{report['config']['repeats']}`",
        "",
        "## ref_8087",
        f"- endpoint: `{report['runs']['ref_8087']['endpoint']}`",
        f"- model: `{report['runs']['ref_8087']['model']}`",
        f"- success: `{ref['success_count']}/{ref['total_count']}` ({ref['success_rate']:.2%})",
        f"- avg latency: `{ref['avg_latency_s']:.3f}s`",
        f"- p50 latency: `{ref['p50_latency_s']:.3f}s`",
        f"- p95 latency: `{ref['p95_latency_s']:.3f}s`",
        f"- avg completion tokens: `{ref['avg_completion_tokens']:.2f}`",
        f"- avg tokens/s: `{ref['avg_tokens_per_s']:.3f}`",
        "",
        "## tt_8088",
        f"- endpoint: `{report['runs']['tt_8088']['endpoint']}`",
        f"- model: `{report['runs']['tt_8088']['model']}`",
        f"- success: `{tt['success_count']}/{tt['total_count']}` ({tt['success_rate']:.2%})",
        f"- avg latency: `{tt['avg_latency_s']:.3f}s`",
        f"- p50 latency: `{tt['p50_latency_s']:.3f}s`",
        f"- p95 latency: `{tt['p95_latency_s']:.3f}s`",
        f"- avg completion tokens: `{tt['avg_completion_tokens']:.2f}`",
        f"- avg tokens/s: `{tt['avg_tokens_per_s']:.3f}`",
        "",
        "## Delta (tt_8088 - ref_8087)",
        f"- avg latency: `{delta['avg_latency_s']:+.3f}s`",
        f"- avg tokens/s: `{delta['avg_tokens_per_s']:+.3f}`",
        "",
        "## Artifacts",
        f"- `{report['artifact_json']}`",
        f"- `{report['artifact_md']}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-base", default="http://localhost:8087/v1")
    parser.add_argument("--ref-model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--tt-base", default="http://localhost:8088/v1")
    parser.add_argument("--tt-model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set chat_template_kwargs.enable_thinking=False",
    )
    parser.add_argument(
        "--artifact-dir",
        default="/home/ttuser/src_docker/plan/glm47_flash/artifacts",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = artifact_dir / f"benchmark_8087_vs_8088_{timestamp}.json"
    md_path = artifact_dir / f"benchmark_8087_vs_8088_{timestamp}.md"

    ref = EndpointConfig("ref_8087", args.ref_base, args.ref_model)
    tt = EndpointConfig("tt_8088", args.tt_base, args.tt_model)

    runs = {
        "ref_8087": _run_endpoint(
            endpoint=ref,
            prompts=DEFAULT_PROMPTS,
            repeats=args.repeats,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
            disable_thinking=args.disable_thinking,
        ),
        "tt_8088": _run_endpoint(
            endpoint=tt,
            prompts=DEFAULT_PROMPTS,
            repeats=args.repeats,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
            disable_thinking=args.disable_thinking,
        ),
    }

    ref_sum = runs["ref_8087"]["summary"]
    tt_sum = runs["tt_8088"]["summary"]
    delta = {
        "avg_latency_s": float(tt_sum["avg_latency_s"]) - float(ref_sum["avg_latency_s"]),
        "avg_tokens_per_s": float(tt_sum["avg_tokens_per_s"]) - float(ref_sum["avg_tokens_per_s"]),
    }

    report = {
        "timestamp_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "config": {
            "prompts": DEFAULT_PROMPTS,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "repeats": args.repeats,
            "disable_thinking": args.disable_thinking,
            "timeout_s": args.timeout_s,
        },
        "runs": runs,
        "delta_tt_minus_ref": delta,
        "artifact_json": str(json_path),
        "artifact_md": str(md_path),
    }

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)

    print(json.dumps({"artifact_json": str(json_path), "artifact_md": str(md_path)}, ensure_ascii=False))
    print(
        json.dumps(
            {
                "ref_avg_tps": round(ref_sum["avg_tokens_per_s"], 3),
                "tt_avg_tps": round(tt_sum["avg_tokens_per_s"], 3),
                "delta_avg_tps": round(delta["avg_tokens_per_s"], 3),
                "ref_avg_latency_s": round(ref_sum["avg_latency_s"], 3),
                "tt_avg_latency_s": round(tt_sum["avg_latency_s"], 3),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
