#!/usr/bin/env python3
"""
Determinism probe for OpenAI-style /v1/chat/completions endpoints.

Purpose:
- Catch nondeterministic greedy decode regressions (temp=0) quickly.
- Produce a small markdown artifact suitable for pasting into the plan directory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True)
class ProbeResult:
    index: int
    latency_s: float
    content: str


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _render_markdown(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    repeats: int,
    temperature: float,
    max_tokens: int,
    stream: bool,
    chat_template_kwargs: dict[str, Any] | None,
    results: list[ProbeResult],
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    uniq = list(dict.fromkeys(r.content for r in results))
    status = "PASS" if len(uniq) == 1 else "FAIL"

    def fmt(s: str) -> str:
        # Keep the markdown readable without losing line breaks.
        return s.replace("\r\n", "\n").replace("\r", "\n")

    lines: list[str] = []
    lines.append(f"# Determinism Probe ({status})")
    lines.append("")
    lines.append(f"Timestamp (UTC): {ts}")
    lines.append("")
    lines.append("## Endpoint")
    lines.append(f"- `{endpoint}`")
    lines.append(f"- model: `{model}`")
    lines.append("")
    lines.append("## Request")
    lines.append(f"- `temperature={temperature}`")
    lines.append(f"- `max_tokens={max_tokens}`")
    lines.append(f"- `stream={str(stream).lower()}`")
    if chat_template_kwargs is not None:
        lines.append(f"- `chat_template_kwargs={json.dumps(chat_template_kwargs, sort_keys=True)}`")
    lines.append(f"- prompt: `{prompt}`")
    lines.append("")
    lines.append(f"## Results ({repeats} repeats, sequential)")
    for r in results:
        content = fmt(r.content)
        if "\n" in content:
            lines.append(f"- run {r.index}: {r.latency_s:.3f}s ->")
            lines.append("```")
            lines.append(content)
            lines.append("```")
        else:
            lines.append(f"- run {r.index}: {r.latency_s:.3f}s -> `{content}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- unique_outputs: {len(uniq)}")
    lines.append(f"- status: {status}")
    lines.append("")
    if status != "PASS":
        lines.append("## Interpretation")
        lines.append("Nondeterminism at `temperature=0` suggests one of:")
        lines.append("- async scheduling / missing synchronization")
        lines.append("- KV cache semantics or memory-lifetime bug")
        lines.append("- numerical nondeterminism in a reduction/top-k path that flips top-1 tokens")
        lines.append("")
        lines.append("Next: rerun this probe with TTNN async disabled (debug) and isolate whether it reproduces without vLLM.")
        lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default=os.environ.get("ENDPOINT", "http://localhost:8088/v1"))
    p.add_argument("--model", default=os.environ.get("MODEL", "zai-org/GLM-4.7-Flash"))
    p.add_argument("--prompt", default="Name 3 animals separated by commas.")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--stream", action="store_true")
    p.add_argument(
        "--chat-template-kwargs-json",
        default=None,
        help='JSON object passed as chat_template_kwargs, e.g. \'{"enable_thinking": false}\'',
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="If set, writes a markdown artifact to this directory.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    chat_template_kwargs = None
    if args.chat_template_kwargs_json is not None:
        chat_template_kwargs = json.loads(args.chat_template_kwargs_json)

    url = args.endpoint.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": bool(args.stream),
    }
    if chat_template_kwargs is not None:
        payload["chat_template_kwargs"] = chat_template_kwargs

    results: list[ProbeResult] = []
    for i in range(args.repeats):
        t0 = time.time()
        r = requests.post(url, json=payload, timeout=600)
        dt = time.time() - t0
        r.raise_for_status()
        j = r.json()
        content = j["choices"][0]["message"].get("content", "")
        results.append(ProbeResult(index=i + 1, latency_s=dt, content=content))
        print(f"run {i+1}: {dt:.3f}s -> {content!r}")

    uniq = list(dict.fromkeys(r.content for r in results))
    print("unique_outputs:", len(uniq))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"determinism_probe_{args.model.replace('/', '_')}_{_utc_now_compact()}.md"
        out_path.write_text(
            _render_markdown(
                endpoint=args.endpoint,
                model=args.model,
                prompt=args.prompt,
                repeats=args.repeats,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=bool(args.stream),
                chat_template_kwargs=chat_template_kwargs,
                results=results,
            )
            + "\n"
        )
        print("wrote:", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

