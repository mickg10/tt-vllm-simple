#!/usr/bin/env python3
"""Performance iteration runner for GLM/Qwen across remote + TT local endpoints.

Key metrics:
- TTFT (time to first token)
- Prefill TPS (prompt_tokens / TTFT)
- Decode TPS ((completion_tokens - 1) / (total_latency - TTFT))
- End-to-end TPS (completion_tokens / total_latency)

Per user workflow this script:
1) Ensures tt-monitor service is running.
2) Runs benchmark suites against:
   - GLM remote endpoint (:8087)
   - GLM local TT endpoint (:8088)
   - Qwen local TT endpoint (:8088)
3) Writes JSON + markdown artifact for the iteration.
4) Appends iteration summary into perf-opt.md.
5) Restores GLM service on :8088 at the end.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def sh(cmd: list[str], cwd: Path, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k] = v
    return out


def write_env(path: Path, env_map: dict[str, str]) -> None:
    lines = [f"{k}={v}" for k, v in env_map.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def wait_health(url: str, timeout_s: int = 360) -> float:
    t0 = time.perf_counter()
    while True:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return time.perf_counter() - t0
        except Exception:
            pass
        if time.perf_counter() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for health: {url}")
        time.sleep(1)


def ensure_ttmonitor(cwd: Path, compose_file: str, env_file: str) -> None:
    # Keep tt-monitor always running as requested.
    out = subprocess.check_output(
        ["docker", "ps", "--format", "{{.Names}}"],
        text=True,
    )
    if "dev-tt-monitor-1" in out:
        return
    sh(
        [
            "docker",
            "compose",
            "--env-file",
            env_file,
            "-f",
            compose_file,
            "up",
            "-d",
            "tt-monitor",
        ],
        cwd,
    )


def count_words(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def build_prompt(prompt_words: int, gen_words: int) -> str:
    prefix = (
        f"You are writing a structured story benchmark output. "
        f"Generate exactly {gen_words} words in English. "
        f"Do not use bullet points. Keep coherent narrative flow. "
        f"Use clear paragraphing and avoid markdown formatting. "
        f"Context follows:"
    )
    seed_words = [
        "river",
        "mountain",
        "signal",
        "archive",
        "lantern",
        "thread",
        "compass",
        "harbor",
        "forest",
        "clock",
        "memory",
        "bridge",
        "window",
        "machine",
        "storm",
        "garden",
        "copper",
        "engine",
        "library",
        "path",
    ]
    prefix_count = count_words(prefix)
    if prompt_words <= prefix_count:
        # Minimal valid prompt if requested count is too small.
        return prefix

    needed = prompt_words - prefix_count
    filler = [seed_words[i % len(seed_words)] for i in range(needed)]
    return prefix + " " + " ".join(filler)


def words_to_max_tokens(gen_words: int, word_to_token_ratio: float) -> int:
    # Keep token budget close to requested word target to avoid runaway
    # generations that distort throughput measurements.
    ratio = max(1.0, float(word_to_token_ratio))
    return max(64, int(gen_words * ratio))


@dataclass
class Endpoint:
    key: str
    base: str
    model: str
    disable_thinking: bool


def stream_chat_metrics(
    endpoint: Endpoint,
    prompt: str,
    max_tokens: int,
    temperature: float,
    per_test_timeout_s: int,
    request_timeout_s: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": endpoint.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if endpoint.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    req = urllib.request.Request(
        endpoint.base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    first_token_at: float | None = None
    content_parts: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None

    try:
        with urllib.request.urlopen(req, timeout=max(1, int(request_timeout_s))) as resp:
            while True:
                if time.perf_counter() - t0 > per_test_timeout_s:
                    raise TimeoutError(f"per-test timeout ({per_test_timeout_s}s)")

                try:
                    raw = resp.readline()
                except (TimeoutError, socket.timeout) as e:
                    raise TimeoutError(f"stream read timeout ({request_timeout_s}s): {e}") from e
                if not raw:
                    break

                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                if not data:
                    continue

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if isinstance(obj, dict) and obj.get("usage"):
                    usage = obj.get("usage")

                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice0 = choices[0] or {}
                finish_reason = choice0.get("finish_reason") or finish_reason
                delta = choice0.get("delta") or {}
                piece = (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("reasoning_content")
                    or ""
                )
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    content_parts.append(piece)
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTPError {e.code}: {text[:400]}") from e

    t1 = time.perf_counter()
    total_latency_s = t1 - t0
    ttft_s = (first_token_at - t0) if first_token_at is not None else None
    content = "".join(content_parts)
    completion_words = count_words(content)

    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)

    e2e_tps = (completion_tokens / total_latency_s) if total_latency_s > 0 and completion_tokens > 0 else 0.0
    decode_window_s = (total_latency_s - (ttft_s or 0.0))
    decode_tps = ((completion_tokens - 1) / decode_window_s) if decode_window_s > 0 and completion_tokens > 1 else 0.0
    prefill_tps = (prompt_tokens / ttft_s) if ttft_s and ttft_s > 0 and prompt_tokens > 0 else 0.0

    return {
        "total_latency_s": total_latency_s,
        "ttft_s": ttft_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_words": completion_words,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "e2e_tps": e2e_tps,
        "finish_reason": finish_reason,
        "content_preview": content[:180],
    }


def prime_endpoint(endpoint: Endpoint, timeout_s: int) -> dict[str, Any]:
    """Force model weights/compilation/traces to be materialized before measurement.

    For TT backends, the first request can be dominated by weight conversion/upload
    and trace capture. Running a tiny streaming request up-front makes the measured
    benchmark cases representative of steady-state performance.
    """
    prompt = "Reply with exactly the single word: OK."
    return stream_chat_metrics(
        endpoint,
        prompt,
        max_tokens=8,
        temperature=0.0,
        per_test_timeout_s=int(timeout_s),
        request_timeout_s=int(timeout_s),
    )


def make_suite_repeat_1k_5k(repeats: int) -> list[dict[str, Any]]:
    return [{"suite": "repeat_1k_5k", "case_id": f"r{i+1}", "prompt_words": 1000, "gen_words": 5000} for i in range(repeats)]


def make_suite_linear_pairs_5() -> list[dict[str, Any]]:
    pairs = [(100, 500), (300, 1500), (500, 2500), (700, 3500), (1000, 5000)]
    out = []
    for i, (pw, gw) in enumerate(pairs, start=1):
        out.append({"suite": "linear_pairs_5", "case_id": f"p{i}", "prompt_words": pw, "gen_words": gw})
    return out


def make_suite_linear_pairs_10() -> list[dict[str, Any]]:
    out = []
    case_idx = 1
    for pw in range(100, 1001, 100):
        gw = pw * 5
        out.append({"suite": "linear_pairs_10", "case_id": f"p{case_idx}", "prompt_words": pw, "gen_words": gw})
        case_idx += 1
    return out


def make_suite_prefix_5() -> list[dict[str, Any]]:
    prompt_words = [1000, 1100, 1200, 1300, 1400]
    out = []
    for i, pw in enumerate(prompt_words, start=1):
        out.append({"suite": "prefix_5", "case_id": f"s{i}", "prompt_words": pw, "gen_words": 1000})
    return out


def make_suite_pair_100_500() -> list[dict[str, Any]]:
    return [{"suite": "pair_100_500", "case_id": "p1", "prompt_words": 100, "gen_words": 500}]


def make_suite_pair_1000_5000() -> list[dict[str, Any]]:
    return [{"suite": "pair_1000_5000", "case_id": "p1", "prompt_words": 1000, "gen_words": 5000}]


def parse_warmup_cases(spec: str) -> list[dict[str, Any]]:
    """Parse warmup spec like `100:200,300:600` into benchmark case dicts."""
    raw = spec.strip()
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw.split(","), start=1):
        piece = item.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"invalid warmup item {piece!r}; expected prompt_words:gen_words")
        ptxt, gtxt = piece.split(":", 1)
        pw = int(ptxt.strip())
        gw = int(gtxt.strip())
        if pw <= 0 or gw <= 0:
            raise ValueError(f"invalid warmup item {piece!r}; words must be > 0")
        out.append({"suite": "custom_warmup", "case_id": f"w{i}", "prompt_words": pw, "gen_words": gw})
    return out


def run_cases(
    endpoint: Endpoint,
    cases: list[dict[str, Any]],
    temperature: float,
    per_test_timeout_s: int,
    request_timeout_s: int,
    word_to_token_ratio: float,
    warmup_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    warmup_out: list[dict[str, Any]] = []
    if warmup_cases:
        for c in warmup_cases:
            print(
                f"[{endpoint.key}] warmup {c['suite']}:{c['case_id']} "
                f"prompt_w={c['prompt_words']} gen_w={c['gen_words']}",
                flush=True,
            )
            ptxt = build_prompt(c["prompt_words"], c["gen_words"])
            max_tokens = words_to_max_tokens(c["gen_words"], word_to_token_ratio)
            row = {"case": c}
            try:
                row["metrics"] = stream_chat_metrics(
                    endpoint,
                    ptxt,
                    max_tokens,
                    temperature,
                    per_test_timeout_s,
                    request_timeout_s,
                )
                row["ok"] = True
                print(
                    f"[{endpoint.key}] warmup ok ttft={row['metrics']['ttft_s']} "
                    f"e2e_tps={row['metrics']['e2e_tps']:.3f}",
                    flush=True,
                )
            except Exception as e:
                row["ok"] = False
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"[{endpoint.key}] warmup fail error={row['error']}", flush=True)
            warmup_out.append(row)

    rows: list[dict[str, Any]] = []
    total_cases = len(cases)
    for idx, c in enumerate(cases, start=1):
        print(
            f"[{endpoint.key}] test {idx}/{total_cases} {c['suite']}:{c['case_id']} "
            f"prompt_w={c['prompt_words']} gen_w={c['gen_words']}",
            flush=True,
        )
        ptxt = build_prompt(c["prompt_words"], c["gen_words"])
        max_tokens = words_to_max_tokens(c["gen_words"], word_to_token_ratio)
        row = {"case": c}
        try:
            row["metrics"] = stream_chat_metrics(
                endpoint,
                ptxt,
                max_tokens,
                temperature,
                per_test_timeout_s,
                request_timeout_s,
            )
            row["ok"] = True
            print(
                f"[{endpoint.key}] ok ttft={row['metrics']['ttft_s']} "
                f"decode_tps={row['metrics']['decode_tps']:.3f} "
                f"e2e_tps={row['metrics']['e2e_tps']:.3f}",
                flush=True,
            )
        except Exception as e:
            row["ok"] = False
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"[{endpoint.key}] fail error={row['error']}", flush=True)
        rows.append(row)

    ok = [r["metrics"] for r in rows if r.get("ok")]
    ttft = [float(x["ttft_s"]) for x in ok if x.get("ttft_s") is not None]
    e2e = [float(x["e2e_tps"]) for x in ok]
    dec = [float(x["decode_tps"]) for x in ok]
    summary = {
        "ok": len(ok),
        "total": len(rows),
        "success_rate": (len(ok) / len(rows)) if rows else 0.0,
        "avg_ttft_s": statistics.fmean(ttft) if ttft else 0.0,
        "avg_e2e_tps": statistics.fmean(e2e) if e2e else 0.0,
        "avg_decode_tps": statistics.fmean(dec) if dec else 0.0,
        "last_e2e_tps": float(ok[-1]["e2e_tps"]) if ok else 0.0,
        "last_decode_tps": float(ok[-1]["decode_tps"]) if ok else 0.0,
    }
    repeat_rows_ok = [
        r["metrics"]
        for r in rows
        if r.get("ok") and r.get("case", {}).get("suite") == "repeat_1k_5k"
    ]
    summary["repeat_1k_5k_count_ok"] = len(repeat_rows_ok)
    summary["repeat_1k_5k_last_e2e_tps"] = float(repeat_rows_ok[-1]["e2e_tps"]) if repeat_rows_ok else 0.0
    summary["repeat_1k_5k_last_decode_tps"] = float(repeat_rows_ok[-1]["decode_tps"]) if repeat_rows_ok else 0.0
    return {"endpoint": endpoint.base, "model": endpoint.model, "warmup": warmup_out, "rows": rows, "summary": summary}


def md_rows(rows: list[dict[str, Any]]) -> list[str]:
    out = [
        "| suite | case | prompt_w | gen_w | ok | ttft_s | prefill_tps | decode_tps | e2e_tps | prompt_tok | completion_tok | completion_words | finish |",
        "|---|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        c = r["case"]
        if not r.get("ok"):
            out.append(
                f"| {c['suite']} | {c['case_id']} | {c['prompt_words']} | {c['gen_words']} | no | - | - | - | - | - | - | - | error |"
            )
            continue
        m = r["metrics"]
        ttft = m["ttft_s"] if m["ttft_s"] is not None else 0.0
        out.append(
            f"| {c['suite']} | {c['case_id']} | {c['prompt_words']} | {c['gen_words']} | yes | "
            f"{ttft:.3f} | {m['prefill_tps']:.3f} | {m['decode_tps']:.3f} | {m['e2e_tps']:.3f} | "
            f"{m['prompt_tokens']} | {m['completion_tokens']} | {m['completion_words']} | {m['finish_reason']} |"
        )
    return out


def append_perf_opt(perf_opt: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    if not perf_opt.exists():
        lines.append("# perf-opt")
        lines.append("")
        lines.append("Iteration history and learnings for TT GLM performance.")
        lines.append("")

    lines.append(f"## Iteration {report['iteration_id']} ({report['timestamp_utc']})")
    lines.append("")
    lines.append(f"- target_tps: `{report['config']['target_tps']}`")
    lines.append(f"- ttmonitor_running: `{report['ttmonitor_running']}`")
    lines.append(f"- startup_glm_s: `{report['startup']['glm_s']:.1f}`")
    lines.append(f"- startup_qwen_s: `{report['startup']['qwen_s']:.1f}`")
    prime = report.get("prime") or {}
    if isinstance(prime, dict) and prime:
        glm_p = prime.get("glm_local_tt_8088") or {}
        qwen_p = prime.get("qwen_local_tt_8088") or {}
        if isinstance(glm_p, dict) and glm_p.get("ok"):
            m = glm_p.get("metrics") or {}
            lines.append(
                f"- prime_glm_tt: ttft_s=`{m.get('ttft_s')}`, decode_tps=`{float(m.get('decode_tps') or 0.0):.3f}`"
            )
        elif isinstance(glm_p, dict) and glm_p.get("error"):
            lines.append(f"- prime_glm_tt: failed error=`{glm_p.get('error')}`")
        if isinstance(qwen_p, dict) and qwen_p.get("ok"):
            m = qwen_p.get("metrics") or {}
            lines.append(
                f"- prime_qwen_tt: ttft_s=`{m.get('ttft_s')}`, decode_tps=`{float(m.get('decode_tps') or 0.0):.3f}`"
            )
        elif isinstance(qwen_p, dict) and qwen_p.get("error"):
            lines.append(f"- prime_qwen_tt: failed error=`{qwen_p.get('error')}`")
    lines.append("")
    for key in ("glm_remote_8087", "glm_local_tt_8088", "qwen_local_tt_8088"):
        s = report["results"][key]["summary"]
        lines.append(
            f"- {key}: avg_e2e_tps=`{s['avg_e2e_tps']:.3f}`, avg_decode_tps=`{s['avg_decode_tps']:.3f}`, "
            f"avg_ttft_s=`{s['avg_ttft_s']:.3f}`, success=`{s['ok']}/{s['total']}`"
        )
    lines.append(
        f"- delta_tt_glm_vs_remote_glm_e2e_tps: `{report['delta']['tt_glm_vs_remote_glm_e2e_tps']:+.3f}`"
    )
    lines.append(f"- delta_tt_glm_vs_qwen_e2e_tps: `{report['delta']['tt_glm_vs_qwen_e2e_tps']:+.3f}`")
    lines.append(f"- goal_reached: `{report['goal_reached']}`")
    lines.append("")
    lines.append(f"- artifact_json: `{report['artifact_json']}`")
    lines.append(f"- artifact_md: `{report['artifact_md']}`")
    lines.append("")
    perf_opt.write_text(
        perf_opt.read_text(encoding="utf-8") + "\n".join(lines) + "\n" if perf_opt.exists() else "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_md(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# Perf Iteration {report['iteration_id']}")
    lines.append("")
    lines.append(f"- timestamp_utc: `{report['timestamp_utc']}`")
    lines.append(f"- target_tps: `{report['config']['target_tps']}`")
    lines.append(f"- suites: `{report['config']['suites']}`")
    lines.append(f"- per_test_timeout_s: `{report['config']['per_test_timeout_s']}`")
    lines.append(f"- request_timeout_s: `{report['config']['request_timeout_s']}`")
    lines.append(f"- warmup_case_count: `{report['config']['warmup_case_count']}`")
    lines.append(f"- ttmonitor_running: `{report['ttmonitor_running']}`")
    lines.append("")
    lines.append("## Startup")
    lines.append(f"- glm_startup_s: `{report['startup']['glm_s']:.1f}`")
    lines.append(f"- qwen_startup_s: `{report['startup']['qwen_s']:.1f}`")
    lines.append("")

    prime = report.get("prime") or {}
    if isinstance(prime, dict) and prime:
        lines.append("## Prime")
        for key, val in prime.items():
            if not isinstance(val, dict):
                continue
            if val.get("ok"):
                m = val.get("metrics") or {}
                lines.append(
                    f"- {key}: ok ttft_s=`{m.get('ttft_s')}` "
                    f"prefill_tps=`{float(m.get('prefill_tps') or 0.0):.3f}` "
                    f"decode_tps=`{float(m.get('decode_tps') or 0.0):.3f}` "
                    f"e2e_tps=`{float(m.get('e2e_tps') or 0.0):.3f}`"
                )
            else:
                lines.append(f"- {key}: fail error=`{val.get('error')}`")
        lines.append("")
    for key in ("glm_remote_8087", "glm_local_tt_8088", "qwen_local_tt_8088"):
        sec = report["results"][key]
        s = sec["summary"]
        lines.append(f"## {key}")
        lines.append(f"- endpoint: `{sec['endpoint']}`")
        lines.append(f"- model: `{sec['model']}`")
        lines.append(f"- success: `{s['ok']}/{s['total']}` ({s['success_rate']:.2%})")
        lines.append(f"- avg_ttft_s: `{s['avg_ttft_s']:.3f}`")
        lines.append(f"- avg_e2e_tps: `{s['avg_e2e_tps']:.3f}`")
        lines.append(f"- avg_decode_tps: `{s['avg_decode_tps']:.3f}`")
        lines.append(f"- last_e2e_tps: `{s['last_e2e_tps']:.3f}`")
        lines.append(f"- last_decode_tps: `{s['last_decode_tps']:.3f}`")
        lines.append("")
        if sec["warmup"]:
            lines.append("### warmup")
            lines.extend(md_rows(sec["warmup"]))
            lines.append("")
        lines.append("### measured")
        lines.extend(md_rows(sec["rows"]))
        lines.append("")

    d = report["delta"]
    lines.append("## Delta")
    lines.append(f"- tt_glm_vs_remote_glm_e2e_tps: `{d['tt_glm_vs_remote_glm_e2e_tps']:+.3f}`")
    lines.append(f"- tt_glm_vs_qwen_e2e_tps: `{d['tt_glm_vs_qwen_e2e_tps']:+.3f}`")
    lines.append("")
    lines.append("## Goal")
    lines.append(f"- reached: `{report['goal_reached']}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- `{report['artifact_json']}`")
    lines.append(f"- `{report['artifact_md']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration-id", required=True)
    ap.add_argument("--target-tps", type=float, default=30.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--per-test-timeout-s", type=int, default=420)
    ap.add_argument("--request-timeout-s", type=int, default=120)
    ap.add_argument("--prime-timeout-s", type=int, default=2400)
    ap.add_argument("--health-timeout-s", type=int, default=360)
    ap.add_argument("--repeat-count", type=int, default=10)
    ap.add_argument("--suites", default="repeat10,linear10,prefix5")
    ap.add_argument("--warmup-cases", default="")
    ap.add_argument("--word-to-token-ratio", type=float, default=1.35)
    ap.add_argument("--artifact-dir", default="/home/ttuser/src_docker/plan/glm47_flash/artifacts/perf_iterations")
    ap.add_argument("--perf-opt", default="/home/ttuser/src_docker/plan/glm47_flash/perf-opt.md")
    ap.add_argument("--compose-file", default="dev/docker-compose.yml")
    ap.add_argument("--glm-env-file", default="dev/.env.glm47")
    ap.add_argument("--qwen-env-file", default="dev/.env.qwen32b")
    ap.add_argument("--glm-overrides-json", default="{}")
    ap.add_argument("--glm-remote-base", default="http://localhost:8087/v1")
    ap.add_argument("--glm-remote-model", default="zai-org/GLM-4.7-Flash")
    args = ap.parse_args()

    cwd = Path(__file__).resolve().parents[1]
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    try:
        glm_overrides = json.loads(args.glm_overrides_json)
        if not isinstance(glm_overrides, dict):
            raise ValueError("glm overrides must be object")
        glm_overrides = {str(k): str(v) for k, v in glm_overrides.items()}
    except Exception as e:
        raise SystemExit(f"invalid --glm-overrides-json: {e}")

    requested = [x.strip() for x in args.suites.split(",") if x.strip()]
    cases: list[dict[str, Any]] = []
    warmup_cases: list[dict[str, Any]] = []
    if args.warmup_cases.strip():
        try:
            warmup_cases.extend(parse_warmup_cases(args.warmup_cases))
        except Exception as e:
            raise SystemExit(f"invalid --warmup-cases: {e}")

    if "repeat10" in requested:
        warmup_cases.append({"suite": "repeat_1k_5k", "case_id": "warmup", "prompt_words": 1000, "gen_words": 5000})
        cases.extend(make_suite_repeat_1k_5k(args.repeat_count))
    if "single100x500" in requested:
        cases.extend(make_suite_pair_100_500())
    if "single1000x5000" in requested:
        cases.extend(make_suite_pair_1000_5000())
    if "linear5" in requested:
        cases.extend(make_suite_linear_pairs_5())
    if "linear10" in requested:
        cases.extend(make_suite_linear_pairs_10())
    if "prefix5" in requested:
        cases.extend(make_suite_prefix_5())
    if not cases:
        raise SystemExit("no cases selected")

    request_timeout_s = max(int(args.request_timeout_s), int(args.per_test_timeout_s) + 30)

    ensure_ttmonitor(cwd, args.compose_file, args.glm_env_file)
    ttmonitor_running = True

    # Endpoint 1: GLM remote reference (always first)
    glm_remote = Endpoint("glm_remote_8087", args.glm_remote_base, args.glm_remote_model, True)
    remote_res = run_cases(
        glm_remote,
        cases,
        args.temperature,
        args.per_test_timeout_s,
        request_timeout_s,
        args.word_to_token_ratio,
        warmup_cases=warmup_cases,
    )

    # Endpoint 2: GLM local TT (with optional env overrides)
    base_glm_env = load_env_file(cwd / args.glm_env_file)
    base_glm_env.update(glm_overrides)
    tmp_glm = cwd / ".tmp_perf_glm.env"
    write_env(tmp_glm, base_glm_env)
    try:
        sh(
            [
                "docker",
                "compose",
                "--env-file",
                str(tmp_glm),
                "-f",
                args.compose_file,
                "up",
                "-d",
                "--force-recreate",
                "vllm-tt",
            ],
            cwd,
        )
        startup_glm_s = wait_health(
            "http://localhost:8088/health",
            timeout_s=int(args.health_timeout_s),
        )
        glm_local = Endpoint("glm_local_tt_8088", "http://localhost:8088/v1", "zai-org/GLM-4.7-Flash", True)
        glm_prime: dict[str, Any] = {"ok": False}
        print(f"[{glm_local.key}] priming (timeout={int(args.prime_timeout_s)}s)...", flush=True)
        try:
            glm_prime_metrics = prime_endpoint(glm_local, int(args.prime_timeout_s))
            glm_prime = {"ok": True, "metrics": glm_prime_metrics}
            print(
                f"[{glm_local.key}] prime ok ttft={glm_prime_metrics.get('ttft_s')} "
                f"decode_tps={glm_prime_metrics.get('decode_tps'):.3f} e2e_tps={glm_prime_metrics.get('e2e_tps'):.3f}",
                flush=True,
            )
        except Exception as e:
            glm_prime = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"[{glm_local.key}] prime fail error={glm_prime['error']}", flush=True)
        glm_local_res = run_cases(
            glm_local,
            cases,
            args.temperature,
            args.per_test_timeout_s,
            request_timeout_s,
            args.word_to_token_ratio,
            warmup_cases=warmup_cases,
        )
    finally:
        tmp_glm.unlink(missing_ok=True)

    # Endpoint 3: Qwen local TT
    sh(
        [
            "docker",
            "compose",
            "--env-file",
            args.qwen_env_file,
            "-f",
            args.compose_file,
            "up",
            "-d",
            "--force-recreate",
            "vllm-tt",
        ],
        cwd,
    )
    startup_qwen_s = wait_health(
        "http://localhost:8088/health",
        timeout_s=int(args.health_timeout_s),
    )
    qwen_local = Endpoint("qwen_local_tt_8088", "http://localhost:8088/v1", "Qwen/Qwen3-32B", False)
    qwen_prime: dict[str, Any] = {"ok": False}
    print(f"[{qwen_local.key}] priming (timeout={int(args.prime_timeout_s)}s)...", flush=True)
    try:
        qwen_prime_metrics = prime_endpoint(qwen_local, int(args.prime_timeout_s))
        qwen_prime = {"ok": True, "metrics": qwen_prime_metrics}
        print(
            f"[{qwen_local.key}] prime ok ttft={qwen_prime_metrics.get('ttft_s')} "
            f"decode_tps={qwen_prime_metrics.get('decode_tps'):.3f} e2e_tps={qwen_prime_metrics.get('e2e_tps'):.3f}",
            flush=True,
        )
    except Exception as e:
        qwen_prime = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(f"[{qwen_local.key}] prime fail error={qwen_prime['error']}", flush=True)
    qwen_local_res = run_cases(
        qwen_local,
        cases,
        args.temperature,
        args.per_test_timeout_s,
        request_timeout_s,
        args.word_to_token_ratio,
        warmup_cases=warmup_cases,
    )

    # Restore GLM as default active server at end.
    sh(
        [
            "docker",
            "compose",
            "--env-file",
            args.glm_env_file,
            "-f",
            args.compose_file,
            "up",
            "-d",
            "--force-recreate",
            "vllm-tt",
        ],
        cwd,
    )
    _ = wait_health(
        "http://localhost:8088/health",
        timeout_s=int(args.health_timeout_s),
    )

    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    json_path = artifact_dir / f"iteration_{args.iteration_id}.json"
    md_path = artifact_dir / f"iteration_{args.iteration_id}.md"

    rr = remote_res["summary"]["avg_e2e_tps"]
    rg = glm_local_res["summary"]["avg_e2e_tps"]
    rq = qwen_local_res["summary"]["avg_e2e_tps"]

    report = {
        "iteration_id": args.iteration_id,
        "timestamp_utc": now,
        "config": {
            "target_tps": args.target_tps,
            "temperature": args.temperature,
            "per_test_timeout_s": args.per_test_timeout_s,
            "request_timeout_s": request_timeout_s,
            "prime_timeout_s": int(args.prime_timeout_s),
            "health_timeout_s": int(args.health_timeout_s),
            "repeat_count": args.repeat_count,
            "word_to_token_ratio": args.word_to_token_ratio,
            "suites": requested,
            "glm_overrides": glm_overrides,
            "glm_remote_base": args.glm_remote_base,
            "glm_remote_model": args.glm_remote_model,
            "case_count": len(cases),
            "warmup_case_count": len(warmup_cases),
        },
        "ttmonitor_running": ttmonitor_running,
        "startup": {"glm_s": startup_glm_s, "qwen_s": startup_qwen_s},
        "prime": {"glm_local_tt_8088": glm_prime, "qwen_local_tt_8088": qwen_prime},
        "results": {
            "glm_remote_8087": remote_res,
            "glm_local_tt_8088": glm_local_res,
            "qwen_local_tt_8088": qwen_local_res,
        },
        "delta": {
            "tt_glm_vs_remote_glm_e2e_tps": rg - rr,
            "tt_glm_vs_qwen_e2e_tps": rg - rq,
        },
        "goal_reached": rg >= args.target_tps,
        "artifact_json": str(json_path),
        "artifact_md": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(md_path, report)
    append_perf_opt(Path(args.perf_opt), report)

    print(json.dumps({"artifact_json": str(json_path), "artifact_md": str(md_path)}, ensure_ascii=False))
    print(
        json.dumps(
            {
                "glm_remote_avg_e2e_tps": round(rr, 3),
                "glm_local_avg_e2e_tps": round(rg, 3),
                "qwen_local_avg_e2e_tps": round(rq, 3),
                "goal_reached": report["goal_reached"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
