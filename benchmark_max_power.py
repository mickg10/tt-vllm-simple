#!/usr/bin/env python3
"""
Benchmark to maximize TT device power consumption.
Monitors power usage correlated with inference activity.
Tests different strategies: long prefills vs continuous decode.
"""

import argparse
import requests
import time
import threading
import json
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Optional

# API endpoints
VLLM_API_URL = "http://localhost:8088/v1/chat/completions"
TT_MONITOR_URL = "http://localhost:9090/api/devices"

# Strategy: Long prompts + short outputs = maximum prefill compute = max power
# Tested: 297W peak with ~2500 token prompts and 32 token outputs

# Long prompt for heavy prefill (maximize initial compute)
LONG_PROMPT = """You are a highly capable AI assistant. I need you to help me with a complex analysis task.

Here is extensive background context that you should consider carefully:

The development of artificial intelligence has been one of the most significant technological achievements of the 21st century. From early perceptrons to modern transformer architectures, the field has undergone remarkable evolution. The attention mechanism, introduced in the landmark "Attention Is All You Need" paper, revolutionized how neural networks process sequential data.

Key milestones include:
1. The development of backpropagation algorithms in the 1980s
2. Convolutional neural networks for image recognition in the 1990s
3. Long short-term memory (LSTM) networks for sequence modeling
4. The transformer architecture and self-attention mechanisms
5. Large language models with billions of parameters
6. Multimodal models combining vision and language

Hardware acceleration has been crucial to these advances. Graphics processing units (GPUs) initially designed for gaming proved remarkably effective for parallel matrix operations. This led to specialized AI accelerators from companies like NVIDIA, Google (TPUs), and Tenstorrent with their innovative Wormhole architecture.

The Wormhole architecture features:
- Tensix cores optimized for neural network operations
- High-bandwidth memory interfaces
- Efficient data movement through NoC (Network-on-Chip)
- Support for various numerical precisions including BF16 and FP8

Consider the following detailed technical specifications and requirements for our analysis:

Memory bandwidth requirements scale with model size and batch size. For a 32B parameter model with BF16 weights, we need approximately 64GB just for model weights. KV cache requirements grow with sequence length and batch size, following the formula: KV_cache_size = 2 * num_layers * hidden_size * num_heads * seq_len * batch_size * bytes_per_element.

Power consumption in AI accelerators comes from several sources:
- Compute operations (matrix multiplications, activations)
- Memory access (DRAM reads/writes, SRAM access)
- Data movement (on-chip interconnects, PCIe transfers)
- Static power (leakage current, always-on circuits)

During prefill (prompt processing), the workload is compute-bound with high arithmetic intensity. During decode (token generation), the workload becomes memory-bound as each token requires reading the full KV cache.

Now, with all this context in mind, please provide a detailed and comprehensive response to the following:

Explain how modern AI accelerators optimize for both compute-bound and memory-bound phases of inference, and discuss the tradeoffs involved in different architectural choices. Be thorough and technical in your response."""

# Short prompt for rapid decode testing
SHORT_PROMPT = "Continue generating text about AI and technology. Be detailed and thorough."


@dataclass
class PowerReading:
    timestamp: float
    total_power: float
    device_powers: list
    temperatures: list


@dataclass
class Stats:
    requests: int = 0
    tokens: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    total_time: float = 0.0


# Global state
stats_lock = threading.Lock()
stats = Stats()
power_history = deque(maxlen=1000)
tps_history = deque(maxlen=100)
running = True


def get_power() -> Optional[PowerReading]:
    """Get current power reading from TT monitor."""
    try:
        resp = requests.get(TT_MONITOR_URL, timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            devices = data.get("devices", [])
            return PowerReading(
                timestamp=time.time(),
                total_power=data.get("totals", {}).get("power", 0),
                device_powers=[d.get("power", 0) for d in devices],
                temperatures=[d.get("temperature", 0) for d in devices],
            )
    except Exception:
        pass
    return None


def power_monitor(interval: float = 0.5):
    """Thread to continuously monitor power."""
    global power_history
    while running:
        reading = get_power()
        if reading:
            power_history.append(reading)
        time.sleep(interval)


def worker_long_prefill(thread_id: int, model: str, max_tokens: int):
    """Worker that sends long prompts for heavy prefill."""
    global stats

    while running:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": LONG_PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }

        try:
            start = time.time()
            response = requests.post(VLLM_API_URL, json=payload, timeout=300)
            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                with stats_lock:
                    stats.requests += 1
                    stats.tokens += completion_tokens
                    stats.prefill_tokens += prompt_tokens
                    stats.decode_tokens += completion_tokens
                    stats.total_time += elapsed
                    if elapsed > 0:
                        tps_history.append(completion_tokens / elapsed)

        except Exception as e:
            if running:
                time.sleep(1)


def worker_continuous_decode(thread_id: int, model: str, max_tokens: int):
    """Worker that generates many tokens with shorter prompts."""
    global stats
    context = SHORT_PROMPT

    while running:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": context}],
            "max_tokens": max_tokens,
            "temperature": 0.8,
        }

        try:
            start = time.time()
            response = requests.post(VLLM_API_URL, json=payload, timeout=300)
            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                # Get response and append to context
                choices = data.get("choices", [])
                if choices:
                    response_text = choices[0].get("message", {}).get("content", "")
                    if response_text:
                        # Keep last 500 chars to maintain context
                        context = response_text[-500:] + "\n\nContinue:"

                with stats_lock:
                    stats.requests += 1
                    stats.tokens += completion_tokens
                    stats.prefill_tokens += prompt_tokens
                    stats.decode_tokens += completion_tokens
                    stats.total_time += elapsed
                    if elapsed > 0:
                        tps_history.append(completion_tokens / elapsed)

        except Exception as e:
            if running:
                time.sleep(1)


def stats_reporter(interval: int = 5):
    """Report stats with power correlation."""
    global stats

    last_tokens = 0
    last_time = time.time()
    max_power_seen = 0

    print("\n" + "=" * 100)
    print(f"{'Time':>8} | {'Reqs':>6} | {'Tokens':>10} | {'TPS':>7} | {'Power':>8} | {'Max Pwr':>8} | {'Temps':>20} | {'Phase'}")
    print("=" * 100)

    while running:
        time.sleep(interval)

        # Get current stats
        with stats_lock:
            current_tokens = stats.tokens
            current_requests = stats.requests
            current_prefill = stats.prefill_tokens
            current_decode = stats.decode_tokens
            avg_tps = sum(tps_history) / len(tps_history) if tps_history else 0

        # Calculate instant TPS
        now = time.time()
        delta_tokens = current_tokens - last_tokens
        delta_time = now - last_time
        instant_tps = delta_tokens / delta_time if delta_time > 0 else 0

        # Get power reading
        power_reading = get_power()
        power_str = "N/A"
        temps_str = "N/A"
        if power_reading:
            power_str = f"{power_reading.total_power:>6.0f}W"
            max_power_seen = max(max_power_seen, power_reading.total_power)
            temps = power_reading.temperatures
            temps_str = "/".join(f"{t:.0f}" for t in temps[:4])

        # Determine phase (more prefill or decode?)
        phase = "mixed"
        if current_prefill > current_decode * 2:
            phase = "PREFILL"
        elif current_decode > current_prefill * 2:
            phase = "DECODE"

        print(f"{datetime.now().strftime('%H:%M:%S'):>8} | "
              f"{current_requests:>6} | "
              f"{current_tokens:>10,} | "
              f"{instant_tps:>6.1f} | "
              f"{power_str:>8} | "
              f"{max_power_seen:>6.0f}W | "
              f"{temps_str:>20} | "
              f"{phase}")

        last_tokens = current_tokens
        last_time = now


def run_phase(name: str, worker_fn, threads: int, model: str, max_tokens: int, duration: int):
    """Run a specific test phase."""
    global stats, running

    print(f"\n>>> Starting {name} phase: {threads} threads, {max_tokens} max_tokens, {duration}s")

    # Reset stats
    with stats_lock:
        stats = Stats()

    # Start workers
    workers = []
    for i in range(threads):
        t = threading.Thread(target=worker_fn, args=(i, model, max_tokens), daemon=True)
        t.start()
        workers.append(t)
        time.sleep(0.05)  # Stagger starts

    # Let it run
    time.sleep(duration)

    # Get final power samples
    final_powers = []
    for _ in range(10):
        reading = get_power()
        if reading:
            final_powers.append(reading.total_power)
        time.sleep(0.2)

    avg_power = sum(final_powers) / len(final_powers) if final_powers else 0
    max_power = max(final_powers) if final_powers else 0

    with stats_lock:
        final_tps = sum(tps_history) / len(tps_history) if tps_history else 0

    print(f"<<< {name} complete: avg_power={avg_power:.0f}W, max_power={max_power:.0f}W, avg_tps={final_tps:.1f}")

    return avg_power, max_power, final_tps


def main():
    global running

    parser = argparse.ArgumentParser(description="Max power benchmark for TT devices")
    parser.add_argument("--threads", "-t", type=int, default=32,
                        help="Number of concurrent threads (default: 32)")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen3-32B",
                        help="Model name")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens per request (default: 512)")
    parser.add_argument("--duration", "-d", type=int, default=60,
                        help="Duration per phase in seconds (default: 60)")
    parser.add_argument("--mode", choices=["prefill", "decode", "mixed", "sweep", "max"], default="sweep",
                        help="Test mode: prefill, decode, mixed, sweep, or max (optimized for peak power)")
    args = parser.parse_args()

    print(f"=" * 100)
    print(f"TT MAX POWER BENCHMARK")
    print(f"=" * 100)
    print(f"Model: {args.model}")
    print(f"Threads: {args.threads}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Mode: {args.mode}")
    print(f"Duration per phase: {args.duration}s")

    # Check initial power
    initial = get_power()
    if initial:
        print(f"\nInitial power: {initial.total_power:.0f}W")
        print(f"Device powers: {initial.device_powers}")
    else:
        print("\nWARNING: Cannot read power from TT monitor!")

    # Start power monitor
    power_thread = threading.Thread(target=power_monitor, args=(0.5,), daemon=True)
    power_thread.start()

    # Start stats reporter
    reporter_thread = threading.Thread(target=stats_reporter, args=(5,), daemon=True)
    reporter_thread.start()

    results = []

    if args.mode == "sweep":
        # Test different configurations
        configs = [
            ("Prefill-Heavy (long prompts)", worker_long_prefill, args.threads, 256),
            ("Decode-Heavy (short prompts, long gen)", worker_continuous_decode, args.threads, 1024),
            ("Balanced (long prompts, long gen)", worker_long_prefill, args.threads, 1024),
            ("Max Parallelism", worker_continuous_decode, args.threads, 512),
        ]

        for name, worker_fn, threads, max_tok in configs:
            avg_p, max_p, tps = run_phase(name, worker_fn, threads, args.model, max_tok, args.duration)
            results.append((name, avg_p, max_p, tps))
            time.sleep(5)  # Cool down between phases

        # Print summary
        print("\n" + "=" * 100)
        print("SUMMARY - Power vs Configuration")
        print("=" * 100)
        print(f"{'Configuration':<45} | {'Avg Power':>10} | {'Max Power':>10} | {'Avg TPS':>10}")
        print("-" * 100)
        for name, avg_p, max_p, tps in results:
            print(f"{name:<45} | {avg_p:>8.0f}W | {max_p:>8.0f}W | {tps:>9.1f}")

        best = max(results, key=lambda x: x[2])  # Best by max power
        print(f"\nBest for MAX POWER: {best[0]} ({best[2]:.0f}W)")

    elif args.mode == "max":
        # Optimized for maximum power: long prompts, very short outputs
        print("\n>>> MAX POWER MODE: Long prompts (~2500 tokens), short outputs (32 tokens)")
        print(">>> This maximizes prefill compute which draws peak power (~297W observed)")

        # Override max_tokens to 32 for max power
        try:
            while True:
                run_phase("MAX_POWER", worker_long_prefill, args.threads, args.model, 32, args.duration)
        except KeyboardInterrupt:
            print("\nStopping...")

    else:
        # Single mode
        if args.mode == "prefill":
            worker_fn = worker_long_prefill
        elif args.mode == "decode":
            worker_fn = worker_continuous_decode
        else:  # mixed - alternate
            worker_fn = worker_long_prefill

        try:
            while True:
                run_phase(args.mode, worker_fn, args.threads, args.model, args.max_tokens, args.duration)
        except KeyboardInterrupt:
            print("\nStopping...")

    running = False
    print("\nDone!")


if __name__ == "__main__":
    main()
