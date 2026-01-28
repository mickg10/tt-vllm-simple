#!/usr/bin/env python3
"""Endless benchmark for vLLM on Tenstorrent devices."""

import requests
import time
import sys
from datetime import datetime

API_URL = "http://localhost:8088/v1/chat/completions"

PROMPTS = [
    "Explain quantum computing in detail, including qubits, superposition, entanglement, and quantum gates.",
    "Write a comprehensive essay about the history of artificial intelligence from the 1950s to today.",
    "Describe the process of photosynthesis at the molecular level, including all chemical reactions.",
    "Explain the theory of general relativity and its implications for our understanding of space and time.",
    "Write a detailed analysis of Shakespeare's Hamlet, focusing on themes of revenge and mortality.",
    "Describe the architecture of modern neural networks, including transformers and attention mechanisms.",
    "Explain the economic principles behind cryptocurrency and blockchain technology.",
    "Write about the causes and consequences of World War I, including political and social factors.",
    "Describe the human immune system and how it responds to viral infections.",
    "Explain the principles of thermodynamics and their applications in engineering.",
]

def run_benchmark():
    iteration = 0
    total_tokens = 0
    total_time = 0.0

    print(f"Starting endless benchmark at {datetime.now().isoformat()}")
    print(f"API: {API_URL}")
    print("-" * 80)

    while True:
        prompt = PROMPTS[iteration % len(PROMPTS)]

        payload = {
            "model": "Qwen/Qwen3-32B",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.7,
        }

        try:
            start = time.time()
            response = requests.post(API_URL, json=payload, timeout=300)
            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens += completion_tokens
                total_time += elapsed

                tps = completion_tokens / elapsed if elapsed > 0 else 0
                avg_tps = total_tokens / total_time if total_time > 0 else 0

                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"iter={iteration:05d} | "
                      f"tokens={completion_tokens:3d} | "
                      f"time={elapsed:5.1f}s | "
                      f"tps={tps:5.1f} | "
                      f"avg_tps={avg_tps:5.1f} | "
                      f"total={total_tokens:,d}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"iter={iteration:05d} | ERROR: {response.status_code}")

        except requests.exceptions.Timeout:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"iter={iteration:05d} | TIMEOUT")
        except requests.exceptions.ConnectionError as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"iter={iteration:05d} | CONNECTION ERROR - waiting 10s...")
            time.sleep(10)
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"iter={iteration:05d} | ERROR: {e}")

        iteration += 1
        sys.stdout.flush()

if __name__ == "__main__":
    run_benchmark()
