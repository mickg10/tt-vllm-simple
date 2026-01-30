#!/usr/bin/env python3
"""Endless multi-threaded benchmark for vLLM with growing context."""

import argparse
import requests
import time
import threading
from datetime import datetime
from collections import deque

API_URL = "http://localhost:8088/v1/chat/completions"

# Base prompt prefix (~1000 chars)
BASE_PREFIX = """You are a helpful AI assistant engaged in a continuous conversation. Please respond thoughtfully to the ongoing discussion. Here is some context about the topics we might explore:

The field of artificial intelligence has undergone remarkable transformation over the past decade. From early rule-based systems to modern deep learning architectures, the evolution has been extraordinary. Neural networks, particularly transformer-based models, have revolutionized natural language processing, computer vision, and many other domains.

Key developments include the attention mechanism, which allows models to focus on relevant parts of input sequences, and the scaling laws that suggest performance improvements with increased model size and training data. The emergence of large language models has demonstrated capabilities previously thought to require human-level intelligence.

Hardware advances have been equally important. Specialized accelerators designed for AI workloads have enabled training and inference at unprecedented scales. Companies like Tenstorrent are pushing the boundaries with novel architectures optimized for neural network computation.

Please continue our discussion on the following topic:"""

# Shared stats
stats_lock = threading.Lock()
total_tokens = 0
total_time = 0.0
total_requests = 0
recent_tps = deque(maxlen=100)

# Constants
MAX_PROMPT_CHARS = 20000
WORDS_TO_APPEND = 200


def get_last_n_words(text: str, n: int) -> str:
    """Get the last n words from text."""
    words = text.split()
    return ' '.join(words[-n:]) if len(words) >= n else text


def worker(thread_id: int, model: str, max_tokens: int):
    """Worker thread that continuously sends requests with growing context."""
    global total_tokens, total_time, total_requests

    # Each thread maintains its own growing prompt
    current_prompt = BASE_PREFIX + f"\n\n[Thread {thread_id}] Let's explore ideas about technology, science, philosophy, or any interesting topic. What are your thoughts?"
    iteration = 0

    while True:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": current_prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.8,
        }

        try:
            start = time.time()
            response = requests.post(API_URL, json=payload, timeout=600)
            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)

                # Get the response text
                response_text = ""
                choices = data.get("choices", [])
                if choices:
                    response_text = choices[0].get("message", {}).get("content", "")

                with stats_lock:
                    total_tokens += completion_tokens
                    total_time += elapsed
                    total_requests += 1
                    tps = completion_tokens / elapsed if elapsed > 0 else 0
                    recent_tps.append(tps)

                # Append last 200 words of response to prompt
                if response_text:
                    last_words = get_last_n_words(response_text, WORDS_TO_APPEND)
                    current_prompt = current_prompt + "\n\nAssistant: " + last_words + "\n\nUser: Please continue and expand on these ideas."

                # Reset if prompt too long
                if len(current_prompt) > MAX_PROMPT_CHARS:
                    current_prompt = BASE_PREFIX + f"\n\n[Thread {thread_id}, Reset after {iteration} iterations] Let's start a fresh discussion. What fascinating topic should we explore?"
                    iteration = 0

            else:
                print(f"[T{thread_id}] ERROR: {response.status_code}", flush=True)

        except requests.exceptions.Timeout:
            print(f"[T{thread_id}] TIMEOUT", flush=True)
        except requests.exceptions.ConnectionError:
            print(f"[T{thread_id}] CONNECTION ERROR - waiting 10s...", flush=True)
            time.sleep(10)
        except Exception as e:
            print(f"[T{thread_id}] ERROR: {e}", flush=True)

        iteration += 1


def stats_reporter(interval: int = 10):
    """Thread that reports aggregate stats periodically."""
    global total_tokens, total_time, total_requests

    last_tokens = 0
    last_time = time.time()

    while True:
        time.sleep(interval)

        with stats_lock:
            current_tokens = total_tokens
            current_requests = total_requests
            avg_tps = sum(recent_tps) / len(recent_tps) if recent_tps else 0

        now = time.time()
        delta_tokens = current_tokens - last_tokens
        delta_time = now - last_time

        instant_tps = delta_tokens / delta_time if delta_time > 0 else 0

        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"reqs={current_requests:5d} | "
              f"tokens={current_tokens:,d} | "
              f"tps={instant_tps:5.1f} | "
              f"avg_tps={avg_tps:5.1f}",
              flush=True)

        last_tokens = current_tokens
        last_time = now


def main():
    parser = argparse.ArgumentParser(description="Multi-threaded vLLM benchmark with growing context")
    parser.add_argument("--threads", "-t", type=int, default=50,
                        help="Number of concurrent request threads (default: 50)")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen3-32B",
                        help="Model name (default: Qwen/Qwen3-32B)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens per request (default: 512)")
    parser.add_argument("--report-interval", type=int, default=10,
                        help="Stats report interval in seconds (default: 10)")
    args = parser.parse_args()

    print(f"Starting multi-threaded benchmark at {datetime.now().isoformat()}")
    print(f"API: {API_URL}")
    print(f"Model: {args.model}")
    print(f"Threads: {args.threads}")
    print(f"Max tokens per response: {args.max_tokens}")
    print(f"Base prompt: ~{len(BASE_PREFIX)} chars, grows by ~{WORDS_TO_APPEND} words per iteration")
    print(f"Prompt resets at: {MAX_PROMPT_CHARS} chars")
    print("-" * 80)

    # Start worker threads
    workers = []
    for i in range(args.threads):
        t = threading.Thread(target=worker, args=(i, args.model, args.max_tokens), daemon=True)
        t.start()
        workers.append(t)
        time.sleep(0.1)  # Stagger thread starts

    # Start stats reporter in main thread
    stats_reporter(args.report_interval)


if __name__ == "__main__":
    main()
