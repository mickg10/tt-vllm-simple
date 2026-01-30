#!/usr/bin/env python3
"""Ultra-aggressive power burn - maximizes prefill compute."""
import requests
import threading
import time
import sys

VLLM_URL = "http://localhost:8088/v1/chat/completions"
MONITOR_URL = "http://localhost:9090/api/devices"

# MASSIVE prompt for maximum prefill compute (~5000 tokens)
MEGA_PROMPT = """You are tasked with an extremely detailed analysis. Consider every aspect thoroughly.

""" + ("The advancement of artificial intelligence represents one of humanity's greatest technological achievements. " * 200) + """

Now provide a brief summary."""

THREADS = 32
MAX_TOKENS = 16  # Minimal decode - pure prefill power

stats = {"reqs": 0, "power_max": 0}

def worker(tid):
    while True:
        try:
            resp = requests.post(VLLM_URL, json={
                "model": "Qwen/Qwen3-32B",
                "messages": [{"role": "user", "content": MEGA_PROMPT}],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.7
            }, timeout=300)
            if resp.status_code == 200:
                stats["reqs"] += 1
        except:
            time.sleep(0.5)

def monitor():
    while True:
        try:
            r = requests.get(MONITOR_URL, timeout=2)
            if r.status_code == 200:
                d = r.json()
                power = d.get("totals", {}).get("power", 0)
                stats["power_max"] = max(stats["power_max"], power)
                temps = [dev.get("temperature", 0) for dev in d.get("devices", [])]
                print(f"\r[{time.strftime('%H:%M:%S')}] Power: {power:>5.0f}W | Max: {stats['power_max']:>5.0f}W | Reqs: {stats['reqs']:>6} | Temps: {'/'.join(f'{t:.0f}' for t in temps)}  ", end="", flush=True)
        except:
            pass
        time.sleep(1)

print(f"ULTRA BURN: {THREADS} threads, {len(MEGA_PROMPT)} char prompt, {MAX_TOKENS} max tokens")
print("Target: Maximize prefill compute = maximum power\n")

threading.Thread(target=monitor, daemon=True).start()
for i in range(THREADS):
    threading.Thread(target=worker, args=(i,), daemon=True).start()
    time.sleep(0.02)

try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print(f"\n\nMax power observed: {stats['power_max']:.0f}W")
