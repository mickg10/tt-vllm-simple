#!/usr/bin/env python3
"""Run 32 coherency tests in parallel against GLM-4.7-Flash endpoint.

Tests include knowledge, math, logic, language, and Python coding tasks.
Code tasks are validated by extracting the function and running it.

Usage:
    python3 tests/run_coherency.py [--url URL] [--model MODEL]
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import aiohttp

TESTS_FILE = os.path.join(os.path.dirname(__file__), "coherency_tests.json")


async def run_test(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    test: dict,
    timeout: float = 120,
) -> dict:
    """Run a single coherency test."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": test["prompt"]}],
        "max_tokens": test.get("max_tokens", 64),
        "temperature": 0,
    }

    t0 = time.monotonic()
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {
                    "id": test["id"],
                    "description": test["description"],
                    "status": "ERROR",
                    "error": f"HTTP {resp.status}: {body[:200]}",
                    "elapsed_s": time.monotonic() - t0,
                }
            data = await resp.json()
    except Exception as e:
        return {
            "id": test["id"],
            "description": test["description"],
            "status": "ERROR",
            "error": str(e)[:200],
            "elapsed_s": time.monotonic() - t0,
        }

    elapsed = time.monotonic() - t0
    choices = data.get("choices", [])
    if not choices:
        return {
            "id": test["id"],
            "description": test["description"],
            "status": "ERROR",
            "error": "No choices in response",
            "elapsed_s": elapsed,
        }

    response = choices[0].get("message", {}).get("content", "")

    # Validate response content
    validate_expr = test.get("validate", "")
    try:
        content_pass = bool(eval(validate_expr)) if validate_expr else len(response) > 0
    except Exception:
        content_pass = False

    # For code tasks, try to extract and run the function
    code_pass = None
    if test.get("validate_runs"):
        code_pass = validate_code_runs(response, test)

    status = "PASS" if content_pass and (code_pass is None or code_pass) else "FAIL"

    return {
        "id": test["id"],
        "description": test["description"],
        "category": test.get("category", ""),
        "status": status,
        "content_pass": content_pass,
        "code_pass": code_pass,
        "response_preview": response[:120].replace("\n", "\\n"),
        "full_response": response,
        "elapsed_s": round(elapsed, 2),
    }


def validate_code_runs(response: str, test: dict) -> bool:
    """Extract Python code from response and try to run it."""
    # Try to extract code from markdown code blocks
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if code_blocks:
        code = code_blocks[0]
    else:
        # Try to find the function definition directly
        lines = response.split("\n")
        code_lines = []
        in_func = False
        for line in lines:
            if line.strip().startswith("def "):
                in_func = True
            if in_func:
                code_lines.append(line)
                if line.strip() and not line.startswith(" ") and not line.startswith("\t") and not line.startswith("def"):
                    break
        code = "\n".join(code_lines)

    if not code.strip():
        return False

    # Add a simple test call based on the function
    test_code = code + "\n"
    desc = test.get("description", "").lower()
    if "fibonacci" in desc or "fib" in desc:
        test_code += "\nassert fib(10) == 55, f'fib(10) = {fib(10)}'\nprint('OK')\n"
    elif "prime" in desc:
        test_code += "\nassert is_prime(7) == True\nassert is_prime(4) == False\nprint('OK')\n"
    elif "reverse" in desc:
        test_code += "\nassert reverse_string('hello') == 'olleh'\nprint('OK')\n"
    elif "factorial" in desc:
        test_code += "\nassert factorial(5) == 120\nprint('OK')\n"
    elif "sum" in desc and "list" in desc:
        test_code += "\nassert sum_list([1,2,3,4,5]) == 15\nprint('OK')\n"
    elif "max" in desc:
        test_code += "\nassert max_in_list([3,1,4,1,5,9]) == 9\nprint('OK')\n"
    elif "vowel" in desc:
        test_code += "\nassert count_vowels('hello') == 2\nprint('OK')\n"
    elif "palindrome" in desc:
        test_code += "\nassert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nprint('OK')\n"
    elif "flatten" in desc:
        test_code += "\nassert flatten([1,[2,[3,4],5],6]) == [1,2,3,4,5,6]\nprint('OK')\n"
    elif "binary_search" in desc or "binary search" in desc:
        test_code += "\nassert binary_search([1,2,3,4,5], 3) == 2\nassert binary_search([1,2,3,4,5], 6) == -1\nprint('OK')\n"
    elif "bubble" in desc or "sort" in desc:
        test_code += "\nassert bubble_sort([3,1,4,1,5]) == [1,1,3,4,5]\nprint('OK')\n"
    else:
        test_code += "\nprint('OK')\n"

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            tmp_path = f.name
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        os.unlink(tmp_path)
        return result.returncode == 0 and "OK" in result.stdout
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return False


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8088")
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    # Load tests
    with open(TESTS_FILE) as f:
        tests_data = json.load(f)
    tests = tests_data["tests"]

    print(f"Running {len(tests)} coherency tests against {args.url}")
    print(f"Model: {args.model}")
    print(f"{'=' * 90}")

    # Run all tests in parallel
    async with aiohttp.ClientSession() as session:
        tasks = [
            run_test(session, args.url, args.model, test, args.timeout)
            for test in tests
        ]
        results = await asyncio.gather(*tasks)

    # Sort by ID
    results.sort(key=lambda r: r["id"])

    # Map test definitions by ID for full response lookup
    test_by_id = {t["id"]: t for t in tests}

    # PHASE 1: Show all generated code for manual inspection FIRST
    code_results = [r for r in results if r.get("category") == "code"]
    if code_results:
        print(f"\n{'=' * 90}")
        print("PHASE 1: CODE OUTPUT REVIEW (inspect before judging)")
        print(f"{'=' * 90}")
        for r in code_results:
            status_marker = {"PASS": "OK", "FAIL": "FAIL", "ERROR": "ERR"}.get(r["status"], "???")
            print(f"\n--- #{r['id']} {r['description']} [{status_marker}] ---")
            print(f"Prompt: {test_by_id[r['id']]['prompt'][:80]}")
            # Show FULL response (the generated code), not just a preview
            full_response = r.get("full_response", r.get("response_preview", "N/A"))
            print(f"Generated code:\n{full_response}")
            if r.get("code_pass") is not None:
                print(f"Runs correctly: {'YES' if r['code_pass'] else 'NO'}")
            print()

    # PHASE 2: Summary table for all tests
    print(f"\n{'=' * 90}")
    print("PHASE 2: ALL RESULTS SUMMARY")
    print(f"{'=' * 90}")

    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    error_count = sum(1 for r in results if r["status"] == "ERROR")

    for r in results:
        status_marker = {"PASS": "OK", "FAIL": "FAIL", "ERROR": "ERR"}.get(r["status"], "???")
        code_info = ""
        if r.get("code_pass") is not None:
            code_info = f" runs={'YES' if r['code_pass'] else 'NO'}"
        preview = r.get("response_preview", "")[:60]
        print(
            f"  [{status_marker:>4}] #{r['id']:>2} {r['description']:<25}{code_info}"
            f"  ({r['elapsed_s']:.1f}s) | {preview}"
        )

    print(f"\n{'=' * 90}")
    print(f"RESULTS: {pass_count} PASS, {fail_count} FAIL, {error_count} ERROR out of {len(results)}")
    print(f"{'=' * 90}")

    # Save results
    output_path = f"/home/ttuser/src_docker/plan/glm47_flash/artifacts/coherency_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "model": args.model,
            "url": args.url,
            "pass": pass_count,
            "fail": fail_count,
            "error": error_count,
            "results": results,
        }, f, indent=2)
    print(f"Results saved to: {output_path}")

    return 0 if fail_count == 0 and error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
