#!/usr/bin/env python3
"""Small dependency-free HTTP load check with percentile reporting."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def request_once(url: str, index: int):
    payload = json.dumps(
        {"text": f"load-check message {index}", "content_reference": f"load://{index}"}
    ).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/moderate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status = response.status
    except (urllib.error.URLError, TimeoutError) as exc:
        return {
            "status": getattr(exc, "code", 0),
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
    return {"status": status, "latency_ms": (time.perf_counter() - started) * 1000}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        rows = list(executor.map(lambda index: request_once(args.url, index), range(args.requests)))
    latency = np.asarray([row["latency_ms"] for row in rows])
    report = {
        "requests": len(rows),
        "concurrency": args.concurrency,
        "success_rate": float(np.mean([row["status"] == 200 for row in rows])),
        "latency_ms": {
            name: float(np.percentile(latency, value))
            for name, value in (("p50", 50), ("p95", 95), ("p99", 99))
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(report["success_rate"] < 0.99)


if __name__ == "__main__":
    raise SystemExit(main())
