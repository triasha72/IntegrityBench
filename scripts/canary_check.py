#!/usr/bin/env python3
"""Compare stable and canary endpoints on a supplied real shadow sample."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np


def predict(url: str, row: dict[str, str]):
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/moderate",
        data=json.dumps(row).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.load(response)
    return payload["decision"], (time.perf_counter() - started) * 1000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-url", required=True)
    parser.add_argument("--canary-url", required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-disagreement", type=float, default=0.05)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.sample.read_text().splitlines() if line]
    transitions = {}
    stable_latency, canary_latency = [], []
    for row in rows:
        stable, stable_ms = predict(args.stable_url, row)
        canary, canary_ms = predict(args.canary_url, row)
        key = f"{stable}->{canary}"
        transitions[key] = transitions.get(key, 0) + 1
        stable_latency.append(stable_ms)
        canary_latency.append(canary_ms)
    disagreements = sum(
        count for key, count in transitions.items() if key.split("->")[0] != key.split("->")[1]
    )
    report = {
        "rows": len(rows),
        "disagreement_rate": disagreements / len(rows),
        "stable_p95_latency_ms": float(np.percentile(stable_latency, 95)),
        "canary_p95_latency_ms": float(np.percentile(canary_latency, 95)),
        "decision_transitions": transitions,
        "contains_source_text": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(report["disagreement_rate"] > args.maximum_disagreement)


if __name__ == "__main__":
    raise SystemExit(main())
