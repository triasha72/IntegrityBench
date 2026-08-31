#!/usr/bin/env python3
"""Register, promote, or roll back IntegrityBench model versions."""

from __future__ import annotations

import argparse
from pathlib import Path

from integritybench.registry import ModelRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--model-id", required=True)
    register.add_argument("--model-uri", required=True)
    register.add_argument("--sha256", required=True)
    register.add_argument("--evidence", required=True)
    promote = subparsers.add_parser("promote")
    promote.add_argument("--model-id", required=True)
    promote.add_argument("--model-path", type=Path, required=True)
    promote.add_argument("--assessment", type=Path, required=True)
    subparsers.add_parser("rollback")
    args = parser.parse_args()
    registry = ModelRegistry(args.registry)
    if args.command == "register":
        registry.register(args.model_id, args.model_uri, args.sha256, args.evidence)
    elif args.command == "promote":
        registry.promote(args.model_id, args.model_path, args.assessment)
    else:
        print(f"production={registry.rollback()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
