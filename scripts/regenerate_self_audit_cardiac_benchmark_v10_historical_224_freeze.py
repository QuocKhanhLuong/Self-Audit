#!/usr/bin/env python
"""Create the active v10 historical-224 CUTS/DFC baseline freeze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import regenerate_self_audit_cardiac_benchmark_v7_historical_224_freeze as _v7


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v10_historical_224"
SCHEMA = "shared_benchmark.cardiac_benchmark_freeze.v10_historical_224"
PREFIX = "cardiac-benchmark-v10-historical-224"


def regenerate(output: Path = OUTPUT) -> dict[str, object]:
    prior = (_v7.SCHEMA, _v7.PREFIX, _v7.GENERATOR_PATH, _v7.VALIDATOR_PATH)
    try:
        _v7.SCHEMA, _v7.PREFIX = SCHEMA, PREFIX
        _v7.GENERATOR_PATH = "scripts/regenerate_self_audit_cardiac_benchmark_v10_historical_224_freeze.py"
        _v7.VALIDATOR_PATH = "scripts/validate_cardiac_benchmark_v10_historical_224_freeze.py"
        return _v7.regenerate(output)
    finally:
        _v7.SCHEMA, _v7.PREFIX, _v7.GENERATOR_PATH, _v7.VALIDATOR_PATH = prior


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(regenerate(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
