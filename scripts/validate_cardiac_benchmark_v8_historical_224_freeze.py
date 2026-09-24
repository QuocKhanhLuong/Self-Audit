#!/usr/bin/env python
"""Validate the immutable v8 historical-224 CUTS/DFC baseline freeze."""
from __future__ import annotations

import argparse
from pathlib import Path

import validate_cardiac_benchmark_v7_historical_224_freeze as _v7


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v8_historical_224"
SCHEMA = "shared_benchmark.cardiac_benchmark_freeze.v8_historical_224"
PREFIX = "cardiac-benchmark-v8-historical-224"
FreezeValidationError = _v7.FreezeValidationError


def validate_freeze(freeze_dir: str | Path = FREEZE, repo_root: str | Path = REPO_ROOT) -> dict[str, object]:
    prior_schema, prior_prefix = _v7.SCHEMA, _v7.PREFIX
    try:
        _v7.SCHEMA, _v7.PREFIX = SCHEMA, PREFIX
        return _v7.validate_freeze(freeze_dir, repo_root)
    finally:
        _v7.SCHEMA, _v7.PREFIX = prior_schema, prior_prefix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", type=Path, default=FREEZE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    import json

    print(json.dumps(validate_freeze(args.freeze_dir, args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
