"""Run bounded, synthetic-only P0 preflight."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardiac_benchmark.preflight import synthetic_preflight
from cardiac_benchmark.provenance import write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("CUTS-2D", "CUTS-2.5D"), default="CUTS-2D")
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()
    result = synthetic_preflight(profile=args.profile, target_hw=(args.size, args.size))
    write_json(args.output, result)
    print(args.output)
