"""Verify seals or compare repeats without GT or producer imports."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_benchmark.native_artifacts import verify_raw_run, compare_repeat_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--repeat", type=Path)
    args = parser.parse_args()
    result = compare_repeat_runs(args.raw_root, args.repeat) if args.repeat else verify_raw_run(args.raw_root)
    print(json.dumps(result, indent=2))
