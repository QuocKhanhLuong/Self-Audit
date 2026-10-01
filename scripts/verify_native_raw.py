"""Verify seals, finalize external seal receipts, or compare repeats without GT or producer imports."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_benchmark.native_artifacts import (
    compare_repeat_runs, verify_raw_run, verify_seal_receipt, write_seal_receipt,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--repeat", type=Path)
    group.add_argument("--write-receipt", type=Path,
                       help="write-once receipt outside the raw root; commit/archive it before evaluation")
    group.add_argument("--receipt", type=Path, help="verify the raw run against a finalized receipt")
    args = parser.parse_args()
    if args.repeat:
        result = compare_repeat_runs(args.raw_root, args.repeat)
    elif args.write_receipt:
        result = write_seal_receipt(args.raw_root, args.write_receipt)
    elif args.receipt:
        result = verify_seal_receipt(args.raw_root, args.receipt)
    else:
        result = verify_raw_run(args.raw_root)
    print(json.dumps(result, indent=2))
