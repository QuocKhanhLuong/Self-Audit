#!/usr/bin/env python
"""Validate the running interpreter against a declared pinned environment.

Exit 0 when it matches, 1 otherwise. Imports no numerical package.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from environment_contract import CANONICAL_ENVIRONMENT, check_environment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=CANONICAL_ENVIRONMENT)
    parser.add_argument("--variant", choices=["cpu", "cu121"], default=None,
                        help="required PyTorch build variant (default: detect)")
    parser.add_argument("--json", type=Path, help="also write the report to this file")
    args = parser.parse_args()
    report = check_environment(args.env, variant=args.variant)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    if not report["matches"]:
        print("ENVIRONMENT MISMATCH: " + "; ".join(report["problems"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
