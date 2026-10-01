#!/usr/bin/env python
"""Run an official entrypoint only inside a declared pinned environment.

Validates the interpreter first (fail fast), writes the environment identity
and critical package versions to --provenance-out (never overwritten), then
executes the target script in-process with the remaining arguments, exactly as
``python <script> <args>`` would. Used for runners whose source is bound by a
benchmark freeze (CUTS/DFC), so their provenance gains the environment identity
without changing freeze-bound files.

    python scripts/run_in_official_environment.py --provenance-out RUN/environment.json \
        -- scripts/run_cuts_scientific.py --split dev ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from environment_contract import CANONICAL_ENVIRONMENT, require_official_environment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default=CANONICAL_ENVIRONMENT)
    parser.add_argument("--variant", choices=["cpu", "cu121"], default=None)
    parser.add_argument("--provenance-out", type=Path, required=True)
    parser.add_argument("script", type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    report = require_official_environment(args.env, variant=args.variant)
    report["entrypoint"] = {"script": str(args.script),
                            "argv": args.script_args[1:] if args.script_args[:1] == ["--"] else args.script_args}
    args.provenance_out.parent.mkdir(parents=True, exist_ok=True)
    with args.provenance_out.open("x", encoding="utf-8") as stream:  # never overwrite provenance
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    script = args.script.resolve()
    script_args = args.script_args[1:] if args.script_args[:1] == ["--"] else args.script_args
    sys.argv = [str(script), *script_args]
    sys.path.insert(0, str(script.parent))  # as `python <script>` would
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
