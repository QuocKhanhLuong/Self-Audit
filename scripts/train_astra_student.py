#!/usr/bin/env python3
"""Compatibility alias for the sole canonical train.py entrypoint."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def main(argv=None):
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("_self_audit_canonical_train", ROOT / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(argv)

if __name__ == "__main__":
    raise SystemExit(main())
