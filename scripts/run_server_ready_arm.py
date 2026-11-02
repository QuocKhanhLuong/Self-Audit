#!/usr/bin/env python3
"""Bind, profile, lock and run one eligible compact server campaign stage."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.server_ready_arm import main

if __name__ == "__main__":
    raise SystemExit(main())
