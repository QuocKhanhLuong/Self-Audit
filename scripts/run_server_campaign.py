#!/usr/bin/env python3
"""Validate, start, inspect, or gracefully stop a pinned server campaign."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.server_campaign import cli


if __name__ == "__main__":
    raise SystemExit(cli())
