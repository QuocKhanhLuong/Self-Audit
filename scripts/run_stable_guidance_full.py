#!/usr/bin/env python3
"""Entrypoint; no remote or synthetic fallback."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from self_audit_guidance.full import main

if __name__ == "__main__":
    raise SystemExit(main())
