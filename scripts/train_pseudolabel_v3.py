#!/usr/bin/env python3
"""Retired training command: intentionally no argument translation or fallback."""
import sys
print("LEGACY_ENTRYPOINT_DISABLED: this trainer was removed. Use python train.py --describe, then docs/RUNBOOK.md. Historical reproduction belongs in archive/pre-cleanup-20261017, never this checkout.", file=sys.stderr)
raise SystemExit(2)
