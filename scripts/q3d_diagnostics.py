#!/usr/bin/env python3
"""Run read-only Q-3D VAL codebook and physiology diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from self_audit.candidate_q3d.diagnostics import diagnose_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="exact Q-3D v1 JSON recipe used for training")
    parser.add_argument("--checkpoint", required=True, help="Q-3D research resume checkpoint; deployment exports are rejected")
    parser.add_argument("--checkpoint-sha256", help="optional expected checkpoint SHA-256")
    parser.add_argument("--device", default="cpu", help="inference device, defaults to CPU")
    parser.add_argument("--out", required=True, help="new JSON report path")
    args = parser.parse_args(argv)

    output = Path(args.out)
    if output.exists():
        parser.error(f"refusing to overwrite existing diagnostic report: {output}")
    result = diagnose_checkpoint(
        args.config, args.checkpoint, device=args.device,
        expected_checkpoint_sha256=args.checkpoint_sha256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
