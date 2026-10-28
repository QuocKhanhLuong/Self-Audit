#!/usr/bin/env python3
"""Assess pinned patient quality and matched inference resource reports."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from self_audit.compact_artifacts import load_compact_export
from self_audit.compact_acceptance import read_pinned_json, assess_compact_deployment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("export", "quality", "benchmark"):
        p.add_argument(f"--{name}", required=True)
        p.add_argument(f"--{name}-sha256", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        p.error("output exists; preserve previous acceptance receipt")
    model, export = load_compact_export(a.export, a.export_sha256, device="cpu")
    del model
    quality = read_pinned_json(a.quality, a.quality_sha256)
    benchmark = read_pinned_json(a.benchmark, a.benchmark_sha256)
    result = assess_compact_deployment(export, quality, benchmark)
    result["evidence_sha256"] = {"export": a.export_sha256, "quality": a.quality_sha256,
                                  "benchmark": a.benchmark_sha256}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
