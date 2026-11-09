#!/usr/bin/env python3
"""Evaluate a frozen Candidate A student; never train or select a checkpoint."""
import argparse
import json
from pathlib import Path
from self_audit.model import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    manifest = Path(args.checkpoint) / "MANIFEST.json"
    if digest(manifest) != args.checkpoint_sha256:
        parser.error("checkpoint manifest hash mismatch")
    schema = json.loads(manifest.read_text()).get("schema")
    if schema in {"self_audit.candidate_a3d.checkpoint.v1", "self_audit.candidate_a3d.student.v1"}:
        from self_audit.candidate_a3d.evaluation import evaluate
    else:
        from self_audit.candidate_a.evaluation import evaluate
    result = evaluate(args.checkpoint, args.checkpoint_sha256, args.data, args.data_sha256,
                      args.out, args.device, args.threshold)
    print(json.dumps(result["macro_patient"], indent=2))


if __name__ == "__main__":
    main()
