#!/usr/bin/env python3
"""Evaluate a frozen Candidate A student; never train or select a checkpoint."""
import argparse
import json
from self_audit.candidate_a.evaluation import evaluate


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
    result = evaluate(args.checkpoint, args.checkpoint_sha256, args.data, args.data_sha256,
                      args.out, args.device, args.threshold)
    print(json.dumps(result["macro_patient"], indent=2))


if __name__ == "__main__":
    main()
