#!/usr/bin/env python3
"""Export a verified candidate-A checkpoint without teacher dependencies."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from self_audit.candidate_a.io import export_student

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(json.dumps(export_student(a.checkpoint, a.checkpoint_sha256, a.out), indent=2))
