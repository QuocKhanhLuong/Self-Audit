#!/usr/bin/env python3
"""Export a verified student-only artifact with embedded memory buffers."""
import argparse
import copy
from pathlib import Path
import json
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.data import write_json
from self_audit.engine import load_checkpoint, source_digest
from self_audit.model import digest


def export(checkpoint, checkpoint_sha256, out):
    checkpoint = Path(checkpoint).resolve()
    out = Path(out).resolve()
    if out.exists() or out.is_relative_to(checkpoint) or checkpoint.is_relative_to(out):
        raise ValueError("export output must be a new directory outside the source checkpoint")
    model, checkpoint_meta = load_checkpoint(checkpoint, checkpoint_sha256, device="cpu")
    config = copy.deepcopy(checkpoint_meta["config"])
    # These training assets are represented in the embedded model state or are
    # unused at inference. Clear them so downstream loading cannot access them.
    for key in ("bank_path", "bank_sha256", "cache_path", "cache_sha256"):
        config["teacher_memory"][key] = None
    out.mkdir(parents=True)
    weights = out / "weights.pt"
    torch.save({"student": model.state_dict()}, weights)
    metadata = {"schema": "self_audit.student_export.v1", "config": config,
                "synthetic": bool(checkpoint_meta["synthetic"]), "prediction_model": "student",
                "source_checkpoint_manifest_sha256": checkpoint_sha256,
                "source_checkpoint_payload_sha256": checkpoint_meta["payload_sha256"],
                "source_sha256": source_digest(), "payload_sha256": digest(weights),
                "teacher_included": False, "external_pseudo_cache_required": False,
                "memory_bank_buffers_embedded": model.teacher_memory is not None}
    write_json(out / "MANIFEST.json", metadata)
    return digest(out / "MANIFEST.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_sha = export(args.checkpoint, args.checkpoint_sha256, args.out)
    print(json.dumps({"status": "EXPORTED", "manifest_sha256": manifest_sha,
                      "out": str(Path(args.out).resolve())}, indent=2))


if __name__ == "__main__":
    main()
