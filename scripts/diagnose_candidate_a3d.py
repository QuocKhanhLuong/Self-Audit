#!/usr/bin/env python3
"""Measure the privileged TRAIN-GT code-capacity diagnostic for A-3D.

This command is deliberately separate from deployment and training. It replaces
the student values in a serialized code with quantized TRAIN-GT measurements to
answer the representation question "can the decoder hold the shape at this
budget?". It never opens development or test manifests, never changes model
parameters, and never reports a patient efficacy result or a mathematical upper
bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from self_audit.candidate_a3d.data import CandidateA3DData
from self_audit.candidate_a3d.io import load_student
from self_audit.metrics import class_metrics, confusion_matrix
from self_audit.model import digest


def _geometry(meta):
    if isinstance(meta, dict) and "geometry" in meta:
        meta = meta["geometry"]
    if not isinstance(meta, dict):
        raise ValueError("A-3D frame loader must return a public geometry mapping")
    required = {"shape", "spacing_zyx", "affine_zyx", "context"}
    if not required.issubset(meta):
        raise ValueError("diagnostic geometry must include shape, spacing_zyx, affine_zyx, context")
    if meta["context"] != "full_volume":
        raise ValueError("TRAIN-GT diagnostic requires full_volume geometry")
    return meta


def _load_frame(data, reference):
    image, label, geometry, _ = data.load_labeled_frame(reference)
    geometry = _geometry(geometry)
    if image.ndim != 5 or tuple(image.shape[1:2]) != (1,):
        raise ValueError("A-3D frame image must have shape [B,1,D,H,W]")
    if label.ndim != 4 or label.shape[0] != image.shape[0]:
        raise ValueError("A-3D frame label must have shape [B,D,H,W]")
    return image, label, geometry


def _ids_for_budget(codec, bank, geometry, budget_bits, *, baseline=()):
    # A fixed prefix is intentionally used across budgets. This diagnostic
    # measures capacity, not policy quality, so ID selection cannot change with
    # the budget. Prefix order is the bank's deterministic canonical order.
    selected = list(baseline)
    if codec.cost(selected, shape=geometry["shape"], spacing=geometry["spacing_zyx"]) > budget_bits:
        raise ValueError("fixed R0 does not fit the diagnostic budget")
    for query in bank:
        if query.index in selected:
            continue
        candidate = tuple(selected + [query.index])
        if codec.cost(candidate, shape=geometry["shape"], spacing=geometry["spacing_zyx"]) > budget_bits:
            break
        selected.append(query.index)
    return tuple(selected)


def diagnose(checkpoint, checkpoint_sha256, train_manifest, train_sha256, out,
             *, budgets, device="cpu"):
    manifest = json.loads(Path(train_manifest).read_text())
    if manifest.get("role") != "training":
        raise ValueError("diagnostic accepts only a training manifest")
    # A development/held-out manifest cannot be made a TRAIN manifest merely
    # by relabelling its role: the TRAIN schema has explicit labeled and
    # unlabeled cohorts, while evaluation manifests have a patients list and
    # an evaluation scope.  This structural guard complements the checkpoint
    # ID and manifest-hash checks below.
    if not isinstance(manifest.get("labeled"), list) or not isinstance(manifest.get("unlabeled"), list):
        raise ValueError("diagnostic requires the canonical TRAIN labeled/unlabeled cohorts")
    if "patients" in manifest or "scope" in manifest:
        raise ValueError("development/held-out manifest cannot be relabelled as TRAIN")
    if digest(train_manifest) != train_sha256:
        raise ValueError("TRAIN manifest hash mismatch")
    model, bank, codec, metadata = load_student(checkpoint, checkpoint_sha256, device=device)
    checkpoint_synthetic = metadata.get("synthetic")
    manifest_synthetic = manifest.get("synthetic")
    if type(checkpoint_synthetic) is not bool or type(manifest_synthetic) is not bool:
        raise ValueError("student/TRAIN synthetic flags must be explicit booleans")
    if checkpoint_synthetic != manifest_synthetic:
        raise ValueError("student/TRAIN synthetic identity mismatch")
    train_ids = {record["id"] for record in manifest.get("labeled", []) + manifest.get("unlabeled", [])}
    checkpoint_ids = set(metadata.get("labeled_ids", [])) | set(metadata.get("unlabeled_ids", []))
    if checkpoint_ids != train_ids or metadata.get("data_sha256") != train_sha256:
        raise ValueError("student checkpoint is not tied to this exact TRAIN cohort/manifest")
    if metadata.get("label_budget") != len(manifest.get("labeled", [])):
        raise ValueError("student checkpoint label budget differs from TRAIN manifest")
    data = CandidateA3DData(train_manifest, train_sha256, synthetic=manifest_synthetic)
    references = tuple(data.labeled_frame_keys())
    if not references:
        raise ValueError("TRAIN manifest has no annotated frames")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    with torch.inference_mode():
        for budget in budgets:
            if isinstance(budget, bool) or int(budget) != budget or int(budget) <= 0:
                raise ValueError("budgets must be positive integers")
            budget = int(budget)
            matrix = np.zeros((4, 5), dtype=np.int64)
            counts = 0
            baseline = tuple(int(v) for v in metadata["config"]["rollout"]["baseline_ids"])
            header_bits = codec.cost((), shape=(1, 1, 1), spacing=(1.0, 1.0, 1.0))
            if budget < header_bits:
                raise ValueError(f"budget {budget} is below the fixed packet header cost {header_bits}")
            selected = None
            for reference in references:
                image, label, geometry = _load_frame(data, reference)
                image, label = image.to(device), label.to(device)
                if tuple(geometry["shape"]) != tuple(label.shape[-3:]):
                    raise ValueError("frame label shape disagrees with public geometry")
                ids = _ids_for_budget(codec, bank, geometry, budget, baseline=baseline)
                if selected is None:
                    selected = ids
                elif ids != selected:
                    raise ValueError("fixed diagnostic ID prefix changed across TRAIN geometry")
                target = bank.targets(label, geometry["spacing_zyx"])
                quantized = codec.quantize_values(target[:, list(ids)], ids, ste=False)
                payload = codec.serialize(ids, quantized.levels[0], geometry["shape"],
                                          geometry["spacing_zyx"], affine_zyx=geometry["affine_zyx"],
                                          budget_bits=budget)
                packet = codec.deserialize(payload, budget_bits=budget)
                dequantized = codec.dequantize(packet, device=label.device, dtype=torch.float32)[None]
                packet_geometry = {**geometry, "shape": packet.shape,
                                   "spacing_zyx": packet.spacing,
                                   "affine_zyx": packet.affine_zyx}
                decoded = model.decode(packet.ids, dequantized, packet_geometry,
                                       chunk_size=metadata["config"]["student"]["chunk_size"])
                if decoded.shape != (label.shape[0], 4, *label.shape[-3:]):
                    raise ValueError("decoder output shape does not match full TRAIN volume")
                matrix += confusion_matrix(label.cpu().numpy(), decoded.argmax(1).cpu().numpy())
                counts += 1
            rows.append({"budget_bits": budget, "actual_packet_bits": codec.packet_bits(payload),
                         "query_count": len(selected or ()),
                         "frames": counts, "per_class": class_metrics(matrix),
                         "privileged_values": "quantized_TRAIN_GT",
                         "deployment": "NOT_DEPLOYMENT", "efficacy": "NOT_MEASURED"})
    result = {"schema": "self_audit.candidate_a3d.gt_code_diagnostic.v1",
              "status": "COMPLETED_SYNTHETIC" if manifest_synthetic else "COMPLETED_TRAIN_DIAGNOSTIC",
              "train_manifest_sha256": train_sha256, "checkpoint_manifest_sha256": checkpoint_sha256,
              "budgets": rows, "selection": "canonical bank prefix fixed across budgets",
              "warning": "Privileged TRAIN-GT representation diagnostic; not deployment, patient efficacy, or a mathematical upper bound."}
    (out / "DIAGNOSTIC.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(diagnose(args.checkpoint, args.checkpoint_sha256, args.train,
                              args.train_sha256, args.out, budgets=args.budgets,
                              device=args.device), indent=2))


if __name__ == "__main__":
    main()
