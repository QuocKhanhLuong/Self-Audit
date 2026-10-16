"""Read historical frozen TRAIN features without mutating their artifact schema."""

from pathlib import Path

import numpy as np
import torch
from self_audit_guidance.io import digest, load_bank_patient, verify

from .geometry import (
    GridGeometry,
    TeacherRegistration,
    normalize_cine,
    require_digest,
    slice_contexts,
)

# Reviewed full-native-FOV producer at main@4d58f3a. A different producer needs
# a separately reviewed mapping; dimensions alone cannot establish orientation.
PRODUCER_HASHES = {
    "teacher.py": "8a7b88548223c30ef570f90d977c71a14f2ca6ec219bf216e0f16efafdada273",
    "pipeline.py": "62b4f5fdd45089d1e3297004d61b643cb9e8e82f7d20a6598db6fd5b30ee181d",
    "io.py": "42c73e38a59463ec409fa1915d425f1e549f510be68d5c2c6fbd135eb5204b5d",
}


def load_training_context(bank, patient_id, indices, *, expected_bank_sha256):
    """Return image/feature inputs only; Q, labels and reliability are not Z mass.

    Conditional Z supervision requires an additional declared mass/support
    artifact. This reader does not invert Q, admit labels, or read references.
    """
    folder = Path(bank)
    before = digest(folder / "FROZEN.json")
    if before != require_digest(expected_bank_sha256):
        raise ValueError("bank manifest differs from the locked input")
    manifest = verify(folder, "guidance_bank")
    if any(
        manifest.get("source_sha256", {}).get(name) != h
        for name, h in PRODUCER_HASHES.items()
    ):
        raise ValueError("unreviewed bank producer coordinate contract")
    splits = manifest.get("locked_splits", {})
    selected = manifest.get("patients", [])
    if manifest.get("split") != "train" or patient_id not in selected:
        raise ValueError("only selected TRAIN bank patients are allowed")
    train, val, test = (set(splits.get(k, [])) for k in ("train", "val", "test"))
    if not set(selected) <= train or train & val or train & test or val & test:
        raise ValueError("bank patient split leakage")
    teacher = manifest.get("teacher", {})
    if teacher.get("frozen") is not True:
        raise ValueError("frozen teacher provenance required")
    data = load_bank_patient(folder, patient_id)
    meta = manifest["image_records"][patient_id]
    if tuple(reversed(meta["shape_xyzt"])) != data["image"].shape:
        raise ValueError("bank image/native geometry mismatch")
    normalized, normalization = normalize_cine(
        torch.from_numpy(data["image"]), meta["image_sha256"]
    )
    contexts = slice_contexts(normalized, indices)
    features = torch.from_numpy(
        np.stack([data["features"][t, z] for t, z in indices]).astype(np.float32)
    ).permute(0, 3, 1, 2)
    if features.shape[1] != 64:
        raise ValueError("v1 requires the declared 64-channel teacher interface")
    registration = TeacherRegistration(
        tuple(normalized.shape[-2:]),
        tuple(features.shape[-2:]),
        meta["image_sha256"],
        tuple(tuple(row) for row in meta["affine"]),
        before,
        tuple(sorted(PRODUCER_HASHES.items())),
    )
    geometry = GridGeometry(
        tuple(normalized.shape[-2:]),
        tuple(features.shape[-2:]),
        meta["image_sha256"],
        tuple(tuple(row) for row in meta["affine"]),
        meta.get("spatial_unit", "unknown"),
        registration,
    )
    # Verify that bytes read belong to the same immutable bank receipt.
    verify(folder, "guidance_bank")
    if digest(folder / "FROZEN.json") != before:
        raise ValueError("bank manifest changed while reading")
    return {
        "images": contexts,
        "teacher_features": features,
        "geometry": geometry,
        "normalization": normalization,
        "bank_manifest_sha256": before,
        "teacher": teacher,
        "patient_id": patient_id,
        "indices": [list(i) for i in indices],
        "semantic_admission": False,
        "named_mass_status": "NOT_INFERRED_FROM_HISTORICAL_Q",
    }
