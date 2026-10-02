"""Synthetic ACDC-shaped fixture for the native ACDC adaptation tests (no real ACDC data).

Each volume is a small HWZ NIfTI with a ring (MYO) around a bright disc (LV) and a crescent
(RV); a ``*_gt.nii`` reference sits next to every image exactly as in ACDC, so tests can
prove producers and Track A never touch it. Records reuse the frozen v12 record schema
and shared grid; only identity, source and geometry fields are synthetic.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmark_freezes/cardiac_benchmark_v12_historical_224/data/acdc_shared_manifest.json"


def _volume(shape, shift, rng):
    height, width, depth = shape
    yy, xx = np.mgrid[:height, :width]
    image = np.zeros(shape, np.int16)
    labels = np.zeros(shape, np.uint8)
    for z in range(depth):
        cy, cx = height // 2 + shift, width // 2 + z
        radius = np.hypot(yy - cy, xx - cx)
        lv = radius < 6
        myo = (radius >= 6) & (radius < 9)
        rv = (np.hypot(yy - cy, xx - (cx - 11)) < 7) & ~lv & ~myo & (radius >= 9)
        plane = np.full((height, width), 40.0)
        plane[myo], plane[lv], plane[rv] = 110.0, 230.0, 190.0
        image[:, :, z] = np.clip(plane + rng.normal(0, 6, plane.shape), 0, 400).astype(np.int16)
        labels[:, :, z][lv], labels[:, :, z][myo], labels[:, :, z][rv] = 3, 2, 1
    return image, labels


def build(tmp_path, *, patients=("patient001",), shape=(40, 48, 2), split="dev"):
    """Write the fixture; return (contract, records, image_root, gt_root)."""
    import nibabel as nib
    from shared_benchmark.acdc_native import AcdcContract
    from shared_benchmark.semantic_contract import (FROZEN_ADAPTER_SPEC_SHA256,
                                                    FROZEN_SELF_AUDIT_HISTORICAL_224_SHARED_GRID_SHA256)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    template = manifest["records"][0]
    image_root = Path(tmp_path) / "acdc"
    rng = np.random.default_rng(0)
    records = []
    for index, patient in enumerate(patients):
        folder = image_root / "training" / patient
        folder.mkdir(parents=True)
        image, labels = _volume(shape, index * 2, rng)
        locator = f"training/{patient}/{patient}_frame01.nii"
        nib.save(nib.Nifti1Image(image, np.eye(4)), str(image_root / locator))
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(folder / f"{patient}_frame01_gt.nii"))
        digest = hashlib.sha256((image_root / locator).read_bytes()).hexdigest()
        for z in range(shape[2]):
            record = copy.deepcopy(template)
            volume_id = f"acdc:self_audit:{patient}:frame-0001"
            record.update({
                "sample_id": f"acdc:patient-{patient}:study-acdc:{patient}:volume-{volume_id}:frame-0001:z-{z:04d}",
                "patient_id": patient, "study_id": f"acdc:{patient}", "volume_id": volume_id, "split": split,
                "frame_index": 1, "slice_index": z, "depth": shape[2], "native_hw": list(shape[:2]),
                "native_shape": list(shape), "stored_shape": list(shape),
                "context_indices": [max(z - 1, 0), z, min(z + 1, shape[2] - 1)]})
            record["source"] = {**template["source"], "locator": locator, "sha256": digest,
                                "frame_fingerprint": hashlib.sha256(f"{digest}:1".encode()).hexdigest()}
            record["spatial_transform"] = {**template["spatial_transform"], "native_shape": list(shape), "slice_index": z,
                                           "forward_resize": {**template["spatial_transform"]["forward_resize"], "input_hw": list(shape[:2])},
                                           "inverse_resize": {**template["spatial_transform"]["inverse_resize"], "output_hw": list(shape[:2])}}
            records.append(record)
    records.sort(key=lambda record: record["sample_id"])
    contract = AcdcContract(manifest={"records": records}, manifest_hash="synthetic-fixture-" + "0" * 46,
                            shared_grid_hash=FROZEN_SELF_AUDIT_HISTORICAL_224_SHARED_GRID_SHA256,
                            adapter_spec_sha256=FROZEN_ADAPTER_SPEC_SHA256, freeze_id="SYNTHETIC_FIXTURE", synthetic=True)
    return contract, records, image_root, image_root
