"""Native Track B: Step I executable, Step II gated. No GT is read before the
external seal receipt, the raw seal and the locked evaluator configuration verify;
this evaluator never writes to or reseals the raw run."""
import hashlib
import json
from pathlib import Path

import numpy as np

from shared_benchmark.native_artifacts import verify_seal_receipt
from shared_benchmark.native_protocol import load_lock, value_hash, ProtocolBlocked

try:
    from .step1 import per_image_segment_dice
except ImportError:  # loaded by file path
    import importlib.util as _u
    _spec = _u.spec_from_file_location("dss_us_track_b__sibling", Path(__file__).with_name("_sibling.py"))
    _sibling = _u.module_from_spec(_spec)
    _spec.loader.exec_module(_sibling)
    per_image_segment_dice = _sibling.load("step1").per_image_segment_dice

GT_SCHEMA = "medical-native.gt-evaluation.v1"


def _load_gt(record, manifest_dir):
    path = Path(record["gt_path"])
    path = path if path.is_absolute() else manifest_dir / path
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["gt_sha256"]:
        raise ValueError(f"GT hash mismatch: {record['sample_id']}")
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        import cv2
        array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if array is None or array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"GT must be a single-channel integer label map: {record['sample_id']}")
    return array


def evaluate_native(raw_root, *, seal_receipt, protocol_path, gt_manifest):
    raw = verify_seal_receipt(Path(raw_root), Path(seal_receipt))
    config = load_lock(protocol_path, purpose="native_track_b")
    if raw["run"]["stage"] != config["stage"] or raw["run"]["config_sha256"] != value_hash(config):
        raise ValueError("native evaluator stage/protocol mismatch with the sealed raw run")
    if config["stage"] != "I":
        raise ProtocolBlocked(["Step II evaluated stage and label-consistency metric unresolved"])
    evaluation = config["native_track_b"]
    # GT is opened only now, after receipt, seal and locked evaluator verified.
    manifest_path = Path(gt_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != GT_SCHEMA or manifest.get("dataset") != config["dataset"]:
        raise ValueError("GT manifest schema/dataset mismatch")
    records = {r["sample_id"]: r for r in manifest["records"]}
    sealed = [r["sample_id"] for r in raw["seal"]["samples"]]
    if sorted(records) != sorted(sealed):
        raise ValueError("GT manifest must cover exactly the sealed samples")
    root = Path(raw_root).resolve()

    def pairs():
        for sample_id in sealed:
            with np.load(root / sample_id / "raw.npz", allow_pickle=False) as data:
                partition = data["partition"]
            gt = _load_gt(records[sample_id], manifest_path.parent)
            if gt.shape != partition.shape:
                raise ValueError(f"native evaluation grid mismatch: {sample_id}")
            yield sample_id, partition, gt

    result = per_image_segment_dice(pairs(), n_classes=evaluation["n_classes"])
    result.update({
        "evaluator": "dss_us.step1.per_image_remapped_dice",
        "raw_seal_sha256": raw["seal"]["seal_sha256"], "seal_receipt_sha256": raw["receipt"]["receipt_sha256"],
        "config_sha256": raw["run"]["config_sha256"], "native_evaluator_spec_sha256": config["native_evaluator_spec_sha256"],
        "gt_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    })
    return result
