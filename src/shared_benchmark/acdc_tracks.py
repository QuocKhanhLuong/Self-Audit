"""ACDC Track A (GT-free common adapter) and Track B (raw_id_majority_vote_v1 diagnostic).

Both tracks consume the same sealed native raw run (``acdc_native``) and verify its external
receipt before reading anything else. Track A semantics come only from the frozen
``cardiac_adapter_v2``; Track B is a GT-assisted diagnostic whose outputs never return to a
producer, a hyperparameter, a seed, Track A or model selection (nothing here writes into a raw
or semantic root it does not own). Metrics reuse the frozen reference evaluator's functions.
"""
from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .acdc_native import (
    FROZEN_ADAPTER_SPEC,
    ROOT,
    TRACKS_CONTRACT,
    AcdcImageSource,
    access_guard,
    working_directory,
    verify_acdc_seal_receipt,
)
from .artifacts import RAW_COMPLETE, run_adapter_after_raw, verify_semantic_partition
from .native_artifacts import NativeContractError
from .native_protocol import canonical_json, file_hash, value_hash
from .semantic_contract import BG, FROZEN_ADAPTER_SPEC_SHA256, LV, MYO, RV, VOID

TRACK_B_RULE = "raw_id_majority_vote_v1"
TRACK_B_KIND = "GT_ASSISTED_DIAGNOSTIC_ONLY"
VALID_REFERENCE = (BG, RV, MYO, LV)  # also the fixed tie order: argmax returns the first maximum
FOREGROUND = (RV, MYO, LV)
REFERENCE_EVALUATOR = ROOT / "scripts" / "evaluate_cardiac_baseline_reference.py"
_TRACK_A_ENTRIES = {"semantic", "access_log.json", "track_a_index.json"}


class TrackError(ValueError):
    pass


def tracks_contract():
    contract = json.loads(TRACKS_CONTRACT.read_text(encoding="utf-8"))
    rule = contract["track_b"]
    if (contract.get("schema_version") != "shared_benchmark.acdc_native_tracks.v1" or rule["name"] != TRACK_B_RULE
            or rule["tie_order"] != ["BG", "RV", "MYO", "LV"] or rule["split_raw_id"] or rule["hungarian"]
            or rule["class_quotas"] or rule["dice_optimisation"] or rule["no_valid_reference_pixels"] != "VOID"
            or contract["track_a"]["adapter_spec_sha256"] != FROZEN_ADAPTER_SPEC_SHA256
            or file_hash(ROOT / contract["metrics"]["reuses"]) != contract["metrics"]["reuses_sha256"]):
        raise TrackError("ACDC tracks contract does not match the implemented rule")
    return contract, file_hash(TRACKS_CONTRACT)


def _reference_evaluator():
    spec = importlib.util.spec_from_file_location("_acdc_reference_evaluator", REFERENCE_EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _outside(path, *roots):
    path = Path(path).resolve()
    for root in roots:
        root = Path(root).resolve()
        if path == root or root in path.parents or path in root.parents:
            raise TrackError(f"{path} must not overlap {root}")
    return path


def _write_exclusive(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical_json(value) + b"\n")


# ---------------------------------------------------------------- Track A (GT-free)

def run_track_a(raw_root, receipt_path, image_root, semantic_root, *, trusted_roots=()):
    """Frozen cardiac_adapter_v2 over every sealed raw map. No GT parameter exists on this path."""
    tracks_contract()
    verified = verify_acdc_seal_receipt(raw_root, receipt_path)
    run, artifacts = verified["run"], verified["artifacts"]
    raw_root = Path(raw_root).resolve()
    if file_hash(FROZEN_ADAPTER_SPEC) != FROZEN_ADAPTER_SPEC_SHA256:
        raise TrackError("frozen adapter spec changed")
    adapter_spec = json.loads(FROZEN_ADAPTER_SPEC.read_text(encoding="utf-8"))
    semantic_root = _outside(semantic_root, raw_root, image_root)
    if semantic_root.exists() and any(semantic_root.iterdir()):
        raise TrackError("refuse nonempty Track A root")
    semantic_root.mkdir(parents=True, exist_ok=True)
    source = AcdcImageSource(run["records"], image_root)
    sealed_files = [raw_root / name for name in ("run.json", "access_log.json", "acdc_raw_seal.json")]
    for artifact in artifacts.values():
        sealed_files += [artifact.directory / name for name in ("raw_partition.npy", "metadata.json", "state.json")]
    guard = access_guard(source, output_root=semantic_root, trusted_roots=trusted_roots,
                         readable_files=[*sealed_files, Path(receipt_path).absolute()])
    rows = []
    with working_directory(semantic_root), guard:
        for record in source.records:
            sample_id = record["sample_id"]
            raw = artifacts[sample_id]
            central = source.central_plane(record)
            if source.input_receipt(record)["central_image_sha256"] != raw.metadata["baseline_provenance"]["central_image_sha256"]:
                raise TrackError(f"central image differs from the producer's sealed input: {sample_id}")
            semantic = run_adapter_after_raw(raw, semantic_root=semantic_root / "semantic", record=record,
                                             central_image=central, adapter_spec=adapter_spec,
                                             baseline_name=run["method"], baseline_mode=run["baseline_mode"])
            rows.append({"sample_id": sample_id, "directory": str(semantic.directory.relative_to(semantic_root)),
                         "scientific_payload_hash": semantic.metadata["scientific_payload_hash"],
                         "coverage": float(semantic.validity_map.mean())})
    (semantic_root / "access_log.json").write_bytes(canonical_json(guard.log) + b"\n")
    if any(not entry["allowed"] for entry in guard.log):
        raise NativeContractError("Track A GT-firewall violation")
    index = {"schema": "native_acdc.track-a-index.v1", "track": "A", "adapter_version": "cardiac_adapter_v2",
             "adapter_spec_sha256": FROZEN_ADAPTER_SPEC_SHA256, "tracks_contract_sha256": file_hash(TRACKS_CONTRACT),
             "raw_seal_sha256": verified["seal"]["seal_sha256"], "receipt_sha256": verified["receipt"]["receipt_sha256"],
             "access_log_sha256": file_hash(semantic_root / "access_log.json"), "samples": rows}
    index["index_sha256"] = value_hash(index)
    _write_exclusive(semantic_root / "track_a_index.json", index)
    return index


def verify_track_a(raw_root, receipt_path, image_root, semantic_root):
    """Recompute every semantic map from the verified raw map and image-only central plane."""
    verified = verify_acdc_seal_receipt(raw_root, receipt_path)
    semantic_root = Path(semantic_root).resolve()
    index = json.loads((semantic_root / "track_a_index.json").read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in index.items() if key != "index_sha256"}
    if (value_hash(unsigned) != index.get("index_sha256") or index["raw_seal_sha256"] != verified["seal"]["seal_sha256"]
            or file_hash(semantic_root / "access_log.json") != index["access_log_sha256"]
            or {path.name for path in semantic_root.iterdir()} != _TRACK_A_ENTRIES):
        raise TrackError("Track A index does not bind this raw run")
    adapter_spec = json.loads(FROZEN_ADAPTER_SPEC.read_text(encoding="utf-8"))
    source = AcdcImageSource(verified["run"]["records"], image_root)
    maps = {}
    for row in index["samples"]:
        record = next(r for r in source.records if r["sample_id"] == row["sample_id"])
        semantic = verify_semantic_partition(semantic_root / row["directory"], raw_artifact=verified["artifacts"][row["sample_id"]],
                                             record=record, central_image=source.central_plane(record),
                                             adapter_spec=adapter_spec)
        if semantic.metadata["scientific_payload_hash"] != row["scientific_payload_hash"]:
            raise TrackError("semantic artifact differs from the Track A index")
        maps[row["sample_id"]] = semantic.semantic_map
    if set(maps) != set(verified["artifacts"]):
        raise TrackError("Track A does not cover the sealed cohort")
    return verified, index, maps


# ---------------------------------------------------------------- Track B (GT-assisted diagnostic)

def restore_raw_partition(partition, native_hw):
    """Frozen nearest-exact label inverse applied to anonymous IDs (same pixel correspondence as Track A)."""
    import torch
    import torch.nn.functional as F
    array = np.asarray(partition)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer) or array.min() < 0 or array.max() >= 2 ** 24:
        raise TrackError("raw partition must be a 2-D nonnegative integer map below 2**24")
    restored = F.interpolate(torch.from_numpy(array.astype(np.float32))[None, None], size=tuple(native_hw),
                             mode="nearest-exact")[0, 0].numpy()
    return np.asarray(np.rint(restored), dtype=np.int64)


def raw_id_majority_vote(raw, reference):
    """raw_id_majority_vote_v1 for one sample: whole raw IDs -> BG/RV/MYO/LV, VOID without valid pixels."""
    raw, reference = np.asarray(raw), np.asarray(reference)
    if raw.shape != reference.shape or raw.ndim != 2:
        raise TrackError("raw map and reference must share one 2-D grid")
    semantic = np.full(raw.shape, VOID, dtype=np.uint8)
    rows = []
    for raw_id in np.unique(raw):
        member = raw == raw_id
        values = reference[member]
        valid = np.isin(values, VALID_REFERENCE)
        counts = np.bincount(values[valid].astype(np.int64), minlength=4)[:4]
        assigned = VOID if counts.sum() == 0 else int(np.argmax(counts))  # first maximum: BG, RV, MYO, LV
        semantic[member] = assigned
        rows.append({"raw_id": int(raw_id), "assigned": int(assigned), "pixels": int(member.sum()),
                     "valid_counts": [int(value) for value in counts]})
    return semantic, rows


# ---------------------------------------------------------------- common ACDC metrics

def _volumes(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record["volume_id"])].append(record)
    for volume_id, members in grouped.items():
        depth = int(members[0]["depth"])
        if sorted(int(r["slice_index"]) for r in members) != list(range(depth)):
            raise TrackError(f"common ACDC metrics need complete volumes; {volume_id} has "
                             f"{len(members)}/{depth} slices")
    return {key: sorted(value, key=lambda r: int(r["slice_index"])) for key, value in sorted(grouped.items())}


def score_native(records, native_by_id, gt_root, *, reference_for_volume=None):
    """Per-volume 3D Dice/IoU + coverage, patient macro (reference evaluator semantics)."""
    from self_audit_maskfree.evaluation.metrics import dice_iou_volume, patient_macro
    evaluator = _reference_evaluator()
    per_volume = []
    for volume_id, members in _volumes(records).items():
        first = members[0]
        native_shape = tuple(int(v) for v in first["native_shape"])
        prediction = np.full(native_shape, VOID, dtype=np.uint8)
        for record in members:
            prediction[:, :, int(record["slice_index"])] = native_by_id[record["sample_id"]]
        truth, locator, gt_sha = reference_for_volume(first) if reference_for_volume else _load_reference(evaluator, first, gt_root)
        scores = dice_iou_volume(np.moveaxis(prediction, 2, 0), np.moveaxis(truth, 2, 0), classes=FOREGROUND)
        validity = prediction != VOID
        per_volume.append({"patient_id": str(first["patient_id"]), "volume_id": volume_id,
                           "frame_index": int(first["frame_index"]), "slice_count": len(members),
                           "coverage": float(validity.mean()), "void_voxels": int((~validity).sum()),
                           "gt_locator": locator, "gt_sha256": gt_sha,
                           "scores": {str(key): value for key, value in scores.items()}})
    patients = evaluator._patient_scores(per_volume)
    coverage = [item["coverage"] for item in per_volume]
    return {"dice_patient_macro": patient_macro(patients, key="dice", classes=FOREGROUND),
            "iou_patient_macro": patient_macro(patients, key="iou", classes=FOREGROUND),
            "coverage": {"mean": float(np.mean(coverage)), "min": float(np.min(coverage)), "max": float(np.max(coverage))},
            "per_volume": per_volume}


def _load_reference(evaluator, record, gt_root):
    path = evaluator.gt_path_for_record(record, Path(gt_root))
    truth = evaluator._load_gt(path, tuple(int(v) for v in record["native_shape"]))
    return truth, str(path.relative_to(Path(gt_root).resolve())), file_hash(path)


def _summary_header(track, verified, contract_sha):
    run = verified["run"]
    return {"schema": f"native_acdc.track-{track.lower()}-summary.v1", "track": track, "method": run["method"],
            "baseline_mode": run["baseline_mode"], "adaptation_sha256": run["adaptation_sha256"],
            "contract": run["contract"], "raw_seal_sha256": verified["seal"]["seal_sha256"],
            "receipt_sha256": verified["receipt"]["receipt_sha256"], "tracks_contract_sha256": contract_sha,
            "reference_evaluator_sha256": file_hash(REFERENCE_EVALUATOR),
            "sample_count": len(verified["artifacts"])}


def _new_output(output_dir, *protected):
    output_dir = _outside(output_dir, *protected)
    if output_dir.exists():
        raise TrackError(f"refusing to overwrite evaluator output: {output_dir}")
    output_dir.mkdir(parents=True)
    return output_dir


def evaluate_track_a(raw_root, receipt_path, image_root, semantic_root, gt_root, output_dir):
    _, contract_sha = tracks_contract()
    verified, index, maps = verify_track_a(raw_root, receipt_path, image_root, semantic_root)  # before GT access
    output_dir = _new_output(output_dir, raw_root, semantic_root, image_root)
    evaluator = _reference_evaluator()
    records = verified["run"]["records"]
    native = {r["sample_id"]: evaluator.restore_semantic_slice(maps[r["sample_id"]], tuple(int(v) for v in r["native_hw"]))
              for r in records}
    summary = {**_summary_header("A", verified, contract_sha), "track_a_index_sha256": index["index_sha256"],
               "adapter_spec_sha256": FROZEN_ADAPTER_SPEC_SHA256, **score_native(records, native, gt_root)}
    _write_exclusive(output_dir / "summary.json", summary)
    return summary


def evaluate_track_b(raw_root, receipt_path, gt_root, output_dir):
    """GT-assisted diagnostic. Its mappings stay in output_dir and are never consumed by any producer or track."""
    _, contract_sha = tracks_contract()
    verified = verify_acdc_seal_receipt(raw_root, receipt_path)  # before GT access
    output_dir = _new_output(output_dir, raw_root)
    evaluator = _reference_evaluator()
    records = verified["run"]["records"]
    references, native, mappings = {}, {}, {}

    def reference_for_volume(record):
        key = str(record["volume_id"])
        if key not in references:
            references[key] = _load_reference(evaluator, record, gt_root)
        return references[key]

    for volumes in _volumes(records).values():
        for record in volumes:
            truth = reference_for_volume(record)[0]
            raw = verified["artifacts"][record["sample_id"]]
            if raw.metadata["completion_status"] != RAW_COMPLETE:
                raise TrackError("Track B requires RAW_COMPLETE raw maps")
            restored = restore_raw_partition(raw.partition, tuple(int(v) for v in record["native_hw"]))
            native[record["sample_id"]], mappings[record["sample_id"]] = raw_id_majority_vote(
                restored, truth[:, :, int(record["slice_index"])])
    summary = {**_summary_header("B", verified, contract_sha), "rule": TRACK_B_RULE, "kind": TRACK_B_KIND,
               "use_restriction": "diagnostic only; never input to producers, hyperparameters, seeds, Track A or selection",
               "raw_id_mappings": mappings,
               **score_native(records, native, gt_root, reference_for_volume=reference_for_volume)}
    _write_exclusive(output_dir / "summary.json", summary)
    return summary
