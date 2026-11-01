#!/usr/bin/env python3
"""Pinned historical VAL18 context diagnostic; inference only, never training.

Run from the repository root. Workers use separate, pinned historical runtimes.
Scoring opens reference masks only after shared-round authorization verifies both
diagnostics' locks/freezes or records an explicit B-blocked A-only exposure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.context_prediction_worker import (  # noqa: E402
    _load_json, _overlap, _read_config, _resolve_ref, _sha256, _validate_config,
    protocol_snapshot, protocol_sha256, run_worker,
)
from scripts import context_prediction_worker as _worker  # noqa: E402
from self_audit import diagnostic_protocol as _round_protocol  # noqa: E402

TEACHER_MANIFEST = "ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa"
TEACHER_PAYLOAD = "0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6"
STUDENT_MANIFEST = "197dd4367877c51bc21b91d172f25442549ef9a3d0e422d28118bb07308067de"
STUDENT_PAYLOAD = "32a35c1fc3b83d57d083de4ed17ab80bda6ba38a8ebc50133e5aede17ab00380"
VAL_MANIFEST = "cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50"
TRAIN_MANIFEST = "e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4"
SPLIT_SHA = "b826eaf494760079aa559b0c0c12f7444951d779ebbb5ed291c0398f20bf835f"
HISTORICAL_WIDE_MACRO4 = 0.777413392930114
# Fixed before opening labels: 0.0001 Dice = 0.01 percentage point.
REPRODUCTION_ATOL = 1e-4


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _source_digest(root):
    root = Path(root)
    files = sorted((root / "self_audit").glob("*.py")) + [root / "train.py"]
    if len(files) < 2 or not all(p.is_file() for p in files):
        raise ValueError(f"incomplete runtime source: {root}")
    return hashlib.sha256("".join(f"{p.name}:{_sha256(p)}\n" for p in files).encode()).hexdigest()


def validate_pair(teacher, student):
    """Bind the compact export to its actual raw-student teacher, not a nickname."""
    identity = student.get("lineage", {}).get("teacher_identity", {})
    if identity.get("payload_sha256") != teacher.get("payload_sha256"):
        raise ValueError("student was not distilled from this teacher payload")
    if identity.get("manifest_sha256") != TEACHER_MANIFEST:
        raise ValueError("student teacher manifest identity differs")
    if identity.get("inference_state") != "student":
        raise ValueError("historical teacher must use raw student state, not EMA")
    if identity.get("train_manifest_sha256") != TRAIN_MANIFEST:
        raise ValueError("teacher TRAIN lineage differs")
    lineage = student.get("lineage", {})
    if lineage.get("data_sha256") != TRAIN_MANIFEST or lineage.get("label_budget") != 8:
        raise ValueError("compact must retain historical TRAIN/L8 lineage")
    if sorted(identity.get("labeled_ids", [])) != sorted(lineage.get("lids", [])):
        raise ValueError("teacher and compact labeled patient identities differ")


def preflight(config):
    """Verify metadata/assets without deserializing weights or reading GT bytes."""
    config = _validate_config(config)
    synthetic = config.get("synthetic_fixture", False)
    required = {
        "teacher": (TEACHER_MANIFEST, TEACHER_PAYLOAD),
        "student": (STUDENT_MANIFEST, STUDENT_PAYLOAD),
    }
    if config["parity"] != {"logit_atol": 1e-6, "logit_rtol": 1e-5}:
        raise ValueError("parity tolerances are fixed for this diagnostic")
    split = None
    split_path = ROOT / "splits/acdc_reviewed_locked_train80_val18.json"
    if not synthetic:
        for role, (manifest_sha, payload_sha) in required.items():
            if (config[role]["manifest_sha256"], config[role]["payload_sha256"]) != (manifest_sha, payload_sha):
                raise ValueError(f"{role}: this protocol is only for the matched historical pair")
        if config["val_manifest"]["sha256"] != VAL_MANIFEST:
            raise ValueError("historical native VAL18 manifest must remain byte-identical")
        if _sha256(split_path) != SPLIT_SHA:
            raise ValueError("locked split metadata changed")
        split = _load_json(split_path)
    missing, checked, models = [], [], {}

    def inspect_file(path, expected, name):
        path = Path(path)
        if not path.is_file():
            missing.append({"asset": name, "path": str(path.resolve())})
            return False
        if _sha256(path) != expected:
            raise ValueError(f"{name} SHA-256 mismatch: {path}")
        checked.append({"asset": name, "path": str(path.resolve()), "sha256": expected})
        return True

    for role in ("teacher", "student"):
        spec = config[role]
        folder = Path(spec["path"])
        if inspect_file(folder / "MANIFEST.json", spec["manifest_sha256"], role + " manifest"):
            meta = models[role] = _load_json(folder / "MANIFEST.json")
            if meta.get("payload_sha256") != spec["payload_sha256"]:
                raise ValueError(f"{role} manifest/payload binding mismatch")
            if synthetic and meta.get("synthetic") is not True:
                raise ValueError(f"{role} fixture manifest must explicitly declare synthetic=true")
        inspect_file(folder / "weights.pt", spec["payload_sha256"], role + " weights")
        runtime = Path(spec["runtime_root"])
        if not runtime.is_dir():
            missing.append({"asset": role + " pinned runtime", "path": str(runtime.resolve())})
        elif _source_digest(runtime) != spec["runtime_source_sha256"]:
            raise ValueError(f"{role} pinned runtime source hash mismatch")
        else:
            checked.append({"asset": role + " runtime", "sha256": spec["runtime_source_sha256"]})
    if set(models) == {"teacher", "student"}:
        if synthetic:
            if any(models[role].get("synthetic") is not True for role in ("teacher", "student")):
                raise ValueError("synthetic fixture requires two explicitly synthetic model manifests")
        else:
            validate_pair(models["teacher"], models["student"])
            if models["teacher"].get("source_sha256") != config["teacher"]["runtime_source_sha256"]:
                raise ValueError("teacher runtime must match checkpoint source closure")
            lineage = models["student"]["lineage"]
            lids, uids = set(lineage["lids"]), set(lineage["uids"])
            if lids & uids or lids | uids != set(split["train"]) or len(lids) != 8:
                raise ValueError("compact L/U IDs must partition the locked TRAIN80")

    vm_path = Path(config["val_manifest"]["path"])
    frames = None
    expected_vm_sha = config["val_manifest"]["sha256"] if synthetic else VAL_MANIFEST
    if inspect_file(vm_path, expected_vm_sha, "development manifest"):
        vm = _load_json(vm_path)
        ids = [p["id"] for p in vm["patients"]]
        if vm.get("role") != "development":
            raise ValueError("context diagnostic manifest must have development role")
        if synthetic and vm.get("synthetic") is not True:
            raise ValueError("synthetic fixture manifest must explicitly declare synthetic=true")
        if not synthetic and vm.get("synthetic") is not False:
            raise ValueError("historical VAL18 must be real, exposed development data")
        if len(ids) != len(set(ids)):
            raise ValueError("development manifest contains duplicate patient IDs")
        if not synthetic and set(ids) != set(split["val"]):
            raise ValueError("VAL18 patient IDs differ from the locked split")
        if not synthetic and set(ids) & set(split["train"]):
            raise ValueError("VAL/TRAIN patient overlap")
        frames = sum(len(p["masks"]) for p in vm["patients"])
        for patient in vm["patients"]:
            image = _resolve_ref(vm_path.parent, patient["image"], "image")
            inspect_file(image, patient["image_sha256"], patient["id"] + " image")
            # Availability only: mask payloads and even their hashes wait until freeze.
            for frame, mask in patient["masks"].items():
                path = _resolve_ref(vm_path.parent, mask["path"], "mask")
                if not path.is_file():
                    missing.append({"asset": f"{patient['id']} frame {frame} GT", "path": str(path)})
    return {
        "schema": "self_audit.context_diagnostic.preflight.v1",
        "status": "READY_FOR_INFERENCE" if not missing else "NOT_RUN_MISSING_ASSETS",
        "patient_inference_run": False, "ground_truth_payloads_opened": False,
        "pair": "synthetic_fixture" if synthetic else "historical_raw_B8_and_wide_plain_KD",
        "synthetic_fixture": synthetic, "patients": len(ids) if frames is not None else 0,
        "frames": frames, "checked": checked, "missing": missing,
        "limits": "Preflight is asset verification, not model loading or inference success.",
    }


def verify_inventory(config, role, receipt, manifest, output, expected_protocol):
    if (receipt.get("stage") != "A" or receipt.get("role") != role
            or receipt.get("schema") != "self_audit.context_diagnostic.predictions.v1"):
        raise ValueError("wrong worker receipt schema/role")
    if receipt.get("protocol") != expected_protocol or receipt.get("protocol_sha256") != protocol_sha256(expected_protocol):
        raise ValueError("worker diagnostic code/environment protocol mismatch")
    for receipt_key, expected in {
        "val_manifest_sha256": config["val_manifest"]["sha256"],
        "model_manifest_sha256": config[role]["manifest_sha256"],
        "model_payload_sha256": config[role]["payload_sha256"],
        "runtime_source_sha256": config[role]["runtime_source_sha256"],
    }.items():
        if receipt.get(receipt_key) != expected:
            raise ValueError(f"worker receipt identity mismatch: {receipt_key}")
    parity = receipt.get("parity", {})
    if parity.get("mismatches") != 0 or parity.get("slices", 0) <= 0:
        raise ValueError("worker must pass lambda=1 parity before scoring")
    images = {p["id"]: p["image_sha256"] for p in manifest["patients"]}
    expected_keys = {(p["id"], int(t), lam) for p in manifest["patients"] for t in p["masks"] for lam in (0., .5, 1.)}
    found = {}
    for item in receipt["records"]:
        key = (item["patient"], item["frame"], item["lambda"])
        if key in found or key not in expected_keys:
            raise ValueError("duplicate or unexpected prediction identity")
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(Path(output).resolve()) or _sha256(path) != item["sha256"]:
            raise ValueError("prediction path/hash mismatch")
        if item["image_sha256"] != images[item["patient"]]:
            raise ValueError("prediction/image identity mismatch")
        found[key] = item
    if set(found) != expected_keys:
        raise ValueError("prediction inventory incomplete; GT must remain unopened")
    return found


def _csv(path, rows):
    rows = list(rows)
    with Path(path).open("x", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _source_files(root):
    root = Path(root)
    return sorted((root / "self_audit").glob("*.py")) + [root / "train.py"]


def _collect_lock_assets(config_path, config_sha, config):
    """Bind config, source/runtime files, weights, manifests, and images only."""
    assets = {}

    def add(name, path, expected_sha):
        path = Path(path).resolve()
        expected_sha = str(expected_sha)
        if not path.is_file() or _sha256(path) != expected_sha:
            raise ValueError(f"locked asset hash mismatch: {name}")
        assets[name] = {"path": str(path), "sha256": expected_sha}

    add("config", config_path, config_sha)
    manifest_path = Path(config["val_manifest"]["path"]).resolve()
    add("development_manifest", manifest_path, config["val_manifest"]["sha256"])
    manifest = _load_json(manifest_path)
    for patient in manifest["patients"]:
        image = _resolve_ref(manifest_path.parent, patient["image"], "image")
        add(f"image:{patient['id']}", image, patient["image_sha256"])
    if not config.get("synthetic_fixture", False):
        split_path = ROOT / "splits/acdc_reviewed_locked_train80_val18.json"
        add("locked_train_val_split", split_path, SPLIT_SHA)
    for role in ("teacher", "student"):
        spec = config[role]
        model_dir = Path(spec["path"]).resolve()
        add(f"{role}_manifest", model_dir / "MANIFEST.json", spec["manifest_sha256"])
        add(f"{role}_weights", model_dir / "weights.pt", spec["payload_sha256"])
        runtime = Path(spec["runtime_root"]).resolve()
        if _source_digest(runtime) != spec["runtime_source_sha256"]:
            raise ValueError(f"{role} runtime source hash mismatch while locking")
        for index, source in enumerate(_source_files(runtime)):
            add(f"{role}_runtime_file_{index:03d}", source, _sha256(source))
    return assets


def _assert_round_disjoint(round_dir, config_path, config):
    output = Path(round_dir).resolve()
    manifest_path = Path(config["val_manifest"]["path"]).resolve()
    protected = [Path(config_path).resolve(), manifest_path.parent]
    protected.extend(Path(config[role]["path"]).resolve() for role in ("teacher", "student"))
    protected.extend(Path(config[role]["runtime_root"]).resolve() for role in ("teacher", "student"))
    # preflight() already records an absent VAL manifest as a structured
    # missing asset.  Do not turn that report into an unstructured JSON-open
    # exception while checking output-directory overlap.
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        for patient in manifest["patients"]:
            protected.append(_resolve_ref(manifest_path.parent, patient["image"], "image").parent)
            protected.extend(
                _resolve_ref(manifest_path.parent, mask["path"], "mask").parent
                for mask in patient["masks"].values()
            )
    if any(_overlap(output, source) for source in protected):
        raise ValueError("round directory overlaps config or input assets")


def _read_preflight(round_dir, config_path, config_sha):
    record = _load_json(Path(round_dir) / "PREFLIGHT_A.json")
    if (record.get("schema") != "self_audit.context_diagnostic.preflight_record.v1"
            or record.get("config_sha256") != config_sha
            or Path(record.get("config_path", "")).resolve() != Path(config_path).resolve()):
        raise ValueError("preflight receipt/config identity mismatch")
    if record["report"].get("status") != "READY_FOR_INFERENCE" or record["report"].get("missing"):
        raise FileNotFoundError("preflight is not ready; inspect PREFLIGHT_A.json")
    return record


def preflight_stage(config_path, config_sha, round_dir):
    """Run asset-only checks and create an append-once round directory."""
    config_path = Path(config_path).resolve()
    config = _read_config(config_path, config_sha)
    report = preflight(config)
    round_dir = Path(round_dir).resolve()
    if round_dir.exists():
        raise ValueError("round directory must be new at preflight")
    _assert_round_disjoint(round_dir, config_path, config)
    round_dir.mkdir(parents=True)
    record = {
        "schema": "self_audit.context_diagnostic.preflight_record.v1",
        "config_path": str(config_path), "config_sha256": config_sha,
        "report": report,
    }
    _round_protocol.write_once(round_dir / "PREFLIGHT_A.json", record)
    return report


def _execution_snapshot(config):
    import torch

    torch.set_num_threads(config["threads"])
    return protocol_snapshot(config["device"], config["threads"])


def lock_stage(config_path, config_sha, round_dir):
    """Create the A lock after preflight, before any model prediction."""
    config_path = Path(config_path).resolve()
    config = _read_config(config_path, config_sha)
    preflight_record = _read_preflight(round_dir, config_path, config_sha)
    current = preflight(config)
    if current != preflight_record["report"]:
        raise ValueError("preflight assets changed before protocol lock")
    execution = _execution_snapshot(config)
    protocol = {"config": config, "config_sha256": config_sha, "execution": execution}
    assets = _collect_lock_assets(config_path, config_sha, config)
    return _round_protocol.create_lock(Path(round_dir).resolve(), "A", protocol, assets)


def _write_synthetic_fixture(input_dir):
    """Compatibility entry point for the shared actual-model synthetic fixture."""
    from self_audit.diagnostic_fixture import build_diagnostic_fixture

    fixture = build_diagnostic_fixture(input_dir)
    return Path(fixture["context_config_path"]), fixture["context_config_sha256"]


def predict_stage(config_path, config_sha, round_dir):
    """Run image-only A predictions and freeze all shards and worker receipts."""
    config_path = Path(config_path).resolve()
    config = _read_config(config_path, config_sha)
    round_dir = Path(round_dir).resolve()
    preflight_record = _read_preflight(round_dir, config_path, config_sha)
    lock = _round_protocol.read_lock(round_dir, "A")
    execution = _execution_snapshot(config)
    expected_protocol = {"config": config, "config_sha256": config_sha, "execution": execution}
    if lock.get("protocol") != expected_protocol:
        raise ValueError("current config/runtime differs from locked A protocol")
    if preflight_record["config_sha256"] != config_sha:
        raise ValueError("preflight changed before prediction")
    stage_dir = round_dir / "A"
    if stage_dir.exists():
        raise ValueError("A prediction directory already exists")
    stage_dir.mkdir()
    execution_sha = protocol_sha256(execution)
    if config.get("synthetic_fixture", False):
        # The fixture uses the same strict pinned-runtime loaders as an
        # ordinary run; only its model/data assets are synthetic.  Calling the
        # worker in-process also lets bounded tests guard GT access directly.
        for role in ("teacher", "student"):
            run_worker(config, role, stage_dir / role, expected_protocol_sha256=execution_sha, stage="A")
    else:
        for role in ("teacher", "student"):
            log_path = stage_dir / f"{role}.worker.log"
            with log_path.open("x") as log:
                subprocess.run(
                    [sys.executable, str(ROOT / "scripts/context_prediction_worker.py"),
                     "--config", str(config_path), "--config-sha256", config_sha,
                     "--protocol-sha256", execution_sha, "--stage", "A", "--role", role,
                     "--out", str(stage_dir / role)],
                    check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                )
    manifest_path = Path(config["val_manifest"]["path"]).resolve()
    manifest = _load_json(manifest_path)
    all_files, receipt_hashes, counts = [], {}, {}
    for role in ("teacher", "student"):
        role_dir = stage_dir / role
        receipt_path = role_dir / "PREDICTIONS_FROZEN.json"
        receipt = _load_json(receipt_path)
        found = verify_inventory(config, role, receipt, manifest, role_dir, execution)
        receipt_hashes[role] = _sha256(receipt_path)
        counts[role] = len(found)
        all_files.extend(item["path"] for item in found.values())
        all_files.append(str(receipt_path.resolve()))
    frozen = _round_protocol.freeze_predictions(
        round_dir, "A", sorted(all_files),
        {"config_sha256": config_sha, "execution_sha256": execution_sha,
         "worker_receipt_sha256": receipt_hashes, "prediction_counts": counts,
         "ground_truth_opened_before_freeze": False},
    )
    return frozen


def _prevalidate_prediction_payloads(manifest_path, manifest, inventories, np):
    """Check every image/prediction array's native shape and label domain pre-GT."""
    for patient in manifest["patients"]:
        image_path = _resolve_ref(manifest_path.parent, patient["image"], "image")
        image = np.load(image_path, mmap_mode="r", allow_pickle=False)
        if image.ndim != 4 or min(image.shape) < 1:
            raise ValueError(f"prepared image is not TZYX for {patient['id']}")
        expected_shape = (int(image.shape[1]), int(image.shape[2]), int(image.shape[3]))
        for frame in patient["masks"]:
            frame_id = int(frame)
            if not 0 <= frame_id < image.shape[0]:
                raise ValueError(f"reference frame out of bounds for {patient['id']}")
            for role in ("teacher", "student"):
                for lam in (0.0, 0.5, 1.0):
                    item = inventories[role][(patient["id"], frame_id, lam)]
                    prediction = np.load(item["path"], mmap_mode="r", allow_pickle=False)
                    if tuple(prediction.shape) != expected_shape or list(prediction.shape) != item["shape_zyx"]:
                        raise ValueError("frozen prediction shape differs from native image geometry")
                    if not np.issubdtype(prediction.dtype, np.integer):
                        raise ValueError("frozen prediction labels must be integer class IDs")
                    if prediction.size and (int(prediction.min()) < 0 or int(prediction.max()) >= 4):
                        raise ValueError("frozen prediction contains a class outside BG/RV/MYO/LV")


def score_frozen(config, output, config_sha, b_blocked_reason=None):
    """Score A only after common-round authorization has frozen required predictions."""
    import numpy as np
    import torch
    from self_audit.context_metrics import (
        analyze_frame, aggregate_records, frame_csv_rows, frame_metric_csv_rows,
        patient_class_common_failure_csv_rows, summary_csv_rows,
    )

    config = _validate_config(config)
    output = Path(output).resolve()
    torch.set_num_threads(config["threads"])
    execution = protocol_snapshot(config["device"], config["threads"])
    lock = _round_protocol.read_lock(output, "A")
    if lock["protocol"].get("config") != config or lock["protocol"].get("config_sha256") != config_sha:
        raise ValueError("score config differs from locked A protocol")
    if lock["protocol"].get("execution") != execution:
        raise ValueError("diagnostic code/environment changed after protocol lock")
    preflight_record = _load_json(output / "PREFLIGHT_A.json")
    if (preflight_record.get("config_sha256") != config_sha
            or preflight_record.get("report", {}).get("status") != "READY_FOR_INFERENCE"):
        raise ValueError("scoring requires the matching successful A preflight")
    preflight_config_path = Path(preflight_record.get("config_path", "")).resolve()
    if not preflight_config_path.is_file() or _sha256(preflight_config_path) != config_sha:
        raise ValueError("preflight config bytes changed before scoring")
    vm_path = Path(config["val_manifest"]["path"]).resolve()
    if _sha256(vm_path) != config["val_manifest"]["sha256"]:
        raise ValueError("development manifest changed after protocol lock")
    manifest = _load_json(vm_path)
    inventories, receipt_hashes = {}, {}
    expected_files = set()
    for role in ("teacher", "student"):
        role_dir = output / "A" / role
        receipt_path = role_dir / "PREDICTIONS_FROZEN.json"
        receipt = _load_json(receipt_path)
        inventories[role] = verify_inventory(config, role, receipt, manifest, role_dir, execution)
        receipt_hashes[role] = _sha256(receipt_path)
        expected_files.add(str(receipt_path.resolve()))
        expected_files.update(item["path"] for item in inventories[role].values())
    frozen = _round_protocol.verify_frozen(output, "A")
    frozen_paths = {item["path"] for item in frozen["files"]}
    if frozen_paths != expected_files:
        raise ValueError("shared A freeze does not match complete teacher/student inventories")
    metadata = frozen.get("metadata", {})
    if (metadata.get("config_sha256") != config_sha
            or metadata.get("execution_sha256") != protocol_sha256(execution)
            or metadata.get("worker_receipt_sha256") != receipt_hashes
            or metadata.get("ground_truth_opened_before_freeze") is not False):
        raise ValueError("shared A freeze metadata differs from the locked prediction run")
    _prevalidate_prediction_payloads(vm_path, manifest, inventories, np)

    stage_dir = output / "A"
    score_paths = [stage_dir / name for name in (
        "FRAME_RECORDS.json", "RESULT.json", "frame_transitions.csv", "frame_metrics.csv",
        "patient_transitions.csv", "patient_class_common_failure.csv", "summary.csv",
    )]
    if any(path.exists() for path in score_paths):
        raise ValueError("A scoring outputs already exist; scoring is append-once")
    if b_blocked_reason is not None and (
            not isinstance(b_blocked_reason, str)
            or not b_blocked_reason.strip().upper().startswith("B BLOCKED:")):
        raise ValueError("A-only scoring requires an explicit reason beginning with 'B BLOCKED:'")
    exposure = _round_protocol.authorize_score(output, "A", b_blocked_reason=b_blocked_reason)
    exposure_path = output / "VAL_EXPOSURE.json"
    records = []
    for patient in manifest["patients"]:
        for t, gt_spec in sorted(patient["masks"].items(), key=lambda kv: int(kv[0])):
            gt_path = _resolve_ref(vm_path.parent, gt_spec["path"], "mask")
            if _sha256(gt_path) != gt_spec["sha256"]:
                raise ValueError("GT mask hash mismatch")
            gt = np.load(gt_path, allow_pickle=False)
            predictions = {}
            for role in ("teacher", "student"):
                predictions[role] = {}
                for lam in (0.0, 0.5, 1.0):
                    item = inventories[role][(patient["id"], int(t), lam)]
                    pred = np.load(item["path"], allow_pickle=False)
                    if list(pred.shape) != item["shape_zyx"]:
                        raise ValueError("prediction shape differs from frozen receipt")
                    predictions[role][lam] = pred
            records.append(analyze_frame(patient["id"], int(t), gt,
                                         predictions["teacher"], predictions["student"]))
    summary = aggregate_records(records)
    actual = summary["patient_macro"]["all"]["predictions"]["student"]["1"]["class_first_macro4"]
    if config.get("synthetic_fixture", False):
        reproduction = {"status": "NOT_APPLICABLE_SYNTHETIC", "actual": actual}
        status = "FIXTURE_ONLY"
        interpretation = "Synthetic fixture software check only; not a patient result or model-quality claim."
    else:
        reproduced = actual is not None and abs(actual - HISTORICAL_WIDE_MACRO4) <= REPRODUCTION_ATOL
        reproduction = {"status": "PASS" if reproduced else "MISMATCH", "expected": HISTORICAL_WIDE_MACRO4,
                        "actual": actual, "atol": REPRODUCTION_ATOL, "passed": reproduced}
        status = "DIAGNOSTIC_COMPLETE" if reproduced else "REPRODUCTION_MISMATCH_DO_NOT_INTERPRET"
        interpretation = "Descriptive exposed-VAL diagnostic. Transfer opportunity is not learned transfer or generalization."
    result = {
        "schema": "self_audit.context_diagnostic.result.v2", "stage": "A", "status": status,
        "synthetic_fixture": bool(config.get("synthetic_fixture", False)),
        "config_sha256": config_sha, "A_lock_sha256": lock["record_sha256"],
        "A_freeze_sha256": frozen["record_sha256"],
        "exposure_ledger_sha256": _sha256(exposure_path),
        "protocol_sha256": protocol_sha256(execution),
        "ground_truth_opened_after_freeze_for_scoring": True,
        "B_blocked_reason": exposure.get("B_blocked_reason"),
        "lambda1_wide_reproduction": reproduction,
        "interpretation": interpretation, "summary": summary,
    }
    _round_protocol.write_once(score_paths[0], records)
    _round_protocol.write_once(score_paths[1], result)
    _csv(score_paths[2], frame_csv_rows(records))
    _csv(score_paths[3], frame_metric_csv_rows(records))
    patient_rows = []
    for patient_id, patient in summary["patients"].items():
        for stratum, values in patient["strata"].items():
            for lam, comparison in values["comparisons"].items():
                for name, item in comparison["foreground"].items():
                    patient_rows.append({"patient_id": patient_id, "stratum": stratum,
                                         "lambda": float(lam), "class_name": name,
                                         **item["counts"], **item["rates"]})
    _csv(score_paths[4], patient_rows)
    _csv(score_paths[5], patient_class_common_failure_csv_rows(summary))
    _csv(score_paths[6], summary_csv_rows(summary))
    return result


def score_stage(config_path, config_sha, round_dir, b_blocked_reason=None):
    config = _read_config(Path(config_path).resolve(), config_sha)
    return score_frozen(config, round_dir, config_sha, b_blocked_reason=b_blocked_reason)


def _fixture_smoke(round_dir):
    round_dir = Path(round_dir).resolve()
    if round_dir.exists():
        raise ValueError("fixture-smoke round directory must be new")
    fixture_dir = round_dir.parent / f"{round_dir.name}.fixture_inputs"
    config_path, config_sha = _write_synthetic_fixture(fixture_dir)
    report = preflight_stage(config_path, config_sha, round_dir)
    if report["missing"]:
        raise FileNotFoundError("synthetic fixture unexpectedly missing an input")
    lock = lock_stage(config_path, config_sha, round_dir)
    frozen = predict_stage(config_path, config_sha, round_dir)
    result = score_stage(
        config_path, config_sha, round_dir,
        b_blocked_reason="B BLOCKED: this command is a bounded context-only synthetic fixture smoke.",
    )
    return {"status": result["status"], "round_dir": str(round_dir),
            "fixture_inputs": str(fixture_dir), "A_lock_sha256": lock["record_sha256"],
            "A_freeze_sha256": frozen["record_sha256"],
            "exposure_ledger_sha256": result["exposure_ledger_sha256"],
            "patient_count": result["summary"]["patient_count"]}


def run(config_path, config_sha, output):
    """Compatibility helper that stops after prediction freeze and never scores."""
    report = preflight_stage(config_path, config_sha, output)
    if report["missing"]:
        raise FileNotFoundError(f"missing {len(report['missing'])} required assets; inspect PREFLIGHT_A.json")
    lock_stage(config_path, config_sha, output)
    frozen = predict_stage(config_path, config_sha, output)
    return {"status": "PREDICTIONS_FROZEN_NOT_SCORED", "A_freeze_sha256": frozen["record_sha256"]}


def _add_common_arguments(parser):
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--round-dir", required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight", help="asset-only validation; creates a new round")
    _add_common_arguments(preflight_parser)
    lock_parser = subparsers.add_parser("lock", help="lock context diagnostic A before prediction")
    _add_common_arguments(lock_parser)
    lock_parser.add_argument("--stage", choices=("A",), required=True)
    predict_parser = subparsers.add_parser("predict", help="image-only A prediction and freeze")
    _add_common_arguments(predict_parser)
    predict_parser.add_argument("--stage", choices=("A",), required=True)
    score_parser = subparsers.add_parser("score", help="score only after common-round authorization")
    _add_common_arguments(score_parser)
    score_parser.add_argument("--stage", choices=("A",), required=True)
    score_parser.add_argument("--b-blocked-reason", help="required by the shared API for explicit A-only exposure")
    smoke_parser = subparsers.add_parser("fixture-smoke", help="run the bounded synthetic image/prediction/scoring smoke")
    smoke_parser.add_argument("--round-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "fixture-smoke":
        result = _fixture_smoke(args.round_dir)
    elif args.command == "preflight":
        report = preflight_stage(args.config, args.config_sha256, args.round_dir)
        result = {"status": report["status"], "missing_assets": len(report["missing"]),
                  "synthetic_fixture": report["synthetic_fixture"]}
    elif args.command == "lock":
        record = lock_stage(args.config, args.config_sha256, args.round_dir)
        result = {"status": "LOCKED", "stage": "A", "record_sha256": record["record_sha256"]}
    elif args.command == "predict":
        record = predict_stage(args.config, args.config_sha256, args.round_dir)
        result = {"status": "PREDICTIONS_FROZEN", "stage": "A",
                  "record_sha256": record["record_sha256"], "files": len(record["files"])}
    else:
        scored = score_stage(args.config, args.config_sha256, args.round_dir,
                             b_blocked_reason=args.b_blocked_reason)
        result = {"status": scored["status"], "stage": "A",
                  "exposure_ledger_sha256": scored["exposure_ledger_sha256"],
                  "patient_macro": scored["summary"]["patient_macro"]}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] in {"READY_FOR_INFERENCE", "LOCKED", "PREDICTIONS_FROZEN",
                                      "DIAGNOSTIC_COMPLETE", "FIXTURE_ONLY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
