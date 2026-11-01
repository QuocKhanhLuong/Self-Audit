#!/usr/bin/env python3
"""Pinned historical VAL18 context diagnostic; inference only, never training.

Run from the repository root. Workers use separate, pinned historical runtimes.
The parent opens reference masks only after both prediction inventories freeze.
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
    protocol_snapshot, protocol_sha256,
)

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
    _validate_config(config)
    required = {
        "teacher": (TEACHER_MANIFEST, TEACHER_PAYLOAD),
        "student": (STUDENT_MANIFEST, STUDENT_PAYLOAD),
    }
    for role, (manifest_sha, payload_sha) in required.items():
        if (config[role]["manifest_sha256"], config[role]["payload_sha256"]) != (manifest_sha, payload_sha):
            raise ValueError(f"{role}: this protocol is only for the matched historical pair")
    if config["val_manifest"]["sha256"] != VAL_MANIFEST:
        raise ValueError("historical native VAL18 manifest must remain byte-identical")
    if config["parity"] != {"logit_atol": 1e-6, "logit_rtol": 1e-5}:
        raise ValueError("parity tolerances are fixed for this diagnostic")
    split_path = ROOT / "splits/acdc_reviewed_locked_train80_val18.json"
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
        inspect_file(folder / "weights.pt", spec["payload_sha256"], role + " weights")
        runtime = Path(spec["runtime_root"])
        if not runtime.is_dir():
            missing.append({"asset": role + " pinned runtime", "path": str(runtime.resolve())})
        elif _source_digest(runtime) != spec["runtime_source_sha256"]:
            raise ValueError(f"{role} pinned runtime source hash mismatch")
        else:
            checked.append({"asset": role + " runtime", "sha256": spec["runtime_source_sha256"]})
    if set(models) == {"teacher", "student"}:
        validate_pair(models["teacher"], models["student"])
        if models["teacher"].get("source_sha256") != config["teacher"]["runtime_source_sha256"]:
            raise ValueError("teacher runtime must match checkpoint source closure")
        lineage = models["student"]["lineage"]
        lids, uids = set(lineage["lids"]), set(lineage["uids"])
        if lids & uids or lids | uids != set(split["train"]) or len(lids) != 8:
            raise ValueError("compact L/U IDs must partition the locked TRAIN80")

    vm_path = Path(config["val_manifest"]["path"])
    frames = None
    if inspect_file(vm_path, VAL_MANIFEST, "VAL18 manifest"):
        vm = _load_json(vm_path)
        ids = [p["id"] for p in vm["patients"]]
        if vm.get("role") != "development" or vm.get("synthetic") is not False:
            raise ValueError("VAL18 must be real, exposed development data")
        if len(ids) != len(set(ids)) or set(ids) != set(split["val"]):
            raise ValueError("VAL18 patient IDs differ from the locked split")
        if set(ids) & set(split["train"]):
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
        "pair": "historical_raw_B8_and_wide_plain_KD", "patients": 18,
        "frames": frames, "checked": checked, "missing": missing,
        "limits": "Preflight is asset verification, not model loading or inference success.",
    }


def verify_inventory(config, role, receipt, manifest, output, expected_protocol):
    if receipt.get("role") != role or receipt.get("schema") != "self_audit.context_diagnostic.predictions.v1":
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
    with Path(path).open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def score_frozen(config, output, config_sha):
    """Validate every prediction, globally freeze, then (and only then) read GT."""
    import numpy as np
    import torch
    from self_audit.context_metrics import (
        analyze_frame, aggregate_records, frame_csv_rows, frame_metric_csv_rows,
        summary_csv_rows,
    )
    output = Path(output).resolve()
    torch.set_num_threads(config["threads"])
    protocol = _load_json(output / "PROTOCOL.json")
    if protocol != protocol_snapshot(config["device"], config["threads"]):
        raise ValueError("diagnostic code/environment changed since protocol was frozen")
    vm_path = Path(config["val_manifest"]["path"])
    if _sha256(vm_path) != config["val_manifest"]["sha256"]:
        raise ValueError("VAL manifest changed after inference")
    manifest = _load_json(vm_path)
    inventories, worker_receipts = {}, {}
    for role in ("teacher", "student"):
        receipt_path = output / role / "PREDICTIONS_FROZEN.json"
        receipt = _load_json(receipt_path)
        inventories[role] = verify_inventory(config, role, receipt, manifest, output / role, protocol)
        worker_receipts[role] = {"path": str(receipt_path), "sha256": _sha256(receipt_path)}
    freeze_path = output / "PREDICTIONS_FROZEN.json"
    if freeze_path.exists():
        raise ValueError("global freeze already exists; scoring is append-once")
    write_json(freeze_path, {
        "schema": "self_audit.context_diagnostic.global_freeze.v1",
        "config_sha256": config_sha, "worker_receipts": worker_receipts,
        "protocol_sha256": protocol_sha256(protocol),
        "ground_truth_opened_before_freeze": False,
        "selection": "all prespecified patients, frames and lambda values; no VAL selection",
    })
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
                for lam in (0., .5, 1.):
                    item = inventories[role][(patient["id"], int(t), lam)]
                    if _sha256(item["path"]) != item["sha256"]:
                        raise ValueError("frozen prediction changed before scoring")
                    pred = np.load(item["path"], allow_pickle=False)
                    if list(pred.shape) != item["shape_zyx"]:
                        raise ValueError("prediction shape differs from frozen receipt")
                    predictions[role][lam] = pred
            records.append(analyze_frame(patient["id"], int(t), gt, predictions["teacher"], predictions["student"]))
    summary = aggregate_records(records)
    actual = summary["patient_macro"]["all"]["predictions"]["student"]["1"]["class_first_macro4"]
    reproduced = actual is not None and abs(actual - HISTORICAL_WIDE_MACRO4) <= REPRODUCTION_ATOL
    result = {
        "schema": "self_audit.context_diagnostic.result.v1",
        "status": "DIAGNOSTIC_COMPLETE" if reproduced else "REPRODUCTION_MISMATCH_DO_NOT_INTERPRET",
        "config_sha256": config_sha, "global_freeze_sha256": _sha256(freeze_path),
        "protocol_sha256": protocol_sha256(protocol),
        "ground_truth_opened_after_freeze_for_scoring": True,
        "lambda1_wide_reproduction": {"expected": HISTORICAL_WIDE_MACRO4, "actual": actual,
                                      "atol": REPRODUCTION_ATOL, "passed": reproduced},
        "interpretation": "Descriptive exposed-VAL diagnostic. Transfer opportunity is not learned transfer or generalization.",
        "summary": summary,
    }
    write_json(output / "FRAME_RECORDS.json", records)
    write_json(output / "RESULT.json", result)
    _csv(output / "frame_transitions.csv", frame_csv_rows(records))
    _csv(output / "frame_metrics.csv", frame_metric_csv_rows(records))
    patient_rows = []
    for patient_id, patient in summary["patients"].items():
        for stratum, values in patient["strata"].items():
            for lam, comparison in values["comparisons"].items():
                for name, item in comparison["foreground"].items():
                    patient_rows.append({"patient_id": patient_id, "stratum": stratum,
                                         "lambda": float(lam), "class_name": name,
                                         **item["counts"], **item["rates"]})
    _csv(output / "patient_transitions.csv", patient_rows)
    _csv(output / "summary.csv", summary_csv_rows(summary))
    return result


def run(config_path, config_sha, output):
    import torch
    config = _read_config(Path(config_path), config_sha)
    report = preflight(config)
    if report["missing"]:
        raise FileNotFoundError(f"missing {len(report['missing'])} required assets; run --preflight-only for inventory")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output must be a new directory")
    protected = [Path(config[role]["path"]).resolve() for role in ("teacher", "student")]
    protected += [Path(config[role]["runtime_root"]).resolve() for role in ("teacher", "student")]
    protected += [Path(config["val_manifest"]["path"]).resolve().parent]
    vm = _load_json(Path(config["val_manifest"]["path"]))
    for patient in vm["patients"]:
        protected.append(_resolve_ref(Path(config["val_manifest"]["path"]).parent, patient["image"], "image").parent)
        protected.extend(_resolve_ref(Path(config["val_manifest"]["path"]).parent, m["path"], "mask").parent for m in patient["masks"].values())
    if any(_overlap(output, p) for p in protected):
        raise ValueError("output overlaps input assets")
    output.mkdir(parents=True)
    write_json(output / "PREFLIGHT.json", report)
    torch.set_num_threads(config["threads"])
    protocol = protocol_snapshot(config["device"], config["threads"])
    write_json(output / "PROTOCOL.json", protocol)
    print(json.dumps({"event": "START", "config": config, "config_sha256": config_sha,
                      "protocol_sha256": protocol_sha256(protocol), "protocol": protocol}), flush=True)
    for role in ("teacher", "student"):
        with (output / f"{role}.log").open("w") as log:
            subprocess.run([sys.executable, str(ROOT / "scripts/context_prediction_worker.py"),
                            "--config", str(Path(config_path).resolve()), "--config-sha256", config_sha,
                            "--protocol-sha256", protocol_sha256(protocol),
                            "--role", role, "--out", str(output / role)], check=True, stdout=log, stderr=subprocess.STDOUT)
    return score_frozen(config, output, config_sha)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--out", required=True, help="new JSON file for preflight; new directory for inference")
    args = parser.parse_args(argv)
    if args.preflight_only:
        if Path(args.out).exists():
            raise ValueError("preflight output already exists")
        report = preflight(_read_config(Path(args.config), args.config_sha256))
        write_json(args.out, report)
        print(json.dumps({"status": report["status"], "missing_assets": len(report["missing"])}))
        return 0 if not report["missing"] else 2
    result = run(args.config, args.config_sha256, args.out)
    print(json.dumps({"status": result["status"], "lambda1_wide_reproduction": result["lambda1_wide_reproduction"],
                      "patient_macro": result["summary"]["patient_macro"]}, allow_nan=False))
    return 0 if result["status"] == "DIAGNOSTIC_COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
