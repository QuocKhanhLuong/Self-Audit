"""Recovery gates, historical lineage verification, and baseline certification.

Locked contract:
1. Historical lineage and payload verification fails closed on missing files or hash mismatches.
2. Distinguishes failure classifications:
   - BLOCKED_MISSING_EVIDENCE: missing artifacts, weights, manifests, or unverified claims.
   - implementation_error: parameter ownership, architecture, or code bugs.
   - not_converged: optimization failed to reach preregistered threshold within budget.
   - empirical_failure: completed run failed locked baseline tolerance or class collapsed.
3. Baseline tolerance: Foreground Dice >= historic (0.831862) - 0.02 = 0.811862,
   with foreground noncollapse across RV, MYO, LV.
4. Certificate MUST be an evidence BUNDLE (paths + SHA) referencing actual physical files:
   checkpoint manifest + payload, TRAIN manifest / config, preregistered protocol,
   frozen predictions receipt + prediction arrays, evaluation metrics, optimizer witness,
   and tinyfit receipt. All physical files must exist and match hashes on disk.
5. Unified validator recomputes primary per-patient macro directly from per-patient evidence,
   requiring finite 0..1, exact class set, expected patient/frame coverage, and positive counts.
   Never authorizes handtyped scalars, empty dictionaries, or arbitrary 64-char hex strings.
6. Block shared training unless baseline certificate is valid and tied to source/weights/data.
   Shared 4+4 must start fresh from SSL DINO (exact weights SHA) or exact 4+4 checkpoint, NEVER ACDC L8.
"""

from __future__ import annotations

import copy
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import digest
from .data import write_json

# Historical reference pins from commit 0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca
HISTORICAL_COMMIT = "0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca"
HISTORICAL_SOURCE_SHA256 = "3c7fa85f00d54a6e087af3cd6c7045351fb72d532f1069bf09d95f9cd72359b7"
HISTORICAL_MANIFEST_SHA256 = "03c1730f88941ada898bed7fcf1b9f13cdd8f08bcce4390539fec665c82e97de"
HISTORICAL_PAYLOAD_SHA256 = "b36bcafe9613325d85ee70dabd6a36fef7c17429882967831a151c0cce6d8437"
HISTORICAL_DATA_SHA256 = "e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4"
HISTORICAL_DINO_WEIGHTS_SHA256 = "21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54"
HISTORICAL_DINO_SOURCE_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
HISTORICAL_PROTOCOL_SHA256 = "a7d4d844288e80041b4acc53790eda79a112b6301058eb1d98577c3b590d4a3c"
HISTORICAL_TEST_MANIFEST_SHA256 = "22a9ea63821f266bd59e810bcd8f9a6958e6f92cc363e4a360b5ab9d4eb1e25c"
HISTORICAL_METRICS_SHA256 = "c898c1c9649da0a365d95d57ff4a84229d83958a85a25b11ff83446b43151a40"
HISTORICAL_FROZEN_PREDICTIONS_RECEIPT_SHA256 = "ce6caa078e6737feece4999e53921f07a33ec553c6efa4cadce4dd01bda09792"

HISTORICAL_ACDC_LIDS = [
    "patient007", "patient010", "patient011", "patient024",
    "patient026", "patient041", "patient077", "patient099"
]

HISTORIC_FG_DICE = 0.831862063046926
HISTORIC_ALL_DICE = 0.8729522993309857
BASELINE_TOLERANCE_FG = 0.8118620630  # historic - 0.02
SHARED_VAL_DOMAIN_FG_THRESHOLD = 0.75
SHARED_VAL_MIN_CLASS_THRESHOLD = 0.60

CERTIFICATE_SCHEMA = "self_audit.baseline_gate_certificate.v1"
PREREGISTERED_PROTOCOL_SCHEMA = "self_audit.recovery_protocol.v1"


class RecoveryFailureType(str, Enum):
    BLOCKED_MISSING_EVIDENCE = "BLOCKED_MISSING_EVIDENCE"
    IMPLEMENTATION_ERROR = "implementation_error"
    NOT_CONVERGED = "not_converged"
    EMPIRICAL_FAILURE = "empirical_failure"
    PASSED = "PASSED"


class RecoveryGateError(RuntimeError):
    """Base error for recovery gate violations."""
    def __init__(self, message: str, failure_type: RecoveryFailureType = RecoveryFailureType.EMPIRICAL_FAILURE,
                 details: dict | None = None):
        super().__init__(message)
        self.failure_type = failure_type
        self.details = details or {}


class MissingEvidenceError(RecoveryGateError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, failure_type=RecoveryFailureType.BLOCKED_MISSING_EVIDENCE, details=details)


class ImplementationVerificationError(RecoveryGateError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, failure_type=RecoveryFailureType.IMPLEMENTATION_ERROR, details=details)


class ConvergenceError(RecoveryGateError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, failure_type=RecoveryFailureType.NOT_CONVERGED, details=details)


class EmpiricalGateFailureError(RecoveryGateError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, failure_type=RecoveryFailureType.EMPIRICAL_FAILURE, details=details)


def _validate_finite_score(val: Any, name: str) -> float:
    """Validate a score is float, finite, not NaN, and within [0.0, 1.0]."""
    if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
        raise EmpiricalGateFailureError(f"Score '{name}' is non-numeric: {val}")
    fval = float(val)
    if not math.isfinite(fval) or math.isnan(fval):
        raise EmpiricalGateFailureError(f"Score '{name}' is non-finite or NaN: {fval}")
    if fval < 0.0 or fval > 1.0:
        raise EmpiricalGateFailureError(f"Score '{name}' out of bounds [0, 1]: {fval}")
    return fval


def verify_historical_lineage(manifest_path: str | Path,
                              payload_path: str | Path | None = None,
                              *,
                              executed_source_digest: str | None = None,
                              require_payload_exists: bool = False) -> dict:
    """Verify historical baseline checkpoint manifest, payload hash, and source closure."""
    manifest_p = Path(manifest_path).resolve()
    if not manifest_p.is_file():
        raise MissingEvidenceError(
            f"Historical checkpoint manifest not found at {manifest_p}",
            details={"manifest_path": str(manifest_p)}
        )

    computed_manifest_sha = digest(manifest_p)
    if computed_manifest_sha != HISTORICAL_MANIFEST_SHA256:
        raise ImplementationVerificationError(
            f"Manifest hash mismatch: expected {HISTORICAL_MANIFEST_SHA256}, got {computed_manifest_sha}",
            details={"expected": HISTORICAL_MANIFEST_SHA256, "actual": computed_manifest_sha}
        )

    manifest_obj = json.loads(manifest_p.read_text())
    if manifest_obj.get("schema") != "self_audit.checkpoint.v2":
        raise ImplementationVerificationError(
            f"Invalid checkpoint schema: {manifest_obj.get('schema')}",
            details={"schema": manifest_obj.get("schema")}
        )

    # Verify labeled IDs match locked ACDC L8
    lids = manifest_obj.get("labeled_ids", [])
    if sorted(lids) != sorted(HISTORICAL_ACDC_LIDS):
        raise ImplementationVerificationError(
            f"Historical labeled IDs do not match locked ACDC L8: {lids} vs {HISTORICAL_ACDC_LIDS}",
            details={"actual_lids": lids, "expected_lids": HISTORICAL_ACDC_LIDS}
        )

    # Check payload hash declaration
    decl_payload_sha = manifest_obj.get("payload_sha256")
    if decl_payload_sha != HISTORICAL_PAYLOAD_SHA256:
        raise ImplementationVerificationError(
            f"Declared payload hash in manifest mismatch: expected {HISTORICAL_PAYLOAD_SHA256}, got {decl_payload_sha}",
            details={"expected": HISTORICAL_PAYLOAD_SHA256, "actual": decl_payload_sha}
        )

    # Check data hash declaration
    decl_data_sha = manifest_obj.get("data_sha256")
    if decl_data_sha != HISTORICAL_DATA_SHA256:
        raise ImplementationVerificationError(
            f"Declared data hash in manifest mismatch: expected {HISTORICAL_DATA_SHA256}, got {decl_data_sha}",
            details={"expected": HISTORICAL_DATA_SHA256, "actual": decl_data_sha}
        )

    # Check source hash declaration
    decl_source_sha = manifest_obj.get("source_sha256")
    if decl_source_sha != HISTORICAL_SOURCE_SHA256:
        raise ImplementationVerificationError(
            f"Declared source hash in manifest mismatch: expected {HISTORICAL_SOURCE_SHA256}, got {decl_source_sha}",
            details={"expected": HISTORICAL_SOURCE_SHA256, "actual": decl_source_sha}
        )

    # Verify physical payload if provided or required
    actual_payload_verified = False
    resolved_payload_p = None
    if payload_path is not None:
        resolved_payload_p = Path(payload_path).resolve()
    else:
        candidate = manifest_p.parent / "weights.pt"
        if candidate.is_file():
            resolved_payload_p = candidate

    if resolved_payload_p is not None:
        if not resolved_payload_p.is_file():
            if require_payload_exists:
                raise MissingEvidenceError(
                    f"Physical payload file missing: {resolved_payload_p}",
                    details={"payload_path": str(resolved_payload_p)}
                )
        else:
            computed_payload_sha = digest(resolved_payload_p)
            if computed_payload_sha != HISTORICAL_PAYLOAD_SHA256:
                raise ImplementationVerificationError(
                    f"Physical payload SHA-256 mismatch: expected {HISTORICAL_PAYLOAD_SHA256}, got {computed_payload_sha}",
                    details={"expected": HISTORICAL_PAYLOAD_SHA256, "actual": computed_payload_sha}
                )
            actual_payload_verified = True
    elif require_payload_exists:
        raise MissingEvidenceError(
            "Physical weights payload file is required but was not provided or found",
            details={"manifest_dir": str(manifest_p.parent)}
        )

    # Verify executed source if requested
    if executed_source_digest is not None:
        if executed_source_digest != HISTORICAL_SOURCE_SHA256:
            raise ImplementationVerificationError(
                f"Executed source digest {executed_source_digest} does not match historical {HISTORICAL_SOURCE_SHA256}. Isolated worktree required.",
                details={"executed": executed_source_digest, "historical": HISTORICAL_SOURCE_SHA256}
            )

    return {
        "verified": True,
        "manifest_path": str(manifest_p),
        "manifest_sha256": computed_manifest_sha,
        "payload_sha256": decl_payload_sha,
        "physical_payload_verified": actual_payload_verified,
        "source_sha256": decl_source_sha,
        "data_sha256": decl_data_sha,
        "labeled_ids": lids,
        "step": manifest_obj.get("step"),
        "recipe_config": manifest_obj.get("config", {}),
    }


def verify_baseline_protocol_config(config: Mapping[str, Any]) -> dict:
    """Verify that a configuration dictionary matches the canonical baseline recovery protocol."""
    cfg = copy.deepcopy(dict(config))
    errors = []

    if cfg.get("schema") != "self_audit.semi_v2":
        errors.append(f"schema must be 'self_audit.semi_v2', got '{cfg.get('schema')}'")

    if cfg.get("method") != "ema_full":
        errors.append(f"method must be 'ema_full', got '{cfg.get('method')}'")

    if cfg.get("update_scope") not in {"head", "decoder"}:
        errors.append(f"update_scope must be 'head' or 'decoder', got '{cfg.get('update_scope')}'")

    window = cfg.get("window", {})
    if not isinstance(window, dict) or window.get("mode") != "none":
        errors.append(f"window mode must be 'none', got '{window.get('mode') if isinstance(window, dict) else None}'")

    if cfg.get("steps") != 10000:
        errors.append(f"steps must be 10000, got {cfg.get('steps')}")

    if cfg.get("warmup_steps") != 1000:
        errors.append(f"warmup_steps must be 1000, got {cfg.get('warmup_steps')}")

    if cfg.get("seed") != 42:
        errors.append(f"seed must be 42, got {cfg.get('seed')}")

    if cfg.get("body_lr") != 3e-5:
        errors.append(f"body_lr must be 3e-5, got {cfg.get('body_lr')}")

    if cfg.get("head_lr") != 0.003:
        errors.append(f"head_lr must be 0.003, got {cfg.get('head_lr')}")

    if cfg.get("pseudo_lr") != 0.001:
        errors.append(f"pseudo_lr must be 0.001, got {cfg.get('pseudo_lr')}")

    if cfg.get("weight_decay") != 1e-4:
        errors.append(f"weight_decay must be 0.0001 (1e-4), got {cfg.get('weight_decay')}")

    if cfg.get("ema_decay") != 0.99:
        errors.append(f"ema_decay must be 0.99, got {cfg.get('ema_decay')}")

    if cfg.get("pseudo_threshold") != 0.7:
        errors.append(f"pseudo_threshold must be 0.7, got {cfg.get('pseudo_threshold')}")

    batch_size = cfg.get("batch_size")
    if batch_size != 8:
        errors.append(f"batch_size must be 8, got {batch_size}")

    crop_size = cfg.get("crop_size")
    if crop_size != 256:
        errors.append(f"crop_size must be 256, got {crop_size}")

    tm = cfg.get("teacher_memory")
    if tm is not None:
        if isinstance(tm, dict):
            if tm.get("arm") != "A0" or tm.get("bank_path") is not None or tm.get("cache_path") is not None:
                errors.append(f"baseline cannot use teacher_memory: got arm {tm.get('arm')}")

    model = cfg.get("model", {})
    if not isinstance(model, dict):
        errors.append("model must be a dictionary")
    else:
        if model.get("backbone") != "dinov3_convnext_tiny":
            errors.append(f"model backbone must be 'dinov3_convnext_tiny', got '{model.get('backbone')}'")
        if model.get("weights_sha256") != HISTORICAL_DINO_WEIGHTS_SHA256:
            errors.append(f"model weights_sha256 must match DINO {HISTORICAL_DINO_WEIGHTS_SHA256}")

    if errors:
        raise ImplementationVerificationError(
            f"Configuration does not match canonical baseline protocol: {'; '.join(errors)}",
            details={"errors": errors, "config": cfg}
        )

    return cfg


def _same(actual, expected, name):
    if actual != expected:
        raise ImplementationVerificationError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _file_entry(path, name):
    if path is None or not Path(path).is_file():
        raise MissingEvidenceError(f"Missing physical {name}: {path}")
    p = Path(path).resolve()
    return {"path": str(p), "sha256": digest(p)}


def _json(entry):
    return json.loads(Path(entry["path"]).read_text())


def evaluate_baseline_gate(metrics, *, tolerance=BASELINE_TOLERANCE_FG,
                           expected_patient_count=None, expected_frame_count=None,
                           synthetic=False):
    """Recompute volume -> frame -> patient metrics from integer confusion evidence."""
    from .evaluation import class_metrics
    tolerance = _validate_finite_score(tolerance, "tolerance")
    if not isinstance(metrics, Mapping):
        raise MissingEvidenceError("A metrics object with per-frame confusion evidence is required")
    target = metrics.get("domains", {}).get("acdc", metrics)
    frames = target.get("frames")
    if not isinstance(frames, list) or not frames:
        raise MissingEvidenceError("Missing per-frame confusion evidence; macro scalars cannot authorize training")
    if not synthetic:
        if expected_patient_count not in (None, 50) or expected_frame_count not in (None, 100):
            raise ImplementationVerificationError("Real baseline coverage is locked to 50 patients / 100 frames")
        expected_patient_count, expected_frame_count = 50, 100
    grouped, keys, counts = {}, set(), np.zeros(4, dtype=np.int64)
    for row in frames:
        pid, frame = row.get("patient"), row.get("frame")
        if not isinstance(pid, str) or not pid or type(frame) is not int or frame < 0:
            raise ImplementationVerificationError("Invalid patient/frame identity")
        if (pid, frame) in keys:
            raise ImplementationVerificationError("Duplicate patient/frame evidence")
        keys.add((pid, frame))
        cm = np.asarray(row.get("confusion", row.get("raw_confusion")))
        if cm.shape != (4, 5) or cm.dtype.kind not in "iu" or (cm < 0).any():
            raise ImplementationVerificationError("Expected nonnegative integer 4x5 confusion matrix")
        if cm.sum() <= 0 or cm[:, 4].sum() != 0:
            raise EmpiricalGateFailureError("Empty support or incomplete raw prediction coverage")
        computed = class_metrics(cm)
        raw = row.get("raw")
        if not isinstance(raw, list) or len(raw) != 4:
            raise MissingEvidenceError("Missing per-class frame metrics")
        for i, name in enumerate(("BG", "RV", "MYO", "LV")):
            _same(raw[i].get("class"), name, "frame class order")
            for field in ("TP", "FP", "FN", "prediction_count", "reference_count", "dice"):
                _same(raw[i].get(field), computed[i][field], f"frame {pid}/{frame} {name}/{field}")
        counts += cm[:, :4].sum(0)
        grouped.setdefault(pid, []).append(computed)
    if expected_patient_count is not None:
        _same(len(grouped), expected_patient_count, "patient coverage")
    if expected_frame_count is not None:
        _same(len(keys), expected_frame_count, "frame coverage")
    if not synthetic:
        _same(sorted(grouped), [f"patient{i:03d}" for i in range(101, 151)], "locked TEST50 cohort")
    patients, patient_fg = {}, []
    for pid, rows in grouped.items():
        values = {}
        for i, name in enumerate(("BG", "RV", "MYO", "LV")):
            ds = [r[i]["dice"] for r in rows if r[i]["dice"] is not None]
            values[name] = float(np.mean(ds)) if ds else None
        fg = [values[c] for c in ("RV", "MYO", "LV") if values[c] is not None]
        if not fg:
            raise EmpiricalGateFailureError(f"Undefined foreground for patient {pid}")
        patient_fg.append(float(np.mean(fg)))
        patients[pid] = values
    declared = target.get("patient_metrics", target.get("patients"))
    if not isinstance(declared, dict) or set(declared) != set(patients):
        raise MissingEvidenceError("Patient metrics and frame identities must agree exactly")
    for pid, classes in patients.items():
        for name, val in classes.items():
            _same(declared[pid].get(name), val, f"patient {pid}/{name}")
    macro = {}
    for name in ("RV", "MYO", "LV"):
        ds = [p[name] for p in patients.values() if p[name] is not None]
        macro[name] = float(np.mean(ds)) if ds else None
        if macro[name] is None or macro[name] <= 0:
            raise EmpiricalGateFailureError(f"Collapsed foreground class {name}")
    if not (counts[1:] > 0).all():
        raise EmpiricalGateFailureError("Every foreground class must have predictions")
    fg = float(np.mean(patient_fg))
    if target.get("foreground_patient_macro") is not None:
        _same(target["foreground_patient_macro"], fg, "patient-macro foreground")
    if fg < tolerance:
        raise EmpiricalGateFailureError(f"Foreground Dice {fg:.6f} below locked {tolerance:.6f}")
    return {"status": "PASSED", "foreground_patient_macro": fg, "class_dices": macro,
            "patient_count": len(patients), "frame_count": len(keys), "tolerance": tolerance,
            "foreground_prediction_counts": dict(zip(("RV", "MYO", "LV"), map(int, counts[1:]))),
            "historic_foreground_dice": HISTORIC_FG_DICE, "delta_from_historic": fg-HISTORIC_FG_DICE,
            "all_foreground_classes_active": True}


def _validate_evidence_bundle(*, checkpoint_manifest_path, checkpoint_payload_path=None,
                              train_manifest_path, train_config_path, protocol_path,
                              frozen_predictions_path, metrics_path, tinyfit_receipt_path,
                              evaluation_manifest_path=None, optimizer_witness=None,
                              synthetic=False, expected_patient_count=None,
                              expected_frame_count=None, tolerance=BASELINE_TOLERANCE_FG,
                              rescore_references=True):
    """Verify bytes and lineage, then rescore frozen predictions against pinned references.

    This is an integrity/evidence check, not a signature or a claim of clinical efficacy.
    Reference arrays are read only here, after the entire prediction inventory is verified.
    """
    from .evaluation import confusion_matrix, class_metrics
    paths = dict(checkpoint_manifest=checkpoint_manifest_path,
                 checkpoint_payload=checkpoint_payload_path or Path(checkpoint_manifest_path).parent/"weights.pt",
                 train_manifest=train_manifest_path, train_config=train_config_path, protocol=protocol_path,
                 frozen_predictions=frozen_predictions_path, metrics=metrics_path,
                 tinyfit_receipt=tinyfit_receipt_path, evaluation_manifest=evaluation_manifest_path,
                 evaluation_recipe=Path(frozen_predictions_path).parent/"RECIPE.json")
    bundle = {k: _file_entry(v, k) for k, v in paths.items()}
    ck, tr, cfg, proto, frozen, metrics, tf, ev, recipe = [
        _json(bundle[k]) for k in ("checkpoint_manifest", "train_manifest", "train_config", "protocol",
                                  "frozen_predictions", "metrics", "tinyfit_receipt", "evaluation_manifest", "evaluation_recipe")]
    _same(ck.get("schema"), "self_audit.checkpoint.v2", "checkpoint schema")
    for obj, name in ((ck,"checkpoint"),(tr,"TRAIN"),(tf,"tinyfit"),(ev,"evaluation"),(recipe,"recipe"),(metrics,"metrics")):
        _same(obj.get("synthetic", False), synthetic, name+" synthetic identity")
    _same(ck.get("payload_sha256"), bundle["checkpoint_payload"]["sha256"], "checkpoint payload")
    _same(ck.get("data_sha256"), bundle["train_manifest"]["sha256"], "TRAIN digest")
    _same(ck.get("config"), cfg, "actual training config")
    lids = sorted(r["id"] for r in tr.get("labeled", []))
    uids = sorted(r["id"] for r in tr.get("unlabeled", []))
    _same(sorted(ck.get("labeled_ids", [])), lids, "checkpoint label lineage")
    _same(sorted(ck.get("unlabeled_ids", [])), uids, "checkpoint U lineage")
    if not lids or not uids or set(lids)&set(uids):
        raise ImplementationVerificationError("Invalid TRAIN lineage")
    historical = bundle["checkpoint_manifest"]["sha256"] == HISTORICAL_MANIFEST_SHA256
    if not synthetic:
        verify_baseline_protocol_config(cfg)
        _same(lids, sorted(HISTORICAL_ACDC_LIDS), "locked ACDC L8")
        _same(len(uids), 72, "locked ACDC U72")
        _same(bundle["train_manifest"]["sha256"], HISTORICAL_DATA_SHA256, "historical TRAIN split")
        _same(bundle["evaluation_manifest"]["sha256"], HISTORICAL_TEST_MANIFEST_SHA256, "locked TEST50")
        _same(ck.get("step"), 10000, "selected final step")
        _same(cfg["model"].get("weights_sha256"), HISTORICAL_DINO_WEIGHTS_SHA256, "DINO weights")
        _same(cfg["model"].get("source_commit"), HISTORICAL_DINO_SOURCE_COMMIT, "DINO source")
        _same(tolerance, BASELINE_TOLERANCE_FG, "locked tolerance")
    if historical:
        _same(bundle["checkpoint_payload"]["sha256"], HISTORICAL_PAYLOAD_SHA256, "historical payload")
        _same(ck.get("source_sha256"), HISTORICAL_SOURCE_SHA256, "historical executed source")
        _same(bundle["protocol"]["sha256"], HISTORICAL_PROTOCOL_SHA256, "historical protocol")
    else:
        _same(proto.get("schema"), PREREGISTERED_PROTOCOL_SCHEMA, "protocol schema")
        _same(proto.get("protocol_type"), "baseline", "protocol role")
        _same(proto.get("executed_source_sha256"), ck.get("source_sha256"), "preregistered source")
        _same(proto.get("config"), cfg, "preregistered config")
        _same(proto.get("train_manifest_sha256"), bundle["train_manifest"]["sha256"], "preregistered TRAIN")
        _same(proto.get("evaluation_manifest_sha256"), bundle["evaluation_manifest"]["sha256"], "preregistered evaluation")
        _same(proto.get("target_steps"), ck.get("step"), "preregistered final step")
        _same(proto.get("warmup_steps"), cfg.get("warmup_steps"), "preregistered warmup")
        _same(proto.get("tolerance"), BASELINE_TOLERANCE_FG, "preregistered tolerance")
    if not isinstance(ck.get("source_sha256"), str) or len(ck["source_sha256"]) != 64:
        raise MissingEvidenceError("Missing executed source digest")
    # Fresh inference may be produced by the original evaluator in the pinned historical checkout.
    _same(recipe.get("checkpoint_sha256"), bundle["checkpoint_manifest"]["sha256"], "inference checkpoint")
    _same(recipe.get("threshold"), None, "raw student evaluation")
    declared_eval = recipe.get("domains", {}).get("acdc", {}).get("manifest_sha256", recipe.get("data_sha256"))
    _same(declared_eval, bundle["evaluation_manifest"]["sha256"], "inference reference manifest")
    _same(frozen.get("recipe_sha256"), bundle["evaluation_recipe"]["sha256"], "frozen recipe")
    modern = "domains" in frozen
    if modern:
        _same(set(frozen["domains"]), {"acdc"}, "baseline domains")
        for obj, name in ((frozen,"frozen"),(metrics,"metrics")):
            _same(obj.get("checkpoint_manifest_sha256"), bundle["checkpoint_manifest"]["sha256"], name+" checkpoint")
            _same(obj.get("checkpoint_payload_sha256"), bundle["checkpoint_payload"]["sha256"], name+" payload")
        _same(metrics.get("recipe_sha256"), bundle["evaluation_recipe"]["sha256"], "metrics recipe")
        _same(metrics.get("frozen_predictions_sha256"), bundle["frozen_predictions"]["sha256"], "metrics frozen inventory")
        inventory = frozen["domains"]["acdc"]
    else:
        if not historical and not synthetic:
            raise ImplementationVerificationError("Legacy evaluation only allowed for pinned historical checkpoint")
        inventory = frozen.get("files")
    patients = ev.get("patients")
    if not isinstance(patients, list) or not patients or not isinstance(inventory, list):
        raise MissingEvidenceError("Missing evaluation patient/prediction inventory")
    ids = [r["id"] for r in patients]
    if len(set(ids)) != len(ids) or set(ids)&set(lids+uids):
        raise ImplementationVerificationError("Evaluation duplicate IDs or TRAIN overlap")
    _same(sorted(p["id"] for p in inventory), sorted(ids), "frozen prediction coverage")
    predroot = Path(frozen_predictions_path).resolve().parent/"predictions"
    predmap = {}
    for p in inventory:
        pp = (predroot/p["file"]).resolve()
        if not pp.is_relative_to(predroot.resolve()):
            raise ImplementationVerificationError("Prediction path outside frozen package")
        entry = _file_entry(pp, "prediction")
        _same(entry["sha256"], p["sha256"], "prediction bytes")
        predmap[p["id"]] = pp
    # Only after the whole frozen inventory has been verified, rescore every reference frame.
    rows = metrics.get("domains", {}).get("acdc", metrics).get("frames", [])
    rowmap = {(r.get("patient"),r.get("frame")):r for r in rows}
    if len(rowmap) != len(rows):
        raise ImplementationVerificationError("Duplicate metric frame")
    expected_keys = set()
    evroot = Path(evaluation_manifest_path).resolve().parent
    for p in patients:
        pred = np.load(predmap[p["id"]], mmap_mode="r", allow_pickle=False)
        if modern:
            if pred.ndim != 4 or pred.dtype != np.uint8 or not np.isin(pred,[0,1,2,3]).all():
                raise ImplementationVerificationError("Expected raw uint8 TZYX prediction")
        else:
            if pred.ndim != 5 or pred.shape[2] != 4 or pred.dtype.kind != "f" or not np.isfinite(pred).all():
                raise ImplementationVerificationError("Expected legacy finite TZCYX probabilities")
        masks = p.get("masks")
        if not isinstance(masks, dict) or not masks:
            raise MissingEvidenceError("Empty evaluation reference frames")
        for frame_text, record in masks.items():
            frame = int(frame_text)
            key = p["id"], frame
            if frame < 0 or frame >= pred.shape[0] or key in expected_keys:
                raise ImplementationVerificationError("Invalid/duplicate reference frame")
            expected_keys.add(key)
            if not rescore_references:
                # Startup verifies the pinned release receipt, never opens query/reference masks.
                continue
            refpath = Path(record["path"])
            if not refpath.is_absolute(): refpath = evroot/refpath
            ref = _file_entry(refpath,"reference")
            _same(ref["sha256"], record["sha256"], "reference bytes")
            y = np.load(refpath, allow_pickle=False)
            if y.dtype.kind not in "iu":
                raise ImplementationVerificationError("Reference must contain integer labels")
            cm = confusion_matrix(y, pred[frame] if modern else pred[frame].argmax(1))
            row = rowmap.get(key, {})
            _same(row.get("confusion",row.get("raw_confusion")), cm.tolist(), "rescored frame confusion")
    _same(set(rowmap), expected_keys, "scored frame coverage")
    # The separate tiny-fit diagnostic must use only this TRAIN budget and real optimization evidence.
    _same(tf.get("train_manifest_sha256"), bundle["train_manifest"]["sha256"], "tiny-fit TRAIN")
    from .engine import source_digest
    expected_tinyfit_source = source_digest() if historical else proto.get("tinyfit_source_sha256")
    _same(tf.get("source_sha256"), expected_tinyfit_source, "tiny-fit executed source")
    if tf.get("status") != "CONVERGED" or tf.get("converged") is not True:
        raise ConvergenceError("Tiny-fit did not converge within its locked budget")
    if not synthetic:
        _same(tf.get("probe_patients"), ["patient007","patient010"], "tiny-fit patients")
        _same(tf.get("target_fg_dice"), .95, "tiny-fit target")
        _same(tf.get("max_steps"), 200, "tiny-fit budget")
        _same(tf.get("model_config"), cfg["model"], "tiny-fit initialization")
    if _validate_finite_score(tf.get("final_fg_dice"), "tiny-fit foreground") < .95:
        raise ConvergenceError("Tiny-fit foreground below .95")
    if type(tf.get("steps_taken")) is not int or not 1 <= tf["steps_taken"] <= 200:
        raise ImplementationVerificationError("Invalid tiny-fit update count")
    if set(tf.get("probe_patients", []))-set(lids):
        raise ImplementationVerificationError("Tiny-fit labels outside TRAIN budget")
    for field in ("step_0_gradient_norms", "step_0_update_deltas", "module_update_deltas"):
        for mod in ("encoder","decoder","detail","fuse","head"):
            val = tf.get(field, {}).get(mod)
            if type(val) not in (int,float) or not math.isfinite(val) or val <= 0:
                raise ImplementationVerificationError(f"Invalid optimizer witness {field}/{mod}")
    if tf.get("optimizer_ownership", {}).get("verified") is not True:
        raise ImplementationVerificationError("Missing actual optimizer ownership witness")
    for c in ("RV","MYO","LV"):
        if _validate_finite_score(tf.get("final_class_dices",{}).get(c), "tiny-fit "+c) <= 0:
            raise ConvergenceError("Tiny-fit collapsed foreground")
    result = evaluate_baseline_gate(metrics, tolerance=tolerance, synthetic=synthetic,
                                   expected_patient_count=expected_patient_count,
                                   expected_frame_count=expected_frame_count)
    return bundle, result


def build_baseline_gate_certificate(*, out_path=None, **kwargs):
    bundle, result = _validate_evidence_bundle(**kwargs)
    cert = {"schema": CERTIFICATE_SCHEMA, "status":"PASSED", "synthetic":kwargs.get("synthetic",False),
            "evidence_bundle":bundle, "recomputed_evaluation":result,
            "verification_scope":"Physical evidence integrity and frozen prediction rescoring; exposed TEST50 development benchmark"}
    cert["integrity_receipt_sha256"] = hashlib.sha256(json.dumps(cert,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if out_path is not None:
        if Path(out_path).exists(): raise ValueError("Gate output must be new")
        write_json(out_path,cert)
    return cert


def verify_baseline_gate_certificate(cert_input, *, require_physical_files=True, read_reference_arrays=True):
    if not require_physical_files:
        raise MissingEvidenceError("Physical verification cannot be disabled")
    cert = json.loads(Path(cert_input).read_text()) if isinstance(cert_input,(str,Path)) else copy.deepcopy(dict(cert_input))
    _same(cert.get("schema"), CERTIFICATE_SCHEMA, "gate schema")
    _same(cert.get("status"), "PASSED", "gate status")
    expected = cert.pop("integrity_receipt_sha256",None)
    actual = hashlib.sha256(json.dumps(cert,sort_keys=True,allow_nan=False).encode()).hexdigest()
    _same(expected,actual,"gate integrity")
    cert["integrity_receipt_sha256"] = expected
    bundle = cert.get("evidence_bundle",{})
    names = ("checkpoint_manifest","checkpoint_payload","train_manifest","train_config","protocol",
             "frozen_predictions","metrics","tinyfit_receipt","evaluation_manifest")
    args = {name+"_path":bundle.get(name,{}).get("path") for name in names}
    args["synthetic"] = cert.get("synthetic",False)
    args["rescore_references"] = read_reference_arrays
    current, result = _validate_evidence_bundle(**args)
    _same(current,bundle,"physical gate bundle")
    _same(result,cert.get("recomputed_evaluation"),"recomputed gate result")
    return cert


def block_shared_training_unless_baseline_gated(baseline_cert_input, *, expected_cert_sha=None,
                                               shared_manifest_path=None, shared_config=None,
                                               require_physical_files=True):
    if not isinstance(baseline_cert_input,(str,Path)) or not expected_cert_sha:
        raise MissingEvidenceError("Real shared training requires physical baseline certificate and expected_cert_sha")
    _same(_file_entry(baseline_cert_input,"baseline gate")["sha256"],expected_cert_sha,"baseline gate pin")
    cert = verify_baseline_gate_certificate(baseline_cert_input,require_physical_files=require_physical_files,
                                            read_reference_arrays=False)
    if cert.get("synthetic") is not False:
        raise ImplementationVerificationError("Synthetic certificate cannot authorize real shared training")
    if shared_manifest_path is None or shared_config is None:
        raise MissingEvidenceError("Shared manifest and resolved config required")
    from .shared_data import validate_shared_manifest
    m = validate_shared_manifest(shared_manifest_path)
    _same(m.get("synthetic"),False,"shared patient identity")
    _same(shared_config.get("model",{}).get("weights_sha256"),HISTORICAL_DINO_WEIGHTS_SHA256,"fresh SSL initialization")
    _same(shared_config.get("model",{}).get("source_commit"),HISTORICAL_DINO_SOURCE_COMMIT,"DINO source")
    verify_baseline_protocol_config(shared_config)
    return cert



def run_tiny_fit_diagnostic(manifest_path: str | Path,
                            manifest_sha256: str,
                            out_dir: str | Path,
                            *,
                            model_config_path: str | Path | None = None,
                            model_config: Mapping[str, Any] | None = None,
                            device: str = "cpu",
                            max_steps: int = 200,
                            target_fg_dice: float = 0.95,
                            diagnostic_lr_factor: float = 1.0,
                            seed: int = 42,
                            synthetic: bool = False) -> dict:
    """Run tiny-fit diagnostic on fixed 2-patient LTRAIN subset to prove optimizer efficacy.

    Enforces:
    1. Heterogeneous patient slice padding with 0.0 for image, 255 for label.
    2. Dice metric calculation strictly excludes 255 (predictions on 255 do NOT count as false positives).
    3. RNG states captured and restored; manual seed set prior to model build.
    4. max_steps validated in 1..200; target_fg_dice locked to 0.95 for patient runs.
    5. Step 0 update deltas required across all canonical modules.
    6. Saves full probe details, step history, and optimizer ownership to TINY_FIT_REPORT.json.
    """
    from .model import build_model
    from .data import TrainingData
    from .engine import supervised_step, source_digest
    from .baseline_recovery import (
        verify_optimizer_ownership,
        snapshot_parameters,
        compute_module_telemetry,
        CANONICAL_MODULES,
    )

    if not (type(max_steps) is int and 1 <= max_steps <= 200):
        raise ImplementationVerificationError(f"tiny-fit max_steps must be an integer between 1 and 200, got {max_steps}")

    _validate_finite_score(target_fg_dice, "tiny-fit target")
    if not synthetic:
        if seed != 42:
            raise ImplementationVerificationError("Real tiny-fit seed is locked to 42")
        if target_fg_dice != 0.95:
            raise ImplementationVerificationError(f"Real patient tiny-fit target_fg_dice must be exactly 0.95, got {target_fg_dice}")
        if abs(diagnostic_lr_factor - 1.0) > 1e-6:
            raise ImplementationVerificationError(
                f"Real patient tiny-fit diagnostic_lr_factor must be 1.0 (no free LR factor for real protocol), got {diagnostic_lr_factor}"
            )

    out_p = Path(out_dir).resolve()
    if out_p.exists():
        raise ImplementationVerificationError(f"tiny-fit output directory already exists: {out_p}")
    out_p.mkdir(parents=True)

    # Save and set RNG
    orig_torch_rng = torch.get_rng_state()
    orig_cuda_rng = torch.cuda.get_rng_state_all() if (torch.cuda.is_available() and device.startswith("cuda")) else None
    torch.manual_seed(seed)
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    try:
        data = TrainingData(manifest_path, manifest_sha256, synthetic=synthetic)

        # Determine 2 diagnostic patients: STRICTLY patient007 and patient010, NO fallback
        if not synthetic:
            required_pair = ["patient007", "patient010"]
            avail = set(data.lids)
            if not set(required_pair).issubset(avail):
                raise ImplementationVerificationError(
                    f"tiny-fit requires exact fixed patient pair {required_pair}; available labeled IDs: {sorted(avail)}. No fallback allowed.",
                    details={"required_pair": required_pair, "available": sorted(avail)}
                )
            probe_pids = required_pair
        else:
            probe_pids = data.lids[:2]

        probe_indices = [i for i, r in enumerate(data.labeled) if r["id"] in probe_pids]
        if len(probe_indices) < 2:
            raise ImplementationVerificationError("tiny-fit requires at least 2 distinct labeled patients")

        # Resolve model config with real pinned asset paths
        if model_config_path is not None:
            mcfg_p = Path(model_config_path).resolve()
            if not mcfg_p.is_file():
                raise MissingEvidenceError(f"model_config file not found: {mcfg_p}")
            eff_model_cfg = json.loads(mcfg_p.read_text())
        elif model_config is not None:
            eff_model_cfg = copy.deepcopy(dict(model_config))
        elif synthetic:
            eff_model_cfg = {"backbone": "tiny_test", "source_repo": "", "source_commit": "", "weights": "", "weights_sha256": ""}
        else:
            raise MissingEvidenceError("Real patient tiny-fit requires explicit model_config with pinned DINO asset paths")

        if not synthetic:
            if eff_model_cfg.get("backbone") != "dinov3_convnext_tiny":
                raise ImplementationVerificationError(f"tiny-fit backbone must be dinov3_convnext_tiny, got {eff_model_cfg.get('backbone')}")
            if eff_model_cfg.get("weights_sha256") != HISTORICAL_DINO_WEIGHTS_SHA256:
                raise ImplementationVerificationError(f"tiny-fit weights_sha256 must match DINO {HISTORICAL_DINO_WEIGHTS_SHA256}")

        model, provenance = build_model(eff_model_cfg, synthetic=synthetic, window={"mode": "none"})
        model = model.to(device)

        # Canonical baseline optimizer split using declared baseline LRs
        base_body_lr = 3e-5 * diagnostic_lr_factor
        base_head_lr = 0.003 * diagnostic_lr_factor
        body_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
        head_params = list(model.head.parameters())
        opt_body = torch.optim.AdamW(body_params, lr=base_body_lr, weight_decay=1e-4)
        opt_head = torch.optim.SGD(head_params, lr=base_head_lr, momentum=0.0)

        # Verify initial parameter ownership
        ownership = verify_optimizer_ownership(model, [opt_body, opt_head])

        rng = np.random.default_rng(seed)
        probe_slices = []
        probe_details = []
        for p_idx in probe_indices:
            x, y, meta = data.sample_labeled(p_idx, rng, foreground_probability=1.0)
            probe_slices.append((x, y))
            probe_details.append({
                "patient_id": data.labeled[p_idx]["id"],
                "image_shape": list(x.shape),
                "voxel_sha256": hashlib.sha256(x.numpy().tobytes() if hasattr(x, "numpy") else bytes(x)).hexdigest(),
            })

        # Heterogeneous spatial padding: determine maximum H and W across probe slices
        max_h = max(b[0].shape[-2] for b in probe_slices)
        max_w = max(b[0].shape[-1] for b in probe_slices)

        padded_xs = []
        padded_ys = []
        for x, y in probe_slices:
            h, w = x.shape[-2], x.shape[-1]
            pad_bottom = max_h - h
            pad_right = max_w - w
            if pad_bottom > 0 or pad_right > 0:
                x_pad = F.pad(x, (0, pad_right, 0, pad_bottom), mode="constant", value=0.0)
                y_pad = F.pad(y, (0, pad_right, 0, pad_bottom), mode="constant", value=255)
            else:
                x_pad = x
                y_pad = y
            padded_xs.append(x_pad)
            padded_ys.append(y_pad)

        x_probe = torch.cat(padded_xs, dim=0).to(device)
        y_probe = torch.cat(padded_ys, dim=0).to(device)

        initial_snapshot = snapshot_parameters(model)
        step_history = []
        final_dice = 0.0
        final_class_dices = {}
        converged = False
        step0_telemetry = None
        step0_deltas = None

        model.train()
        for step in range(max_steps):
            # Step 0 gradient witness check
            if step == 0:
                loss, norm, telem0 = supervised_step(model, [opt_body, opt_head], x_probe, y_probe, return_telemetry=True)
                step0_telemetry = telem0
                for mod in CANONICAL_MODULES:
                    gnorm = telem0.get("gradient_norms", {}).get(mod, 0.0)
                    if not math.isfinite(gnorm) or gnorm <= 0:
                        raise ImplementationVerificationError(f"tiny-fit step 0 canonical module '{mod}' gradient is nonfinite or zero: {gnorm}")

                # First-step deltas check
                step0_deltas = compute_module_telemetry(model, initial_snapshot)
                for mod in CANONICAL_MODULES:
                    dval = step0_deltas.get("update_deltas", {}).get(mod, 0.0)
                    if not math.isfinite(dval) or dval <= 0:
                        raise ImplementationVerificationError(f"tiny-fit step 0 canonical module '{mod}' weight delta is nonfinite or zero: {dval}")
            else:
                loss, norm = supervised_step(model, [opt_body, opt_head], x_probe, y_probe)

            if not math.isfinite(loss):
                raise ImplementationVerificationError(f"tiny-fit nonfinite loss at step {step}: {loss}")

            # Evaluate current probe Dice: valid mask excludes 255
            model.eval()
            with torch.no_grad():
                logits = model(x_probe)
                preds = logits.argmax(dim=1)
                valid = (y_probe != 255)
                dices = {}
                for c_idx, c_name in [(1, "RV"), (2, "MYO"), (3, "LV")]:
                    gt = (y_probe == c_idx)
                    pr = (preds == c_idx) & valid
                    inter = (gt & pr).sum().item()
                    denom = gt.sum().item() + pr.sum().item()
                    dices[c_name] = (2.0 * inter / denom) if denom > 0 else 0.0

                cur_fg_dice = float(np.mean(list(dices.values())))
                step_history.append({"step": step, "loss": float(loss), "fg_dice": cur_fg_dice})

                if cur_fg_dice >= target_fg_dice and all(d > 0.0 for d in dices.values()):
                    converged = True
                    final_dice = cur_fg_dice
                    final_class_dices = dices
                    break
            model.train()

        if not converged:
            final_dice = step_history[-1]["fg_dice"]
            final_class_dices = dices

        # Verify final weight update deltas across all modules
        final_telemetry = compute_module_telemetry(model, initial_snapshot)
        for mod in CANONICAL_MODULES:
            delta_val = final_telemetry.get("update_deltas", {}).get(mod, 0.0)
            if not math.isfinite(delta_val) or delta_val <= 0:
                raise ImplementationVerificationError(f"tiny-fit canonical module '{mod}' final weight delta is nonfinite or zero: {delta_val}")

        result = {
            "status": "CONVERGED" if converged else "NOT_CONVERGED",
            "failure_classification": RecoveryFailureType.PASSED.value if converged else RecoveryFailureType.NOT_CONVERGED.value,
            "converged": converged,
            "synthetic": synthetic,
            "target_fg_dice": target_fg_dice,
            "final_fg_dice": final_dice,
            "final_class_dices": final_class_dices,
            "steps_taken": len(step_history),
            "max_steps": max_steps,
            "probe_patients": probe_pids,
            "probe_details": probe_details,
            "train_manifest_sha256": manifest_sha256,
            "model_config": eff_model_cfg,
            "provenance": provenance,
            "source_sha256": source_digest(),
            "optimizer_ownership": ownership,
            "step_0_gradient_norms": step0_telemetry.get("gradient_norms", {}) if step0_telemetry else {},
            "step_0_update_deltas": step0_deltas.get("update_deltas", {}) if step0_deltas else {},
            "all_modules_updated": final_telemetry.get("all_deltas_nonzero", False),
            "module_update_deltas": final_telemetry.get("update_deltas", {}),
            "step_history": step_history,
        }

        write_json(out_p / "TINY_FIT_REPORT.json", result)

        if not converged:
            raise ConvergenceError(
                f"tiny-fit diagnostic did not converge: final Dice {final_dice:.4f} < {target_fg_dice:.4f} within {max_steps} steps. NO_GO.",
                details=result
            )

        return result

    finally:
        torch.set_rng_state(orig_torch_rng)
        if orig_cuda_rng is not None and torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.set_rng_state_all(orig_cuda_rng)
