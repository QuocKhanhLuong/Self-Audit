#!/usr/bin/env python3
"""Common checkpoint evaluation wrapper for shared student models.

Locked contract:
1. Two-phase evaluation: freeze predictions for ALL evaluation datasets BEFORE opening
   any reference masks. Never opens masks during inference.
2. Same payloadSHA and checkpoint manifest SHA verified for both datasets.
3. Raw student logits argmax before probability rounding. Finite valid shape logits required.
4. Per-class, per-patient macro: average present frames within patient, then mean
   available foreground classes (RV, MYO, LV) within each patient, then patients across dataset.
5. Optional equal-domain mean: average of domain patient macros (only if both defined).
6. Handles absent classes with explicit fixed-denominator false positive tracking.
   Ignored 255 has zero mass.
7. Verification of strict disjointness against TRAIN (by patient ID, canonical dataset:id,
   and image content / voxel SHA-256). Fails closed on missing TRAIN image arrays.
8. Never claims external clinical generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.data import TrainingData, write_json
from self_audit.model import digest
from self_audit.engine import load_checkpoint
from self_audit.evaluation import class_metrics
from self_audit.shared_data import parse_namespaced_id, namespace_patient_id


def safe_confusion_matrix(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Compute confusion matrix with strict integer dtype and class range validation.

    Rejects fractional labels, non-integer dtypes, shape mismatches, and out-of-range classes.
    """
    if not (np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.unsignedinteger)):
        raise ValueError(f"Ground truth mask must have integer dtype, got {y.dtype}")
    if not (np.issubdtype(prediction.dtype, np.integer) or np.issubdtype(prediction.dtype, np.unsignedinteger)):
        raise ValueError(f"Prediction array must have integer dtype, got {prediction.dtype}")
    if y.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: mask {y.shape} vs prediction {prediction.shape}")
    if not np.isin(y, [0, 1, 2, 3, 255]).all():
        raise ValueError("Mask contains invalid class values (must be in {0, 1, 2, 3, 255})")
    if not np.isin(prediction, [0, 1, 2, 3, 255]).all():
        raise ValueError("Prediction contains invalid class values (must be in {0, 1, 2, 3, 255})")

    valid = (y != 255)
    return np.bincount(
        (y[valid].astype(int) * 5 + np.where(prediction[valid] == 255, 4, prediction[valid])).astype(int),
        minlength=20
    ).reshape(4, 5)


def evaluate_shared_student(checkpoint_path: str | Path,
                            checkpoint_sha256: str,
                            domains_data: Mapping[str, tuple[str | Path, str]],
                            out_dir: str | Path,
                            *,
                            train_manifest_path: str | Path,
                            train_manifest_sha256: str,
                            device: str = "cpu",
                            threshold: float | None = None) -> dict:
    """Run two-phase frozen evaluation across one or multiple domain datasets.

    Args:
        checkpoint_path: Path to checkpoint directory containing MANIFEST.json and weights.pt.
        checkpoint_sha256: SHA-256 of checkpoint MANIFEST.json.
        domains_data: Dict mapping domain_name -> (manifest_path, manifest_sha256).
        out_dir: New output directory.
        train_manifest_path: Mandatory training manifest to check TRAIN disjointness.
        train_manifest_sha256: Mandatory SHA-256 of training manifest.
        device: 'cpu' or 'cuda'.
        threshold: Optional confidence threshold for gated predictions in [0.0, 1.0].
    """
    # ==========================================
    # PRE-EXECUTION CHECKS (BEFORE SIDE EFFECTS)
    # ==========================================
    if not train_manifest_path or not train_manifest_sha256:
        raise ValueError("train_manifest_path and train_manifest_sha256 are strictly required")

    if threshold is not None:
        if not (isinstance(threshold, (int, float)) and np.isfinite(threshold) and 0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be finite in [0.0, 1.0], got {threshold}")

    out = Path(out_dir).resolve()
    if out.exists():
        raise ValueError(f"Output directory already exists: {out}")

    ckpt_p = Path(checkpoint_path).resolve()
    if not ckpt_p.exists() or not (ckpt_p / "MANIFEST.json").is_file():
        raise ValueError(f"checkpoint manifest not found at {ckpt_p}")

    train_m_p = Path(train_manifest_path).resolve()
    if not train_m_p.is_file():
        raise ValueError(f"Training manifest not found at {train_m_p}")

    # Normalize domains once and reject duplicate case aliases
    if not domains_data:
        raise ValueError("Strict nonempty domains required; got empty domains_data")

    raw_keys = list(domains_data.keys())
    normalized_keys = [k.strip().lower() for k in raw_keys]
    if len(set(normalized_keys)) != len(raw_keys):
        raise ValueError(f"Duplicate case alias in evaluation domains: {raw_keys}")

    for dom_name in raw_keys:
        if not dom_name or "/" in dom_name or "\\" in dom_name or ".." in dom_name or not re.match(r'^[a-zA-Z0-9_-]+$', dom_name):
            raise ValueError(f"Unsafe or illegal domain name: '{dom_name}'")

    norm_keys_set = set(normalized_keys)
    if norm_keys_set not in [{"acdc", "mnm"}, {"acdc"}]:
        raise ValueError(f"Evaluation domains must be strictly 'acdc' and 'mnm' or single explicit 'acdc' mode; got {sorted(norm_keys_set)}")

    # Construct single normalized domains mapping
    normalized_domains = {}
    for raw_k in raw_keys:
        norm_k = raw_k.strip().lower()
        normalized_domains[norm_k] = domains_data[raw_k]

    # Load and verify checkpoint ONCE
    model, meta = load_checkpoint(ckpt_p, checkpoint_sha256, device=device)
    model.eval()

    payload_sha = meta.get("payload_sha256")
    if not payload_sha:
        raise ValueError("Checkpoint metadata missing payload_sha256")

    ckpt_data_sha = meta.get("data_sha256")
    if not ckpt_data_sha:
        raise ValueError("Checkpoint metadata missing data_sha256")
    if ckpt_data_sha != train_manifest_sha256:
        raise ValueError(
            f"Checkpoint data_sha256 ({ckpt_data_sha}) does not match expected train_manifest_sha256 ({train_manifest_sha256})"
        )

    ckpt_synthetic = bool(meta.get("synthetic", False))

    # Verify training manifest hash and records
    actual_train_sha = digest(train_m_p)
    if actual_train_sha != train_manifest_sha256:
        raise ValueError(f"Training manifest SHA mismatch: got {actual_train_sha}, expected {train_manifest_sha256}")

    tm_obj = json.loads(train_m_p.read_text())
    tm_labeled = tm_obj.get("labeled", [])
    tm_unlabeled = tm_obj.get("unlabeled", [])

    ckpt_lids = set(meta.get("labeled_ids", []))
    ckpt_uids = set(meta.get("unlabeled_ids", []))
    tm_lids = {r["id"] for r in tm_labeled}
    tm_uids = {r["id"] for r in tm_unlabeled}

    if ckpt_lids != tm_lids or ckpt_uids != tm_uids:
        raise ValueError(
            f"Checkpoint labeled/unlabeled IDs do not match training manifest IDs: "
            f"lids delta={ckpt_lids ^ tm_lids}, uids delta={ckpt_uids ^ tm_uids}"
        )

    # Collect TRAIN IDs and content hashes (failing closed for image identity proof)
    train_raw_ids = set(tm_lids | tm_uids)
    train_canonical_ids = set()
    for tid in train_raw_ids:
        d, raw_id = parse_namespaced_id(tid)
        train_canonical_ids.add((d.lower(), raw_id))

    resolve_tr = lambda p: Path(p) if Path(p).is_absolute() else train_m_p.parent / p
    train_image_shas = set()
    train_voxel_shas = set()
    referenced_paths_to_protect = [ckpt_p, train_m_p, train_m_p.parent]

    for r in tm_labeled + tm_unlabeled:
        if "image_sha256" in r:
            train_image_shas.add(r["image_sha256"])
        if "image" in r:
            ip = resolve_tr(r["image"])
            referenced_paths_to_protect.extend([ip, ip.parent])
            if not ip.is_file():
                raise FileNotFoundError(f"TRAIN image not found for patient '{r['id']}': {ip}")
            actual_img_sha = digest(ip)
            if "image_sha256" in r and actual_img_sha != r["image_sha256"]:
                raise ValueError(f"TRAIN image hash mismatch for patient '{r['id']}': {ip}")
            train_image_shas.add(actual_img_sha)

            tr_arr = np.load(ip, mmap_mode="r", allow_pickle=False)
            if tr_arr.ndim != 4 or not np.isfinite(tr_arr).all():
                raise ValueError(f"Invalid TRAIN image array for patient '{r['id']}': {ip}")
            v_sha = hashlib.sha256(np.ascontiguousarray(tr_arr).tobytes()).hexdigest()
            train_voxel_shas.add(v_sha)

    # Load and validate evaluation domain manifests
    domain_manifests = {}
    for dom_name, (m_path, m_sha) in normalized_domains.items():
        mp = Path(m_path).resolve()
        referenced_paths_to_protect.extend([mp, mp.parent])
        actual_dom_sha = digest(mp)
        if actual_dom_sha != m_sha:
            raise ValueError(f"Evaluation manifest hash mismatch for domain '{dom_name}': got {actual_dom_sha}, expected {m_sha}")
        m_obj = json.loads(mp.read_text())
        if m_obj.get("schema") != "self_audit.data.v2":
            raise ValueError(f"Domain '{dom_name}' manifest schema must be self_audit.data.v2")
        if m_obj.get("role") not in {"development", "external_frozen", "evaluation"}:
            raise ValueError(f"Domain '{dom_name}' role must be development or evaluation")

        dom_synthetic = bool(m_obj.get("synthetic", False))
        if dom_synthetic != ckpt_synthetic:
            raise ValueError(f"Domain '{dom_name}' synthetic flag ({dom_synthetic}) does not match checkpoint ({ckpt_synthetic})")

        patients = m_obj.get("patients", [])
        if not patients:
            raise ValueError(f"Evaluation domain '{dom_name}' has no patients (empty dataset)")

        eval_ids = []
        resolve_ev = lambda p: Path(p) if Path(p).is_absolute() else mp.parent / p

        for r in patients:
            pid = r.get("id")
            if not pid or not isinstance(pid, str):
                raise ValueError(f"Invalid patient record without string id in domain '{dom_name}'")
            if "/" in pid or "\\" in pid or ".." in pid or not re.match(r'^[a-zA-Z0-9_:-]+$', pid):
                raise ValueError(f"Unsafe patient ID '{pid}' in domain '{dom_name}'")
            eval_ids.append(pid)

            # Check direct ID collision with TRAIN
            if pid in train_raw_ids:
                raise ValueError(f"Evaluation domain '{dom_name}' patient ID overlap with TRAIN: {pid}")

            # Patient prefix conflicts (e.g. domain acdc with mnm:pid)
            eval_d, eval_raw = parse_namespaced_id(pid)
            if eval_d and eval_d != dom_name:
                raise ValueError(f"Patient ID '{pid}' has conflicting domain prefix '{eval_d}' for domain '{dom_name}'")

            if not eval_d:
                eval_d = dom_name

            for tr_d, tr_raw in train_canonical_ids:
                if tr_d and eval_d and tr_d == eval_d and tr_raw == eval_raw:
                    raise ValueError(f"Evaluation domain '{dom_name}' patient ID '{pid}' matches TRAIN patient '{tr_raw}' in domain '{tr_d}'")
                if not tr_d and (pid == tr_raw or eval_raw == tr_raw):
                    raise ValueError(f"Evaluation domain '{dom_name}' patient ID '{pid}' matches unnamespaced TRAIN patient '{tr_raw}'")

            # Check image SHA against TRAIN
            img_sha = r.get("image_sha256")
            if img_sha and img_sha in train_image_shas:
                raise ValueError(f"Evaluation domain '{dom_name}' image content overlap with TRAIN: patient {pid}")

            # Protect resolved eval image path
            if "image" in r:
                ev_ip = resolve_ev(r["image"])
                referenced_paths_to_protect.extend([ev_ip, ev_ip.parent])

            # Check annotated frames non-empty and validate mask records
            masks = r.get("masks", {})
            if not masks or not isinstance(masks, dict):
                raise ValueError(f"Evaluation patient '{pid}' in domain '{dom_name}' has no annotated frames")

            seen_ti = set()
            for ti_str, mr in masks.items():
                try:
                    ti = int(ti_str)
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid non-integer frame index '{ti_str}' for patient '{pid}'")
                if ti in seen_ti:
                    raise ValueError(f"Duplicate frame index {ti} for patient '{pid}'")
                seen_ti.add(ti)
                if not isinstance(mr, dict) or "path" not in mr or "sha256" not in mr:
                    raise ValueError(f"Malformed mask record for frame {ti} in patient '{pid}'")
                if not isinstance(mr["path"], str) or not mr["path"]:
                    raise ValueError(f"Missing mask path for frame {ti} in patient '{pid}'")
                if not isinstance(mr["sha256"], str) or len(mr["sha256"]) != 64:
                    raise ValueError(f"Invalid mask SHA-256 for frame {ti} in patient '{pid}'")

                ev_mp = resolve_ev(mr["path"])
                referenced_paths_to_protect.extend([ev_mp, ev_mp.parent])

        if len(set(eval_ids)) != len(eval_ids):
            raise ValueError(f"Duplicate patient IDs within evaluation domain '{dom_name}'")

        domain_manifests[dom_name] = (mp, m_sha, m_obj)

    # Protect output directory against ALL referenced resolved image/mask parents BEFORE ANY WRITES
    for ref_p in referenced_paths_to_protect:
        try:
            rp = ref_p.resolve()
            if out == rp or out.is_relative_to(rp) or rp.is_relative_to(out):
                raise ValueError(f"Output directory '{out}' conflicts with referenced input/parent path '{rp}'")
        except (AttributeError, ValueError) as err:
            if "conflicts with referenced input/parent path" in str(err):
                raise

    # All pre-execution checks passed; create output directories
    out.mkdir(parents=True)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True)

    recipe = {
        "schema": "self_audit.shared_evaluation.v1",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_payload_sha256": payload_sha,
        "train_manifest_sha256": train_manifest_sha256,
        "device": device,
        "threshold": threshold,
        "domains": {
            dom: {
                "manifest_sha256": d_sha,
                "patients": len(d_obj.get("patients", [])),
            }
            for dom, (d_p, d_sha, d_obj) in domain_manifests.items()
        },
        "synthetic": ckpt_synthetic,
    }
    recipe_file = out / "RECIPE.json"
    write_json(recipe_file, recipe)
    actual_recipe_sha = digest(recipe_file)

    # ==========================================
    # PHASE 1: GENERATE & FREEZE PREDICTIONS
    # NO REFERENCE MASKS OPENED IN THIS BLOCK
    # ==========================================
    frozen_inventory = {}
    with torch.inference_mode():
        for dom_name, (mp, d_sha, m_obj) in domain_manifests.items():
            dom_pred_dir = pred_dir / dom_name
            dom_pred_dir.mkdir(parents=True, exist_ok=True)
            resolve = lambda p: Path(p) if Path(p).is_absolute() else mp.parent / p

            dom_inventory = []
            for r in m_obj.get("patients", []):
                pid = r["id"]
                ip = resolve(r["image"])
                if not ip.is_file():
                    raise FileNotFoundError(f"Evaluation image not found: {ip}")
                if digest(ip) != r["image_sha256"]:
                    raise ValueError(f"Evaluation image hash mismatch: {ip}")
                x = np.load(ip, mmap_mode="r", allow_pickle=False)
                if x.ndim != 4 or not np.isfinite(x).all():
                    raise ValueError(f"Invalid evaluation image: {ip}")

                # Check voxel hash against TRAIN
                eval_voxel_sha = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
                if eval_voxel_sha in train_voxel_shas:
                    raise ValueError(f"Evaluation domain '{dom_name}' voxel content overlap with TRAIN: patient {pid}")

                t, z, h, w = x.shape

                # Check reference frame bounds 0 <= t < T before opening mask arrays
                for ti_str in r.get("masks", {}).keys():
                    ti = int(ti_str)
                    if ti < 0 or ti >= t:
                        raise ValueError(f"Frame index {ti} out of bounds for patient '{pid}' with T={t}")

                safe_pred_name = f"{pid}.npy"
                op = dom_pred_dir / safe_pred_name
                if not op.resolve().is_relative_to(dom_pred_dir.resolve()):
                    raise ValueError(f"Prediction path escapes domain directory: {op}")

                raw_pred = np.zeros((t, z, h, w), dtype=np.uint8)
                prob_map = np.zeros((t, z, 4, h, w), dtype=np.float32) if threshold is not None else None

                for ti in range(t):
                    for zi in range(z):
                        ctx = TrainingData.context(x, ti, zi).to(device)
                        logits = model(ctx)

                        if not torch.isfinite(logits).all():
                            raise ValueError(f"Non-finite logits produced for patient '{pid}', frame {ti}, slice {zi}")
                        if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 4 or logits.shape[2] != h or logits.shape[3] != w:
                            raise ValueError(f"Invalid logits shape {logits.shape} for patient '{pid}', expected (1, 4, {h}, {w})")

                        raw_pred[ti, zi] = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                        if threshold is not None:
                            prob_map[ti, zi] = logits.softmax(dim=1)[0].cpu().numpy()

                np.save(op, raw_pred)
                pred_sha = digest(op)
                entry = {"id": pid, "file": f"{dom_name}/{op.name}", "sha256": pred_sha}
                if threshold is not None:
                    prob_op = dom_pred_dir / f"{pid}_prob.npy"
                    np.save(prob_op, prob_map)
                    entry["prob_file"] = f"{dom_name}/{prob_op.name}"
                    entry["prob_sha256"] = digest(prob_op)

                dom_inventory.append(entry)

            frozen_inventory[dom_name] = dom_inventory

    frozen_file = out / "PREDICTIONS_FROZEN.json"
    write_json(frozen_file, {
        "recipe_sha256": actual_recipe_sha,
        "checkpoint_manifest_sha256": checkpoint_sha256,
        "checkpoint_payload_sha256": payload_sha,
        "domains": frozen_inventory,
    })
    actual_frozen_sha = digest(frozen_file)

    # ==========================================
    # PHASE 2: SCORE FROZEN PREDICTIONS
    # MASKS OPENED ONLY AFTER ALL PREDICTIONS FROZEN
    # ==========================================
    for dom_name, inventory in frozen_inventory.items():
        for pr in inventory:
            op = pred_dir / pr["file"]
            if not op.is_file():
                raise FileNotFoundError(f"Missing frozen prediction file: {op}")
            if digest(op) != pr["sha256"]:
                raise ValueError(f"Frozen prediction file modified: {op}")
            if threshold is not None and "prob_file" in pr:
                prob_op = pred_dir / pr["prob_file"]
                if not prob_op.is_file():
                    raise FileNotFoundError(f"Missing frozen probability file: {prob_op}")
                if digest(prob_op) != pr["prob_sha256"]:
                    raise ValueError(f"Frozen probability file modified: {prob_op}")

    domain_results = {}
    failure_list = []
    all_frame_rows = []

    for dom_name, (mp, d_sha, m_obj) in domain_manifests.items():
        resolve = lambda p: Path(p) if Path(p).is_absolute() else mp.parent / p
        inventory = frozen_inventory[dom_name]
        dom_pred_dir = pred_dir / dom_name

        frame_rows = []
        domain_confusion = np.zeros((4, 5), dtype=np.int64)

        for r, pr in zip(m_obj.get("patients", []), inventory):
            op = pred_dir / pr["file"]
            pred_arr = np.load(op, allow_pickle=False)

            prob_arr = None
            if threshold is not None and "prob_file" in pr:
                prob_op = pred_dir / pr["prob_file"]
                prob_arr = np.load(prob_op, allow_pickle=False)

            for ti_str, mr in r.get("masks", {}).items():
                ti = int(ti_str)
                ref_p = resolve(mr["path"])
                if not ref_p.is_file():
                    raise FileNotFoundError(f"Reference mask not found: {ref_p}")
                if digest(ref_p) != mr["sha256"]:
                    raise ValueError(f"Reference mask hash mismatch: {ref_p}")

                y = np.load(ref_p, allow_pickle=False)
                pred_slice = pred_arr[ti]

                cm = safe_confusion_matrix(y, pred_slice)
                domain_confusion += cm
                c_metrics = class_metrics(cm)

                valid_voxels_frame = int(cm.sum())
                for cm_row in c_metrics:
                    cm_row["valid_voxels"] = valid_voxels_frame
                    if not cm_row["reference_present"]:
                        cm_row["absent_fp_rate"] = float(cm_row["prediction_count"] / valid_voxels_frame) if valid_voxels_frame > 0 else 0.0
                    else:
                        cm_row["absent_fp_rate"] = None

                row = {
                    "domain": dom_name,
                    "patient": r["id"],
                    "frame": ti,
                    "raw": c_metrics,
                    "confusion": cm.tolist(),
                    "valid_voxels": valid_voxels_frame,
                }

                if threshold is not None and prob_arr is not None:
                    p_slice = prob_arr[ti]
                    gated_pred = np.where(p_slice.max(axis=1) >= threshold, pred_slice, 255)
                    g_cm = safe_confusion_matrix(y, gated_pred)
                    row["gated"] = class_metrics(g_cm)
                    valid_y = (y != 255)
                    row["coverage"] = float((gated_pred[valid_y] != 255).mean()) if valid_y.any() else 0.0

                frame_rows.append(row)
                all_frame_rows.append(row)

        # Per-patient aggregation
        patient_metrics = {}
        patient_fg_dices = []
        patient_ids = [r["id"] for r in m_obj.get("patients", [])]

        for pid in patient_ids:
            p_frames = [row for row in frame_rows if row["patient"] == pid]
            p_classes = {}
            for j, c_name in enumerate(["BG", "RV", "MYO", "LV"]):
                vals = [f["raw"][j]["dice"] for f in p_frames if f["raw"][j]["dice"] is not None]
                p_classes[c_name] = float(np.mean(vals)) if vals else None

            # Valid prediction counts and coverage per patient
            p_valid_voxels = sum(f["valid_voxels"] for f in p_frames)
            p_pred_counts = {
                c_name: sum(f["raw"][j]["prediction_count"] for f in p_frames)
                for j, c_name in enumerate(["BG", "RV", "MYO", "LV"])
            }
            p_pred_valid = sum(p_pred_counts.values())
            p_coverage = float(p_pred_valid / p_valid_voxels) if p_valid_voxels > 0 else 0.0

            p_classes["valid_voxels"] = p_valid_voxels
            p_classes["prediction_coverage"] = p_coverage
            p_classes["class_prediction_counts"] = p_pred_counts

            # Foreground mean within patient (RV, MYO, LV only; absent means None)
            avail_fg = [p_classes[c] for c in ["RV", "MYO", "LV"] if p_classes[c] is not None]
            patient_fg = float(np.mean(avail_fg)) if avail_fg else None
            p_classes["foreground_mean"] = patient_fg
            patient_metrics[pid] = p_classes
            if patient_fg is not None:
                patient_fg_dices.append(patient_fg)

            if patient_fg is None:
                failure_list.append({
                    "domain": dom_name,
                    "patient": pid,
                    "reason": "absent_or_undefined_foreground",
                    "foreground_mean": None,
                    "scores": p_classes,
                })
            elif patient_fg < 0.50 or any(p_classes[c] is not None and p_classes[c] == 0.0 for c in ["RV", "MYO", "LV"]):
                failure_list.append({
                    "domain": dom_name,
                    "patient": pid,
                    "reason": "collapsed_or_low_foreground",
                    "foreground_mean": patient_fg,
                    "scores": p_classes,
                })

        # Dataset macro: mean of patient foreground means across patients with defined FG
        dom_fg_macro = float(np.mean(patient_fg_dices)) if patient_fg_dices else None

        # Class macros across patients
        class_macros = {}
        for c_name in ["BG", "RV", "MYO", "LV"]:
            vals = [p[c_name] for p in patient_metrics.values() if p[c_name] is not None]
            class_macros[c_name] = {
                "dice": float(np.mean(vals)) if vals else None,
                "present_patients": len(vals),
            }

        all_class_vals = [class_macros[c]["dice"] for c in ["BG", "RV", "MYO", "LV"] if class_macros[c]["dice"] is not None]
        all_class_macro = float(np.mean(all_class_vals)) if all_class_vals else None

        # Per-class counts and absent FP rate
        dom_valid_voxels = int(domain_confusion.sum())
        domain_class_counts = {}
        for j, c_name in enumerate(["BG", "RV", "MYO", "LV"]):
            tp = int(domain_confusion[j, j])
            gt = int(domain_confusion[j].sum())
            pred = int(domain_confusion[:, j].sum())
            fp = pred - tp
            fn = gt - tp
            domain_class_counts[c_name] = {
                "prediction_count": pred,
                "reference_count": gt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "reference_present": bool(gt > 0),
                "absent_fp_rate": float(pred / dom_valid_voxels) if gt == 0 and dom_valid_voxels > 0 else None,
            }

        total_annotated_frames = sum(len(r.get("masks", {})) for r in m_obj.get("patients", []))
        scored_frames = len(frame_rows)
        frame_coverage = float(scored_frames / total_annotated_frames) if total_annotated_frames > 0 else 0.0
        dom_predicted_valid = int(domain_confusion[:, :4].sum())
        dom_pred_coverage = float(dom_predicted_valid / dom_valid_voxels) if dom_valid_voxels > 0 else 0.0

        # Development gate requires score, complete coverage, and non-collapse prediction counts
        fg_pass = (dom_fg_macro is not None and dom_fg_macro >= 0.75)
        fg_classes_pass = all(
            class_macros[c]["dice"] is not None and class_macros[c]["dice"] >= 0.60
            for c in ["RV", "MYO", "LV"]
        )
        complete_coverage = (frame_coverage == 1.0 and dom_pred_coverage == 1.0 and len(patient_metrics) == len(patient_ids))
        all_classes_predicted = all(domain_class_counts[c]["prediction_count"] > 0 for c in ["BG", "RV", "MYO", "LV"])

        if dom_fg_macro is None:
            gate_status = "UNDEFINED_NO_GO"
        elif fg_pass and fg_classes_pass and complete_coverage and all_classes_predicted:
            gate_status = "GO"
        else:
            gate_status = "NO_GO"

        dev_gate = {
            "status": gate_status,
            "foreground_threshold": 0.75,
            "min_class_threshold": 0.60,
            "foreground_patient_macro": dom_fg_macro,
            "class_dices": {c: class_macros[c]["dice"] for c in ["RV", "MYO", "LV"]},
            "complete_coverage": bool(complete_coverage),
            "all_classes_predicted": bool(all_classes_predicted),
            "frame_coverage": frame_coverage,
            "prediction_coverage": dom_pred_coverage,
            "informational_only": True,
            "note": "Fixed development NO_GO, never hyperparam adjust; requires complete coverage and non-collapse counts",
        }

        domain_results[dom_name] = {
            "manifest_sha256": d_sha,
            "foreground_patient_macro": dom_fg_macro,
            "all_class_macro": all_class_macro,
            "patient_count": {
                "expected": len(patient_ids),
                "scored": len(patient_metrics),
                "with_defined_fg": len(patient_fg_dices),
            },
            "macro_patient": class_macros,
            "class_counts": domain_class_counts,
            "frame_coverage": frame_coverage,
            "prediction_coverage": dom_pred_coverage,
            "valid_voxels": dom_valid_voxels,
            "development_gate": dev_gate,
            "patient_metrics": patient_metrics,
            "pooled_confusion": domain_confusion.tolist(),
            "frames": frame_rows,
        }

    # Combined macro across domains
    if "acdc" in domain_results and "mnm" in domain_results:
        acdc_fg = domain_results["acdc"]["foreground_patient_macro"]
        mnm_fg = domain_results["mnm"]["foreground_patient_macro"]
        if acdc_fg is not None and mnm_fg is not None:
            equal_domain_fg_macro = float((acdc_fg + mnm_fg) / 2.0)
        else:
            equal_domain_fg_macro = None
    else:
        equal_domain_fg_macro = None

    final_result = {
        "schema": "self_audit.shared_evaluation_metrics.v2",
        "checkpoint_manifest_sha256": checkpoint_sha256,
        "checkpoint_payload_sha256": payload_sha,
        "train_manifest_sha256": train_manifest_sha256,
        "recipe_sha256": actual_recipe_sha,
        "frozen_predictions_sha256": actual_frozen_sha,
        "synthetic": ckpt_synthetic,
        "equal_domain_foreground_macro": equal_domain_fg_macro,
        "domains": domain_results,
        "failures": failure_list,
        "failure_count": len(failure_list),
        "frames": all_frame_rows,
        "guarantee_limits": (
            "Predictions frozen before masks opened. "
            "Evaluated on development/test domain sets; explicitly does not certify external clinical domain generalization."
        ),
    }

    # Single domain backwards compatibility
    if "acdc" in domain_results and len(domain_results) == 1:
        final_result["foreground_patient_macro"] = domain_results["acdc"]["foreground_patient_macro"]
        final_result["macro_patient"] = domain_results["acdc"]["macro_patient"]
    elif equal_domain_fg_macro is not None:
        final_result["foreground_patient_macro"] = equal_domain_fg_macro

    write_json(out / "METRICS.json", final_result)
    return final_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--checkpoint-sha256", required=True, help="Checkpoint MANIFEST.json SHA-256")
    parser.add_argument("--train-manifest", required=True, help="Mandatory TRAIN manifest path")
    parser.add_argument("--train-manifest-sha256", required=True, help="Mandatory TRAIN manifest SHA-256")
    parser.add_argument("--data-acdc", help="ACDC evaluation manifest")
    parser.add_argument("--data-acdc-sha256", help="ACDC evaluation manifest SHA-256")
    parser.add_argument("--data-mnm", help="MnM evaluation manifest")
    parser.add_argument("--data-mnm-sha256", help="MnM evaluation manifest SHA-256")
    parser.add_argument("--data", help="Single domain evaluation manifest (ACDC)")
    parser.add_argument("--data-sha256", help="Single domain evaluation manifest SHA-256")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, help="Optional probability threshold in [0.0, 1.0]")
    args = parser.parse_args()

    # Reject partial arg pairs
    if (args.data_acdc is None) != (args.data_acdc_sha256 is None):
        parser.error("--data-acdc and --data-acdc-sha256 must be provided together")
    if (args.data_mnm is None) != (args.data_mnm_sha256 is None):
        parser.error("--data-mnm and --data-mnm-sha256 must be provided together")
    if (args.data is None) != (args.data_sha256 is None):
        parser.error("--data and --data-sha256 must be provided together")

    domains = {}
    if args.data_acdc and args.data_acdc_sha256:
        domains["acdc"] = (args.data_acdc, args.data_acdc_sha256)
    if args.data_mnm and args.data_mnm_sha256:
        domains["mnm"] = (args.data_mnm, args.data_mnm_sha256)
    if args.data and args.data_sha256:
        domains["acdc"] = (args.data, args.data_sha256)

    if not domains:
        parser.error("At least one evaluation dataset must be specified (--data-acdc/--data-mnm or --data)")

    res = evaluate_shared_student(
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        domains_data=domains,
        out_dir=args.out,
        train_manifest_path=args.train_manifest,
        train_manifest_sha256=args.train_manifest_sha256,
        device=args.device,
        threshold=args.threshold,
    )
    print(json.dumps({
        "status": "COMPLETED",
        "equal_domain_foreground_macro": res.get("equal_domain_foreground_macro"),
        "domains": {k: v.get("foreground_patient_macro") for k, v in res.get("domains", {}).items()},
        "failure_count": res.get("failure_count", 0),
    }, indent=2))


if __name__ == "__main__":
    main()
