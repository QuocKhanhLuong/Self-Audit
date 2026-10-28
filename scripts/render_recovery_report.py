#!/usr/bin/env python3
"""Offline recovery reporting CLI: training curves, frozen overlays, and provenance."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Supported schema versions for strict fail-closed validation
VALID_MANIFEST_SCHEMAS = {"self_audit.data.v2"}
VALID_EVALUATION_ROLES = {"development", "external_frozen", "evaluation"}
CANONICAL_CLASSES = ["BG", "RV", "MYO", "LV"]
ALLOWED_VOXEL_LABELS = {0, 1, 2, 3, 255}


def digest(path: Path | str) -> str:
    """Compute SHA-256 digest of a file in streaming 1MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path | str, data: Any) -> None:
    """Write formatted JSON without NaN/inf."""
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def validate_sha256_format(val: str, field_name: str = "hash") -> None:
    """Ensure string is a valid 64-character lowercase hexadecimal sha256."""
    if not isinstance(val, str) or len(val) != 64:
        raise ValueError(f"invalid {field_name}: expected 64-character hex string, got {val!r}")
    try:
        int(val, 16)
    except ValueError:
        raise ValueError(f"invalid {field_name}: non-hex characters in {val!r}")


def validate_safe_identifier(val: str, name: str = "identifier") -> None:
    """Ensure identifier has no path traversal, slashes, or unsafe characters."""
    if not isinstance(val, str) or not val:
        raise ValueError(f"empty or invalid {name}: {val!r}")
    if ".." in val or "/" in val or "\\" in val or "\0" in val:
        raise ValueError(f"unsafe characters or path traversal in {name}: {val!r}")


def validate_output_directory(out: Path, protected_paths: Sequence[Path]) -> None:
    """Enforce strict brand-new output location outside all input directories and parents."""
    out = out.resolve()
    if out.exists():
        raise ValueError(f"output directory {out} already exists; brand new directory required")

    for p in protected_paths:
        if p is None:
            continue
        p = p.resolve()
        p_dir = p if p.is_dir() else p.parent
        if out == p or out == p_dir:
            raise ValueError(f"output directory {out} conflicts with input path: {p}")
        if out.is_relative_to(p_dir):
            raise ValueError(f"output directory {out} must not be inside input directory: {p_dir}")
        if p.is_relative_to(out) or p_dir.is_relative_to(out):
            raise ValueError(f"input path {p} must not be inside output directory: {out}")

    out.mkdir(parents=True, exist_ok=False)


def label_to_rgba(labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Map class indices to RGBA with canonical colors and transparent background / ignore.

    Canonical colors:
      Class 0 (BG): transparent (alpha=0.0)
      Class 1 (RV): red [1.0, 0.0, 0.0]
      Class 2 (MYO): green [0.0, 0.8, 0.0]
      Class 3 (LV): blue [0.0, 0.2, 1.0]
      Class 255 (Ignore): transparent (alpha=0.0)
    """
    if labels.ndim != 2:
        raise ValueError(f"label_to_rgba requires 2D array, got shape {labels.shape}")

    # Check for unknown segmentation labels
    unique_labels = set(np.unique(labels).tolist())
    unknown = unique_labels - ALLOWED_VOXEL_LABELS
    if unknown:
        raise ValueError(f"unknown segmentation label(s) in slice: {sorted(unknown)}")

    h, w = labels.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    # Class 1: RV -> red
    mask_rv = (labels == 1)
    rgba[mask_rv] = [1.0, 0.0, 0.0, alpha]

    # Class 2: MYO -> green
    mask_myo = (labels == 2)
    rgba[mask_myo] = [0.0, 0.8, 0.0, alpha]

    # Class 3: LV -> blue
    mask_lv = (labels == 3)
    rgba[mask_lv] = [0.0, 0.2, 1.0, alpha]

    # Class 0 (BG) and 255 (Ignore) remain [0, 0, 0, 0] transparent.
    return rgba


def render_learning_curves(steps_jsonl: Path, out_file: Path, synthetic: bool = False) -> dict[str, Any]:
    """Render standalone training learning curve PNG from JSONL."""
    if not steps_jsonl.is_file():
        raise FileNotFoundError(f"training steps JSONL not found: {steps_jsonl}")

    steps = []
    sup_losses = []
    pseudo_series = []
    has_pseudo_updates = False
    ema_series = []
    has_ema_updates = False

    raw_pseudo_per_step = []

    with open(steps_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                raise ValueError(f"malformed JSON in {steps_jsonl} line {line_no}: {e}")

            # Step numbers must be finite positive, strictly ordered, and unique
            # Strict bool reject: in Python, isinstance(True, int) is True, so type check is mandatory
            raw_step = row.get("step")
            if type(raw_step) is bool or not isinstance(raw_step, int) or raw_step <= 0:
                raise ValueError(
                    f"step number in line {line_no} must be a positive integer (strict bool reject), got: {raw_step!r}"
                )
            if steps and raw_step <= steps[-1]:
                raise ValueError(
                    f"step numbers must be strictly increasing and unique: step {raw_step} <= {steps[-1]} at line {line_no}"
                )
            steps.append(raw_step)

            # Supervised loss (required, finite)
            sloss = row.get("supervised_loss")
            if sloss is None or type(sloss) is bool or not math.isfinite(sloss):
                raise ValueError(f"nonfinite or missing supervised_loss at step {raw_step}: {sloss}")
            sup_losses.append(float(sloss))

            # Pseudo update tracking:
            # First check for cumulative engine counters emitted directly
            opt_counters = row.get("optimizer_counters") or {}
            slices_opt = row.get("slices_optimized") or {}
            counts_dict = row.get("counts") or {}

            direct_pseudo = None
            if "pseudo_optimizer_updates" in row:
                direct_pseudo = row["pseudo_optimizer_updates"]
            elif isinstance(opt_counters, dict) and "pseudo_accepted" in opt_counters:
                direct_pseudo = opt_counters["pseudo_accepted"]
            elif isinstance(opt_counters, dict) and "pseudo" in opt_counters:
                direct_pseudo = opt_counters["pseudo"]
            elif isinstance(counts_dict, dict) and "pseudo_accepted" in counts_dict:
                direct_pseudo = counts_dict["pseudo_accepted"]
            elif isinstance(slices_opt, dict) and "unlabeled" in slices_opt:
                direct_pseudo = slices_opt["unlabeled"]
            elif "pseudo_accepted" in row and type(row["pseudo_accepted"]) is not bool and isinstance(row["pseudo_accepted"], (int, float)):
                direct_pseudo = row["pseudo_accepted"]
            elif "nonzero_displacement_updates" in row and type(row["nonzero_displacement_updates"]) is not bool:
                direct_pseudo = row["nonzero_displacement_updates"]

            if direct_pseudo is not None:
                if type(direct_pseudo) is bool or not isinstance(direct_pseudo, (int, float)) or not math.isfinite(direct_pseudo) or direct_pseudo < 0:
                    raise ValueError(f"invalid pseudo counter value at line {line_no}: {direct_pseudo!r}")
                pseudo_series.append(float(direct_pseudo))
                has_pseudo_updates = True
            else:
                # Check per-step actual optimization signals (control scale > 0, etc.)
                step_accepted = 0
                has_step_signal = False
                if "control" in row and isinstance(row["control"], dict):
                    scale = row["control"].get("scale")
                    if scale is not None and math.isfinite(scale):
                        step_accepted = 1 if scale > 0 else 0
                        has_step_signal = True
                elif "pseudo_accepted" in row and type(row["pseudo_accepted"]) is bool:
                    step_accepted = 1 if row["pseudo_accepted"] else 0
                    has_step_signal = True
                elif "pseudo_directions_accepted" in row and type(row["pseudo_directions_accepted"]) is not bool:
                    step_accepted = 1 if row["pseudo_directions_accepted"] > 0 else 0
                    has_step_signal = True

                if has_step_signal:
                    raw_pseudo_per_step.append(step_accepted)
                    has_pseudo_updates = True
                else:
                    raw_pseudo_per_step.append(None)

            # EMA updates tracking:
            # Check for cumulative EMA counters emitted directly
            direct_ema = None
            if "cumulative_ema_updates" in row:
                direct_ema = row["cumulative_ema_updates"]
            elif isinstance(opt_counters, dict) and "ema_updates" in opt_counters:
                direct_ema = opt_counters["ema_updates"]
            elif isinstance(opt_counters, dict) and "ema" in opt_counters:
                direct_ema = opt_counters["ema"]
            elif isinstance(counts_dict, dict) and "ema_updates" in counts_dict:
                direct_ema = counts_dict["ema_updates"]
            elif "ema_updates" in row and type(row["ema_updates"]) is not bool and isinstance(row["ema_updates"], (int, float)):
                direct_ema = row["ema_updates"]
            elif "ema_update_coverage" in row and type(row["ema_update_coverage"]) is not bool and isinstance(row["ema_update_coverage"], (int, float)):
                direct_ema = row["ema_update_coverage"]

            if direct_ema is not None:
                if type(direct_ema) is bool or not isinstance(direct_ema, (int, float)) or not math.isfinite(direct_ema) or direct_ema < 0:
                    raise ValueError(f"invalid EMA counter value at line {line_no}: {direct_ema!r}")
                ema_series.append(float(direct_ema))
                has_ema_updates = True
            else:
                ema_series.append(None)

    if not steps:
        raise ValueError(f"steps JSONL is empty: {steps_jsonl}")

    # Resolve pseudo series if per-step signals were accumulated
    if not pseudo_series and has_pseudo_updates:
        accum = 0.0
        for s in raw_pseudo_per_step:
            if s is not None:
                accum += s
                pseudo_series.append(accum)
            else:
                pseudo_series.append(None)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel 1: Supervised loss
    axs[0].plot(steps, sup_losses, color="#1f77b4", lw=1.8)
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Supervised Loss")
    axs[0].set_title("TRAIN Supervised Loss")
    axs[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Pseudo accepted updates
    if has_pseudo_updates and any(v is not None for v in pseudo_series):
        valid_pseudo = [(s, v) for s, v in zip(steps, pseudo_series) if v is not None]
        ps_steps, ps_vals = zip(*valid_pseudo)
        axs[1].plot(ps_steps, ps_vals, color="#2ca02c", lw=1.8)
        axs[1].set_xlabel("Step")
        axs[1].set_ylabel("Pseudo Updates / Optimized")
        axs[1].set_title("Pseudo Updates Accepted")
        axs[1].grid(True, linestyle="--", alpha=0.5)
        total_pseudo = float(ps_vals[-1])
    else:
        axs[1].text(
            0.5, 0.5,
            "Pseudo updates accepted:\nN/A (not recorded in training log)",
            ha="center", va="center", transform=axs[1].transAxes,
            fontsize=10, color="gray",
        )
        axs[1].set_title("Pseudo Updates Accepted (N/A)")
        axs[1].axis("off")
        total_pseudo = None

    # Panel 3: EMA update coverage
    if has_ema_updates and any(v is not None for v in ema_series):
        valid_ema = [(s, v) for s, v in zip(steps, ema_series) if v is not None]
        vs_steps, vs_vals = zip(*valid_ema)
        axs[2].plot(vs_steps, vs_vals, color="#ff7f0e", lw=1.8)
        axs[2].set_xlabel("Step")
        axs[2].set_ylabel("EMA Updates / Coverage")
        axs[2].set_title("EMA Updates / Recorded Coverage")
        axs[2].grid(True, linestyle="--", alpha=0.5)
    else:
        axs[2].text(
            0.5, 0.5,
            "EMA update coverage:\nN/A (not recorded in training log)",
            ha="center", va="center", transform=axs[2].transAxes,
            fontsize=10, color="gray",
        )
        axs[2].set_title("EMA Update Coverage (N/A)")
        axs[2].axis("off")

    if synthetic:
        fig.text(0.5, 0.02, "SYNTHETIC FIXTURE — NOT CLINICAL DATA", ha="center",
                 fontsize=11, color="red", weight="bold")
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    else:
        fig.tight_layout()

    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "total_steps": len(steps),
        "initial_supervised_loss": sup_losses[0],
        "final_supervised_loss": sup_losses[-1],
        "total_pseudo_accepted": total_pseudo,
        "has_ema_coverage": has_ema_updates,
    }


def load_prediction_file(pred_path: Path, expected_t: int, expected_z: int, expected_h: int, expected_w: int) -> np.ndarray:
    """Load prediction array with strict shape and uint8 class validation."""
    if not pred_path.is_file():
        raise FileNotFoundError(f"prediction file not found: {pred_path}")
    if pred_path.name.endswith(".gz"):
        with gzip.open(pred_path, "rb") as stream:
            arr = np.load(stream, allow_pickle=False)
    else:
        arr = np.load(pred_path, allow_pickle=False)

    if not np.isfinite(arr).all():
        raise ValueError(f"prediction contains non-finite values: {pred_path}")

    # Exact shapes normalized T/Z/H/W current labels uint8; no guessed 4D probability ambiguity
    if arr.ndim == 4:
        if arr.shape != (expected_t, expected_z, expected_h, expected_w):
            raise ValueError(
                f"prediction shape {arr.shape} does not match expected cine dimensions "
                f"({expected_t}, {expected_z}, {expected_h}, {expected_w}): {pred_path}"
            )
        if arr.dtype != np.uint8:
            raise ValueError(f"4D prediction array must be uint8 labels, got dtype {arr.dtype}: {pred_path}")
    elif arr.ndim == 5:
        # 5D float logits/probabilities: (T, Z, 4, H, W)
        if arr.shape != (expected_t, expected_z, 4, expected_h, expected_w):
            raise ValueError(
                f"5D prediction probabilities shape {arr.shape} does not match expected "
                f"({expected_t}, {expected_z}, 4, {expected_h}, {expected_w}): {pred_path}"
            )
        arr = arr.argmax(axis=2).astype(np.uint8)
    else:
        raise ValueError(
            f"prediction array must be 4D uint8 or 5D (T, Z, 4, H, W) float, got ndim={arr.ndim}: {pred_path}"
        )

    # Validate label vocabulary
    unique_labels = set(np.unique(arr).tolist())
    unknown = unique_labels - ALLOWED_VOXEL_LABELS
    if unknown:
        raise ValueError(f"unknown prediction labels {sorted(unknown)} in: {pred_path}")

    return arr


def render_overlay_figure(
    image_slice: np.ndarray,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    dataset: str,
    patient_id: str,
    frame_idx: int,
    slice_idx: int,
    checkpoint_sha: str,
    out_file: Path,
    frame_metrics_str: str | None = None,
    synthetic: bool = False,
) -> None:
    """Render deterministic 3-panel overlay PNG: Grayscale image, GT overlay, Pred overlay."""
    if image_slice.ndim != 2 or gt_slice.ndim != 2 or pred_slice.ndim != 2:
        raise ValueError("all slices must be 2D arrays")

    # Normalize grayscale image to [0, 1] for robust display
    img_min, img_max = float(image_slice.min()), float(image_slice.max())
    denom = max(img_max - img_min, 1e-8)
    norm_img = (image_slice.astype(np.float32) - img_min) / denom

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.8))

    # Panel 1: Pure grayscale cine slice
    axs[0].imshow(norm_img, cmap="gray", aspect="equal", origin="upper")
    axs[0].set_title("Grayscale Cine Image", fontsize=11)
    axs[0].axis("off")

    # Panel 2: Ground Truth overlay
    axs[1].imshow(norm_img, cmap="gray", aspect="equal", origin="upper")
    gt_rgba = label_to_rgba(gt_slice, alpha=0.55)
    axs[1].imshow(gt_rgba, aspect="equal", origin="upper")
    axs[1].set_title("Reference Ground Truth", fontsize=11)
    axs[1].axis("off")

    # Panel 3: Prediction overlay
    axs[2].imshow(norm_img, cmap="gray", aspect="equal", origin="upper")
    pred_rgba = label_to_rgba(pred_slice, alpha=0.55)
    axs[2].imshow(pred_rgba, aspect="equal", origin="upper")
    axs[2].set_title("Frozen Student Prediction", fontsize=11)
    axs[2].axis("off")

    # Add legend patches
    patches = [
        mpatches.Patch(color="red", label="RV (1)"),
        mpatches.Patch(color="green", label="MYO (2)"),
        mpatches.Patch(color="blue", label="LV (3)"),
    ]
    fig.legend(handles=patches, loc="lower center", bbox_to_anchor=(.5, .065 if synthetic else .005),
               ncol=3, frameon=False, fontsize=10)

    # Explicit dataset/patient/frame/slice and checkpointSHA label; no fabricated scores
    title_text = (
        f"Dataset: {dataset} | Patient: {patient_id} | Frame: {frame_idx} | Slice: {slice_idx}\n"
        f"Checkpoint SHA: {checkpoint_sha[:16]}..."
    )
    if frame_metrics_str:
        title_text += f" | {frame_metrics_str}"
    else:
        title_text += " | Measured FG Dice: N/A"

    fig.suptitle(title_text, fontsize=11, y=0.98)

    if synthetic:
        fig.text(0.5, 0.02, "SYNTHETIC FIXTURE — NOT CLINICAL DATA", ha="center",
                 fontsize=11, color="red", weight="bold")
        fig.tight_layout(rect=[0, 0.14, 1, 0.92])
    else:
        fig.tight_layout(rect=[0, 0.06, 1, 0.92])

    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_recovery_report(
    out_dir: Path,
    *,
    train_log: Path | None = None,
    evaluations: Sequence[Path] | None = None,
    manifests: Sequence[Path] | None = None,
    checkpoint_sha: str | None = None,
    max_overlays_per_dataset: int = 3,
) -> dict[str, Any]:
    """Execute recovery report generation with strict verification before opening reference masks."""
    out_dir = out_dir.resolve()
    evaluations = list(evaluations or [])
    manifests = list(manifests or [])

    if train_log is None and not evaluations:
        raise ValueError("At least --train-log or (--evaluation and --manifest) must be provided")
    if evaluations and not manifests:
        raise ValueError("manifests required when evaluation directory is specified")

    if not isinstance(max_overlays_per_dataset, int) or max_overlays_per_dataset <= 0:
        raise ValueError(
            f"max_overlays_per_dataset must be a positive integer, got: {max_overlays_per_dataset!r}"
        )

    # Collect protected input paths for output safety check
    protected_paths = []
    if train_log is not None:
        protected_paths.append(train_log)
    for ed in evaluations:
        protected_paths.append(ed)
    for m in manifests:
        protected_paths.append(m)

    # If only train-log is provided, render curves only and exit cleanly
    if train_log is not None and not evaluations:
        validate_output_directory(out_dir, protected_paths)
        lc_out = out_dir / "learning_curves.png"
        lc_info = render_learning_curves(train_log, lc_out, synthetic=False)
        prov = {
            "schema": "self_audit.recovery_report.v1",
            "checkpoint_sha256": None,
            "datasets": [],
            "learning_curves": {
                "file": lc_out.name,
                "sha256": digest(lc_out),
                "metrics": lc_info,
            },
            "overlays": [],
            "metrics_pointers": {},
            "synthetic": False,
        }
        write_json(out_dir / "PROVENANCE.json", prov)
        return prov

    # 1. Parse and verify evaluation packages BEFORE opening reference masks
    eval_packages = []
    recipe_manifest_shas = {}  # sha -> domain

    common_ckpt_sha = None
    common_payload_sha = None

    for eval_dir in evaluations:
        eval_dir = eval_dir.resolve()
        frozen_receipt = eval_dir / "PREDICTIONS_FROZEN.json"
        recipe_path = eval_dir / "RECIPE.json"
        metrics_path = eval_dir / "METRICS.json"

        if not frozen_receipt.is_file():
            raise ValueError(f"PREDICTIONS_FROZEN.json missing in evaluation dir: {eval_dir}")
        if not recipe_path.is_file():
            raise ValueError(f"RECIPE.json missing in evaluation dir: {eval_dir}")

        try:
            frozen_data = json.loads(frozen_receipt.read_text())
        except Exception as e:
            raise ValueError(f"malformed PREDICTIONS_FROZEN.json in {eval_dir}: {e}")

        expected_recipe_sha = frozen_data.get("recipe_sha256")
        if not expected_recipe_sha:
            raise ValueError(f"PREDICTIONS_FROZEN.json missing 'recipe_sha256': {eval_dir}")
        validate_sha256_format(expected_recipe_sha, "recipe_sha256")

        actual_recipe_sha = digest(recipe_path)
        if actual_recipe_sha != expected_recipe_sha:
            raise ValueError(
                f"RECIPE.json hash mismatch: expected {expected_recipe_sha}, got {actual_recipe_sha} ({eval_dir})"
            )

        recipe = json.loads(recipe_path.read_text())
        rec_ckpt_sha = recipe.get("checkpoint_sha256")
        if not rec_ckpt_sha:
            raise ValueError(f"RECIPE.json missing 'checkpoint_sha256': {recipe_path}")
        validate_sha256_format(rec_ckpt_sha, "checkpoint_sha256")

        # Checkpoint SHA consistency
        if checkpoint_sha is not None and rec_ckpt_sha != checkpoint_sha:
            raise ValueError(
                f"checkpoint SHA mismatch: expected {checkpoint_sha}, got {rec_ckpt_sha} in {recipe_path}"
            )
        if common_ckpt_sha is None:
            common_ckpt_sha = rec_ckpt_sha
        elif common_ckpt_sha != rec_ckpt_sha:
            raise ValueError(
                f"checkpoint SHA mismatch across evaluation packages: {common_ckpt_sha} vs {rec_ckpt_sha}"
            )

        # Mandatory checkpoint payload SHA: must be present in BOTH recipe and frozen receipt and match
        payload_sha = recipe.get("checkpoint_payload_sha256")
        if not payload_sha:
            raise ValueError(f"RECIPE.json missing mandatory 'checkpoint_payload_sha256': {recipe_path}")
        validate_sha256_format(payload_sha, "checkpoint_payload_sha256 in RECIPE")

        frozen_payload_sha = frozen_data.get("checkpoint_payload_sha256")
        if not frozen_payload_sha:
            raise ValueError(f"PREDICTIONS_FROZEN.json missing mandatory 'checkpoint_payload_sha256': {frozen_receipt}")
        validate_sha256_format(frozen_payload_sha, "checkpoint_payload_sha256 in PREDICTIONS_FROZEN")

        if payload_sha != frozen_payload_sha:
            raise ValueError(
                f"payload SHA mismatch between recipe ({payload_sha}) and frozen receipt ({frozen_payload_sha}) in {eval_dir}"
            )

        if common_payload_sha is None:
            common_payload_sha = payload_sha
        elif common_payload_sha != payload_sha:
            raise ValueError(
                f"checkpoint payload SHA mismatch across packages: {common_payload_sha} vs {payload_sha}"
            )

        # Record manifest SHAs declared in RECIPE
        if "domains" in recipe and isinstance(recipe["domains"], dict):
            for dom, dinfo in recipe["domains"].items():
                if isinstance(dinfo, dict) and "manifest_sha256" in dinfo:
                    msha = dinfo["manifest_sha256"]
                    validate_sha256_format(msha, f"manifest_sha256 for domain {dom}")
                    recipe_manifest_shas[msha] = dom
        elif "data_sha256" in recipe:
            msha = recipe["data_sha256"]
            validate_sha256_format(msha, "data_sha256 in RECIPE")
            scope_dom = recipe.get("scope", "default")
            recipe_manifest_shas[msha] = scope_dom

        # Build verified prediction map keyed strictly by (domain, pid)
        verified_preds = {}
        pred_dir = eval_dir / "predictions"
        if not pred_dir.is_dir():
            pred_dir = eval_dir

        if "domains" in frozen_data and isinstance(frozen_data["domains"], dict):
            for dom, entries in frozen_data["domains"].items():
                validate_safe_identifier(dom, "domain in frozen predictions")
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict) or "id" not in entry or "file" not in entry or "sha256" not in entry:
                        raise ValueError(f"invalid prediction file record in {frozen_receipt}: {entry}")
                    pid = entry["id"]
                    validate_safe_identifier(pid, "patient id in frozen predictions")
                    validate_sha256_format(entry["sha256"], f"prediction SHA for {dom}:{pid}")

                    key = (dom, pid)
                    if key in verified_preds:
                        raise ValueError(f"duplicate prediction entry for (domain={dom}, patient={pid})")

                    pfile = entry["file"]
                    if pfile.startswith("/") or pfile.startswith("\\") or ".." in pfile.replace("\\", "/").split("/"):
                        raise ValueError(f"illegal traversal in prediction filename: {pfile}")

                    actual_pred_path = (pred_dir / pfile).resolve()
                    if not actual_pred_path.is_relative_to(pred_dir.resolve()):
                        raise ValueError(f"prediction file path escapes prediction directory: {pfile}")
                    if not actual_pred_path.is_file():
                        raise FileNotFoundError(f"frozen prediction file does not exist: {actual_pred_path}")

                    actual_pred_sha = digest(actual_pred_path)
                    if actual_pred_sha != entry["sha256"]:
                        raise ValueError(
                            f"frozen prediction hash mismatch for {dom}:{pid}: "
                            f"expected {entry['sha256']}, got {actual_pred_sha}"
                        )
                    verified_preds[key] = actual_pred_path
        elif "files" in frozen_data and isinstance(frozen_data["files"], list):
            dom = recipe.get("scope", "default")
            for entry in frozen_data["files"]:
                if not isinstance(entry, dict) or "id" not in entry or "file" not in entry or "sha256" not in entry:
                    raise ValueError(f"invalid prediction file record in {frozen_receipt}: {entry}")
                pid = entry["id"]
                validate_safe_identifier(pid, "patient id in frozen predictions")
                validate_sha256_format(entry["sha256"], f"prediction SHA for {pid}")

                key = (dom, pid)
                if key in verified_preds:
                    raise ValueError(f"duplicate prediction entry for (domain={dom}, patient={pid})")

                pfile = entry["file"]
                if pfile.startswith("/") or pfile.startswith("\\") or ".." in pfile.replace("\\", "/").split("/"):
                    raise ValueError(f"illegal traversal in prediction filename: {pfile}")

                actual_pred_path = (pred_dir / pfile).resolve()
                if not actual_pred_path.is_relative_to(pred_dir.resolve()):
                    raise ValueError(f"prediction file path escapes prediction directory: {pfile}")
                if not actual_pred_path.is_file():
                    raise FileNotFoundError(f"frozen prediction file does not exist: {actual_pred_path}")

                actual_pred_sha = digest(actual_pred_path)
                if actual_pred_sha != entry["sha256"]:
                    raise ValueError(
                        f"frozen prediction hash mismatch for {dom}:{pid}: "
                        f"expected {entry['sha256']}, got {actual_pred_sha}"
                    )
                verified_preds[key] = actual_pred_path
        else:
            raise ValueError(f"PREDICTIONS_FROZEN.json has no valid 'domains' or 'files' records: {eval_dir}")

        # Strict METRICS.json binding check:
        # Checkpoint manifest, payload, and recipe hashes MUST match actual package before displaying measured score
        metrics_data = None
        metrics_sha = None
        metrics_bound = False

        if metrics_path.is_file():
            metrics_sha = digest(metrics_path)
            raw_metrics = json.loads(metrics_path.read_text())

            m_ckpt_m = raw_metrics.get("checkpoint_manifest_sha256")
            m_ckpt_s = raw_metrics.get("checkpoint_sha256")
            m_payload = raw_metrics.get("checkpoint_payload_sha256") or raw_metrics.get("payload_sha256")
            m_recipe = raw_metrics.get("recipe_sha256")
            m_frozen = raw_metrics.get("frozen_predictions_sha256")
            actual_frozen_sha = digest(frozen_receipt)

            # Missing or mismatched binding marks metrics unbound
            if not m_ckpt_m and not m_ckpt_s:
                metrics_bound = False
            elif m_ckpt_m and m_ckpt_m != rec_ckpt_sha:
                metrics_bound = False
            elif m_ckpt_s and m_ckpt_s != rec_ckpt_sha:
                metrics_bound = False
            elif not m_payload or m_payload != payload_sha:
                metrics_bound = False
            elif not m_recipe or m_recipe != actual_recipe_sha:
                metrics_bound = False
            elif m_frozen and m_frozen != actual_frozen_sha:
                metrics_bound = False
            else:
                metrics_bound = True
                metrics_data = raw_metrics

        eval_packages.append({
            "eval_dir": eval_dir,
            "recipe": recipe,
            "recipe_sha": actual_recipe_sha,
            "checkpoint_sha": rec_ckpt_sha,
            "payload_sha": payload_sha,
            "predictions": verified_preds,
            "metrics": metrics_data,
            "metrics_bound": metrics_bound,
            "metrics_sha": metrics_sha,
            "metrics_path": metrics_path if metrics_path.is_file() else None,
            "synthetic": bool(recipe.get("synthetic", False)),
        })

    # 2. Strict verification of manifests BEFORE opening ANY reference mask
    manifest_records = []
    is_synthetic_overall = False

    for m_path in manifests:
        m_path = m_path.resolve()
        if not m_path.is_file():
            raise FileNotFoundError(f"manifest file not found: {m_path}")

        m_sha = digest(m_path)
        try:
            m_data = json.loads(m_path.read_text())
        except Exception as e:
            raise ValueError(f"malformed manifest JSON in {m_path}: {e}")

        # Validate schema fail-closed: support ONLY actual documented schemas
        schema = m_data.get("schema")
        if schema not in VALID_MANIFEST_SCHEMAS:
            raise ValueError(
                f"unsupported manifest schema: {schema!r}; expected one of {sorted(VALID_MANIFEST_SCHEMAS)}"
            )

        # Validate evaluation role: NEVER allow training manifest to masquerade as evaluation
        role = m_data.get("role")
        if role not in VALID_EVALUATION_ROLES:
            raise ValueError(
                f"manifest role {role!r} is not an allowed evaluation role ({sorted(VALID_EVALUATION_ROLES)}); "
                f"refusing to open masks"
            )

        # Match calculated manifest_sha against corresponding RECIPE.domains[domain].manifest_sha256
        # Derive domain from MATCHED manifest hash, not heuristic pid/scope
        if m_sha not in recipe_manifest_shas:
            raise ValueError(
                f"manifest {m_path} (SHA-256: {m_sha}) does not match any domain manifest_sha256 "
                f"in evaluation recipes: {recipe_manifest_shas}"
            )
        matched_domain = recipe_manifest_shas[m_sha]

        patients = m_data.get("patients")
        if not isinstance(patients, list) or not patients:
            raise ValueError(f"manifest has no patients list: {m_path}")

        synth = bool(m_data.get("synthetic", False))
        if synth:
            is_synthetic_overall = True

        manifest_records.append({
            "manifest_path": m_path,
            "manifest_sha": m_sha,
            "manifest_data": m_data,
            "domain": matched_domain,
            "patients": patients,
            "synthetic": synth,
        })

    # 3. Patient validation, mandatory image/mask SHA-256, and metadata frame bounds check
    # Check all manifest patients are mapped exactly with NO silent continue
    dataset_records = {}

    for m_info in manifest_records:
        domain = m_info["domain"]
        m_parent = m_info["manifest_path"].parent

        for p in m_info["patients"]:
            pid = p.get("id")
            if not pid:
                raise ValueError("patient record missing 'id'")
            validate_safe_identifier(pid, "patient id in manifest")

            # Look for frozen prediction across eval packages by (domain, pid)
            key = (domain, pid)
            matched_eval = None
            pred_file_path = None
            for epkg in eval_packages:
                if key in epkg["predictions"]:
                    matched_eval = epkg
                    pred_file_path = epkg["predictions"][key]
                    break

            # Require all manifest patients mapped exactly, no silent continue
            if matched_eval is None or pred_file_path is None:
                raise ValueError(
                    f"missing frozen prediction for patient {pid!r} in domain {domain!r}; "
                    f"refusing to proceed (no silent continue allowed)"
                )

            # Image SHA is MANDATORY
            img_rel = p.get("image")
            if not img_rel:
                raise ValueError(f"patient {pid} has no image specified")
            img_path = Path(img_rel) if Path(img_rel).is_absolute() else m_parent / img_rel
            if not img_path.is_file():
                raise FileNotFoundError(f"cine image file not found for {pid}: {img_path}")

            expected_img_sha = p.get("image_sha256")
            if not expected_img_sha:
                raise ValueError(f"mandatory 'image_sha256' missing for patient {pid} in {m_info['manifest_path']}")
            validate_sha256_format(expected_img_sha, f"image_sha256 for {pid}")

            actual_img_sha = digest(img_path)
            if actual_img_sha != expected_img_sha:
                raise ValueError(
                    f"image hash mismatch for patient {pid}: expected {expected_img_sha}, got {actual_img_sha}"
                )

            # Load image and check exact shape 4D (T, Z, H, W)
            cine_img = np.load(img_path, mmap_mode="r", allow_pickle=False)
            if not np.isfinite(cine_img).all():
                raise ValueError(f"cine image contains non-finite values: {img_path}")
            if cine_img.ndim != 4:
                raise ValueError(f"expected 4D cine image (T, Z, H, W), got {cine_img.shape} for {img_path}")
            t_len, z_len, h_len, w_len = cine_img.shape

            # Add to protected paths
            protected_paths.append(img_path)

            masks = p.get("masks", {})
            if not isinstance(masks, dict) or not masks:
                raise ValueError(f"patient {pid} has missing or empty 'masks' dict")

            # Validate frame bounds BEFORE opening any mask
            for frame_str, minfo in masks.items():
                try:
                    f_idx = int(frame_str)
                except ValueError:
                    raise ValueError(f"invalid non-integer frame index {frame_str!r} for patient {pid}")
                if f_idx < 0 or f_idx >= t_len:
                    raise ValueError(
                        f"frame index {f_idx} out of bounds for image with T={t_len} (patient {pid})"
                    )

                # Mask SHA is MANDATORY
                m_sha = minfo.get("sha256")
                if not m_sha:
                    raise ValueError(f"mandatory 'sha256' missing for mask of patient {pid} frame {f_idx}")
                validate_sha256_format(m_sha, f"mask sha256 for {pid} frame {f_idx}")

                m_rel = minfo.get("path")
                if not m_rel:
                    raise ValueError(f"mask path missing for patient {pid} frame {f_idx}")
                mp = Path(m_rel) if Path(m_rel).is_absolute() else m_parent / m_rel
                protected_paths.append(mp)

            if domain not in dataset_records:
                dataset_records[domain] = []

            dataset_records[domain].append({
                "patient_id": pid,
                "domain": domain,
                "image_path": img_path,
                "image_shape": (t_len, z_len, h_len, w_len),
                "pred_path": pred_file_path,
                "masks": masks,
                "manifest_parent": m_parent,
                "eval_package": matched_eval,
            })

    # Protect all resolved image/mask/input parents not just manifest dirs
    validate_output_directory(out_dir, protected_paths)

    # 4. Render learning curves (if train_log provided)
    learning_curves_info = None
    learning_curves_path = None
    if train_log is not None:
        lc_out = out_dir / "learning_curves.png"
        learning_curves_info = render_learning_curves(train_log, lc_out, synthetic=is_synthetic_overall)
        learning_curves_path = lc_out.name

    # 5. Deterministic Overlay Generation per Dataset
    overlays_summary = []

    for domain, records in sorted(dataset_records.items(), key=lambda x: x[0]):
        candidate_frames = []
        for r in records:
            epkg = r["eval_package"]
            metrics = epkg["metrics"]
            pid = r["patient_id"]

            for frame_str, minfo in sorted(r["masks"].items(), key=lambda x: int(x[0])):
                frame_idx = int(frame_str)
                frame_dice = None
                frame_metrics_str = None

                # Extract accurate score if metrics strictly binds verified checkpoint and recipe
                if metrics:
                    dom_dict = (
                        metrics.get("domains", {}).get(domain)
                        or metrics.get("domains", {}).get(domain.lower())
                        or metrics.get("domains", {}).get(domain.upper())
                        or {}
                    )
                    dom_frames = dom_dict.get("frames", [])
                    global_frames = metrics.get("frames", [])
                    search_frames = dom_frames if dom_frames else global_frames

                    for frow in search_frames:
                        if frow.get("patient") == pid and frow.get("frame") == frame_idx:
                            raw_metrics = frow.get("raw")
                            if isinstance(raw_metrics, list) and len(raw_metrics) >= 4:
                                dice_vals = [
                                    c.get("dice") for c in raw_metrics[1:]
                                    if isinstance(c, dict) and c.get("dice") is not None and math.isfinite(c.get("dice"))
                                ]
                                if dice_vals:
                                    frame_dice = float(np.mean(dice_vals))
                                    frame_metrics_str = f"Frame FG Dice: {frame_dice:.3f}"
                            break

                    if frame_metrics_str is None:
                        pat_metrics = dom_dict.get("patient_metrics", {}).get(pid, {})
                        if isinstance(pat_metrics, dict):
                            fg_m = pat_metrics.get("foreground_mean")
                            if fg_m is not None and math.isfinite(fg_m):
                                frame_dice = float(fg_m)
                                frame_metrics_str = f"Patient FG Mean Dice: {frame_dice:.3f}"

                candidate_frames.append({
                    "record": r,
                    "frame_idx": frame_idx,
                    "mask_info": minfo,
                    "frame_dice": frame_dice if frame_dice is not None else float("inf"),
                    "has_dice": frame_dice is not None,
                    "frame_metrics_str": frame_metrics_str,
                })

        # Deterministic sorting:
        # If evaluated score available, sort by (frame_dice, pid, frame_idx) for worst-case inspection
        # Otherwise sort by (pid, frame_idx)
        candidate_frames.sort(
            key=lambda x: (x["frame_dice"], x["record"]["patient_id"], x["frame_idx"])
        )

        chosen = candidate_frames[:max_overlays_per_dataset]

        for item in chosen:
            r = item["record"]
            pid = r["patient_id"]
            frame_idx = item["frame_idx"]
            minfo = item["mask_info"]
            m_parent = r["manifest_parent"]
            t_len, z_len, h_len, w_len = r["image_shape"]

            # Open mask and verify hash
            m_rel = minfo["path"]
            mask_path = Path(m_rel) if Path(m_rel).is_absolute() else m_parent / m_rel
            if not mask_path.is_file():
                raise FileNotFoundError(f"reference mask file not found: {mask_path}")

            actual_m_sha = digest(mask_path)
            if actual_m_sha != minfo["sha256"]:
                raise ValueError(
                    f"reference mask hash mismatch for {pid} frame {frame_idx}: "
                    f"expected {minfo['sha256']}, got {actual_m_sha}"
                )

            # Load mask
            gt_mask = np.load(mask_path, allow_pickle=False)
            if not np.isfinite(gt_mask).all():
                raise ValueError(f"non-finite values in reference mask: {mask_path}")
            if gt_mask.ndim != 3 or gt_mask.shape != (z_len, h_len, w_len):
                raise ValueError(
                    f"reference mask shape {gt_mask.shape} must match image (Z, H, W)=({z_len}, {h_len}, {w_len})"
                )

            # Load prediction with strict shape and label validation
            pred_vol = load_prediction_file(r["pred_path"], t_len, z_len, h_len, w_len)

            # Load cine slice
            cine_img = np.load(r["image_path"], mmap_mode="r", allow_pickle=False)

            # Deterministic slice selection: slice with maximum foreground area in GT
            fg_voxels = ((gt_mask > 0) & (gt_mask < 4) & (gt_mask != 255))
            if fg_voxels.any():
                slice_idx = int(fg_voxels.sum(axis=(1, 2)).argmax())
            else:
                slice_idx = z_len // 2

            image_slice = cine_img[frame_idx, slice_idx]
            gt_slice = gt_mask[slice_idx]
            pred_slice = pred_vol[frame_idx, slice_idx]

            # Path-safe overlay filename: replace colons or punctuation with underscores
            clean_dom = re.sub(r"[^A-Za-z0-9_\-]", "_", domain)
            clean_pid = re.sub(r"[^A-Za-z0-9_\-]", "_", pid)
            overlay_fname = f"overlay_{clean_dom}_{clean_pid}_f{frame_idx}_s{slice_idx}.png"
            overlay_out = out_dir / overlay_fname

            render_overlay_figure(
                image_slice=image_slice,
                gt_slice=gt_slice,
                pred_slice=pred_slice,
                dataset=domain,
                patient_id=pid,
                frame_idx=frame_idx,
                slice_idx=slice_idx,
                checkpoint_sha=r["eval_package"]["checkpoint_sha"],
                out_file=overlay_out,
                frame_metrics_str=item["frame_metrics_str"],
                synthetic=is_synthetic_overall,
            )

            overlays_summary.append({
                "dataset": domain,
                "patient_id": pid,
                "frame": frame_idx,
                "slice": slice_idx,
                "file": overlay_fname,
                "sha256": digest(overlay_out),
                "checkpoint_sha256": r["eval_package"]["checkpoint_sha"],
                "has_measured_dice": item["has_dice"],
                "measured_fg_dice": item["frame_dice"] if item["has_dice"] else None,
            })

    # 6. Build durable provenance
    metrics_pointers = {}
    for epkg in eval_packages:
        scope = epkg["recipe"].get("scope", epkg["eval_dir"].name)
        metrics_pointers[scope] = {
            "evaluation_dir": str(epkg["eval_dir"]),
            "recipe_sha256": epkg["recipe_sha"],
            "checkpoint_sha256": epkg["checkpoint_sha"],
            "checkpoint_payload_sha256": epkg["payload_sha"],
            "metrics_bound": epkg["metrics_bound"],
            "metrics_file": str(epkg["metrics_path"]) if epkg["metrics_path"] else None,
            "metrics_sha256": epkg["metrics_sha"],
            "macro_patient": epkg["metrics"].get("macro_patient") if (epkg["metrics"] and epkg["metrics_bound"]) else None,
        }

    provenance = {
        "schema": "self_audit.recovery_report.v1",
        "checkpoint_sha256": common_ckpt_sha,
        "checkpoint_payload_sha256": common_payload_sha,
        "datasets": sorted(dataset_records.keys()),
        "learning_curves": {
            "file": learning_curves_path,
            "sha256": digest(out_dir / learning_curves_path) if learning_curves_path else None,
            "metrics": learning_curves_info,
        } if learning_curves_path else None,
        "overlays": overlays_summary,
        "metrics_pointers": metrics_pointers,
        "canonical_colors": {
            "BG": "transparent",
            "RV": "red (#ff0000)",
            "MYO": "green (#00cc00)",
            "LV": "blue (#0033ff)",
            "ignore_255": "transparent",
        },
        "synthetic": is_synthetic_overall,
    }

    write_json(out_dir / "PROVENANCE.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-log", "--train-run", "--steps-jsonl",
        dest="train_log",
        type=Path,
        help="Path to training steps.jsonl or training run directory",
    )
    parser.add_argument(
        "--evaluation", "--evaluations",
        dest="evaluations",
        type=Path,
        nargs="*",
        default=[],
        help="One or more evaluation artifact directories containing PREDICTIONS_FROZEN.json",
    )
    parser.add_argument(
        "--manifest", "--manifests", "--data",
        dest="manifests",
        type=Path,
        nargs="*",
        default=[],
        help="One or more evaluation reference manifest JSON files",
    )
    parser.add_argument(
        "--out",
        dest="out",
        type=Path,
        required=True,
        help="Brand new output directory for generated figures and report provenance",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        dest="checkpoint_sha256",
        type=str,
        default=None,
        help="Expected checkpoint SHA-256 for strict lineage verification",
    )
    parser.add_argument(
        "--max-overlays",
        dest="max_overlays",
        type=int,
        default=3,
        help="Maximum number of deterministic overlays per dataset (positive integer, default: 3)",
    )

    args = parser.parse_args()

    train_log = None
    if args.train_log is not None:
        p = args.train_log.resolve()
        if p.is_dir() and (p / "steps.jsonl").is_file():
            train_log = p / "steps.jsonl"
        elif p.is_file():
            train_log = p
        else:
            raise FileNotFoundError(f"invalid training log or directory: {args.train_log}")

    expanded_evals = []
    for ed in args.evaluations:
        ed = ed.resolve()
        if (ed / "PREDICTIONS_FROZEN.json").is_file():
            expanded_evals.append(ed)
        else:
            subdirs = [sd for sd in ed.iterdir() if sd.is_dir() and (sd / "PREDICTIONS_FROZEN.json").is_file()]
            if subdirs:
                expanded_evals.extend(subdirs)
            else:
                expanded_evals.append(ed)

    prov = render_recovery_report(
        out_dir=args.out,
        train_log=train_log,
        evaluations=expanded_evals,
        manifests=args.manifests,
        checkpoint_sha=args.checkpoint_sha256,
        max_overlays_per_dataset=args.max_overlays,
    )

    print(json.dumps({
        "status": "COMPLETED",
        "output_directory": str(args.out.resolve()),
        "datasets": prov["datasets"],
        "overlays_count": len(prov["overlays"]),
        "has_learning_curves": prov["learning_curves"] is not None,
    }, indent=2))


if __name__ == "__main__":
    main()
