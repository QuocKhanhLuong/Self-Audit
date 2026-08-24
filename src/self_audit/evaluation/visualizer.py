"""Visualizer module for Self-Audit training phases, transition auditing, and volume QA.

Provides headless matplotlib figure generation, semantic segmentation overlays,
counterfactual transition heatmaps, multi-turn refinement decision traces,
and export utilities for local files and Weights & Biases.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# Ensure headless backend and cache directories for matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "self_audit_mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# ─── Color Palettes & Labels ──────────────────────────────────────────────────

CLASS_NAMES = {
    0: "Background",
    1: "RV (Right Ventricle)",
    2: "MYO (Myocardium)",
    3: "LV (Left Ventricle)",
}

CLASS_SHORT_NAMES = {
    0: "BG",
    1: "RV",
    2: "MYO",
    3: "LV",
}

# RGBA color mapping for segmentation overlays (alpha is handled during blending)
CLASS_COLORS = {
    0: np.array([0.0, 0.0, 0.0], dtype=np.float32),  # Background (transparent)
    1: np.array([0.2, 0.5, 1.0], dtype=np.float32),  # RV: Blue
    2: np.array([0.1, 0.85, 0.3], dtype=np.float32), # MYO: Green
    3: np.array([1.0, 0.25, 0.25], dtype=np.float32),# LV: Red
}

TRANSITION_NAMES = {
    0: "FIX (Repair)",
    1: "UNCHANGED",
    2: "REGRESS (Error)",
}

TRANSITION_COLORS = {
    0: np.array([0.15, 0.9, 0.25], dtype=np.float32), # FIX: Lime Green
    1: np.array([0.45, 0.45, 0.45], dtype=np.float32), # UNCHANGED: Gray
    2: np.array([0.95, 0.15, 0.15], dtype=np.float32), # REGRESS: Crimson Red
}


# ─── Array / Tensor Helpers ───────────────────────────────────────────────────

def _to_numpy_2d(array_or_tensor: Any) -> np.ndarray:
    """Extract a 2-D float32 or int64 numpy array."""
    if torch.is_tensor(array_or_tensor):
        t = array_or_tensor.detach().cpu()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3:
            if t.shape[0] in (1, 3):  # [1, H, W] or [3, H, W]
                t = t[t.shape[0] // 2]  # center slice
            elif t.shape[0] == 4:     # [4, H, W] logits
                t = t.argmax(dim=0)
            else:
                t = t[0]
        return t.numpy()
    arr = np.asarray(array_or_tensor)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3):
            arr = arr[arr.shape[0] // 2]
        elif arr.shape[0] == 4:
            arr = arr.argmax(axis=0)
        else:
            arr = arr[0]
    return arr


def _normalize_image_display(image: Any) -> np.ndarray:
    """Normalize a 2-D image slice to [0, 1] RGB for display."""
    arr = _to_numpy_2d(image).astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[:, :, 0]
    finite = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(finite, [1.0, 99.0])
    if high > low:
        scaled = np.clip((finite - low) / (high - low), 0.0, 1.0)
    else:
        scaled = np.zeros_like(finite)
    # Convert to 3-channel grayscale RGB [H, W, 3]
    return np.repeat(scaled[:, :, np.newaxis], 3, axis=2)


def render_overlay(
    image: Any,
    mask: Any,
    num_classes: int = 4,
    alpha: float = 0.55,
) -> np.ndarray:
    """Render a semantic cardiac segmentation overlay on a grayscale image slice."""
    base_rgb = _normalize_image_display(image)
    label_map = _to_numpy_2d(mask).astype(np.int64)

    overlay = base_rgb.copy()
    for cls_id in range(1, min(num_classes, len(CLASS_COLORS))):
        cls_mask = (label_map == cls_id)
        if not np.any(cls_mask):
            continue
        color = CLASS_COLORS.get(cls_id, np.array([1.0, 1.0, 0.0], dtype=np.float32))
        overlay[cls_mask] = (1.0 - alpha) * base_rgb[cls_mask] + alpha * color
    return np.clip(overlay, 0.0, 1.0).astype(np.float32)


def render_transition_overlay(
    image: Any,
    transition_map: Any,
    alpha: float = 0.65,
) -> np.ndarray:
    """Render local auditor transitions (FIX=Green, UNCHANGED=Dim, REGRESS=Red)."""
    base_rgb = _normalize_image_display(image)
    t_map = _to_numpy_2d(transition_map).astype(np.int64)

    overlay = base_rgb.copy()
    for trans_id, color in TRANSITION_COLORS.items():
        if trans_id == 1:  # UNCHANGED - slight dimming or leave as is
            continue
        mask = (t_map == trans_id)
        if not np.any(mask):
            continue
        overlay[mask] = (1.0 - alpha) * base_rgb[mask] + alpha * color
    return np.clip(overlay, 0.0, 1.0).astype(np.float32)


def render_difference_map(
    image: Any,
    gt_mask: Any,
    pred_mask: Any,
    alpha: float = 0.6,
) -> np.ndarray:
    """Render confusion/error map: True Positives (Green), False Positives (Yellow/Orange), False Negatives (Red)."""
    base_rgb = _normalize_image_display(image)
    gt = _to_numpy_2d(gt_mask).astype(np.int64) > 0
    pred = _to_numpy_2d(pred_mask).astype(np.int64) > 0

    tp = gt & pred
    fp = (~gt) & pred
    fn = gt & (~pred)

    overlay = base_rgb.copy()
    if np.any(tp):
        overlay[tp] = (1.0 - alpha) * base_rgb[tp] + alpha * np.array([0.2, 0.85, 0.2], dtype=np.float32)  # Green: TP
    if np.any(fp):
        overlay[fp] = (1.0 - alpha) * base_rgb[fp] + alpha * np.array([1.0, 0.6, 0.0], dtype=np.float32)   # Orange: FP
    if np.any(fn):
        overlay[fn] = (1.0 - alpha) * base_rgb[fn] + alpha * np.array([0.95, 0.2, 0.2], dtype=np.float32)  # Red: FN

    return np.clip(overlay, 0.0, 1.0).astype(np.float32)


def _compute_dice_fast(pred: np.ndarray, gt: np.ndarray, num_classes: int = 4) -> dict[int, float]:
    """Compute per-class Dice scores for titles."""
    scores = {}
    for c in range(1, num_classes):
        p_c = (pred == c)
        g_c = (gt == c)
        denom = p_c.sum() + g_c.sum()
        if denom == 0:
            scores[c] = 1.0
        else:
            scores[c] = 2.0 * float((p_c & g_c).sum()) / float(denom)
    return scores


# ─── Figure Plotting: Phase A, Phase B, Phase C ───────────────────────────────

def plot_phase_a_samples(
    images: Any,
    masks: Any,
    initial_preds: Any,
    final_preds: Any,
    *,
    case_ids: Sequence[str] | None = None,
    max_samples: int = 4,
    title: str = "Phase A: Annotation Network Predictions",
) -> plt.Figure:
    """Plot Phase A comparison: [MRI Image, Ground Truth, Initial A0, Final AT, Error Map]."""
    if torch.is_tensor(images):
        images = images.detach().cpu()
    if torch.is_tensor(masks):
        masks = masks.detach().cpu()
    if torch.is_tensor(initial_preds):
        initial_preds = initial_preds.detach().cpu()
    if torch.is_tensor(final_preds):
        final_preds = final_preds.detach().cpu()

    n = min(len(images), max_samples)
    fig, axes = plt.subplots(n, 5, figsize=(18, 3.6 * max(n, 1)), squeeze=False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    headers = ["Input MRI (z)", "Ground Truth", "Initial Pred (A0)", "Final Pred (AT)", "Error Map (AT vs GT)"]

    for col, h in enumerate(headers):
        axes[0, col].set_title(h, fontsize=12, fontweight="bold", pad=8)

    for i in range(n):
        img = images[i]
        gt = _to_numpy_2d(masks[i]).astype(np.int64)
        init_p = _to_numpy_2d(initial_preds[i]).astype(np.int64)
        fin_p = _to_numpy_2d(final_preds[i]).astype(np.int64)
        cid = case_ids[i] if case_ids and i < len(case_ids) else f"Sample {i+1}"

        dice_init = _compute_dice_fast(init_p, gt)
        dice_fin = _compute_dice_fast(fin_p, gt)
        mean_init = np.mean(list(dice_init.values())) if dice_init else 0.0
        mean_fin = np.mean(list(dice_fin.values())) if dice_fin else 0.0

        # Col 0: Grayscale Image
        axes[i, 0].imshow(_normalize_image_display(img))
        axes[i, 0].set_ylabel(f"{cid}", fontsize=11, fontweight="bold")

        # Col 1: Ground Truth Overlay
        axes[i, 1].imshow(render_overlay(img, gt))

        # Col 2: Initial A0
        axes[i, 2].imshow(render_overlay(img, init_p))
        axes[i, 2].set_xlabel(f"A0 Dice: {mean_init:.3f}", fontsize=10)

        # Col 3: Final AT
        axes[i, 3].imshow(render_overlay(img, fin_p))
        axes[i, 3].set_xlabel(f"AT Dice: {mean_fin:.3f} (Δ={mean_fin - mean_init:+.3f})", fontsize=10)

        # Col 4: Difference Map
        axes[i, 4].imshow(render_difference_map(img, gt, fin_p))

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    # Legend
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[1], label="RV"),
        mpatches.Patch(color=CLASS_COLORS[2], label="MYO"),
        mpatches.Patch(color=CLASS_COLORS[3], label="LV"),
        mpatches.Patch(color=[0.2, 0.85, 0.2], label="TP (Green)"),
        mpatches.Patch(color=[1.0, 0.6, 0.0], label="FP (Orange)"),
        mpatches.Patch(color=[0.95, 0.2, 0.2], label="FN (Red)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.01), fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    return fig


def plot_phase_b_transitions(
    images: Any,
    masks: Any,
    previous_states: Any,
    candidate_states: Any,
    local_preds: Any,
    local_targets: Any,
    *,
    delta_qs: Any | None = None,
    delta_dices: Any | None = None,
    case_ids: Sequence[str] | None = None,
    max_samples: int = 4,
    title: str = "Phase B: Counterfactual Transition Auditor Verification",
) -> plt.Figure:
    """Plot Phase B transition auditor outputs vs ground-truth transitions."""
    n = min(len(images), max_samples)
    fig, axes = plt.subplots(n, 6, figsize=(21, 3.6 * max(n, 1)), squeeze=False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    headers = [
        "Input MRI (z)",
        "Ground Truth",
        "Previous State (A_t)",
        "Candidate State (A')",
        "GT Transition (Target)",
        "Auditor Pred (Local Map)",
    ]
    for col, h in enumerate(headers):
        axes[0, col].set_title(h, fontsize=12, fontweight="bold", pad=8)

    for i in range(n):
        img = images[i]
        gt = _to_numpy_2d(masks[i]).astype(np.int64)
        prev_p = _to_numpy_2d(previous_states[i]).astype(np.int64)
        cand_p = _to_numpy_2d(candidate_states[i]).astype(np.int64)
        loc_tgt = _to_numpy_2d(local_targets[i]).astype(np.int64)
        loc_prd = _to_numpy_2d(local_preds[i]).astype(np.int64)
        cid = case_ids[i] if case_ids and i < len(case_ids) else f"Sample {i+1}"

        dq = float(delta_qs[i]) if delta_qs is not None and i < len(delta_qs) else None
        dd = float(delta_dices[i]) if delta_dices is not None and i < len(delta_dices) else None

        axes[i, 0].imshow(_normalize_image_display(img))
        axes[i, 0].set_ylabel(f"{cid}", fontsize=11, fontweight="bold")
        axes[i, 1].imshow(render_overlay(img, gt))
        axes[i, 2].imshow(render_overlay(img, prev_p))
        axes[i, 3].imshow(render_overlay(img, cand_p))
        axes[i, 4].imshow(render_transition_overlay(img, loc_tgt))
        axes[i, 5].imshow(render_transition_overlay(img, loc_prd))

        cap_text = ""
        if dq is not None and dd is not None:
            cap_text = f"Pred ΔQ: {dq:+.3f} | Actual ΔDice: {dd:+.3f}"
        elif dq is not None:
            cap_text = f"Pred ΔQ: {dq:+.3f}"
        elif dd is not None:
            cap_text = f"Actual ΔDice: {dd:+.3f}"

        if cap_text:
            axes[i, 5].set_xlabel(cap_text, fontsize=10, fontweight="bold")

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    legend_patches = [
        mpatches.Patch(color=TRANSITION_COLORS[0], label="FIX (Green)"),
        mpatches.Patch(color=TRANSITION_COLORS[1], label="UNCHANGED (Gray)"),
        mpatches.Patch(color=TRANSITION_COLORS[2], label="REGRESS (Red)"),
        mpatches.Patch(color=CLASS_COLORS[1], label="RV"),
        mpatches.Patch(color=CLASS_COLORS[2], label="MYO"),
        mpatches.Patch(color=CLASS_COLORS[3], label="LV"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.01), fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    return fig


def plot_phase_c_audit_trace(
    images: Any,
    masks: Any,
    initial_logits: Any,
    final_logits: Any,
    *,
    transition_candidates: Sequence[Any] | None = None,
    delta_qs: Sequence[Any] | None = None,
    accepted_turns: Sequence[Sequence[int]] | None = None,
    halted_turns: Sequence[int] | None = None,
    tau_accept: float = 0.0,
    case_ids: Sequence[str] | None = None,
    max_samples: int = 4,
    title: str = "Phase C: Threshold-Controlled Self-Audit Refinement Traces",
) -> plt.Figure:
    """Plot Phase C multi-turn refinement trace showing intermediate candidates and acceptance decisions."""
    n = min(len(images), max_samples)
    num_cols = 5
    fig, axes = plt.subplots(n, num_cols, figsize=(19, 3.6 * max(n, 1)), squeeze=False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    headers = ["Input MRI (z)", "Ground Truth", "Initial State (A0)", "Turn Candidates (A1..)", "Final Accepted (A_final)"]
    for col, h in enumerate(headers):
        axes[0, col].set_title(h, fontsize=12, fontweight="bold", pad=8)

    for i in range(n):
        img = images[i]
        gt = _to_numpy_2d(masks[i]).astype(np.int64)
        init_p = _to_numpy_2d(initial_logits[i]).astype(np.int64)
        fin_p = _to_numpy_2d(final_logits[i]).astype(np.int64)
        cid = case_ids[i] if case_ids and i < len(case_ids) else f"Sample {i+1}"

        dice_init = _compute_dice_fast(init_p, gt)
        dice_fin = _compute_dice_fast(fin_p, gt)
        mean_init = np.mean(list(dice_init.values())) if dice_init else 0.0
        mean_fin = np.mean(list(dice_fin.values())) if dice_fin else 0.0

        axes[i, 0].imshow(_normalize_image_display(img))
        axes[i, 0].set_ylabel(f"{cid}", fontsize=11, fontweight="bold")
        axes[i, 1].imshow(render_overlay(img, gt))
        axes[i, 2].imshow(render_overlay(img, init_p))
        axes[i, 2].set_xlabel(f"A0 Dice: {mean_init:.3f}", fontsize=10)

        # Col 3: Intermediate candidate if available, else difference
        if transition_candidates is not None and i < len(transition_candidates):
            cand = _to_numpy_2d(transition_candidates[i]).astype(np.int64)
            axes[i, 3].imshow(render_overlay(img, cand))
            axes[i, 3].set_xlabel("Candidate Turn 1", fontsize=10)
        else:
            axes[i, 3].imshow(render_difference_map(img, init_p, fin_p))
            axes[i, 3].set_xlabel("Update Map (A_fin vs A0)", fontsize=10)

        # Col 4: Final accepted
        axes[i, 4].imshow(render_overlay(img, fin_p))
        halt_info = f"Halt: T={halted_turns[i]}" if halted_turns and i < len(halted_turns) and halted_turns[i] >= 0 else "Full Turns"
        axes[i, 4].set_xlabel(f"Final Dice: {mean_fin:.3f} | {halt_info}", fontsize=10, fontweight="bold")

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[1], label="RV"),
        mpatches.Patch(color=CLASS_COLORS[2], label="MYO"),
        mpatches.Patch(color=CLASS_COLORS[3], label="LV"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01), fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    return fig


def plot_patient_volume_qa(
    volume_image: Any,
    gt_volume: Any,
    pred_volume: Any,
    *,
    initial_pred_volume: Any | None = None,
    slice_indices: Sequence[int] | None = None,
    patient_id: str = "Patient QA",
    title: str | None = None,
) -> plt.Figure:
    """Plot multi-slice axial QA across a patient volume (Apex, Mid, Base)."""
    img_vol = _to_numpy_2d(volume_image) if not isinstance(volume_image, np.ndarray) or volume_image.ndim != 3 else volume_image
    gt_vol = _to_numpy_2d(gt_volume) if not isinstance(gt_volume, np.ndarray) or gt_volume.ndim != 3 else gt_volume
    pred_vol = _to_numpy_2d(pred_volume) if not isinstance(pred_volume, np.ndarray) or pred_volume.ndim != 3 else pred_volume

    z_slices = img_vol.shape[0] if img_vol.ndim == 3 else 1
    if slice_indices is None:
        if z_slices <= 5:
            selected_slices = list(range(z_slices))
        else:
            selected_slices = np.linspace(1, z_slices - 2, 5, dtype=int).tolist()
    else:
        selected_slices = [int(s) for s in slice_indices if int(s) < z_slices]

    n_slices = len(selected_slices)
    num_cols = 4 if initial_pred_volume is not None else 3
    fig, axes = plt.subplots(n_slices, num_cols, figsize=(4 * num_cols, 3.6 * max(n_slices, 1)), squeeze=False)
    fig.suptitle(title or f"Patient Volume QA: {patient_id}", fontsize=15, fontweight="bold", y=0.98)

    headers = ["Input Slice (z)", "Ground Truth", "Initial (A0)", "Self-Audit (Final)"] if num_cols == 4 else ["Input Slice (z)", "Ground Truth", "Self-Audit Prediction"]
    for col, h in enumerate(headers):
        axes[0, col].set_title(h, fontsize=12, fontweight="bold", pad=8)

    for i, s_idx in enumerate(selected_slices):
        img_s = img_vol[s_idx]
        gt_s = gt_vol[s_idx]
        pred_s = pred_vol[s_idx]

        axes[i, 0].imshow(_normalize_image_display(img_s))
        axes[i, 0].set_ylabel(f"Slice z={s_idx}/{z_slices-1}", fontsize=11, fontweight="bold")
        axes[i, 1].imshow(render_overlay(img_s, gt_s))

        if num_cols == 4 and initial_pred_volume is not None:
            init_s = initial_pred_volume[s_idx]
            axes[i, 2].imshow(render_overlay(img_s, init_s))
            axes[i, 3].imshow(render_overlay(img_s, pred_s))
        else:
            axes[i, 2].imshow(render_overlay(img_s, pred_s))

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    return fig


# ─── Export & Logging Utilities ───────────────────────────────────────────────

def save_figure(
    fig: plt.Figure,
    output_path: str | Path,
    *,
    dpi: int = 150,
    close: bool = True,
) -> str:
    """Save a matplotlib figure safely to disk, creating parent directories."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return str(path)


def log_figures_to_wandb(
    figures_dict: Mapping[str, Any],
    *,
    step: int | None = None,
) -> None:
    """Safely log matplotlib figures or image paths to Weights & Biases."""
    try:
        import wandb
        if not wandb.run:
            return
        log_payload: dict[str, Any] = {}
        for tag, item in figures_dict.items():
            if isinstance(item, plt.Figure):
                log_payload[tag] = wandb.Image(item)
            elif isinstance(item, (str, Path)) and Path(item).is_file():
                log_payload[tag] = wandb.Image(str(item))
        if log_payload:
            wandb.log(log_payload, step=step)
    except Exception:
        pass
