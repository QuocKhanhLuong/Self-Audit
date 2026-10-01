#!/usr/bin/env python
"""Evaluate and visualize shared-benchmark semantic artifacts on Self-Audit ACDC."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from self_audit.evaluation.contracts import (
    compute_dice_from_stats,
    compute_sufficient_statistics,
    resolve_metric_contract,
    score_volume_from_stats,
)


CLASS_NAMES = ["Background", "RV", "MYO", "LV"]
FOREGROUND_CLASSES = [1, 2, 3]
SEMANTIC_COLORS = np.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.90, 0.26, 0.21, 0.75],
        [0.13, 0.59, 0.95, 0.75],
        [0.30, 0.69, 0.31, 0.75],
        [0.95, 0.77, 0.06, 0.85],
    ],
    dtype=np.float32,
)
ERROR_COLORS = np.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.85, 0.20, 0.20, 0.85],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class SampleEvaluation:
    sample_id: str
    patient_id: str
    volume_key: str
    volume_id: str
    slice_index: int
    image: np.ndarray
    gt: np.ndarray
    raw_partition: np.ndarray
    semantic_map: np.ndarray
    validity_map: np.ndarray
    slice_proxy_foreground_macro_dice: float
    slice_proxy_per_class_dice: dict[str, float | None]
    coverage: float


def _safe_float(value: float) -> float | None:
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def _surface_distances(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
    pred_border = pred_mask ^ binary_erosion(pred_mask, iterations=1)
    gt_border = gt_mask ^ binary_erosion(gt_mask, iterations=1)
    if not np.any(pred_border) and not np.any(gt_border):
        return np.array([0.0], dtype=np.float64)
    if not np.any(pred_border) or not np.any(gt_border):
        return np.array([float("inf")], dtype=np.float64)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
    return np.concatenate([dt_pred[gt_border], dt_gt[pred_border]])


def _hd95(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple[float, ...]) -> float:
    distances = _surface_distances(pred_mask, gt_mask, spacing)
    if not len(distances):
        return float("nan")
    if not np.all(np.isfinite(distances)):
        return float("inf")
    return float(np.percentile(distances, 95))


def _assd(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple[float, ...]) -> float:
    distances = _surface_distances(pred_mask, gt_mask, spacing)
    if not len(distances):
        return float("nan")
    if not np.all(np.isfinite(distances)):
        return float("inf")
    return float(np.mean(distances))


def _normalize_image(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    low = float(np.percentile(value, 1.0))
    high = float(np.percentile(value, 99.0))
    if not high > low:
        high = low + 1.0
    return np.clip((value - low) / (high - low), 0.0, 1.0)


def _overlay(base: np.ndarray, label_map: np.ndarray, *, alpha: float = 0.85, is_error: bool = False) -> np.ndarray:
    base_rgb = np.stack([base, base, base], axis=-1)
    cmap = ERROR_COLORS if is_error else SEMANTIC_COLORS
    rgba = cmap[label_map]
    out = base_rgb.copy()
    weight = rgba[..., 3:4] * alpha
    out = out * (1.0 - weight) + rgba[..., :3] * weight
    return np.clip(out, 0.0, 1.0)


def _read_info_cfg(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if key in {"ED", "ES"}:
            mapping[int(value)] = key
    if not mapping:
        raise ValueError(f"missing ED/ES mapping in {path}")
    return mapping


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    return payload


def _collect_artifacts(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    raw_by_sample: dict[str, Path] = {}
    semantic_by_sample: dict[str, Path] = {}
    for metadata_path in sorted(root.rglob("metadata.json")):
        directory = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sample_id = str(metadata["sample_id"])
        if (directory / "raw_partition.npy").is_file():
            raw_by_sample[sample_id] = directory
        if (directory / "semantic_map.npy").is_file():
            semantic_by_sample[sample_id] = directory
    return raw_by_sample, semantic_by_sample


def _sample_to_volume_key(
    record: dict[str, Any],
    *,
    source_root: Path,
    phase_cache: dict[str, dict[int, str]],
) -> str:
    patient_id = str(record["patient_id"])
    if patient_id not in phase_cache:
        phase_cache[patient_id] = _read_info_cfg(source_root / "training" / patient_id / "Info.cfg")
    frame_index = int(record["frame_index"])
    phase = phase_cache[patient_id].get(frame_index)
    if phase is None:
        raise ValueError(f"frame_index={frame_index} is not ED/ES for {patient_id}")
    return f"{patient_id}_{phase}"


def _named_foreground_dice(per_class: dict[int, float]) -> dict[str, float | None]:
    return {CLASS_NAMES[c]: _safe_float(float(per_class.get(c, float("nan")))) for c in FOREGROUND_CLASSES}


def evaluate_artifacts(
    artifact_root: Path,
    manifest_path: Path,
    preprocessed_root: Path,
    source_root: Path,
    sample_list_path: Path | None = None,
) -> dict[str, Any]:
    slice_contract = resolve_metric_contract("foreground_dice_exclude_v1")
    volume_contract = resolve_metric_contract("foreground_dice_volume_resized_v1")
    manifest = _load_manifest(manifest_path)
    records_by_id = {str(record["sample_id"]): dict(record) for record in manifest["records"]}
    metadata = json.loads((preprocessed_root / "metadata.json").read_text(encoding="utf-8"))
    volume_info = dict(metadata["volume_info"])
    raw_by_sample, semantic_by_sample = _collect_artifacts(artifact_root)
    if not semantic_by_sample:
        raise ValueError(f"no semantic artifacts found in {artifact_root}")
    if sample_list_path is not None:
        selected = {
            line.strip()
            for line in sample_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        raw_by_sample = {sample_id: path for sample_id, path in raw_by_sample.items() if sample_id in selected}
        semantic_by_sample = {sample_id: path for sample_id, path in semantic_by_sample.items() if sample_id in selected}
        if not semantic_by_sample:
            raise ValueError(f"no semantic artifacts matched sample list: {sample_list_path}")

    phase_cache: dict[str, dict[int, str]] = {}
    image_cache: dict[str, np.ndarray] = {}
    mask_cache: dict[str, np.ndarray] = {}
    sample_rows: list[SampleEvaluation] = []
    volume_rows: dict[str, list[SampleEvaluation]] = defaultdict(list)

    for sample_id, semantic_dir in sorted(semantic_by_sample.items()):
        if sample_id not in records_by_id:
            raise ValueError(f"artifact sample not present in manifest: {sample_id}")
        if sample_id not in raw_by_sample:
            raise ValueError(f"missing raw artifact for semantic sample: {sample_id}")

        record = records_by_id[sample_id]
        volume_key = _sample_to_volume_key(record, source_root=source_root, phase_cache=phase_cache)
        if volume_key not in image_cache:
            image_cache[volume_key] = np.load(preprocessed_root / "volumes" / f"{volume_key}.npy")
            mask_cache[volume_key] = np.load(preprocessed_root / "masks" / f"{volume_key}.npy")
        image_volume = image_cache[volume_key]
        mask_volume = mask_cache[volume_key]
        z = int(record["slice_index"])

        semantic = np.load(semantic_dir / "semantic_map.npy")
        validity = np.load(semantic_dir / "validity_map.npy")
        raw_partition = np.load(raw_by_sample[sample_id] / "raw_partition.npy")
        image = np.asarray(image_volume[:, :, z], dtype=np.float32)
        gt = np.asarray(mask_volume[:, :, z], dtype=np.uint8)

        slice_stats = compute_sufficient_statistics(semantic, gt, classes=slice_contract.classes)
        slice_per_class, slice_macro = compute_dice_from_stats(slice_stats, slice_contract)

        row = SampleEvaluation(
            sample_id=sample_id,
            patient_id=str(record["patient_id"]),
            volume_key=volume_key,
            volume_id=str(record["volume_id"]),
            slice_index=z,
            image=image,
            gt=gt,
            raw_partition=raw_partition,
            semantic_map=semantic,
            validity_map=validity,
            slice_proxy_foreground_macro_dice=float(slice_macro),
            slice_proxy_per_class_dice=_named_foreground_dice(slice_per_class),
            coverage=float(validity.mean()),
        )
        sample_rows.append(row)
        volume_rows[volume_key].append(row)

    volume_metrics: list[dict[str, Any]] = []
    coverage_values: list[float] = []
    foreground_values: list[float] = []
    for volume_key, rows in sorted(volume_rows.items()):
        ordered = sorted(rows, key=lambda row: row.slice_index)
        pred = np.stack([row.semantic_map for row in ordered], axis=0)
        gt = np.stack([row.gt for row in ordered], axis=0)
        info = volume_info[volume_key]
        spacing = tuple(float(v) for v in info["effective_spacing"])
        volume_stats = [
            compute_sufficient_statistics(row.semantic_map, row.gt, classes=volume_contract.classes)
            for row in ordered
        ]
        volume_score = score_volume_from_stats(volume_stats, volume_contract)
        hd95 = {
            CLASS_NAMES[c]: _safe_float(_hd95(pred == c, gt == c, spacing))
            for c in FOREGROUND_CLASSES
        }
        assd = {
            CLASS_NAMES[c]: _safe_float(_assd(pred == c, gt == c, spacing))
            for c in FOREGROUND_CLASSES
        }
        coverage = float(np.mean([row.coverage for row in ordered]))
        volume_macro = _safe_float(volume_score.macro_dice)
        coverage_values.append(coverage)
        if volume_macro is not None:
            foreground_values.append(volume_macro)
        volume_metrics.append(
            {
                "volume_key": volume_key,
                "patient_id": ordered[0].patient_id,
                "slice_count": len(ordered),
                "metric_contract": volume_contract.name,
                "metric_space": volume_contract.metric_space,
                "coverage_mean": coverage,
                "foreground_macro_dice": volume_macro,
                "foreground_per_class_dice": _named_foreground_dice(volume_score.per_class_dice),
                "hd95_per_class_mm": hd95,
                "assd_per_class_mm": assd,
                "spacing_zyx_mm": [float(v) for v in spacing],
            }
        )

    summary = {
        "metric_contract": slice_contract.name,
        "metric_space": slice_contract.metric_space,
        "volume_metric_contract": volume_contract.name,
        "volume_metric_space": volume_contract.metric_space,
        "sample_count": len(sample_rows),
        "volume_count": len(volume_metrics),
        "coverage_mean": _safe_float(np.mean(coverage_values)) if coverage_values else None,
        "coverage_min": _safe_float(np.min(coverage_values)) if coverage_values else None,
        "coverage_max": _safe_float(np.max(coverage_values)) if coverage_values else None,
        "final_foreground_macro_dice": _safe_float(np.mean(foreground_values)) if foreground_values else None,
    }

    return {
        "manifest_hash": manifest["manifest_hash"],
        "summary": summary,
        "sample_metrics": [
            {
                "sample_id": row.sample_id,
                "patient_id": row.patient_id,
                "volume_key": row.volume_key,
                "volume_id": row.volume_id,
                "slice_index": row.slice_index,
                "coverage": row.coverage,
                "metric_contract": slice_contract.name,
                "metric_space": slice_contract.metric_space,
                "foreground_macro_dice": _safe_float(row.slice_proxy_foreground_macro_dice),
                "foreground_per_class_dice": row.slice_proxy_per_class_dice,
                "raw_cluster_cardinality": int(np.unique(row.raw_partition).size),
            }
            for row in sorted(sample_rows, key=lambda row: (row.volume_key, row.slice_index))
        ],
        "volume_metrics": volume_metrics,
        "samples": sample_rows,
    }


def _save_summary_plot(report: dict[str, Any], output_dir: Path, baseline_name: str) -> Path:
    metrics = list(report["volume_metrics"])
    labels = [entry["volume_key"].replace("_", "\n") for entry in metrics]
    coverage = [entry["coverage_mean"] for entry in metrics]
    fg_scores = [entry["foreground_macro_dice"] or 0.0 for entry in metrics]
    class_means = []
    for class_name in CLASS_NAMES[1:]:
        values = [entry["foreground_per_class_dice"][class_name] for entry in metrics if entry["foreground_per_class_dice"][class_name] is not None]
        class_means.append(float(np.mean(values)) if values else 0.0)
    hd95_means = []
    for class_name in CLASS_NAMES[1:]:
        values = [entry["hd95_per_class_mm"][class_name] for entry in metrics if entry["hd95_per_class_mm"][class_name] is not None]
        hd95_means.append(float(np.mean(values)) if values else 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    x = np.arange(len(metrics))
    axes[0, 0].bar(x, coverage, color="#1f77b4")
    axes[0, 0].set_title(f"{baseline_name}: adapter coverage by volume")
    axes[0, 0].set_xticks(x, labels, rotation=0, fontsize=8)
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_ylabel("coverage")

    axes[0, 1].bar(x, fg_scores, color="#2ca02c")
    axes[0, 1].set_title(f"{baseline_name}: final_foreground_macro_dice by volume")
    axes[0, 1].set_xticks(x, labels, rotation=0, fontsize=8)
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_ylabel("Dice")

    class_x = np.arange(len(FOREGROUND_CLASSES))
    axes[1, 0].bar(class_x, class_means, color=["#d62728", "#1f77b4", "#2ca02c"])
    axes[1, 0].set_title(f"{baseline_name}: mean foreground Dice per class")
    axes[1, 0].set_xticks(class_x, CLASS_NAMES[1:], rotation=20)
    axes[1, 0].set_ylim(0.0, 1.0)

    hd_x = np.arange(len(FOREGROUND_CLASSES))
    axes[1, 1].bar(hd_x, hd95_means, color=["#d62728", "#1f77b4", "#2ca02c"])
    axes[1, 1].set_title(f"{baseline_name}: mean HD95 per foreground class (mm)")
    axes[1, 1].set_xticks(hd_x, CLASS_NAMES[1:], rotation=20)
    axes[1, 1].set_ylabel("mm")

    path = output_dir / "summary_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_volume_montage(report: dict[str, Any], output_dir: Path, volume_key: str) -> Path:
    rows = [row for row in report["samples"] if row.volume_key == volume_key]
    ordered = sorted(rows, key=lambda row: row.slice_index)
    selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    fig, axes = plt.subplots(len(selected), 5, figsize=(16, 4 * len(selected)), constrained_layout=True)
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row_index, row in enumerate(selected):
        image = _normalize_image(row.image)
        semantic_overlay = _overlay(image, row.semantic_map)
        gt_overlay = _overlay(image, row.gt)
        validity_overlay = _overlay(image, row.validity_map.astype(np.uint8), alpha=0.95, is_error=True)
        error_map = np.asarray(row.semantic_map != row.gt, dtype=np.uint8)
        error_overlay = _overlay(image, error_map, is_error=True)
        panels = [
            ("Image", image, "gray"),
            ("GT", gt_overlay, None),
            ("Semantic", semantic_overlay, None),
            ("Validity", validity_overlay, None),
            ("Error", error_overlay, None),
        ]
        for col_index, (title, panel, cmap) in enumerate(panels):
            ax = axes[row_index, col_index]
            if cmap is None:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap=cmap)
            ax.set_title(title)
            ax.set_axis_off()
        axes[row_index, 0].set_ylabel(
            f"z={row.slice_index}\n"
            f"cov={row.coverage:.3f}\n"
            f"fg={row.slice_proxy_foreground_macro_dice:.3f}\n"
            f"clusters={np.unique(row.raw_partition).size}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=35,
        )
    fig.suptitle(f"{volume_key} representative slices", fontsize=14)
    path = output_dir / f"{volume_key}_montage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_extremes_plot(report: dict[str, Any], output_dir: Path, baseline_name: str) -> Path:
    rows = sorted(
        report["samples"],
        key=lambda row: (
            float("-inf") if np.isnan(row.slice_proxy_foreground_macro_dice) else row.slice_proxy_foreground_macro_dice,
            row.sample_id,
        ),
    )
    selected = rows[:3] + rows[-3:]
    fig, axes = plt.subplots(len(selected), 4, figsize=(14, 3.6 * len(selected)), constrained_layout=True)
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row_index, row in enumerate(selected):
        image = _normalize_image(row.image)
        panels = [
            ("Image", image, "gray"),
            ("GT", _overlay(image, row.gt), None),
            ("Semantic", _overlay(image, row.semantic_map), None),
            ("Error", _overlay(image, np.asarray(row.semantic_map != row.gt, dtype=np.uint8), is_error=True), None),
        ]
        for col_index, (title, panel, cmap) in enumerate(panels):
            ax = axes[row_index, col_index]
            if cmap is None:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap=cmap)
            ax.set_title(title)
            ax.set_axis_off()
        axes[row_index, 0].set_ylabel(
            f"fg={row.slice_proxy_foreground_macro_dice:.3f}\n"
            f"cov={row.coverage:.3f}\n"
            f"{row.patient_id}:z{row.slice_index}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=34,
        )
    fig.suptitle(f"{baseline_name}: lowest/highest foreground_dice_exclude_v1 slices", fontsize=14)
    path = output_dir / "slice_extremes.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report_files(
    report: dict[str, Any],
    *,
    output_dir: Path,
    baseline_name: str,
    artifact_root: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_ready = {key: value for key, value in report.items() if key != "samples"}
    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(json_ready, indent=2, sort_keys=True), encoding="utf-8")

    summary = report["summary"]
    volume_metrics = report["volume_metrics"]
    best = max(
        volume_metrics,
        key=lambda row: float("-inf") if row["foreground_macro_dice"] is None else row["foreground_macro_dice"],
    )
    worst = min(
        volume_metrics,
        key=lambda row: float("inf") if row["foreground_macro_dice"] is None else row["foreground_macro_dice"],
    )
    lines = [
        f"# {baseline_name} Shared-Benchmark Semantic Audit",
        "",
        "## Run Scope",
        f"- Artifact root: `{artifact_root}`",
        f"- Samples: `{summary['sample_count']}`",
        f"- Volumes: `{summary['volume_count']}`",
        "",
        "## Summary",
        f"- Slice metric contract: `{summary['metric_contract']}`",
        f"- Slice metric space: `{summary['metric_space']}`",
        f"- Volume metric contract: `{summary['volume_metric_contract']}`",
        f"- Volume metric space: `{summary['volume_metric_space']}`",
        f"- Mean adapter coverage: `{summary['coverage_mean']}`",
        f"- `final_foreground_macro_dice`: `{summary['final_foreground_macro_dice']}`",
        "",
        "## Volume Highlights",
        f"- Best `foreground_dice_volume_resized_v1`: `{best['volume_key']}` -> `{best['foreground_macro_dice']}`",
        f"- Worst `foreground_dice_volume_resized_v1`: `{worst['volume_key']}` -> `{worst['foreground_macro_dice']}`",
        "",
        "## Visual Outputs",
        "- `summary_metrics.png`",
        "- `slice_extremes.png`",
        "- one `*_montage.png` per volume",
        "",
        "## Notes",
        "- Headline metrics follow the Self-Audit canonical foreground Dice contracts on the resized grid.",
        "- Anonymous raw clusters are retained only as debug metadata; headline visuals focus on semantic cardiac output.",
        "- The scientific runner itself remains GT-free; GT is used only in this downstream audit report.",
        "- VOID pixels stay outside {RV, MYO, LV}; under `foreground_dice_exclude_v1`, both-empty classes are excluded and one-sided empties score 0.0.",
        "",
    ]
    md_path = output_dir / "REPORT.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preprocessed-root", type=Path, default=Path("preprocessed_data/ACDC"))
    parser.add_argument("--source-root", type=Path, default=Path("data/ACDC"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--sample-list", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = evaluate_artifacts(
        artifact_root=args.artifact_root,
        manifest_path=args.manifest,
        preprocessed_root=args.preprocessed_root,
        source_root=args.source_root,
        sample_list_path=args.sample_list,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_plot = _save_summary_plot(report, args.output_dir, args.baseline_name)
    extremes_plot = _save_extremes_plot(report, args.output_dir, args.baseline_name)
    montage_paths = [
        _save_volume_montage(report, args.output_dir, volume_key)
        for volume_key in sorted({row.volume_key for row in report["samples"]})
    ]
    json_path, md_path = _write_report_files(
        report,
        output_dir=args.output_dir,
        baseline_name=args.baseline_name,
        artifact_root=args.artifact_root,
    )
    print(
        json.dumps(
            {
                "baseline": args.baseline_name,
                "artifact_root": str(args.artifact_root),
                "report_json": str(json_path),
                "report_md": str(md_path),
                "summary_plot": str(summary_plot),
                "extremes_plot": str(extremes_plot),
                "montages": [str(path) for path in montage_paths],
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
