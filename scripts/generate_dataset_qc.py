"""Generate visual overlays for audited CMR-MULTI and CMRxMotion samples."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_audit.data.cmr_multi import CMRMultiAdapter
from self_audit.data.cmrxmotion import CMRxMotionAdapter
from self_audit.data.common import build_25d_triplet


CLASS_COLORS = ListedColormap(
    [(0.0, 0.0, 0.0, 0.0), (0.1, 0.3, 1.0, 0.55), (1.0, 0.8, 0.0, 0.55), (1.0, 0.1, 0.1, 0.55)]
)


def _display_slice(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(value, [1.0, 99.0])
    return np.clip((value - low) / max(float(high - low), 1e-6), 0.0, 1.0)


def _save_overlay(
    image_zhw: np.ndarray,
    mask_zhw: np.ndarray,
    *,
    z: int,
    title: str,
    output: Path,
) -> None:
    context = build_25d_triplet(np.asarray(image_zhw), z)
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    for index, axis in enumerate(axes[:3]):
        axis.imshow(_display_slice(context[index]), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(("z-1", "z", "z+1")[index])
        axis.axis("off")
    axes[3].imshow(_display_slice(image_zhw[z]), cmap="gray", vmin=0.0, vmax=1.0)
    axes[3].imshow(np.asarray(mask_zhw[z]), cmap=CLASS_COLORS, vmin=0, vmax=3, interpolation="nearest")
    axes[3].set_title("center + GT")
    axes[3].axis("off")
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)


def generate_cmr_multi(data_root: str | Path, output_dir: str | Path, *, max_cases: int = 10) -> int:
    adapter = CMRMultiAdapter(data_root, require_confident=True)
    units = adapter.discover_units()
    by_case: dict[str, list] = defaultdict(list)
    for unit in units:
        by_case[str(unit.metadata["parent_case_id"])].append(unit)
    selected_cases = sorted(by_case)[: int(max_cases)]
    generated = 0
    for case_id in selected_cases:
        case_units = by_case[case_id]
        for unit in (case_units[0], case_units[len(case_units) // 2], case_units[-1]):
            loaded = adapter.load_unit(unit)
            z_values = (0, loaded.image_zhw.shape[0] // 2, loaded.image_zhw.shape[0] - 1)
            for z in z_values:
                _save_overlay(
                    loaded.image_zhw,
                    loaded.mask_zhw,
                    z=z,
                    title=(
                        f"CMR-MULTI case={case_id} subject={unit.subject_id} "
                        f"z={z} t={unit.time_index} confidence={unit.metadata.get('zt_confidence')}"
                    ),
                    output=Path(output_dir) / "cmr_multi" / f"{case_id}_t{unit.time_index:02d}_z{z:02d}.png",
                )
                generated += 1
    if len(selected_cases) < int(max_cases):
        raise RuntimeError(f"CMR-MULTI visual QC produced only {len(selected_cases)} cases")
    return generated


def generate_cmrxmotion(data_root: str | Path, output_dir: str | Path, *, max_samples: int = 12) -> int:
    adapter = CMRxMotionAdapter(data_root, allow_affine_mismatch=True)
    units = adapter.discover_units(supervised_only=True)
    selected = []
    subjects: set[str] = set()
    phases: set[str] = set()
    for unit in units:
        if unit.subject_id not in subjects or unit.phase not in phases or len(selected) < max_samples:
            selected.append(unit)
            subjects.add(unit.subject_id)
            phases.add(unit.phase)
        if len(selected) >= int(max_samples):
            break
    generated = 0
    for unit in selected:
        loaded = adapter.load_unit(unit)
        for z in (0, loaded.image_zhw.shape[0] // 2, loaded.image_zhw.shape[0] - 1):
            _save_overlay(
                loaded.image_zhw,
                loaded.mask_zhw,
                z=z,
                title=(
                    f"CMRxMotion subject={unit.subject_id} case={unit.case_id} "
                    f"phase={unit.phase} acquisition={unit.acquisition_id} z={z} "
                    f"affine_equal={unit.metadata.get('affine_equal')}"
                ),
                output=Path(output_dir) / "cmrxmotion" / f"{unit.case_id}_z{z:02d}.png",
            )
            generated += 1
    if not selected:
        raise RuntimeError("CMRxMotion visual QC found no labeled units")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmr-multi-root", required=True)
    parser.add_argument("--cmrxmotion-root", required=True)
    parser.add_argument("--output-dir", default="reports/qc")
    parser.add_argument("--cmr-multi-cases", type=int, default=10)
    parser.add_argument("--cmrxmotion-samples", type=int, default=12)
    args = parser.parse_args()
    multi_count = generate_cmr_multi(args.cmr_multi_root, args.output_dir, max_cases=args.cmr_multi_cases)
    motion_count = generate_cmrxmotion(args.cmrxmotion_root, args.output_dir, max_samples=args.cmrxmotion_samples)
    print(f"cmr_multi={multi_count} cmrxmotion={motion_count}")


if __name__ == "__main__":
    main()
