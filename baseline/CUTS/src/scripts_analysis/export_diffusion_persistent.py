"""Materialize CUTS's GT-free persistent diffusion partition for each output.

``generate_diffusion.py`` produces every Multiscale-PHATE condensation level.
Upstream CUTS derives its single persistent representation with
``get_persistent_structures``.  This companion intentionally reads only the
diffusion hierarchy and writes only anonymous partitions; it never reads or
writes the ``label`` field present in upstream result NPZ files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

SRC_ROOT = Path(__file__).resolve().parents[1]
for _path in (SRC_ROOT, SRC_ROOT / "utils"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from utils.attribute_hashmap import AttributeHashmap
from utils.parse import parse_settings


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _upstream_persistent_structures_safe(labels: np.ndarray) -> np.ndarray:
    """The upstream persistence rule with int64 storage for a 224x224 map.

    The original helper creates an ``int16`` output.  A 224x224 hierarchy can
    legitimately contain an anonymous label above 32767, where that dtype
    aliases labels.  Cluster IDs have no semantic meaning, so retaining the
    algorithm while storing IDs safely is required before dense renumbering.
    The upstream loop deliberately excludes its final hierarchy frame; this
    function preserves that behavior for reproducibility.
    """
    min_area = 1e-2 * labels.shape[1] * labels.shape[2]
    persistent_label = np.zeros(labels.shape[1:], dtype=np.int64)
    persistence_tuples = []
    for label_idx in np.unique(labels):
        current_persistence = max_persistence = max_area = 0
        best_frame = None
        for frame_idx in range(labels.shape[0] - 1):
            current_area = int(np.sum(labels[frame_idx] == label_idx))
            if current_area > 0 and best_frame is None:
                best_frame = frame_idx
                max_area = current_area
            if current_area > 0:
                current_persistence += 1
                max_persistence = max(max_persistence, current_persistence)
                if current_area > max_area:
                    max_area = current_area
                    best_frame = frame_idx
        if best_frame is None or max_area < min_area or max_persistence < 2:
            continue
        persistence_tuples.append((max_persistence, max_area, int(label_idx), best_frame))
    for _, _, label_idx, frame_idx in sorted(persistence_tuples, key=lambda value: (-value[1], -value[0])):
        persistent_label[labels[frame_idx] == label_idx] = label_idx
    return persistent_label


def _continuous_renumber(labels: np.ndarray) -> np.ndarray:
    values = np.unique(labels)
    dense = np.empty_like(labels, dtype=np.int64)
    for new_value, old_value in enumerate(values):
        dense[labels == old_value] = new_value
    return dense


def persistent_partition(labels_diffusion: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Apply the upstream persistence rule and return safe, dense raw IDs."""
    levels = np.asarray(labels_diffusion)
    if levels.ndim != 2 or levels.shape[1] != height * width:
        raise ValueError("labels_diffusion must have shape [num_levels, height * width]")
    if levels.shape[0] < 2:
        raise ValueError("CUTS persistent diffusion requires at least two condensation levels")
    if not np.issubdtype(levels.dtype, np.integer):
        raise ValueError("labels_diffusion must be integral")
    if levels.min() < 0:
        raise ValueError("labels_diffusion must contain non-negative cluster IDs")
    persistent = _upstream_persistent_structures_safe(levels.reshape(levels.shape[0], height, width))
    return _continuous_renumber(persistent)


def export(input_root: Path, output_root: Path) -> list[dict[str, object]]:
    input_paths = sorted(input_root.glob("*.npz"))
    if not input_paths:
        raise ValueError(f"no diffusion NPZ files under {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for input_path in input_paths:
        with np.load(input_path, allow_pickle=False) as payload:
            levels = np.asarray(payload["labels_diffusion"])
            if "image" not in payload:
                raise ValueError(f"{input_path} has no image needed to establish spatial shape")
            height, width = np.asarray(payload["image"]).shape[:2]
            granularities = np.asarray(payload["granularities_diffusion"])
        partition = persistent_partition(levels, height=int(height), width=int(width))
        raw_path = output_root / f"{input_path.stem}.npy"
        np.save(raw_path, partition, allow_pickle=False)
        records.append({
            "source_diffusion_file": input_path.name,
            "raw_partition_file": raw_path.name,
            "shape": [int(height), int(width)],
            "num_diffusion_levels": int(levels.shape[0]),
            "granularities": [int(value) for value in granularities.tolist()],
            "hierarchy_sha256": _sha256_array(levels),
            "raw_partition_sha256": _sha256_array(partition),
            "selection_rule": "upstream_cuts.get_persistent_structures.v1",
        })
    manifest = {"schema_version": "cuts.diffusion-persistent.v1", "records": records}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = AttributeHashmap(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    config.config_file_name = str(args.config)
    config = parse_settings(config, log_settings=False)
    input_root = args.input_root or Path(config.output_save_path) / "numpy_files_seg_diffusion"
    output_root = args.output_root or Path(config.output_save_path) / "numpy_files_seg_diffusion_persistent"
    export(input_root, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
