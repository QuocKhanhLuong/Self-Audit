"""Write raw partition artifacts with provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .provenance import sha256_array, sha256_file, write_json


def write_raw_partition(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write raw partition .npy + provenance .json for one sample.

    Returns the result dict augmented with file paths and hashes.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    sample_id = result["sample_id"]
    safe_id = sample_id.replace(":", "_").replace("/", "_")

    partition = result["partition"]
    npy_path = root / f"{safe_id}.npy"
    np.save(npy_path, partition, allow_pickle=False)

    provenance = {
        "sample_id": sample_id,
        "patient_id": result["patient_id"],
        "split": result["split"],
        "partition_shape": result["partition_shape"],
        "partition_hash": result["partition_hash"],
        "n_clusters": result["n_clusters"],
        "raw_partition_path": str(npy_path),
        "raw_file_hash": sha256_file(npy_path),
        "elapsed_seconds": result["elapsed_seconds"],
        "status": result["status"],
    }

    json_path = root / f"{safe_id}.json"
    write_json(json_path, provenance)

    result["raw_partition_path"] = str(npy_path)
    result["raw_file_hash"] = provenance["raw_file_hash"]
    return result
