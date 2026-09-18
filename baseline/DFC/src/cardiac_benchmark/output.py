"""Lossless raw DFC artifact writing and terminal attempt accounting."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from .provenance import canonical_json, file_sha256, sha256_bytes


class OutputError(ValueError): pass


def write_raw_artifact(directory: str | Path, cluster_map: np.ndarray, metadata: Mapping[str, Any]) -> dict[str, Any]:
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(np.asarray(cluster_map, dtype="<i4"))
    if array.ndim != 2: raise OutputError("raw partition must be [H,W]")
    raw_path = directory / "raw_cluster_map.npy"
    np.save(raw_path, array, allow_pickle=False)
    record = dict(metadata)
    record["partition"] = {"path": raw_path.name, "shape": list(array.shape), "dtype": "<i4", "content_hash": sha256_bytes(array.tobytes(order="C")), "file_hash": file_sha256(raw_path)}
    meta_path = directory / "metadata.json"
    meta_path.write_bytes(canonical_json(record))
    return {"raw_path": str(raw_path), "metadata_path": str(meta_path), "metadata_hash": file_sha256(meta_path), **record["partition"]}


class AttemptLedger:
    """One current terminal record per requested ID, retaining every retry attempt."""
    def __init__(self, expected_sample_ids: list[str]):
        if len(set(expected_sample_ids)) != len(expected_sample_ids): raise OutputError("duplicate expected sample IDs")
        self.expected = set(expected_sample_ids); self.attempts: list[dict[str, Any]] = []
    def terminal(self, sample_id: str, status: str, **details: Any) -> None:
        if sample_id not in self.expected or status not in {"success", "failure"}: raise OutputError("invalid terminal attempt")
        self.attempts.append({"sample_id": sample_id, "status": status, "attempt_id": len(self.attempts), **details})
    def current(self) -> dict[str, dict[str, Any]]:
        result = {}; [result.__setitem__(a["sample_id"], a) for a in self.attempts]
        return result
    def assert_complete(self) -> None:
        if set(self.current()) != self.expected: raise OutputError("missing explicit terminal record")
