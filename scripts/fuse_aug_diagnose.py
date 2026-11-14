#!/usr/bin/env python3
"""Evaluate SHA-verified old A exports on locked L8 and existing VAL only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pins(directory: Path) -> dict[str, dict[str, str]]:
    result = {}
    for mode in ("teacher", "student", "fusion"):
        artifact = directory / f"{mode}.pt"
        sidecar = artifact.with_suffix(artifact.suffix + ".manifest.json")
        if not artifact.is_file() or not sidecar.is_file():
            raise ValueError(f"missing {mode} export or SHA sidecar under {directory}")
        meta = json.loads(sidecar.read_text())
        digest = _sha(artifact)
        if (meta.get("schema") != "self_audit.candidate_fuse.export.v1"
                or digest != meta.get("artifact_sha256")
                or meta.get("mode") != mode or meta.get("artifact") != artifact.name
                or meta.get("artifact_bytes") != artifact.stat().st_size):
            raise ValueError(f"{mode} export SHA/mode does not match sidecar")
        result[mode] = {"path": str(artifact), "sha256": digest}
    return result


def run(config_path: Path, train_manifest: Path, train_sha: str, val_manifest: Path,
        val_sha: str, exports_dir: Path, output: Path, *, device="cpu",
        synthetic: bool = False) -> dict:
    """Load verified exports; evaluate only TRAIN L frames and development VAL."""
    workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None,":4096:8",":16:8"):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    from self_audit.candidate_fuse.config import load_config
    from self_audit.candidate_fuse.data import FuseData, open_fuse_data
    from self_audit.candidate_fuse.evaluation import evaluate_modes
    from self_audit.candidate_fuse.export import load_export

    cfg = load_config(config_path, real_run=not synthetic)
    if cfg.get("experiment", {}).get("track") != "A":
        raise ValueError("diagnosis accepts only the locked A/v2 recipe")
    for key, supplied_path, supplied_sha in (
        ("train_manifest", train_manifest, train_sha), ("val_manifest", val_manifest, val_sha)
    ):
        if (Path(cfg["data"][key]).resolve() != Path(supplied_path).resolve()
                or cfg["data"][f"{key}_sha256"] != supplied_sha):
            raise ValueError(f"diagnosis {key} differs from the configured locked identity")
    torch.set_num_threads(int(cfg["runtime"]["num_threads"]))
    torch.use_deterministic_algorithms(bool(cfg["runtime"]["deterministic"]), warn_only=not synthetic)
    deterministic = bool(cfg["runtime"]["deterministic"])
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    pins = _pins(exports_dir)
    train = FuseData(train_manifest, train_sha, role="train", synthetic=synthetic)
    if not synthetic and len(train._labeled_ids) != 8:
        raise ValueError("diagnosis requires exactly locked L8")
    val = open_fuse_data(val_manifest, val_sha, role="development", synthetic=synthetic)
    models = {name: load_export(pin["path"], expected_sha256=pin["sha256"], device=device)
              for name, pin in pins.items()}
    # Restrict the training-role wrapper to its authorized L keys; evaluation
    # never requests U images or masks outside this list.
    class _L8View:
        patients = tuple(sorted(train._labeled_ids))
        def labeled_keys(self): return train.labeled_keys()
        def load(self, key, *, labeled):
            if not labeled or key not in train.labeled_keys():
                raise ValueError("diagnosis attempted a non-L8 read")
            return train.load(key, labeled=True)
    results = {}
    for label, dataset in (("L8_TRAIN_UNAUGMENTED", _L8View()), ("VAL18", val)):
        results[label] = evaluate_modes(models["teacher"].teacher,
            models["student"].student, dataset, device=device,
            fusion_teacher=models["fusion"].teacher,
            fusion_student=models["fusion"].student)
    result = {"schema": "self_audit.fuse_aug_diagnosis.v1", "status": "COMPLETED",
        "evidence_status": "SOFTWARE_ONLY" if synthetic else "REAL_DATA_EVALUATION",
        "modes": ["teacher", "student", "fusion"],
        "train_scope": "L8 ED/ES TRAIN keys, no augmentation" if not synthetic else "synthetic labeled TRAIN keys",
        "test_opened": False,
        "config_sha256": _sha(config_path), "train_manifest_sha256": train.manifest_sha256,
        "val_manifest_sha256": val.manifest_sha256, "exports": pins, "metrics": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--old-run", type=Path)
    p.add_argument("--train-manifest", type=Path); p.add_argument("--val-manifest", type=Path)
    p.add_argument("--exports", type=Path); p.add_argument("--out", required=True, type=Path)
    p.add_argument("--train-sha256"); p.add_argument("--val-sha256")
    p.add_argument("--device", default="cuda")
    p.add_argument("--synthetic", action="store_true", help="software-only fixture mode")
    a = p.parse_args(argv)
    if a.old_run:
        cfg=json.loads(a.config.read_text()); data=cfg["data"]
        a.train_manifest=Path(data["train_manifest"]); a.train_sha256=data["train_manifest_sha256"]
        a.val_manifest=Path(data["val_manifest"]); a.val_sha256=data["val_manifest_sha256"]
        result_path=a.old_run/"RESULT.json"
        if not result_path.is_file(): raise FileNotFoundError(f"old run lacks RESULT.json: {result_path}")
        old=json.loads(result_path.read_text())
        a.exports=a.old_run/"exports"
        for mode,row in old.get("exports",{}).items():
            source=Path(row.get("path",""))
            candidate=a.exports/f"{mode}.pt"
            if not candidate.is_file() and source.is_file():
                a.exports=source.parent
                break
    if not all((a.train_manifest,a.val_manifest,a.exports,a.train_sha256,a.val_sha256)):
        p.error("provide --old-run or explicit manifest, SHA, and exports arguments")
    print(json.dumps(run(a.config, a.train_manifest, a.train_sha256, a.val_manifest,
                         a.val_sha256, a.exports, a.out, device=a.device,
                         synthetic=a.synthetic), indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
