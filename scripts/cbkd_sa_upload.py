#!/usr/bin/env python3
"""Back up one finished stage's reusable artifacts to Google Drive (rclone), verify, and record Drive file IDs.

Destination: Drive folder ``--folder-id`` (default: the "Self-Audit" folder), ``pg-artifacts-20261119/<stage>/``.
Per stage a ``SHA256SUMS`` file is written next to the artifacts; after upload the files are checked with
``rclone check`` (Drive MD5) and the Drive file IDs are written to ``<run>/markers/<stage>.UPLOADED.json`` and merged into
``reports/cbkd-sa-a4-20261119/ARTIFACTS.md`` when ``--artifacts-md`` is given.

Never uploads raw/prepared data or optimizer/resume checkpoints, except with ``--include-resume`` (the last resume
checkpoint of an aborted stage).  Failure is non-fatal by default (warning, exit 0); ``--strict`` makes it fatal.
The Drive token is never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FOLDER_ID = "1KLYuulOT6riQjeF1oarmeBDmwcsD3aEX"
DEST_ROOT = "pg-artifacts-20261119"

# Reusable artifacts per stage, relative to the run directory.
ARTIFACTS = {
    "S1": ["S1/student_init.pt", "S1_RECEIPT.json", "S1/metrics.jsonl", "S1/epochs.csv"],
    "S2": ["S2/teacher_best.pt", "S2_RECEIPT.json", "S2/epochs.csv", "S2/diag.jsonl"],
    "S3": ["S3/codebook.pt", "S3/S3_SA_RECEIPT.json", "S3_RECEIPT.json", "S3/diag.jsonl", "S3/epochs.csv"],
    "A4": ["exports/A4_SA.pt", "exports/A4_SA.pt.manifest.json", "FINAL_RESULT.json", "arms/A4/epochs.csv", "arms/A4/epochs.jsonl",
           "arms/A4/diag.jsonl", "arms/A4/EXPERIMENT_LOCK.json", "A4_SA_CONFIG.json"],
}
ARTIFACTS["FINAL"] = ARTIFACTS["A4"]  # the A4-SA folder is completed by the FINAL stage (export + final receipt)
REMOTE_FOLDER = {"S1": "S1", "S2": "S2", "S3": "S3-SA", "A4": "A4-SA", "FINAL": "A4-SA"}
# Only for an aborted stage (``--include-resume``).
RESUME = {"S1": [], "S2": ["S2/teacher_latest.pt"], "S3": ["S3/checkpoint.pt"], "A4": ["resume_A4.pt"], "FINAL": []}
FORBIDDEN_PARTS = ("prepared", "raw", "arrays")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rclone(args: list[str], folder_id: str, *, capture: bool = True) -> subprocess.CompletedProcess:
    cmd = ["rclone", *args, "--drive-root-folder-id", folder_id]
    return subprocess.run(cmd, capture_output=capture, text=True)


def collect(out: Path, stage: str, include_resume: bool) -> list[Path]:
    names = list(ARTIFACTS[stage]) + (RESUME[stage] if include_resume else [])
    files = []
    for name in names:
        path = out / name
        if any(part in FORBIDDEN_PARTS for part in Path(name).parts):
            raise ValueError(f"refusing to upload {name}")
        if path.is_file():
            files.append(path)
    if not files:
        raise FileNotFoundError(f"no artifacts found for stage {stage} under {out}")
    return files


def upload_stage(out: Path, stage: str, *, folder_id: str = FOLDER_ID, dest_root: str = DEST_ROOT,
                 include_resume: bool = False) -> dict:
    if stage not in ARTIFACTS:
        raise ValueError(f"unknown stage {stage}")
    files = collect(out, stage, include_resume)
    remote = f"gdrive:{dest_root}/{REMOTE_FOLDER[stage]}"
    with tempfile.TemporaryDirectory() as tmp:
        stage_dir = Path(tmp) / stage
        stage_dir.mkdir()
        sums = []
        for path in files:
            digest = sha256_file(path)
            sums.append((digest, path.name, path.stat().st_size))
            (stage_dir / path.name).symlink_to(path.resolve())
        # Distinct source paths must not collide on file name inside the stage folder.
        if len({name for _, name, _ in sums}) != len(sums):
            raise ValueError("artifact file names collide inside one stage folder")
        (stage_dir / "SHA256SUMS").write_text("".join(f"{d}  {n}\n" for d, n, _ in sums))
        res = rclone(["copy", str(stage_dir), remote, "--copy-links", "--transfers", "4", "--checksum"], folder_id)
        if res.returncode:
            raise RuntimeError(f"rclone copy failed: {res.stderr.strip()[-300:]}")
        chk = rclone(["check", str(stage_dir), remote, "--copy-links", "--one-way"], folder_id)
        if chk.returncode:
            raise RuntimeError(f"rclone check failed: {(chk.stderr or chk.stdout).strip()[-300:]}")
        listing = rclone(["lsjson", remote], folder_id)
        if listing.returncode:
            raise RuntimeError(f"rclone lsjson failed: {listing.stderr.strip()[-300:]}")
        remote_rows = {row["Name"]: row for row in json.loads(listing.stdout)}
    record = {"stage": stage, "remote_path": f"{dest_root}/{REMOTE_FOLDER[stage]}", "folder_id": folder_id, "verified": True,
              "uploaded_unix": time.time(), "include_resume": include_resume, "files": {}}
    for digest, name, size in sums + [(None, "SHA256SUMS", None)]:
        row = remote_rows.get(name)
        if row is None:
            raise RuntimeError(f"{name} missing on Drive after upload")
        if size is not None and int(row.get("Size", -1)) != size:
            raise RuntimeError(f"{name}: Drive size {row.get('Size')} != local {size}")
        record["files"][name] = {"drive_id": row.get("ID"), "size": row.get("Size"), "sha256": digest}
    return record


def record_artifacts(out: Path, stage: str, record: dict, artifacts_md: Path | None) -> None:
    marker = out / "markers" / f"{stage}.UPLOADED.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_name(f".{marker.name}.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, marker)
    # Also fold the IDs into the stage DONE marker when it is not being written concurrently by the pipeline.
    done = out / "markers" / f"{stage}.DONE.json"
    if done.is_file():
        data = json.loads(done.read_text())
        data["drive"] = {"remote_path": record["remote_path"], "files": {n: v["drive_id"] for n, v in record["files"].items()}}
        tmp = done.with_name(f".{done.name}.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, done)
    if artifacts_md is not None:
        artifacts_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"\n## {stage}  (`{record['remote_path']}`)\n", "| file | Drive ID | bytes | sha256 |\n|---|---|---|---|\n"]
        for name, v in record["files"].items():
            lines.append(f"| {name} | `{v['drive_id']}` | {v['size']} | `{(v['sha256'] or '')[:16]}` |\n")
        with artifacts_md.open("a", encoding="utf-8") as fh:
            fh.writelines(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", required=True, choices=sorted(ARTIFACTS))
    ap.add_argument("--folder-id", default=FOLDER_ID)
    ap.add_argument("--dest-root", default=DEST_ROOT)
    ap.add_argument("--include-resume", action="store_true")
    ap.add_argument("--artifacts-md", type=Path)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)
    try:
        record = upload_stage(args.out, args.stage, folder_id=args.folder_id, dest_root=args.dest_root, include_resume=args.include_resume)
        record_artifacts(args.out, args.stage, record, args.artifacts_md)
        print(f"[upload] {args.stage}: {len(record['files'])} files verified on Drive ({record['remote_path']})", flush=True)
        return 0
    except Exception as exc:  # non-fatal by design
        print(f"[upload] WARNING {args.stage}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
