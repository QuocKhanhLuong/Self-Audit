"""Drive backup: file selection, SHA256SUMS, verification and ID recording against a fake rclone (no network)."""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cbkd_sa_upload", ROOT / "scripts/cbkd_sa_upload.py")
up = importlib.util.module_from_spec(spec); sys.modules["cbkd_sa_upload"] = up; spec.loader.exec_module(up)

FAKE_RCLONE = r'''#!{python}
import hashlib, json, os, shutil, sys
from pathlib import Path
drive = Path(os.environ["FAKE_DRIVE"])
args = sys.argv[1:]
cmd, rest = args[0], [a for a in args[1:] if not a.startswith("--")]
rest = [a for a in rest if a not in ("4", os.environ.get("FOLDER", ""))]
def remote(p): return drive / p.split(":", 1)[1]
if os.environ.get("FAKE_FAIL") == cmd:
    print("boom", file=sys.stderr); sys.exit(5)
if cmd == "copy":
    src, dst = Path(rest[0]), remote(rest[1]); dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copyfile(f, dst / f.name)   # follows symlinks
elif cmd == "check":
    src, dst = Path(rest[0]), remote(rest[1])
    bad = [f.name for f in src.iterdir() if not (dst / f.name).is_file() or (dst / f.name).read_bytes() != f.read_bytes()]
    if os.environ.get("FAKE_CORRUPT"):
        bad = ["x"]
    sys.exit(1 if bad else 0)
elif cmd == "lsjson":
    dst = remote(rest[0])
    print(json.dumps([{"Name": f.name, "Size": f.stat().st_size, "ID": "id_" + hashlib.md5(f.name.encode()).hexdigest()[:8]} for f in sorted(dst.iterdir())]))
'''


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    exe = bin_dir / "rclone"
    exe.write_text(FAKE_RCLONE.replace("{python}", sys.executable)); exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    drive = tmp_path / "drive"; drive.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DRIVE", str(drive))
    monkeypatch.setenv("FOLDER", up.FOLDER_ID)
    return drive


def _run_dir(tmp_path: Path) -> Path:
    out = tmp_path / "run"
    for rel, data in {"S1/student_init.pt": b"enc", "S1_RECEIPT.json": b"{}", "S2/teacher_best.pt": b"teach", "S2/teacher_latest.pt": b"resume",
                      "S2_RECEIPT.json": b"{}", "resume_A4.pt": b"big", "exports/A4_SA.pt": b"student", "exports/A4_SA.pt.manifest.json": b"{}",
                      "FINAL_RESULT.json": b"{}", "arms/A4/epochs.csv": b"a,b", "arms/A4/diag.jsonl": b"{}", "A4_SA_CONFIG.json": b"{}"}.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True); (out / rel).write_bytes(data)
    (out / "markers").mkdir(); (out / "markers" / "S1.DONE.json").write_text(json.dumps({"stage": "S1"}))
    return out


def test_upload_selects_files_writes_sums_verifies_and_records_ids(fake, tmp_path) -> None:
    out = _run_dir(tmp_path)
    md = tmp_path / "ARTIFACTS.md"
    assert up.main(["--out", str(out), "--stage", "S1", "--artifacts-md", str(md), "--strict"]) == 0
    folder = fake / up.DEST_ROOT / "S1"
    assert {p.name for p in folder.iterdir()} == {"student_init.pt", "S1_RECEIPT.json", "SHA256SUMS"}  # absent files skipped
    sums = (folder / "SHA256SUMS").read_text()
    assert up.sha256_file(out / "S1/student_init.pt") in sums
    rec = json.loads((out / "markers" / "S1.UPLOADED.json").read_text())
    assert rec["verified"] and rec["files"]["student_init.pt"]["drive_id"].startswith("id_")
    done = json.loads((out / "markers" / "S1.DONE.json").read_text())
    assert done["drive"]["files"]["student_init.pt"] == rec["files"]["student_init.pt"]["drive_id"]
    assert "student_init.pt" in md.read_text()


def test_resume_checkpoint_only_when_requested_and_no_forbidden_data(fake, tmp_path) -> None:
    out = _run_dir(tmp_path)
    assert up.main(["--out", str(out), "--stage", "S2", "--strict"]) == 0
    assert not (fake / up.DEST_ROOT / "S2" / "teacher_latest.pt").exists()
    assert up.main(["--out", str(out), "--stage", "S2", "--include-resume", "--strict"]) == 0
    assert (fake / up.DEST_ROOT / "S2" / "teacher_latest.pt").is_file()
    assert up.main(["--out", str(out), "--stage", "FINAL", "--strict"]) == 0
    names = {p.name for p in (fake / up.DEST_ROOT / "A4-SA").iterdir()}
    assert "A4_SA.pt" in names and "resume_A4.pt" not in names
    with pytest.raises(ValueError):
        up.ARTIFACTS["S1"].append("prepared/x.json"); up.collect(out, "S1", False)
    up.ARTIFACTS["S1"].pop()


def test_failures_are_nonfatal_unless_strict_and_corruption_is_caught(fake, tmp_path, monkeypatch, capsys) -> None:
    out = _run_dir(tmp_path)
    monkeypatch.setenv("FAKE_FAIL", "copy")
    assert up.main(["--out", str(out), "--stage", "S1"]) == 0  # warning only
    assert "WARNING" in capsys.readouterr().err and not (out / "markers" / "S1.UPLOADED.json").exists()
    assert up.main(["--out", str(out), "--stage", "S1", "--strict"]) == 1
    monkeypatch.delenv("FAKE_FAIL"); monkeypatch.setenv("FAKE_CORRUPT", "1")
    assert up.main(["--out", str(out), "--stage", "S1", "--strict"]) == 1  # rclone check failure detected
    assert up.main(["--out", str(tmp_path / "empty"), "--stage", "S1"]) == 0  # nothing to upload is also non-fatal
