#!/usr/bin/env python3
"""Complete the authorized ACDC Drive folder manifest on the Vast instance.

This temporary deployment helper is intentionally outside the reusable training
pipeline.  It downloads only the exact patient folders in the pinned receipt,
validates the NIfTI container before accepting a file, and writes a complete
raw-data receipt.  It is removed from the checkout after the transfer.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request


ROOT = Path("/root/mount_data/datasets/ACDC/raw")
RECEIPT = Path("/root/mount_data/receipts/setup-20261112")
FOLDER_MANIFEST = RECEIPT / "ACDC_DRIVE_FOLDERS.json"
HEADERS = {"User-Agent": "Mozilla/5.0"}
ID_RE = re.compile(r'data-id="([^"]+)"')
STRONG_RE = re.compile(r"<strong[^>]*>([^<]+)</strong>")


def fetch_folder(folder_id: str) -> str:
    url = "https://drive.google.com/drive/folders/" + folder_id
    last: Exception | None = None
    for attempt in range(1, 7):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=90) as response:
                body = response.read().decode("utf-8", "replace")
            if "<html" not in body.lower() or "data-id=" not in body:
                raise RuntimeError(f"unexpected Drive folder response for {folder_id}")
            return body
        except Exception as exc:  # bounded retry for transient Drive responses
            last = exc
            print("FETCH_RETRY", folder_id, attempt, repr(exc), flush=True)
            time.sleep(min(3 * attempt, 20))
    raise RuntimeError(f"folder fetch failed {folder_id}: {last!r}")


def rows(body: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in ID_RE.finditer(body):
        window = body[match.end() : match.end() + 2200]
        name_match = STRONG_RE.search(window)
        if not name_match:
            continue
        file_id = match.group(1)
        name = html.unescape(name_match.group(1)).strip()
        key = (file_id, name)
        if name and key not in seen:
            seen.add(key)
            found.append(key)
    return found


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_nifti(path: Path) -> bool:
    # A real ACDC release contains uncompressed .nii files.  The magic is a
    # cheap consent-page guard; nibabel then verifies the full NIfTI header.
    if path.stat().st_size < 352:
        return False
    with path.open("rb") as handle:
        header = handle.read(352)
    if header[344:348] not in (b"n+1\x00", b"ni1\x00"):
        return False
    try:
        import nibabel as nib

        image = nib.load(str(path), mmap=False)
        return len(image.shape) >= 3 and image.header.get_data_dtype() is not None
    except Exception:
        return False


def valid_file(path: Path, name: str) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    if name.endswith(".nii"):
        return valid_nifti(path)
    if name == "Info.cfg":
        text = path.read_text(errors="replace")
        return "ED:" in text and "ES:" in text
    return True


def download(item: tuple[str, str, str, str]) -> dict[str, object]:
    split, patient, file_id, name = item
    destination = ROOT / split / patient / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if valid_file(destination, name):
        return {
            "split": split,
            "patient": patient,
            "id": file_id,
            "name": name,
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "status": "EXISTING_VALIDATED",
        }
    if destination.exists():
        # Preserve a failed prior payload for diagnosis; never treat it as data.
        quarantine = destination.with_name(destination.name + ".invalid-20261112")
        if quarantine.exists():
            quarantine.unlink()
        os.replace(destination, quarantine)

    part = destination.with_name(destination.name + ".part")
    if part.exists() and not valid_file(part, name):
        part.unlink()
    url = (
        "https://drive.usercontent.google.com/download?id="
        + file_id
        + "&export=download&confirm=t"
    )
    command = [
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--retry", "5", "--retry-all-errors", "--retry-delay", "2",
        "--connect-timeout", "30", "--max-time", "1800", "-o", str(part), url,
    ]
    last = ""
    for attempt in range(1, 6):
        try:
            proc = subprocess.run(command, text=True, capture_output=True, timeout=1900)
            if proc.returncode == 0 and valid_file(part, name):
                os.replace(part, destination)
                return {
                    "split": split,
                    "patient": patient,
                    "id": file_id,
                    "name": name,
                    "path": str(destination),
                    "bytes": destination.stat().st_size,
                    "sha256": sha256(destination),
                    "status": "DOWNLOADED_VALIDATED",
                }
            last = proc.stderr[-1200:] or "invalid payload"
        except Exception as exc:
            last = repr(exc)
        if part.exists():
            part.unlink()
        time.sleep(min(3 * attempt, 20))
    raise RuntimeError(f"{split}/{patient}/{name}: {last}")


def main() -> None:
    manifest = json.loads(FOLDER_MANIFEST.read_text())
    folders = manifest["folders"]
    expected = {"training": 100, "testing": 50}
    if {key: len(value) for key, value in folders.items()} != expected:
        raise RuntimeError("folder manifest coverage is not exactly 100 training/50 testing")
    patients = {
        split: sorted(entries, key=lambda entry: entry["patient"])
        for split, entries in folders.items()
    }
    if [e["patient"] for e in patients["training"]] != [f"patient{i:03d}" for i in range(1, 101)]:
        raise RuntimeError("training patient IDs are not patient001..patient100")
    if [e["patient"] for e in patients["testing"]] != [f"patient{i:03d}" for i in range(101, 151)]:
        raise RuntimeError("testing patient IDs are not patient101..patient150")

    # Folder metadata is small but there are 150 patient folders.  Fetch it in
    # bounded parallelism so the actual file transfer can start promptly.
    requests = [
        (split, entry["patient"], entry["id"])
        for split, entries in patients.items()
        for entry in entries
    ]
    source_rows: dict[str, list[tuple[str, str]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(fetch_folder, folder_id): (split, patient, folder_id)
            for split, patient, folder_id in requests
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            split, patient, folder_id = futures[future]
            source_rows[patient] = rows(future.result())
            if index % 25 == 0 or index == len(futures):
                print("SOURCE_METADATA", index, "/", len(futures), flush=True)

    source_patients: dict[str, dict[str, object]] = {}
    items: list[tuple[str, str, str, str]] = []
    folder_ids = {
        entry["patient"]: (split, entry["id"])
        for split, entries in patients.items()
        for entry in entries
    }
    for patient in sorted(source_rows):
        split, folder_id = folder_ids[patient]
        unique_by_name = {name: file_id for file_id, name in source_rows[patient]}
        expected_names = {"Info.cfg", "MANDATORY_CITATION.md", f"{patient}_4d.nii"}
        expected_names.update(
            name
            for name in unique_by_name
            if name.startswith(patient + "_frame") and name.endswith(".nii")
        )
        if len(unique_by_name) != 7 or not expected_names.issubset(unique_by_name):
            raise RuntimeError(
                f"{split}/{patient}: expected 7 unique source files, got {sorted(unique_by_name)}"
            )
        source_patients[patient] = {
            "split": split,
            "folder_id": folder_id,
            "files_in_source": len(unique_by_name),
            "source_names": sorted(unique_by_name),
        }
        for name, file_id in sorted(unique_by_name.items()):
            if name != "MANDATORY_CITATION.md":
                items.append((split, patient, file_id, name))
    print("SOURCE_COVERAGE", json.dumps({k: len(v) for k, v in patients.items()}), flush=True)
    print("FILES_TO_VALIDATE_OR_DOWNLOAD", len(items), flush=True)

    completed: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, item) for item in items]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            completed.append(record)
            if index % 20 == 0 or index == len(items):
                print("PROGRESS", index, "/", len(items), "bytes", sum(int(x["bytes"]) for x in completed), flush=True)
    completed.sort(key=lambda value: (str(value["split"]), str(value["patient"]), str(value["name"])))
    inventory = {
        "schema": "self_audit.acdc.drive_download.v2",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "exact authenticated Drive folder manifest + direct Drive usercontent download",
        "folder_manifest": str(FOLDER_MANIFEST),
        "raw_root": str(ROOT),
        "patients": source_patients,
        "files": completed,
        "counts": {
            "patients_by_split": {split: len(entries) for split, entries in patients.items()},
            "files": len(completed),
            "bytes": sum(int(item["bytes"]) for item in completed),
            "downloaded_validated": sum(item["status"] == "DOWNLOADED_VALIDATED" for item in completed),
            "existing_validated": sum(item["status"] == "EXISTING_VALIDATED" for item in completed),
        },
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    final = RECEIPT / "ACDC_RAW_INVENTORY.json"
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(inventory, indent=2) + "\n")
    os.replace(tmp, final)
    print("COMPLETE", json.dumps(inventory["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
