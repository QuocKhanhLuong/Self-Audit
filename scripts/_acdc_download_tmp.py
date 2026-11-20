#!/usr/bin/env python3
import concurrent.futures
import datetime
import hashlib
import html
import json
import os
import pathlib
import re
import subprocess
import time
import urllib.request

ROOT = pathlib.Path("/root/mount_data/datasets/ACDC/raw")
RECEIPT = pathlib.Path("/root/mount_data/receipts/setup-20261112")
ROOT.mkdir(parents=True, exist_ok=True)
RECEIPT.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0"}
SOURCES = {
    "training": "15BKahUSkHyJv0tctCcavnoLS30YsvNY6",
    "testing": "1RQkxK1y5C8ydaJoOCiH-WHi51y7aMvz_",
}
ID_RE = re.compile(r'data-id="([^"]+)"')
STRONG_RE = re.compile(r"<strong[^>]*>([^<]+)</strong>")


def fetch_folder(fid):
    url = "https://drive.google.com/drive/folders/" + fid
    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", "replace")
        except Exception as exc:
            print("FETCH_RETRY", fid, attempt, repr(exc), flush=True)
            time.sleep(min(attempt * 2, 15))
    raise RuntimeError("folder fetch failed " + fid)


def rows(body):
    out, seen = [], set()
    for match in ID_RE.finditer(body):
        window = body[match.end() : match.end() + 2200]
        name_match = STRONG_RE.search(window)
        if name_match:
            name = html.unescape(name_match.group(1)).strip()
            fid = match.group(1)
            if name and fid not in seen:
                seen.add(fid)
                out.append((fid, name))
    return out


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(item):
    split, patient, fid, name = item
    dest = ROOT / split / patient / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if dest.exists() and dest.stat().st_size > 0:
        size = dest.stat().st_size
        return {
            "split": split,
            "patient": patient,
            "id": fid,
            "name": name,
            "path": str(dest),
            "bytes": size,
            "sha256": sha256(dest),
            "status": "EXISTING",
        }
    url = (
        "https://drive.usercontent.google.com/download?id="
        + fid
        + "&export=download&confirm=t"
    )
    command = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "5",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--max-time",
        "1800",
        "-o",
        str(part),
        url,
    ]
    last = None
    for attempt in range(1, 6):
        try:
            proc = subprocess.run(command, text=True, capture_output=True, timeout=1900)
            if proc.returncode == 0 and part.exists() and part.stat().st_size > 0:
                os.replace(part, dest)
                size = dest.stat().st_size
                return {
                    "split": split,
                    "patient": patient,
                    "id": fid,
                    "name": name,
                    "path": str(dest),
                    "bytes": size,
                    "sha256": sha256(dest),
                    "status": "DOWNLOADED",
                }
            last = proc.stderr[-1200:]
        except Exception as exc:
            last = repr(exc)
        time.sleep(min(attempt * 3, 20))
    if part.exists():
        part.unlink()
    raise RuntimeError(f"{split}/{patient} {name}: {last}")


def main():
    inventory = {
        "schema": "self_audit.acdc.drive_download.v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": "Google Drive public download endpoints for authenticated shared folders",
        "source_folders": SOURCES,
        "raw_root": str(ROOT),
        "patients": {},
        "files": [],
        "errors": [],
    }
    items = []
    for split, folder_id in SOURCES.items():
        patient_rows = [
            (fid, name)
            for fid, name in rows(fetch_folder(folder_id))
            if name.startswith("patient") and len(name) == 10 and name[7:].isdigit()
        ]
        patient_rows.sort(key=lambda x: x[1])
        print("PATIENT_FOLDERS", split, len(patient_rows), flush=True)
        if len(patient_rows) != 50:
            raise RuntimeError(f"expected 50 patients in {split}, got {len(patient_rows)}")
        for patient_id, patient in patient_rows:
            file_rows = rows(fetch_folder(patient_id))
            inventory["patients"][patient] = {
                "split": split,
                "folder_id": patient_id,
                "files_in_source": len(file_rows),
                "source_names": [name for _, name in file_rows],
            }
            for fid, name in file_rows:
                if name != "MANDATORY_CITATION.md":
                    items.append((split, patient, fid, name))
    print("FILES_TO_DOWNLOAD", len(items), flush=True)
    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, item) for item in items]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            completed.append(record)
            if index % 20 == 0 or index == len(items):
                print("PROGRESS", index, "/", len(items), "bytes", sum(x["bytes"] for x in completed), flush=True)
    completed.sort(key=lambda x: (x["split"], x["patient"], x["name"]))
    inventory["files"] = completed
    inventory["counts"] = {
        "patients_by_split": {
            split: sum(1 for value in inventory["patients"].values() if value["split"] == split)
            for split in SOURCES
        },
        "files": len(completed),
        "bytes": sum(item["bytes"] for item in completed),
        "downloaded": sum(item["status"] == "DOWNLOADED" for item in completed),
        "existing": sum(item["status"] == "EXISTING" for item in completed),
    }
    inventory["completed_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tmp = RECEIPT / "ACDC_RAW_INVENTORY.json.tmp"
    final = RECEIPT / "ACDC_RAW_INVENTORY.json"
    tmp.write_text(json.dumps(inventory, indent=2) + "\n")
    os.replace(tmp, final)
    print("COMPLETE", json.dumps(inventory["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
