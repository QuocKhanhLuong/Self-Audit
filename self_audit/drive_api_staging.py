"""Read-only Drive API v3 transfers. No rclone process, browser cookie or MRI client relay."""
from __future__ import annotations

import configparser
import hashlib
import json
import os
from pathlib import Path
import re
import time

from .asset_staging import atomic_json, sha256

TRAINING_FOLDER_ID = "15BKahUSkHyJv0tctCcavnoLS30YsvNY6"
SMOKE_FILE_ID = "1KYy4ktn-9WtkCMpYHJwGIp0K0D2BTkWV"
SMOKE_NAME = "patient007_4d.nii"
SMOKE_SIZE = 15913312
OLD_STAGING_SHA = "c4ffe94c921820f0bc3c67586f3ad64f5c6ec02fe2359146c1f2fa8c7b22029d"


class SafeDriveError(RuntimeError):
    def __init__(self, status, reasons):
        self.status = status
        self.reasons = reasons
        super().__init__(json.dumps({"http_status": status, "reasons": reasons}))


def sanitized_error(exc):
    """Only HTTP code/reason enums leave the CPU process, never response messages or tokens."""
    if isinstance(exc, SafeDriveError):
        return {"http_status": exc.status, "reasons": exc.reasons}
    status = getattr(getattr(exc, "resp", None), "status", None)
    reasons = []
    try:
        body = json.loads(exc.content)
        for item in body.get("error", {}).get("errors", []):
            reason = item.get("reason", "")
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", reason):
                reasons.append(reason)
    except (AttributeError, TypeError, ValueError):
        pass
    return {"http_status": status, "reasons": sorted(set(reasons)) or [type(exc).__name__]}


def credentials_from_environment(environ=None):
    """Reuse the already granted access token; never pair a refresh token with a guessed client."""
    from google.oauth2.credentials import Credentials
    env = os.environ if environ is None else environ
    user_json = env.get("SELFAUDIT_DRIVE_AUTHORIZED_USER_JSON")
    if user_json:
        try:
            info = json.loads(user_json)
            creds = Credentials.from_authorized_user_info(info)
        except Exception:
            raise SafeDriveError(None, ["INVALID_AUTHORIZED_USER_SECRET"]) from None
        return creds, {"source": "authorized_user_secret", "matching_refresh_client_available": True}
    try:
        config = configparser.ConfigParser(interpolation=None)
        config.read_string(env.get("RCLONE_CONFIG_TEXT", ""))
        remote = env.get("RCLONE_REMOTE_NAME", "").rstrip(":")
        section = config[remote]
        token = json.loads(section.get("token", "{}"))
        access = token.get("access_token")
        if not access:
            raise ValueError("missing access token")
        client_id = section.get("client_id", "").strip()
        client_secret = section.get("client_secret", "").strip()
        paired = bool(client_id and client_secret and token.get("refresh_token"))
        # A missing client means access-token-only: an HTTP 401 is exposed safely;
        # a new read-only OAuth grant is needed rather than guessing another client.
        creds = Credentials(token=access, refresh_token=token.get("refresh_token") if paired else None,
            client_id=client_id or None, client_secret=client_secret or None,
            token_uri="https://oauth2.googleapis.com/token" if paired else None)
    except Exception:
        raise SafeDriveError(None, ["MISSING_USABLE_OAUTH_SECRET"]) from None
    return creds, {"source": "existing_grant_from_secret_no_rclone_execution",
                   "matching_refresh_client_available": paired}


def make_service():
    import google_auth_httplib2
    import httplib2
    from googleapiclient.discovery import build
    credentials, identity = credentials_from_environment()
    # Do not hide a missing matching client behind implicit refresh attempts.
    http = google_auth_httplib2.AuthorizedHttp(credentials, http=httplib2.Http(timeout=25),
        refresh_status_codes=(401,) if identity["matching_refresh_client_available"] else (),
        max_refresh_attempts=1)
    return build("drive", "v3", http=http, cache_discovery=False, static_discovery=True), identity


def historical_files_unchanged(root="/assets"):
    root = Path(root)
    receipt = root / "receipts/STAGING.json"
    if sha256(receipt) != OLD_STAGING_SHA:
        raise SafeDriveError(None, ["HISTORICAL_RECEIPT_CHANGED"])
    files = json.loads(receipt.read_text())["files"]
    if len(files) != 17:
        raise SafeDriveError(None, ["HISTORICAL_FILE_COUNT_CHANGED"])
    for record in files:
        if sha256(root / record["path"]) != record["sha256"]:
            raise SafeDriveError(None, ["HISTORICAL_PAYLOAD_CHANGED"])
    return {"count": len(files), "receipt_sha256": OLD_STAGING_SHA, "all_file_hashes_match": True}


def download_file(service, file_id, target, expected_name, expected_size=None):
    from googleapiclient.http import MediaIoBaseDownload
    target = Path(target)
    # Keep receipts outside prepare_acdc's '*.nii*' glob.
    proof = target.parent / ".verified" / (target.name + ".json")
    if target.is_file() and proof.is_file():
        prior = json.loads(proof.read_text())
        if prior["file_id"] == file_id and prior["name"] == expected_name and sha256(target) == prior["sha256"]:
            if target.stat().st_size == prior["bytes"] and (expected_size is None or prior["bytes"] == expected_size):
                return dict(prior, reused_verified_file=True)
        raise SafeDriveError(None, ["EXISTING_PAYLOAD_IDENTITY_MISMATCH"])
    meta = service.files().get(fileId=file_id, supportsAllDrives=True,
        fields="id,name,size,md5Checksum,sha1Checksum,sha256Checksum,parents,capabilities(canDownload)").execute(num_retries=0)
    if meta["id"] != file_id or meta["name"] != expected_name or not meta.get("capabilities", {}).get("canDownload", False):
        raise SafeDriveError(None, ["SOURCE_IDENTITY_OR_DOWNLOAD_PERMISSION_MISMATCH"])
    size = int(meta["size"])
    if expected_size is not None and expected_size != size:
        raise SafeDriveError(None, ["SOURCE_SIZE_MISMATCH"])
    target.parent.mkdir(parents=True, exist_ok=True)
    recovering_complete_payload = target.is_file()
    part = target if recovering_complete_payload else target.with_name(target.name + ".part")
    started = time.monotonic()
    if not recovering_complete_payload:
        with part.open("wb") as stream:
            downloader = MediaIoBaseDownload(stream, service.files().get_media(fileId=file_id, supportsAllDrives=True), chunksize=8 << 20)
            done = False
            while not done:
                _, done = downloader.next_chunk(num_retries=0)
                if time.monotonic() - started > 180:
                    raise SafeDriveError(None, ["SINGLE_FILE_DEADLINE"])
            stream.flush()
            os.fsync(stream.fileno())
    if part.stat().st_size != size:
        raise SafeDriveError(None, ["DOWNLOADED_SIZE_MISMATCH"])
    checksums = {}
    for name, key in (("md5", "md5Checksum"), ("sha1", "sha1Checksum"), ("sha256", "sha256Checksum")):
        if meta.get(key):
            h = hashlib.new(name)
            with part.open("rb") as f:
                for block in iter(lambda: f.read(8 << 20), b""):
                    h.update(block)
            if h.hexdigest() != meta[key]:
                raise SafeDriveError(None, ["SOURCE_CHECKSUM_MISMATCH"])
            checksums[name] = h.hexdigest()
    digest = sha256(part)
    if recovering_complete_payload and not checksums:
        raise SafeDriveError(None, ["UNVERIFIED_EXISTING_TARGET_NO_SOURCE_CHECKSUM"])
    if not recovering_complete_payload:
        os.replace(part, target)
    record = {"file_id": file_id, "name": expected_name, "bytes": size, "sha256": digest,
              "source_checksums": checksums, "source_checksums_available": bool(checksums),
              "path": str(target), "reused_verified_file": recovering_complete_payload}
    atomic_json(proof, record)
    return record


def smoke(namespace):
    import importlib.metadata
    base = Path(namespace)
    base.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "self_audit.drive_api.smoke.v1", "status": "RUNNING",
        "gpu_allocated": False, "rclone_called": False, "dataset_through_mac": False,
        "source_sha256": sha256(__file__),
        "stage_container": os.environ.get("MODAL_TASK_ID"),
        "package_versions": {p: importlib.metadata.version(p) for p in
            ("google-api-python-client", "google-auth", "google-auth-httplib2")}}
    started = time.monotonic()
    try:
        receipt["historical_preservation"] = historical_files_unchanged()
        service, identity = make_service()
        receipt["credential_identity"] = identity
        receipt["file"] = download_file(service, SMOKE_FILE_ID, base / "raw/patient007" / SMOKE_NAME, SMOKE_NAME, SMOKE_SIZE)
        receipt["status"] = "COPIED_PENDING_FRESH_CONTAINER"
    except Exception as exc:
        receipt.update(status="BLOCKED", error=sanitized_error(exc))
    receipt["elapsed_seconds"] = time.monotonic() - started
    atomic_json(base / "API_SMOKE.json", receipt)
    return receipt


def verify_smoke(namespace):
    base = Path(namespace)
    receipt = json.loads((base / "API_SMOKE.json").read_text())
    if receipt["status"] not in ("COPIED_PENDING_FRESH_CONTAINER", "PASS"):
        return receipt
    if os.environ.get("MODAL_TASK_ID") == receipt["stage_container"]:
        raise SafeDriveError(None, ["VERIFIER_MUST_BE_NEW_CONTAINER"])
    if any(k in os.environ for k in ("RCLONE_CONFIG_TEXT", "SELFAUDIT_DRIVE_AUTHORIZED_USER_JSON")):
        raise SafeDriveError(None, ["VERIFIER_MUST_NOT_RECEIVE_DRIVE_SECRET"])
    f = receipt["file"]
    if Path(f["path"]).stat().st_size != SMOKE_SIZE or sha256(f["path"]) != f["sha256"]:
        raise SafeDriveError(None, ["PERSISTED_PAYLOAD_MISMATCH"])
    receipt.update(status="PASS", verification_container=os.environ.get("MODAL_TASK_ID"),
        persistence_verified=True, historical_preservation=historical_files_unchanged())
    atomic_json(base / "API_SMOKE.json", receipt)
    return receipt


def list_files(service, query):
    """Bounded pagination of a declared folder; never scans the Drive root."""
    result, token, seen = [], None, set()
    for _ in range(10):
        page = service.files().list(q=query, pageSize=1000, pageToken=token,
            fields="nextPageToken,files(id,name,size,mimeType)",
            supportsAllDrives=True, includeItemsFromAllDrives=True).execute(num_retries=0)
        result.extend(page.get("files", []))
        token = page.get("nextPageToken")
        if not token:
            return result
        if token in seen:
            raise SafeDriveError(None, ["REPEATED_PAGINATION_TOKEN"])
        seen.add(token)
    raise SafeDriveError(None, ["INVENTORY_PAGE_BUDGET_EXCEEDED"])


def cohort_plan(service, split):
    from .modal_data_prep import L8, download_plan
    folders = list_files(service, f"'{TRAINING_FOLDER_ID}' in parents and trashed = false and mimeType = 'application/vnd.google-apps.folder'")
    needed = set(split["train"] + split["val"])
    by_name = {}
    for f in folders:
        if f["name"] in needed:
            if f["name"] in by_name:
                raise SafeDriveError(None, ["DUPLICATE_PATIENT_FOLDER"])
            by_name[f["name"]] = f["id"]
    if set(by_name) != needed:
        raise SafeDriveError(None, ["INCOMPLETE_LOCKED_COHORT"])
    rows = []
    reference_ids = set(L8 + split["val"])
    for pid in sorted(needed):
        name_filter = f"(name = '{pid}_4d.nii' or name = '{pid}_4d.nii.gz'"
        if pid in reference_ids:
            name_filter += f" or name contains '{pid}_frame'"
        name_filter += ")"
        for f in list_files(service, f"'{by_name[pid]}' in parents and trashed = false and {name_filter}"):
            if f["mimeType"] != "application/vnd.google-apps.folder":
                rows.append({"Path": pid + "/" + f["name"], "Size": int(f["size"]), "ID": f["id"], "IsDir": False})
    return download_plan(rows, split)


def transfer_and_prepare(namespace, commit, repo="/repo"):
    from .modal_data_prep import L8, SPLIT_SHA, locked_split, validate_prepared
    import sys
    sys.path.insert(0, repo)
    from prepare_acdc import prepare
    base = Path(namespace)
    smoke_receipt = json.loads((base / "API_SMOKE.json").read_text())
    if smoke_receipt["status"] != "PASS" or not smoke_receipt.get("persistence_verified"):
        raise SafeDriveError(None, ["SMOKE_MUST_PASS_FRESH_CONTAINER_VERIFICATION"])
    historical_files_unchanged()
    split_path = Path(repo) / "splits/acdc_reviewed_locked_train80_val18.json"
    split = locked_split(split_path)
    receipt = {"schema": "self_audit.fresh_s0.data.v1", "status": "PREPARING", "namespace": str(base),
        "seed": 42, "label_budget": 8, "labeled_ids": L8, "split_sha256": SPLIT_SHA,
        "stage_container": os.environ.get("MODAL_TASK_ID"), "teacher_required": False,
        "rclone_called": False, "dataset_through_mac": False, "historical_staging_sha256": OLD_STAGING_SHA,
        "source_files": {p: sha256(Path(repo) / p) for p in
            ("prepare_acdc.py", "self_audit/data.py", "self_audit/modal_data_prep.py", "self_audit/drive_api_staging.py")}}
    atomic_json(base / "DATA_PREP.json", receipt)
    started = time.monotonic()
    try:
        service, identity = make_service()
        receipt["credential_identity"] = identity
        plan_path = base / "API_DOWNLOAD_PLAN.json"
        if plan_path.exists():
            saved = json.loads(plan_path.read_text())
            if saved["split_sha256"] != SPLIT_SHA or saved["folder_id"] != TRAINING_FOLDER_ID:
                raise SafeDriveError(None, ["SAVED_PLAN_PROTOCOL_MISMATCH"])
            # Re-run the exact allowlist checks even for a saved inventory.
            from .modal_data_prep import download_plan
            plan = download_plan([{"Path": x["path"], "Size": x["size"], "ID": x["drive_id"]}
                                  for x in saved["files"]], split)
        else:
            plan = cohort_plan(service, split)
            atomic_json(plan_path, {"folder_id": TRAINING_FOLDER_ID, "split_sha256": SPLIT_SHA, "files": plan})
            commit()
        records = []
        raw = base / "raw"
        for index, item in enumerate(plan, 1):
            target = raw / item["path"]
            records.append(download_file(service, item["drive_id"], target, target.name, item["size"]))
            atomic_json(base / "API_TRANSFER.json", {"completed_files": index, "total_files": len(plan), "files": records})
            if index % 8 == 0 or index == len(plan):
                print(json.dumps({"transfer": f"{index}/{len(plan)}", "verified_bytes": sum(x["bytes"] for x in records)}), flush=True)
                commit()
        expected = {x["path"] for x in plan}
        actual = {str(p.relative_to(raw)) for p in raw.rglob("*") if p.is_file() and ".verified" not in p.parts}
        if actual != expected:
            raise SafeDriveError(None, ["RAW_PAYLOAD_ALLOWLIST_MISMATCH"])
        atomic_json(base / "RAW_FILES.json", [dict(x, sha256=sha256(raw / x["path"])) for x in plan])
        prepared = base / "prepared"
        if not (prepared / "PREPARATION.json").is_file():
            # Preserve interrupted outputs; each canonical preparation requires a new directory.
            if prepared.exists():
                preserved = base / f"prepared_incomplete_{time.time_ns()}"
                os.replace(prepared, preserved)
            prepare(raw, split_path, prepared, 8, 42, evaluation=True)
        receipt.update(validate_prepared(raw, prepared, split, Path(repo) / "historical_val18_native.json"))
        receipt.update(train_manifest=str(prepared / "train.json"), train_manifest_sha256=sha256(prepared / "train.json"),
            val_manifest=str(prepared / "val18_fresh_native.json"), raw_file_count=len(plan),
            prepared_image_count=98, prepared_mask_count=52, unlabeled_mask_downloads=0, unlabeled_mask_reads=0,
            train_count=80, val_count=18, reused_verified_files=sum(x["reused_verified_file"] for x in records))
        comparison = receipt["train_manifest_comparison"]
        receipt["status"] = "PREPARED_PENDING_FRESH_CONTAINER" if comparison["exact_bytes"] or comparison["path_only_witness"] else "BLOCKED_HISTORICAL_TRAIN_COMPARISON"
    except Exception as exc:
        receipt.update(status="BLOCKED", error=sanitized_error(exc))
    receipt["elapsed_seconds"] = time.monotonic() - started
    receipt["historical_preservation"] = historical_files_unchanged()
    receipt["historical_receipt_unchanged"] = True
    atomic_json(base / "DATA_PREP.json", receipt)
    commit()
    return {k: receipt.get(k) for k in ("status", "namespace", "error", "train_manifest_sha256", "val_manifest_sha256", "elapsed_seconds")}


def historical_kd_prerequisites(root="/assets"):
    """Missing physical prerequisites block only S1/S2, never S0."""
    from .asset_staging import PINS
    root = Path(root)
    wanted = {
        "teacher_manifest": (root / "b8_recovery/full/final/MANIFEST.json", PINS["teacher_manifest"]),
        "teacher_raw_student_payload": (root / "b8_recovery/full/final/weights.pt", PINS["teacher_weights"]),
        "TRAIN_U_cache_manifest": (root / "compact_teacher_b8/CACHE_MANIFEST.json", "783624211e760fbeb2c3a80ded7b8dacb9d1407f0dcc1e484010cda2c343897b"),
    }
    evidence = {}
    for name, (path, expected) in wanted.items():
        actual = sha256(path) if path.is_file() else None
        evidence[name] = {"path": str(path), "expected_sha256": expected, "actual_sha256": actual,
                          "status": "MATCH" if actual == expected else "MISSING" if actual is None else "MISMATCH"}
    missing = [k for k, v in evidence.items() if v["status"] != "MATCH"]
    return {"status": "BLOCKED" if missing else "REQUIRES_FULL_CACHE_AND_LINEAGE_PREFLIGHT",
            "missing_or_mismatched": missing, "evidence": evidence,
            "raw_student_required": True, "reproduced_B8_or_B52_allowed": False}
