"""Offline failure witnesses for the CPU Drive boundary; never accesses patient data."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import pytest

from self_audit import drive_api_staging as api
from self_audit.asset_staging import atomic_json, sha256
from self_audit.modal_data_prep import L8, locked_split

ROOT = Path(__file__).resolve().parents[1]


class Request:
    def __init__(self, value):
        self.value = value

    def execute(self, *, num_retries):
        assert num_retries == 0
        return self.value


class Files:
    def __init__(self, payload=b"fixture cine", **metadata):
        self.payload = payload
        self.metadata = dict(id="file-id", name="patient007_4d.nii", size=str(len(payload)),
                             md5Checksum=hashlib.md5(payload).hexdigest(), capabilities={"canDownload": True})
        self.metadata.update(metadata)
        self.calls = []

    def files(self):
        return self

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        return Request(self.metadata)

    def get_media(self, **kwargs):
        self.calls.append(("get_media", kwargs))
        return self.payload


@pytest.fixture
def download_client(monkeypatch):
    module = ModuleType("googleapiclient.http")

    class Download:
        def __init__(self, stream, payload, *, chunksize):
            self.stream, self.payload = stream, payload
            assert chunksize == 8 << 20

        def next_chunk(self, *, num_retries):
            assert num_retries == 0
            assert self.stream.name.endswith(".part")
            self.stream.write(self.payload)
            return None, True

    module.MediaIoBaseDownload = Download
    monkeypatch.setitem(sys.modules, "googleapiclient.http", module)
    return Files()


def test_download_atomic_source_checksums_and_idempotent_reuse(tmp_path, download_client):
    target = tmp_path / "patient007_4d.nii"
    result = api.download_file(download_client, "file-id", target, target.name, len(download_client.payload))
    assert result["sha256"] == sha256(target)
    assert result["source_checksums"]["md5"] == hashlib.md5(download_client.payload).hexdigest()
    assert not target.with_name(target.name + ".part").exists()
    assert list(tmp_path.glob("patient007_4d.nii*")) == [target]
    assert (tmp_path / ".verified" / (target.name + ".json")).is_file()
    download_client.calls.clear()
    reused = api.download_file(download_client, "file-id", target, target.name, len(download_client.payload))
    assert reused["reused_verified_file"] and download_client.calls == []


@pytest.mark.parametrize("change,reason", [
    ({"size": "999"}, "DOWNLOADED_SIZE_MISMATCH"),
    ({"md5Checksum": "0" * 32}, "SOURCE_CHECKSUM_MISMATCH"),
    ({"name": "wrong.nii"}, "SOURCE_IDENTITY_OR_DOWNLOAD_PERMISSION_MISMATCH"),
    ({"capabilities": {"canDownload": False}}, "SOURCE_IDENTITY_OR_DOWNLOAD_PERMISSION_MISMATCH"),
])
def test_invalid_source_never_publishes_target(tmp_path, download_client, change, reason):
    download_client.metadata.update(change)
    target = tmp_path / "patient007_4d.nii"
    with pytest.raises(api.SafeDriveError, match=reason):
        api.download_file(download_client, "file-id", target, target.name)
    assert not target.exists()
    assert not (tmp_path / ".verified" / (target.name + ".json")).exists()


def test_size_lock_rejects_before_media_read(tmp_path, download_client):
    with pytest.raises(api.SafeDriveError, match="SOURCE_SIZE_MISMATCH"):
        api.download_file(download_client, "file-id", tmp_path / "patient007_4d.nii", "patient007_4d.nii", 123)
    assert [x[0] for x in download_client.calls] == ["get"]


def test_corrupt_previously_verified_payload_is_not_overwritten(tmp_path, download_client):
    target = tmp_path / "patient007_4d.nii"
    api.download_file(download_client, "file-id", target, target.name)
    target.write_bytes(b"corrupt")
    download_client.calls.clear()
    with pytest.raises(api.SafeDriveError, match="EXISTING_PAYLOAD_IDENTITY_MISMATCH"):
        api.download_file(download_client, "file-id", target, target.name)
    assert target.read_bytes() == b"corrupt" and download_client.calls == []


def test_crash_after_rename_recovers_by_source_checksum_without_redownload(tmp_path, download_client):
    target = tmp_path / "patient007_4d.nii"
    target.write_bytes(download_client.payload)
    result = api.download_file(download_client, "file-id", target, target.name)
    assert result["reused_verified_file"]
    assert [x[0] for x in download_client.calls] == ["get"]


def test_existing_unverified_target_without_source_checksum_blocks(tmp_path, download_client):
    target = tmp_path / "patient007_4d.nii"
    target.write_bytes(download_client.payload)
    download_client.metadata.pop("md5Checksum")
    with pytest.raises(api.SafeDriveError, match="UNVERIFIED_EXISTING_TARGET_NO_SOURCE_CHECKSUM"):
        api.download_file(download_client, "file-id", target, target.name)


def test_google_error_excludes_message_and_uri():
    exc = RuntimeError("secret-token-value")
    exc.resp = SimpleNamespace(status=403)
    exc.content = json.dumps({"error": {"message": "secret-token-value", "errors": [
        {"reason": "rateLimitExceeded", "message": "secret-token-value"}]}})
    result = api.sanitized_error(exc)
    assert result == {"http_status": 403, "reasons": ["rateLimitExceeded"]}
    assert "secret-token-value" not in json.dumps(result)


@pytest.fixture
def fake_credentials(monkeypatch):
    module = ModuleType("google.oauth2.credentials")

    class Credentials:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @classmethod
        def from_authorized_user_info(cls, info):
            if not all(info.get(k) for k in ("client_id", "client_secret", "refresh_token")):
                raise ValueError("secret parse details")
            return cls(**info)

    module.Credentials = Credentials
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", module)
    return Credentials


def test_no_guessed_oauth_client_or_implicit_refresh(fake_credentials):
    token = json.dumps({"access_token": "fixture-access", "refresh_token": "fixture-refresh"})
    env = {"RCLONE_CONFIG_TEXT": "[KhanhDrive]\ntoken = " + token, "RCLONE_REMOTE_NAME": "KhanhDrive:"}
    creds, identity = api.credentials_from_environment(env)
    assert creds.kwargs["token"] == "fixture-access"
    assert creds.kwargs["refresh_token"] is None and creds.kwargs["client_id"] is None
    assert not identity["matching_refresh_client_available"]
    assert "fixture-access" not in json.dumps(identity)


def test_explicit_authorized_user_has_priority_and_safe_errors(fake_credentials):
    creds, identity = api.credentials_from_environment({
        "SELFAUDIT_DRIVE_AUTHORIZED_USER_JSON": json.dumps({
            "client_id": "fixture-id", "client_secret": "fixture-secret", "refresh_token": "fixture-refresh"}),
        "RCLONE_CONFIG_TEXT": "malformed secret config"})
    assert creds.kwargs["client_id"] == "fixture-id" and identity["matching_refresh_client_available"]
    with pytest.raises(api.SafeDriveError, match="INVALID_AUTHORIZED_USER_SECRET") as error:
        api.credentials_from_environment({"SELFAUDIT_DRIVE_AUTHORIZED_USER_JSON": "private-value"})
    assert "private-value" not in str(error.value)


def test_list_pagination_is_bounded():
    calls = []
    class Service:
        def files(self):
            return self
        def list(self, **kwargs):
            calls.append(kwargs)
            return Request({"files": [], "nextPageToken": "same-token"})
    with pytest.raises(api.SafeDriveError, match="REPEATED_PAGINATION_TOKEN"):
        api.list_files(Service(), "declared-folder-query")
    assert len(calls) == 2 and all(c["q"] == "declared-folder-query" for c in calls)


def test_live_api_inventory_never_requests_u_reference_metadata(monkeypatch):
    split = locked_split(ROOT / "splits/acdc_reviewed_locked_train80_val18.json")
    ids = split["train"] + split["val"]
    queries = []
    refs = set(L8 + split["val"])
    def listing(service, query):
        queries.append(query)
        if api.TRAINING_FOLDER_ID in query:
            return [{"id": pid, "name": pid} for pid in ids]
        pid = query.split("'")[1]
        names = [pid + "_4d.nii"]
        if pid in refs:
            names += [f"{pid}_frame{f:02d}{s}.nii" for f in (1, 12) for s in ("", "_gt")]
        else:
            assert "frame" not in query and "_gt" not in query
        return [{"id": n, "name": n, "size": "100", "mimeType": "application/octet-stream"} for n in names]
    monkeypatch.setattr(api, "list_files", listing)
    plan = api.cohort_plan(None, split)
    assert len(plan) == 202 and len(queries) == 99
    assert sum(x["kind"] == "cine" for x in plan) == 98
    assert all(x["path"].split("/")[0] in refs for x in plan if "_gt" in x["path"])
    assert not any("patient055" in x["path"] or "patient076" in x["path"] for x in plan)


def test_historical_preservation_checks_every_file_and_receipt(tmp_path, monkeypatch):
    records = []
    for n in range(17):
        file = tmp_path / f"asset{n}"
        file.write_bytes(bytes([n]))
        records.append({"path": file.name, "sha256": sha256(file)})
    receipt = tmp_path / "receipts/STAGING.json"
    atomic_json(receipt, {"files": records})
    monkeypatch.setattr(api, "OLD_STAGING_SHA", sha256(receipt))
    assert api.historical_files_unchanged(tmp_path)["count"] == 17
    (tmp_path / "asset16").write_bytes(b"tampered")
    with pytest.raises(api.SafeDriveError, match="HISTORICAL_PAYLOAD_CHANGED"):
        api.historical_files_unchanged(tmp_path)


def test_new_container_readback_requires_persistence_and_no_secret(tmp_path, monkeypatch):
    target = tmp_path / "patient007_4d.nii"
    target.write_bytes(b"fixture")
    monkeypatch.setattr(api, "SMOKE_SIZE", target.stat().st_size)
    monkeypatch.setattr(api, "historical_files_unchanged", lambda: {"count": 17})
    atomic_json(tmp_path / "API_SMOKE.json", {"status": "COPIED_PENDING_FRESH_CONTAINER",
        "stage_container": "stage", "file": {"path": str(target), "sha256": sha256(target)}})
    monkeypatch.setenv("MODAL_TASK_ID", "stage")
    with pytest.raises(api.SafeDriveError, match="VERIFIER_MUST_BE_NEW_CONTAINER"):
        api.verify_smoke(tmp_path)
    monkeypatch.setenv("MODAL_TASK_ID", "new-container")
    monkeypatch.setenv("SELFAUDIT_DRIVE_AUTHORIZED_USER_JSON", "fixture")
    with pytest.raises(api.SafeDriveError, match="VERIFIER_MUST_NOT_RECEIVE_DRIVE_SECRET"):
        api.verify_smoke(tmp_path)
    monkeypatch.delenv("SELFAUDIT_DRIVE_AUTHORIZED_USER_JSON")
    monkeypatch.delenv("RCLONE_CONFIG_TEXT", raising=False)
    assert api.verify_smoke(tmp_path)["persistence_verified"]
    target.write_bytes(b"changed")
    with pytest.raises(api.SafeDriveError, match="PERSISTED_PAYLOAD_MISMATCH"):
        api.verify_smoke(tmp_path)


def test_missing_teacher_blocks_kd_without_claiming_s0_blocked(tmp_path):
    result = api.historical_kd_prerequisites(tmp_path)
    assert result["status"] == "BLOCKED"
    assert "teacher_raw_student_payload" in result["missing_or_mismatched"]
    assert "TRAIN_U_cache_manifest" in result["missing_or_mismatched"]
    assert result["evidence"]["teacher_raw_student_payload"]["expected_sha256"] == (
        "0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6")
    assert not result["reproduced_B8_or_B52_allowed"]


def test_api_modal_resource_and_secret_boundary():
    tree = ast.parse((ROOT / "scripts/modal_drive_api.py").read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("verify_smoke", "verify_data", "smoke_s0", "train_s0", "pipeline"):
        options = {k.arg: k.value for k in functions[name].decorator_list[0].keywords}
        assert "secrets" not in options
    gpu = {k.arg: ast.literal_eval(k.value) for k in functions["train_s0"].decorator_list[0].keywords
           if k.arg in ("gpu", "retries", "max_containers", "timeout")}
    assert gpu == {"gpu": "A100-40GB", "timeout": 7200, "retries": 0, "max_containers": 1}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "apt_install"  # no rclone binary in either image


def test_locked_s1_s2_only_change_crop_and_identity():
    s1 = json.loads((ROOT / "configs/fresh_s0/s1_plain_kd.json").read_text())
    s2 = json.loads((ROOT / "configs/fresh_s0/s2_plain_kd.json").read_text())
    assert s1["crop_size"] == 160 and s2["crop_size"] == 256
    for cfg in (s1, s2):
        assert (cfg["steps"], cfg["warmup_steps"], cfg["seed"]) == (10000, 1000, 42)
        assert cfg["labeled_batch_size"] == cfg["unlabeled_batch_size"] == 8
        assert cfg["model"]["widths"] == [32, 64, 128, 256]
    assert {k for k in s1 if s1[k] != s2[k]} == {"crop_size", "experiment"}


def test_progress_logs_every_hundred_and_skips_resumed_prefix(tmp_path, capsys):
    from self_audit.modal_s0 import progress
    path = tmp_path / "TRAIN_LOG.jsonl"
    path.write_text("".join(json.dumps({"step": n, "supervised_loss": 1.0}) + "\n" for n in range(400)))
    class Stop:
        def wait(self, seconds):
            return True
    progress(path, Stop(), resume_step=100)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["progress"] for r in rows] == ["200/10000", "300/10000", "400/10000"]


def test_remote_controller_stops_on_auth_failure_and_prevents_duplicate(tmp_path, monkeypatch):
    tree = ast.parse((ROOT / "scripts/modal_drive_api.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pipeline")
    function.decorator_list = []
    namespace = ast.Module(body=[function], type_ignores=[])
    events = []
    class Stage:
        def with_options(self, **kwargs):
            assert list(kwargs) == ["secrets"]
            return self
        def remote(self, path):
            events.append("cpu-smoke")
            return {"status": "BLOCKED", "error": {"http_status": 401, "reasons": ["authError"]}}
    monkeypatch.setattr(api, "historical_kd_prerequisites", lambda: {"status": "BLOCKED"})
    context = {"Path": Path, "json": json, "stage_smoke": Stage(),
        "volume": SimpleNamespace(commit=lambda: events.append("commit")),
        "modal": SimpleNamespace(Secret=SimpleNamespace(from_name=lambda name: "secret-handle"))}
    exec(compile(namespace, "controller-fixture", "exec"), context)
    result = context["pipeline"](str(tmp_path), "readonly-secret", "source-sha", {}, {})
    assert result["DATA_PREP"] == "BLOCKED" and result["S0"] == "BLOCKED_DATA_PREP"
    assert result["S1"] == result["S2"] == "BLOCKED"
    assert events.count("cpu-smoke") == 1
    with pytest.raises(FileExistsError):
        context["pipeline"](str(tmp_path), "readonly-secret", "source-sha", {}, {})
    assert events.count("cpu-smoke") == 1


def test_api_transfer_canonical_prepare_and_new_container_batch_on_fixture(tmp_path, monkeypatch, download_client):
    """98 manufactured cines; hashes pin this fixture only, never the historical data."""
    import nibabel as nib
    import numpy as np
    from prepare_acdc import prepare
    from self_audit import modal_data_prep as prep

    split_path = ROOT / "splits/acdc_reviewed_locked_train80_val18.json"
    split = locked_split(split_path)
    source = tmp_path / "fixture-source"
    refs = set(L8 + split["val"])
    rng = np.random.default_rng(91)
    affine = np.diag([1.2, 1.2, 5.0, 1.0])
    for pid in split["train"] + split["val"]:
        folder = source / pid
        folder.mkdir(parents=True)
        cine = rng.integers(0, 1000, size=(16, 16, 4, 3), dtype=np.int16)
        nib.save(nib.Nifti1Image(cine, affine), folder / f"{pid}_4d.nii")
        if pid in refs:
            for frame in (1, 3):
                nib.save(nib.Nifti1Image(cine[..., frame - 1], affine), folder / f"{pid}_frame{frame:02d}.nii")
                mask = rng.integers(0, 4, size=(16, 16, 4), dtype=np.uint8)
                nib.save(nib.Nifti1Image(mask, affine), folder / f"{pid}_frame{frame:02d}_gt.nii")
    witness = tmp_path / "fixture-prepared-witness"
    prepare(source, split_path, witness, 8, 42, evaluation=True)
    historical = json.loads((witness / "development.json").read_text())
    for rec in historical["patients"]:
        rec.update(affine=affine.tolist(), geometry_source_image={
            "sha256": sha256(source / rec["id"] / (rec["id"] + "_4d.nii"))})
    history_path = tmp_path / "fixture-native-history.json"
    atomic_json(history_path, historical)
    monkeypatch.setattr(prep, "HISTORICAL_VAL_SHA", sha256(history_path))
    monkeypatch.setattr(prep, "HISTORICAL_TRAIN_SHA", sha256(witness / "train.json"))
    compare = prep.manifest_comparison
    monkeypatch.setattr(prep, "manifest_comparison", lambda p: compare(p, witness / "train.json"))
    validate = prep.validate_prepared
    monkeypatch.setattr(prep, "validate_prepared", lambda r, p, s, h: validate(r, p, s, history_path))

    rows = [{"Path": str(p.relative_to(source)), "Size": p.stat().st_size,
             "ID": str(p.relative_to(source))} for p in source.rglob("*.nii")]
    plan = prep.download_plan(rows, split)
    class Source(Files):
        def get(self, **kwargs):
            p = source / kwargs["fileId"]
            data = p.read_bytes()
            return Request(dict(id=kwargs["fileId"], name=p.name, size=str(len(data)),
                md5Checksum=hashlib.md5(data).hexdigest(), capabilities={"canDownload": True}))
        def get_media(self, **kwargs):
            return (source / kwargs["fileId"]).read_bytes()
    monkeypatch.setattr(api, "make_service", lambda: (Source(), {"fixture": True}))
    monkeypatch.setattr(api, "cohort_plan", lambda service, s: plan)
    monkeypatch.setattr(api, "historical_files_unchanged", lambda: {"count": 17, "fixture": True})
    base = tmp_path / "api-run"
    atomic_json(base / "API_SMOKE.json", {"status": "PASS", "persistence_verified": True})
    monkeypatch.setenv("MODAL_TASK_ID", "fixture-stage")
    opened_masks = []
    original_load = nib.load
    def observed_load(path, *args, **kwargs):
        if "_gt.nii" in str(path):
            pid = Path(path).parent.name
            assert pid in refs
            opened_masks.append(pid)
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(nib, "load", observed_load)
    commits = []
    result = api.transfer_and_prepare(base, lambda: commits.append("commit"), repo=str(ROOT))
    assert result["status"] == "PREPARED_PENDING_FRESH_CONTAINER"
    assert len(commits) >= 26 and set(opened_masks) == refs
    receipt = json.loads((base / "DATA_PREP.json").read_text())
    assert len(receipt["files"]) == 150 and receipt["raw_file_count"] == 202
    assert receipt["unlabeled_mask_downloads"] == receipt["unlabeled_mask_reads"] == 0
    assert receipt["train_manifest_comparison"]["exact_bytes"]
    monkeypatch.setenv("MODAL_TASK_ID", "fixture-new-container")
    monkeypatch.delenv("RCLONE_CONFIG_TEXT", raising=False)
    monkeypatch.delenv("SELFAUDIT_DRIVE_AUTHORIZED_USER_JSON", raising=False)
    original_sha = prep.sha256
    monkeypatch.setattr(prep, "sha256", lambda p: prep.OLD_STAGING_SHA
        if str(p) == "/assets/receipts/STAGING.json" else original_sha(p))
    verified = prep.verify_fresh_container(base)
    assert verified["status"] == "PASS" and verified["cpu_batch"]["x_shape"] == [8, 3, 160, 160]
    assert set(verified["cpu_batch"]["label_reads"]).issubset(L8)
    # All payloads are reusable without network after a staged interruption.
    monkeypatch.setenv("MODAL_TASK_ID", "fixture-stage-retry")
    monkeypatch.setattr(api, "make_service", lambda: (None, {"fixture": True}))
    assert api.transfer_and_prepare(base, lambda: None, repo=str(ROOT))["status"] == "PREPARED_PENDING_FRESH_CONTAINER"
    assert json.loads((base / "DATA_PREP.json").read_text())["reused_verified_files"] == 202
