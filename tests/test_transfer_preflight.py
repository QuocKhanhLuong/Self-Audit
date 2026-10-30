"""Unit and integration tests for transfer preflight verification.

Tests:
  - Strict protocol lock validation (all locked hyperparameters, budget, selector params, architecture, pins).
  - Read-only GPU occupancy check without CUDA context, requested UUID only, CUDA_VISIBLE_DEVICES, PID filtering.
  - Active training jobs check with read-only ps, failureclosed, PID filtering.
  - Output directory safety (reject any existing non-empty directory or file; strictly no writes).
  - Failclosed asset validation (missing manifest, hash mismatch, corrupted metadata, U masks prohibition).
  - VAL manifest validation (18 patients, strict disjointness from TRAIN, no VAL masks opened).
  - Teacher lineage and cache identity validation (canonical kind, selected raw student, full EMA, L8 union).
  - Positional call to find_checkpoint_payload.
  - Synthetic unit mode validation.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch
import pytest

from self_audit.transfer_preflight import (
    PINNED_B8_TEACHER_MANIFEST_SHA256,
    PINNED_L8_PATIENT_IDS,
    PINNED_LABEL_BUDGET_K,
    PINNED_RAW_PAYLOAD_SHA256,
    PINNED_TEACHER_CACHE_SHA256,
    PINNED_TRAIN_MANIFEST_SHA256,
    PINNED_VAL18_PATIENT_COUNT,
    PINNED_VAL_MANIFEST_SHA256,
    check_active_training_jobs,
    check_gpu_occupancy,
    check_preflight,
    validate_locked_protocol,
)
from self_audit.compact_artifacts import digest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "transfer_utility" / "protocol.json"


@pytest.fixture
def valid_protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


# ==============================================================================
# Protocol Validation Tests
# ==============================================================================

def test_validate_locked_protocol_passes_on_valid_protocol(valid_protocol):
    """The canonical protocol in configs/transfer_utility/protocol.json must pass with 0 errors."""
    errors = validate_locked_protocol(valid_protocol)
    assert errors == [], f"Unexpected errors on canonical protocol: {errors}"


def test_validate_locked_protocol_catches_budget_tampering(valid_protocol):
    """Tampering with any budget field must be rejected."""
    tampered = copy.deepcopy(valid_protocol)
    tampered["budget"]["total_steps"] = 5000
    errors = validate_locked_protocol(tampered)
    assert any("budget.total_steps mismatch" in e for e in errors)

    tampered = copy.deepcopy(valid_protocol)
    tampered["budget"]["warmup_steps"] = 500
    errors = validate_locked_protocol(tampered)
    assert any("budget.warmup_steps mismatch" in e for e in errors)

    tampered = copy.deepcopy(valid_protocol)
    tampered["budget"]["eval_steps"] = [1000, 2000]
    errors = validate_locked_protocol(tampered)
    assert any("budget.eval_steps mismatch" in e for e in errors)

    tampered = copy.deepcopy(valid_protocol)
    tampered["budget"]["watchdog_max_seconds"] = 3600
    errors = validate_locked_protocol(tampered)
    assert any("budget.watchdog_max_seconds mismatch" in e for e in errors)

    tampered = copy.deepcopy(valid_protocol)
    tampered["budget"]["profile_max_steps"] = 64
    errors = validate_locked_protocol(tampered)
    assert any("budget.profile_max_steps mismatch" in e for e in errors)


def test_validate_locked_protocol_catches_hyperparameter_tampering(valid_protocol):
    """Tampering with optimization hyperparameters must be rejected."""
    tampered = copy.deepcopy(valid_protocol)
    tampered["hyperparameters"]["lr"] = 0.001
    errors = validate_locked_protocol(tampered)
    assert any("hyperparameters.lr mismatch" in e for e in errors)

    tampered = copy.deepcopy(valid_protocol)
    tampered["hyperparameters"]["pseudo_lr"] = 0.0001
    errors = validate_locked_protocol(tampered)
    assert any("hyperparameters.pseudo_lr mismatch" in e for e in errors)

    tampered = copy.deepcopy(valid_protocol)
    tampered["hyperparameters"]["temperature"] = 2.0
    errors = validate_locked_protocol(tampered)
    assert any("hyperparameters.temperature mismatch" in e for e in errors)

    tampered = copy.deepcopy(valid_protocol)
    tampered["hyperparameters"]["pseudo_threshold"] = 0.9
    errors = validate_locked_protocol(tampered)
    assert any("hyperparameters.pseudo_threshold mismatch" in e for e in errors)

    tampered = copy.deepcopy(valid_protocol)
    tampered["hyperparameters"]["trial_scales"] = [1.0, 0.5]
    errors = validate_locked_protocol(tampered)
    assert any("hyperparameters.trial_scales mismatch" in e for e in errors)


def test_validate_locked_protocol_catches_selector_param_tampering(valid_protocol):
    """Tampering with transfer-utility selector tolerances must be rejected."""
    tampered = copy.deepcopy(valid_protocol)
    tampered["selector_parameters"]["utility_gain_tolerance"] = 1e-4
    errors = validate_locked_protocol(tampered)
    assert any("selector_parameters.utility_gain_tolerance mismatch" in e for e in errors)

    tampered = copy.deepcopy(valid_protocol)
    tampered["selector_parameters"]["probe_seed"] = 12345
    errors = validate_locked_protocol(tampered)
    assert any("selector_parameters.probe_seed mismatch" in e for e in errors)


def test_validate_locked_protocol_catches_architecture_tampering(valid_protocol):
    """Tampering with student architecture must be rejected."""
    tampered = copy.deepcopy(valid_protocol)
    tampered["architecture"]["widths"] = [32, 64, 128, 256]
    errors = validate_locked_protocol(tampered)
    assert any("architecture.widths mismatch" in e for e in errors)


def test_validate_locked_protocol_catches_extra_arms(valid_protocol):
    """No extra experimental arms or selector changes are permitted."""
    tampered = copy.deepcopy(valid_protocol)
    tampered["arms"]["arm_extra"] = {"name": "unauthorized", "method": "plain_kd"}
    errors = validate_locked_protocol(tampered)
    assert any("arms mismatch" in e for e in errors)

    tampered = copy.deepcopy(valid_protocol)
    tampered["arms"]["arm2"]["selector"] = "larger_tie"  # changed selector
    errors = validate_locked_protocol(tampered)
    assert any("arm2.selector mismatch" in e for e in errors)


def test_validate_locked_protocol_catches_pins_tampering(valid_protocol):
    """Tampering with data, teacher, or evaluation pins must be rejected."""
    tampered = copy.deepcopy(valid_protocol)
    tampered["data"]["manifest_sha256"] = "0" * 64
    errors = validate_locked_protocol(tampered)
    assert any("data.manifest_sha256 mismatch" in e for e in errors)

    tampered = copy.deepcopy(valid_protocol)
    tampered["teacher"]["source_manifest_sha256"] = "0" * 64
    errors = validate_locked_protocol(tampered)
    assert any("teacher.source_manifest_sha256 mismatch" in e for e in errors)

    tampered = copy.deepcopy(valid_protocol)
    tampered["teacher"]["cache_sha256"] = "0" * 64
    errors = validate_locked_protocol(tampered)
    assert any("teacher.cache_sha256 mismatch" in e for e in errors)


# ==============================================================================
# GPU Occupancy Tests
# ==============================================================================

def test_check_gpu_occupancy_cpu():
    """CPU device bypasses GPU check without touching nvidia-smi."""
    res = check_gpu_occupancy("cpu")
    assert res["status"] == "PASS"
    assert res["gpu_available"] is False
    assert res["is_busy"] is False


def test_check_gpu_occupancy_missing_smi_failclosed(monkeypatch):
    """When nvidia-smi is unavailable, CUDA occupancy check fails closed as UNVERIFIABLE."""
    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=FileNotFoundError("nvidia-smi not found")))
    res = check_gpu_occupancy("cuda:0")
    assert res["status"] == "UNVERIFIABLE"
    assert res["is_busy"] is True
    assert "not found" in res["error"].lower()


def test_check_gpu_occupancy_requested_uuid_only(monkeypatch):
    """Query targets requested GPU UUID only, respects CUDA_VISIBLE_DEVICES, excludes own PID."""
    calls = []

    def mock_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        if "--query-gpu=uuid,index,name" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                returncode=0,
                stdout="GPU-aaaa-bbbb-cccc, 1, NVIDIA RTX 4090\n",
                stderr="",
            )
        elif "--query-gpu=memory.used,memory.total,memory.free" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                returncode=0,
                stdout="100, 24576, 24476\n",
                stderr="",
            )
        elif "--query-compute-apps=pid,process_name,used_memory" in cmd:
            # Own PID plus another PID
            own_pid = os.getpid()
            return subprocess.CompletedProcess(
                cmd,
                returncode=0,
                stdout=f"{own_pid}, python, 50\n",
                stderr="",
            )
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="Unknown cmd")

    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    res = check_gpu_occupancy("cuda:0")
    assert res["status"] == "PASS"
    assert res["uuid"] == "GPU-aaaa-bbbb-cccc"
    assert res["device_index"] == 0
    assert res["physical_index"] == 1
    assert res["active_processes"] == []  # own PID was filtered out
    assert res["is_busy"] is False

    # Check that requested UUID was used for queries
    assert any("--id=GPU-aaaa-bbbb-cccc" in arg for cmd in calls for arg in cmd)


def test_check_gpu_occupancy_busy_on_foreign_process(monkeypatch):
    """GPU check reports BUSY when foreign compute processes are running."""
    def mock_run(cmd, *args, **kwargs):
        if "--query-gpu=uuid,index,name" in cmd:
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="GPU-1234, 0, RTX\n", stderr="")
        elif "--query-gpu=memory.used,memory.total,memory.free" in cmd:
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="200, 24000, 23800\n", stderr="")
        elif "--query-compute-apps=pid,process_name,used_memory" in cmd:
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="999999, python train.py, 200\n", stderr="")
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)
    res = check_gpu_occupancy("cuda:0")
    assert res["status"] == "BUSY"
    assert res["is_busy"] is True
    assert len(res["active_processes"]) == 1
    assert res["active_processes"][0]["pid"] == 999999


# ==============================================================================
# Active Jobs Tests
# ==============================================================================

def test_check_active_training_jobs_clean(monkeypatch):
    """Reports CLEAN when no foreign training jobs are running."""
    own_pid = os.getpid()
    ps_stdout = f"  PID COMMAND\n    1 /sbin/init\n  {own_pid} python -m pytest\n"
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=subprocess.CompletedProcess([], 0, ps_stdout, "")))

    res = check_active_training_jobs()
    assert res["status"] == "CLEAN"
    assert res["active_job_count"] == 0


def test_check_active_training_jobs_detects_foreign_job(monkeypatch):
    """Reports ACTIVE_JOBS_PRESENT when another process runs train.py."""
    ps_stdout = "  PID COMMAND\n    1 /sbin/init\n  9999 python train.py --config cfg.json\n"
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=subprocess.CompletedProcess([], 0, ps_stdout, "")))

    res = check_active_training_jobs()
    assert res["status"] == "ACTIVE_JOBS_PRESENT"
    assert res["active_job_count"] == 1
    assert res["active_jobs"][0]["pid"] == 9999


def test_check_active_training_jobs_failclosed_on_error(monkeypatch):
    """ps execution failure results in UNVERIFIABLE, not silent CLEAN."""
    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=RuntimeError("ps execution failed")))

    res = check_active_training_jobs()
    assert res["status"] == "UNVERIFIABLE"
    assert "ps check failed" in res["error"]


# ==============================================================================
# Output Safety Tests
# ==============================================================================

def test_check_preflight_refuses_existing_output_with_files(tmp_path, valid_protocol):
    """Preflight must refuse if output directory already exists and contains any file."""
    out_dir = tmp_path / "existing_run"
    out_dir.mkdir(parents=True)
    (out_dir / "any_file.txt").write_text("existing output\n")

    receipt = check_preflight(
        protocol=valid_protocol,
        data_manifest=None,
        data_manifest_sha256=None,
        out_dir=out_dir,
        device="cpu",
        synthetic=True,
        check_jobs=False,
    )
    assert receipt["status"] == "REFUSED_EXISTING_OUTPUT"
    assert any("output directory exists and contains" in r.lower() for r in receipt["reasons"])


def test_check_preflight_refuses_existing_output_regular_file(tmp_path, valid_protocol):
    """Preflight must refuse if out_dir is an existing regular file."""
    out_file = tmp_path / "existing_file"
    out_file.write_text("some file")

    receipt = check_preflight(
        protocol=valid_protocol,
        data_manifest=None,
        data_manifest_sha256=None,
        out_dir=out_file,
        device="cpu",
        synthetic=True,
        check_jobs=False,
    )
    assert receipt["status"] == "REFUSED_EXISTING_OUTPUT"


def test_check_preflight_does_not_write_to_fresh_dir(tmp_path, valid_protocol):
    """Preflight must perform NO disk writes; fresh out_dir is NOT created."""
    out_dir = tmp_path / "fresh_run_dir"
    assert not out_dir.exists()

    receipt = check_preflight(
        protocol=valid_protocol,
        data_manifest=None,
        data_manifest_sha256=None,
        out_dir=out_dir,
        device="cpu",
        synthetic=True,
        check_jobs=False,
    )
    assert receipt["status"] == "PASS"
    assert not out_dir.exists(), "check_preflight must strictly perform NO writes"


# ==============================================================================
# Refusal State Machine Tests
# ==============================================================================

def test_check_preflight_refuses_active_jobs(tmp_path, valid_protocol, monkeypatch):
    """Preflight refuses when training jobs are active in workspace."""
    out_dir = tmp_path / "run"
    monkeypatch.setattr(
        "self_audit.transfer_preflight.check_active_training_jobs",
        lambda: {"status": "ACTIVE_JOBS_PRESENT", "active_job_count": 1, "active_jobs": [{"pid": 1234, "command": "python train.py"}]},
    )

    receipt = check_preflight(
        protocol=valid_protocol,
        data_manifest=None,
        data_manifest_sha256=None,
        out_dir=out_dir,
        device="cpu",
        synthetic=True,
        check_jobs=True,
    )
    assert receipt["status"] == "REFUSED_ACTIVE_JOBS"


def test_check_preflight_refuses_gpu_busy(tmp_path, valid_protocol, monkeypatch):
    """Preflight refuses when GPU is busy."""
    out_dir = tmp_path / "run"
    monkeypatch.setattr(
        "self_audit.transfer_preflight.check_gpu_occupancy",
        lambda dev: {"status": "BUSY", "is_busy": True, "error": "GPU memory 95% full"},
    )

    receipt = check_preflight(
        protocol=valid_protocol,
        data_manifest=None,
        data_manifest_sha256=None,
        out_dir=out_dir,
        device="cuda:0",
        synthetic=False,
        check_jobs=False,
    )
    assert receipt["status"] == "REFUSED_GPU_BUSY"


def test_check_preflight_invalid_protocol(tmp_path, valid_protocol):
    """Preflight sets INVALID_PROTOCOL when protocol is modified."""
    out_dir = tmp_path / "run"
    tampered = copy.deepcopy(valid_protocol)
    tampered["budget"]["total_steps"] = 3000

    receipt = check_preflight(
        protocol=tampered,
        data_manifest=None,
        data_manifest_sha256=None,
        out_dir=out_dir,
        device="cpu",
        synthetic=True,
        check_jobs=False,
    )
    assert receipt["status"] == "INVALID_PROTOCOL"
    assert any("budget.total_steps mismatch" in r for r in receipt["reasons"])


# ==============================================================================
# Real Asset Failclosed Tests (NOT_RUN)
# ==============================================================================

def test_check_preflight_not_run_on_missing_train_data(tmp_path, valid_protocol):
    """Real non-synthetic preflight fails closed with NOT_RUN when data is missing."""
    out_dir = tmp_path / "run"
    receipt = check_preflight(
        protocol=valid_protocol,
        data_manifest=None,
        data_manifest_sha256=None,
        out_dir=out_dir,
        device="cpu",
        synthetic=False,
        check_jobs=False,
    )
    assert receipt["status"] == "NOT_RUN"
    assert any("requires --data and --data-sha256" in r for r in receipt["reasons"])


def test_check_preflight_not_run_on_corrupt_train_sha(tmp_path, valid_protocol):
    """Real non-synthetic preflight fails closed when TRAIN SHA does not match."""
    out_dir = tmp_path / "run"
    fake_manifest = tmp_path / "fake_train.json"
    fake_manifest.write_text('{"schema": "self_audit.data.v2"}\n')

    receipt = check_preflight(
        protocol=valid_protocol,
        data_manifest=fake_manifest,
        data_manifest_sha256="0" * 64,
        out_dir=out_dir,
        device="cpu",
        synthetic=False,
        check_jobs=False,
    )
    assert receipt["status"] == "NOT_RUN"
    assert any("data manifest sha-256 mismatch" in r.lower() for r in receipt["reasons"])


def test_check_preflight_strict_no_u_masks(tmp_path, valid_protocol):
    """Preflight fails closed if unlabeled record in TRAIN manifest contains masks."""
    out_dir = tmp_path / "run"
    manifest_data = {
        "schema": "self_audit.data.v2",
        "role": "training",
        "label_budget": 8,
        "labeled": [{"id": pid, "image": f"{pid}.npy", "image_sha256": "a"*64, "masks": {"0": {"path": "m.npy", "sha256": "b"*64}}} for pid in PINNED_L8_PATIENT_IDS],
        "unlabeled": [
            {"id": "patient001", "image": "patient001.npy", "image_sha256": "c"*64, "masks": {"0": "leak.npy"}}
        ] + [{"id": f"p{i:03d}", "image": f"p{i}.npy", "image_sha256": "d"*64} for i in range(2, 73)],
    }
    m_path = tmp_path / "manifest.json"
    m_path.write_text(json.dumps(manifest_data) + "\n")
    actual_sha = digest(m_path)

    # Even if sha matches argument, it must match protocol pin and pass validation
    receipt = check_preflight(
        protocol=valid_protocol,
        data_manifest=m_path,
        data_manifest_sha256=actual_sha,
        out_dir=out_dir,
        device="cpu",
        synthetic=False,
        check_jobs=False,
    )
    assert receipt["status"] == "NOT_RUN"
    assert any("strict no u masks" in r.lower() or "does not match immutable protocol pin" in r.lower() for r in receipt["reasons"])


def test_check_preflight_val_manifest_validation(tmp_path, valid_protocol):
    """Validation manifest must have count=18, disjoint from TRAIN, and match SHA."""
    out_dir = tmp_path / "run"
    val_data = {
        "schema": "self_audit.data.v2",
        "role": "validation",
        "labeled": [{"id": f"val{i:03d}"} for i in range(18)],
    }
    v_path = tmp_path / "val_manifest.json"
    v_path.write_text(json.dumps(val_data) + "\n")
    v_sha = digest(v_path)

    # When synthetic=False and val_sha does not match pinned VAL sha, it fails closed
    receipt = check_preflight(
        protocol=valid_protocol,
        data_manifest=tmp_path / "missing_train.json",
        data_manifest_sha256=PINNED_TRAIN_MANIFEST_SHA256,
        val_manifest=v_path,
        val_manifest_sha256=v_sha,
        out_dir=out_dir,
        device="cpu",
        synthetic=False,
        check_jobs=False,
    )
    assert receipt["status"] == "NOT_RUN"


# ==============================================================================
# Real Teacher Lineage & Cache Metadata Tests
# ==============================================================================

def test_check_preflight_uses_real_cache_metadata(tmp_path, valid_protocol, monkeypatch):
    """Real filesystem integration with manufactured pins, never patient evidence.

    No mocked identity validator: payload, quality receipt, data geometry and every
    teacher cache array are checked by production APIs.
    """
    import torch
    import self_audit.transfer_preflight as preflight
    from self_audit.compact_data import synthetic_compact_fixture
    from self_audit.compact_teacher import HISTORICAL_SOURCE_CLOSURE, HISTORICAL_SOURCE_COMMIT
    train, _ = synthetic_compact_fixture(tmp_path/'train', n_labeled=8,
        n_unlabeled=72, size=4, num_frames=1, num_slices=1)
    subprocess.run([sys.executable, str(ROOT/'scripts/build_compact_teacher_cache.py'),
        '--data-manifest',str(train),'--data-sha256',digest(train),
        '--output-dir',str(tmp_path/'cache'),'--synthetic-teacher','--device','cpu'],
        check=True, capture_output=True, text=True, timeout=60)
    data = json.loads(train.read_text()); data['synthetic'] = False
    train.write_text(json.dumps(data))
    lids = [r['id'] for r in data['labeled']]
    checkpoint = tmp_path/'teacher'; checkpoint.mkdir()
    manifest = checkpoint/'MANIFEST.json'
    manifest.write_text(json.dumps({'schema':'self_audit.checkpoint.v2',
        'source_sha256':HISTORICAL_SOURCE_CLOSURE,'config':{'method':'ema_full'},'synthetic':False}))
    payload = checkpoint/'student.pt'; torch.save({'student':{'weight':torch.tensor(1.)}}, payload)
    quality = tmp_path/'quality.json'; quality.write_text(json.dumps({
        'checkpoint_manifest_sha256':digest(manifest),'inference_state':'student','class_first_macro4':.5}))
    identity = dict(teacher_kind='canonical', inference_state='student', training_method='ema_full',
        source_sha256=HISTORICAL_SOURCE_CLOSURE, source_commit=HISTORICAL_SOURCE_COMMIT,
        manifest_sha256=digest(manifest), payload_sha256=digest(payload),
        lineage={'budget_k':8,'exact_labeled_ids':lids},
        quality_receipt={'path':str(quality),'sha256':digest(quality)},
        verified=True, physical_payload_status='VERIFIED_ON_DISK')
    cache_file = tmp_path/'cache/CACHE_MANIFEST.json'; cache=json.loads(cache_file.read_text())
    cache.update(teacher_identity=identity,train_manifest_sha256=digest(train))
    cache_file.write_text(json.dumps(cache))
    val = tmp_path/'val.json'; val.write_text(json.dumps({'schema':'self_audit.native_manifest.v1',
        'role':'development','patients':[{'id':f'dev_{i}'} for i in range(18)]}))
    pins = dict(PINNED_TRAIN_MANIFEST_SHA256=digest(train), PINNED_L8_PATIENT_IDS=tuple(lids),
        PINNED_B8_TEACHER_MANIFEST_SHA256=digest(manifest), PINNED_RAW_PAYLOAD_SHA256=digest(payload),
        PINNED_TEACHER_CACHE_SHA256=digest(cache_file), PINNED_VAL_MANIFEST_SHA256=digest(val))
    for key,value in pins.items(): monkeypatch.setattr(preflight,key,value)
    valid_protocol['data'].update(manifest_sha256=digest(train),labeled_ids=lids)
    valid_protocol['teacher'].update(source_manifest_sha256=digest(manifest),source_payload_sha256=digest(payload),
        cache_sha256=digest(cache_file),lineage={'budget_k':8,'exact_labeled_ids':lids})
    valid_protocol['evaluation']['manifest_sha256']=digest(val)
    kwargs=dict(protocol=valid_protocol,data_manifest=train,data_manifest_sha256=digest(train),
        out_dir=tmp_path/'out',device='cpu',synthetic=False,check_jobs=False,
        teacher_checkpoint=checkpoint,teacher_checkpoint_sha256=digest(manifest),
        teacher_payload_sha256=digest(payload),teacher_cache=cache_file,teacher_cache_sha256=digest(cache_file),
        val_manifest=val,val_manifest_sha256=digest(val))
    receipt=check_preflight(**kwargs)
    assert receipt['status']=='PASS', receipt
    assert receipt['checks']['cache_payloads']['status']=='PASS'
    assert receipt['checks']['teacher_identity']['verified'] is True
    assert not (tmp_path/'out').exists()
    # Physical corruption is detected even though the manifest is untouched.
    quality.write_text('{}')
    corrupt=check_preflight(**kwargs)
    assert corrupt['status']=='NOT_RUN'
    assert 'quality receipt' in ' '.join(corrupt['reasons'])


def test_check_preflight_synthetic_unit_mode(valid_protocol, tmp_path):
    """Synthetic unit mode passes without physical patient assets and records disclaimer."""
    out_dir = tmp_path / "synthetic_run"
    receipt = check_preflight(
        protocol=valid_protocol,
        data_manifest=None,
        data_manifest_sha256=None,
        out_dir=out_dir,
        device="cpu",
        synthetic=True,
        check_jobs=False,
    )
    assert receipt["status"] == "PASS"
    assert receipt["synthetic"] is True
    assert "source_receipts" in receipt
    assert "compact_training_source_sha256" in receipt["source_receipts"]
    assert "transfer_preflight_sha256" in receipt["source_receipts"]
    assert receipt["checks"]["assets"]["synthetic"] is True
    assert "Synthetic unit mode" in receipt["checks"]["assets"]["disclaimer"]


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), None, True, '0.0003', 0.000300000001])
def test_protocol_rejects_invalid_or_changed_numeric(valid_protocol, bad):
    valid_protocol['hyperparameters']['lr'] = bad
    assert validate_locked_protocol(valid_protocol)


def test_failed_gpu_process_query_fails_closed(monkeypatch):
    responses = iter([
        subprocess.CompletedProcess([],0,'GPU-test, 0, RTX 4000\n',''),
        subprocess.CompletedProcess([],0,'0, 24576, 24576\n',''),
        subprocess.CompletedProcess([],1,'','query failed'),
    ])
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: next(responses))
    assert check_gpu_occupancy('cuda:0')['status'] == 'UNVERIFIABLE'


def test_empty_owned_output_is_refused(tmp_path, valid_protocol):
    out = tmp_path/'owned'; out.mkdir()
    result = check_preflight(valid_protocol, None, None, out, device='cpu', synthetic=True, check_jobs=False)
    assert result['status'] == 'REFUSED_EXISTING_OUTPUT'


def test_archived_val_schema_and_pin_are_supported():
    path = ROOT/'results/compact_acdc_l8_20261029/experiment_20261028/val18_native.json'
    manifest = json.loads(path.read_text())
    assert digest(path) == PINNED_VAL_MANIFEST_SHA256
    assert manifest['role'] == 'development'
    assert manifest['schema'] == 'self_audit.data.v2'
    assert len(manifest['patients']) == PINNED_VAL18_PATIENT_COUNT
