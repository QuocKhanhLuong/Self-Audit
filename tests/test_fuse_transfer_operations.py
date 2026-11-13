from __future__ import annotations

import importlib.util
import json
import tempfile
import hashlib
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location("test_" + name[:-3], ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def b_config():
    return {"schema": "self_audit.candidate_fuse.v3", "experiment": {"track": "B", "source_teacher": {
                "path": "source.pt", "sha256": "b"*64, "metadata_sha256": "c"*64,
                "identity": {"config_sha256": "d"*64, "train_manifest_sha256": "e"*64,
                             "val_manifest_sha256": "f"*64}}},
            "data": {"test_manifest": "test.json", "test_manifest_sha256": "a" * 64}}


def pin_source(config, root):
    source = config["experiment"]["source_teacher"]
    path = root / "source.pt"
    path.write_bytes(b"selected source")
    source["path"] = str(path)
    source["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = path.with_name(path.name + ".meta.json")
    metadata.write_text(json.dumps({"status": "SELECTED", "identity": source["identity"]}))
    source["metadata_sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    return config


def test_export_bundle_supports_four_modes_only_when_requested():
    campaign = load_script("fuse_campaign.py")
    import hashlib
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); exports = root / "exports"; exports.mkdir()
        for mode in ("teacher_zero_shot", "teacher", "student", "fusion"):
            artifact = exports / f"{mode}.pt"; artifact.write_bytes(mode.encode())
            artifact.with_suffix(".pt.manifest.json").write_text(json.dumps({
                "schema": "self_audit.candidate_fuse.export.v1", "artifact": artifact.name,
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "artifact_bytes": artifact.stat().st_size,
                "mode": "teacher" if mode == "teacher_zero_shot" else mode}))
        assert set(campaign.export_bundle(root, track_b=True)) == {"teacher_zero_shot", "teacher", "student", "fusion"}
        with __import__("pytest").raises(ValueError):
            campaign.export_bundle(root, track_b=False)


def test_b_campaign_pins_source_t_and_adapted_teacher_for_j():
    campaign = load_script("fuse_campaign.py")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); config = root / "b.json"; config.write_text(json.dumps(b_config()))
        plan = campaign.commands(config, root / "run", device="cpu")
        assert "--fuse-teacher-artifact" in plan["T"]
        assert "source.pt" in plan["T"]
        assert str(root / "run" / "teacher.pt") in plan["J"]
        assert "--fuse-max-updates" not in plan["T"]


def test_b_test_call_is_one_four_mode_evaluation_transaction():
    campaign = load_script("fuse_campaign.py")
    import hashlib
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); exports = root / "exports"; exports.mkdir()
        for mode in ("teacher_zero_shot", "teacher", "student", "fusion"):
            artifact = exports / f"{mode}.pt"; artifact.write_bytes(mode.encode())
            artifact.with_suffix(".pt.manifest.json").write_text(json.dumps({
                "schema": "self_audit.candidate_fuse.export.v1", "artifact": artifact.name,
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "artifact_bytes": artifact.stat().st_size,
                "mode": "teacher" if mode == "teacher_zero_shot" else mode}))
        config = root / "b.json"; config.write_text(json.dumps(b_config()))
        result = {"status": "COMMITTED", "role": "test"}
        with mock.patch("self_audit.candidate_fuse.evaluation.evaluate_exports", return_value=result) as evaluator:
            assert campaign.evaluate_test_once(config, root, device="cpu") == result
        assert len(evaluator.call_args.args[0]) == 4
        evaluator.assert_called_once()


def test_predecessor_gate_requires_completed_committed_test():
    queue = load_script("fuse_transfer_queue.py")
    with tempfile.TemporaryDirectory() as td:
        transaction = Path(td) / "TEST_TRANSACTION.json"
        transaction.write_text(json.dumps({"status": "COMMITTED", "role": "test"}))
        state = {"status": "COMPLETED", "completed": queue.REQUIRED_A_STAGES,
                 "config_sha256": "a" * 64, "test_ledger": {"status": "COMMITTED", "completed_utc": "now",
                     "transaction_path": str(transaction)},
                 "export_bundle": {"teacher": {"sha256": "b" * 64}}}
        path = Path(td) / "RUN_STATE.json"; path.write_text(json.dumps(state))
        assert queue.verify_predecessor(path)["status"] == "COMPLETED"
        state["test_ledger"]["status"] = "FAILED_LOCKED"; path.write_text(json.dumps(state))
        with __import__("pytest").raises(ValueError, match="TEST ledger"):
            queue.verify_predecessor(path)


def test_queue_defaults_to_plan_only_and_preserves_separate_run_identity():
    queue = load_script("fuse_transfer_queue.py")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); config = root / "b.json"; config.write_text(json.dumps(b_config()))
        transaction = root / "A" / "TEST_TRANSACTION.json"
        transaction.parent.mkdir()
        transaction.write_text(json.dumps({"status": "COMMITTED", "role": "test"}))
        state = {"status": "COMPLETED", "completed": queue.REQUIRED_A_STAGES,
                 "config_sha256": "a" * 64, "test_ledger": {"status": "COMMITTED", "completed_utc": "now",
                     "transaction_path": str(transaction)},
                 "artifacts": {"teacher": "b" * 64}}
        pred = root / "A" / "RUN_STATE.json"; pred.write_text(json.dumps(state))
        plan = queue.run_queue(config, root / "B", pred, execute=False, device="cpu")
        assert plan["status"] == "PLAN_ONLY"
        assert plan["automatic_retries"] == 0
        assert not (root / "B").exists()


def test_queue_disk_headroom_failure_is_explicit_and_no_subprocess_runs():
    queue = load_script("fuse_transfer_queue.py")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); config = root / "b.json"
        cfg = pin_source(b_config(), root)
        config.write_text(json.dumps(cfg))
        transaction = root / "A" / "TEST_TRANSACTION.json"
        transaction.parent.mkdir()
        transaction.write_text(json.dumps({"status": "COMMITTED", "role": "test"}))
        state = {"status": "COMPLETED", "completed": queue.REQUIRED_A_STAGES,
                 "config_sha256": "a" * 64, "test_ledger": {"status": "COMMITTED", "completed_utc": "now",
                     "transaction_path": str(transaction)},
                 "artifacts": {"teacher": "b" * 64}}
        pred = root / "A" / "RUN_STATE.json"; pred.write_text(json.dumps(state))
        with mock.patch.object(queue, "real_data_preflight", return_value={"ok": True}), \
             mock.patch.object(queue, "disk_headroom", return_value={"free_bytes": 100, "required_bytes": 8*1024**3}), \
             mock.patch.object(queue.subprocess, "run", side_effect=AssertionError("must not run")):
            result = queue.run_queue(config, root / "B", pred, execute=True, device="cpu",
                                     idle_probe=lambda: (True, ""), wait_for_predecessor=False)
        assert result["status"] == "BLOCKED"
        assert "8 GiB" in result["blocker"]


def test_failed_predecessor_is_blocked_without_waiting_or_retry():
    queue = load_script("fuse_transfer_queue.py")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); config = root / "b.json"
        cfg = b_config()
        config.write_text(json.dumps(cfg))
        pred = root / "A" / "RUN_STATE.json"; pred.parent.mkdir()
        pred.write_text(json.dumps({"status": "FAILED", "completed": [],
                                    "test_ledger": {"status": "NOT_RUN"}}))
        result = queue.run_queue(config, root / "B", pred, execute=True, device="cpu",
                                 idle_probe=lambda: (_ for _ in ()).throw(AssertionError("GPU must not be queried")))
        assert result["status"] == "BLOCKED"
        assert "no automatic retry" in result["blocker"]


def test_plan_requires_complete_source_teacher_pin():
    queue = load_script("fuse_transfer_queue.py")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); config = root / "b.json"
        cfg = b_config(); cfg["experiment"]["source_teacher"]["path"] = None
        config.write_text(json.dumps(cfg))
        predecessor = root / "A" / "RUN_STATE.json"
        predecessor.parent.mkdir()
        predecessor.write_text("{}")
        with __import__("pytest").raises(ValueError, match="fully pinned"):
            queue.run_queue(config, root / "B", predecessor, execute=False)


def test_unexpected_queue_launch_failure_is_persisted_as_terminal():
    queue = load_script("fuse_transfer_queue.py")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); config = root / "b.json"
        config.write_text(json.dumps(pin_source(b_config(), root)))
        transaction = root / "A" / "TEST_TRANSACTION.json"
        transaction.parent.mkdir()
        transaction.write_text(json.dumps({"status": "COMMITTED", "role": "test"}))
        predecessor = root / "A" / "RUN_STATE.json"
        predecessor.write_text(json.dumps({
            "status": "COMPLETED", "completed": queue.REQUIRED_A_STAGES,
            "config_sha256": "a" * 64,
            "test_ledger": {"status": "COMMITTED", "completed_utc": "now",
                            "transaction_path": str(transaction)},
            "artifacts": {"teacher": "b" * 64},
        }))
        with mock.patch.object(queue, "real_data_preflight", return_value={"ok": True}), \
             mock.patch.object(queue, "disk_headroom", return_value={
                "free_bytes": 9 * 1024**3, "required_bytes": 8 * 1024**3}), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=OSError("launch failed")):
            result = queue.run_queue(config, root / "B", predecessor, execute=True,
                                     device="cpu", idle_probe=lambda: (True, ""),
                                     wait_for_predecessor=False)
        assert result["status"] == "FAILED"
        assert result["active"] is None
        assert result["error"]["stage"] == "PROFILE_TJ"
        saved = json.loads((root / "B" / "TRANSFER_QUEUE_STATE.json").read_text())
        assert saved["status"] == "FAILED"


def test_real_preflight_validates_config_and_all_roles_without_decoding():
    queue = load_script("fuse_transfer_queue.py")
    with mock.patch("self_audit.candidate_fuse.config.load_config", return_value={
            "runtime": {"synthetic": False}, "data": {
                "train_manifest": "train.json", "train_manifest_sha256": "a"*64,
                "val_manifest": "val.json", "val_manifest_sha256": "b"*64,
                "test_manifest": "test.json", "test_manifest_sha256": "c"*64,
            }}) as validate, \
         mock.patch("self_audit.candidate_fuse.data.open_fuse_data") as open_reader:
        readers = {role: mock.Mock(verify_sources=mock.Mock(return_value={"containers": 2}))
                   for role in ("train", "development", "test")}
        open_reader.side_effect = lambda path, sha, *, role, synthetic: readers[role]
        result = queue.real_data_preflight(Path("b.json"))
    validate.assert_called_once_with(Path("b.json"), real_run=True)
    assert [call.kwargs["role"] for call in open_reader.call_args_list] == [
        "train", "development", "test"]
    assert all(call.kwargs["synthetic"] is False for call in open_reader.call_args_list)
    for reader in readers.values():
        reader.verify_sources.assert_called_once_with()
    assert set(result) == {"train", "development", "test"}


def test_real_preflight_rejects_synthetic_and_unverifiable_reader():
    queue = load_script("fuse_transfer_queue.py")
    with mock.patch("self_audit.candidate_fuse.config.load_config", return_value={
            "runtime": {"synthetic": True}, "data": {}}):
        with __import__("pytest").raises(ValueError, match="non-synthetic"):
            queue.real_data_preflight(Path("b.json"))
    with mock.patch("self_audit.candidate_fuse.config.load_config", return_value={
            "runtime": {"synthetic": False}, "data": {
                "train_manifest": "train.json", "train_manifest_sha256": "a"*64,
                "val_manifest": "val.json", "val_manifest_sha256": "b"*64,
                "test_manifest": "test.json", "test_manifest_sha256": "c"*64,
            }}), \
         mock.patch("self_audit.candidate_fuse.data.open_fuse_data", return_value=object()):
        with __import__("pytest").raises(RuntimeError, match="metadata-only"):
            queue.real_data_preflight(Path("b.json"))


def test_queue_rechecks_gpu_after_profile_before_campaign():
    queue = load_script("fuse_transfer_queue.py")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); config = root / "b.json"
        config.write_text(json.dumps(pin_source(b_config(), root)))
        transaction = root / "A" / "TEST_TRANSACTION.json"
        transaction.parent.mkdir()
        transaction.write_text(json.dumps({"status": "COMMITTED", "role": "test"}))
        predecessor = root / "A" / "RUN_STATE.json"
        predecessor.write_text(json.dumps({
            "status": "COMPLETED", "completed": queue.REQUIRED_A_STAGES,
            "config_sha256": "a" * 64,
            "test_ledger": {"status": "COMMITTED", "completed_utc": "now",
                            "transaction_path": str(transaction)},
            "artifacts": {"teacher": "b" * 64},
        }))

        class ProfileProc:
            stdout = iter(())
            def __init__(self, cmd, **kwargs):
                assert "fuse_profile.py" in cmd[1]
                profile_dir = Path(cmd[cmd.index("--out") + 1])
                profile_dir.mkdir(parents=True)
                (profile_dir / "PROFILE.json").write_text(json.dumps({
                    "status": "PROFILE_ONLY", "track": "B"}))
            def wait(self):
                return 0

        popen = mock.Mock(side_effect=ProfileProc)
        with mock.patch.object(queue, "real_data_preflight", return_value={"ok": True}), \
             mock.patch.object(queue, "disk_headroom", return_value={
                 "free_bytes": 9 * 1024**3, "required_bytes": 8 * 1024**3}), \
             mock.patch.object(queue.subprocess, "Popen", popen):
            result = queue.run_queue(config, root / "B", predecessor, execute=True,
                                     device="cpu", idle_probe=mock.Mock(side_effect=[
                                         (True, ""), (False, "1234")]),
                                     wait_for_predecessor=False)
        assert result["status"] == "BLOCKED"
        assert "after profile" in result["blocker"]
        assert result["campaign"] == "NOT_RUN"
        popen.assert_called_once()
