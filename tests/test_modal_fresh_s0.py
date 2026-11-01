import json
from pathlib import Path

import pytest
import torch

from self_audit.asset_staging import sha256
from self_audit.modal_data_prep import (
    BoundedRclone, HISTORICAL_TRAIN_SHA, L8, download_plan, locked_split, manifest_comparison,
)
from self_audit.modal_s0 import atomic_save, budget_plan, checkpoint_io, s0_config

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "splits/acdc_reviewed_locked_train80_val18.json"
HISTORY = ROOT / "configs/fresh_s0/acdc_l8_historical_train.json"


def test_locked_split_and_recovered_historical_manifest():
    split = locked_split(SPLIT)
    assert len(split["train"]) == 80 and len(split["val"]) == 18
    assert sha256(HISTORY) == HISTORICAL_TRAIN_SHA
    manifest = json.loads(HISTORY.read_text())
    assert sorted(r["id"] for r in manifest["labeled"]) == L8
    assert manifest_comparison(HISTORY)["exact_bytes"]


def inventory_fixture():
    rows = []
    for i in range(1, 151):
        pid = f"patient{i:03d}"
        for name in [f"{pid}_4d.nii.gz", "Info.cfg"] + [
            f"{pid}_frame{f:02d}{suffix}.nii.gz" for f in (1, 12) for suffix in ("", "_gt")
        ]:
            rows.append({"Path": f"{pid}/{name}", "Size": 1024, "IsDir": False})
    return rows


def test_allowlist_contains_no_u_masks_or_test_quarantine():
    split = locked_split(SPLIT)
    plan = download_plan(inventory_fixture(), split)
    assert len(plan) == 202
    refs = {r["path"].split("/")[0] for r in plan if r["kind"] == "reference"}
    assert refs == set(L8 + split["val"])
    cine = {r["path"].split("/")[0] for r in plan if r["kind"] == "cine"}
    assert cine == set(split["train"] + split["val"])
    assert "patient055" not in cine and "patient076" not in cine


def test_allowlist_requires_complete_reference_and_rejects_duplicate_cine():
    split = locked_split(SPLIT)
    rows = inventory_fixture()
    with pytest.raises(AssertionError):
        download_plan([r for r in rows if r["Path"] != "patient007/patient007_frame01.nii.gz"], split)
    rows.append({"Path": "patient007/patient007_4d.nii", "Size": 1})
    with pytest.raises(AssertionError):
        download_plan(rows, split)


def test_path_only_manifest_witness_is_not_payload_bypass(tmp_path):
    obj = json.loads(HISTORY.read_text())
    for r in obj["labeled"] + obj["unlabeled"]:
        r["image"] = "/different/root/" + r["image"]
    path = tmp_path / "train.json"
    path.write_text(json.dumps(obj, sort_keys=True))
    comp = manifest_comparison(path)
    assert not comp["exact_bytes"] and comp["path_only_witness"]["payload_hashes_unchanged"]
    obj["unlabeled"][0]["image_sha256"] = "f" * 64
    path.write_text(json.dumps(obj))
    comp = manifest_comparison(path)
    assert comp["path_only_witness"] is None and comp["historical_semantic_differences"] == ["unlabeled"]


@pytest.mark.parametrize("credit,spent", [(30, 0), (10, 2), (4, 0), (30, 9)])
def test_budget_respects_credit_and_total_cap(credit, spent):
    p = budget_plan(credit, spent)
    assert p["max_container_cost_usd"] + p["reserve_usd"] <= min(credit, 15 - spent)
    assert 0 < p["training_seconds"] < p["function_timeout_seconds"] <= 7200
    assert p["gpu"] == "A100-40GB" and p["max_containers"] == 1 and p["retries"] == 0


@pytest.mark.parametrize("credit,spent", [(0, 0), (2, 0), (30, 14), (float("nan"), 0), (-1, 0)])
def test_invalid_or_insufficient_credit_blocks(credit, spent):
    with pytest.raises(ValueError):
        budget_plan(credit, spent)


def test_s0_config_has_no_teacher_and_fixed_supervised_recipe():
    c = s0_config(6600)
    assert c["teacher"] == {} and c["method"] == "supervised"
    assert (c["steps"], c["crop_size"], c["labeled_batch_size"], c["seed"]) == (10000, 160, 8, 42)
    assert c["model"]["widths"] == [32, 64, 128, 256]
    assert c["save_every"] == 1000 and c["equal_wall_time"] == {"enabled": False}


def test_atomic_save_publishes_only_after_hash_check(tmp_path):
    out = tmp_path / "step_001000"
    events = []
    def save(*args, **kwargs):
        target = Path(args[4])
        target.mkdir()
        (target / "weights.pt").write_bytes(b"weights-real-payload")
        (target / "MANIFEST.json").write_text(json.dumps({"payload_sha256": sha256(target / "weights.pt")}))
        assert not out.exists()
        return sha256(target / "MANIFEST.json")
    def commit():
        assert out.is_dir() and (out / "weights.pt").is_file()
        events.append("commit")
    atomic_save(save, commit, None, None, {}, None, out, step=1000)
    assert events == ["commit"]
    with pytest.raises(FileExistsError):
        atomic_save(save, commit, None, None, {}, None, out, step=1000)


def test_partial_save_is_not_published_or_committed(tmp_path):
    out = tmp_path / "final"
    def fail(*args, **kwargs):
        Path(args[4]).mkdir()
        (Path(args[4]) / "weights.pt").write_bytes(b"incomplete")
        raise OSError("interrupted")
    with pytest.raises(OSError):
        atomic_save(fail, lambda: pytest.fail("must not commit partial"), None, None, {}, None, out, step=2)
    assert not out.exists() and (tmp_path / ".final.partial/weights.pt").exists()


def test_rclone_retry_is_bounded_and_sanitized(monkeypatch):
    calls = []
    def fail(self, *args):
        calls.append(args)
        raise RuntimeError("RCLONE_FAILED category=DRIVE_RATE_LIMIT; raw stderr withheld")
    monkeypatch.setattr("self_audit.asset_staging.Rclone.run", fail)
    monkeypatch.setattr("self_audit.modal_data_prep.time.sleep", lambda x: None)
    with pytest.raises(RuntimeError, match="DRIVE_RATE_LIMIT"):
        BoundedRclone("/tmp/private.conf", "Remote").run("lsjson", "Remote:path")
    assert len(calls) == 3 and "--tpslimit" in calls[0]


def test_modal_gpu_has_no_drive_secret_and_required_resource_limits():
    import ast
    tree = ast.parse((ROOT / "scripts/modal_fresh_s0.py").read_text())
    funcs = {x.name: x for x in tree.body if isinstance(x, ast.FunctionDef)}
    decorated = {name: {k.arg: k.value for k in f.decorator_list[0].keywords}
                 for name, f in funcs.items() if f.decorator_list and isinstance(f.decorator_list[0], ast.Call)}
    assert "secrets" in decorated["prepare_data"]
    for name in ("verify_data", "smoke_s0", "train_s0"):
        assert "secrets" not in decorated[name]
    assert ast.literal_eval(decorated["train_s0"]["gpu"]) == "A100-40GB"
    assert ast.literal_eval(decorated["train_s0"]["max_containers"]) == 1
    assert ast.literal_eval(decorated["train_s0"]["retries"]) == 0


def test_wide_supervised_atomic_smoke_export_and_resume(tmp_path):
    from self_audit.compact_data import synthetic_compact_fixture
    from self_audit.compact_training import train_compact, _state_sha256
    from self_audit.compact_artifacts import export_compact_student, load_compact_export, load_compact_checkpoint
    torch.set_num_threads(2)
    manifest, digest = synthetic_compact_fixture(tmp_path / "data", n_labeled=2, n_unlabeled=2, size=32)
    c = s0_config(6600)
    # Full wide model/crop/batch; only a bounded two-update manufactured execution.
    commits = []
    with checkpoint_io(lambda: commits.append(1)):
        r = train_compact(c, manifest, digest, tmp_path / "smoke", synthetic=True,
                          device="cpu", stop_after_step=2, allow_full_run=True)
    assert len(commits) == 1 and r["steps"] == 2
    assert r["optimizer_calls"] == {"adamw": 2, "sgd": 0}
    for field in ("initial_gradient_norms", "initial_weight_changes"):
        assert all(r["warm_start_summary"][field][k] > 0 for k in ("encoder", "decoder", "head"))
    export_sha = export_compact_student(tmp_path / "smoke/final", r["final_manifest_sha256"], tmp_path / "export")
    reloaded, _ = load_compact_export(tmp_path / "export", export_sha)
    trained, _ = load_compact_checkpoint(tmp_path / "smoke/final", r["final_manifest_sha256"])
    assert _state_sha256(trained) == _state_sha256(reloaded)
    x = torch.linspace(0, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    trained.eval(); reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(trained(x), reloaded(x), rtol=0, atol=0)
    with checkpoint_io(lambda: commits.append(1)):
        resumed = train_compact(c, manifest, digest, tmp_path / "resumed", synthetic=True,
            device="cpu", stop_after_step=3, allow_full_run=True,
            resume=tmp_path / "smoke/final", resume_sha256=r["final_manifest_sha256"])
    assert resumed["steps"] == 3 and resumed["resumed_step"] == 2
    assert resumed["initialization"]["initial_state_sha256"] == r["initialization"]["initial_state_sha256"]
    with checkpoint_io(lambda: None):
        uninterrupted = train_compact(c, manifest, digest, tmp_path / "uninterrupted", synthetic=True,
                                      device="cpu", stop_after_step=3, allow_full_run=True)
    resumed_model, _ = load_compact_checkpoint(tmp_path / "resumed/final", resumed["final_manifest_sha256"])
    uninterrupted_model, _ = load_compact_checkpoint(tmp_path / "uninterrupted/final", uninterrupted["final_manifest_sha256"])
    assert _state_sha256(resumed_model) == _state_sha256(uninterrupted_model)
