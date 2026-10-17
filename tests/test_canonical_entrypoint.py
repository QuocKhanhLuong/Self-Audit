"""Routing tests use lightweight fixtures; they do not claim patient training."""
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train
from scripts.check_repo import check_layout, REQUIRED, REMOVED, LEGACY_STUBS


@pytest.fixture
def clean(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for name in REQUIRED:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("" if file.suffix != ".json" else "{}")
    (root / "ACTIVE_PIPELINE.json").write_text((ROOT / "ACTIVE_PIPELINE.json").read_text())
    for name in ("pilot", "full50"):
        (root / "configs" / (name + ".json")).write_text((ROOT / "configs" / (name + ".json")).read_text())
    return root


def test_describe_is_lightweight():
    proc = subprocess.run([sys.executable, "-c", "import train,sys; assert train.main(['--describe'])==0; assert 'torch' not in sys.modules"],
                          cwd=ROOT, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["backbone"] == "dinov3_convnext_tiny"


def test_alias_describes_same_model(tmp_path):
    outputs = []
    for script in (ROOT / "train.py", ROOT / "scripts/train_astra_student.py"):
        proc = subprocess.run([sys.executable, str(script), "--describe"], cwd=tmp_path, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        outputs.append(json.loads(proc.stdout))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("name", LEGACY_STUBS)
def test_legacy_commands_never_train(name, tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / name), "--help"], cwd=tmp_path, text=True, capture_output=True)
    assert proc.returncode == 2
    assert "LEGACY_ENTRYPOINT_DISABLED" in proc.stderr


@pytest.mark.parametrize("args", [[], ["--phase", "native"], ["--model", "GuidanceStudent"], ["--des"]])
def test_invalid_arguments_stop(args):
    with pytest.raises(SystemExit) as exc:
        train.main(args)
    assert exc.value.code == 2


def test_layout_accepts_canonical_fixture(clean):
    assert check_layout(clean)["status"] == "PASS_CANONICAL_LAYOUT"


@pytest.mark.parametrize("name", ["baseline", "src/self_audit_guidance/student.py", "experiments/guidance_signal_20261014"])
def test_stale_code_is_rejected(clean, name):
    path = clean / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("old") if path.suffix else path.mkdir()
    with pytest.raises(ValueError, match="retired"):
        check_layout(clean)


def test_hidden_legacy_class_is_rejected(clean):
    path = clean / train.IMPLEMENTATION / "layer_upgrade/model.py"
    path.write_text("class GuidanceStudent: pass\n")
    with pytest.raises(ValueError, match="Retired CNN"):
        check_layout(clean)


def test_removed_import_is_rejected(clean):
    path = clean / train.IMPLEMENTATION / "training_v1/runner.py"
    path.write_text("from self_audit_guidance.student import train_student\n")
    with pytest.raises(ValueError, match="legacy dependency"):
        check_layout(clean)


def test_pilot_is_bounded():
    assert train.read_recipe(ROOT / "configs/pilot.json")["max_steps"] == 32


def test_full_budget_is_explicit():
    with pytest.raises(ValueError, match="allow-full-run"):
        train.read_recipe(ROOT / "configs/full50.json")
    assert train.read_recipe(ROOT / "configs/full50.json", True)["max_steps"] is None


@pytest.mark.parametrize("key,value", [("architecture", "GuidanceStudent"), ("semantic_policy", "Z"), ("refiner", "on"), ("max_steps", True)])
def test_bad_recipe_is_rejected(clean, key, value):
    path = clean / "configs/pilot.json"
    cfg = json.loads(path.read_text()); cfg[key] = value
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        train.read_recipe(path)


def test_wrong_identity_rejected(clean):
    path = clean / "ACTIVE_PIPELINE.json"
    cfg = json.loads(path.read_text()); cfg["backbone"] = "random"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="backbone"):
        train.identity(clean)


def test_foreign_import_refused(monkeypatch, clean):
    foreign = types.ModuleType("training_v1"); foreign.__file__ = "/other/checkout/training_v1/__init__.py"
    monkeypatch.setitem(sys.modules, "training_v1", foreign)
    with pytest.raises(RuntimeError, match="Foreign module"):
        train._load_runner(clean)


def test_missing_bank_stops_before_loader(monkeypatch, clean, capsys):
    monkeypatch.setattr(train, "ROOT", clean)
    monkeypatch.setattr(train, "_load_runner", lambda _: pytest.fail("must not import trainer"))
    out = clean / "run"
    args = ["--phase", "coarse", "--bank", str(clean / "missing"), "--bank-manifest-sha256", "0"*64,
            "--approval", str(clean / "absent.json"), "--approval-sha256", "1"*64,
            "--out", str(out), "--recipe", str(clean / "configs/pilot.json"), "--device", "cpu"]
    assert train.main(args) == 1
    assert not out.exists()
    assert "Original frozen TRAIN bank" in capsys.readouterr().err


def test_canonical_dispatch_exact_arguments(monkeypatch, clean, capsys):
    monkeypatch.setattr(train, "ROOT", clean)
    bank = clean / "bank"; bank.mkdir(); (bank / "FROZEN.json").write_text("{}")
    approval = clean / "approval.json"; approval.write_text("{}")
    out = clean / "output"
    recorded = {}
    def runner(**kwargs):
        recorded.update(kwargs)
        return {"status": "MOCK_DISPATCH_ONLY"}
    monkeypatch.setattr(train, "_load_runner", lambda root: runner)
    assert train.main(["--phase", "coarse", "--bank", str(bank), "--bank-manifest-sha256", "0"*64,
                       "--approval", str(approval), "--approval-sha256", "1"*64,
                       "--out", str(out), "--recipe", str(clean / "configs/pilot.json"), "--device", "cpu"]) == 0
    assert recorded["config"]["architecture"] == train.ARCHITECTURE
    assert recorded["config"]["max_steps"] == 32
    assert recorded["device"] == "cpu"
    assert recorded["expected_bank_sha256"] == "0"*64
    assert "NATIVE_DETAIL_UNTRAINED" in capsys.readouterr().out
    assert not out.exists()  # mocked dispatch is not claimed as execution
