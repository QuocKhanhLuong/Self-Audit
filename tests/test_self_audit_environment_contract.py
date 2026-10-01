"""The canonical environment declaration, its locks and the fail-fast validator agree."""
from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path

import pytest

import environment_contract as contract

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "environments" / "self-audit-canonical"


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\;]+)", line)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def test_declaration_requirements_and_both_locks_agree_on_critical_pins():
    spec = contract.load_environment()
    assert spec["python"]["series"] == "3.10"
    requirements = _pins(CANONICAL / "requirements.in")
    for variant, details in spec["variants"].items():
        lock = _pins(CANONICAL / details["lock"])
        for name, version in spec["critical_packages"].items():
            key = name.lower()
            assert lock[key].split("+")[0] == version, (variant, name)
            if key not in {"torch", "torchvision"}:
                assert requirements[key] == version, name
    assert _pins(CANONICAL / "requirements-cpu.lock")["torch"] == "2.4.1+cpu"
    assert _pins(CANONICAL / "requirements-cu121.lock")["torch"] == "2.4.1"
    assert _pins(CANONICAL / "requirements-cu121.lock")["nvidia-cuda-runtime-cu12"].startswith("12.1.")


def test_native_baselines_use_the_canonical_lock():
    for package in ("DSS_US", "SGSCN"):
        text = (ROOT / "baseline" / package / "environment" / "requirements.lock").read_text()
        assert "-r ../../../environments/self-audit-canonical/requirements-cpu.lock" in text
        assert not re.search(r"^[A-Za-z0-9_.-]+==", text, re.M)


def _fake_versions(monkeypatch, overrides):
    real = importlib.metadata.version

    def version(name):
        if name in overrides:
            if overrides[name] is None:
                raise importlib.metadata.PackageNotFoundError(name)
            return overrides[name]
        return real(name)

    monkeypatch.setattr(contract.importlib.metadata, "version", version)


def test_mismatch_fails_fast_and_lists_every_problem(monkeypatch):
    monkeypatch.delenv(contract.ALLOW_UNOFFICIAL_ENV_VAR, raising=False)
    _fake_versions(monkeypatch, {"numpy": "2.2.6", "phate": None, "torch": "2.7.0+cpu"})
    with pytest.raises(contract.EnvironmentMismatch) as error:
        contract.require_official_environment(variant="cpu")
    problems = error.value.report["problems"]
    assert "numpy 2.2.6 != 1.26.4" in problems
    assert "phate missing != 2.0.0" in problems
    assert "torch 2.7.0+cpu != 2.4.1" in problems
    assert error.value.report["official"] is False


def test_unofficial_override_is_recorded_not_silent(monkeypatch, capsys):
    monkeypatch.setenv(contract.ALLOW_UNOFFICIAL_ENV_VAR, "1")
    _fake_versions(monkeypatch, {"numpy": "2.2.6"})
    report = contract.require_official_environment(variant="cpu")
    assert report["official"] is False
    assert report["unofficial_override"] == contract.ALLOW_UNOFFICIAL_ENV_VAR
    assert "UNOFFICIAL ENVIRONMENT" in capsys.readouterr().err


def test_variant_detection_from_build_metadata(monkeypatch):
    assert contract._torch_variant("2.4.1+cpu") == "cpu"
    monkeypatch.setattr(contract.importlib.metadata, "requires",
                        lambda name: ["nvidia-cuda-runtime-cu12==12.1.105; platform_system == 'Linux'"])
    assert contract._torch_variant("2.4.1") == "cu121"
    monkeypatch.setattr(contract.importlib.metadata, "requires", lambda name: [])
    assert contract._torch_variant("2.4.1") == "unknown"


# Official entrypoints of the active milestone (Self-Audit core, shared_benchmark,
# CUTS, DSS-US, SGSCN). DFC, STEGO and PiCIE are legacy / out of scope.
CLI_ENTRYPOINTS = [
    "scripts/train_self_audit.py",
    "scripts/train_maskfree.py",
    "src/self_audit/training/train_annotation.py",
    "src/self_audit/training/train_auditor.py",
    "src/self_audit/training/finetune_joint.py",
    "scripts/evaluate_external_mnms.py",
    "scripts/evaluate_maskfree_epoch.py",
    "scripts/evaluate_maskfree_reference.py",
    "scripts/evaluate_cardiac_baseline_reference.py",
    "scripts/evaluate_visualize_shared_benchmark.py",
]


def _main_block(path: Path):
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            return ast.unparse(node)
    raise AssertionError(f"no __main__ block: {path}")


@pytest.mark.parametrize("entrypoint", CLI_ENTRYPOINTS)
def test_official_cli_entrypoints_enforce_the_canonical_environment_first(entrypoint):
    block = _main_block(ROOT / entrypoint)
    enforce = block.index("enforce_official_entrypoint(__file__)")
    assert enforce < block.index("main()")


def test_native_and_cuts_producers_enforce_the_canonical_environment():
    sgscn = (ROOT / "baseline/SGSCN/scripts/run_native.py").read_text()
    assert "environment = require_official_environment()" in sgscn
    assert '"environment_contract": environment' in sgscn
    cuts = (ROOT / "baseline/CUTS/src/cardiac_benchmark/provenance.py").read_text()
    assert "environment_contract = _official_environment_contract()" in cuts


def _load_dss_runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location("dss_run_native_contract", ROOT / "baseline/DSS_US/scripts/run_native.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dss_blocked_profiles_report_blocked_protocol_without_environment_check(monkeypatch, capsys):
    runner = _load_dss_runner()
    monkeypatch.setattr(contract, "require_official_environment",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("environment checked before gate")))
    monkeypatch.setattr(runner.sys, "argv", ["run_native.py", "--config",
                                              str(ROOT / "baseline/DSS_US/config/native/step1_ours_comb.yaml")])
    assert runner.main() == 2
    assert '"BLOCKED_PROTOCOL"' in capsys.readouterr().out


def test_dss_executable_profile_requires_canonical_environment_before_running(monkeypatch, capsys):
    runner = _load_dss_runner()
    monkeypatch.setattr(runner, "load_lock", lambda path: {"profile": "hypothetically-unblocked"})

    def mismatch(*args, **kwargs):
        raise contract.EnvironmentMismatch({"environment_id": "self-audit-canonical", "environment_version": "1",
                                            "problems": ["torch 2.7.0 != 2.4.1"], "official": False})

    monkeypatch.setattr(contract, "require_official_environment", mismatch)
    monkeypatch.setattr(runner.sys, "argv", ["run_native.py", "--config", "unblocked.yaml"])
    with pytest.raises(contract.EnvironmentMismatch, match="torch 2.7.0"):
        runner.main()
    assert capsys.readouterr().out == ""
