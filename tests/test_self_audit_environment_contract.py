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
