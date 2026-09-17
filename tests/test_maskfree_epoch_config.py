"""Focused checks for the report-only per-epoch validation control plane."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_audit_maskfree.config import (  # noqa: E402
    OPERATIONAL_FIELDS,
    ConfigError,
    config_from_dict,
    load_config,
)


def _base_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset": "acdc",
        "data_root": "/images",
        "output_dir": "/runs",
    }
    payload.update(overrides)
    return payload


def test_epoch_validation_defaults_are_operational_and_reference_paths_stay_opaque() -> None:
    config = config_from_dict(_base_payload())
    assert config.epoch_validation is True
    assert config.epoch_reference_config is None
    assert "epoch_validation" in OPERATIONAL_FIELDS
    assert "epoch_reference_config" in OPERATIONAL_FIELDS

    configured = config.replace(
        epoch_validation=False,
        epoch_reference_config="/does/not/exist/epoch-reference.json",
    )
    assert configured.epoch_validation is False
    assert configured.epoch_reference_config == "/does/not/exist/epoch-reference.json"
    # Neither the optional control-plane path nor a legacy mask path is opened
    # or accepted by the image-only training configuration.
    with pytest.raises(ConfigError, match="forbidden configuration keys"):
        config_from_dict(_base_payload(reference_root="/manual/masks"))
    with pytest.raises(ConfigError, match="forbidden configuration keys"):
        config_from_dict(_base_payload(mask_root="/manual/masks"))
    with pytest.raises(ConfigError, match="non-empty string"):
        config_from_dict(_base_payload(epoch_reference_config=""))


def test_production_configs_and_train_cli_resolve_epoch_controls_without_opening_path(tmp_path: Path) -> None:
    for dataset in ("acdc", "mnms"):
        config = load_config(REPO_ROOT / f"configs/maskfree_{dataset}_150.yaml")
        assert config.epoch_validation is True
        assert config.epoch_reference_config is None

    missing_reference = tmp_path / "missing-reference-config.json"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/train_maskfree.py"),
        "--config",
        str(REPO_ROOT / "configs/maskfree_acdc_150.yaml"),
        "--no-epoch-validation",
        "--epoch-reference-config",
        str(missing_reference),
        "--print-config",
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    resolved = json.loads(result.stdout)
    assert resolved["epoch_validation"] is False
    assert resolved["epoch_reference_config"] == str(missing_reference)
    assert not missing_reference.exists()

    help_result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/train_maskfree.py"), "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert help_result.returncode == 0
    assert "--epoch-validation" in help_result.stdout
    assert "--no-epoch-validation" in help_result.stdout
    assert "separate evaluator" in help_result.stdout
    assert "no checkpoint selection" in " ".join(help_result.stdout.split())


@pytest.mark.parametrize(
    "script, args",
    [
        ("scripts/run_maskfree_full.sh", ["acdc"]),
        ("scripts/run_maskfree_acdc_mnms.sh", ["--datasets", "acdc,mnms"]),
    ],
)
def test_launcher_preview_records_epoch_control_per_dataset(
    tmp_path: Path, script: str, args: list[str],
) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable on this host")
    env = os.environ.copy()
    env.update({
        "RUN_FULL": "0",
        "WORKSPACE": str(tmp_path / "workspace"),
        "EPOCH_VALIDATION": "0",
        "ACDC_EPOCH_REFERENCE_CONFIG": str(tmp_path / "acdc-epoch.json"),
        "MNMS_EPOCH_REFERENCE_CONFIG": str(tmp_path / "mnms-epoch.json"),
        # Keep this invocation independent of any caller's shell default.
        "EPOCH_REFERENCE_CONFIG": "",
    })
    result = subprocess.run(
        [bash, str(REPO_ROOT / script), *args],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "epoch validation" in result.stdout
    assert "enabled=0" in result.stdout
    if script.endswith(("sequential.sh", "acdc_mnms.sh")):
        assert str(tmp_path / "acdc-epoch.json") in result.stdout
        assert str(tmp_path / "mnms-epoch.json") in result.stdout
    else:
        assert "separate evaluator" in result.stdout
    assert not (tmp_path / "workspace").exists()
