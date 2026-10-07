from __future__ import annotations

import json
import math
import sys
from unittest.mock import MagicMock, patch

from scripts.run_full_pipeline_v3 import _log_stage_metrics, build_parser

from src.self_audit_pseudolabel.wandb_v3 import WandbV3Tracker


def _fake_wandb() -> MagicMock:
    module = MagicMock()
    module.init.return_value = MagicMock()
    return module


def test_disabled_tracker_does_not_import_or_call_wandb() -> None:
    fake = _fake_wandb()
    with patch.dict(sys.modules, {"wandb": fake}):
        tracker = WandbV3Tracker(
            enabled=False,
            mode="online",
            project="test-project",
            entity=None,
            run_name="disabled",
            run_dir=None,
            config={},
        )
        tracker.log({"loss": 1.0})
        tracker.finish()

    fake.init.assert_not_called()
    fake.log.assert_not_called()


def test_enabled_tracker_initializes_once_and_prefixes_history() -> None:
    fake = _fake_wandb()
    with patch.dict(sys.modules, {"wandb": fake}):
        tracker = WandbV3Tracker(
            enabled=True,
            mode="offline",
            project="test-project",
            entity="test-entity",
            run_name="run-1",
            run_dir="runs/run-1",
            config={"seed": 42},
        )
        tracker.log_history(
            "teacher",
            [
                {
                    "epoch": 0,
                    "step": 2,
                    "loss": 1.25,
                    "nested": {"accepted": 3},
                    "bad": math.nan,
                }
            ],
        )

    fake.init.assert_called_once()
    init_kwargs = fake.init.call_args.kwargs
    assert init_kwargs["project"] == "test-project"
    assert init_kwargs["entity"] == "test-entity"
    assert init_kwargs["name"] == "run-1"
    assert init_kwargs["mode"] == "offline"
    assert init_kwargs["dir"] == "runs/run-1"
    payload = fake.log.call_args.args[0]
    assert payload["teacher/epoch"] == 0
    assert payload["teacher/step"] == 2
    assert payload["teacher/loss"] == 1.25
    assert payload["teacher/nested/accepted"] == 3
    assert payload["teacher/bad"] is None


def test_json_logging_handles_nested_values_and_sdk_errors(tmp_path) -> None:
    fake = _fake_wandb()
    fake.log.side_effect = RuntimeError("temporary W&B failure")
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps({"foreground_mean": 0.8, "coverage": {"known": 0.9}}))

    with patch.dict(sys.modules, {"wandb": fake}):
        tracker = WandbV3Tracker(
            enabled=True,
            mode="online",
            project="test-project",
            entity=None,
            run_name="run-1",
            run_dir=tmp_path,
            config={},
        )
        tracker.log_json("evaluation", path)
        tracker.finish()

    assert tracker.warnings
    assert any("temporary W&B failure" in warning for warning in tracker.warnings)


def test_v3_parser_keeps_wandb_disabled_by_default(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--dataset",
            "acdc",
            "--root",
            str(tmp_path),
            "--split-manifest",
            str(tmp_path / "split.json"),
            "--out",
            str(tmp_path / "run"),
        ]
    )

    assert args.wandb is False
    assert args.wandb_mode == "disabled"


def test_v3_parser_accepts_explicit_wandb_settings(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--dataset",
            "acdc",
            "--root",
            str(tmp_path),
            "--split-manifest",
            str(tmp_path / "split.json"),
            "--out",
            str(tmp_path / "run"),
            "--wandb",
            "--wandb-mode",
            "online",
            "--wandb-project",
            "self-audit-v3-test",
            "--wandb-entity",
            "team",
            "--wandb-run-name",
            "run-name",
        ]
    )

    assert args.wandb is True
    assert args.wandb_mode == "online"
    assert args.wandb_project == "self-audit-v3-test"
    assert args.wandb_entity == "team"
    assert args.wandb_run_name == "run-name"


def test_completed_stage_metrics_route_to_tracker(tmp_path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text("[{\"epoch\": 0, \"loss\": 0.5}]")
    tracker = MagicMock()

    _log_stage_metrics(tracker, "teacher", metrics)

    tracker.log_json.assert_called_once_with("teacher", metrics)
