from __future__ import annotations

import json
import math
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from self_audit.artifact_io import (
    ArtifactSerializationError,
    atomic_write_json,
    clean_wandb_payload,
    json_safe_artifact,
    read_json_artifact,
)
from self_audit.serialization import atomic_save_torch
from self_audit.training._utils import WandbLogger
import scripts.calibrate_threshold as calibrate_threshold_mod


class SubclassedTensor(torch.Tensor):
    pass


class ArbitraryObject:
    def __init__(self, value: int = 42) -> None:
        self.value = value


# ---------------------------------------------------------------------------
# json_safe_artifact
# ---------------------------------------------------------------------------

def test_json_safe_primitives_and_nesting() -> None:
    data = {
        "int": 1,
        "float": 2.5,
        "str": "hello",
        "bool": True,
        "none": None,
        "list": [1, "two", 3.0],
        "tuple": (4, 5),
    }
    safe = json_safe_artifact(data)
    assert safe == {
        "int": 1,
        "float": 2.5,
        "str": "hello",
        "bool": True,
        "none": None,
        "list": [1, "two", 3.0],
        "tuple": [4, 5],
    }


def test_json_safe_torch_tensors() -> None:
    # 0D tensors
    assert json_safe_artifact(torch.tensor(42)) == 42
    assert json_safe_artifact(torch.tensor(3.14)) == pytest.approx(3.14)
    assert json_safe_artifact(torch.tensor(float("nan"))) is None
    assert json_safe_artifact(torch.tensor(float("inf"))) is None
    assert json_safe_artifact(torch.tensor(float("-inf"))) is None

    # ND tensors
    t_1d = torch.tensor([1, 2, 3])
    assert json_safe_artifact(t_1d) == [1, 2, 3]

    t_nan = torch.tensor([1.0, float("nan"), float("inf"), float("-inf")])
    assert json_safe_artifact(t_nan) == [1.0, None, None, None]

    t_2d = torch.tensor([[1.0, 2.0], [float("nan"), 4.0]])
    assert json_safe_artifact(t_2d) == [[1.0, 2.0], [None, 4.0]]

    # Float8 scalar conversion without lossy int truncation
    if hasattr(torch, "float8_e4m3fn"):
        t_f8 = torch.tensor(1.5, dtype=torch.float8_e4m3fn)
        res = json_safe_artifact(t_f8)
        assert isinstance(res, float)
        assert res == 1.5

    # Complex tensors fail closed with exact path
    with pytest.raises(ArtifactSerializationError) as exc_info:
        json_safe_artifact({"scalar_complex": torch.tensor(1 + 2j)})
    assert "scalar_complex" in str(exc_info.value)

    with pytest.raises(ArtifactSerializationError) as exc_info:
        json_safe_artifact({"vector_complex": torch.tensor([1 + 2j, 3 + 4j])})
    assert "vector_complex[0]" in str(exc_info.value)


def test_json_safe_rejects_subclassed_tensor() -> None:
    base = torch.tensor([1.0, 2.0])
    sub = base.as_subclass(SubclassedTensor)
    payload = {"data": {"tensor": sub}}
    with pytest.raises(ArtifactSerializationError) as exc_info:
        json_safe_artifact(payload)
    assert "data.tensor" in str(exc_info.value)
    assert "SubclassedTensor" in str(exc_info.value)


def test_json_safe_numpy_types() -> None:
    assert json_safe_artifact(np.int64(10)) == 10
    assert json_safe_artifact(np.float64(2.5)) == 2.5
    assert json_safe_artifact(np.float32(float("nan"))) is None
    assert json_safe_artifact(np.float64(float("inf"))) is None

    arr = np.array([[1.0, np.nan], [np.inf, 4.0]])
    assert json_safe_artifact(arr) == [[1.0, None], [None, 4.0]]


def test_json_safe_paths() -> None:
    p = Path("/tmp/foo/bar.json")
    assert json_safe_artifact(p) == str(p)
    assert json_safe_artifact({"path": p}) == {"path": str(p)}


def test_json_safe_mapping_keys_and_collision() -> None:
    # Numeric and Path keys normalized to strings
    payload = {1: "a", 2: "b", Path("path_key"): "c"}
    safe = json_safe_artifact(payload)
    assert safe == {"1": "a", "2": "b", "path_key": "c"}

    # Collision between int and str key
    collision_int_str = {1: "one_int", "1": "one_str"}
    with pytest.raises(ArtifactSerializationError) as exc_info:
        json_safe_artifact(collision_int_str)
    assert "Mapping key collision" in str(exc_info.value)
    assert "'1'" in str(exc_info.value)

    # Collision between Path and str key
    collision_path_str = {Path("foo"): 1, "foo": 2}
    with pytest.raises(ArtifactSerializationError) as exc_info:
        json_safe_artifact(collision_path_str)
    assert "Mapping key collision" in str(exc_info.value)

    # Non-stringifiable key
    with pytest.raises(ArtifactSerializationError) as exc_info:
        json_safe_artifact({(1, 2): "tuple_key"})
    assert "Unsupported mapping key type" in str(exc_info.value)


def test_json_safe_rejects_arbitrary_objects() -> None:
    obj = ArbitraryObject()
    payload = {"records": [{"id": 1, "obj": obj}]}
    with pytest.raises(ArtifactSerializationError) as exc_info:
        json_safe_artifact(payload)
    assert "records[0].obj" in str(exc_info.value)
    assert "ArbitraryObject" in str(exc_info.value)


# ---------------------------------------------------------------------------
# atomic_write_json & read_json_artifact
# ---------------------------------------------------------------------------

def test_atomic_write_and_read_json(tmp_path: Path) -> None:
    target = tmp_path / "subdir" / "test_artifact.json"
    payload = {
        "metrics": {"dice": 0.85, "empty_dice": float("nan"), "inf_metric": float("inf")},
        "path": tmp_path / "foo.txt",
        "tensor": torch.tensor([1, 2]),
    }
    atomic_write_json(target, payload, indent=2, sort_keys=True)
    assert target.exists()

    # Raw content check: NaN/Inf should be null
    raw_text = target.read_text(encoding="utf-8")
    assert '"empty_dice": null' in raw_text
    assert '"inf_metric": null' in raw_text
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text

    # Read back with read_json_artifact
    loaded = read_json_artifact(target)
    assert loaded["metrics"]["dice"] == 0.85
    assert loaded["metrics"]["empty_dice"] is None
    assert loaded["metrics"]["inf_metric"] is None
    assert loaded["tensor"] == [1, 2]


def test_atomic_write_json_fsync_failure_preserves_original(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    target.write_text('{"status": "original"}', encoding="utf-8")

    def failing_fsync(fd: int) -> None:
        raise OSError("Simulated disk error during fsync")

    with patch("os.fsync", side_effect=failing_fsync):
        with pytest.raises(OSError, match="Simulated disk error during fsync"):
            atomic_write_json(target, {"status": "corrupted"})

    # Original file must remain untouched
    assert target.read_text(encoding="utf-8") == '{"status": "original"}'

    # No leftover temporary files
    tmps = list(tmp_path.glob(".*tmp*")) + list(tmp_path.glob("*.tmp"))
    assert tmps == []


def test_atomic_write_json_replace_failure_preserves_original(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    target.write_text('{"status": "original"}', encoding="utf-8")

    def failing_replace(src: str | Path, dst: str | Path) -> None:
        raise OSError("Simulated permission error during replace")

    with patch("os.replace", side_effect=failing_replace):
        with pytest.raises(OSError, match="Simulated permission error during replace"):
            atomic_write_json(target, {"status": "corrupted"})

    assert target.read_text(encoding="utf-8") == '{"status": "original"}'
    tmps = list(tmp_path.glob(".*tmp*")) + list(tmp_path.glob("*.tmp"))
    assert tmps == []


def test_read_json_artifact_rejects_nonstandard_constants(tmp_path: Path) -> None:
    target = tmp_path / "nan.json"
    target.write_text('{"bad": NaN}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_json_artifact(target)

    target.write_text('{"bad": Infinity}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_json_artifact(target)

    target.write_text('{"bad": -Infinity}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_json_artifact(target)


# ---------------------------------------------------------------------------
# clean_wandb_payload & WandbLogger
# ---------------------------------------------------------------------------

def test_clean_wandb_payload() -> None:
    payload = {
        "loss": 0.5,
        "nan_metric": float("nan"),
        "inf_metric": float("inf"),
        "scalar_tensor": torch.tensor(1.23),
        "nan_tensor": torch.tensor(float("nan")),
        "vector_tensor": torch.tensor([1.0, float("nan")]),
        "numpy_val": np.float64(2.71),
        "string": "text",
        "bool": True,
        "nested": {"inner_nan": float("nan"), "valid": 10},
    }
    cleaned = clean_wandb_payload(payload)
    assert cleaned["loss"] == 0.5
    assert cleaned["nan_metric"] is None
    assert cleaned["inf_metric"] is None
    assert cleaned["scalar_tensor"] == pytest.approx(1.23)
    assert cleaned["nan_tensor"] is None
    assert cleaned["vector_tensor"] == [1.0, None]
    assert cleaned["numpy_val"] == 2.71
    assert cleaned["string"] == "text"
    assert cleaned["bool"] is True
    assert cleaned["nested"] == {"inner_nan": None, "valid": 10}

    # Strict converter reuse: mapping key collisions must fail
    with pytest.raises(ArtifactSerializationError) as exc_info:
        clean_wandb_payload({1: "first", "1": "second"})
    assert "Mapping key collision" in str(exc_info.value)

    # Strict converter reuse: arbitrary custom objects must fail
    with pytest.raises(ArtifactSerializationError) as exc_info:
        clean_wandb_payload(ArbitraryObject())
    assert "ArbitraryObject" in str(exc_info.value)


def test_wandb_logger_telemetry_observability() -> None:
    with patch.dict("sys.modules", {"wandb": MagicMock()}):
        import wandb
        mock_run = MagicMock()
        mock_run.summary = {}
        wandb.init.return_value = mock_run

        logger = WandbLogger(
            enabled=True,
            project="test-proj",
            config={"lr": 0.001, "bad_cfg": float("nan")},
        )
        assert logger.enabled
        assert logger.config["bad_cfg"] is None
        assert logger.failed_log_count == 0
        assert logger.last_error is None

        # Normal log with NaN metric
        logger.log({"loss": 0.1, "empty_dice": float("nan")}, step=1)
        logged_metrics = wandb.log.call_args[0][0]
        assert logged_metrics["loss"] == 0.1
        assert logged_metrics["empty_dice"] is None

        # Simulated wandb log exception
        wandb.log.side_effect = RuntimeError("Network error")
        logger.log({"loss": 0.2}, step=2)
        assert logger.failed_log_count == 1
        assert isinstance(logger.last_error, RuntimeError)
        assert len(logger.telemetry_errors) == 1
        assert "log: Network error" in logger.telemetry_errors[0]

        # Summary
        wandb.log.side_effect = None
        logger.set_summary({"best_dice": 0.92, "nan_metric": float("nan")})
        assert mock_run.summary["best_dice"] == 0.92
        assert mock_run.summary["nan_metric"] is None

        # Finish with exit code
        logger.finish(exit_code=0)
        wandb.finish.assert_called_with(exit_code=0)
        assert logger._run is None


# ---------------------------------------------------------------------------
# calibrate_threshold weights_only enforcement
# ---------------------------------------------------------------------------

def test_calibrate_threshold_load_enforces_weights_only(tmp_path: Path) -> None:
    # Valid transition cache
    valid_cache = {
        "transitions": [
            {
                "step": 0,
                "delta_q": 0.15,
                "confidence": 0.8,
            }
        ]
    }
    cache_path = tmp_path / "valid_cache.pt"
    atomic_save_torch(valid_cache, cache_path)

    loaded = calibrate_threshold_mod._load(cache_path)
    assert "transitions" in loaded
    assert len(loaded["transitions"]) == 1

    # Malicious or custom pickled object
    class Exploit:
        def __reduce__(self):
            return (os.system, ("echo exploited",))

    exploit_cache = {"exploit": Exploit()}
    exploit_path = tmp_path / "exploit_cache.pt"
    # Save with standard pickle to simulate legacy / malicious cache
    torch.save(exploit_cache, exploit_path)

    with pytest.raises(ValueError) as exc_info:
        calibrate_threshold_mod._load(exploit_path)
    assert "unsupported legacy cache format" in str(exc_info.value)
    assert "weights_only=True" in str(exc_info.value)


# ---------------------------------------------------------------------------
# External evaluation report undefined metrics
# ---------------------------------------------------------------------------

def test_external_evaluation_nan_dice_to_null(tmp_path: Path) -> None:
    out_file = tmp_path / "report.json"
    report = {
        "evidence_class": "independent_external_evaluation",
        "macro_foreground_dice": float("nan"),
        "per_class_dice": {
            "LV": 0.88,
            "MYO": float("nan"),
            "RV": 0.0,
        },
    }
    atomic_write_json(out_file, report, indent=2, sort_keys=True)
    raw = out_file.read_text(encoding="utf-8")
    assert '"macro_foreground_dice": null' in raw
    assert '"MYO": null' in raw
    assert '"RV": 0.0' in raw

    loaded = read_json_artifact(out_file)
    assert loaded["macro_foreground_dice"] is None
    assert loaded["per_class_dice"]["MYO"] is None
    # Crucial distinction: None (undefined) vs 0.0 (total failure)
    assert loaded["per_class_dice"]["RV"] == 0.0
    assert loaded["per_class_dice"]["MYO"] is not loaded["per_class_dice"]["RV"]


# ---------------------------------------------------------------------------
# Real writer entrypoint tests (external evaluation, save_calibration, dump_bank)
# ---------------------------------------------------------------------------

def test_run_external_evaluation_entrypoint_with_undefined_metrics(tmp_path: Path, monkeypatch) -> None:
    # Setup fixture files for data loader and checkpoint
    volumes = tmp_path / "testing" / "volumes"
    masks = tmp_path / "testing" / "masks"
    volumes.mkdir(parents=True)
    masks.mkdir(parents=True)
    np.save(volumes / "A100_t00.npy", np.zeros((16, 16, 2), dtype=np.float32))
    np.save(masks / "A100_t00.npy", np.zeros((16, 16, 2), dtype=np.uint8))

    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save({"training_dataset": "acdc", "model": {}}, ckpt_path)

    import scripts.evaluate_external_mnms as external_evaluator

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    class FakeBinding:
        state_digest = "live-digest"
        model_identity = {"architecture": "fake"}

        def as_dict(self) -> dict[str, Any]:
            return {
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": "fake-sha",
                "state_digest": self.state_digest,
                "model_identity": self.model_identity,
            }

    monkeypatch.setattr(external_evaluator, "build_model_from_config", lambda config, device: FakeModel().to(device))
    monkeypatch.setattr(external_evaluator, "bind_evaluation_checkpoint", lambda *args, **kwargs: FakeBinding())
    monkeypatch.setattr(external_evaluator, "verify_bound_state", lambda *args, **kwargs: "live-digest")

    # Inject nested evaluation results containing NaN / undefined metrics (e.g. absent class Dice)
    monkeypatch.setattr(
        external_evaluator,
        "evaluate_volume_cohort",
        lambda *args, **kwargs: {
            "metric_space": "volume_resized",
            "native_dice_available": False,
            "volumes_evaluated": 1,
            "volumes_available": 1,
            "macro_dice": float("nan"),
            "patients": {
                "A100_t00": {
                    "per_class_dice": {
                        "LV": 0.90,
                        "MYO": float("nan"),
                        "RV": 0.0,
                    }
                }
            },
            "cohorts": {"unknown": {"case_ids": ["A100_t00"]}},
        },
    )

    out_file = tmp_path / "external_out.json"
    result = external_evaluator.run_external_evaluation(
        config={
            "external_test": {
                "dataset": "mnms",
                "split": "testing",
                "raw_to_acdc": {0: 0, 1: 3, 2: 2, 3: 1},
            },
            "image_size": 16,
            "num_classes": 4,
            "depth_axis": 2,
            "audit": {"tau_accept": 0.25, "t_max": 1},
            "model": {"num_classes": 4},
        },
        checkpoint=ckpt_path,
        data_root=tmp_path,
        split="testing",
        tau_accept=None,
        device=torch.device("cpu"),
        output=out_file,
    )

    assert out_file.is_file()
    raw_text = out_file.read_text(encoding="utf-8")
    assert '"macro_dice": null' in raw_text
    assert '"MYO": null' in raw_text
    assert '"RV": 0.0' in raw_text
    assert "NaN" not in raw_text

    loaded = read_json_artifact(out_file)
    assert loaded["metrics"]["macro_dice"] is None
    assert loaded["metrics"]["patients"]["A100_t00"]["per_class_dice"]["MYO"] is None
    assert loaded["metrics"]["patients"]["A100_t00"]["per_class_dice"]["RV"] == 0.0


def test_save_calibration_writer_preserves_old_file_on_error(tmp_path: Path) -> None:
    from self_audit.evaluation.threshold import save_calibration

    cal_path = tmp_path / "calibration.json"
    cal_path.write_text('{"preexisting": "original_calibration"}', encoding="utf-8")

    with patch("os.fsync", side_effect=OSError("Simulated disk error in fsync")):
        with pytest.raises(OSError, match="Simulated disk error in fsync"):
            save_calibration(
                cal_path,
                tau_accept=0.1,
                neutral_margin=0.005,
                source_split="val",
                checkpoint_path=None,
                t_max=3,
                threshold_grid=[0.0, 0.1, 0.2],
                selected_row={"threshold": 0.1, "final_macro_dice": 0.8, "metric_space": "slice_proxy"},
                metric_space="slice_proxy",
            )

    # Pre-existing file must remain intact
    assert cal_path.read_text(encoding="utf-8") == '{"preexisting": "original_calibration"}'
    # Temporary files must be cleaned up
    assert list(tmp_path.glob(".*tmp*")) == []


def test_dump_bank_writer_preserves_old_file_on_error(tmp_path: Path) -> None:
    from self_audit.evaluation.transition_bank import dump_bank

    bank_path = tmp_path / "bank.json"
    bank_path.write_text('{"preexisting": "original_bank"}', encoding="utf-8")

    with patch("self_audit.evaluation.transition_bank.validate_bank"):
        with patch("os.replace", side_effect=OSError("Simulated replace error")):
            with pytest.raises(OSError, match="Simulated replace error"):
                dump_bank({"valid": "bank_data"}, bank_path)

    # Pre-existing file must remain intact
    assert bank_path.read_text(encoding="utf-8") == '{"preexisting": "original_bank"}'
    assert list(tmp_path.glob(".*tmp*")) == []
