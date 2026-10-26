from __future__ import annotations

import json
import math
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import volume_preflight  # noqa: E402
import volume_profile  # noqa: E402
import volume_smoke  # noqa: E402


class _FakeVolume(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(3, 4, kernel_size=1)

    def forward(self, image: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        del valid_mask
        return self.conv(image)


def _install_fake_builder(monkeypatch) -> None:
    module = types.ModuleType("self_audit.volume.models")
    module.build_volume_model = lambda config, synthetic=False: (_FakeVolume(), {"synthetic": synthetic, "test": True})
    monkeypatch.setitem(sys.modules, "self_audit.volume.models", module)


def test_preflight_cli_emits_json_receipt(capsys) -> None:
    code = volume_preflight.main(["--config", str(ROOT / "configs/native_volume/V04.json")])
    assert code == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "PASS"
    assert receipt["config_id"] == "V04"
    assert receipt["real_train_smoke"] == "NOT_RUN"


def test_smoke_calls_contract_and_is_bounded(monkeypatch) -> None:
    _install_fake_builder(monkeypatch)
    receipt = volume_smoke.run_smoke(ROOT / "configs/native_volume/V04.json", requested_steps=1, max_seconds=30)
    assert receipt["status"] == "PASS"
    assert receipt["output_shape"] == [1, 4, 3, 64, 64]
    assert receipt["optimizer_steps"] == 2  # one supervised + one pseudo commit
    assert receipt["coverage"]["trainer"] is True
    assert receipt["teacher_copy_finite"] is True
    assert math.isfinite(receipt["elapsed_seconds"]) and receipt["elapsed_seconds"] > 0


def test_smoke_deadline_checks_after_final_transaction_without_sleep(monkeypatch) -> None:
    _install_fake_builder(monkeypatch)
    samples = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(volume_smoke, "time", types.SimpleNamespace(perf_counter=lambda: next(samples)))
    with pytest.raises(TimeoutError, match="after transaction 0"):
        volume_smoke.run_smoke(ROOT / "configs/native_volume/V04.json", requested_steps=1,
                               crop_size=32, max_seconds=1.0)


def test_smoke_deadline_checks_after_correction_witness_without_sleep(monkeypatch) -> None:
    _install_fake_builder(monkeypatch)
    samples = iter((0.0, 0.0, 0.0, 0.0, 0.0, 2.0))
    monkeypatch.setattr(volume_smoke, "time", types.SimpleNamespace(perf_counter=lambda: next(samples)))
    with pytest.raises(TimeoutError, match="after correction witness"):
        volume_smoke.run_smoke(
            ROOT / "configs/native_volume/I01.json", requested_steps=1, crop_size=32, max_seconds=1.0,
            correction_config_path=ROOT / "configs/native_volume/I01_correction.json",
        )


def test_smoke_deadline_checks_after_teacher_witness_without_sleep(monkeypatch) -> None:
    _install_fake_builder(monkeypatch)
    samples = iter((0.0, 0.0, 0.0, 0.0, 0.0, 2.0))
    monkeypatch.setattr(volume_smoke, "time", types.SimpleNamespace(perf_counter=lambda: next(samples)))
    with pytest.raises(TimeoutError, match="after teacher witness"):
        volume_smoke.run_smoke(ROOT / "configs/native_volume/V04.json", requested_steps=1,
                               crop_size=32, max_seconds=1.0)


def test_profile_reports_warmup_and_valid_voxel_rates(monkeypatch) -> None:
    _install_fake_builder(monkeypatch)
    receipt = volume_profile.run_profile(ROOT / "configs/native_volume/V04.json", iterations=1, warmup=1, max_minutes=1)
    assert receipt["status"] == "PASS"
    assert receipt["warmup_excluded"] == 1
    assert receipt["valid_voxels_per_second"] > 0
    assert receipt["gpu_profile"] == "NOT_RUN"


def test_enabled_region_temporal_and_extra_update_witnesses() -> None:
    region = volume_smoke.run_smoke(ROOT / "configs/native_volume/R01.json", requested_steps=1, max_seconds=30)
    assert region["coverage"]["region_witness"]["region_loss"] > 0
    temporal = volume_smoke.run_smoke(ROOT / "configs/native_volume/T01.json", requested_steps=1, max_seconds=30)
    assert temporal["coverage"]["temporal_witness"]["temporal_loss"] > 0
    assert temporal["coverage"]["temporal_witness"]["temporal_valid_voxels"] > 0
    extra = volume_smoke.run_smoke(ROOT / "configs/native_volume/C01.json", requested_steps=1, max_seconds=30)
    assert extra["coverage"]["extra_supervised_updates"] == 1


def test_historical_v00_smoke_is_provenance_only() -> None:
    receipt = volume_smoke.run_smoke(ROOT / "configs/native_volume/V00.json", max_seconds=5)
    assert receipt["status"] == "HISTORICAL_REFERENCE_ONLY"
    assert receipt["steps"] == 0
    assert receipt["historical_config_path"].endswith("completed_20261025/configs/A0.json")


def test_absolute_preflight_bootstraps_repo_and_unknown_flags_fail() -> None:
    script = ROOT / "scripts/volume_preflight.py"
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(ROOT / "configs/native_volume/V04.json")],
        cwd="/tmp", capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "PASS"
    unknown = subprocess.run(
        [sys.executable, str(script), "--config", str(ROOT / "configs/native_volume/V04.json"), "--not-a-flag"],
        cwd="/tmp", capture_output=True, text=True, check=False,
    )
    assert unknown.returncode != 0
    assert "unrecognized arguments" in unknown.stderr


def test_smoke_and_profile_use_configured_logical_batch(monkeypatch) -> None:
    _install_fake_builder(monkeypatch)
    smoke = volume_smoke.run_smoke(ROOT / "configs/native_volume/V04.json", requested_steps=1,
                                   batch_size=2, max_seconds=30)
    assert smoke["logical_batch_size"] == 2
    assert smoke["microbatch"] == 2
    assert smoke["output_shape"] == [2, 4, 3, 64, 64]
    profile = volume_profile.run_profile(ROOT / "configs/native_volume/V04.json", iterations=1,
                                         warmup=1, microbatch=2, max_minutes=1)
    assert profile["logical_batch_size"] == 2
    assert profile["microbatch"] == 2
    assert profile["effective_batch_volumes"] == 4


def test_correction_witness_requires_explicit_protocol_before_builder(monkeypatch) -> None:
    _install_fake_builder(monkeypatch)
    built = []
    def spy_builder(*args, **kwargs):
        built.append(True)
        return _FakeVolume(), {"synthetic": True, "test": True}

    monkeypatch.setattr(volume_smoke, "_builder", spy_builder)
    with pytest.raises(ValueError, match="explicit matching --correction-config"):
        volume_smoke.run_smoke(ROOT / "configs/native_volume/I01.json", requested_steps=1, max_seconds=30)
    assert built == []
    explicit = volume_smoke.run_smoke(
        ROOT / "configs/native_volume/I01.json", requested_steps=1, max_seconds=30,
        correction_config_path=ROOT / "configs/native_volume/I01_correction.json",
    )
    assert explicit["coverage"]["correction_witness"]["protocol"] == "interactive_extra_supervision"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol", "automatic_only", "protocol"),
        ("mode", "geodesic_3d", "mode"),
        ("dataset_role", "TRAIN", "dataset_role"),
        ("click_budget", 0, "click_budget"),
    ],
)
def test_declared_correction_mismatch_fails_before_builder(tmp_path, monkeypatch, field, value, message) -> None:
    _install_fake_builder(monkeypatch)
    sidecar = json.loads((ROOT / "configs/native_volume/I01_correction.json").read_text())
    sidecar[field] = value
    sidecar_path = tmp_path / f"mismatch_{field}.json"
    sidecar_path.write_text(json.dumps(sidecar))
    built = []
    monkeypatch.setattr(volume_smoke, "_builder", lambda *args, **kwargs: built.append(True))
    with pytest.raises(ValueError, match=message):
        volume_smoke.run_smoke(
            ROOT / "configs/native_volume/I01.json", requested_steps=1, max_seconds=30,
            correction_config_path=sidecar_path,
        )
    assert built == []


def test_v04_explicit_correction_records_protocol_and_budget(monkeypatch) -> None:
    _install_fake_builder(monkeypatch)
    receipt = volume_smoke.run_smoke(
        ROOT / "configs/native_volume/V04.json", requested_steps=1, crop_size=32, max_seconds=30,
        correction_config_path=ROOT / "configs/native_volume/I01_correction.json",
    )
    witness = receipt["coverage"]["correction_witness"]
    assert witness["protocol"] == "interactive_extra_supervision"
    assert witness["mode"] == "image_guided"
    assert witness["supervision_budget"] == witness["prompt_count"] == 1
    assert witness["declared_protocol"] is None
    assert witness["declared_mode"] is None
    assert witness["declared_supervision_budget"] is None


def test_train_router_rejects_native_flags_on_legacy_schema_before_dispatch():
    script = ROOT / "train.py"
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(ROOT / "configs/pilot.json"),
         "--resume", "/missing/checkpoint", "--resume-sha256", "0" * 64,
         "--steps", "10000", "--geometry", "/missing/geometry.json"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "native-volume-only flags" in result.stderr
    assert "cannot verify then fresh-train" not in result.stderr


def test_train_router_rejects_legacy_flags_on_volume_schema_before_dispatch():
    script = ROOT / "train.py"
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(ROOT / "configs/native_volume/V04.json"),
         "--legacy-checkpoint", "/missing/checkpoint", "--legacy-manifest-sha256", "0" * 64,
         "--synthetic", "--out", "/tmp/native-volume-cli-noop"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "legacy-only flags" in result.stderr


def test_automatic_correction_has_zero_annotation_budget_and_interactive_refines(monkeypatch) -> None:
    _install_fake_builder(monkeypatch)
    automatic = volume_smoke.run_smoke(
        ROOT / "configs/native_volume/I00.json", requested_steps=1, max_seconds=30,
        correction_config_path=ROOT / "configs/native_volume/I00_correction.json",
    )["coverage"]["correction_witness"]
    interactive = volume_smoke.run_smoke(
        ROOT / "configs/native_volume/I01.json", requested_steps=1, max_seconds=30,
        correction_config_path=ROOT / "configs/native_volume/I01_correction.json",
    )["coverage"]["correction_witness"]
    assert automatic["protocol"] == "automatic_only"
    assert automatic["prompt_count"] == automatic["manual_voxels"] == 0
    assert automatic["prior_unchanged"] is True
    assert interactive["protocol"] == "interactive_extra_supervision"
    assert interactive["prompt_count"] == 1
    assert interactive["manual_voxels"] > 0
    assert interactive["propagated_voxels"] > 0
    assert interactive["prior_unchanged"] is False
    assert interactive["frozen_student_probability_sha256"] == automatic["frozen_student_probability_sha256"]


def test_runtime_bounds_reject_huge_and_nonfinite_before_builder(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("builder must not be called after invalid runtime input")
    monkeypatch.setattr(volume_smoke, "_builder", forbidden)
    monkeypatch.setattr(volume_profile, "_build", forbidden)
    with pytest.raises(ValueError, match="microbatch"):
        volume_smoke.run_smoke(ROOT / "configs/native_volume/V04.json", batch_size=100000, crop_size=16)
    with pytest.raises(ValueError, match="finite"):
        volume_smoke.run_smoke(ROOT / "configs/native_volume/V04.json", max_seconds=math.nan, crop_size=16)
    with pytest.raises(ValueError, match="microbatch"):
        volume_profile.run_profile(ROOT / "configs/native_volume/V04.json", iterations=1, warmup=0,
                                   microbatch=100000, crop_size=16, max_minutes=1)
    with pytest.raises(ValueError, match="finite"):
        volume_profile.run_profile(ROOT / "configs/native_volume/V04.json", iterations=1, warmup=0,
                                   crop_size=16, max_minutes=math.nan)


def test_production_declared_decoder_widths_match_mock_lateral(monkeypatch) -> None:
    config = volume_preflight.load_config(ROOT / "configs/native_volume/V01.json")
    declared = tuple(config["architecture"]["widths"])
    assert declared == (96, 128, 192, 256)

    class _Lateral:
        def __init__(self, channels: int) -> None:
            self.out_channels = channels

    class _Production(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.slice_model = types.SimpleNamespace(lateral=[_Lateral(x) for x in declared])

        def forward(self, image, valid_mask=None):
            del valid_mask
            return torch.zeros((image.shape[0], 4, *image.shape[2:]), device=image.device)

    builder = types.ModuleType("self_audit.volume.models")
    builder.build_volume_model = lambda cfg, synthetic=False: (_Production(), {"synthetic": synthetic,
                                                                               "decoder_widths": list(declared)})
    monkeypatch.setitem(sys.modules, "self_audit.volume.models", builder)
    model, provenance = volume_smoke._builder(config)
    assert tuple(layer.out_channels for layer in model.slice_model.lateral) == declared
    assert provenance["decoder_widths"] == list(declared)


def test_c01_counter_oracle_includes_extra_update_and_accumulation(tmp_path, monkeypatch) -> None:
    _install_fake_builder(monkeypatch)
    config = json.loads((ROOT / "configs/native_volume/C01.json").read_text())
    config["training"]["microbatch"] = 2
    config["training"]["accumulation"] = 2
    path = tmp_path / "C01.json"
    path.write_text(json.dumps(config))
    receipt = volume_smoke.run_smoke(path, requested_steps=1, crop_size=16, max_seconds=30)
    assert receipt["logical_batch_size"] == 4
    assert receipt["supervised_updates"] == 1
    assert receipt["supervised_extra_updates"] == 1
    assert receipt["pseudo_updates"] == 0
    assert receipt["ema_updates"] == 1
    assert receipt["optimizer_steps"] == 2
    assert receipt["coverage"]["extra_supervised_updates"] == 1


def test_cnn_runtime_override_keeps_four_matching_stages(monkeypatch) -> None:
    _install_fake_builder(monkeypatch)
    config = volume_preflight.load_config(ROOT / "configs/native_volume/V06.json")
    runtime, overrides = volume_preflight.synthetic_runtime_config(config, steps=1, crop_size=16, microbatch=1)
    assert config["architecture"]["widths"] == [16, 32, 64, 128]
    assert runtime["architecture"]["widths"] == [8, 16, 24, 32]
    assert overrides["architecture.widths"]["from"] == [16, 32, 64, 128]
    receipt = volume_smoke.run_smoke(ROOT / "configs/native_volume/V06.json", requested_steps=1,
                                     crop_size=16, max_seconds=30)
    assert receipt["status"] == "PASS"
    assert receipt["runtime_overrides"]["architecture.widths"]["to"] == [8, 16, 24, 32]


def test_cnn_anisotropic_plan_downsamples_xy_and_restores_output_with_finite_gradients() -> None:
    from self_audit.volume.models import build_volume_model

    config = volume_preflight.load_config(ROOT / "configs/native_volume/V06.json")
    runtime, overrides = volume_preflight.synthetic_runtime_config(config, steps=1, crop_size=32, microbatch=1)
    assert runtime["architecture"]["widths"] == [8, 16, 24, 32]
    assert runtime["architecture"]["z_kernels"] == config["architecture"]["z_kernels"]
    assert runtime["architecture"]["z_strides"] == [[1, 1, 1], [1, 2, 2], [1, 2, 2], [1, 2, 2]]
    model, _ = build_volume_model(runtime, synthetic=True)
    x = torch.rand(1, 3, 5, 32, 32, requires_grad=True)
    valid = torch.ones(1, 5, 32, 32, dtype=torch.bool)
    intermediate: list[tuple[int, ...]] = []
    hooks = [stage.register_forward_hook(lambda _m, _i, out: intermediate.append(tuple(out.shape)))
             for stage in model.encoder]
    try:
        output = model(x, valid_mask=valid)
    finally:
        for hook in hooks:
            hook.remove()
    assert output.shape == (1, 4, 5, 32, 32)
    assert any(shape[-1] < 32 and shape[-2] < 32 for shape in intermediate)
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
