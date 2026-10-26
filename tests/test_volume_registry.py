from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from volume_preflight import IDS, bounded_steps, load_config, load_registry, validate_config  # noqa: E402


def test_registry_covers_every_requested_arm() -> None:
    registry, entries = load_registry(ROOT / "configs/native_volume/REGISTRY.json")
    assert registry["schema"] == "self_audit.volume.registry.v1"
    assert set(entries) == set(IDS)
    for exp_id, entry in entries.items():
        config = load_config(ROOT / "configs/native_volume" / entry["config"])
        assert (ROOT / "configs/native_volume" / entry["config"]).stem == exp_id
        assert entry["changed_factors"]


def test_historical_a0_descriptor_is_hash_pinned_and_bounded() -> None:
    config = load_config(ROOT / "configs/native_volume/V00.json")
    registry, entries = load_registry(ROOT / "configs/native_volume/REGISTRY.json")
    metadata = entries["V00"]
    source = ROOT / metadata["historical_a0_source"]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == metadata["historical_a0_sha256"]
    assert metadata["historical_recipe"]["outer_iterations"] == 10000
    assert metadata["historical_recipe"]["supervised_only_delay"] == 1000
    assert metadata["historical_recipe"]["batch_size"] == 24
    assert metadata["historical_recipe"]["crop_size"] == 256
    assert metadata["status"] == "HISTORICAL_REFERENCE_ONLY"
    assert config["training"]["microbatch"] == 24
    assert config["training"]["precision"] == "fp32"
    assert bounded_steps(config) == 8


def test_registry_declares_schedule_exposure_and_optimizer_confound_factors() -> None:
    _, entries = load_registry(ROOT / "configs/native_volume/REGISTRY.json")
    v01 = " ".join(entries["V01"]["changed_factors"]).lower()
    for token in ("batch", "prediction", "schedule", "horizon", "delay", "exposure"):
        assert token in v01
    o02 = " ".join(entries["O02"]["changed_factors"]).lower()
    for token in ("optimizer", "encoder_lr", "decoder_lr", "head_lr", "scheduler"):
        assert token in o02


def test_preflight_rejects_forbidden_mask_reference() -> None:
    config = load_config(ROOT / "configs/native_volume/V04.json")
    bad = dict(config)
    bad["val_mask"] = "should-never-be-read"
    with pytest.raises(ValueError, match="forbidden"):
        validate_config(bad)


def test_preflight_rejects_unbounded_nonhistorical_recipe() -> None:
    config = load_config(ROOT / "configs/native_volume/V04.json")
    bad = dict(config)
    bad["training"] = dict(config["training"], steps=11)
    with pytest.raises(ValueError, match="steps"):
        validate_config(bad)


def test_production_provenance_and_single_factor_controls_are_explicit() -> None:
    canonical = {
        "backbone": "dinov3_convnext_tiny",
        "source_repo": "/root/self-audit-pilot-artifacts/backbone/dinov3",
        "source_commit": "6876159a11b4df116f30f667f8c9888617df0751",
        "weights": "/root/self-audit-pilot-artifacts/backbone/pretrained/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
        "weights_sha256": "21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54",
    }
    for exp_id in ("V01", "V02", "V03", "V04", "V05"):
        config = load_config(ROOT / "configs/native_volume" / f"{exp_id}.json")
        assert config["model"] == canonical
        assert config["window"]["mode"] == "fixed"
        assert config["architecture"]["widths"] == [96, 128, 192, 256]
        assert config["training"]["pseudo_view"] == "strong_intensity"
        assert config["training"]["pseudo_student_mode"] == "eval"
    configs = {exp: load_config(ROOT / "configs/native_volume" / f"{exp}.json") for exp in ("V03", "V04", "V05")}
    assert [configs[exp]["architecture"]["depth_mode"] for exp in ("V03", "V04", "V05")] == ["off", "mix", "planar_control"]
    assert {configs[exp]["architecture"]["skip_fusion"] for exp in configs} == {"legacy"}
    o03 = load_config(ROOT / "configs/native_volume/O03.json")
    assert o03["training"]["pseudo_weight"] == 1.0
    assert o03["training"]["pseudo_ramp_steps"] > 0
    t00 = load_config(ROOT / "configs/native_volume/T00.json")
    assert t00["temporal"] == {"mode": "none", "weight": 0.0}


def test_cnn3d_rows_are_separate_four_stage_anisotropic_topologies() -> None:
    for exp_id in ("V06", "V07"):
        config = load_config(ROOT / "configs/native_volume" / f"{exp_id}.json")
        arch = config["architecture"]
        assert arch["name"] == "cnn3d"
        assert arch["widths"] == [16, 32, 64, 128]
        assert len(arch["widths"]) == len(arch["z_kernels"]) == len(arch["z_strides"]) == 4
        assert all(len(kernel) == 3 for kernel in arch["z_kernels"])
        assert all(len(stride) == 3 for stride in arch["z_strides"])
        assert arch["z_strides"] == [[1, 1, 1], [1, 2, 2], [1, 2, 2], [1, 2, 2]]
        if exp_id == "V06":
            assert arch["z_kernels"] == [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3]]
        else:
            assert arch["z_kernels"] == [[3, 3, 3]] * 4


def test_o02_optimizer_and_lr_confounds_are_explicitly_unvalidated() -> None:
    _, entries = load_registry(ROOT / "configs/native_volume/REGISTRY.json")
    config = load_config(ROOT / "configs/native_volume/O02.json")
    assert config["optimization"]["recipe"] == "grouped_adamw"
    assert config["optimization"]["head_lr"] == pytest.approx(3e-4)
    assert config["optimization"]["encoder_lr"] == pytest.approx(2e-5)
    factors = " ".join(entries["O02"]["changed_factors"]).lower()
    assert "scheduler" in factors and "unchanged" in factors
    assert "unvalidated" in factors
