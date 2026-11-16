import json
from pathlib import Path

import pytest

from self_audit.candidate_pg.config import apply_arm, config_sha256, default_config, load_config, validate_config


def test_pg_base_has_exact_schema_and_arm_files_are_switch_only():
    config = load_config(Path("configs/candidate_pg/pg_k8_base.json"))
    assert config["schema"] == "self_audit.candidate_pg.v2"
    assert config["seed"] == 42
    assert config["data"]["split_lock"] == "configs/candidate_fuse/semi_k8.json"
    assert config["data"]["split_lock_sha256"] == "cdec2b85c549a1cf9a19eea8790b7d6dbe155fdbe7a4798b60315412cbead639"
    assert config["data"]["source_k8_sha256"] == "60dd11747570f3ac183e8636050a3ac3d73651cb43b37deba9ff7422b442a433"
    assert config["data"]["per_step"] == {"labeled": 1, "unlabeled": 2}
    assert config["teacher"]["semi"]["stream_weights"] == {"s1": 0.25, "s2": 0.25, "fp": 0.5}
    assert config["teacher"]["heads"]["sdf"] == {"channels": 3, "radius_mm": 5.0}
    assert config["teacher"]["semi"]["fp"] == "dropout3d_0.5_on_encoder_skips"
    assert config["training"]["optimizer"] == {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001, "schedule": "reuse candidate_fuse _schedule_lr"}
    assert config["cinema_init"]["optimizer"]["name"] == config["teacher"]["optimizer"]["name"] == "adamw"
    for name in ("A1", "A2", "A3", "A4"):
        arm = load_config(Path("configs/candidate_pg") / f"pg_k8_{name}.json")
        assert set(arm) == {"pg", "cbkd"}
        assert all(set(value) == {"enabled"} for value in arm.values())
    assert not any((Path("configs/candidate_pg") / f"a{i}.json").exists() for i in range(1, 5))
    assert apply_arm(config, "configs/candidate_pg/pg_k8_A4.json")["pg"]["enabled"] is True


def _assert_reference_subset(actual, reference):
    assert isinstance(actual, dict) and isinstance(reference, dict)
    for key, expected in reference.items():
        assert key in actual
        if isinstance(expected, dict):
            _assert_reference_subset(actual[key], expected)
        else:
            assert actual[key] == expected


def test_pg_base_recursively_matches_fixed_user_reference_fixture():
    actual = load_config(Path("configs/candidate_pg/pg_k8_base.json"))
    reference = json.loads(Path("tests/fixtures/pg_k8_base_reference.json").read_text())
    _assert_reference_subset(actual, reference)


def test_pg_config_rejects_hidden_or_unpaired_manifest_defaults(tmp_path):
    path = tmp_path / "bad.json"
    value = json.loads(Path("configs/candidate_pg/pg_k8_base.json").read_text())
    value["data"]["manifests"]["train"]["sha256"] = None
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="path/SHA"):
        load_config(path)


def test_pg_config_rejects_transcribed_stream_or_head_values(tmp_path):
    value = json.loads(Path("configs/candidate_pg/pg_k8_base.json").read_text())
    value["teacher"]["semi"]["stream_weights"]["s1"] = 1.25
    path = tmp_path / "wrong-stream.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="exact reference value changed"):
        load_config(path)


@pytest.mark.parametrize(
    ("arm_name", "pg_enabled", "cbkd_enabled", "expected_sha"),
    (
        ("A1", False, False, "f68212d85ae0c89e458954b00d32826393c4551afa13f5c8db3592d03f0c7e44"),
        ("A2", True, False, "8b4e7a9ccd66e82517939f30d7ceae10bb10992d5b109eb4355f2d557eabd9d1"),
        ("A3", False, True, "8cdb1042968c24d0da16e014ab0ee8923cd55dd5f9b7a5d2ab16200534914c17"),
        ("A4", True, True, "1f1d28c8a3f7226eca9934169515e1c4105e2686beac7f2ef119d5db3f87e850"),
    ),
)
def test_pg_each_resolved_arm_validates_and_has_locked_hash(arm_name, pg_enabled, cbkd_enabled, expected_sha):
    resolved = apply_arm(default_config(), Path("configs/candidate_pg") / f"pg_k8_{arm_name}.json")
    assert validate_config(resolved) == resolved
    assert resolved["pg"]["enabled"] is pg_enabled
    assert resolved["cbkd"]["enabled"] is cbkd_enabled
    assert config_sha256(resolved) == expected_sha


def test_pg_arm_switches_are_the_only_reference_values_allowed_to_change():
    resolved = apply_arm(default_config(), "configs/candidate_pg/pg_k8_A1.json")
    resolved["pg"]["enabled"] = 0
    with pytest.raises(ValueError, match="arm switch must be Boolean"):
        validate_config(resolved)
