"""Candidate C configuration and baseline-interface contract tests.

These cover the schema half of the Candidate C contract only: that each
execution mode is accepted, validated, serialized, and actually delivered to
the model constructor, that historical configurations still load unchanged,
and that the ACDC baseline recipe is untouched.  Nothing here says anything
about whether Candidate C works.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from self_audit.training._utils import (
    CANDIDATE_C_DEFAULTS,
    DEFAULT_WINDOW_MODE,
    WINDOW_MODES,
    build_model_from_config,
    filter_model_config,
    validate_candidate_c_settings,
    validate_window_mode,
)
from self_audit.training.unified_config import (
    CandidateCSettings,
    ModelConfig,
    load_unified_config,
    parse_unified_config,
    resolve_downstream_config,
)

CANONICAL = "configs/self_audit_full.yaml"

EXPECTED_MODES = (
    "current",
    "feature_only",
    "free_offsets",
    "candidate_c",
    "candidate_c_no_fix",
    "direct_rollback",
)


def _canonical_dict() -> dict[str, Any]:
    return yaml.safe_load(Path(CANONICAL).read_text(encoding="utf-8"))


def _legacy_model_dict() -> dict[str, Any]:
    """A pre-Candidate-C model section, exactly as older configs wrote it."""
    return {
        "encoder_name": "convnext_tiny",
        "pretrained_encoder": False,
        "encoder_allow_fallback": True,
        "shared_channels": 96,
        "num_classes": 4,
        "window_k": 8,
        "max_turns": 3,
    }


class _ModeAwareBuilder:
    """Spy constructor recording exactly what the config path handed over.

    Used so the canonical config -- including ``pretrained_encoder: true`` --
    can be checked end to end without downloading encoder weights.  The real
    constructor is exercised separately in section 7.
    """

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def __call__(
        self,
        *,
        window_mode: str = DEFAULT_WINDOW_MODE,
        candidate_c: Any = None,
        **rest: Any,
    ) -> torch.nn.Module:
        self.kwargs = dict(rest)
        self.kwargs["window_mode"] = window_mode
        self.kwargs["candidate_c"] = candidate_c
        return torch.nn.Identity()


# ---------------------------------------------------------------------------
# 1. Mode surface and strict validation
# ---------------------------------------------------------------------------


def test_window_modes_are_exactly_the_contract_set() -> None:
    assert tuple(WINDOW_MODES) == EXPECTED_MODES
    assert DEFAULT_WINDOW_MODE == "current"


@pytest.mark.parametrize("mode", EXPECTED_MODES)
def test_every_mode_parses_from_config(mode: str) -> None:
    raw = _canonical_dict()
    raw["model"]["window_mode"] = mode
    config = parse_unified_config(raw)
    assert config.model.window_mode == mode
    assert config.model.runs_candidate_c_solver == (
        mode in {"candidate_c", "candidate_c_no_fix"}
    )


def test_unknown_window_mode_is_rejected() -> None:
    raw = _canonical_dict()
    raw["model"]["window_mode"] = "candidate_d"
    with pytest.raises(ValueError, match="window_mode"):
        parse_unified_config(raw)


def test_non_string_window_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        validate_window_mode(3)


def test_candidate_c_defaults_match_the_contract() -> None:
    settings = CandidateCSettings()
    assert settings.rho_feature_pixels == 1.0
    assert settings.lam == 1.0
    assert settings.lr is None
    assert settings.fix_threshold == 0.5
    assert settings.regress_threshold == 0.5
    assert settings.margin_fraction == 0.5
    assert settings.min_regress_mass == pytest.approx(1e-3)
    assert settings.replay_atol == pytest.approx(1e-5)
    assert settings.replay_rtol == pytest.approx(1e-5)
    assert settings.max_backtracks == 2
    # No enabled/enforce_fix knob: window_mode alone selects activation.
    assert set(settings.to_mapping()) == set(CANDIDATE_C_DEFAULTS)
    assert "enabled" not in settings.to_mapping()
    assert "enforce_fix" not in settings.to_mapping()


def test_candidate_c_unknown_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported model.candidate_c key"):
        validate_candidate_c_settings({"enabled": True})


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"lam": float("nan")}, "finite"),
        ({"lam": float("inf")}, "finite"),
        ({"rho_feature_pixels": 0.0}, "> 0"),
        ({"rho_feature_pixels": -1.0}, "> 0"),
        ({"fix_threshold": 1.5}, r"\[0, 1\]"),
        ({"regress_threshold": -0.1}, r"\[0, 1\]"),
        ({"margin_fraction": -0.5}, r"\[0, 1\]"),
        ({"margin_fraction": 1.5}, r"\[0, 1\]"),
        ({"min_regress_mass": -1e-9}, r"\[0, 1\]"),
        ({"min_regress_mass": 1.01}, r"\[0, 1\]"),
        ({"replay_atol": float("nan")}, "finite"),
        ({"max_backtracks": 3}, "<= 2"),
        ({"max_backtracks": 1.5}, "integer"),
        ({"lr": 0.0}, "> 0"),
    ],
)
def test_candidate_c_invalid_values_are_rejected(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_candidate_c_settings(payload)


def test_candidate_c_partial_mapping_fills_documented_defaults() -> None:
    resolved = validate_candidate_c_settings({"lam": 2.0})
    assert resolved["lam"] == 2.0
    assert resolved["rho_feature_pixels"] == 1.0
    assert set(resolved) == set(CANDIDATE_C_DEFAULTS)


def test_candidate_c_null_is_the_default_mapping() -> None:
    assert validate_candidate_c_settings(None) == dict(CANDIDATE_C_DEFAULTS)


def test_model_section_still_rejects_unknown_keys() -> None:
    raw = _canonical_dict()
    raw["model"]["free_offsets"] = True
    with pytest.raises(ValueError, match="Unknown keys in section 'model'"):
        parse_unified_config(raw)


def test_model_section_still_requires_historical_keys() -> None:
    raw = _canonical_dict()
    del raw["model"]["window_k"]
    with pytest.raises(ValueError, match="Missing required keys in section 'model'"):
        parse_unified_config(raw)


# ---------------------------------------------------------------------------
# 2. Old configurations still load with baseline behaviour
# ---------------------------------------------------------------------------


def test_config_without_switches_defaults_to_current() -> None:
    raw = _canonical_dict()
    raw["model"].pop("window_mode")
    raw["model"].pop("candidate_c")
    raw["training"]["rollout"].pop("predicted_history_exposure")
    raw["training"]["rollout"].pop("predicted_history_weight")
    config = parse_unified_config(raw)
    assert config.model.window_mode == "current"
    assert config.model.candidate_c == CandidateCSettings()
    assert config.training.rollout.predicted_history_exposure is False
    assert config.training.rollout.predicted_history_weight == pytest.approx(0.1)


def test_legacy_flat_config_resolves_to_current_mode() -> None:
    flat = {"model": _legacy_model_dict(), "num_classes": 4, "device": "cpu"}
    resolved = resolve_downstream_config(flat)
    assert resolved.is_unified is False
    assert resolved.window_mode == "current"
    assert resolved.candidate_c_settings == dict(CANDIDATE_C_DEFAULTS)
    # The resolved model config states the execution mode rather than omitting it.
    assert resolved.model_config["model"]["window_mode"] == "current"
    assert resolved.model_config["window_mode"] == "current"


def test_legacy_flat_config_with_bad_mode_is_rejected() -> None:
    flat = {"model": {**_legacy_model_dict(), "window_mode": "nope"}}
    resolved = resolve_downstream_config(flat)
    with pytest.raises(ValueError, match="window_mode"):
        _ = resolved.model_config


# ---------------------------------------------------------------------------
# 3. The switch actually reaches the model constructor
# ---------------------------------------------------------------------------


def test_filter_model_config_does_not_drop_the_mode() -> None:
    kwargs = filter_model_config({**_legacy_model_dict(), "window_mode": "candidate_c"})
    assert kwargs["window_mode"] == "candidate_c"
    assert kwargs["candidate_c"] == dict(CANDIDATE_C_DEFAULTS)


def test_filter_model_config_rejects_an_invalid_mode() -> None:
    with pytest.raises(ValueError, match="window_mode"):
        filter_model_config({**_legacy_model_dict(), "window_mode": "candidate_z"})


def test_filter_model_config_still_rejects_image_size_and_unknowns() -> None:
    with pytest.raises(ValueError, match="image_size belongs to the dataset config"):
        filter_model_config({**_legacy_model_dict(), "image_size": 256})
    with pytest.raises(ValueError, match="Unsupported model config key"):
        filter_model_config({**_legacy_model_dict(), "mystery": 1})


@pytest.mark.parametrize("mode", EXPECTED_MODES)
def test_each_mode_reaches_the_constructor(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _canonical_dict()
    raw["model"]["window_mode"] = mode
    raw["model"]["candidate_c"]["lam"] = 0.25
    legacy = parse_unified_config(raw).to_legacy_model_config()

    builder = _ModeAwareBuilder()
    monkeypatch.setattr("self_audit.models.self_audit_net.build_self_audit_net", builder)
    build_model_from_config(legacy, torch.device("cpu"))

    assert builder.kwargs["window_mode"] == mode
    assert builder.kwargs["candidate_c"]["lam"] == 0.25
    assert builder.kwargs["candidate_c"]["max_backtracks"] == 2
    # Baseline architecture kwargs are unchanged by the switch.
    assert builder.kwargs["encoder_name"] == "convnext_tiny"
    assert builder.kwargs["pretrained_encoder"] is True
    assert builder.kwargs["encoder_allow_fallback"] is False
    assert builder.kwargs["shared_channels"] == 96
    assert builder.kwargs["window_k"] == 8
    assert builder.kwargs["max_turns"] == 3


# ---------------------------------------------------------------------------
# 4. Rollout curriculum fields
# ---------------------------------------------------------------------------


def test_rollout_exposure_fields_parse_and_serialize() -> None:
    raw = _canonical_dict()
    raw["training"]["rollout"]["predicted_history_exposure"] = True
    raw["training"]["rollout"]["predicted_history_weight"] = 0.25
    config = parse_unified_config(raw)
    assert config.training.rollout.predicted_history_exposure is True
    assert config.training.rollout.predicted_history_weight == pytest.approx(0.25)
    assert config.to_dict()["training"]["rollout"]["predicted_history_exposure"] is True


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"predicted_history_exposure": "yes"}, "must be a boolean"),
        ({"predicted_history_exposure": 1}, "must be a boolean"),
        ({"predicted_history_weight": -0.1}, ">= 0"),
        ({"predicted_history_weight": float("inf")}, "finite"),
        ({"predicted_history_weight": True}, "must be a float"),
        ({"predicted_history_gain": 0.1}, "Unknown keys in section 'training.rollout'"),
    ],
)
def test_rollout_exposure_strict_validation(payload: dict[str, Any], message: str) -> None:
    raw = _canonical_dict()
    raw["training"]["rollout"].update(payload)
    with pytest.raises((ValueError, TypeError), match=message):
        parse_unified_config(raw)


def test_rollout_still_requires_tau_and_max_turns() -> None:
    raw = _canonical_dict()
    del raw["training"]["rollout"]["tau"]
    with pytest.raises(ValueError, match="Missing required keys in section 'training.rollout'"):
        parse_unified_config(raw)


# ---------------------------------------------------------------------------
# 5. Canonical serialization roundtrip and baseline preservation
# ---------------------------------------------------------------------------


def test_canonical_config_roundtrips_with_every_knob() -> None:
    config = load_unified_config(CANONICAL)
    data = config.to_dict()
    assert parse_unified_config(copy.deepcopy(data)).to_dict() == data
    assert set(data["model"]) == {
        "encoder_name",
        "pretrained_encoder",
        "fallback",
        "shared_channels",
        "num_classes",
        "window_k",
        "max_turns",
        "window_mode",
        "candidate_c",
    }
    assert data["model"]["candidate_c"] == dict(CANDIDATE_C_DEFAULTS)


@pytest.mark.parametrize("mode", EXPECTED_MODES)
def test_roundtrip_preserves_a_non_default_mode(mode: str) -> None:
    raw = _canonical_dict()
    raw["model"]["window_mode"] = mode
    raw["model"]["candidate_c"]["margin_fraction"] = 0.75
    data = parse_unified_config(raw).to_dict()
    reparsed = parse_unified_config(copy.deepcopy(data))
    assert reparsed.model.window_mode == mode
    assert reparsed.model.candidate_c.margin_fraction == pytest.approx(0.75)
    assert reparsed.to_dict() == data


def test_canonical_baseline_recipe_is_unchanged() -> None:
    config = load_unified_config(CANONICAL)
    assert config.model.window_mode == "current"
    assert config.model.encoder_name == "convnext_tiny"
    assert config.model.pretrained_encoder is True
    assert config.model.fallback is False
    assert config.model.shared_channels == 96
    assert config.model.window_k == 8
    assert config.model.max_turns == 3
    assert config.training.schedule.total_epochs == 130
    assert config.training.annotation_loss.stage_weights == (0.5, 0.7, 0.8, 1.0)
    assert config.training.rollout.predicted_history_exposure is False
    boundaries = [
        (i.start_epoch, i.end_epoch, i.name)
        for i in config.training.schedule.intervals
    ]
    assert boundaries == [
        (0, 100, "annotation_bootstrap"),
        (100, 120, "auditor_training"),
        (120, 130, "joint_self_audit"),
    ]


def test_model_config_has_no_redundant_activation_flags() -> None:
    fields = set(ModelConfig.__dataclass_fields__)
    assert "window_mode" in fields
    assert "candidate_c" in fields
    assert fields.isdisjoint({"enabled", "candidate_c_enabled", "enforce_fix"})


# ---------------------------------------------------------------------------
# 6. Checkpoint lineage of the execution mode
#
# The authoritative binding lives in provenance.resolve_model_identity /
# verify_model_config and is owned by the runtime worker.  This stream is
# responsible only for putting the mode and the complete solver settings into
# the config the verifier is handed.  Nothing here re-implements or bypasses
# that verifier.
# ---------------------------------------------------------------------------


def _tiny_model_section(mode: str, **candidate_c: Any) -> dict[str, Any]:
    """A small real model section; tiny so this stays a CPU boundary test."""
    return {
        "encoder_name": "convnext_tiny",
        "pretrained_encoder": False,
        "encoder_allow_fallback": True,
        "shared_channels": 16,
        "num_classes": 4,
        "window_k": 4,
        "max_turns": 2,
        "window_mode": mode,
        **({"candidate_c": candidate_c} if candidate_c else {}),
    }


def _build_tiny_net(mode: str, candidate_c: dict[str, Any] | None = None) -> torch.nn.Module:
    section = _tiny_model_section(mode, **(candidate_c or {}))
    return build_model_from_config({"model": section}, torch.device("cpu"))


def test_checkpoint_config_carries_the_mode_and_every_solver_setting() -> None:
    """What the verifier is given must be enough to bind the execution mode."""
    raw = _canonical_dict()
    raw["model"]["window_mode"] = "candidate_c"
    raw["model"]["candidate_c"]["lam"] = 0.25
    config = parse_unified_config(raw)

    # The dict written into the checkpoint payload.
    saved = config.to_dict()
    assert saved["model"]["window_mode"] == "candidate_c"
    assert saved["model"]["candidate_c"]["lam"] == 0.25
    assert set(saved["model"]["candidate_c"]) == set(CANDIDATE_C_DEFAULTS)

    # The dict handed to verify_model_config through the legacy adapter.
    legacy = config.to_legacy_model_config()
    assert legacy["model"]["window_mode"] == "candidate_c"
    assert set(legacy["model"]["candidate_c"]) == set(CANDIDATE_C_DEFAULTS)


def test_execution_mode_mismatch_is_refused_by_the_central_verifier() -> None:
    """A candidate_c checkpoint must not bind to a current model.

    The check belongs to the centralized verifier owned by the runtime worker;
    this asserts the config this stream produces actually drives it.  If the
    verifier stops reporting the mode the test fails rather than falling back
    to a local check that would bypass provenance.
    """
    from self_audit.provenance import resolve_model_identity, verify_model_config

    identity = resolve_model_identity(_build_tiny_net("candidate_c"))
    assert "window_mode" in identity, (
        "provenance.resolve_model_identity must report the execution mode; the "
        "binding is owned by the runtime worker and this stream adds no local fallback"
    )
    assert identity["window_mode"] == "candidate_c"

    declared = _tiny_model_section("candidate_c")
    with pytest.raises(ValueError, match="window_mode"):
        verify_model_config(_build_tiny_net("current"), {"model": declared})
    # A matching model binds and returns the identity.
    assert verify_model_config(_build_tiny_net("candidate_c"), {"model": declared})


def test_solver_setting_mismatch_is_refused_by_the_central_verifier() -> None:
    """The mode alone is not enough: the solver settings must bind too."""
    from self_audit.provenance import verify_model_config

    declared = _tiny_model_section("candidate_c")
    declared["candidate_c"] = {**dict(CANDIDATE_C_DEFAULTS), "lam": 0.25}
    with pytest.raises(ValueError, match="candidate_c.lam"):
        verify_model_config(_build_tiny_net("candidate_c"), {"model": declared})
    matching = _build_tiny_net("candidate_c", candidate_c={"lam": 0.25})
    assert verify_model_config(matching, {"model": declared})


def test_old_checkpoints_fail_closed_on_the_changed_schema() -> None:
    """An old last.pt is refused on resume; it is never migrated in silence.

    This is a different claim from "old config files still load": a plain YAML
    without the new keys parses with documented defaults, but a checkpoint whose
    saved config predates them cannot be resumed into the new schema.
    """
    from self_audit.training.unified_trainer import compare_execution_configs

    current = load_unified_config(CANONICAL).to_dict()
    old_saved = copy.deepcopy(current)
    old_saved["model"].pop("window_mode")
    old_saved["model"].pop("candidate_c")
    old_saved["training"]["rollout"].pop("predicted_history_exposure")
    old_saved["training"]["rollout"].pop("predicted_history_weight")

    with pytest.raises(ValueError, match="missing key 'model.candidate_c' in saved config"):
        compare_execution_configs(old_saved, current)

    # A saved config that does carry the keys but disagrees is also refused.
    changed = copy.deepcopy(current)
    changed["model"]["window_mode"] = "candidate_c"
    with pytest.raises(ValueError, match="Config mismatch on resume at 'model.window_mode'"):
        compare_execution_configs(changed, current)


# ---------------------------------------------------------------------------
# 7. End to end against the real constructor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", EXPECTED_MODES)
def test_mode_reaches_the_real_self_audit_net(mode: str) -> None:
    """The switch survives the whole config path into a real network.

    Deliberately tiny (and ``pretrained_encoder: False``) so this stays a CPU
    boundary test, not a model test.
    """
    model_section = {
        "encoder_name": "convnext_tiny",
        "pretrained_encoder": False,
        "encoder_allow_fallback": True,
        "shared_channels": 16,
        "num_classes": 4,
        "window_k": 4,
        "max_turns": 2,
        "window_mode": mode,
        "candidate_c": {"lam": 0.25, "margin_fraction": 0.75},
    }
    model = build_model_from_config({"model": model_section}, torch.device("cpu"))
    assert model.window_mode == mode
    assert model.candidate_c.lam == pytest.approx(0.25)
    assert model.candidate_c.margin_fraction == pytest.approx(0.75)
    assert model.candidate_c.max_backtracks == 2


def test_schema_rejects_everything_the_model_package_rejects() -> None:
    """The schema must not admit a value the runtime config would refuse."""
    from self_audit.models.self_audit_net import CandidateCConfig, resolve_candidate_c_config

    resolved = validate_candidate_c_settings({})
    assert resolve_candidate_c_config(resolved) == CandidateCConfig()
    for payload in ({"margin_fraction": 1.5}, {"lr": -1.0}, {"lam": float("nan")}):
        with pytest.raises((ValueError, TypeError)):
            validate_candidate_c_settings(payload)
        with pytest.raises((ValueError, TypeError)):
            resolve_candidate_c_config({**CANDIDATE_C_DEFAULTS, **payload})
