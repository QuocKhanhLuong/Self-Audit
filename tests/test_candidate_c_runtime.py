"""Runtime binding tests: equal weights must not describe different mechanisms.

Candidate C adds no parameter and no buffer, so a ``current`` network and a
``candidate_c`` network share every tensor, every ``state_digest`` and every
checkpoint hash while computing different transitions.  Everything here is
about the one consequence of that fact: the recorded identity, and therefore
every calibration lineage comparison built on it, must still be able to tell
the two apart -- while leaving a baseline model's historical signature exactly
as it was, so existing artifacts stay comparable.

Scope note: these are CPU fixture tests on small modules and random tensors.
They say nothing about GPU numerics, real data, or Candidate C's effectiveness.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from self_audit.provenance import (
    DEFAULT_WINDOW_MODE,
    UNKNOWN,
    resolve_model_identity,
    state_digest,
    verify_model_config,
)

CANDIDATE_C_DEFAULTS: dict[str, Any] = {
    "rho_feature_pixels": 1.0,
    "lam": 1.0,
    "lr": None,
    "fix_threshold": 0.5,
    "regress_threshold": 0.5,
    "margin_fraction": 0.5,
    "min_regress_mass": 1e-3,
    "replay_atol": 1e-5,
    "replay_rtol": 1e-5,
    "max_backtracks": 2,
}


class _Settings:
    """Stands in for the frozen solver-settings dataclass, via ``as_dict``."""

    def __init__(self, **overrides: Any) -> None:
        self._values = dict(CANDIDATE_C_DEFAULTS)
        self._values.update(overrides)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)


class _LegacyModule(nn.Module):
    """A module from before the execution-mode switches existed."""

    def __init__(self) -> None:
        super().__init__()
        self.num_classes = 4
        self.max_turns = 3
        self.weight = nn.Parameter(torch.zeros(3, 3))


class _MechanismModule(_LegacyModule):
    """The same architecture, able to report which mechanism it runs."""

    def __init__(self, mode: str, **setting_overrides: Any) -> None:
        super().__init__()
        self.window_mode = mode
        self.candidate_c = _Settings(**setting_overrides)


def _legacy_signature(identity: dict[str, Any]) -> str:
    """Recompute the pre-mechanism signature formula, independently.

    Hard-coding the historical construction here is the point: it fails if a
    later edit changes what a baseline model signs as, which would silently
    invalidate every artifact recorded before that edit.
    """

    payload = {
        key: identity[key]
        for key in sorted(identity)
        if key
        not in {
            "parameter_count",
            "signature",
            "window_mode",
            "candidate_c_settings",
            "candidate_c_signature",
        }
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"model_identity.v1\x00{encoded}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. What the identity records
# ---------------------------------------------------------------------------


def test_a_legacy_module_reports_an_unknown_mechanism_never_a_guessed_one() -> None:
    identity = resolve_model_identity(_LegacyModule())
    assert identity["window_mode"] == UNKNOWN
    assert identity["candidate_c_settings"] is None
    assert identity["candidate_c_signature"] == UNKNOWN


def test_a_mechanism_module_reports_its_live_mode_and_settings() -> None:
    identity = resolve_model_identity(_MechanismModule("candidate_c", lam=0.25))
    assert identity["window_mode"] == "candidate_c"
    assert identity["candidate_c_settings"]["lam"] == 0.25
    assert identity["candidate_c_settings"]["lr"] is None
    assert identity["candidate_c_signature"] != UNKNOWN


def test_unreadable_settings_sign_as_unknown_rather_than_as_absent() -> None:
    """An unreportable settings block must not collide with a readable one."""

    class _Opaque:
        def as_dict(self) -> dict[str, Any]:
            return {"lam": object()}

    module = _MechanismModule("candidate_c")
    module.candidate_c = _Opaque()
    identity = resolve_model_identity(module)
    assert identity["candidate_c_settings"] is None
    assert identity["candidate_c_signature"] == UNKNOWN
    readable = resolve_model_identity(_MechanismModule("candidate_c"))
    assert identity["signature"] != readable["signature"]


# ---------------------------------------------------------------------------
# 2. What the signature does and does not change
# ---------------------------------------------------------------------------


def test_the_baseline_signature_is_byte_for_byte_the_historical_one() -> None:
    for module in (_LegacyModule(), _MechanismModule(DEFAULT_WINDOW_MODE)):
        identity = resolve_model_identity(module)
        assert identity["signature"] == _legacy_signature(identity)


def test_teaching_a_module_to_report_the_baseline_mode_changes_no_signature() -> None:
    """Compatibility: gaining the ability to report ``current`` changes nothing.

    The same instance is compared with and without the mechanism attributes, so
    the class name -- which is itself part of the signature -- is held fixed.
    """

    module = _MechanismModule(DEFAULT_WINDOW_MODE)
    declared = resolve_model_identity(module)
    del module.window_mode
    del module.candidate_c
    legacy = resolve_model_identity(module)
    assert legacy["window_mode"] == UNKNOWN
    assert legacy["signature"] == declared["signature"]


def test_inert_solver_settings_do_not_change_the_baseline_signature() -> None:
    """Under ``current`` the solver never runs, so its settings are not identity."""

    plain = resolve_model_identity(_MechanismModule(DEFAULT_WINDOW_MODE))
    tweaked = resolve_model_identity(_MechanismModule(DEFAULT_WINDOW_MODE, lam=99.0))
    assert plain["signature"] == tweaked["signature"]


@pytest.mark.parametrize(
    "mode", ["candidate_c", "candidate_c_no_fix", "direct_rollback", "feature_only", "free_offsets"]
)
def test_every_non_default_mechanism_signs_differently_from_the_baseline(mode: str) -> None:
    baseline = resolve_model_identity(_MechanismModule(DEFAULT_WINDOW_MODE))
    other = resolve_model_identity(_MechanismModule(mode))
    assert other["signature"] != baseline["signature"]


def test_two_different_non_default_mechanisms_sign_differently() -> None:
    left = resolve_model_identity(_MechanismModule("candidate_c"))
    right = resolve_model_identity(_MechanismModule("candidate_c_no_fix"))
    assert left["signature"] != right["signature"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lam", 0.5),
        ("rho_feature_pixels", 2.0),
        ("fix_threshold", 0.7),
        ("margin_fraction", 0.25),
        ("replay_atol", 1e-4),
        ("max_backtracks", 1),
        ("lr", 0.01),
    ],
)
def test_changed_solver_settings_change_the_signature_under_a_solver_mode(
    field: str, value: Any
) -> None:
    baseline = resolve_model_identity(_MechanismModule("candidate_c"))
    changed = resolve_model_identity(_MechanismModule("candidate_c", **{field: value}))
    assert changed["signature"] != baseline["signature"]


def test_the_lr_sentinel_is_not_equal_to_zero() -> None:
    """``lr: null`` selects the normalized proposal rule; ``0.0`` is not that."""

    sentinel = resolve_model_identity(_MechanismModule("candidate_c"))
    zero = resolve_model_identity(_MechanismModule("candidate_c", lr=0.0))
    assert sentinel["signature"] != zero["signature"]


# ---------------------------------------------------------------------------
# 3. What verify_model_config refuses
# ---------------------------------------------------------------------------


def _config(**model: Any) -> dict[str, Any]:
    return {"model": {"num_classes": 4, "max_turns": 3, **model}}


def test_a_matching_mechanism_verifies() -> None:
    identity = verify_model_config(
        _MechanismModule("candidate_c"),
        _config(window_mode="candidate_c", candidate_c=dict(CANDIDATE_C_DEFAULTS)),
    )
    assert identity["window_mode"] == "candidate_c"


def test_a_disagreeing_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="window_mode"):
        verify_model_config(
            _MechanismModule("candidate_c"), _config(window_mode=DEFAULT_WINDOW_MODE)
        )


def test_a_non_default_mode_against_a_module_without_one_is_refused() -> None:
    """The baseline network must never run under another mode's name."""

    with pytest.raises(ValueError, match="refusing to bind without verification"):
        verify_model_config(_LegacyModule(), _config(window_mode="candidate_c"))


def test_the_default_mode_against_a_module_without_one_is_allowed() -> None:
    identity = verify_model_config(_LegacyModule(), _config(window_mode=DEFAULT_WINDOW_MODE))
    assert identity["window_mode"] == UNKNOWN


def test_a_mode_that_is_not_a_name_is_refused() -> None:
    with pytest.raises(ValueError, match="window_mode"):
        verify_model_config(_MechanismModule("candidate_c"), _config(window_mode=7))


def test_disagreeing_solver_settings_are_refused_under_a_solver_mode() -> None:
    with pytest.raises(ValueError, match="candidate_c.lam"):
        verify_model_config(
            _MechanismModule("candidate_c", lam=1.0),
            _config(window_mode="candidate_c", candidate_c={"lam": 0.5}),
        )


def test_an_unreportable_solver_setting_is_refused_under_a_solver_mode() -> None:
    with pytest.raises(ValueError, match="candidate_c.not_a_setting"):
        verify_model_config(
            _MechanismModule("candidate_c"),
            _config(window_mode="candidate_c", candidate_c={"not_a_setting": 1.0}),
        )


def test_inert_solver_settings_are_not_compared_under_the_baseline_mode() -> None:
    """Two ``current`` runs compute the same thing whatever the inert values say."""

    identity = verify_model_config(
        _MechanismModule(DEFAULT_WINDOW_MODE, lam=1.0),
        _config(window_mode=DEFAULT_WINDOW_MODE, candidate_c={"lam": 0.5}),
    )
    assert identity["window_mode"] == DEFAULT_WINDOW_MODE


def test_an_integer_valued_setting_matches_its_float_form() -> None:
    identity = verify_model_config(
        _MechanismModule("candidate_c"),
        _config(window_mode="candidate_c", candidate_c={"lam": 1}),
    )
    assert identity["candidate_c_settings"]["lam"] == 1.0


def test_an_explicit_null_settings_block_means_the_resolved_defaults() -> None:
    identity = verify_model_config(
        _MechanismModule("candidate_c"), _config(window_mode="candidate_c", candidate_c=None)
    )
    assert identity["window_mode"] == "candidate_c"


def test_a_mechanism_disagreement_does_not_mask_an_architecture_disagreement() -> None:
    with pytest.raises(ValueError) as excinfo:
        verify_model_config(
            _MechanismModule("candidate_c"),
            _config(num_classes=7, window_mode=DEFAULT_WINDOW_MODE),
        )
    message = str(excinfo.value)
    assert "num_classes" in message and "window_mode" in message


def test_a_config_without_a_mechanism_key_still_verifies() -> None:
    """Backward compatibility: nothing is invented for an unset key."""

    identity = verify_model_config(_MechanismModule("candidate_c"), _config())
    assert identity["window_mode"] == "candidate_c"


# ---------------------------------------------------------------------------
# 4. The real network, and the calibration lineage built on it
# ---------------------------------------------------------------------------


def _real_net(mode: str) -> nn.Module:
    from self_audit.models.self_audit_net import SelfAuditNet

    torch.manual_seed(17)
    return SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=2,
        window_mode=mode,
    ).eval()


def test_identical_weights_under_two_mechanisms_differ_only_in_the_signature() -> None:
    baseline = _real_net(DEFAULT_WINDOW_MODE)
    candidate = _real_net("candidate_c")
    assert state_digest(baseline) == state_digest(candidate)
    assert sum(1 for _ in baseline.buffers()) == sum(1 for _ in candidate.buffers())
    left = resolve_model_identity(baseline)
    right = resolve_model_identity(candidate)
    assert left["parameter_count"] == right["parameter_count"]
    assert left["signature"] != right["signature"]


def test_a_baseline_calibration_cannot_be_reused_under_candidate_c(tmp_path: Path) -> None:
    """The end-to-end hole this patch closes.

    A tau calibrated with ``window_mode: current`` used to bind under
    ``candidate_c``: same weights, same file hash, same source tree, so every
    compared lineage field agreed.  It must now be refused.
    """

    from torch.utils.data import DataLoader, Dataset

    from self_audit.evaluation.calibration_lineage import (
        LineageMismatchError,
        build_expected_lineage,
        verify_calibration_lineage,
    )
    from self_audit.data.common import VolumeRecord
    from self_audit.training._utils import bind_evaluation_checkpoint, save_checkpoint

    class _Records(Dataset):
        def __init__(self) -> None:
            self.records = [
                VolumeRecord(
                    case_id=f"{patient}_ED",
                    patient_id=patient,
                    image_path=Path(f"/fixture/{patient}_image.npy"),
                    mask_path=Path(f"/fixture/{patient}_mask.npy"),
                    split="val",
                )
                for patient in ("patient001", "patient002")
            ]
            self.image_size = 32
            self.depth_axis = 2
            self.foreground_only = False
            self.augment = False
            self.lower_percentile = 0.5
            self.upper_percentile = 99.5
            self.transform = None

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            record = self.records[index]
            return {
                "image": torch.zeros(3, 32, 32),
                "mask": torch.zeros(32, 32, dtype=torch.long),
                "case_id": record.case_id,
                "patient_id": record.patient_id,
            }

    loader = DataLoader(_Records(), batch_size=1, shuffle=False)
    checkpoint = tmp_path / "best_snapshot.pt"
    # The checkpoint carries no mode, exactly like every pre-Candidate-C file.
    save_checkpoint(checkpoint, _real_net(DEFAULT_WINDOW_MODE), epoch=1)

    def _lineage(model: nn.Module) -> dict[str, Any]:
        binding = bind_evaluation_checkpoint(
            model, [("checkpoint", checkpoint)], map_location="cpu"
        )
        return build_expected_lineage(
            binding=binding,
            loader=loader,
            split_name="val",
            metric_contract="foreground_dice_exclude_v1",
            metric_space="slice_proxy",
            neutral_margin=0.005,
            t_max=2,
            batch_size=1,
        )

    artifact_lineage = _lineage(_real_net(DEFAULT_WINDOW_MODE))
    same_mode = _lineage(_real_net(DEFAULT_WINDOW_MODE))
    # Sanity: the baseline still verifies against itself.
    verify_calibration_lineage(artifact_lineage, same_mode, artifact_name="baseline artifact")

    candidate_mode = _lineage(_real_net("candidate_c"))
    with pytest.raises(LineageMismatchError, match="model_identity.signature"):
        verify_calibration_lineage(
            artifact_lineage, candidate_mode, artifact_name="baseline artifact"
        )
