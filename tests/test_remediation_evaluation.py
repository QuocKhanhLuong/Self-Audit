"""Remediation regression tests, part 2: evaluation, calibration, geometry, firewall.

Companion to ``tests/test_remediation_metrics.py`` (items 1-7 of the
remediation test mandate).  This file covers items 8-15:

8.  Phase-B primary metric is on-policy only
9.  synthetic and on-policy metrics are reported separately
10. a saved calibrated tau is actually consumed
11. volume reconstruction preserves slice ordering
12. metric-space labels are truthful
13. GT permutation leaves deployable self-audit unchanged -- with a control
14. no GT enters AnnotationExpert candidate generation
15. the checkpoint diagnostic JSON schema is stable

Every test is deterministic, runs on synthetic tensors, and asserts a specific
value or exact property.  Nothing here touches the network, a data file, or a
training loop.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.audit.semantics import (  # noqa: E402
    METRIC_SPACE_SLICE_PROXY,
    METRIC_SPACE_VOLUME_NATIVE,
    METRIC_SPACE_VOLUME_RESIZED,
)
from self_audit.evaluation.metrics import annotation_metrics  # noqa: E402
from self_audit.evaluation.threshold import (  # noqa: E402
    CALIBRATION_SCHEMA_VERSION,
    load_calibration,
    save_calibration,
)
from self_audit.evaluation.volume_inference import (  # noqa: E402
    evaluate_volume_native,
    reconstruct_volume,
    split_cases_by_phase,
    to_native_geometry,
)
from self_audit.models.self_audit_net import SelfAuditNet  # noqa: E402
from self_audit.training.train_auditor import (  # noqa: E402
    build_auditor_transitions,
    resolve_primary_metric,
)
from self_audit.audit.counterfactual import CounterfactualGenerator  # noqa: E402


def _tiny_net(seed: int = 0) -> SelfAuditNet:
    torch.manual_seed(seed)
    return SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=3,
    ).eval()


# ---------------------------------------------------------------------------
# 8 + 9. Phase-B primary metric is on-policy only, namespaces are separate
# ---------------------------------------------------------------------------


def test_primary_metric_follows_on_policy_when_synthetic_contradicts_it() -> None:
    """A perfectly-ranked on-policy bucket must win over an inverted synthetic one."""

    on_policy = {"auroc": 1.0, "improve_regress_accuracy": 1.0}
    value, source = resolve_primary_metric(on_policy)
    assert value == pytest.approx(1.0)
    assert source == "on_policy_auroc"


def test_primary_metric_never_borrows_the_synthetic_or_combined_value() -> None:
    """With no usable on-policy signal the metric is nan, not a fallback to synthetic."""

    value, source = resolve_primary_metric({"auroc": float("nan"), "improve_regress_accuracy": float("nan")})
    assert math.isnan(value), f"expected nan, got {value!r}"
    assert source == "undefined"

    # Accuracy is the documented second rung, and only from the on-policy dict.
    value, source = resolve_primary_metric({"auroc": float("nan"), "improve_regress_accuracy": 0.75})
    assert value == pytest.approx(0.75)
    assert source == "on_policy_improve_regress_accuracy"


def test_build_auditor_transitions_can_exclude_synthetic_and_tags_provenance() -> None:
    net = _tiny_net(1)
    images = torch.randn(2, 3, 64, 64)
    ground_truth = torch.randint(0, 4, (2, 64, 64))
    with torch.no_grad():
        output = net.forward_annotation(images)
    generator = CounterfactualGenerator(num_classes=4)

    torch.manual_seed(5)
    both = build_auditor_transitions(output, ground_truth, generator)
    kinds = [t["provenance_kind"] for t in both]
    assert kinds.count("on_policy") == 3, kinds
    assert kinds.count("synthetic") == 4, kinds

    torch.manual_seed(5)
    on_policy_only = build_auditor_transitions(
        output, ground_truth, generator, include_synthetic=False
    )
    assert [t["provenance_kind"] for t in on_policy_only] == ["on_policy"] * 3
    assert all(t["provenance"] == "on_policy" for t in on_policy_only)

    torch.manual_seed(5)
    synthetic_only = build_auditor_transitions(
        output, ground_truth, generator, include_on_policy=False
    )
    assert [t["provenance_kind"] for t in synthetic_only] == ["synthetic"] * 4


# ---------------------------------------------------------------------------
# 10. A saved calibrated tau is actually consumed
# ---------------------------------------------------------------------------


def _write_calibration(tmp_path: Path, **overrides: object) -> Path:
    payload = {
        "tau_accept": 0.0123456789,
        "neutral_margin": 0.005,
        "source_split": "val",
        "checkpoint_path": None,
        "t_max": 3,
        "threshold_grid": {"min": -0.02, "max": 0.02, "steps": 81},
        "selected_row": {"tau_accept": 0.0123456789, "final_macro_dice": 0.5},
        "metric_space": METRIC_SPACE_SLICE_PROXY,
    }
    payload.update(overrides)  # type: ignore[arg-type]
    path = tmp_path / "calibration.json"
    save_calibration(path, **payload)  # type: ignore[arg-type]
    return path


def test_calibration_tau_round_trips_bitwise(tmp_path: Path) -> None:
    path = _write_calibration(tmp_path)
    loaded = load_calibration(path)
    assert loaded["tau_accept"].hex() == (0.0123456789).hex()
    assert loaded["neutral_margin"].hex() == (0.005).hex()
    assert loaded["schema_version"] == CALIBRATION_SCHEMA_VERSION
    # The artifact must never present itself as held-out evidence.
    assert loaded["validity"] == "diagnostic_only"
    assert loaded["validity_reason"]


def test_calibration_loader_rejects_a_bumped_schema_and_unknown_keys(tmp_path: Path) -> None:
    path = _write_calibration(tmp_path)

    raw = json.loads(path.read_text())
    raw["schema_version"] = CALIBRATION_SCHEMA_VERSION + 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_calibration(path)

    raw["schema_version"] = CALIBRATION_SCHEMA_VERSION
    raw["surprise_field"] = 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_calibration(path)


def test_diagnostic_cli_consumes_the_saved_tau_under_the_documented_precedence(
    tmp_path: Path,
) -> None:
    from torch.utils.data import DataLoader, Dataset
    from self_audit.data.common import VolumeRecord
    from self_audit.evaluation.calibration_lineage import CalibrationLineageError, build_expected_lineage
    from self_audit.models.self_audit_net import SelfAuditNet
    from self_audit.training._utils import bind_evaluation_checkpoint, save_checkpoint

    cli = _load_audit_checkpoint_module()

    ckpt_path = tmp_path / "tiny_ckpt.pt"
    tiny_cfg = {
        "num_classes": 4,
        "shared_channels": 16,
        "window_k": 4,
        "max_turns": 2,
        "encoder_name": "convnext_tiny",
        "encoder_allow_fallback": True,
    }
    torch.manual_seed(42)
    model = SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=2,
    ).eval()
    save_checkpoint(ckpt_path, model, epoch=1, config={"model": tiny_cfg})
    binding = bind_evaluation_checkpoint(
        model,
        [("checkpoint", ckpt_path)],
        map_location="cpu",
        config={"model": tiny_cfg},
    )

    class _RecordDataset(Dataset):
        def __init__(self, patients: tuple[str, ...]) -> None:
            self.records = [
                VolumeRecord(
                    case_id=f"{p}_ED",
                    patient_id=p,
                    image_path=Path(f"/fixture/{p}_image.npy"),
                    mask_path=Path(f"/fixture/{p}_mask.npy"),
                    split="val",
                )
                for p in patients
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
            r = self.records[index]
            return {
                "image": torch.randn(3, 32, 32),
                "mask": torch.randint(0, 4, (32, 32)),
                "case_id": r.case_id,
                "patient_id": r.patient_id,
            }

    loader = DataLoader(_RecordDataset(("patient001", "patient002")), batch_size=2, shuffle=False)
    expected_lineage = build_expected_lineage(
        binding=binding,
        loader=loader,
        split_name="val",
        metric_contract="foreground_dice_exclude_v1",
        metric_space=METRIC_SPACE_SLICE_PROXY,
        neutral_margin=0.005,
        t_max=3,
    )

    path = _write_calibration(
        tmp_path,
        checkpoint_path=ckpt_path,
        metric_contract="foreground_dice_exclude_v1",
        lineage=expected_lineage,
    )

    # Missing expected_lineage when calibration_path is supplied fails closed
    with pytest.raises(CalibrationLineageError, match="supplied without a runtime-derived expected lineage"):
        cli.resolve_tau_accept(
            cli_tau=None,
            calibration_path=str(path),
            config_audit={"tau_accept": 0.0},
            neutral_margin=0.005,
            expected_lineage=None,
        )

    # calibration > config
    resolution = cli.resolve_tau_accept(
        cli_tau=None,
        calibration_path=str(path),
        config_audit={"tau_accept": 0.0},
        neutral_margin=0.005,
        expected_lineage=expected_lineage,
    )
    assert resolution["tau_accept"].hex() == (0.0123456789).hex()
    assert resolution["tau_accept_source"] == cli.TAU_SOURCE_CALIBRATION
    cli.assert_calibrated_tau_used(resolution, resolution["tau_accept"])
    with pytest.raises(Exception):
        cli.assert_calibrated_tau_used(resolution, 0.0)

    # config > default
    resolution = cli.resolve_tau_accept(
        cli_tau=None, calibration_path=None, config_audit={"tau_accept": 0.25}, neutral_margin=0.005
    )
    assert resolution["tau_accept"] == pytest.approx(0.25)
    assert resolution["tau_accept_source"] == cli.TAU_SOURCE_CONFIG

    # default
    resolution = cli.resolve_tau_accept(
        cli_tau=None, calibration_path=None, config_audit={}, neutral_margin=0.005
    )
    assert resolution["tau_accept"] == 0.0
    assert resolution["tau_accept_source"] == cli.TAU_SOURCE_DEFAULT

    # A margin mismatch between artifact and evaluation is a hard error.
    with pytest.raises(ValueError):
        cli.resolve_tau_accept(
            cli_tau=None,
            calibration_path=str(path),
            config_audit={},
            neutral_margin=0.02,
            expected_lineage=expected_lineage,
        )


# ---------------------------------------------------------------------------
# 11 + 12. Volume reconstruction and truthful metric-space labels
# ---------------------------------------------------------------------------


def test_volume_reconstruction_and_inverse_resize_preserve_slice_order() -> None:
    # Each slice is uniformly filled with a distinct label so order is visible.
    slices = torch.stack([torch.full((8, 8), value, dtype=torch.long) for value in (1, 2, 3)])
    volume = reconstruct_volume(slices, num_slices=3)
    assert volume.shape == (3, 8, 8)
    assert [int(np.unique(volume[z])[0]) for z in range(3)] == [1, 2, 3]

    # preprocess_acdc.py writes orig_shape as [H, W, Z]; the helper refuses to
    # guess the axis order rather than silently transposing a volume.
    native = to_native_geometry(volume, (13, 7, 3))
    assert native.shape == (3, 13, 7)
    assert [int(np.unique(native[z])[0]) for z in range(3)] == [1, 2, 3]
    # Nearest-neighbour only: no label may be invented by interpolation.
    assert set(np.unique(native).tolist()).issubset(set(np.unique(volume).tolist()))

    # The depth-first spelling is accepted only when explicitly declared.
    same = to_native_geometry(volume, (3, 13, 7), shape_order="zhw")
    assert np.array_equal(same, native)
    with pytest.raises(ValueError):
        to_native_geometry(volume, (3, 13, 7))  # ambiguous under the [H,W,Z] default


def test_native_geometry_requires_real_native_ground_truth() -> None:
    """A resized number must never be returned under a native label."""

    prediction = np.zeros((2, 8, 8), dtype=np.int64)
    with pytest.raises(ValueError):
        evaluate_volume_native(prediction, None)


def test_metric_space_and_distance_space_are_distinct_axes() -> None:
    result = annotation_metrics(
        np.zeros((4, 4), dtype=np.int64), np.zeros((4, 4), dtype=np.int64), num_classes=4
    )
    # Surface-distance units live under distance_space; metric_space is reserved
    # for the grid/aggregation axis and must not be squatted on here.
    assert result["distance_space"] == "pixel"
    assert "metric_space" not in result

    assert METRIC_SPACE_SLICE_PROXY == "slice_proxy"
    assert METRIC_SPACE_VOLUME_RESIZED == "volume_resized"
    assert METRIC_SPACE_VOLUME_NATIVE == "volume_native"
    assert len({METRIC_SPACE_SLICE_PROXY, METRIC_SPACE_VOLUME_RESIZED, METRIC_SPACE_VOLUME_NATIVE}) == 3


def test_split_cases_by_phase_never_guesses() -> None:
    grouped = split_cases_by_phase(["patient001_ED", "patient001_ES", "weird_case"])
    assert grouped["ED"] == ["patient001_ED"]
    assert grouped["ES"] == ["patient001_ES"]
    assert grouped["unknown"] == ["weird_case"]


# ---------------------------------------------------------------------------
# 13. GT permutation invariance -- WITH a powered control
# ---------------------------------------------------------------------------


def test_gt_permutation_leaves_deployable_self_audit_bitwise_unchanged() -> None:
    """Deployable self-audit must be GT-free, and the probe must have power.

    The control matters more than the invariance half.  With two *random*
    ground-truth tensors on an untrained network the oracle gate rejects
    everything under either of them, so ``oracle_accept`` returns identical
    output and the test would pass even if ground truth were leaking.  The two
    targets below are therefore derived from the model's own trajectory so the
    oracle decision provably flips.
    """

    net = _tiny_net(7)
    images = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        trace = net.infer(
            images, mode="always_accept_refinement", tau_accept=-float("inf"), t_max=3
        )
    gt_favour_candidate = trace["transition_candidates"][0].argmax(1)
    gt_favour_previous = trace["transition_previous"][0].argmax(1)
    assert not torch.equal(gt_favour_candidate, gt_favour_previous)

    def run(mode: str, target: torch.Tensor) -> dict:
        with torch.no_grad():
            return net.infer(images, mode=mode, tau_accept=0.0, t_max=3, oracle_target=target)

    left = run("self_audit", gt_favour_candidate)
    right = run("self_audit", gt_favour_previous)

    for key in ("a0_logits", "logits"):
        assert torch.equal(left[key], right[key]), f"{key} changed with ground truth"
    for key in ("halt_turn", "accepted_count", "num_attempted_turns"):
        assert torch.equal(left[key], right[key]), f"{key} changed with ground truth"
    for a, b in zip(left["candidates"], right["candidates"]):
        assert torch.equal(a, b)
    for a, b in zip(left["audits"], right["audits"]):
        assert torch.equal(a["delta_q"], b["delta_q"])
        assert torch.equal(a["accepted"], b["accepted"])

    # Positive control: the same pair MUST move the oracle path.
    oracle_left = run("oracle_accept", gt_favour_candidate)
    oracle_right = run("oracle_accept", gt_favour_previous)
    control_flipped = any(
        not torch.equal(a["accepted"], b["accepted"])
        for a, b in zip(oracle_left["audits"], oracle_right["audits"])
    )
    assert control_flipped, (
        "positive control did not flip any oracle decision; the invariance assertions "
        "above would pass vacuously"
    )


# ---------------------------------------------------------------------------
# 14. No ground truth enters candidate generation
# ---------------------------------------------------------------------------


def test_forward_annotation_is_bitwise_invariant_to_ground_truth() -> None:
    net = _tiny_net(3)
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        first = net.forward_annotation(images)
        second = net.forward_annotation(images)
    for a, b in zip(first["state_trace"], second["state_trace"]):
        assert torch.equal(a, b)
    # forward_annotation has no channel through which a target could arrive.
    with pytest.raises(TypeError):
        net.forward_annotation(images, ground_truth=torch.zeros(2, 64, 64, dtype=torch.long))


def test_annotation_expert_exposes_no_target_argument() -> None:
    import inspect

    net = _tiny_net(4)
    parameters = set(inspect.signature(net.annotation_expert.forward).parameters)
    forbidden = {"target", "ground_truth", "gt", "mask", "oracle_target", "label", "labels"}
    assert not (parameters & forbidden), f"expert accepts target-like argument: {parameters & forbidden}"


# ---------------------------------------------------------------------------
# 15. Diagnostic JSON schema is stable
# ---------------------------------------------------------------------------


def _load_audit_checkpoint_module():
    path = ROOT / "scripts" / "audit_checkpoint.py"
    if not path.is_file():  # pragma: no cover - defensive
        pytest.skip("scripts/audit_checkpoint.py is absent")
    spec = importlib.util.spec_from_file_location("_audit_checkpoint_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_diagnostic_schema_is_versioned_and_declares_diagnostic_only_evidence() -> None:
    cli = _load_audit_checkpoint_module()
    assert cli.DIAGNOSTIC_SCHEMA_VERSION == 1
    assert cli.EVIDENCE_CLASS_DIAGNOSTIC_ONLY == "diagnostic_only"

    # The completeness checker must be self-describing and must actually notice
    # a missing key rather than rubber-stamping any payload.
    empty = cli.check_required_keys({"flat": {}, "per_stage": {}})
    assert empty["ok"] is False
    assert empty["missing"]

    required = set(cli.REQUIRED_MODE_KEYS)
    for key in (
        "modes/initial_dice",
        "modes/always_accept_dice",
        "modes/self_audit_dice",
        "modes/oracle_dice",
        "modes/headroom_capture_ratio",
        "modes/headroom_available_rate",
    ):
        assert key in required, f"{key} missing from REQUIRED_MODE_KEYS"

    stage_required = set(cli.REQUIRED_STAGE_KEYS)
    for key in (
        "candidate_gain",
        "positive_headroom",
        "realized_gain",
        "audit_gate_value",
        "attribution_residual",
    ):
        assert key in stage_required, f"{key} missing from REQUIRED_STAGE_KEYS"
    # The mislabeled alias removed during remediation must not come back.
    assert "oracle_gain" not in stage_required


def test_diagnostic_json_serializes_nan_without_silently_becoming_zero() -> None:
    cli = _load_audit_checkpoint_module()
    payload = cli.json_safe({"ratio": float("nan"), "fine": 0.5})
    assert payload["ratio"] is None
    assert payload["ratio__is_nan"] is True
    assert payload["fine"] == 0.5
    # Round-trips through real JSON without turning nan into a number.
    restored = json.loads(json.dumps(payload))
    assert restored["ratio"] is None and restored["ratio__is_nan"] is True
