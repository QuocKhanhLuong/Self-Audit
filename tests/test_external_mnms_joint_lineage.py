"""External M&Ms evaluator regression tests for real training-config lineage.

The existing external tests in ``tests/test_candidate_c_mnms.py`` build the
checkpoint's ``config`` payload by hand as a small flat dict.  A real run does
not: ``UnifiedTrainer`` stamps ``UnifiedConfig.to_dict()`` into the checkpoint,
which is the nested Schema Version 1 tree with ``dataset`` as a mapping and a
fully populated ``model.candidate_c`` block.  Nothing pinned that shape, so a
rename inside ``to_dict()`` could break the external evaluator's ACDC identity
check or its Candidate C inheritance with every test still green.

These tests parse the **actual repository configs** -- the staged baseline and
both joint-from-epoch-1 profiles -- shrink only the model size, and drive
``scripts.evaluate_external_mnms.run_external_evaluation`` end to end against a
tiny synthetic M&Ms ``testing`` cohort.

Deliberate limits:

* Only lineage, provenance and refusal behaviour are asserted.  No segmentation
  metric is asserted and none may be read as a performance claim: the weights
  are randomly initialised and the cohort is synthetic noise.
* The models built here use ``_FallbackHierarchicalEncoder`` unless ``timm`` is
  installed, so these tests say nothing about the real ConvNeXt-Tiny backbone.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from self_audit.training._utils import build_model_from_config, save_checkpoint
from self_audit.training.checkpoint_commit import make_best_reference, publish_best_alias
from self_audit.training.unified_config import parse_unified_config

from scripts.evaluate_external_mnms import run_external_evaluation


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
PROTOCOL_CONFIG = CONFIG_DIR / "self_audit_acdc_to_mnms.yaml"

STAGED_ACDC = CONFIG_DIR / "self_audit_full.yaml"
JOINT_ACDC = CONFIG_DIR / "self_audit_joint_from_start.yaml"
JOINT_MNMS = CONFIG_DIR / "self_audit_joint_from_start_mnms.yaml"

# Small enough that a full external evaluation of one two-slice volume is a
# couple of CPU seconds.  Only the size changes; the nested config structure
# under test is exactly the repository's.
TINY_MODEL = {"shared_channels": 16, "window_k": 4, "max_turns": 2}
IMAGE_SIZE = 32
CANONICAL_MAPPING = {0: 0, 1: 3, 2: 2, 3: 1}


def _tiny_unified_config(config_path: Path, *, window_mode: str | None = None):
    """Parse a repository config, shrinking only the model and encoder needs.

    ``pretrained_encoder`` is turned off and ``fallback`` on so the test never
    reaches the network or requires ``timm``.  Everything else -- the dataset
    contract, the schedule, the selection rule, the ``candidate_c`` block --
    is the file's own.
    """

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["model"].update(TINY_MODEL)
    raw["model"]["pretrained_encoder"] = False
    raw["model"]["fallback"] = True
    if window_mode is not None:
        raw["model"]["window_mode"] = window_mode
    return parse_unified_config(raw)


def _write_lineage_checkpoint(
    config_path: Path,
    destination: Path,
    *,
    window_mode: str | None = None,
) -> dict[str, Any]:
    """Save a frozen tiny checkpoint stamped with the real nested config tree."""

    unified = _tiny_unified_config(config_path, window_mode=window_mode)
    checkpoint_config = unified.to_dict()
    model = build_model_from_config(unified.to_legacy_model_config(), torch.device("cpu"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(destination, model, config=checkpoint_config, epoch=1)
    return checkpoint_config


def _populate_mnms_testing(root: Path, case_id: str = "A0S9V9_t00") -> Path:
    """Write one synthetic four-class M&Ms case under ``<root>/testing``."""

    split_dir = root / "testing"
    (split_dir / "volumes").mkdir(parents=True, exist_ok=True)
    (split_dir / "masks").mkdir(parents=True, exist_ok=True)
    shape = (IMAGE_SIZE, IMAGE_SIZE, 2)  # [H, W, Z]; depth_axis 2
    rng = np.random.default_rng(0)
    np.save(split_dir / "volumes" / f"{case_id}.npy", rng.normal(size=shape).astype(np.float32))
    mask = np.zeros(shape, dtype=np.uint8)
    # Raw M&Ms labels in disjoint blocks: 1 = LV, 2 = MYO, 3 = RV.
    mask[2:8, 2:8, :] = 1
    mask[10:16, 10:16, :] = 2
    mask[20:26, 20:26, :] = 3
    np.save(split_dir / "masks" / f"{case_id}.npy", mask)
    return root


def _protocol_config(data_root: Path) -> dict[str, Any]:
    """The shipped protocol config, resized and pointed at the synthetic cohort."""

    config = yaml.safe_load(PROTOCOL_CONFIG.read_text(encoding="utf-8"))
    config["image_size"] = IMAGE_SIZE
    config["model"].update(TINY_MODEL)
    config["external_test"]["data_root"] = str(data_root)
    return config


@pytest.fixture(scope="module")
def mnms_cohort(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _populate_mnms_testing(tmp_path_factory.mktemp("mnm_external") / "mnm")


def _assert_frozen_external_report(report: dict[str, Any], *, window_mode: str) -> None:
    """Assert the lineage claims an independent external evaluation must carry."""

    assert report["training_dataset"] == "acdc"
    assert report["evidence_class"] == "independent_external_evaluation"
    assert report["scientific_protocol_label"] == "acdc_frozen_to_mnms_external_v1"
    assert report["dataset"] == "mnms"
    assert report["split"] == "testing"
    assert report["window_mode"] == window_mode
    # tau is fixed by the protocol config; nothing on this path may derive it
    # from the target cohort.
    assert report["tau_accept"] == 0.0
    assert report["tau_accept_source"] == "config.audit.tau_accept"
    # The report serialises mapping keys as strings.
    assert report["label_mapping"] == {str(k): v for k, v in CANONICAL_MAPPING.items()}
    assert report["case_count"] == 1
    assert report["metric_space"] == "volume_resized"
    assert report["native_dice_available"] is False


@pytest.mark.parametrize(
    ("config_path", "declared_mode"),
    [
        pytest.param(STAGED_ACDC, None, id="staged_acdc_current"),
        pytest.param(JOINT_ACDC, None, id="joint_from_start_acdc_current"),
        pytest.param(JOINT_ACDC, "candidate_c", id="joint_from_start_acdc_candidate_c"),
    ],
)
def test_real_config_lineage_evaluates_externally(
    tmp_path: Path,
    mnms_cohort: Path,
    config_path: Path,
    declared_mode: str | None,
) -> None:
    """A checkpoint stamped with the real nested config tree evaluates.

    Covers the staged baseline and the joint-from-epoch-1 profile, the latter
    both at its shipped ``current`` mode and switched to ``candidate_c`` the way
    the native Candidate C runner does in its generated run config.
    """

    checkpoint_config = _write_lineage_checkpoint(
        config_path, tmp_path / "weights" / "frozen.pt", window_mode=declared_mode
    )
    # Precondition: the shape these tests exist to protect.
    assert checkpoint_config["dataset"]["name"] == "acdc"
    expected_mode = declared_mode or checkpoint_config["model"]["window_mode"]
    assert checkpoint_config["model"]["window_mode"] == expected_mode
    assert isinstance(checkpoint_config["model"]["candidate_c"], dict)

    report = run_external_evaluation(
        config=_protocol_config(mnms_cohort),
        checkpoint=tmp_path / "weights" / "frozen.pt",
        device="cpu",
        output=None,
    )
    _assert_frozen_external_report(report, window_mode=expected_mode)

    # The evaluation config declares neither the mode nor the solver settings,
    # so both must be inherited from the checkpoint and recorded verbatim.
    assert report["candidate_c_settings"] == checkpoint_config["model"]["candidate_c"]
    assert report["model_identity"]["window_mode"] == expected_mode


def test_cli_window_mode_conflicting_with_joint_checkpoint_is_refused(
    tmp_path: Path, mnms_cohort: Path
) -> None:
    """``--window-mode`` may not override the mode the checkpoint declares."""

    _write_lineage_checkpoint(
        JOINT_ACDC, tmp_path / "frozen.pt", window_mode="candidate_c"
    )
    with pytest.raises(ValueError, match="conflicts with checkpoint declared window_mode"):
        run_external_evaluation(
            config=_protocol_config(mnms_cohort),
            checkpoint=tmp_path / "frozen.pt",
            window_mode="current",
            device="cpu",
            output=None,
        )


def test_eval_config_candidate_c_conflicting_with_joint_checkpoint_is_refused(
    tmp_path: Path, mnms_cohort: Path
) -> None:
    """Evaluation-config solver settings may not contradict the frozen source."""

    checkpoint_config = _write_lineage_checkpoint(
        JOINT_ACDC, tmp_path / "frozen.pt", window_mode="candidate_c"
    )
    evaluation_config = _protocol_config(mnms_cohort)
    divergent = dict(checkpoint_config["model"]["candidate_c"])
    divergent["rho_feature_pixels"] = 2.0
    evaluation_config["model"]["candidate_c"] = divergent

    with pytest.raises(ValueError, match=r"candidate_c\.rho_feature_pixels"):
        run_external_evaluation(
            config=evaluation_config,
            checkpoint=tmp_path / "frozen.pt",
            device="cpu",
            output=None,
        )


def test_native_mnms_joint_checkpoint_is_refused_as_external_evidence(
    tmp_path: Path, mnms_cohort: Path
) -> None:
    """An M&Ms-trained joint-from-start checkpoint is never external evidence."""

    checkpoint_config = _write_lineage_checkpoint(JOINT_MNMS, tmp_path / "native.pt")
    assert checkpoint_config["dataset"]["name"] == "mnms"

    with pytest.raises(ValueError, match="cannot evaluate M&Ms-trained model"):
        run_external_evaluation(
            config=_protocol_config(mnms_cohort),
            checkpoint=tmp_path / "native.pt",
            device="cpu",
            output=None,
        )


def test_committed_best_alias_binds_and_a_stale_alias_is_refused(
    tmp_path: Path, mnms_cohort: Path
) -> None:
    """The external path honours the Wave 4 best-alias commit protocol.

    The alias is produced here the way a real run produces it -- an immutable
    ``selected_best/`` snapshot, a hash-verified ``best_reference`` recorded in
    ``last.pt``, then ``publish_best_alias`` -- rather than by copying a file
    into place, so this exercises ``_require_committed_best_alias`` for real.
    """

    weights = tmp_path / "weights"
    snapshot = weights / "selected_best" / "epoch_1_lineage.pt"
    checkpoint_config = _write_lineage_checkpoint(
        JOINT_ACDC, snapshot, window_mode="candidate_c"
    )
    reference = make_best_reference(snapshot, weights, epoch=1, metric=0.1)

    unrelated = _tiny_unified_config(JOINT_ACDC, window_mode="candidate_c")
    other_model = build_model_from_config(
        unrelated.to_legacy_model_config(), torch.device("cpu")
    )
    save_checkpoint(
        weights / "last.pt",
        other_model,
        config=checkpoint_config,
        epoch=1,
        extra={"best_reference": reference},
    )
    alias = publish_best_alias(reference, weights)

    report = run_external_evaluation(
        config=_protocol_config(mnms_cohort),
        checkpoint=alias,
        device="cpu",
        output=None,
    )
    _assert_frozen_external_report(report, window_mode="candidate_c")
    # The bound bytes are the committed selection, not merely a file named
    # best.pt sitting in the directory.
    assert report["checkpoint_binding"]["checkpoint_sha256"] == reference["sha256"]

    # Interrupted alias publication: last.pt commits one snapshot, best.pt
    # holds other bytes.  Evaluation must refuse rather than repair.
    alias.write_bytes((weights / "last.pt").read_bytes())
    with pytest.raises(ValueError, match="the public best alias has sha256"):
        run_external_evaluation(
            config=_protocol_config(mnms_cohort),
            checkpoint=alias,
            device="cpu",
            output=None,
        )


def test_protocol_config_declares_the_documented_mapping_and_depth_axis() -> None:
    """The shipped protocol config pins the M&Ms input format explicitly.

    ``raw_to_acdc`` and ``depth_axis`` are evaluation *input-format*
    declarations, not tuning knobs, and the evaluator honours whatever the
    config declares.  What makes the shipped protocol correct is that this file
    declares the documented values, so a silent edit here is a review event.
    """

    config = yaml.safe_load(PROTOCOL_CONFIG.read_text(encoding="utf-8"))
    declared = {int(k): int(v) for k, v in config["external_test"]["raw_to_acdc"].items()}
    assert declared == CANONICAL_MAPPING
    assert int(config["depth_axis"]) == 2
    assert str(config["external_test"]["split"]) == "testing"
    assert float(config["audit"]["tau_accept"]) == 0.0


def test_external_evaluation_never_mutates_the_frozen_checkpoint(
    tmp_path: Path, mnms_cohort: Path
) -> None:
    """Evaluation is read-only with respect to the artifact it measures."""

    checkpoint = tmp_path / "frozen.pt"
    _write_lineage_checkpoint(JOINT_ACDC, checkpoint)
    before = checkpoint.read_bytes()
    evaluation_config = copy.deepcopy(_protocol_config(mnms_cohort))

    run_external_evaluation(
        config=evaluation_config, checkpoint=checkpoint, device="cpu", output=None
    )

    assert checkpoint.read_bytes() == before
