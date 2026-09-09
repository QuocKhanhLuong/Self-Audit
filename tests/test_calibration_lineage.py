"""Regression tests for Wave 3.2: strict calibration lineage at the consumer.

The failure guarded against here is a calibrated ``tau_accept`` being applied
to something it was never measured on -- different weights behind the same
filename, a different preprocessing recipe, a different metric contract or
decision margin, a different ``t_max``, or a substituted cohort.  A calibration
artifact is not evidence about a run; it becomes evidence only once its lineage
has been checked against what that run actually constructed.

Every expectation in these tests is derived from live objects -- a bound
checkpoint, a constructed dataset, the protocol's declared semantics -- and
never from the artifact under test.  A test that read its expectation out of
the artifact would pass for any file at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Imported under the ``self_audit`` package name, which is what
# ``scripts/audit_checkpoint.py`` imports.  Reaching the same modules through
# the ``src.self_audit`` alias would create a second set of exception classes,
# and the CLI's refusals would then not be catchable here.
from self_audit.data.common import VolumeRecord
from self_audit.evaluation.calibration_lineage import (
    CALIBRATION_LINEAGE_SCHEMA_VERSION,
    COHORT_ROLE_CALIBRATION,
    COHORT_ROLE_INDEPENDENT_EVALUATION,
    IDENTITY_FIELDS,
    CalibrationLineageError,
    CohortPolicy,
    CohortRoleError,
    LineageMismatchError,
    MissingLineageError,
    REQUIRED_PRESENT,
    build_expected_lineage,
    inspect_legacy_calibration,
    runtime_membership_signature,
    validate_lineage_completeness,
    verify_calibration_lineage,
)
from self_audit.evaluation.threshold import (
    CALIBRATION_SCHEMA_VERSION,
    LEGACY_CALIBRATION_SCHEMA_VERSIONS,
    load_calibration,
    save_calibration,
)
from self_audit.models.self_audit_net import SelfAuditNet
from self_audit.provenance import git_source_provenance, state_digest
from self_audit.training._utils import bind_evaluation_checkpoint, save_checkpoint

TINY_MODEL_CONFIG: dict[str, Any] = {
    "encoder_name": "convnext_tiny",
    "shared_channels": 16,
    "num_classes": 4,
    "window_k": 4,
    "max_turns": 2,
}

METRIC_CONTRACT = "foreground_dice_exclude_v1"
METRIC_SPACE = "slice_proxy"
NEUTRAL_MARGIN = 0.005
T_MAX = 2


def _tiny_net(seed: int) -> SelfAuditNet:
    torch.manual_seed(seed)
    return SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=2,
    ).eval()


class _RecordDataset(Dataset):
    """A dataset with the record metadata the split/cohort descriptors read.

    Patient identity is explicit so a genuinely disjoint second cohort can be
    constructed, rather than asserted.
    """

    def __init__(self, patients: tuple[str, ...], *, size: int = 16) -> None:
        torch.manual_seed(len(patients) + size)
        self.records = [
            VolumeRecord(
                case_id=f"{patient}_ED",
                patient_id=patient,
                image_path=Path(f"/fixture/{patient}_image.npy"),
                mask_path=Path(f"/fixture/{patient}_mask.npy"),
                split="val",
            )
            for patient in patients
        ]
        self.image_size = 32
        self.depth_axis = 2
        self.foreground_only = False
        self.augment = False
        self.lower_percentile = 0.5
        self.upper_percentile = 99.5
        self.transform = None
        self.images = torch.randn(len(self.records), 3, size, size)
        self.masks = torch.randint(0, 4, (len(self.records), size, size))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "image": self.images[index],
            "mask": self.masks[index],
            "case_id": record.case_id,
            "patient_id": record.patient_id,
        }


CALIBRATION_PATIENTS = ("patient001", "patient002", "patient003")
DISJOINT_PATIENTS = ("patient101", "patient102", "patient103")


def _loader(patients: tuple[str, ...]) -> DataLoader:
    return DataLoader(_RecordDataset(patients), batch_size=2, shuffle=False)


def _bind(path: Path, model: nn.Module) -> Any:
    return bind_evaluation_checkpoint(
        model,
        [("checkpoint", path)],
        map_location="cpu",
        config={"model": TINY_MODEL_CONFIG},
    )


def _expected(binding: Any, loader: DataLoader, *, split_name: str = "val", **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "binding": binding,
        "loader": loader,
        "split_name": split_name,
        "metric_contract": METRIC_CONTRACT,
        "metric_space": METRIC_SPACE,
        "neutral_margin": NEUTRAL_MARGIN,
        "t_max": T_MAX,
        "batch_size": loader.batch_size,
    }
    kwargs.update(overrides)
    return build_expected_lineage(**kwargs)


def _write_artifact(path: Path, lineage: Any, *, tau: float = 0.0123456789, **overrides: Any) -> Path:
    payload: dict[str, Any] = {
        "tau_accept": tau,
        "neutral_margin": NEUTRAL_MARGIN,
        "source_split": "val",
        "checkpoint_path": None,
        "t_max": T_MAX,
        "threshold_grid": {"min": -0.02, "max": 0.02, "steps": 5},
        "selected_row": {"tau_accept": tau, "final_macro_dice": 0.5, "metric_space": METRIC_SPACE},
        "metric_space": METRIC_SPACE,
        "metric_contract": METRIC_CONTRACT,
        "lineage": lineage,
    }
    payload.update(overrides)
    save_calibration(path, **payload)
    return path


@pytest.fixture()
def producer(tmp_path: Path) -> dict[str, Any]:
    """One matched producer: bound checkpoint, cohort, artifact on disk."""

    checkpoint = tmp_path / "phase_c_best.pt"
    model = _tiny_net(101)
    save_checkpoint(checkpoint, model, epoch=3, config={"model": TINY_MODEL_CONFIG})

    live = _tiny_net(303)
    binding = _bind(checkpoint, live)
    loader = _loader(CALIBRATION_PATIENTS)
    lineage = _expected(binding, loader)
    artifact_path = _write_artifact(tmp_path / "calibration.json", lineage)
    return {
        "tmp_path": tmp_path,
        "checkpoint": checkpoint,
        "model": live,
        "binding": binding,
        "loader": loader,
        "lineage": lineage,
        "artifact_path": artifact_path,
        "artifact": load_calibration(artifact_path),
    }


# ---------------------------------------------------------------------------
# 1. Matched producer -> artifact -> consumer roundtrip
# ---------------------------------------------------------------------------


def test_matched_producer_artifact_consumer_roundtrip(producer: dict[str, Any]) -> None:
    """A consumer that rebuilds the same runtime accepts the artifact."""

    artifact = producer["artifact"]
    assert artifact["schema_version"] == CALIBRATION_SCHEMA_VERSION
    assert artifact["lineage"] is not None

    # The consumer re-derives its expectation from its own objects: a freshly
    # constructed model, a fresh bind of the checkpoint file, a fresh dataset.
    consumer_model = _tiny_net(777)
    consumer_binding = _bind(producer["checkpoint"], consumer_model)
    consumer_loader = _loader(CALIBRATION_PATIENTS)
    expected = _expected(consumer_binding, consumer_loader)

    record = verify_calibration_lineage(
        artifact, expected, cohort_policy=CohortPolicy(role=COHORT_ROLE_CALIBRATION)
    )
    assert record["verified"] is True
    assert record["calibration_lineage_schema_version"] == CALIBRATION_LINEAGE_SCHEMA_VERSION
    assert record["cohort"]["role"] == COHORT_ROLE_CALIBRATION
    assert set(record["compared_fields"]) == set(IDENTITY_FIELDS)
    # The calibrating revision and the consuming revision are recorded apart.
    assert "calibration_code" in record and "consuming_code" in record


def test_expectation_survives_a_json_roundtrip(producer: dict[str, Any]) -> None:
    """The artifact is compared as it was persisted, not as it was in memory."""

    on_disk = json.loads(producer["artifact_path"].read_text())
    assert on_disk["lineage"] is not None
    verify_calibration_lineage(on_disk, producer["lineage"])


# ---------------------------------------------------------------------------
# 2. Every identity field is load-bearing
# ---------------------------------------------------------------------------


def _mutate(lineage: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    record = json.loads(json.dumps(lineage, default=str))
    node = record
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    return record


def _drop(lineage: dict[str, Any], path: str) -> dict[str, Any]:
    record = json.loads(json.dumps(lineage, default=str))
    node = record
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node.pop(parts[-1])
    return record


_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
def test_each_identity_field_mutation_hard_fails(producer: dict[str, Any], field: str) -> None:
    """Change any one identity field in the artifact and verification refuses."""

    expected = producer["lineage"]
    current = expected
    for part in field.split("."):
        current = current[part]
    if isinstance(current, bool):
        replacement: Any = not current
    elif isinstance(current, int):
        replacement = current + 1
    elif isinstance(current, float):
        replacement = current + 1.0
    elif isinstance(current, str) and _HEX64.match(current):
        # A well-formed but different digest: the point is the comparison, not
        # the format check that a garbled digest would trip first.
        replacement = ("0" if not current.startswith("0") else "1") * 64
    elif isinstance(current, str):
        replacement = current + "_tampered"
    elif isinstance(current, list):
        replacement = current + ["tampered"]
    elif isinstance(current, dict):
        replacement = dict(current)
        replacement["tampered"] = True
    else:  # pragma: no cover - IDENTITY_FIELDS carries no other types
        pytest.fail(f"unhandled identity field type for {field}: {type(current).__name__}")

    tampered = _mutate(expected, field, replacement)
    if field == "lineage_schema_version":
        # An unrecognised lineage schema is refused before any comparison: this
        # build cannot claim to know what its fields mean.
        with pytest.raises(MissingLineageError, match="lineage_schema_version"):
            verify_calibration_lineage(tampered, expected)
        return
    with pytest.raises(LineageMismatchError, match="different lineage"):
        verify_calibration_lineage(tampered, expected)


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
def test_each_identity_field_absence_hard_fails(producer: dict[str, Any], field: str) -> None:
    """A missing field is a refusal, not an absence of objection."""

    expected = producer["lineage"]
    pattern = (
        "lineage_schema_version"
        if field == "lineage_schema_version"
        else "missing required lineage field"
    )
    with pytest.raises(CalibrationLineageError, match=pattern):
        verify_calibration_lineage(_drop(expected, field), expected)
    with pytest.raises(CalibrationLineageError, match=pattern):
        verify_calibration_lineage(_mutate(expected, field, None), expected)


def test_no_lineage_block_is_refused_and_only_readable_as_legacy(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path / "no_lineage.json", None)
    payload = load_calibration(path)
    assert payload["lineage"] is None

    with pytest.raises(MissingLineageError, match="carries no lineage block"):
        verify_calibration_lineage(payload, {"lineage_schema_version": 1})

    inspected = inspect_legacy_calibration(path)
    assert inspected["verified"] is False
    assert inspected["usable_for_calibrated_evaluation"] is False
    assert inspected["lineage_present"] is False
    assert "tau_accept" not in inspected
    assert inspected["declared_tau_accept"] == pytest.approx(0.0123456789)


def test_v1_artifact_cannot_be_loaded_for_calibrated_use(tmp_path: Path, producer: dict[str, Any]) -> None:
    """The pre-lineage schema is legacy: inspect only, never load."""

    path = tmp_path / "legacy_v1.json"
    raw = json.loads(producer["artifact_path"].read_text())
    raw["schema_version"] = LEGACY_CALIBRATION_SCHEMA_VERSIONS[0]
    raw.pop("lineage")
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="predates the required lineage block"):
        load_calibration(path)

    inspected = inspect_legacy_calibration(path)
    assert inspected["verified"] is False
    assert inspected["usable_for_calibrated_evaluation"] is False
    assert inspected["schema_version"] == LEGACY_CALIBRATION_SCHEMA_VERSIONS[0]


def test_incomplete_lineage_is_refused_at_write_time(tmp_path: Path, producer: dict[str, Any]) -> None:
    """A half-filled lineage never reaches disk to fail later."""

    with pytest.raises(MissingLineageError):
        _write_artifact(
            tmp_path / "incomplete.json",
            _drop(producer["lineage"], "preprocessing.signature"),
        )


# ---------------------------------------------------------------------------
# 3. The specific substitutions this gate exists for
# ---------------------------------------------------------------------------


def test_different_weights_under_the_same_filename_fail(tmp_path: Path) -> None:
    """Overwriting the checkpoint in place does not sneak past the artifact."""

    checkpoint = tmp_path / "phase_c_best.pt"
    save_checkpoint(checkpoint, _tiny_net(101), epoch=3, config={"model": TINY_MODEL_CONFIG})
    calibrated_binding = _bind(checkpoint, _tiny_net(303))
    loader = _loader(CALIBRATION_PATIENTS)
    artifact_lineage = _expected(calibrated_binding, loader)

    # Same path, different weights.
    replacement = _tiny_net(202)
    assert state_digest(replacement) != state_digest(_tiny_net(101))
    save_checkpoint(checkpoint, replacement, epoch=9, config={"model": TINY_MODEL_CONFIG})
    later_binding = _bind(checkpoint, _tiny_net(404))
    assert later_binding.path == calibrated_binding.path

    expected = _expected(later_binding, _loader(CALIBRATION_PATIENTS))
    with pytest.raises(LineageMismatchError) as excinfo:
        verify_calibration_lineage(artifact_lineage, expected)
    message = str(excinfo.value)
    assert "checkpoint.state_digest" in message
    assert "checkpoint.checkpoint_sha256" in message


def test_same_weights_at_a_different_path_still_verify(tmp_path: Path) -> None:
    """Location is not identity; the digests are."""

    first = tmp_path / "a" / "phase_c_best.pt"
    second = tmp_path / "b" / "renamed.pt"
    first.parent.mkdir()
    second.parent.mkdir()
    model = _tiny_net(101)
    save_checkpoint(first, model, epoch=3, config={"model": TINY_MODEL_CONFIG})
    # A byte copy, not a second save: ``save_checkpoint`` stamps RNG state, so
    # two writes of one model are two different files.
    shutil.copy2(first, second)

    artifact_lineage = _expected(_bind(first, _tiny_net(9)), _loader(CALIBRATION_PATIENTS))
    expected = _expected(_bind(second, _tiny_net(9)), _loader(CALIBRATION_PATIENTS))
    assert artifact_lineage["checkpoint"]["checkpoint_path"] != expected["checkpoint"]["checkpoint_path"]
    assert verify_calibration_lineage(artifact_lineage, expected)["verified"] is True


@pytest.mark.parametrize(
    "overrides, expect_in_message",
    [
        (
            {
                "metric_contract": "foreground_dice_volume_resized_v1",
                "metric_space": "volume_resized",
            },
            "semantics.metric_contract",
        ),
        ({"t_max": 3}, "semantics.t_max"),
    ],
)
def test_semantic_protocol_changes_fail(
    producer: dict[str, Any], overrides: dict[str, Any], expect_in_message: str
) -> None:
    """A different contract or turn cap is a different tau.

    ``metric_space`` and ``neutral_margin`` cannot be varied independently:
    ``build_lineage`` derives them from the contract and refuses a caller value
    that contradicts it, so they move with the contract.  Their comparison is
    covered field-by-field in ``test_each_identity_field_mutation_hard_fails``.
    """

    expected = _expected(producer["binding"], producer["loader"], **overrides)
    with pytest.raises(LineageMismatchError) as excinfo:
        verify_calibration_lineage(producer["artifact"], expected)
    assert expect_in_message in str(excinfo.value)


def test_preprocessing_recipe_change_fails(producer: dict[str, Any]) -> None:
    """A resize or clipping change invalidates the calibration."""

    other = _RecordDataset(CALIBRATION_PATIENTS)
    other.image_size = 128
    expected = _expected(producer["binding"], DataLoader(other, batch_size=2, shuffle=False))
    with pytest.raises(LineageMismatchError, match="preprocessing.signature"):
        verify_calibration_lineage(producer["artifact"], expected)


def test_model_config_and_backend_change_fails(tmp_path: Path, producer: dict[str, Any]) -> None:
    """A differently-shaped model cannot inherit another model's threshold."""

    torch.manual_seed(55)
    wider = SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=24,
        window_k=4,
        max_turns=2,
    ).eval()
    checkpoint = tmp_path / "wider.pt"
    save_checkpoint(checkpoint, wider, epoch=1, config=None)
    binding = bind_evaluation_checkpoint(wider, [("checkpoint", checkpoint)], map_location="cpu")
    expected = _expected(binding, producer["loader"])
    with pytest.raises(LineageMismatchError) as excinfo:
        verify_calibration_lineage(producer["artifact"], expected)
    assert "checkpoint.model_identity.shared_channels" in str(excinfo.value)


def test_class_mapping_is_compared(producer: dict[str, Any]) -> None:
    """Reordering the semantic classes changes what every Dice meant."""

    swapped = list(producer["lineage"]["semantics"]["class_names"])
    swapped[1], swapped[3] = swapped[3], swapped[1]
    tampered = _mutate(producer["lineage"], "semantics.class_names", swapped)
    with pytest.raises(LineageMismatchError, match="semantics.class_names"):
        verify_calibration_lineage(tampered, producer["lineage"])


def test_an_artifact_is_never_its_own_witness(producer: dict[str, Any]) -> None:
    """Verifying an artifact against itself is not a verification.

    A tampered artifact compared with an expectation lifted out of that same
    artifact passes trivially -- which is exactly why the expectation is built
    from the runtime.  This test pins the distinction rather than the loophole.
    """

    tampered = _mutate(producer["lineage"], "checkpoint.state_digest", "0" * 64)
    assert verify_calibration_lineage(tampered, tampered)["verified"] is True
    with pytest.raises(LineageMismatchError):
        verify_calibration_lineage(tampered, producer["lineage"])


# ---------------------------------------------------------------------------
# 4. Cohort roles
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3b. The four bypasses reproduced against the draft
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", REQUIRED_PRESENT)
def test_recorded_but_not_compared_fields_may_not_be_deleted(
    producer: dict[str, Any], field: str
) -> None:
    """"Not compared" must not mean "may be removed".

    Deleting ``evaluation_code`` or the whole ``cohort`` block used to leave a
    record that verified clean, because nothing checked for their presence.
    """

    expected = producer["lineage"]
    with pytest.raises(MissingLineageError, match="missing required lineage field"):
        verify_calibration_lineage(_drop(expected, field), expected)
    with pytest.raises(MissingLineageError, match="missing required lineage field"):
        verify_calibration_lineage(_mutate(expected, field, None), expected)


def test_a_boolean_is_not_a_schema_version(producer: dict[str, Any]) -> None:
    """``True == 1``, so an equality check alone accepts a bool as version 1."""

    tampered = _mutate(producer["lineage"], "lineage_schema_version", True)
    with pytest.raises(MissingLineageError, match="integer version"):
        verify_calibration_lineage(tampered, producer["lineage"])


@pytest.mark.parametrize(
    "field, value",
    [
        ("semantics.t_max", True),
        ("semantics.num_classes", True),
        ("split.num_records", True),
        ("checkpoint.model_identity.window_k", False),
        ("checkpoint.state_digest", "not-a-digest"),
        ("preprocessing.signature", "ABC"),
        ("split.membership_signature", 1),
        ("semantics.class_names", "Background,RV,MYO,LV"),
        ("semantics.metric_contract_definition", {}),
    ],
)
def test_malformed_field_types_are_refused(
    producer: dict[str, Any], field: str, value: Any
) -> None:
    """A present value of the wrong kind is not a present value."""

    tampered = _mutate(producer["lineage"], field, value)
    with pytest.raises(MissingLineageError, match="malformed lineage field"):
        verify_calibration_lineage(tampered, producer["lineage"])


@pytest.mark.parametrize(
    "field, value, expected_message",
    [
        ("cohort.covers_full_split", False, "covers_full_split"),
        ("cohort.sampling.iterates_every_record", False, "iterates_every_record"),
        ("cohort.max_batches", 1, "truncated at max_batches"),
        ("cohort.sampling.subset_indices_signature", "a" * 64, "iterated a subset"),
        ("cohort.iterated_slices", 1, "iterated_slices"),
        ("cohort.coverage", "content_hash", "cohort.coverage"),
        ("cohort.membership_signature", "b" * 64, "disagrees with split.membership_signature"),
    ],
)
def test_an_altered_calibration_cohort_is_refused(
    producer: dict[str, Any], field: str, value: Any, expected_message: str
) -> None:
    """The same split name with a different measured population is refused.

    Truncating with ``max_batches``, iterating a ``Subset``, or reporting a
    partial extent all leave the membership signature intact, so the signature
    alone was never enough.
    """

    tampered = _mutate(producer["lineage"], field, value)
    with pytest.raises(CohortRoleError, match=expected_message):
        verify_calibration_lineage(tampered, producer["lineage"])


def test_a_same_signature_cohort_with_a_different_definition_is_refused(
    producer: dict[str, Any],
) -> None:
    """A signature says which case ids existed, not how the cohort was built."""

    tampered = _mutate(producer["lineage"], "cohort.available_slices", 999)
    with pytest.raises(CohortRoleError, match="not the same cohort"):
        verify_calibration_lineage(producer["lineage"], tampered)


# ---------------------------------------------------------------------------
# 3c. Source identity of the measuring run
# ---------------------------------------------------------------------------


def test_source_identity_is_the_shared_producer_signature(producer: dict[str, Any]) -> None:
    """The lineage carries the producer's scoped signature, not a second one."""

    recorded = producer["lineage"]["evaluation_code"]
    current = git_source_provenance()
    assert recorded["source_content_signature_known"] is True
    assert recorded["source_content_signature"] == current["source_content_signature"]
    assert recorded["source_signature_version"] == current["source_signature_version"]
    assert _HEX64.match(recorded["source_content_signature"])
    # Scoped, not the whole workspace: reports and data are excluded.
    scope = recorded["source_signature_scope"]
    assert "reports" in scope["excluded_dirs"]
    assert "tests" in scope["excluded_dirs"]


def test_changed_scoped_source_invalidates_the_artifact(producer: dict[str, Any]) -> None:
    """A functional source change expires the calibration; the git SHA does not."""

    tampered = _mutate(
        producer["lineage"], "evaluation_code.source_content_signature", "c" * 64
    )
    with pytest.raises(
        LineageMismatchError,
        match=r"different lineage than this run.*evaluation_code\.source_content_signature",
    ):
        verify_calibration_lineage(tampered, producer["lineage"])


def test_a_moved_git_revision_alone_does_not_expire_a_calibration(
    producer: dict[str, Any],
) -> None:
    """A report-only commit changes the SHA and nothing else; that must pass."""

    report_only_commit = _mutate(producer["lineage"], "evaluation_code.git_sha", "f" * 40)
    report_only_commit = _mutate(report_only_commit, "evaluation_code.dirty", True)
    assert verify_calibration_lineage(report_only_commit, producer["lineage"])["verified"] is True


@pytest.mark.parametrize("side", ["artifact", "runtime"])
def test_an_unknown_source_signature_fails_closed(producer: dict[str, Any], side: str) -> None:
    """An unidentified source is never treated as a matching source."""

    unknown = _mutate(
        producer["lineage"], "evaluation_code.source_content_signature_known", False
    )
    artifact, expected = (
        (unknown, producer["lineage"]) if side == "artifact" else (producer["lineage"], unknown)
    )
    with pytest.raises(LineageMismatchError, match="never a match"):
        verify_calibration_lineage(artifact, expected)


def test_a_non_digest_source_signature_is_refused(producer: dict[str, Any]) -> None:
    tampered = _mutate(producer["lineage"], "evaluation_code.source_content_signature", "unknown")
    with pytest.raises(MissingLineageError, match="malformed lineage field"):
        verify_calibration_lineage(tampered, producer["lineage"])


def test_a_different_checkpoint_producing_source_is_refused(producer: dict[str, Any]) -> None:
    """The producing source is a separate fact and is compared when present."""

    tampered = _mutate(
        producer["lineage"],
        "checkpoint.producer.producer_source_content_signature",
        "d" * 64,
    )
    with pytest.raises(LineageMismatchError, match="different checkpoint-producing source"):
        verify_calibration_lineage(tampered, producer["lineage"])


def test_producer_provenance_is_not_claimed_complete_when_unknown(
    producer: dict[str, Any],
) -> None:
    """A legacy checkpoint's absent producing source stays visibly unknown."""

    record = verify_calibration_lineage(producer["artifact"], producer["lineage"])
    assert record["producer_provenance_complete"] is True
    assert record["source_identity"]["checkpoint_producing_source_known"] is True

    legacy = _mutate(producer["lineage"], "checkpoint.producer.producer_git_sha", "unknown")
    legacy = _mutate(legacy, "checkpoint.producer.producer_recorded", False)
    legacy = _mutate(legacy, "checkpoint.producer.producer_source_content_signature", None)
    legacy = _mutate(legacy, "checkpoint.producer.producer_source_content_signature_known", False)
    result = verify_calibration_lineage(legacy, legacy)
    assert result["checkpoint_producing_git_sha_unknown"] is True
    assert result["producer_provenance_complete"] is False
    assert result["source_identity"]["checkpoint_producing_source_known"] is False


# ---------------------------------------------------------------------------
# 4. Cohort roles
# ---------------------------------------------------------------------------


def test_calibration_role_requires_the_calibration_cohort(producer: dict[str, Any]) -> None:
    expected = _expected(producer["binding"], _loader(DISJOINT_PATIENTS))
    with pytest.raises(CohortRoleError, match="cohort role 'calibration'"):
        verify_calibration_lineage(
            producer["artifact"], expected, cohort_policy=CohortPolicy(role=COHORT_ROLE_CALIBRATION)
        )


def test_authorized_disjoint_evaluation_cohort_is_accepted(producer: dict[str, Any]) -> None:
    """The positive control: a different cohort the protocol actually named."""

    eval_loader = _loader(DISJOINT_PATIENTS)
    authorized = runtime_membership_signature(eval_loader, split_name="val")
    expected = _expected(producer["binding"], eval_loader)

    record = verify_calibration_lineage(
        producer["artifact"],
        expected,
        cohort_policy=CohortPolicy(
            role=COHORT_ROLE_INDEPENDENT_EVALUATION,
            authorized_membership_signature=authorized,
            authorized_split_name="val",
        ),
    )
    assert record["verified"] is True
    assert record["cohort"]["patients_disjoint"] is True
    assert record["cohort"]["calibration_patient_count"] == len(CALIBRATION_PATIENTS)
    assert record["cohort"]["evaluation_patient_count"] == len(DISJOINT_PATIENTS)


def test_arbitrary_replacement_cohort_is_rejected(producer: dict[str, Any]) -> None:
    """Disjoint is not enough; the protocol has to have named it."""

    authorized_loader = _loader(DISJOINT_PATIENTS)
    authorized = runtime_membership_signature(authorized_loader, split_name="val")

    substituted = _loader(("patient901", "patient902"))
    expected = _expected(producer["binding"], substituted)
    with pytest.raises(CohortRoleError, match="not the cohort the protocol authorised"):
        verify_calibration_lineage(
            producer["artifact"],
            expected,
            cohort_policy=CohortPolicy(
                role=COHORT_ROLE_INDEPENDENT_EVALUATION,
                authorized_membership_signature=authorized,
            ),
        )


def test_independent_role_without_an_authorization_is_rejected() -> None:
    with pytest.raises(CohortRoleError, match="requires the protocol to declare"):
        CohortPolicy(role=COHORT_ROLE_INDEPENDENT_EVALUATION)


def test_overlapping_patients_are_rejected_even_when_authorized(producer: dict[str, Any]) -> None:
    overlapping = _loader(("patient002", "patient777"))
    authorized = runtime_membership_signature(overlapping, split_name="val")
    expected = _expected(producer["binding"], overlapping)
    with pytest.raises(CohortRoleError, match="shares patient"):
        verify_calibration_lineage(
            producer["artifact"],
            expected,
            cohort_policy=CohortPolicy(
                role=COHORT_ROLE_INDEPENDENT_EVALUATION,
                authorized_membership_signature=authorized,
            ),
        )


def test_the_calibration_cohort_cannot_be_relabelled_independent(producer: dict[str, Any]) -> None:
    """The selection split may not be presented as independent evidence."""

    same = _loader(CALIBRATION_PATIENTS)
    authorized = runtime_membership_signature(same, split_name="val")
    expected = _expected(producer["binding"], same)
    with pytest.raises(CohortRoleError, match="is the calibration cohort"):
        verify_calibration_lineage(
            producer["artifact"],
            expected,
            cohort_policy=CohortPolicy(
                role=COHORT_ROLE_INDEPENDENT_EVALUATION,
                authorized_membership_signature=authorized,
            ),
        )


def test_empty_patient_identity_is_not_proof_of_disjointness(producer: dict[str, Any]) -> None:
    stripped = _mutate(producer["lineage"], "split.patient_ids", [])
    eval_loader = _loader(DISJOINT_PATIENTS)
    authorized = runtime_membership_signature(eval_loader, split_name="val")
    expected = _expected(producer["binding"], eval_loader)
    with pytest.raises(CohortRoleError, match="not proof of disjointness"):
        verify_calibration_lineage(
            stripped,
            expected,
            cohort_policy=CohortPolicy(
                role=COHORT_ROLE_INDEPENDENT_EVALUATION,
                authorized_membership_signature=authorized,
            ),
        )


# ---------------------------------------------------------------------------
# 5. The diagnostic CLI: override cannot bypass, emission is gated
# ---------------------------------------------------------------------------


def _cli() -> Any:
    import scripts.audit_checkpoint as module

    return module


def test_cli_resolution_verifies_and_stamps_the_artifact(producer: dict[str, Any]) -> None:
    cli = _cli()
    resolution = cli.resolve_tau_accept(
        cli_tau=None,
        calibration_path=str(producer["artifact_path"]),
        config_audit={"tau_accept": 0.0},
        neutral_margin=NEUTRAL_MARGIN,
        expected_lineage=producer["lineage"],
    )
    assert resolution["tau_accept_source"] == cli.TAU_SOURCE_CALIBRATION
    assert resolution["lineage_verified"] is True
    assert resolution["lineage_verification"]["verified"] is True
    cli.assert_calibration_lineage_verified(resolution)


def test_cli_tau_override_cannot_bypass_the_lineage_check(producer: dict[str, Any]) -> None:
    """An override changes the number reported, not whether the artifact is valid."""

    cli = _cli()
    mismatched_loader = _loader(DISJOINT_PATIENTS)
    expected = _expected(producer["binding"], mismatched_loader)
    with pytest.raises(CohortRoleError):
        cli.resolve_tau_accept(
            cli_tau=0.5,
            calibration_path=str(producer["artifact_path"]),
            config_audit={},
            neutral_margin=NEUTRAL_MARGIN,
            allow_tau_override=True,
            expected_lineage=expected,
        )


def test_cli_override_still_fails_on_a_tampered_artifact(tmp_path: Path, producer: dict[str, Any]) -> None:
    cli = _cli()
    tampered_path = _write_artifact(
        tmp_path / "tampered.json",
        _mutate(producer["lineage"], "checkpoint.state_digest", "0" * 64),
    )
    with pytest.raises(LineageMismatchError):
        cli.resolve_tau_accept(
            cli_tau=0.5,
            calibration_path=str(tampered_path),
            config_audit={},
            neutral_margin=NEUTRAL_MARGIN,
            allow_tau_override=True,
            expected_lineage=producer["lineage"],
        )


def test_report_emission_is_blocked_without_a_verification(producer: dict[str, Any]) -> None:
    """The resolver refuses a consulted artifact with no runtime expectation.

    It fails closed rather than returning a usable tau stamped unverified: a
    caller that forgot to build an expectation must not get a number that looks
    calibrated.
    """

    cli = _cli()
    with pytest.raises(cli.CalibrationLineageError, match="without a runtime-derived"):
        cli.resolve_tau_accept(
            cli_tau=None,
            calibration_path=str(producer["artifact_path"]),
            config_audit={},
            neutral_margin=NEUTRAL_MARGIN,
        )
    # The same refusal applies under an explicit override.
    with pytest.raises(cli.CalibrationLineageError, match="without a runtime-derived"):
        cli.resolve_tau_accept(
            cli_tau=0.5,
            calibration_path=str(producer["artifact_path"]),
            config_audit={},
            neutral_margin=NEUTRAL_MARGIN,
            allow_tau_override=True,
        )
    # The emission gate stays as a second, independent guard.
    with pytest.raises(cli.CalibrationLineageError, match="never verified against this run"):
        cli.assert_calibration_lineage_verified(
            {"calibration_path": "x", "lineage_verified": False}
        )


def test_no_artifact_means_no_lineage_gate(producer: dict[str, Any]) -> None:
    """Config and default taus are unaffected; nothing new is demanded of them."""

    cli = _cli()
    resolution = cli.resolve_tau_accept(
        cli_tau=None, calibration_path=None, config_audit={"tau_accept": 0.25}, neutral_margin=NEUTRAL_MARGIN
    )
    assert resolution["tau_accept_source"] == cli.TAU_SOURCE_CONFIG
    assert resolution["lineage_verified"] is None
    cli.assert_calibration_lineage_verified(resolution)


# ---------------------------------------------------------------------------
# 6. The unified runner's calibration branch, best != last
# ---------------------------------------------------------------------------


class _SyntheticDataset(Dataset):
    def __init__(self, count: int = 4, size: int = 32) -> None:
        torch.manual_seed(11)
        self.images = torch.randn(count, 3, size, size)
        self.masks = torch.randint(0, 4, (count, size, size))
        self.case_ids = [f"case_{index // 2:03d}" for index in range(count)]

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self.images[index],
            "mask": self.masks[index],
            "case_id": self.case_ids[index],
        }


def test_runner_calibration_branch_lineage_names_best_not_last(tmp_path: Path) -> None:
    """The runner's artifact lineage must name the bound best weights.

    ``run_post_training_calibration`` is owned by W3.1; this test consumes it
    read-only.  It asserts the property W3.2 depends on -- the lineage carried
    by the artifact is the *bound* checkpoint's, never the live last-epoch
    weights -- and records which channel currently carries it.
    """

    from scripts.train_self_audit_legacy import run_post_training_calibration

    output_dir = tmp_path / "weights"
    report_dir = tmp_path / "reports"
    output_dir.mkdir()
    report_dir.mkdir()

    best = _tiny_net(101)
    last = _tiny_net(202)
    best_digest = state_digest(best)
    last_digest = state_digest(last)
    assert best_digest != last_digest, "fixture must have best != last"
    save_checkpoint(output_dir / "phase_c_best.pt", best, epoch=3, config={"model": TINY_MODEL_CONFIG})
    save_checkpoint(output_dir / "phase_c_last.pt", last, epoch=5, config={"model": TINY_MODEL_CONFIG})

    live = _tiny_net(202)
    assert state_digest(live) == last_digest, "runner starts from the live last-epoch weights"

    args = argparse.Namespace(
        no_tqdm=True,
        max_val_batches=1,
        threshold_min=-0.02,
        threshold_max=0.02,
        threshold_steps=5,
        use_calibrated_tau=False,
        skip_calibration=False,
    )
    report: dict[str, Any] = {"tau": {}}
    calibration_path = report_dir / "calibration.json"
    run_post_training_calibration(
        live,
        DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False),
        torch.device("cpu"),
        args=args,
        config_c={"model": TINY_MODEL_CONFIG, "val_split": "val"},
        output_dir=output_dir,
        report_dir=report_dir,
        calibration_path=calibration_path,
        report=report,
        tau_accept=0.0,
        tau_source="default:0.0",
        t_max=2,
        neutral_margin_c=NEUTRAL_MARGIN,
        headline_tau=0.0,
        headline_tau_source="default:0.0",
    )

    artifact = json.loads(calibration_path.read_text())
    carried = artifact.get("lineage")
    if carried is None:
        # The runner still passes its lineage through ``extra``; W3.2 requested
        # the one-line promotion to the top-level field.  Until that lands the
        # artifact is legacy-class at every consumer, which is asserted below.
        carried = artifact["extra"]["lineage"]
        with pytest.raises(MissingLineageError, match="carries no lineage block"):
            verify_calibration_lineage(load_calibration(calibration_path), carried)

    validate_lineage_completeness(carried, where="runner artifact lineage")
    assert carried["checkpoint"]["state_digest"] == best_digest
    assert carried["checkpoint"]["state_digest"] != last_digest
    assert carried["checkpoint"]["checkpoint_role"] == "best"


def test_runner_lineage_is_rejected_against_the_last_epoch_weights(tmp_path: Path) -> None:
    """An old calibration must not be applied to different starting weights."""

    best_path = tmp_path / "best.pt"
    last_path = tmp_path / "last.pt"
    save_checkpoint(best_path, _tiny_net(101), epoch=3, config={"model": TINY_MODEL_CONFIG})
    save_checkpoint(last_path, _tiny_net(202), epoch=5, config={"model": TINY_MODEL_CONFIG})

    calibrated_on_best = _expected(_bind(best_path, _tiny_net(1)), _loader(CALIBRATION_PATIENTS))
    resuming_from_last = _expected(_bind(last_path, _tiny_net(1)), _loader(CALIBRATION_PATIENTS))
    with pytest.raises(LineageMismatchError, match="checkpoint.state_digest"):
        verify_calibration_lineage(calibrated_on_best, resuming_from_last)


# ---------------------------------------------------------------------------
# 7. Header-lineage consistency and strict artifact validation
# ---------------------------------------------------------------------------


def test_matched_artifact_header_cross_check_succeeds(producer: dict[str, Any]) -> None:
    """Artifact payload header matches lineage and verify_calibration_lineage succeeds."""
    artifact_path = producer["artifact_path"]
    loaded = load_calibration(artifact_path)
    result = verify_calibration_lineage(loaded, producer["lineage"])
    assert result["verified"] is True
    assert loaded["checkpoint_sha256"] == producer["lineage"]["checkpoint"]["checkpoint_sha256"]
    assert loaded["source_split"] == producer["lineage"]["split"]["split_name"]
    assert loaded["metric_contract"] == producer["lineage"]["semantics"]["metric_contract"]
    assert loaded["metric_contract_version"] == producer["lineage"]["semantics"]["metric_contract_version"]
    assert loaded["t_max"] == producer["lineage"]["semantics"]["t_max"]
    assert loaded["neutral_margin"] == producer["lineage"]["semantics"]["neutral_margin"]
    assert loaded["metric_space"] == producer["lineage"]["semantics"]["metric_space"]


def test_save_calibration_derives_checkpoint_sha_from_lineage_when_path_omitted(
    tmp_path: Path, producer: dict[str, Any]
) -> None:
    """When checkpoint_path is omitted, checkpoint_sha256 is derived from lineage."""
    out_path = tmp_path / "derived_cal.json"
    payload = save_calibration(
        out_path,
        tau_accept=0.01,
        neutral_margin=NEUTRAL_MARGIN,
        source_split="val",
        checkpoint_path=None,
        t_max=T_MAX,
        threshold_grid=[-0.01, 0.01],
        selected_row={"tau_accept": 0.01, "metric_space": METRIC_SPACE},
        metric_space=METRIC_SPACE,
        metric_contract=METRIC_CONTRACT,
        lineage=producer["lineage"],
    )
    assert payload["checkpoint_path"] is None
    assert payload["checkpoint_sha256"] == producer["lineage"]["checkpoint"]["checkpoint_sha256"]
    loaded = load_calibration(out_path)
    assert loaded["checkpoint_sha256"] == producer["lineage"]["checkpoint"]["checkpoint_sha256"]
    assert verify_calibration_lineage(loaded, producer["lineage"])["verified"] is True


def test_save_calibration_rejects_conflicting_checkpoint_file(
    tmp_path: Path, producer: dict[str, Any]
) -> None:
    """Passing a checkpoint_path with different weights than lineage raises ContractMismatchError."""
    from self_audit.evaluation.contracts import ContractMismatchError

    other_ckpt = tmp_path / "different.pt"
    save_checkpoint(other_ckpt, _tiny_net(999), epoch=9, config={"model": TINY_MODEL_CONFIG})

    with pytest.raises(ContractMismatchError, match="conflicts with lineage checkpoint_sha256"):
        save_calibration(
            tmp_path / "failed.json",
            tau_accept=0.01,
            neutral_margin=NEUTRAL_MARGIN,
            source_split="val",
            checkpoint_path=other_ckpt,
            t_max=T_MAX,
            threshold_grid=[-0.01, 0.01],
            selected_row={"tau_accept": 0.01, "metric_space": METRIC_SPACE},
            metric_space=METRIC_SPACE,
            metric_contract=METRIC_CONTRACT,
            lineage=producer["lineage"],
        )


def test_relocated_checkpoint_passes_when_sha_matches(
    tmp_path: Path, producer: dict[str, Any]
) -> None:
    """A relocated checkpoint with identical SHA256 passes verification."""
    import shutil

    relocated_dir = tmp_path / "relocated"
    relocated_dir.mkdir()
    relocated_path = relocated_dir / "copied_best.pt"
    shutil.copy2(producer["checkpoint"], relocated_path)

    cal_path = tmp_path / "relocated_cal.json"
    save_calibration(
        cal_path,
        tau_accept=0.01,
        neutral_margin=NEUTRAL_MARGIN,
        source_split="val",
        checkpoint_path=relocated_path,
        t_max=T_MAX,
        threshold_grid=[-0.01, 0.01],
        selected_row={"tau_accept": 0.01, "metric_space": METRIC_SPACE},
        metric_space=METRIC_SPACE,
        metric_contract=METRIC_CONTRACT,
        lineage=producer["lineage"],
    )
    loaded = load_calibration(cal_path)
    assert verify_calibration_lineage(loaded, producer["lineage"])["verified"] is True


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("checkpoint_sha256", "f" * 64),
        ("source_split", "test"),
        ("metric_contract", "other_contract_v1"),
        ("metric_contract_version", 99),
        ("t_max", 99),
        ("neutral_margin", 0.05),
        ("metric_space", "volume_native"),
    ],
)
def test_top_level_field_mismatch_against_lineage_is_rejected(
    tmp_path: Path, producer: dict[str, Any], field: str, bad_value: Any
) -> None:
    """Mutating any top-level header field away from the lineage causes load and verify to fail."""
    artifact = dict(load_calibration(producer["artifact_path"]))
    artifact[field] = bad_value

    with pytest.raises(LineageMismatchError, match=f"top-level {field}"):
        verify_calibration_lineage(artifact, producer["lineage"])

    tampered_file = tmp_path / f"tampered_{field}.json"
    tampered_file.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match=f"top-level {field}"):
        load_calibration(tampered_file)


@pytest.mark.parametrize(
    "bad_tau",
    [float("nan"), float("inf"), -float("inf"), True],
)
def test_top_level_tau_must_be_finite_real(
    tmp_path: Path, producer: dict[str, Any], bad_tau: Any
) -> None:
    """Non-finite or boolean tau_accept is rejected on save and verify."""
    with pytest.raises(ValueError):
        save_calibration(
            tmp_path / "bad_tau.json",
            tau_accept=bad_tau,
            neutral_margin=NEUTRAL_MARGIN,
            source_split="val",
            t_max=T_MAX,
            threshold_grid=[-0.01, 0.01],
            selected_row={"tau_accept": 0.0, "metric_space": METRIC_SPACE},
            metric_space=METRIC_SPACE,
            lineage=producer["lineage"],
        )

    artifact = dict(load_calibration(producer["artifact_path"]))
    artifact["tau_accept"] = bad_tau
    with pytest.raises(CalibrationLineageError):
        verify_calibration_lineage(artifact, producer["lineage"])


@pytest.mark.parametrize("bad_tmax", [True, 0, -1, "3"])
def test_top_level_tmax_must_be_strict_positive_integer(
    tmp_path: Path, producer: dict[str, Any], bad_tmax: Any
) -> None:
    """Boolean, zero, negative, or string t_max is rejected."""
    with pytest.raises(ValueError):
        save_calibration(
            tmp_path / "bad_tmax.json",
            tau_accept=0.01,
            neutral_margin=NEUTRAL_MARGIN,
            source_split="val",
            t_max=bad_tmax,
            threshold_grid=[-0.01, 0.01],
            selected_row={"tau_accept": 0.01, "metric_space": METRIC_SPACE},
            metric_space=METRIC_SPACE,
            lineage=producer["lineage"],
        )

    artifact = dict(load_calibration(producer["artifact_path"]))
    artifact["t_max"] = bad_tmax
    with pytest.raises(CalibrationLineageError):
        verify_calibration_lineage(artifact, producer["lineage"])


def test_selected_row_tau_must_equal_chosen_tau(
    tmp_path: Path, producer: dict[str, Any]
) -> None:
    """A disagreement between selected_row tau and top-level tau_accept is rejected."""
    from self_audit.evaluation.contracts import ContractMismatchError

    with pytest.raises(ContractMismatchError, match="selected_row tau"):
        save_calibration(
            tmp_path / "bad_selected_row.json",
            tau_accept=0.01,
            neutral_margin=NEUTRAL_MARGIN,
            source_split="val",
            t_max=T_MAX,
            threshold_grid=[-0.01, 0.01],
            selected_row={"tau_accept": 0.05, "metric_space": METRIC_SPACE},
            metric_space=METRIC_SPACE,
            lineage=producer["lineage"],
        )

    artifact = dict(load_calibration(producer["artifact_path"]))
    artifact["selected_row"]["tau_accept"] = 0.05
    with pytest.raises(CalibrationLineageError, match="selected_row tau"):
        verify_calibration_lineage(artifact, producer["lineage"])


@pytest.mark.parametrize(
    "field",
    [
        "checkpoint_sha256",
        "source_split",
        "metric_contract",
        "metric_contract_version",
        "t_max",
        "neutral_margin",
        "metric_space",
    ],
)
def test_top_level_null_field_on_lineage_artifact_is_rejected(
    tmp_path: Path, producer: dict[str, Any], field: str
) -> None:
    """Explicitly setting any required header field to None/null is rejected."""
    artifact = dict(load_calibration(producer["artifact_path"]))
    artifact[field] = None

    with pytest.raises(LineageMismatchError, match=f"top-level {field}"):
        verify_calibration_lineage(artifact, producer["lineage"])

    null_file = tmp_path / f"null_{field}.json"
    null_file.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match=f"top-level {field}"):
        load_calibration(null_file)


@pytest.mark.parametrize(
    "field",
    [
        "checkpoint_sha256",
        "source_split",
        "metric_contract",
        "metric_contract_version",
        "t_max",
        "neutral_margin",
        "metric_space",
    ],
)
def test_top_level_deleted_field_on_lineage_artifact_is_rejected(
    tmp_path: Path, producer: dict[str, Any], field: str
) -> None:
    """Deleting any required header field from a lineage-bearing artifact is rejected."""
    artifact = dict(load_calibration(producer["artifact_path"]))
    del artifact[field]

    with pytest.raises(LineageMismatchError, match=f"top-level {field}"):
        verify_calibration_lineage(artifact, producer["lineage"])

    deleted_file = tmp_path / f"deleted_{field}.json"
    deleted_file.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match=f"top-level {field}"):
        load_calibration(deleted_file)


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("metric_contract_version", 1.5),
        ("metric_contract_version", True),
        ("metric_contract_version", "1"),
        ("checkpoint_sha256", 123),
        ("checkpoint_sha256", True),
        ("checkpoint_sha256", "not_a_valid_hex"),
        ("checkpoint_sha256", "f" * 63),
        ("neutral_margin", True),
        ("neutral_margin", float("nan")),
        ("neutral_margin", float("inf")),
        ("neutral_margin", "0.005"),
        ("metric_space", 123),
        ("metric_space", True),
        ("metric_space", "unknown_metric_space"),
        ("source_split", 123),
        ("source_split", True),
        ("source_split", ""),
        ("metric_contract", 123),
        ("metric_contract", True),
        ("metric_contract", ""),
    ],
)
def test_top_level_field_invalid_type_is_rejected(
    tmp_path: Path, producer: dict[str, Any], field: str, bad_value: Any
) -> None:
    """Mistyped header fields are rejected by verify_calibration_lineage and load_calibration."""
    artifact = dict(load_calibration(producer["artifact_path"]))
    artifact[field] = bad_value

    with pytest.raises(LineageMismatchError, match=f"top-level {field}"):
        verify_calibration_lineage(artifact, producer["lineage"])

    mistyped_file = tmp_path / f"mistyped_{field}.json"
    mistyped_file.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match=f"top-level {field}"):
        load_calibration(mistyped_file)


def test_audit_checkpoint_cli_preflight_rejects_non_canonical_contract() -> None:
    """scripts/audit_checkpoint.py preflight rejects non-canonical metric contracts early."""
    import importlib.util
    import subprocess

    script_path = _SRC.parent / "scripts" / "audit_checkpoint.py"
    spec = importlib.util.spec_from_file_location("audit_checkpoint_cli", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with pytest.raises(SystemExit, match="Unsupported --metric_contract 'foreground_dice_legacy_one_v1'"):
        mod.main(["--checkpoint", "nonexistent.pt", "--metric_contract", "foreground_dice_legacy_one_v1"])

    with pytest.raises(SystemExit, match="Unsupported --metric_contract 'unknown_target_contract'"):
        mod.main(["--checkpoint", "nonexistent.pt", "--metric_contract", "unknown_target_contract"])

    proc = subprocess.run(
        [sys.executable, str(script_path), "--checkpoint", "nonexistent.pt", "--metric_contract", "legacy_target"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Unsupported --metric_contract 'legacy_target'" in proc.stderr


def test_calibrated_artifact_with_stale_source_signature_fails_closed(producer: dict[str, Any]) -> None:
    """Artifact with stale source content signature fails closed against runtime expectation."""
    from self_audit.evaluation.calibration_lineage import verify_source_identity

    artifact = dict(load_calibration(producer["artifact_path"]))
    stale_lineage = json.loads(json.dumps(artifact["lineage"]))
    stale_lineage["evaluation_code"]["source_content_signature"] = "0" * 64
    artifact["lineage"] = stale_lineage

    # Full lineage verification fails closed under IDENTITY_FIELDS
    with pytest.raises(LineageMismatchError, match="evaluation_code.source_content_signature"):
        verify_calibration_lineage(artifact, producer["lineage"])

    # Dedicated source identity verification specifically reports different source than this run
    with pytest.raises(LineageMismatchError, match="measured by different source than this run"):
        verify_source_identity(stale_lineage, producer["lineage"])


