"""Strict lineage verification for calibration artifacts (W3.2).

A calibration artifact hands a ``tau_accept`` to a deployable evaluation path.
That number is only meaningful for the exact weights, preprocessing recipe,
metric semantics and decision margin it was measured under.  This module is
the gate that proves the artifact and the run about to consume it describe the
same thing.

Three rules shape everything here:

1.  **The expectation is built from the runtime, never from the artifact.**
    :func:`build_expected_lineage` takes the *live* checkpoint binding, the
    *constructed* loader and the protocol's declared semantics.  An artifact
    can therefore never be its own witness: comparing a record against values
    read out of that same record would manufacture success for any file.
2.  **Missing is not passing.**  Every field in :data:`IDENTITY_FIELDS` must be
    present and non-``None`` on both sides.  An artifact that omits a field, or
    carries ``None`` where an identity is required, is rejected -- it is not
    treated as "no objection".
3.  **Legacy stays visibly unverified.**  A pre-lineage artifact can be read
    with :func:`inspect_legacy_calibration`, which always reports
    ``verified=False`` and never yields a value usable for calibrated
    evaluation.  There is no warning-and-continue path.

Cohort roles are kept distinct.  The cohort a threshold was *calibrated* on and
the cohort an *independent evaluation* runs on are different roles with
different permitted membership.  The evaluation cohort is not required to equal
the calibration cohort, but it may not be an arbitrary substitute either: the
protocol must name the exact membership signature it authorises for that role,
and the authorised cohort must be patient-disjoint from the calibration cohort.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from ..audit.semantics import METRIC_SPACES
from ..provenance import (
    LINEAGE_SCHEMA_VERSION,
    MEMBERSHIP_COVERAGE,
    UNKNOWN,
    build_lineage,
    git_source_provenance,
    split_membership_descriptor,
)

#: Top-level schema version for calibration artifacts.
CALIBRATION_SCHEMA_VERSION = 2

#: Version of the *verification* contract implemented by this module: which
#: fields are compared, which are recorded only, and how the cohort roles are
#: resolved.  Bumped whenever any of those change meaning.
CALIBRATION_LINEAGE_SCHEMA_VERSION = 1

#: The evaluated cohort is the same cohort the threshold was calibrated on.
COHORT_ROLE_CALIBRATION = "calibration"

#: The evaluated cohort is a separate, protocol-authorised cohort that must be
#: patient-disjoint from the calibration cohort.
COHORT_ROLE_INDEPENDENT_EVALUATION = "independent_evaluation"

COHORT_ROLES: tuple[str, ...] = (
    COHORT_ROLE_CALIBRATION,
    COHORT_ROLE_INDEPENDENT_EVALUATION,
)

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


def _is_int(value: Any) -> bool:
    """``True`` for a real integer.  ``bool`` is not an integer here.

    ``True == 1`` in Python, so a presence-and-equality check alone accepts a
    boolean wherever a version or a count is expected.
    """

    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64.match(value))


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


#: Type/format rule per field, so a structurally present value cannot be the
#: wrong kind of thing.  Anything not listed must simply be a non-empty string.
_FIELD_RULES: dict[str, tuple[Any, str]] = {
    "lineage_schema_version": (_is_int, "an integer (not a bool)"),
    "checkpoint.checkpoint_sha256": (_is_hex64, "a 64-character lowercase hex digest"),
    "checkpoint.state_digest": (_is_hex64, "a 64-character lowercase hex digest"),
    "checkpoint.file_state_digest": (_is_hex64, "a 64-character lowercase hex digest"),
    "checkpoint.model_identity.signature": (_is_hex64, "a 64-character lowercase hex digest"),
    "checkpoint.model_identity.num_classes": (_is_int, "an integer (not a bool)"),
    "checkpoint.model_identity.shared_channels": (_is_int, "an integer (not a bool)"),
    "checkpoint.model_identity.window_k": (_is_int, "an integer (not a bool)"),
    "checkpoint.model_identity.max_turns": (_is_int, "an integer (not a bool)"),
    "semantics.metric_contract_version": (_is_int, "an integer (not a bool)"),
    "semantics.metric_contract_definition": (
        lambda value: isinstance(value, Mapping) and bool(value),
        "a non-empty mapping",
    ),
    "semantics.neutral_margin": (_is_real, "a real number (not a bool)"),
    "semantics.t_max": (_is_int, "an integer (not a bool)"),
    "semantics.num_classes": (_is_int, "an integer (not a bool)"),
    "semantics.class_names": (_is_str_list, "a list of strings"),
    "preprocessing.signature": (_is_hex64, "a 64-character lowercase hex digest"),
    "split.membership_signature": (_is_hex64, "a 64-character lowercase hex digest"),
    "split.num_records": (_is_int, "an integer (not a bool)"),
    "split.case_ids": (_is_str_list, "a list of strings"),
    "split.patient_ids": (_is_str_list, "a list of strings"),
    "cohort.membership_signature": (_is_hex64, "a 64-character lowercase hex digest"),
    "cohort.covers_full_split": (lambda value: value is True or value is False, "a bool"),
    "cohort.available_slices": (_is_int, "an integer (not a bool)"),
    "cohort.iterated_slices": (_is_int, "an integer (not a bool)"),
    "cohort.sampling.iterates_every_record": (
        lambda value: value is True or value is False,
        "a bool",
    ),
    "evaluation_code.git_sha": (
        lambda value: isinstance(value, str) and bool(value),
        "a non-empty string",
    ),
}

_FIELD_RULES.update(
    {
        "evaluation_code.source_content_signature": (
            _is_hex64,
            "a 64-character lowercase hex digest (an unknown source signature fails closed)",
        ),
        "evaluation_code.source_content_signature_known": (
            lambda value: value is True or value is False,
            "a bool",
        ),
        "evaluation_code.source_signature_version": (_is_int, "an integer (not a bool)"),
        "checkpoint.producer.producer_source_content_signature_known": (
            lambda value: value is True or value is False,
            "a bool",
        ),
    }
)

class CalibrationLineageError(ValueError):
    """Base class for every refusal in this module."""


class MissingLineageError(CalibrationLineageError):
    """The artifact carries no lineage, or an identity field is absent/``None``."""


class LineageMismatchError(CalibrationLineageError):
    """An artifact identity field disagrees with the runtime it is used in."""


class CohortRoleError(CalibrationLineageError):
    """The evaluated cohort is not permitted for its declared role."""


# ---------------------------------------------------------------------------
# Field policy
# ---------------------------------------------------------------------------

#: Dotted paths that must be present, non-``None``, and *equal* on both sides.
#:
#: Each entry answers "would a difference here change what ``tau_accept``
#: means?".  Weights (``checkpoint_sha256`` / the two state digests), the
#: resolved model configuration and backend (``model_identity``), entropy
#: semantics, the metric contract in full, the metric space, the decision
#: margin, ``t_max``, the class mapping and the preprocessing recipe all do.
IDENTITY_FIELDS: tuple[str, ...] = (
    "lineage_schema_version",
    "checkpoint.checkpoint_sha256",
    "checkpoint.state_digest",
    "checkpoint.file_state_digest",
    "checkpoint.producer.producer_git_sha",
    "checkpoint.model_identity.signature",
    "checkpoint.model_identity.class_name",
    "checkpoint.model_identity.num_classes",
    "checkpoint.model_identity.shared_channels",
    "checkpoint.model_identity.window_k",
    "checkpoint.model_identity.max_turns",
    "checkpoint.model_identity.encoder_name",
    "checkpoint.model_identity.encoder_backend",
    "checkpoint.model_identity.entropy_version",
    # The scoped source-content signature of the run that *measured* the
    # threshold.  Shared with the producer (``git_source_provenance``) rather
    # than re-derived here, and compared instead of the git SHA: a report-only
    # commit leaves it unchanged, any scoped source edit changes it.
    "evaluation_code.source_content_signature",
    "evaluation_code.source_signature_version",
    "semantics.entropy_version",
    "semantics.metric_contract",
    "semantics.metric_contract_version",
    "semantics.metric_contract_definition",
    "semantics.metric_space",
    "semantics.neutral_margin",
    "semantics.empty_class_policy",
    "semantics.t_max",
    "semantics.num_classes",
    "semantics.class_names",
    "preprocessing.signature",
)

#: Compared only when either side records one, because a checkpoint written
#: before source signatures existed honestly has none.  A disagreement is
#: always a refusal; a mutual absence is recorded as unknown and never
#: presented as verified producing provenance.
CONDITIONAL_IDENTITY_FIELDS: tuple[str, ...] = (
    "checkpoint.producer.producer_source_content_signature",
    "checkpoint.producer.producer_source_signature_version",
)

#: Paths that must exist but are deliberately *not* compared, with the reason.
#:
#: Every path listed here is still required to be *present*; see
#: :data:`REQUIRED_PRESENT`.
RECORDED_NOT_COMPARED: dict[str, str] = {
    "evaluation_code.git_sha": (
        "the commit that happened to be checked out; a report-only commit moves it "
        "without changing behaviour, so evaluation_code.source_content_signature is "
        "compared instead"
    ),
    "evaluation_code.dirty": (
        "whole-tree dirt includes reports and scratch files; the scoped source content "
        "signature is compared instead"
    ),
    "checkpoint.checkpoint_path": (
        "filesystem location is not identity; the same weights at another path "
        "are the same weights, and different weights at the same path are "
        "caught by checkpoint_sha256/state_digest"
    ),
    "checkpoint.epoch": "bookkeeping, not identity",
    "checkpoint.global_step": "bookkeeping, not identity",
    "semantics.semantics_source": (
        "says whether the producer read its semantics off a transition cache or "
        "resolved them from the contract registry; the values themselves are compared"
    ),
    "semantics.cache_schema_version": (
        "version of the transition cache the producer measured; a consumer that "
        "reads no cache has none, and the metric contract is compared instead"
    ),
    "cohort": "measurement extent; membership is enforced by the cohort role",
}

#: Cohort paths, enforced by role rather than by blanket equality.
COHORT_FIELDS: tuple[str, ...] = ("split.split_name", "split.membership_signature")

#: Paths that must be present and non-``None`` even though they are not
#: compared field-for-field.  Without this, "recorded but not compared" would
#: mean "may be deleted", and dropping a block would weaken the record instead
#: of failing it.
REQUIRED_PRESENT: tuple[str, ...] = (
    "evaluation_code",
    "evaluation_code.git_sha",
    "evaluation_code.source_content_signature_known",
    "evaluation_code.source_signature_scope",
    "checkpoint.checkpoint_path",
    "checkpoint.checkpoint_role",
    "checkpoint.producer",
    "checkpoint.producer.producer_recorded",
    "checkpoint.producer.producer_source_content_signature_known",
    "checkpoint.model_identity",
    "split.case_ids",
    "split.patient_ids",
    "split.num_records",
    "split.coverage",
    "cohort",
    "cohort.membership_signature",
    "cohort.coverage",
    "cohort.covers_full_split",
    "cohort.available_slices",
    "cohort.iterated_slices",
    "cohort.sampling",
    "cohort.sampling.iterates_every_record",
)

#: The cohort *definition* -- which records are in it -- compared when the
#: evaluated cohort is claimed to be the calibration cohort.  Run extent
#: (``max_batches``, ``observed_samples``, ``loader_batches``) is deliberately
#: absent: a consumer may measure a prefix of the same cohort, but it may not
#: measure a differently-populated one.
CALIBRATION_COHORT_FIELDS: tuple[str, ...] = (
    "split.case_ids",
    "split.patient_ids",
    "split.num_records",
    "cohort.membership_signature",
    "cohort.available_slices",
    "cohort.sampling.subset_indices_signature",
    "cohort.sampling.subset_size",
)

_MISSING = object()


def _get_path(record: Any, path: str) -> Any:
    """Resolve a dotted path, returning :data:`_MISSING` when absent."""

    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _normalize(value: Any) -> Any:
    """Canonicalise for comparison without loosening it.

    Lists and tuples become lists; mappings are sorted recursively; everything
    else is returned unchanged.  Numeric types are *not* coerced across the
    int/float boundary and floats are compared exactly: a decision margin that
    differs in the last bit is a different margin.
    """

    if isinstance(value, Mapping):
        return {str(key): _normalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _render(value: Any) -> str:
    try:
        return json.dumps(_normalize(value), sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return repr(value)


# ---------------------------------------------------------------------------
# Cohort policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortPolicy:
    """The protocol's declaration of what the evaluated cohort is allowed to be.

    ``role`` says which role the cohort about to be evaluated plays.  For
    :data:`COHORT_ROLE_INDEPENDENT_EVALUATION` the protocol must additionally
    name ``authorized_membership_signature``: the exact membership signature it
    permits for that role.  That value comes from the evaluation protocol, not
    from the artifact and not from the cohort itself -- otherwise "authorised"
    would mean "whatever was supplied".
    """

    role: str = COHORT_ROLE_CALIBRATION
    authorized_membership_signature: str | None = None
    authorized_split_name: str | None = None

    def __post_init__(self) -> None:
        if self.role not in COHORT_ROLES:
            raise CohortRoleError(
                f"Unknown cohort role {self.role!r}; expected one of {list(COHORT_ROLES)}"
            )
        if self.role == COHORT_ROLE_INDEPENDENT_EVALUATION:
            if not self.authorized_membership_signature:
                raise CohortRoleError(
                    "cohort role 'independent_evaluation' requires the protocol to declare "
                    "authorized_membership_signature; an unnamed replacement cohort is refused"
                )
        elif self.authorized_membership_signature is not None:
            raise CohortRoleError(
                "authorized_membership_signature is only meaningful for cohort role "
                f"'independent_evaluation', not {self.role!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": str(self.role),
            "authorized_membership_signature": self.authorized_membership_signature,
            "authorized_split_name": self.authorized_split_name,
            # Patient disjointness is a property of this protocol, not a knob:
            # an independent evaluation that shares patients with the
            # calibration cohort is not independent, so there is no setting
            # that permits it.
            "patient_disjointness": "required_for_independent_evaluation",
        }


# ---------------------------------------------------------------------------
# Expected lineage, built from the runtime
# ---------------------------------------------------------------------------


def build_expected_lineage(
    *,
    binding: Any,
    loader: Any,
    split_name: str,
    metric_contract: Any,
    metric_space: str,
    neutral_margin: float,
    t_max: int,
    max_batches: int | None = None,
    batch_size: int | None = None,
    observed_samples: int | None = None,
) -> dict[str, Any]:
    """Describe what the run about to consume a tau *actually* is.

    ``binding`` is the live :class:`~self_audit.provenance.CheckpointBinding`
    (or its dict form) produced by ``bind_evaluation_checkpoint``, so the
    weights described here are the weights loaded into the module.  ``loader``
    is the constructed dataset/loader, so preprocessing and split membership are
    read off the objects that will be iterated.  ``metric_contract``,
    ``metric_space``, ``neutral_margin`` and ``t_max`` come from the evaluation
    protocol.  Nothing is read from a calibration artifact.
    """

    record = binding.as_dict() if hasattr(binding, "as_dict") else dict(binding)
    return build_lineage(
        binding=record,
        loader=loader,
        split_name=str(split_name),
        metric_contract=metric_contract,
        metric_space=str(metric_space),
        neutral_margin=float(neutral_margin),
        t_max=int(t_max),
        max_batches=max_batches,
        batch_size=batch_size,
        observed_samples=observed_samples,
    )


def validate_lineage_completeness(lineage: Any, *, where: str) -> dict[str, Any]:
    """Require the record to be complete, well-typed and of a known schema.

    Three separate checks, because each alone is bypassable:

    * every field in :data:`IDENTITY_FIELDS`, :data:`COHORT_FIELDS` and
      :data:`REQUIRED_PRESENT` must be present and non-``None`` -- deleting a
      block must fail, not quietly narrow the comparison;
    * every field with a rule in :data:`_FIELD_RULES` must be of the right
      kind.  ``True == 1`` in Python, so a version or a count that is a
      ``bool`` passes an equality check while meaning nothing; a digest that is
      not 64 hex characters is not a digest;
    * the schema version must be exactly the one this build knows how to read.

    Used on both sides: an incomplete *expected* record would silently narrow
    the comparison, and an incomplete *artifact* record is exactly the case a
    permissive loader would wave through.
    """

    if not isinstance(lineage, Mapping):
        raise MissingLineageError(
            f"{where} carries no lineage record (got {type(lineage).__name__}); "
            "a calibration without lineage cannot enter a calibrated evaluation"
        )
    version = lineage.get("lineage_schema_version")
    if not _is_int(version) or version != LINEAGE_SCHEMA_VERSION:
        raise MissingLineageError(
            f"{where} declares lineage_schema_version {version!r}; this build verifies "
            f"integer version {LINEAGE_SCHEMA_VERSION} only"
        )
    absent: list[str] = []
    for path in IDENTITY_FIELDS + COHORT_FIELDS + REQUIRED_PRESENT:
        value = _get_path(lineage, path)
        if value is _MISSING:
            absent.append(f"{path} (absent)")
        elif value is None:
            absent.append(f"{path} (null)")
    if absent:
        raise MissingLineageError(
            f"{where} is missing required lineage field(s): " + "; ".join(sorted(absent))
        )
    malformed: list[str] = []
    for path, (rule, description) in _FIELD_RULES.items():
        value = _get_path(lineage, path)
        if value is _MISSING:
            continue
        if not rule(value):
            malformed.append(f"{path} must be {description}, got {value!r}")
    if malformed:
        raise MissingLineageError(
            f"{where} has malformed lineage field(s): " + "; ".join(sorted(malformed))
        )
    return dict(lineage)


# ---------------------------------------------------------------------------
# Source identity of the measuring run
# ---------------------------------------------------------------------------


def verify_source_identity(
    artifact: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    artifact_name: str = "calibration artifact",
) -> dict[str, Any]:
    """Prove the artifact was measured by the source this run is executing.

    The signature compared here is the producer's shared
    ``source_content_signature`` (``self_audit.provenance.git_source_provenance``),
    not a second digest maintained by this module: a functional change anywhere
    in the scoped source -- the model, inference, the counterfactual generator,
    the metric contracts, this gate -- changes what a threshold means, so the
    whole scope is what must agree.

    Two things are deliberately *not* the git SHA.  A report-only commit moves
    the SHA without changing behaviour and must not expire a calibration; an
    uncommitted edit to scoring code leaves the SHA alone and must.  The
    content signature answers both.

    It fails closed: a signature either side could not compute is ``unknown``,
    and an unknown is never treated as agreement.
    """

    problems: list[str] = []
    for side, record in (("artifact", artifact), ("runtime", expected)):
        known = _get_path(record, "evaluation_code.source_content_signature_known")
        if known is not True:
            problems.append(
                f"{side} source_content_signature_known is {_render(known)}: the scoped source "
                "could not be identified, and an unidentified source is never a match"
            )
    if problems:
        raise LineageMismatchError(
            f"{artifact_name} cannot be certified against this run's source: "
            + "; ".join(problems)
        )

    artifact_signature = _get_path(artifact, "evaluation_code.source_content_signature")
    runtime_signature = _get_path(expected, "evaluation_code.source_content_signature")
    if artifact_signature != runtime_signature:
        raise LineageMismatchError(
            f"{artifact_name} was measured by different source than this run "
            f"(artifact source_content_signature={_render(artifact_signature)}, "
            f"runtime={_render(runtime_signature)}). The git revision is not what is compared "
            "here: a report-only commit leaves this signature unchanged, so a difference means "
            "scoped source content actually changed. Re-run calibration."
        )

    # The revision that *produced the checkpoint* is a separate fact, recorded
    # separately.  It is compared whenever either side claims one; a checkpoint
    # written before source signatures existed has none on both sides and is
    # reported as unknown rather than dressed up as verified.
    producing: dict[str, Any] = {}
    for path in CONDITIONAL_IDENTITY_FIELDS:
        left = _get_path(artifact, path)
        right = _get_path(expected, path)
        left = None if left is _MISSING else left
        right = None if right is _MISSING else right
        producing[path] = left
        if (left is not None or right is not None) and _normalize(left) != _normalize(right):
            raise LineageMismatchError(
                f"{artifact_name} names a different checkpoint-producing source than this run "
                f"at {path}: artifact={_render(left)} runtime={_render(right)}"
            )
    producing_known = _get_path(artifact, "checkpoint.producer.producer_source_content_signature_known") is True

    return {
        "verified": True,
        "source_content_signature": artifact_signature,
        "source_signature_version": _get_path(artifact, "evaluation_code.source_signature_version"),
        "source_signature_scope": _get_path(artifact, "evaluation_code.source_signature_scope"),
        "compared": "evaluation_code.source_content_signature (not git_sha)",
        "checkpoint_producing_source": producing,
        "checkpoint_producing_source_known": producing_known,
        "note": (
            "the measuring run's scoped source content is compared; the checkpoint-producing "
            "source is a separate record and is compared only when one exists"
        ),
    }
# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _patient_ids(record: Any, path: str) -> list[str]:
    value = _get_path(record, path)
    if value is _MISSING or value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    if isinstance(value, Sequence):
        return sorted({str(item) for item in value})
    return []


def _assert_calibration_cohort_is_complete(artifact: Mapping[str, Any]) -> None:
    """The cohort a threshold was calibrated on must be the whole cohort.

    A split *name* and a membership signature say which records exist, not how
    many were measured.  A calibration run that truncated with ``max_batches``,
    iterated a ``Subset``, or dropped a trailing partial batch selected its
    threshold on a different population than the one it names -- and would
    otherwise pass every signature comparison.
    """

    problems: list[str] = []
    if _get_path(artifact, "cohort.covers_full_split") is not True:
        problems.append(
            "cohort.covers_full_split is "
            f"{_render(_get_path(artifact, 'cohort.covers_full_split'))}, not true"
        )
    if _get_path(artifact, "cohort.sampling.iterates_every_record") is not True:
        problems.append("cohort.sampling.iterates_every_record is not true")
    subset_signature = _get_path(artifact, "cohort.sampling.subset_indices_signature")
    if subset_signature not in (None, _MISSING):
        problems.append("the calibration iterated a subset of the split")
    max_batches = _get_path(artifact, "cohort.max_batches")
    if max_batches not in (None, _MISSING):
        problems.append(f"the calibration was truncated at max_batches={max_batches!r}")
    coverage = _get_path(artifact, "cohort.coverage")
    if coverage != MEMBERSHIP_COVERAGE:
        problems.append(f"cohort.coverage is {_render(coverage)}, not {MEMBERSHIP_COVERAGE!r}")
    cohort_signature = _get_path(artifact, "cohort.membership_signature")
    split_signature = _get_path(artifact, "split.membership_signature")
    if cohort_signature != split_signature:
        problems.append(
            "cohort.membership_signature disagrees with split.membership_signature "
            f"({_render(cohort_signature)} vs {_render(split_signature)})"
        )
    available = _get_path(artifact, "cohort.available_slices")
    iterated = _get_path(artifact, "cohort.iterated_slices")
    if available != iterated:
        problems.append(
            f"cohort.iterated_slices {iterated!r} != cohort.available_slices {available!r}"
        )
    observed = _get_path(artifact, "cohort.observed_samples")
    if observed not in (None, _MISSING):
        if available not in (None, _MISSING) and observed != available:
            problems.append(
                f"cohort.observed_samples {observed!r} != cohort.available_slices {available!r}"
            )
    if problems:
        raise CohortRoleError(
            "the artifact's calibration cohort is not a complete cohort, so its threshold was "
            "selected on a different population than the split it names: " + "; ".join(problems)
        )


def _verify_cohort(
    artifact: Mapping[str, Any],
    expected: Mapping[str, Any],
    policy: CohortPolicy,
) -> dict[str, Any]:
    """Apply the role-specific membership rule and report what it decided."""

    # True in both roles: an artifact calibrated on a truncated or subsetted
    # cohort is not usable, whatever the evaluated cohort turns out to be.
    _assert_calibration_cohort_is_complete(artifact)

    artifact_split = str(_get_path(artifact, "split.split_name"))
    artifact_signature = str(_get_path(artifact, "split.membership_signature"))
    runtime_split = str(_get_path(expected, "split.split_name"))
    runtime_signature = str(_get_path(expected, "split.membership_signature"))
    artifact_patients = _patient_ids(artifact, "split.patient_ids")
    runtime_patients = _patient_ids(expected, "split.patient_ids")

    decision: dict[str, Any] = {
        "role": policy.role,
        "calibration_split_name": artifact_split,
        "calibration_membership_signature": artifact_signature,
        "evaluation_split_name": runtime_split,
        "evaluation_membership_signature": runtime_signature,
        "authorized_membership_signature": policy.authorized_membership_signature,
        "calibration_patient_count": len(artifact_patients),
        "evaluation_patient_count": len(runtime_patients),
        "patients_disjoint": None,
        "coverage_note": (
            "membership signatures prove which records a split contained, never that "
            "their pixel content is unchanged"
        ),
    }

    if policy.role == COHORT_ROLE_CALIBRATION:
        if artifact_split != runtime_split or artifact_signature != runtime_signature:
            raise CohortRoleError(
                "cohort role 'calibration' requires the evaluated cohort to be the cohort the "
                f"threshold was calibrated on, but the artifact was calibrated on "
                f"{artifact_split!r}/{artifact_signature} and this run evaluates "
                f"{runtime_split!r}/{runtime_signature}. Declare cohort role "
                "'independent_evaluation' with the protocol's authorised membership signature "
                "if a separate cohort is intended."
            )
        # A matching signature proves which case ids existed, not that the
        # cohort was populated the same way, so the definition is compared
        # field for field as well.
        differences = [
            f"{path}: artifact={_render(_get_path(artifact, path))} "
            f"runtime={_render(_get_path(expected, path))}"
            for path in CALIBRATION_COHORT_FIELDS
            if _normalize(_get_path(artifact, path)) != _normalize(_get_path(expected, path))
        ]
        if differences:
            raise CohortRoleError(
                "the evaluated cohort carries the calibration cohort's membership signature but "
                "is not the same cohort: " + "; ".join(differences)
            )
        decision["patients_disjoint"] = False
        decision["accepted_because"] = "evaluated cohort is the calibration cohort itself"
        decision["cohort_definition_fields_compared"] = list(CALIBRATION_COHORT_FIELDS)
        return decision

    # independent_evaluation
    authorized = str(policy.authorized_membership_signature)
    if runtime_signature != authorized:
        raise CohortRoleError(
            f"evaluated cohort {runtime_split!r}/{runtime_signature} is not the cohort the "
            f"protocol authorised for independent evaluation ({authorized}). An arbitrary "
            "replacement cohort is refused."
        )
    if policy.authorized_split_name is not None and runtime_split != str(policy.authorized_split_name):
        raise CohortRoleError(
            f"evaluated split {runtime_split!r} is not the protocol's authorised split "
            f"{policy.authorized_split_name!r} for independent evaluation"
        )
    if runtime_signature == artifact_signature:
        raise CohortRoleError(
            "cohort role 'independent_evaluation' was declared but the evaluated cohort is the "
            f"calibration cohort ({artifact_signature}); declare role 'calibration' instead of "
            "presenting the selection split as independent evidence"
        )
    if not artifact_patients or not runtime_patients:
        raise CohortRoleError(
            "cannot certify cohort independence: patient identity is unavailable "
            f"(calibration={len(artifact_patients)} patients, evaluation="
            f"{len(runtime_patients)} patients). An empty patient list is not proof of disjointness."
        )
    overlap = sorted(set(artifact_patients) & set(runtime_patients))
    decision["patients_disjoint"] = not overlap
    if overlap:
        raise CohortRoleError(
            "independent evaluation cohort shares patient(s) with the calibration cohort: "
            + ", ".join(overlap[:10])
            + ("" if len(overlap) <= 10 else f" (+{len(overlap) - 10} more)")
        )
    decision["accepted_because"] = (
        "cohort matches the protocol-authorised membership signature and is patient-disjoint "
        "from the calibration cohort"
    )
    return decision


def validate_calibration_header(
    header: Mapping[str, Any],
    *,
    where: str,
    error_cls: type[Exception] = LineageMismatchError,
) -> None:
    """Validate top-level calibration artifact fields for strict types and consistency."""
    ver = header.get("schema_version")
    if not _is_int(ver) or ver != CALIBRATION_SCHEMA_VERSION:
        raise error_cls(
            f"{where} top-level schema_version {ver!r} is not valid integer {CALIBRATION_SCHEMA_VERSION}"
        )
    if "tau_accept" not in header or header.get("tau_accept") is None:
        raise error_cls(f"{where} top-level tau_accept is required and cannot be null")
    tau = header.get("tau_accept")
    if not _is_real(tau) or not math.isfinite(float(tau)):
        raise error_cls(
            f"{where} top-level tau_accept {tau!r} must be a finite real number (not bool, NaN, or Inf)"
        )
    if "t_max" not in header or header.get("t_max") is None:
        raise error_cls(f"{where} top-level t_max is required and cannot be null")
    t_max = header.get("t_max")
    if not _is_int(t_max) or int(t_max) <= 0:
        raise error_cls(
            f"{where} top-level t_max {t_max!r} must be a positive integer (not bool)"
        )
    selected_row = header.get("selected_row")
    if isinstance(selected_row, Mapping):
        sel_tau = selected_row.get("tau_accept") if "tau_accept" in selected_row else selected_row.get("tau")
        if sel_tau is not None:
            if not _is_real(sel_tau) or not math.isfinite(float(sel_tau)):
                raise error_cls(
                    f"{where} selected_row tau {sel_tau!r} must be a finite real number"
                )
            if float(sel_tau) != float(tau):
                raise error_cls(
                    f"{where} selected_row tau {float(sel_tau)!r} does not match chosen tau_accept {float(tau)!r}"
                )
    if "checkpoint_sha256" in header and header.get("checkpoint_sha256") is not None:
        sha = header.get("checkpoint_sha256")
        if not _is_hex64(sha):
            raise error_cls(
                f"{where} top-level checkpoint_sha256 {sha!r} must be a 64-character lowercase hex digest"
            )
    if "source_split" in header and header.get("source_split") is not None:
        split = header.get("source_split")
        if not isinstance(split, str) or not split:
            raise error_cls(
                f"{where} top-level source_split {split!r} must be a non-empty string"
            )
    if "metric_contract" in header and header.get("metric_contract") is not None:
        contract = header.get("metric_contract")
        if not isinstance(contract, str) or not contract:
            raise error_cls(
                f"{where} top-level metric_contract {contract!r} must be a non-empty string"
            )
    if "metric_contract_version" in header and header.get("metric_contract_version") is not None:
        c_ver = header.get("metric_contract_version")
        if not _is_int(c_ver):
            raise error_cls(
                f"{where} top-level metric_contract_version {c_ver!r} must be an integer (not bool or float)"
            )
    if "neutral_margin" in header and header.get("neutral_margin") is not None:
        margin = header.get("neutral_margin")
        if not _is_real(margin) or not math.isfinite(float(margin)):
            raise error_cls(
                f"{where} top-level neutral_margin {margin!r} must be a finite real number (not bool, NaN, or Inf)"
            )
    if "metric_space" in header and header.get("metric_space") is not None:
        space = header.get("metric_space")
        if not isinstance(space, str) or space not in METRIC_SPACES:
            raise error_cls(
                f"{where} top-level metric_space {space!r} must be a valid metric space in {sorted(METRIC_SPACES)}"
            )


def validate_header_lineage_consistency(
    header: Mapping[str, Any],
    lineage: Mapping[str, Any],
    *,
    where: str,
    error_cls: type[Exception] = LineageMismatchError,
) -> None:
    """Ensure top-level artifact header fields agree with the embedded lineage.

    For lineage-bearing artifacts, required header fields must be present,
    non-null, strictly typed, and match the corresponding lineage values.
    """
    # 1. checkpoint_sha256
    if "checkpoint_sha256" not in header or header.get("checkpoint_sha256") is None:
        raise error_cls(
            f"{where} top-level checkpoint_sha256 is required and cannot be null or missing on a lineage-bearing artifact"
        )
    top_sha = header["checkpoint_sha256"]
    if not _is_hex64(top_sha):
        raise error_cls(
            f"{where} top-level checkpoint_sha256 {top_sha!r} must be a 64-character lowercase hex digest"
        )
    lineage_ckpt = lineage.get("checkpoint")
    lineage_sha = lineage_ckpt.get("checkpoint_sha256") if isinstance(lineage_ckpt, Mapping) else None
    if lineage_sha is not None and str(top_sha) != str(lineage_sha):
        raise error_cls(
            f"{where} top-level checkpoint_sha256 {top_sha!r} conflicts with lineage checkpoint_sha256 {lineage_sha!r}"
        )

    # 2. source_split
    if "source_split" not in header or header.get("source_split") is None:
        raise error_cls(
            f"{where} top-level source_split is required and cannot be null or missing on a lineage-bearing artifact"
        )
    top_split = header["source_split"]
    if not isinstance(top_split, str) or not top_split:
        raise error_cls(
            f"{where} top-level source_split {top_split!r} must be a non-empty string"
        )
    lineage_split = lineage.get("split")
    lineage_split_name = lineage_split.get("split_name") if isinstance(lineage_split, Mapping) else None
    if lineage_split_name is not None and str(top_split) != str(lineage_split_name):
        raise error_cls(
            f"{where} top-level source_split {top_split!r} conflicts with lineage split_name {lineage_split_name!r}"
        )

    semantics = lineage.get("semantics") if isinstance(lineage.get("semantics"), Mapping) else {}

    # 3. metric_contract
    if "metric_contract" not in header or header.get("metric_contract") is None:
        raise error_cls(
            f"{where} top-level metric_contract is required and cannot be null or missing on a lineage-bearing artifact"
        )
    top_contract = header["metric_contract"]
    if not isinstance(top_contract, str) or not top_contract:
        raise error_cls(
            f"{where} top-level metric_contract {top_contract!r} must be a non-empty string"
        )
    lineage_contract = semantics.get("metric_contract")
    if lineage_contract is not None and str(top_contract) != str(lineage_contract):
        raise error_cls(
            f"{where} top-level metric_contract {top_contract!r} conflicts with lineage metric_contract {lineage_contract!r}"
        )

    # 4. metric_contract_version
    if "metric_contract_version" not in header or header.get("metric_contract_version") is None:
        raise error_cls(
            f"{where} top-level metric_contract_version is required and cannot be null or missing on a lineage-bearing artifact"
        )
    top_version = header["metric_contract_version"]
    if not _is_int(top_version):
        raise error_cls(
            f"{where} top-level metric_contract_version {top_version!r} must be an integer (not bool or float)"
        )
    lineage_version = semantics.get("metric_contract_version")
    if lineage_version is not None and top_version != lineage_version:
        raise error_cls(
            f"{where} top-level metric_contract_version {top_version!r} conflicts with lineage metric_contract_version {lineage_version!r}"
        )

    # 5. t_max
    if "t_max" not in header or header.get("t_max") is None:
        raise error_cls(
            f"{where} top-level t_max is required and cannot be null or missing on a lineage-bearing artifact"
        )
    top_t_max = header["t_max"]
    if not _is_int(top_t_max) or int(top_t_max) <= 0:
        raise error_cls(
            f"{where} top-level t_max {top_t_max!r} must be a positive integer (not bool)"
        )
    lineage_t_max = semantics.get("t_max")
    if lineage_t_max is not None and top_t_max != lineage_t_max:
        raise error_cls(
            f"{where} top-level t_max {top_t_max!r} conflicts with lineage t_max {lineage_t_max!r}"
        )

    # 6. neutral_margin
    if "neutral_margin" not in header or header.get("neutral_margin") is None:
        raise error_cls(
            f"{where} top-level neutral_margin is required and cannot be null or missing on a lineage-bearing artifact"
        )
    top_margin = header["neutral_margin"]
    if not _is_real(top_margin) or not math.isfinite(float(top_margin)):
        raise error_cls(
            f"{where} top-level neutral_margin {top_margin!r} must be a finite real number (not bool, NaN, or Inf)"
        )
    lineage_margin = semantics.get("neutral_margin")
    if lineage_margin is not None and abs(float(top_margin) - float(lineage_margin)) > 1e-7:
        raise error_cls(
            f"{where} top-level neutral_margin {top_margin!r} conflicts with lineage neutral_margin {lineage_margin!r}"
        )

    # 7. metric_space
    if "metric_space" not in header or header.get("metric_space") is None:
        raise error_cls(
            f"{where} top-level metric_space is required and cannot be null or missing on a lineage-bearing artifact"
        )
    top_space = header["metric_space"]
    if not isinstance(top_space, str) or top_space not in METRIC_SPACES:
        raise error_cls(
            f"{where} top-level metric_space {top_space!r} must be a valid metric space in {sorted(METRIC_SPACES)}"
        )
    lineage_space = semantics.get("metric_space")
    if lineage_space is not None and str(top_space) != str(lineage_space):
        raise error_cls(
            f"{where} top-level metric_space {top_space!r} conflicts with lineage metric_space {lineage_space!r}"
        )


def verify_calibration_lineage(
    artifact: Any,
    expected: Mapping[str, Any],
    *,
    cohort_policy: CohortPolicy | None = None,
    artifact_name: str = "calibration artifact",
) -> dict[str, Any]:
    """Hard-fail unless the artifact's lineage matches this runtime.

    ``artifact`` is either a loaded calibration payload (its ``lineage`` key is
    used) or a lineage record directly.  ``expected`` must come from
    :func:`build_expected_lineage`.  Returns the verification record to embed in
    a report; raises :class:`CalibrationLineageError` on any missing field,
    any identity mismatch, any cohort-role violation, or any change to the
    scoring code the artifact was measured by.
    """

    policy = cohort_policy if cohort_policy is not None else CohortPolicy()
    # A calibration payload is recognised by its own artifact keys, never by
    # guessing: handing this function a payload whose lineage key was dropped
    # must fail as "no lineage", not silently compare the payload itself.
    is_payload = isinstance(artifact, Mapping) and {"schema_version", "tau_accept"} <= set(artifact)
    if is_payload:
        validate_calibration_header(artifact, where=artifact_name, error_cls=LineageMismatchError)
        artifact_lineage = artifact.get("lineage")
        if artifact_lineage is not None and isinstance(artifact_lineage, Mapping):
            validate_header_lineage_consistency(
                artifact, artifact_lineage, where=artifact_name, error_cls=LineageMismatchError
            )
    else:
        artifact_lineage = artifact
    if artifact_lineage is None:
        raise MissingLineageError(
            f"{artifact_name} carries no lineage block. It predates the strict calibration "
            "schema and is unverified: inspect it with inspect_legacy_calibration() if you need "
            "to read it, but it may never supply a threshold to a calibrated evaluation."
        )

    artifact_record = validate_lineage_completeness(artifact_lineage, where=artifact_name)
    expected_record = validate_lineage_completeness(
        expected, where="expected runtime lineage"
    )

    mismatches: list[str] = []
    for path in IDENTITY_FIELDS:
        left = _normalize(_get_path(artifact_record, path))
        right = _normalize(_get_path(expected_record, path))
        if left != right:
            mismatches.append(f"{path}: artifact={_render(left)} runtime={_render(right)}")
    if mismatches:
        raise LineageMismatchError(
            f"{artifact_name} was produced under a different lineage than this run; refusing to "
            "apply its threshold: " + "; ".join(mismatches)
        )

    cohort = _verify_cohort(artifact_record, expected_record, policy)
    source = verify_source_identity(artifact_record, expected_record, artifact_name=artifact_name)

    producer_sha = _get_path(artifact_record, "checkpoint.producer.producer_git_sha")
    producer_recorded = _get_path(artifact_record, "checkpoint.producer.producer_recorded") is True
    producer_sha_known = isinstance(producer_sha, str) and producer_sha != UNKNOWN
    return {
        "calibration_lineage_schema_version": int(CALIBRATION_LINEAGE_SCHEMA_VERSION),
        "lineage_schema_version": int(LINEAGE_SCHEMA_VERSION),
        "verified": True,
        "verified_scope": (
            "identity fields, required recorded fields, scoped source content, cohort role"
        ),
        "compared_fields": list(IDENTITY_FIELDS),
        "conditionally_compared_fields": list(CONDITIONAL_IDENTITY_FIELDS),
        "required_present_fields": list(REQUIRED_PRESENT),
        "recorded_not_compared": dict(RECORDED_NOT_COMPARED),
        "cohort": cohort,
        "cohort_policy": policy.as_dict(),
        "source_identity": source,
        "checkpoint_producing_git_sha": producer_sha,
        "checkpoint_producer_recorded": producer_recorded,
        "checkpoint_producing_git_sha_unknown": not producer_sha_known,
        # The producing revision is compared on both sides, so a match proves
        # they agree.  When that agreed value is "unknown" the checkpoint
        # simply predates the record: that is an honest absence, and this
        # verification must not be read as having established a producing
        # provenance it never saw.
        "producer_provenance_complete": bool(
            producer_recorded and producer_sha_known and source["checkpoint_producing_source_known"]
        ),
        "calibration_code": _get_path(artifact_record, "evaluation_code"),
        "consuming_code": git_source_provenance(),
        "note": (
            "the calibrating revision and the consuming revision are recorded separately and "
            "are not required to match; what pins the code is the compared scoped "
            "evaluation_code.source_content_signature, and the checkpoint-producing source is "
            "compared separately"
        ),
    }


# ---------------------------------------------------------------------------
# Legacy inspection -- read-only, never a calibrated result
# ---------------------------------------------------------------------------


def inspect_legacy_calibration(source: Any) -> dict[str, Any]:
    """Read a calibration artifact this build cannot verify, marked as such.

    Accepts a path or an already-parsed payload of *any* schema version.  The
    returned record always carries ``verified=False`` and
    ``usable_for_calibrated_evaluation=False``; the declared threshold is
    reported under ``declared_tau_accept`` so it can never be mistaken for a
    verified ``tau_accept``.  There is no flag that promotes the result.
    """

    if isinstance(source, Mapping):
        payload: Mapping[str, Any] = source
        origin = "<in-memory payload>"
    else:
        path = Path(os.fspath(source))
        origin = str(path)
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, Mapping):
            raise CalibrationLineageError(
                f"Legacy calibration artifact {origin} is not a JSON object"
            )
        payload = parsed

    lineage = payload.get("lineage")
    lineage_complete = False
    lineage_problem: str | None = None
    if lineage is not None:
        try:
            validate_lineage_completeness(lineage, where=f"legacy artifact {origin}")
            lineage_complete = True
        except CalibrationLineageError as err:
            lineage_problem = str(err)

    return {
        "verified": False,
        "usable_for_calibrated_evaluation": False,
        "source": origin,
        "schema_version": payload.get("schema_version"),
        "declared_tau_accept": payload.get("tau_accept"),
        "declared_neutral_margin": payload.get("neutral_margin"),
        "declared_source_split": payload.get("source_split"),
        "declared_checkpoint_path": payload.get("checkpoint_path"),
        "declared_checkpoint_sha256": payload.get("checkpoint_sha256"),
        "lineage_present": lineage is not None,
        "lineage_complete": lineage_complete,
        "lineage_problem": lineage_problem,
        "reason": (
            "read through the explicit legacy-inspection API: no runtime lineage was verified, "
            "so nothing here is evidence that this threshold applies to any particular weights, "
            "preprocessing recipe, metric contract or cohort"
        ),
    }


def runtime_membership_signature(loader: Any, *, split_name: str) -> str:
    """Membership signature of a constructed loader/dataset.

    Exposed so an operator can read the signature a protocol needs to authorise
    for :data:`COHORT_ROLE_INDEPENDENT_EVALUATION` off the real objects rather
    than transcribing it from an artifact.
    """

    descriptor = split_membership_descriptor(loader, split_name=str(split_name))
    return str(descriptor["membership_signature"])


__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "CALIBRATION_LINEAGE_SCHEMA_VERSION",
    "CALIBRATION_COHORT_FIELDS",
    "COHORT_ROLES",
    "COHORT_ROLE_CALIBRATION",
    "COHORT_ROLE_INDEPENDENT_EVALUATION",
    "CONDITIONAL_IDENTITY_FIELDS",
    "CalibrationLineageError",
    "CohortPolicy",
    "CohortRoleError",
    "IDENTITY_FIELDS",
    "LineageMismatchError",
    "MissingLineageError",
    "RECORDED_NOT_COMPARED",
    "REQUIRED_PRESENT",
    "build_expected_lineage",
    "inspect_legacy_calibration",
    "runtime_membership_signature",
    "validate_calibration_header",
    "validate_header_lineage_consistency",
    "validate_lineage_completeness",
    "verify_calibration_lineage",
    "verify_source_identity",
]
