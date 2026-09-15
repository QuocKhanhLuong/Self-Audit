"""Evaluation package for the mask-free pipeline (W6).

Two strictly separated halves:

* **Image-only** (:mod:`metrics` helpers, :mod:`partitions`, :mod:`verification`,
  :mod:`freeze`). Importable from anywhere in the pipeline, including the
  trainer, because nothing in them can read a manual mask.
* **Reference** (:mod:`self_audit_maskfree.evaluation.reference`). Deliberately
  *not* imported here. It is the only module allowed to load manual masks, it
  runs as a standalone post-freeze process, and importing it from this package
  root would put a mask-reading code path one attribute access away from the
  trainer.

Import it explicitly - ``from self_audit_maskfree.evaluation import reference``
- and only from the isolated evaluator entry point.
"""
from __future__ import annotations

from .freeze import (
    FreezeValidationError,
    load_freeze_manifest,
    sha256_file,
    validate_freeze_manifest,
)
from .metrics import (
    COVERAGE_LEVELS,
    FOREGROUND_CLASSES,
    MetricContractError,
    MetricRow,
    coverage_error_curve,
    dice_iou_volume,
    edit_attribution,
    evidence_quality_association,
    metric_row,
    paired_patient_bootstrap,
    patient_macro,
    surface_metrics,
    write_csv,
    write_json,
    write_markdown,
)
from .partitions import (
    all_background_hypothesis,
    class_permutation_hypothesis,
    excessive_partition_hypothesis,
    intensity_grouping_hypothesis,
    random_mask_hypothesis,
)
from .verification import (
    VerificationReuseError,
    reset_verification_registry,
    verify_student_coverage,
    verification_registry_state,
    verify_frozen_bank,
)

__all__ = [
    "COVERAGE_LEVELS",
    "FOREGROUND_CLASSES",
    "FreezeValidationError",
    "MetricContractError",
    "MetricRow",
    "VerificationReuseError",
    "all_background_hypothesis",
    "class_permutation_hypothesis",
    "coverage_error_curve",
    "dice_iou_volume",
    "edit_attribution",
    "evidence_quality_association",
    "excessive_partition_hypothesis",
    "intensity_grouping_hypothesis",
    "load_freeze_manifest",
    "metric_row",
    "paired_patient_bootstrap",
    "patient_macro",
    "random_mask_hypothesis",
    "reset_verification_registry",
    "sha256_file",
    "surface_metrics",
    "validate_freeze_manifest",
    "verify_student_coverage",
    "verification_registry_state",
    "verify_frozen_bank",
    "write_csv",
    "write_json",
    "write_markdown",
]
