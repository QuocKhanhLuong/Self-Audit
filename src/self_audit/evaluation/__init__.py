"""Patient-volume inference and evaluation utilities for Self-Audit.

Three metric spaces are exposed and must never be conflated (the constants
live in :mod:`self_audit.audit.semantics`):

``slice_proxy``
    2-D per-slice Dice on the network grid.  Training/monitoring proxy only.
``volume_resized``
    3-D per-volume Dice on the resized network grid (256x256 in-plane).
``volume_native``
    3-D per-volume Dice after inverse-mapping to the original acquisition
    grid.  Requires genuine native ground truth; it is *not* derivable from
    ``preprocessed_data/``, whose masks were destructively downsampled.

Note also that ``annotation_metrics`` reports surface-distance units under
``distance_space`` (``physical`` / ``pixel``), an axis independent of
``metric_space``.
"""

from .metrics import (
    acceptance_metrics,
    annotation_metrics,
    binary_auprc,
    binary_auroc,
    dice_score,
    f1_for_label,
    per_class_dice,
    per_class_precision_recall,
    slice_proxy_dice,
    surface_metrics,
    transition_audit_metrics,
)
from .transition_accumulator import TransitionMetricAccumulator
from .cohort import evaluate_volume_cohort
from .contracts import (
    AUDIT_TARGET_LEGACY_ONE_V1_CONTRACT,
    ContractMismatchError,
    FOREGROUND_DICE_EXCLUDE_V1_CONTRACT,
    FOREGROUND_DICE_VOLUME_NATIVE_V1_CONTRACT,
    FOREGROUND_DICE_VOLUME_RESIZED_V1_CONTRACT,
    MetricContract,
    StateScore,
    SufficientStatistics,
    TransitionScore,
    compute_batch_sufficient_statistics,
    compute_dice_from_stats,
    compute_sufficient_statistics,
    resolve_metric_contract,
    score_state,
    score_transition,
    score_volume_from_stats,
    validate_contract,
    validate_contract_compatibility,
)
from .threshold import (
    CACHE_SCHEMA_VERSION,
    SUPPORTED_CACHE_SCHEMA_VERSIONS,
    NoFeasibleThresholdError,
    evaluate_threshold,
    json_safe,
    load_calibration,
    save_calibration,
    select_threshold,
    sweep_thresholds,
    validate_and_normalize_transition_cache,
)
from .volume_inference import (
    COMPARISON_MODES,
    VolumeInferenceResult,
    evaluate_comparison_modes,
    evaluate_volume_native,
    infer_patient_volume,
    per_class_dice_with_policy,
    reconstruct_volume,
    resolve_class_names,
    split_cases_by_phase,
    to_native_geometry,
)
from .audit_decomposition import (
    AuditModeSamples,
    StageTransition,
    annotation_headroom_metrics,
    audit_mode_metrics,
    decompose_annotation_output,
    decompose_self_audit_output,
    evaluate_annotation_headroom,
    evaluate_audit_decomposition,
    evaluate_audit_modes_batch,
    extract_stage_transitions,
    probe_gt_leakage,
    stage_transition_metrics,
)

from .visualizer import (
    log_figures_to_wandb,
    plot_patient_volume_qa,
    plot_phase_a_samples,
    plot_phase_b_transitions,
    plot_phase_c_audit_trace,
    render_difference_map,
    render_overlay,
    render_transition_overlay,
    save_figure,
)

__all__ = [
    "AUDIT_TARGET_LEGACY_ONE_V1_CONTRACT",
    "CACHE_SCHEMA_VERSION",
    "COMPARISON_MODES",
    "AuditModeSamples",
    "ContractMismatchError",
    "FOREGROUND_DICE_EXCLUDE_V1_CONTRACT",
    "FOREGROUND_DICE_VOLUME_NATIVE_V1_CONTRACT",
    "FOREGROUND_DICE_VOLUME_RESIZED_V1_CONTRACT",
    "MetricContract",
    "NoFeasibleThresholdError",
    "SUPPORTED_CACHE_SCHEMA_VERSIONS",
    "StageTransition",
    "StateScore",
    "SufficientStatistics",
    "TransitionScore",
    "VolumeInferenceResult",
    "acceptance_metrics",
    "annotation_headroom_metrics",
    "annotation_metrics",
    "audit_mode_metrics",
    "binary_auprc",
    "binary_auroc",
    "compute_batch_sufficient_statistics",
    "compute_dice_from_stats",
    "compute_sufficient_statistics",
    "decompose_annotation_output",
    "decompose_self_audit_output",
    "dice_score",
    "evaluate_annotation_headroom",
    "evaluate_audit_decomposition",
    "evaluate_audit_modes_batch",
    "evaluate_comparison_modes",
    "evaluate_threshold",
    "evaluate_volume_native",
    "extract_stage_transitions",
    "f1_for_label",
    "infer_patient_volume",
    "json_safe",
    "load_calibration",
    "log_figures_to_wandb",
    "per_class_dice",
    "per_class_dice_with_policy",
    "per_class_precision_recall",
    "plot_patient_volume_qa",
    "plot_phase_a_samples",
    "plot_phase_b_transitions",
    "plot_phase_c_audit_trace",
    "probe_gt_leakage",
    "reconstruct_volume",
    "render_difference_map",
    "render_overlay",
    "render_transition_overlay",
    "resolve_class_names",
    "resolve_metric_contract",
    "save_calibration",
    "save_figure",
    "score_state",
    "score_transition",
    "score_volume_from_stats",
    "select_threshold",
    "slice_proxy_dice",
    "split_cases_by_phase",
    "stage_transition_metrics",
    "surface_metrics",
    "sweep_thresholds",
    "to_native_geometry",
    "transition_audit_metrics",
    "TransitionMetricAccumulator",
    "evaluate_volume_cohort",
    "validate_and_normalize_transition_cache",
    "validate_contract",
    "validate_contract_compatibility",
]
