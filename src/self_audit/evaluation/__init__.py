"""Patient-volume inference and evaluation utilities for Self-Audit."""

from .metrics import acceptance_metrics, annotation_metrics, dice_score, transition_audit_metrics
from .threshold import evaluate_threshold, select_threshold, sweep_thresholds
from .volume_inference import (
    COMPARISON_MODES,
    evaluate_comparison_modes,
    infer_patient_volume,
    reconstruct_volume,
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
    "COMPARISON_MODES",
    "acceptance_metrics",
    "annotation_metrics",
    "dice_score",
    "evaluate_comparison_modes",
    "infer_patient_volume",
    "reconstruct_volume",
    "transition_audit_metrics",
    "evaluate_threshold",
    "select_threshold",
    "sweep_thresholds",
    "render_overlay",
    "render_transition_overlay",
    "render_difference_map",
    "plot_phase_a_samples",
    "plot_phase_b_transitions",
    "plot_phase_c_audit_trace",
    "plot_patient_volume_qa",
    "save_figure",
    "log_figures_to_wandb",
]
