"""Counterfactual audit targets, generators, and threshold gate."""

from .counterfactual import CounterfactualGenerator, CounterfactualSample, generate_counterfactual
from .gate import GateDecision, ThresholdGate, accept_reject, threshold_accept
from .targets import (
    FIX,
    REGRESS,
    UNCHANGED,
    LOCAL_AUDIT_NAMES,
    TransitionTargets,
    build_local_audit_targets,
    build_transition_targets,
    delta_dice_target,
    local_audit_targets,
)

__all__ = [
    "CounterfactualGenerator",
    "CounterfactualSample",
    "FIX",
    "REGRESS",
    "UNCHANGED",
    "LOCAL_AUDIT_NAMES",
    "GateDecision",
    "ThresholdGate",
    "TransitionTargets",
    "accept_reject",
    "build_local_audit_targets",
    "build_transition_targets",
    "delta_dice_target",
    "generate_counterfactual",
    "local_audit_targets",
    "threshold_accept",
]
