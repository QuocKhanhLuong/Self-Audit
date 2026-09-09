#!/usr/bin/env python3
"""Checkpoint diagnostics CLI: the primary research instrument for Self-Audit.

This script does not train, does not modify the model, and does not select a
threshold.  It *measures* an existing checkpoint and emits exactly one
machine-readable JSON document with a stable, versioned schema.

What it reports
---------------
1. Global comparison modes (``initial_only`` / ``always_accept_refinement`` /
   ``self_audit`` / ``oracle_accept``) and the headroom attribution between
   them.
2. Per-stage transition attribution (candidate gain, gate value, capture and
   block rates, attribution residual).
3. Patient-volume Dice per class (RV/MYO/LV), per patient, per cohort, with ED
   and ES reported separately.
4. Three research probes:

   * **Probe 1 - synthetic positive quality.**  Measures the counterfactual
     generator's ``kind="positive"`` output around every state of the real
     checkpoint's ``forward_annotation`` trajectory.  Generator behaviour is
     not modified; this only asks whether the "positive counterfactuals are
     degenerate" finding reproduces on a trained checkpoint.
   * **Probe 2 - audit-evidence conditioning value.**  Phase A and Phase B run
     the Annotation Expert with ``previous_audit_evidence=None``; inference
     feeds real evidence.  This probe runs the expert twice on the *same*
     active state - once with the real evidence, once with zeros - and reports
     how much those three channels change the candidate.  Both candidates are
     generated GT-free; ground truth enters only afterwards, to score two
     already-computed candidates.
   * **Probe 3 - GT firewall.**  Permuting the oracle target must not change
     deployable ``self_audit`` inference.  Checked over several batches, with
     a strict pass criterion (exact zero, no tolerance).

Evidence class
--------------
Every report is stamped ``evidence_class="diagnostic_only"``.  There is no
independent test set in this project: the checkpoint was trained on the
existing 80/20 patient split, and any threshold reported here was selected on
the same split it is measured on.  Nothing this CLI emits is held-out
evidence.

Threshold precedence
--------------------
``--tau_accept`` > ``--calibration`` > config ``audit.tau_accept`` > ``0.0``.
The resolved value and its source are echoed into the JSON as ``tau_accept``
and ``tau_accept_source``.  When ``--calibration`` is supplied the tau actually
handed to inference is asserted equal to the artifact's saved value, and the
artifact's ``neutral_margin`` must agree with the margin being evaluated with
(a disagreement is a hard error, not a warning).

Calibration lineage
-------------------
Whenever ``--calibration`` is supplied the artifact's lineage is verified
against an expectation built from the objects this run actually constructed --
the bound checkpoint, the live model, the constructed validation dataset and
the protocol's metric semantics -- *before* any tau is used and before any
calibrated number is emitted.  The expectation is never read out of the
artifact, so an artifact cannot be its own witness.  ``--tau_accept`` does not
skip this: an override changes which number is reported, not whether a
consulted artifact has to be valid.  An artifact without a lineage block is
refused; ``self_audit.evaluation.calibration_lineage.inspect_legacy_calibration``
reads it as explicitly unverified instead.

The cohort the threshold was calibrated on and the cohort this run evaluates
are separate roles.  ``--cohort_role calibration`` (the default) requires them
to be the same cohort.  ``--cohort_role independent_evaluation`` requires the
protocol to name the permitted membership with
``--authorized_cohort_signature``, and that cohort must be patient-disjoint
from the calibration cohort.  There is no mode that accepts an unnamed
replacement cohort.

NaN handling
------------
``nan`` is a legitimate, meaningful value here (for example
``headroom_capture_ratio`` when no row carried material headroom).  It is
never coerced to ``0.0``.  In JSON it serialises as ``null`` *and* a sibling
key ``"<key>__is_nan": true`` is emitted, so a reader cannot mistake it for a
missing key.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.audit.counterfactual import CounterfactualGenerator
from self_audit.audit.semantics import (
    BENEFICIAL,
    DEFAULT_EMPTY_CLASS_POLICY,
    HARMFUL,
    METRIC_SPACE_SLICE_PROXY,
    METRIC_SPACE_VOLUME_NATIVE,
    METRIC_SPACE_VOLUME_RESIZED,
    NEUTRAL,
    classify_delta,
    macro_mean,
    resolve_empty_policy,
    resolve_neutral_margin,
)
from self_audit.audit.targets import multiclass_dice
from self_audit.evaluation.audit_decomposition import (
    evaluate_annotation_headroom,
    evaluate_audit_decomposition,
)
from self_audit.evaluation.calibration_lineage import (
    CALIBRATION_LINEAGE_SCHEMA_VERSION,
    COHORT_ROLES,
    COHORT_ROLE_CALIBRATION,
    COHORT_ROLE_INDEPENDENT_EVALUATION,
    CalibrationLineageError,
    CohortPolicy,
    build_expected_lineage,
    verify_calibration_lineage,
)
from self_audit.evaluation.threshold import CALIBRATION_SCHEMA_VERSION, load_calibration
from self_audit.evaluation.cohort import evaluate_volume_cohort
from self_audit.evaluation.volume_inference import (
    COMPARISON_MODES,
    evaluate_comparison_modes,
    resolve_class_names,
    split_cases_by_phase,
)
from self_audit.training.unified_config import resolve_downstream_config
from self_audit.training._utils import (
    bind_evaluation_checkpoint,
    build_data_loader,
    build_model_from_config,
    build_patient_dataset,
    load_config,
    move_batch,
    resolve_device,
    validate_dataset_splits,
    verify_bound_state,
)


# --------------------------------------------------------------------------
# Schema constants
# --------------------------------------------------------------------------

#: Bumped whenever the emitted JSON changes shape in a non-additive way.
DIAGNOSTIC_SCHEMA_VERSION = 1

#: The only evidence class this pass is allowed to claim.
EVIDENCE_CLASS_DIAGNOSTIC_ONLY = "diagnostic_only"

EVIDENCE_REASON = (
    "This checkpoint was trained on the existing 80/20 patient split; no independent test set "
    "exists in this project, and tau was selected on the same split it is reported on. Every "
    "number in this document is therefore an optimistically-biased in-sample diagnostic, not "
    "held-out evidence, and must not be quoted as a benchmark result."
)

#: The audit-decomposition mode Dice comes from
#: ``self_audit.audit.targets.multiclass_dice``, which scores a class that is
#: empty in both prediction and target as 1.0.  ``--empty_policy`` does not
#: reach it; it governs the volume blocks only.  Stated in the JSON rather than
#: silently implied.
MODE_DICE_EMPTY_POLICY = "legacy_one"

TAU_SOURCE_CLI = "cli_tau_accept"
TAU_SOURCE_CALIBRATION = "calibration_artifact"
TAU_SOURCE_CONFIG = "config_audit_tau_accept"
TAU_SOURCE_DEFAULT = "default_zero"


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def json_safe(value: Any) -> Any:
    """Recursively convert to JSON, marking every non-finite float explicitly.

    A ``nan`` dict value serialises as ``null`` plus a sibling
    ``"<key>__is_nan": true``; ``+/-inf`` serialises as ``null`` plus
    ``"<key>__is_inf": true``.  Non-finite values inside a *list* become
    ``null`` with no sibling (a list has no key to hang one on), so metrics
    that can be ``nan`` are always emitted as named dict entries.
    """

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            converted = json_safe(raw_value)
            out[key] = converted
            if isinstance(raw_value, (float, np.floating)):
                number = float(raw_value)
                if math.isnan(number):
                    out[f"{key}__is_nan"] = True
                elif math.isinf(number):
                    out[f"{key}__is_inf"] = True
        return out
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return json_safe(value.item())
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def _file_sha256(path: Any) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with open(candidate, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    head = completed.stdout.strip()
    return head or None


# --------------------------------------------------------------------------
# Threshold resolution
# --------------------------------------------------------------------------


def resolve_tau_accept(
    *,
    cli_tau: float | None,
    calibration_path: str | None,
    config_audit: Mapping[str, Any],
    neutral_margin: float,
    allow_tau_override: bool = False,
    expected_lineage: Mapping[str, Any] | None = None,
    cohort_policy: CohortPolicy | None = None,
) -> dict[str, Any]:
    """Resolve ``tau_accept`` under the documented precedence.

    Precedence: explicit ``--tau_accept`` > ``--calibration`` (via
    :func:`load_calibration`) > config ``audit.tau_accept`` > ``0.0``.

    When a calibration artifact is supplied it is loaded and validated even if
    an explicit ``--tau_accept`` outranks it, because the artifact's
    ``neutral_margin`` must agree with the margin being evaluated with.  A
    disagreement is a hard error.  An explicit ``--tau_accept`` that
    contradicts the artifact is also a hard error unless
    ``allow_tau_override`` is set: silently ignoring a calibrated threshold
    while stamping ``calibration_path`` into the report would misrepresent
    which threshold produced the numbers.

    ``expected_lineage`` is the runtime-derived expectation from
    :func:`build_expected_lineage` and is **required** whenever
    ``calibration_path`` is given.  This function fails closed: it does not
    return a usable tau alongside an unverified artifact, because a caller that
    forgot to verify would otherwise get a number that looks calibrated.  The
    artifact's lineage is checked against the runtime before the resolved tau
    leaves here.  ``allow_tau_override`` does not reach that check -- an
    override decides which number is reported, never whether a consulted
    artifact is valid.  To read an artifact without verifying it, use
    :func:`self_audit.evaluation.calibration_lineage.inspect_legacy_calibration`,
    which never yields a usable threshold.
    """

    calibration: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    if calibration_path is not None:
        if expected_lineage is None:
            raise CalibrationLineageError(
                f"Calibration artifact {calibration_path} was supplied without a runtime-derived "
                "expected lineage. Resolving a threshold from an artifact that has not been "
                "checked against this run's checkpoint, model, preprocessing and cohort is "
                "refused. Build one with build_expected_lineage(), or read the artifact with "
                "inspect_legacy_calibration(), which is explicitly unverified."
            )
        calibration = load_calibration(calibration_path)
        artifact_margin = float(calibration["neutral_margin"])
        if artifact_margin != neutral_margin:
            raise ValueError(
                f"Calibration artifact {calibration_path} was produced at neutral_margin="
                f"{artifact_margin!r} but this run evaluates at neutral_margin={neutral_margin!r}. "
                "A threshold calibrated under a different decision margin does not transfer; "
                "re-run calibration at the evaluation margin or evaluate at the calibrated one."
            )
        verification = verify_calibration_lineage(
            calibration,
            expected_lineage,
            cohort_policy=cohort_policy,
            artifact_name=f"Calibration artifact {calibration_path}",
        )

    if cli_tau is not None:
        tau = float(cli_tau)
        source = TAU_SOURCE_CLI
        if calibration is not None and tau != float(calibration["tau_accept"]):
            if not allow_tau_override:
                raise ValueError(
                    f"--tau_accept={tau!r} contradicts the calibration artifact "
                    f"{calibration_path} (tau_accept={calibration['tau_accept']!r}). Drop one of "
                    "the two, or pass --allow_tau_override to deliberately report an "
                    "uncalibrated threshold alongside the artifact."
                )
            source = "cli_tau_accept_override_of_calibration"
    elif calibration is not None:
        tau = float(calibration["tau_accept"])
        source = TAU_SOURCE_CALIBRATION
    elif config_audit.get("tau_accept") is not None:
        tau = float(config_audit["tau_accept"])
        source = TAU_SOURCE_CONFIG
    else:
        tau = 0.0
        source = TAU_SOURCE_DEFAULT

    resolution: dict[str, Any] = {
        "tau_accept": tau,
        "tau_accept_source": source,
        "calibration_path": None if calibration_path is None else str(calibration_path),
        "calibration_schema_version": None if calibration is None else int(calibration["schema_version"]),
        "calibration_tau_accept": None if calibration is None else float(calibration["tau_accept"]),
        "calibration_neutral_margin": None if calibration is None else float(calibration["neutral_margin"]),
        "calibration_source_split": None if calibration is None else calibration.get("source_split"),
        "calibration_checkpoint_path": None if calibration is None else calibration.get("checkpoint_path"),
        "calibration_checkpoint_sha256": None if calibration is None else calibration.get("checkpoint_sha256"),
        "calibration_metric_space": None if calibration is None else calibration.get("metric_space"),
        "calibration_validity": None if calibration is None else calibration.get("validity"),
        "expected_calibration_schema_version": int(CALIBRATION_SCHEMA_VERSION),
        "expected_calibration_lineage_schema_version": int(CALIBRATION_LINEAGE_SCHEMA_VERSION),
        "lineage_verified": None if calibration is None else True,
        "lineage_verification": verification,
        "precedence": "--tau_accept > --calibration > config audit.tau_accept > 0.0",
    }
    return resolution


def assert_calibration_lineage_verified(resolution: Mapping[str, Any]) -> None:
    """Refuse to emit a calibrated result that was never checked against this run.

    This is the emission-side gate.  It fires whenever an artifact was
    consulted at all -- including under ``--tau_accept`` override, where the
    artifact still appears in the report and would otherwise lend it
    unearned authority.
    """

    if resolution.get("calibration_path") is None:
        return
    if resolution.get("lineage_verified") is True:
        return
    raise CalibrationLineageError(
        f"Calibration artifact {resolution.get('calibration_path')} was consulted but its "
        "lineage was never verified against this run's checkpoint, model, preprocessing and "
        "cohort. Refusing to emit a calibrated result."
    )


def assert_calibrated_tau_used(resolution: Mapping[str, Any], tau_used: float) -> None:
    """Hard-assert that inference ran at the calibrated tau, when one was given."""

    if resolution.get("calibration_path") is None:
        return
    if resolution["tau_accept_source"] != TAU_SOURCE_CALIBRATION:
        return
    saved = float(resolution["calibration_tau_accept"])
    if float(tau_used) != saved:
        raise AssertionError(
            f"tau handed to inference ({tau_used!r}) is not the calibrated value ({saved!r})"
        )


# --------------------------------------------------------------------------
# Probe 1 - synthetic positive counterfactual quality
# --------------------------------------------------------------------------


@torch.no_grad()
def probe_synthetic_positive_quality(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    generator: CounterfactualGenerator,
    neutral_margin: float,
    num_classes: int = 4,
    max_batches: int | None = None,
    high_confidence: float = 0.95,
) -> dict[str, float]:
    """Measure ``kind="positive"`` counterfactuals on the real trajectory.

    The generator is used exactly as training uses it - nothing about its
    behaviour is changed here.  For every state of ``forward_annotation``'s
    trajectory (``A0`` and each refinement state) one positive counterfactual
    batch is generated, and its realised ``delta_dice`` is classified with the
    canonical :func:`classify_delta` margin.

    ``positive_argmax_change_rate`` is the fraction of *valid* generated
    samples whose candidate argmax differs from the previous argmax in at least
    one pixel; its complement is the "zero-pixel edit" rate the audit reported.
    The mean ``p_max`` of the states is reported alongside so the confidence
    regime the generator was operating in is on the record.
    """

    was_training = model.training
    model.eval()

    generated = 0
    valid = 0
    changed_rows = 0
    changed_pixel_fractions: list[float] = []
    deltas: list[np.ndarray] = []
    pmax_sums = 0.0
    pmax_count = 0
    high_conf_pixels = 0
    total_pixels = 0
    states_visited = 0

    try:
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_batch(raw_batch, device)
            images = batch["image"]
            target = batch["mask"]
            output = model.forward_annotation(images)
            trace = [value for value in output.get("state_trace", []) if torch.is_tensor(value)]
            for state in trace:
                states_visited += 1
                probs = state.detach().softmax(dim=1)
                pmax = probs.max(dim=1).values
                pmax_sums += float(pmax.sum().item())
                pmax_count += int(pmax.numel())
                high_conf_pixels += int((pmax >= float(high_confidence)).sum().item())
                total_pixels += int(pmax.numel())

                sample = generator.generate(state.detach(), target, kind="positive")
                rows = int(sample.previous_probs.shape[0])
                generated += rows
                valid_mask = (
                    sample.valid_mask
                    if sample.valid_mask is not None
                    else torch.ones(rows, dtype=torch.bool, device=state.device)
                )
                valid_count = int(valid_mask.sum().item())
                valid += valid_count
                if valid_count == 0:
                    continue
                previous_labels = sample.previous_probs.argmax(dim=1)
                candidate_labels = sample.candidate_probs.argmax(dim=1)
                differing = (candidate_labels != previous_labels).flatten(1)
                row_changed = differing.any(dim=1)
                changed_rows += int((row_changed & valid_mask).sum().item())
                fractions = differing.float().mean(dim=1)
                changed_pixel_fractions.extend(
                    float(value) for value in fractions[valid_mask].detach().cpu().tolist()
                )
                delta = sample.actual_delta_dice
                if delta is None:
                    delta = multiclass_dice(
                        sample.candidate_probs, target, num_classes
                    ) - multiclass_dice(sample.previous_probs, target, num_classes)
                deltas.append(delta[valid_mask].detach().cpu().numpy().astype(np.float64))
    finally:
        model.train(was_training)

    all_deltas = np.concatenate(deltas) if deltas else np.zeros((0,), dtype=np.float64)
    codes = classify_delta(all_deltas, neutral_margin) if all_deltas.size else np.zeros((0,), dtype=np.int64)

    def _rate(count: int, total: int) -> float:
        return float(count) / float(total) if total > 0 else float("nan")

    return {
        "positive_generated_count": float(generated),
        "positive_valid_count": float(valid),
        "positive_invalid_count": float(generated - valid),
        "positive_states_visited": float(states_visited),
        "positive_argmax_change_rate": _rate(changed_rows, valid),
        "positive_zero_pixel_edit_rate": (
            float("nan") if valid == 0 else 1.0 - _rate(changed_rows, valid)
        ),
        "positive_mean_changed_pixel_fraction": (
            float(np.mean(changed_pixel_fractions)) if changed_pixel_fractions else float("nan")
        ),
        "positive_beneficial_rate": _rate(int((codes == BENEFICIAL).sum()), int(codes.size)),
        "positive_neutral_rate": _rate(int((codes == NEUTRAL).sum()), int(codes.size)),
        "positive_harmful_rate": _rate(int((codes == HARMFUL).sum()), int(codes.size)),
        "positive_mean_delta_dice": float(all_deltas.mean()) if all_deltas.size else float("nan"),
        "positive_median_delta_dice": float(np.median(all_deltas)) if all_deltas.size else float("nan"),
        "positive_state_mean_p_max": (
            pmax_sums / float(pmax_count) if pmax_count else float("nan")
        ),
        "positive_state_high_confidence_pixel_rate": _rate(high_conf_pixels, total_pixels),
        "positive_high_confidence_threshold": float(high_confidence),
        "positive_rate_denominator": "valid_generated_samples",
        "positive_neutral_margin": float(neutral_margin),
        "positive_generator_epsilon_neutral": float(generator.epsilon_neutral),
    }


# --------------------------------------------------------------------------
# Probe 2 - audit-evidence conditioning value
# --------------------------------------------------------------------------


def _reconstruct_previous_audit_evidence(audit: Mapping[str, Any]) -> Tensor:
    """Rebuild the recurrent evidence tensor the model feeds into the next turn.

    ``SelfAuditNet.infer`` computes, for turn ``t``::

        local_evidence = softmax(local_logits_t)
        previous_audit = where(accepted_t, local_evidence, 0)

    and hands that to the expert at turn ``t+1``.  This reproduces the formula
    exactly; :func:`probe_audit_evidence_value` verifies the reconstruction by
    re-running the expert and comparing against the candidate the model itself
    produced.
    """

    local_logits = audit["local_logits"]
    accepted = audit["accepted"].to(torch.bool)
    evidence = local_logits.detach().softmax(dim=1)
    return torch.where(accepted[:, None, None, None], evidence, torch.zeros_like(evidence))


@torch.no_grad()
def probe_audit_evidence_value(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    tau_accept: float,
    t_max: int,
    trajectory: str = "self_audit",
    num_classes: int = 4,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Does the recurrent audit-evidence input change the Expert's candidate?

    Phase A (``forward_annotation``) and Phase B never reassign
    ``previous_audit_evidence``; it stays ``None``, so the three audit channels
    are all-zero throughout those phases.  Inference *does* feed real evidence.
    This probe asks whether, after Phase C, the trained Expert responds to it
    at all.

    GT DISCIPLINE (the reason this function is written in three explicit
    steps): both candidates are produced GT-free.  Step 1 runs deployable
    inference with no ground-truth argument.  Step 2 re-runs the Annotation
    Expert twice on the identical active state, differing only in the evidence
    tensor - still no ground truth anywhere.  Only step 3 touches ``mask``, and
    only to score two candidates that already exist.  An inert result is a
    finding, not a failure.
    """

    if trajectory not in {"self_audit", "always_accept_refinement"}:
        raise ValueError(f"trajectory must be self_audit or always_accept_refinement, got {trajectory!r}")

    was_training = model.training
    model.eval()

    logit_l1_sum = 0.0
    logit_element_count = 0
    logit_max_abs = 0.0
    pixel_changes = 0
    pixel_count = 0
    row_changes = 0
    rows_measured = 0
    stages_measured = 0
    dice_deltas: list[np.ndarray] = []
    reconstruction_max_abs = 0.0
    evidence_mass_sum = 0.0

    try:
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_batch(raw_batch, device)
            images = batch["image"]

            # ---- STEP 1 (GT-FREE): deployable inference, no ground truth. ----
            output = model.infer(
                images,
                mode=trajectory,
                tau_accept=float(tau_accept),
                t_max=int(t_max),
            )
            shared = output["shared_features"]
            previous_states = output["transition_previous"]
            candidates = output["transition_candidates"]
            actives = output["transition_active_masks"]
            audits = output["audits"]

            # ---- STEP 2 (GT-FREE): same state, real evidence vs zeros. ----
            for stage in range(1, min(len(previous_states), len(audits) + 1)):
                active = actives[stage].to(torch.bool)
                indices = active.nonzero(as_tuple=False).flatten()
                if indices.numel() == 0:
                    continue
                evidence_full = _reconstruct_previous_audit_evidence(audits[stage - 1])
                evidence = evidence_full.index_select(0, indices)
                shared_active = shared.index_select(0, indices)
                state_active = previous_states[stage].index_select(0, indices)

                real_output = model.annotation_expert(
                    shared_active,
                    state_active,
                    previous_audit_evidence=evidence,
                    turn_index=stage,
                    iteration_index=stage,
                    return_metadata=False,
                )
                zero_output = model.annotation_expert(
                    shared_active,
                    state_active,
                    previous_audit_evidence=torch.zeros_like(evidence),
                    turn_index=stage,
                    iteration_index=stage,
                    return_metadata=False,
                )
                real_logits = real_output.candidate_logits
                zero_logits = zero_output.candidate_logits

                # Self-check: the "real evidence" branch must reproduce the
                # candidate the model itself produced at this stage.  If this
                # is not ~0 the evidence reconstruction is wrong and every
                # number below would be meaningless.
                reference = candidates[stage].index_select(0, indices)
                reconstruction_max_abs = max(
                    reconstruction_max_abs,
                    float((real_logits - reference).abs().max().item()),
                )

                difference = (real_logits - zero_logits).abs()
                logit_l1_sum += float(difference.sum().item())
                logit_element_count += int(difference.numel())
                logit_max_abs = max(logit_max_abs, float(difference.max().item()))
                evidence_mass_sum += float(evidence.abs().sum().item())

                real_labels = real_logits.argmax(dim=1)
                zero_labels = zero_logits.argmax(dim=1)
                differing = real_labels != zero_labels
                pixel_changes += int(differing.sum().item())
                pixel_count += int(differing.numel())
                row_changes += int(differing.flatten(1).any(dim=1).sum().item())
                rows_measured += int(indices.numel())
                stages_measured += 1

                # ---- STEP 3: ground truth enters HERE and only here, to
                # score two candidates that were already computed above. ----
                target_active = batch["mask"].index_select(0, indices)
                real_dice = multiclass_dice(real_logits, target_active, num_classes)
                zero_dice = multiclass_dice(zero_logits, target_active, num_classes)
                dice_deltas.append((real_dice - zero_dice).detach().cpu().numpy().astype(np.float64))
    finally:
        model.train(was_training)

    all_dice_deltas = (
        np.concatenate(dice_deltas) if dice_deltas else np.zeros((0,), dtype=np.float64)
    )
    inert = (
        rows_measured > 0
        and logit_max_abs == 0.0
        and pixel_changes == 0
    )
    return {
        "audit_evidence/candidate_logit_l1": (
            logit_l1_sum / float(logit_element_count) if logit_element_count else float("nan")
        ),
        "audit_evidence/candidate_logit_l1_sum": float(logit_l1_sum),
        "audit_evidence/candidate_logit_max_abs_diff": float(logit_max_abs),
        "audit_evidence/argmax_change_rate": (
            float(pixel_changes) / float(pixel_count) if pixel_count else float("nan")
        ),
        "audit_evidence/row_argmax_change_rate": (
            float(row_changes) / float(rows_measured) if rows_measured else float("nan")
        ),
        "audit_evidence/dice_delta_real_vs_zero": (
            float(all_dice_deltas.mean()) if all_dice_deltas.size else float("nan")
        ),
        "audit_evidence/rows_measured": float(rows_measured),
        "audit_evidence/stages_measured": float(stages_measured),
        "audit_evidence/mean_evidence_abs_mass": (
            evidence_mass_sum / float(logit_element_count) if logit_element_count else float("nan")
        ),
        "audit_evidence/reconstruction_max_abs_diff": float(reconstruction_max_abs),
        "audit_evidence/trajectory": trajectory,
        "audit_evidence/gt_free_candidate_generation": True,
        "audit_evidence/branch_inert": bool(inert),
        "audit_evidence/note": (
            "Both candidates are generated GT-free; ground truth is used only to score two "
            "already-computed candidates. rows_measured==0 means the trajectory never reached a "
            "stage with real (non-zero) recurrent evidence -- typically every row halted at "
            "stage 0 -- which is itself a finding about the audit gate, not a probe failure."
        ),
    }


# --------------------------------------------------------------------------
# Probe 3 - GT firewall beyond batch 0
# --------------------------------------------------------------------------


def strict_gt_firewall(
    aggregate: Mapping[str, float],
    *,
    requested_batches: int,
) -> dict[str, Any]:
    """Re-derive the firewall verdict under a strict, tolerance-free criterion.

    ``evaluate_audit_decomposition`` aggregates the per-batch probes and emits
    its own ``gt_firewall/passed`` using a ``<= 1e-7`` tolerance.  This report
    does not soften the criterion: a deployable path that is GT-independent
    produces bitwise-identical logits, so anything above exact zero is a
    failure worth seeing.  Both verdicts are emitted, the strict one as the
    headline.
    """

    checked = int(float(aggregate.get("gt_firewall/num_batches_checked", 0.0)))
    max_abs = float(aggregate.get("gt_firewall/max_abs_logit_diff", float("nan")))
    mismatch_rate = float(aggregate.get("gt_firewall/decision_mismatch_rate", float("nan")))
    passed = bool(checked >= 1 and max_abs == 0.0 and mismatch_rate == 0.0)
    return {
        "gt_firewall/num_batches_requested": float(requested_batches),
        "gt_firewall/num_batches_checked": float(checked),
        "gt_firewall/max_abs_logit_diff": max_abs,
        "gt_firewall/decision_mismatch_count": float(
            aggregate.get("gt_firewall/decision_mismatch_count", float("nan"))
        ),
        "gt_firewall/decision_total": float(aggregate.get("gt_firewall/decision_total", float("nan"))),
        "gt_firewall/decision_mismatch_rate": mismatch_rate,
        "gt_firewall/passed": passed,
        "gt_firewall/criterion": (
            "num_batches_checked >= 1 AND max_abs_logit_diff == 0.0 AND "
            "decision_mismatch_rate == 0.0 (exact, no tolerance)"
        ),
        "gt_firewall/upstream_tolerant_passed": bool(
            float(aggregate.get("gt_firewall/passed", 0.0)) == 1.0
        ),
        "gt_firewall/upstream_tolerance": 1e-7,
    }


# --------------------------------------------------------------------------
# Volume-level reporting
# --------------------------------------------------------------------------


def load_volume_geometry(data_root: Any, split: str | None = None) -> dict[str, dict[str, Any]]:
    """Read declared per-case geometry from ``metadata.json`` if one exists.

    Nothing here is computed or defaulted: geometry that the preprocessing run
    did not record simply stays absent.
    """

    root = Path(str(data_root))
    candidates = [root / "metadata.json"]
    if split:
        candidates.insert(0, root / str(split) / "metadata.json")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        info = payload.get("volume_info") or {}
        if not isinstance(info, dict):
            continue
        return {
            str(case_id): {
                key: value
                for key, value in entry.items()
                if key in {"orig_shape", "orig_spacing", "effective_spacing", "num_slices"}
            }
            for case_id, entry in info.items()
            if isinstance(entry, dict)
        }
    return {}


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def _stage_view(metrics: Mapping[str, float]) -> dict[str, Any]:
    """Regroup the flat ``stage_<i>/...`` keys into a per-stage mapping."""

    stages: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        if not key.startswith("stage_"):
            continue
        name, _, metric = key.partition("/")
        if not metric:
            continue
        stages.setdefault(name, {})[metric] = value
    return {name: stages[name] for name in sorted(stages, key=lambda item: int(item.split("_")[1]))}


REQUIRED_MODE_KEYS = (
    "modes/initial_dice",
    "modes/always_accept_dice",
    "modes/self_audit_dice",
    "modes/oracle_dice",
    "modes/candidate_path_gain",
    "modes/self_audit_gain",
    "modes/oracle_headroom",
    "modes/audit_rescue_vs_always",
    "modes/headroom_capture_ratio",
    "modes/headroom_available_rate",
)

REQUIRED_STAGE_KEYS = (
    "prev_dice",
    "candidate_dice",
    "candidate_gain",
    "positive_headroom",
    "realized_gain",
    "audit_gate_value",
    "attempt_rate",
    "accept_rate",
    "reject_rate",
    "beneficial_candidate_rate",
    "neutral_candidate_rate",
    "harmful_candidate_rate",
    "beneficial_capture_rate",
    "harmful_block_rate",
    "attribution_residual",
)


def build_report(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    checkpoint: str,
    config_path: str,
    split: str,
    resolution: Mapping[str, Any],
    neutral_margin: float,
    empty_policy: str,
    t_max: int,
    metric_space: str,
    max_val_batches: int | None,
    firewall_batches: int,
    probe_batches: int | None,
    evidence_trajectory: str,
    num_classes: int = 4,
    volume: Mapping[str, Any] | None = None,
    disable_tqdm: bool = False,
    generator: CounterfactualGenerator | None = None,
) -> dict[str, Any]:
    """Run every diagnostic and assemble the single versioned JSON payload."""

    tau_accept = float(resolution["tau_accept"])
    # Nothing is measured until the consulted artifact has been proved to
    # describe this run.
    assert_calibration_lineage_verified(resolution)
    assert_calibrated_tau_used(resolution, tau_accept)

    annotation = evaluate_annotation_headroom(
        model,
        loader,
        device,
        neutral_margin=neutral_margin,
        max_batches=max_val_batches,
        disable_tqdm=disable_tqdm,
    )
    audit = evaluate_audit_decomposition(
        model,
        loader,
        device,
        tau_accept=tau_accept,
        t_max=t_max,
        neutral_margin=neutral_margin,
        max_batches=max_val_batches,
        disable_tqdm=disable_tqdm,
        gt_probe_batches=int(firewall_batches),
    )
    firewall = strict_gt_firewall(audit, requested_batches=int(firewall_batches))
    positive = probe_synthetic_positive_quality(
        model,
        loader,
        device,
        generator=generator or CounterfactualGenerator(num_classes=int(num_classes)),
        neutral_margin=neutral_margin,
        num_classes=int(num_classes),
        max_batches=probe_batches,
    )
    evidence = probe_audit_evidence_value(
        model,
        loader,
        device,
        tau_accept=tau_accept,
        t_max=t_max,
        trajectory=evidence_trajectory,
        num_classes=int(num_classes),
        max_batches=probe_batches,
    )

    global_modes = {key: audit.get(key, float("nan")) for key in REQUIRED_MODE_KEYS}
    global_modes.update(
        {
            key: value
            for key, value in audit.items()
            if key.startswith("modes/") and key not in global_modes
        }
    )
    per_stage = _stage_view(audit)
    audit_aggregate = {key: value for key, value in audit.items() if key.startswith("audit/")}

    flat: dict[str, Any] = {}
    flat.update({key: value for key, value in audit.items() if not key.startswith("gt_firewall/")})
    flat.update(firewall)
    flat.update(positive)
    flat.update(evidence)
    flat.update(annotation)

    payload: dict[str, Any] = {
        "diagnostic_schema_version": int(DIAGNOSTIC_SCHEMA_VERSION),
        "evidence_class": EVIDENCE_CLASS_DIAGNOSTIC_ONLY,
        "evidence_reason": EVIDENCE_REASON,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "config_path": str(config_path),
        "split": str(split),
        "tau_accept": tau_accept,
        "tau_accept_source": resolution["tau_accept_source"],
        "neutral_margin": float(neutral_margin),
        "empty_policy": str(empty_policy),
        "t_max": int(t_max),
        "metric_space": str(metric_space),
        "num_classes": int(num_classes),
        "threshold_resolution": dict(resolution),
        "calibration_lineage_verification": resolution.get("lineage_verification"),
        "metric_space_notes": {
            "slice_level": METRIC_SPACE_SLICE_PROXY,
            "volume_level": METRIC_SPACE_VOLUME_RESIZED,
            "native_available": False,
            "mode_dice_empty_policy": MODE_DICE_EMPTY_POLICY,
            "mode_dice_empty_policy_note": (
                "modes/* and stage_*/ Dice come from self_audit.audit.targets.multiclass_dice, "
                "which scores a class empty in both prediction and target as 1.0. --empty_policy "
                "governs the volume blocks only and does not reach these keys."
            ),
        },
        "annotation_headroom": annotation,
        "global_modes": global_modes,
        "per_stage": per_stage,
        "audit_aggregate": audit_aggregate,
        "gt_firewall": firewall,
        "synthetic_positive_quality": positive,
        "audit_evidence": evidence,
        "volume": dict(volume) if volume is not None else None,
        "flat": flat,
    }
    payload["required_keys_present"] = check_required_keys(payload)
    return payload


def check_required_keys(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Self-describing completeness check emitted inside the report itself."""

    missing: list[str] = []
    flat = payload.get("flat", {})
    for key in REQUIRED_MODE_KEYS:
        if key not in flat:
            missing.append(key)
    for stage_name, stage in (payload.get("per_stage") or {}).items():
        for key in REQUIRED_STAGE_KEYS:
            if key not in stage:
                missing.append(f"{stage_name}/{key}")
    for key in (
        "positive_generated_count",
        "positive_valid_count",
        "positive_argmax_change_rate",
        "positive_beneficial_rate",
        "positive_neutral_rate",
        "positive_harmful_rate",
        "positive_mean_delta_dice",
        "audit_evidence/candidate_logit_l1",
        "audit_evidence/argmax_change_rate",
        "audit_evidence/dice_delta_real_vs_zero",
        "gt_firewall/num_batches_checked",
        "gt_firewall/max_abs_logit_diff",
        "gt_firewall/decision_mismatch_rate",
        "gt_firewall/passed",
    ):
        if key not in flat:
            missing.append(key)
    return {"ok": not missing, "missing": missing}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit an existing Self-Audit checkpoint and emit one versioned diagnostic JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "tau_accept precedence (highest first):\n"
            "  1. --tau_accept <float>      explicit command-line value\n"
            "  2. --calibration <path>      schema-v2 artifact, read with load_calibration()\n"
            "  3. config audit.tau_accept   from the YAML config\n"
            "  4. 0.0                       documented fallback\n"
            "The resolved value and its source are echoed into the JSON as tau_accept and\n"
            "tau_accept_source. With --calibration, the artifact's neutral_margin must match\n"
            "the evaluation margin (hard error otherwise), and a contradicting --tau_accept is\n"
            "rejected unless --allow_tau_override is given.\n\n"
            "A consulted --calibration artifact always has its lineage verified against this\n"
            "run's bound checkpoint, live model, preprocessing recipe, metric semantics and\n"
            "cohort before any tau is used or any calibrated number is emitted.\n"
            "--allow_tau_override does not skip that check. An artifact with no lineage block\n"
            "is refused.\n\n"
            "Every report is stamped evidence_class=\"diagnostic_only\": no independent test\n"
            "set exists for this project."
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="configs/self_audit_full.yaml",
        help="Path to unified or historical config YAML (default: configs/self_audit_full.yaml)",
    )
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--tau_accept",
        type=float,
        default=None,
        help="Explicit acceptance threshold; outranks --calibration and the config.",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Path to a schema-v2 calibration artifact written by save_calibration().",
    )
    parser.add_argument(
        "--allow_tau_override",
        action="store_true",
        help="Permit --tau_accept to contradict a supplied --calibration artifact.",
    )
    parser.add_argument(
        "--metric_contract",
        default="foreground_dice_exclude_v1",
        help=(
            "Metric contract this evaluation runs under; part of the expected calibration "
            "lineage (default: foreground_dice_exclude_v1)."
        ),
    )
    parser.add_argument(
        "--cohort_role",
        default=COHORT_ROLE_CALIBRATION,
        choices=list(COHORT_ROLES),
        help=(
            "Role the evaluated cohort plays. 'calibration' requires it to be the cohort the "
            "threshold was calibrated on. 'independent_evaluation' requires "
            "--authorized_cohort_signature and a patient-disjoint cohort."
        ),
    )
    parser.add_argument(
        "--authorized_cohort_signature",
        default=None,
        help=(
            "Membership signature the evaluation protocol authorises for "
            "--cohort_role independent_evaluation. Required for that role; an unnamed "
            "replacement cohort is refused."
        ),
    )
    parser.add_argument(
        "--authorized_cohort_split",
        default=None,
        help="Optional split name the protocol authorises alongside --authorized_cohort_signature.",
    )
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument("--neutral_margin", type=float, default=None)
    parser.add_argument(
        "--empty_policy",
        default=None,
        choices=("exclude", "legacy_one", "zero"),
        help=f"Empty-class Dice policy for volume blocks (default: {DEFAULT_EMPTY_CLASS_POLICY}).",
    )
    parser.add_argument(
        "--metric_space",
        default=METRIC_SPACE_SLICE_PROXY,
        choices=(METRIC_SPACE_SLICE_PROXY, METRIC_SPACE_VOLUME_RESIZED, METRIC_SPACE_VOLUME_NATIVE),
        help="Declared headline metric space. volume_native is rejected: it is not derivable "
        "from preprocessed_data/.",
    )
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument(
        "--probe_batches",
        type=int,
        default=None,
        help="Batch cap for probes 1 and 2 (default: --max_val_batches).",
    )
    parser.add_argument(
        "--firewall_batches",
        type=int,
        default=8,
        help="Number of batches the GT-firewall probe checks (default: 8).",
    )
    parser.add_argument(
        "--evidence_trajectory",
        default="self_audit",
        choices=("self_audit", "always_accept_refinement"),
        help="Trajectory probe 2 walks to find states carrying real audit evidence.",
    )
    parser.add_argument("--skip_volume", action="store_true", help="Skip the patient-volume pass.")
    parser.add_argument("--max_volumes", type=int, default=None)
    parser.add_argument("--volume_batch_size", type=int, default=8)
    parser.add_argument("--output", default="reports/audit_decomposition.json")
    parser.add_argument("--no_tqdm", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    if str(args.metric_contract) != "foreground_dice_exclude_v1":
        raise SystemExit(
            f"Unsupported --metric_contract {args.metric_contract!r}; "
            "audit_checkpoint computes foreground_dice_exclude_v1 slice proxy metrics. "
            "Legacy audit-target contracts are rejected."
        )
    if args.metric_space == METRIC_SPACE_VOLUME_NATIVE:
        raise SystemExit(
            "--metric_space volume_native is not available: the preprocessed mask was "
            "destructively nearest-downsampled to the network grid (scripts/preprocess_acdc.py:80), "
            "so a native-geometry Dice is not recoverable from preprocessed_data/. Supply raw ACDC "
            "NIfTI labels to evaluate_volume_native instead."
        )

    resolved = resolve_downstream_config(
        args.config,
        overrides={
            "data_root": args.data_root,
            "split_manifest": args.split_manifest,
            "num_workers": args.num_workers,
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )
    resolved.validate_splits()
    device = resolve_device(args.device or str(resolved.device))
    model = resolved.build_model(device)
    # Bind rather than merely load: the binding carries the file hash, the
    # digest of the weights that are actually live after the load, the
    # producing revision and the resolved model identity.  Those are the
    # objects the expected calibration lineage is derived from -- never the
    # artifact.
    binding = bind_evaluation_checkpoint(
        model,
        [("checkpoint", args.checkpoint)],
        map_location=device,
        config=resolved.to_legacy_dict(),
    )
    model.eval()

    split = str(resolved.val_split)
    val_dataset = resolved.build_dataset(split=split, train=False)
    val_loader = resolved.build_dataloader(
        val_dataset,
        train=False,
        batch_size=args.batch_size,
    )

    legacy_config = resolved.to_legacy_dict()
    audit_cfg = dict(legacy_config.get("audit", {}) or {})
    if "tau_accept" not in audit_cfg and "tau_accept" in legacy_config:
        audit_cfg["tau_accept"] = legacy_config["tau_accept"]

    neutral_margin = resolve_neutral_margin(
        args.neutral_margin if args.neutral_margin is not None else resolved.neutral_margin
    )
    empty_policy = resolve_empty_policy(args.empty_policy)
    t_max = int(
        args.t_max
        if args.t_max is not None
        else resolved.rollout_max_turns
    )

    # The expectation the calibration artifact is checked against is assembled
    # from what this process built: the bound weights, the constructed
    # validation dataset/loader, and the protocol's own metric semantics.  It
    # is built unconditionally so it cannot be shaped by what the artifact
    # happens to contain.
    cohort_policy = CohortPolicy(
        role=str(args.cohort_role),
        authorized_membership_signature=args.authorized_cohort_signature,
        authorized_split_name=args.authorized_cohort_split,
    )
    expected_lineage = build_expected_lineage(
        binding=binding,
        loader=val_loader,
        split_name=split,
        metric_contract=str(args.metric_contract),
        metric_space=str(args.metric_space),
        neutral_margin=neutral_margin,
        t_max=t_max,
        max_batches=args.max_val_batches,
        batch_size=getattr(val_loader, "batch_size", None),
    )
    resolution = resolve_tau_accept(
        cli_tau=args.tau_accept,
        calibration_path=args.calibration,
        config_audit=audit_cfg,
        neutral_margin=neutral_margin,
        allow_tau_override=bool(args.allow_tau_override),
        expected_lineage=expected_lineage,
        cohort_policy=cohort_policy,
    )
    tau_accept = float(resolution["tau_accept"])
    # The weights that were bound must still be the weights about to be
    # measured; a mutation between binding and use fails here.
    verify_bound_state(model, binding, boundary="audit_checkpoint_evaluation")

    volume_block: dict[str, Any] | None = None
    if not args.skip_volume:
        volume_block = evaluate_volume_cohort(
            model,
            val_dataset,
            device,
            tau_accept=tau_accept,
            t_max=t_max,
            neutral_margin=neutral_margin,
            empty_policy=empty_policy,
            image_size=int(legacy_config.get("image_size", 256)),
            num_classes=int(legacy_config.get("num_classes", 4)),
            batch_size=int(args.volume_batch_size),
            max_volumes=args.max_volumes,
            geometry_by_case=load_volume_geometry(
                legacy_config.get("data_root", "preprocessed_data/ACDC"), split=split
            ),
        )

    payload = build_report(
        model,
        val_loader,
        device,
        checkpoint=args.checkpoint,
        config_path=args.config,
        split=split,
        resolution=resolution,
        neutral_margin=neutral_margin,
        empty_policy=empty_policy,
        t_max=t_max,
        metric_space=str(args.metric_space),
        max_val_batches=args.max_val_batches,
        firewall_batches=int(args.firewall_batches),
        probe_batches=args.probe_batches if args.probe_batches is not None else args.max_val_batches,
        evidence_trajectory=str(args.evidence_trajectory),
        num_classes=int(legacy_config.get("num_classes", 4)),
        volume=volume_block,
        disable_tqdm=bool(args.no_tqdm),
    )
    payload["checkpoint_binding"] = binding.as_dict()
    payload["expected_calibration_lineage"] = expected_lineage
    payload["cohort_policy"] = cohort_policy.as_dict()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")
    _print_summary(payload, output)
    return payload


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None or _is_nan(value):
        return "nan"
    if isinstance(value, bool):
        return str(value)
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _print_summary(payload: Mapping[str, Any], output: Path) -> None:
    modes = payload["global_modes"]
    annotation = payload["annotation_headroom"]
    positive = payload["synthetic_positive_quality"]
    evidence = payload["audit_evidence"]
    firewall = payload["gt_firewall"]
    print(
        f"schema=v{payload['diagnostic_schema_version']} evidence_class={payload['evidence_class']} "
        f"tau={_fmt(payload['tau_accept'], '.5f')} src={payload['tau_accept_source']} "
        f"eps={_fmt(payload['neutral_margin'], '.5f')}"
    )
    verification = payload.get("calibration_lineage_verification")
    resolution = payload.get("threshold_resolution", {})
    if resolution.get("calibration_path") is None:
        print("calibration_lineage: no artifact consulted")
    else:
        cohort = (verification or {}).get("cohort", {})
        print(
            f"calibration_lineage: verified={(verification or {}).get('verified')} "
            f"role={cohort.get('role')} disjoint={cohort.get('patients_disjoint')} "
            f"producer_git_sha={(verification or {}).get('checkpoint_producing_git_sha')}"
        )
    print(
        f"A0={_fmt(annotation.get('phase_a/a0_dice'))} "
        f"refine_gain={_fmt(annotation.get('phase_a/total_refinement_gain'), '+.5f')} "
        f"stage_headroom={_fmt(annotation.get('phase_a/mean_stage_headroom'), '.5f')}"
    )
    print(
        f"initial={_fmt(modes.get('modes/initial_dice'))} "
        f"always={_fmt(modes.get('modes/always_accept_dice'))} "
        f"self={_fmt(modes.get('modes/self_audit_dice'))} "
        f"oracle={_fmt(modes.get('modes/oracle_dice'))}"
    )
    print(
        f"audit_rescue={_fmt(modes.get('modes/audit_rescue_vs_always'), '+.5f')} "
        f"oracle_headroom={_fmt(modes.get('modes/oracle_headroom'), '+.5f')} "
        f"headroom_capture={_fmt(modes.get('modes/headroom_capture_ratio'), '.3f')} "
        f"headroom_available={_fmt(modes.get('modes/headroom_available_rate'), '.3f')}"
    )
    print(
        f"positive_cf: n={_fmt(positive.get('positive_valid_count'), '.0f')} "
        f"argmax_change={_fmt(positive.get('positive_argmax_change_rate'), '.3f')} "
        f"neutral={_fmt(positive.get('positive_neutral_rate'), '.3f')} "
        f"mean_delta={_fmt(positive.get('positive_mean_delta_dice'), '+.5f')} "
        f"p_max={_fmt(positive.get('positive_state_mean_p_max'), '.3f')}"
    )
    print(
        f"audit_evidence: rows={_fmt(evidence.get('audit_evidence/rows_measured'), '.0f')} "
        f"logit_l1={_fmt(evidence.get('audit_evidence/candidate_logit_l1'), '.6f')} "
        f"argmax_change={_fmt(evidence.get('audit_evidence/argmax_change_rate'), '.5f')} "
        f"dice_delta={_fmt(evidence.get('audit_evidence/dice_delta_real_vs_zero'), '+.5f')} "
        f"inert={evidence.get('audit_evidence/branch_inert')}"
    )
    print(
        f"gt_firewall: batches={_fmt(firewall.get('gt_firewall/num_batches_checked'), '.0f')} "
        f"max_abs={_fmt(firewall.get('gt_firewall/max_abs_logit_diff'), '.3e')} "
        f"mismatch={_fmt(firewall.get('gt_firewall/decision_mismatch_rate'), '.5f')} "
        f"passed={firewall.get('gt_firewall/passed')}"
    )
    volume = payload.get("volume")
    if volume is None:
        print("volume: skipped (--skip_volume)")
    else:
        for phase in ("ED", "ES", "all"):
            block = volume["cohorts"][phase]["modes"]["self_audit"]
            print(
                f"volume[{phase}] metric_space={volume['metric_space']} "
                f"n={block['patient_count']} self_audit_macro={_fmt(block['macro_dice'])}"
            )
    required = payload["required_keys_present"]
    print(f"required_keys_ok={required['ok']} missing={required['missing']}")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
