"""Focused checks for the mask-free experiments and isolated evaluation (W6).

Three checks, matching the W6 acceptance criteria:

1. paired-bank selectors E0-E5, the six negative controls, and the
   equal-number/equal-cost random-edit versus active-challenge comparison -
   including that an E4/E5 tie is reported as a tie and never as a gain, and
   that a missing incompatible study queues instead of fabricating evidence;
2. reference metric edge cases - both-empty excluded and counted, one-empty
   scored as an error, patient-level (not slice-level) aggregation and
   bootstrap, surface metrics unavailable without valid geometry, ranking
   metrics unavailable without support, and coverage reported with its level;
3. freeze tamper rejection, no-refit verification, and the sealed-set reuse
   guard - plus the fixed named class mapping surviving a permutation attack.

Run with::

    PYTHONPATH=src python3 -m pytest tests/test_maskfree_evaluation.py -v

Fixtures are CPU synthetic phantoms. They are software evidence about the
implemented equations only. No real ACDC or M&Ms data exists in this checkout,
and nothing here measures segmentation quality on real images.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from self_audit_maskfree.contracts import (
    AuditResult,
    FittingView,
    Hypothesis,
    ScoringView,
    hypothesis_from_labels,
)
from self_audit_maskfree.evaluation import (
    FreezeValidationError,
    VerificationReuseError,
    coverage_error_curve,
    dice_iou_volume,
    edit_attribution,
    evidence_quality_association,
    metric_row,
    paired_patient_bootstrap,
    patient_macro,
    reset_verification_registry,
    sha256_file,
    surface_metrics,
    validate_freeze_manifest,
    verify_frozen_bank,
)
from self_audit_maskfree.evaluation.metrics import MetricContractError
from self_audit_maskfree.evaluation.reference import (
    ReferenceConfig,
    evaluate_reference,
    oracle_cluster_matching_diagnostic,
)
from self_audit_maskfree.experiments import (
    CONTROL_IDS,
    SELECTOR_IDS,
    ForeignEvidenceCache,
    coverage_masks,
    matched_coverage,
    run_bank_experiments,
)
from self_audit_maskfree.observation import ObservationModel

SIZE = 32
STUDY = "synthetic_study"
UNIT = "synthetic_study_z04"
PARTITION = "partition_hash_abc"
REGION_MEANS = torch.tensor([-1.0, 0.0, 1.0, 2.0])


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------
def _true_labels(shift: int = 0) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(SIZE, dtype=torch.float32),
        torch.arange(SIZE, dtype=torch.float32),
        indexing="ij",
    )
    center = (SIZE - 1) / 2 + shift
    radius = ((ys - center).pow(2) + (xs - center).pow(2)).sqrt()
    labels = torch.zeros(SIZE, SIZE, dtype=torch.long)
    labels[radius < 12.0] = 1
    labels[radius < 8.0] = 2
    labels[radius < 4.0] = 3
    return labels


def _phantom(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(SIZE, SIZE, generator=generator) * 0.2
    return REGION_MEANS[_true_labels()] + noise


def _role_masks() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit, select and verify blocks, pairwise disjoint.

    The three roles must not overlap: a selection or verification pixel that is
    also a fitting pixel would be scored against a model that had already seen
    it, which is the leak the whole design exists to prevent.
    """
    block = torch.arange(SIZE) // 8
    block_y, block_x = torch.meshgrid(block, block, indexing="ij")
    parity = (block_y + block_x) % 4
    select = parity == 0
    verify = parity == 2
    return ~(select | verify), select, verify


def _views(study_id: str = STUDY, unit_id: str = UNIT) -> tuple[FittingView, ScoringView, ScoringView]:
    image = _phantom()
    fit_mask, select_mask, verify_mask = _role_masks()
    fit_image = torch.where(fit_mask, image, torch.zeros_like(image)).unsqueeze(0)
    fitting = FittingView(
        image=fit_image,
        support=fit_mask,
        context=fit_image.repeat(3, 1, 1),
        study_id=study_id,
        unit_id=unit_id,
        partition_id=PARTITION,
        metadata={"dataset": "synthetic", "split": "train", "epoch": 0},
    )
    selection = ScoringView(
        image=torch.where(select_mask, image, torch.zeros_like(image)).unsqueeze(0),
        support=select_mask,
        study_id=study_id,
        unit_id=unit_id,
        role="select",
        partition_id=PARTITION,
    )
    # The verification role is disjoint from both fit and select, so these
    # observations were never fitted on and never used to rank a candidate.
    verification = ScoringView(
        image=torch.where(verify_mask, image, torch.zeros_like(image)).unsqueeze(0),
        support=verify_mask,
        study_id=study_id,
        unit_id=unit_id,
        role="verify",
        partition_id=PARTITION,
    )
    return fitting, selection, verification


def _bank() -> list[Hypothesis]:
    """Four distinct candidates: incumbent, a shifted draft, a merge, and a challenge edit."""
    incumbent = hypothesis_from_labels(_true_labels(), "incumbent", "feature_grouping")
    incumbent.metadata["round"] = 0
    shifted = hypothesis_from_labels(_true_labels(shift=3), "shifted", "boundary_edit")
    shifted.metadata["round"] = 0
    merged_labels = _true_labels().clone()
    merged_labels[merged_labels == 3] = 2
    merged = hypothesis_from_labels(merged_labels, "merged", "split_merge")
    merged.metadata["round"] = 0
    challenge_labels = _true_labels().clone()
    challenge_labels[:6, :] = 1
    challenge = hypothesis_from_labels(challenge_labels, "challenge_0", "challenger")
    challenge.metadata["round"] = 1
    return [incumbent, shifted, merged, challenge]


def _audit_result(model: ObservationModel, fitting: FittingView, selection: ScoringView) -> AuditResult:
    bank = _bank()
    fitted = [model.fit(candidate, fitting) for candidate in bank]
    scores = [model.score(entry, selection) for entry in fitted]
    best = min(range(len(scores)), key=lambda i: scores[i].total)
    for entry, score in zip(fitted, scores):
        entry.metadata["selection_score"] = score.total
    return AuditResult(
        initial=bank[0],
        selected=bank[best],
        bank=bank,
        fitted=fitted,
        scores=scores,
        regional_margin=torch.zeros(SIZE, SIZE),
        validity=torch.ones(SIZE, SIZE),
        trace={"challenge_candidate_ids": ["challenge_0"], "rounds": 2},
    )


# ---------------------------------------------------------------------------
# check 1 - paired-bank selectors, controls, and the search comparison
# ---------------------------------------------------------------------------
def test_paired_bank_selectors_controls_and_search_comparison() -> None:
    """Every selector sees one bank; controls stay diagnostics; ties stay ties.

    The comparison is only meaningful if E1-E5 all choose from the same bank, so
    the test asserts that their chosen candidate ids are bank members and that
    the negative controls are reported in their own section. It also asserts the
    two honesty rules that are easy to violate silently: an E4/E5 tie is
    recorded as a tie rather than converted into a gain, and a shuffled-evidence
    control with no cached incompatible study reports itself unavailable and
    queues instead of inventing foreign numbers.
    """
    model = ObservationModel()
    fitting, selection, _ = _views()
    audit = _audit_result(model, fitting, selection)
    cache = ForeignEvidenceCache()

    result = run_bank_experiments(audit, fitting, selection, model, seed=7, foreign_cache=cache)

    assert tuple(result["selectors"]) == SELECTOR_IDS
    bank_ids = {candidate.candidate_id for candidate in audit.bank}
    for experiment in SELECTOR_IDS[1:]:
        entry = result["selectors"][experiment]
        assert entry["from_common_bank"] is True
        assert entry["candidate_id"] in bank_ids
    # E0 builds its own intensity-grouping partition and says so.
    assert result["selectors"]["E0_intensity_grouping"]["from_common_bank"] is False
    assert result["selectors"]["E1_cuts_inspired_control"]["name"] == "cuts_inspired_control"
    assert result["selectors"]["E1_cuts_inspired_control"]["external_cuts_reproduction"] == "pending"

    # An E4/E5 agreement is a tie, never a claimed improvement.
    e4_id = result["selectors"]["E4_selection_evidence"]["candidate_id"]
    e5 = result["selectors"]["E5_evidence_plus_challenge"]
    assert e5["tie_with_e4"] is (e4_id == e5["candidate_id"])
    if e5["tie_with_e4"]:
        assert "not a mask-quality gain" in e5["tie_note"]

    # All six negative controls are present, and none of them is a selector.
    control_names = [entry["control"] for entry in result["controls"]]
    assert control_names == list(CONTROL_IDS)
    assert not any(name.startswith("control_") for name in result["selectors"])

    # Shuffled evidence with an empty cache: unavailable, queued, not fabricated.
    shuffled = next(e for e in result["controls"] if e["control"] == "control_shuffled_evidence")
    assert shuffled["available"] is False and shuffled["queued"] is True
    assert "fabricat" in shuffled["reason"]
    assert "candidate_id" not in shuffled
    # ...and this unit's scores are now cached for a later, different study.
    assert cache.foreign("another_study") is not None
    assert cache.foreign(STUDY) is None

    # The random-edit control set matches the challenger in number and in fit cost.
    search = result["search_comparison"]
    assert search["equal_number"] is True
    assert search["equal_fitting_cost"] is True
    assert len(search["random_edits"]) == len(search["challenge_candidate_ids"])

    # Degenerate controls were fitted at the same budget as real candidates.
    degenerate = next(e for e in result["controls"] if e["control"] == "control_all_background")
    assert degenerate["selection"]["available"] is True
    assert degenerate["note"].startswith("diagnostic only")

    # Rows are serializable and carry full provenance with honest availability.
    payload = json.dumps(result["rows"])
    assert "contract_version" in payload
    for row in result["rows"]:
        assert row["available"] is (row["value"] is not None)
        if not row["available"]:
            assert row["reason"]

    # Matched coverage removes the "keep fewer pixels" degree of freedom.
    sparse = torch.zeros(SIZE, SIZE)
    sparse[:8, :] = 1.0
    dense = torch.ones(SIZE, SIZE)
    mask_a, mask_b = matched_coverage(sparse, dense)
    assert int(mask_a.sum()) == int(mask_b.sum()) == int((sparse > 0).sum())
    levels = coverage_masks(dense)
    assert int(levels[0.25].sum()) * 4 == int(levels[1.0].sum())


# ---------------------------------------------------------------------------
# check 2 - reference metric edge cases and patient-level aggregation
# ---------------------------------------------------------------------------
def test_reference_metric_edge_cases_and_patient_aggregation() -> None:
    """Empty classes, denominators, aggregation units and unavailable metrics.

    These are the places where a metric silently flatters itself: a both-empty
    class scored as a perfect 1.0, a one-empty class quietly dropped, slices
    bootstrapped as if they were patients, surface distances reported without
    geometry, or a ranking metric computed on two points.
    """
    reference = np.zeros((2, 8, 8), dtype=np.int64)
    reference[0, 2:6, 2:6] = 1  # RV present in one patient only
    prediction = reference.copy()

    scores = dice_iou_volume(prediction, reference)
    assert scores[1]["defined"] is True and scores[1]["dice"] == pytest.approx(1.0)
    # MYO and LV are absent from both sides: excluded and counted, never 1.0 or 0.0.
    assert scores[2]["defined"] is False and scores[2]["dice"] is None
    assert scores[2]["both_empty"] is True

    # One-empty is an error that stays in the aggregate.
    one_empty_prediction = reference.copy()
    one_empty_prediction[one_empty_prediction == 1] = 0
    one_empty = dice_iou_volume(one_empty_prediction, reference)
    assert one_empty[1]["defined"] is True
    assert one_empty[1]["one_empty"] is True
    assert one_empty[1]["dice"] == 0.0

    aggregate = patient_macro({"p1": scores, "p2": one_empty}, key="dice")
    assert aggregate["patients_counted"] == 2
    assert aggregate["macro_mean"] == pytest.approx(0.5)
    assert aggregate["per_class"][2]["available"] is False
    assert aggregate["per_class"][2]["both_empty_excluded"] == 2
    assert aggregate["per_class"][1]["one_empty_errors"] == 1

    # Surface metrics without valid geometry are unavailable, not pixel-labelled mm.
    without_geometry = surface_metrics(prediction, reference, class_id=1, spacing=None,
                                       geometry_valid=False)
    assert without_geometry["available"] is False
    assert without_geometry["hd95"] is None
    with_geometry = surface_metrics(prediction, reference, class_id=1, spacing=(1.0, 1.0, 1.0),
                                    geometry_valid=True)
    assert with_geometry["available"] is True and with_geometry["unit"] == "pixel"
    empty_class = surface_metrics(prediction, reference, class_id=3, spacing=(2.0, 1.0, 1.0),
                                  geometry_valid=True)
    assert empty_class["available"] is False and empty_class["undefined_case"] is True

    # The bootstrap resamples patients; a single pair cannot produce an interval.
    single = paired_patient_bootstrap({"p1": 0.6}, {"p1": 0.5})
    assert single["available"] is False and single["ci_low"] is None
    paired = paired_patient_bootstrap(
        {"p1": 0.6, "p2": 0.7, "p3": 0.65}, {"p1": 0.5, "p2": 0.6, "p3": 0.55}, iterations=200
    )
    assert paired["available"] is True and paired["count"] == 3
    assert paired["ci_low"] <= paired["mean_difference"] <= paired["ci_high"]

    # Ranking metrics refuse to report on insufficient support, and count ties.
    thin = evidence_quality_association([0.1, 0.2, -0.1], [0.0, 0.05, -0.05])
    assert thin["ranking_available"] is False and thin["auroc"] is None
    assert thin["ties"] == 1 and "insufficient ranking support" in thin["reason"]

    # Edit attribution separates fixes from destruction of previously correct pixels.
    initial = reference.copy()
    initial[0, 2, 2] = 0  # one wrong pixel to fix
    final = reference.copy()
    final[0, 3, 3] = 0  # one previously correct pixel destroyed
    attribution = edit_attribution(initial, final, reference)
    assert attribution["fix"] == 1 and attribution["regress"] == 1
    assert attribution["net_quality_change"] == 0
    assert attribution["harm_rate"] == pytest.approx(1 / attribution["initially_correct_pixels"])

    # Coverage rows always carry the coverage that produced them.
    validity = np.ones((2, 8, 8), dtype=np.float64)
    validity[0, 3, 3] = 0.0
    curve = coverage_error_curve(final, reference, validity)
    assert [entry["coverage_level"] for entry in curve] == [0.25, 0.5, 0.75, 1.0]
    assert curve[0]["error_rate"] == 0.0  # the low-validity error is dropped first
    assert curve[-1]["error_rate"] > 0.0

    # An "available" metric with no finite value is refused outright.
    with pytest.raises(MetricContractError):
        metric_row("bogus", None, unit="dice", dataset="d", split="test", protocol="p",
                   epoch=1, checkpoint="c", population="patient", count=1)


# ---------------------------------------------------------------------------
# check 3 - freeze tamper rejection, no-refit verification, reuse guard
# ---------------------------------------------------------------------------
def _write_freeze(tmp_path: Path, *, patients: tuple[str, ...] = ("p1", "p2")) -> dict:
    """Freeze a small prediction set through W7's real ``freeze_predictions``.

    Building the manifest with the owner's own API - rather than hand-rolling
    one - is what makes the tamper check meaningful: the hashes, the manifest
    identity and the compared-method enumeration are all produced by the code
    that will run in production.
    """
    from self_audit_maskfree.export import freeze_predictions

    reference = np.zeros((2, 8, 8), dtype=np.int64)
    reference[0, 2:6, 2:6] = 1
    entries = []
    for name, degrade in (("E1_cuts_inspired_control", True), ("E5_evidence_plus_challenge", False),
                          ("student_audited", False), ("student_no_audit", True)):
        for patient in patients:
            prediction = reference.copy()
            if degrade:
                prediction[0, 2, 2] = 0
            volume_dir = tmp_path / name / patient
            volume_dir.mkdir(parents=True, exist_ok=True)
            np.save(volume_dir / "labels.npy", prediction)
            np.save(volume_dir / "validity.npy", np.ones_like(prediction, dtype=np.float32))
            entries.append(
                {
                    "kind": "volume",
                    "prediction_name": name,
                    "study_id": patient,
                    "patient_id": patient,
                    "split": "test",
                    "dataset": "synthetic",
                    "protocol": "spatial_predictive",
                    "checkpoint_id": f"{name}_epoch149",
                    "grid": "stored",
                    "native_export_available": False,
                    "native_export_reason": "synthetic fixture has no native affine",
                    "volume_shape": list(prediction.shape),
                    "complete_volume": True,
                    "files": [f"{name}/{patient}/labels.npy", f"{name}/{patient}/validity.npy"],
                }
            )
    for patient in patients:
        np.save(tmp_path / f"reference_{patient}.npy", reference)
    return freeze_predictions(tmp_path, entries, {}, dataset="synthetic",
                              protocol="spatial_predictive", epoch=149,
                              required_methods=sorted({e["prediction_name"] for e in entries}))


def test_freeze_tamper_rejection_no_refit_verification_and_reuse_guard(tmp_path: Path) -> None:
    """A tampered freeze stops the evaluation; a sealed set is verified once.

    The three attacks covered here are the cheap ones: edit a prediction after
    the freeze and re-run the evaluator; let verification quietly refit the
    model on the sealed observations; and re-verify the same sealed set until a
    favourable number appears.
    """
    reset_verification_registry()
    manifest = _write_freeze(tmp_path)

    # An intact freeze validates and enumerates every compared method.
    receipt = validate_freeze_manifest(
        manifest, required_methods=["E5_evidence_plus_challenge", "student_audited"]
    )
    assert receipt["validated"] is True and receipt["files_checked"] == 16

    # Verification never refits: fits are made once, then only scored.
    model = ObservationModel()
    fitting, selection, verification = _views()
    audit = _audit_result(model, fitting, selection)
    before = [dict(entry.parameters) for entry in audit.fitted]
    # Degenerate challengers must have been fitted BEFORE the freeze; the audit
    # pass already produced them, and verification only scores them.
    experiments = run_bank_experiments(audit, fitting, selection, model, seed=7,
                                       foreign_cache=ForeignEvidenceCache())
    degenerate_fitted = list(experiments["fitted_controls"].values())
    report = verify_frozen_bank(audit.fitted, verification, model, manifest,
                                freeze_root=tmp_path, degenerate_fitted=degenerate_fitted)
    after = [dict(entry.parameters) for entry in audit.fitted]
    assert report["refit_performed"] is False
    assert [entry["components"] for entry in before] == [entry["components"] for entry in after]
    assert report["selected_by_verification"] in {c.candidate_id for c in audit.bank}
    assert report["perturbation_sensitivity"], "perturbation sensitivity must be measured"
    assert {entry["degenerate"] for entry in report["degenerate_challenges"]} == {
        "all_background", "random_mask", "excessive_partition", "class_permutation"
    }
    assert all(
        entry["method"].endswith("no refit") for entry in report["degenerate_challenges"]
    )
    assert report["candidates"][0]["selection_to_verification_gap"] is not None

    # Without pre-freeze degenerate fits the challenge is unavailable, not "passed".
    reset_verification_registry()
    without = verify_frozen_bank(audit.fitted, verification, model, manifest,
                                 freeze_root=tmp_path)
    assert without["degenerate_challenges"] == []
    assert "fitting after a freeze is not permitted" in without["degenerate_challenge_reason"]
    assert without["falsification_flag"] is False

    # A second verification of the same sealed set is refused unless disclosed.
    with pytest.raises(VerificationReuseError):
        verify_frozen_bank(audit.fitted, verification, model, manifest, freeze_root=tmp_path)
    repeated = verify_frozen_bank(audit.fitted, verification, model, manifest,
                                  freeze_root=tmp_path, allow_repeat=True)
    assert repeated["repeat_verification"] is True and repeated["freeze_reuse_count"] >= 1

    # The verify role is the only role this entry point accepts.
    with pytest.raises(ValueError):
        verify_frozen_bank(audit.fitted, selection, model, manifest, freeze_root=tmp_path)

    # Reference evaluation runs on the intact freeze...
    config = ReferenceConfig.from_mapping(
        {
            "dataset": "synthetic",
            "split": "test",
            "protocol": "spatial_predictive",
            "epoch": 149,
            "reference_root": str(tmp_path),
            "reference_label_map": {"0": "BG", "1": "RV", "2": "MYO", "3": "LV"},
            "cases": [
                {"patient_id": "p1", "mask_path": "reference_p1.npy", "geometry_valid": False},
                {"patient_id": "p2", "mask_path": "reference_p2.npy", "geometry_valid": False},
            ],
            "final_prediction": "E5_evidence_plus_challenge",
            "oracle_diagnostic": True,
        }
    )
    report = evaluate_reference(manifest, config, output_dir=tmp_path / "reference",
                                freeze_root=tmp_path)
    assert report["available"] is True
    assert (tmp_path / "reference" / "reference_metrics.json").is_file()
    assert (tmp_path / "reference" / "reference_metrics.csv").is_file()
    assert (tmp_path / "reference" / "reference_metrics.md").is_file()
    hd95_rows = [row for row in report["rows"] if row["name"].endswith(".hd95")]
    assert hd95_rows and all(row["available"] is False for row in hd95_rows)
    assert report["student_bootstrap"]["count"] == 2
    assert report["oracle_cluster_matching_diagnostic"]["diagnostic"] is True
    assert "never used for any selection" in report["oracle_cluster_matching_diagnostic"]["warning"]

    # ...and refuses to run once a frozen prediction is edited after the freeze.
    tampered = tmp_path / "E5_evidence_plus_challenge" / "p1" / "labels.npy"
    volume = np.load(tampered)
    volume[0, 4, 4] = 3
    np.save(tampered, volume)
    with pytest.raises(FreezeValidationError):
        evaluate_reference(manifest, config, output_dir=tmp_path / "reference2",
                           freeze_root=tmp_path)

    # The named class mapping is fixed: a permutation is a diagnostic, not a metric.
    permuted = {"p1": np.array([[[0, 1], [2, 3]]]), "p2": np.array([[[0, 1], [2, 3]]])}
    references = {"p1": np.array([[[0, 2], [3, 1]]]), "p2": np.array([[[0, 2], [3, 1]]])}
    diagnostic = oracle_cluster_matching_diagnostic(permuted, references)
    assert diagnostic["diagnostic"] is True
    assert diagnostic["permutation"] != [0, 1, 2, 3]
    reset_verification_registry()
