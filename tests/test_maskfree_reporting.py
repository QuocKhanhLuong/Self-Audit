"""Focused synthetic checks for the bounded image-only report writer.

These checks exercise report arithmetic and provenance only.  They do not read
images or manual annotations and do not imply any real-data segmentation
result.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from self_audit_maskfree.reporting import write_image_only_reports


def _verification_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    values = {
        "u1": {"patient_id": "p1", "e5": (4.0, 4), "e1": (8.0, 4)},
        "u2": {"patient_id": "p1", "e5": (12.0, 4), "e1": (8.0, 4)},
        "u3": {"patient_id": "p2", "e5": (8.0, 4), "e1": (4.0, 4)},
    }
    for unit_id, item in values.items():
        patient_id = str(item["patient_id"])
        e5_sum, e5_count = item["e5"]  # type: ignore[misc]
        e1_sum, e1_count = item["e1"]  # type: ignore[misc]
        candidates = []
        for candidate_id, nll_sum, count, gap in (
            (f"{unit_id}-e5", e5_sum, e5_count, 0.2),
            (f"{unit_id}-e1", e1_sum, e1_count, 0.4),
            (f"{unit_id}-e2", e1_sum + 4.0, e1_count, 0.5),
            (f"{unit_id}-e3", e1_sum + 2.0, e1_count, 0.6),
            (f"{unit_id}-e4", e5_sum + 1.0, e5_count, 0.7),
            (f"{unit_id}-sa", e5_sum + 2.0, e5_count, 0.8),
            (f"{unit_id}-sn", e5_sum + 3.0, e5_count, 0.9),
        ):
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "semantic_unresolved": False,
                    "alternatives": [],
                    "verify_score": {
                        "nll_sum": nll_sum,
                        "count": count,
                        "normalized_nll": nll_sum / count,
                        "available": True,
                        "role": "verify",
                    },
                    "selection_to_verification_gap": gap,
                }
            )
        rows.append(
            {
                "unit_id": unit_id,
                "patient_id": patient_id,
                "split": "dev",
                "method_to_candidate": {
                    "E5_evidence_plus_challenge": f"{unit_id}-e5",
                    "E1_cuts_inspired_control": f"{unit_id}-e1",
                    "E2_confidence_consistency": f"{unit_id}-e2",
                    "E3_fitting_score": f"{unit_id}-e3",
                    "E4_selection_evidence": f"{unit_id}-e4",
                    "student_audited": f"{unit_id}-sa",
                    "student_no_audit": f"{unit_id}-sn",
                },
                "candidates": candidates,
                "falsification_flag": False,
                "annotation_stability": {"available": True, "agreement": 0.8},
                "perturbation_sensitivity": [{"delta_vs_unperturbed": 0.1}],
            }
        )
    return rows


def _split_verification_rows(
    split: str,
    *,
    e5_sum: float,
    e1_sum: float,
    matched_no_audit: float | None = None,
    matched_audited: float | None = None,
) -> list[dict[str, object]]:
    """Create one split whose E5/E1 effects can be deliberately reversed."""
    rows = _verification_rows()
    for index, row in enumerate(rows):
        row["split"] = split
        for candidate in row["candidates"]:  # type: ignore[index]
            candidate_id = str(candidate["candidate_id"])
            if candidate_id.endswith("-e5"):
                score = candidate["verify_score"]
                score["nll_sum"] = e5_sum
                score["normalized_nll"] = e5_sum / score["count"]
            elif candidate_id.endswith("-e1"):
                score = candidate["verify_score"]
                score["nll_sum"] = e1_sum
                score["normalized_nll"] = e1_sum / score["count"]
        if index == 0 and matched_no_audit is not None and matched_audited is not None:
            row["student_coverage"] = {
                "available": True,
                "natural": {
                    "student_no_audit": {"supported_pixels": 2, "available": True},
                    "student_audited": {"supported_pixels": 2, "available": True},
                },
                "matched": {
                    "student_no_audit": {
                        "supported_pixels": 2,
                        "available": True,
                        "score": {
                            "nll_sum": matched_no_audit * 2,
                            "count": 2,
                            "normalized_nll": matched_no_audit,
                            "available": True,
                        },
                    },
                    "student_audited": {
                        "supported_pixels": 2,
                        "available": True,
                        "score": {
                            "nll_sum": matched_audited * 2,
                            "count": 2,
                            "normalized_nll": matched_audited,
                            "available": True,
                        },
                    },
                },
                "matched_supported_pixels": 2,
            }
    return rows


def test_image_only_report_preserves_provenance_and_patient_aggregation(tmp_path: Path) -> None:
    manifest = {
        "dataset": "acdc",
        "resolved_protocol": "spatial_predictive",
        "manifest_id": "synthetic-manifest",
    }
    epoch_history = [
        {
            "global_epoch": 0,
            "units_visited": 3,
            "audit": {"select_nll_mean": 1.25, "accepted_edit_rate": 0.5},
        }
    ]
    experiments = [
        {
            "name": "E5_evidence_plus_challenge.selection_total",
            "value": 1.25,
            "unit": "nats/pixel",
            "split": "dev",
            "available": True,
            "count": 3,
        },
        {
            "name": "control_shuffled_evidence.selection_total",
            "value": None,
            "unit": "nats/pixel",
            "split": "dev",
            "available": False,
            "reason": "no incompatible study was cached",
            "count": 0,
        },
    ]
    coverage = [
        {
            "unit_id": "u1",
            "split": "dev",
            "pixels": 10,
            "natural_support_student_no_audit": 7,
            "natural_support_student_audited": 6,
            "valid_foreground_student_no_audit": 3,
            "valid_foreground_student_audited": 4,
            "matched_support": 5,
        },
        {
            "unit_id": "u2",
            "split": "dev",
            "pixels": 10,
            "natural_support_student_no_audit": 8,
            "natural_support_student_audited": 7,
            "valid_foreground_student_no_audit": 4,
            "valid_foreground_student_audited": 5,
            "matched_support": 6,
        },
    ]

    paths = write_image_only_reports(
        tmp_path,
        manifest,
        epoch_history,
        _verification_rows(),
        experiments,
        coverage,
        epoch=149,
        checkpoint=tmp_path / "last.pt",
    )

    assert set(paths) == {"json", "csv", "md"}
    assert all(Path(path).is_file() for path in paths.values())
    report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert report["provenance"] == {
        "available": True,
        "checkpoint": str(tmp_path / "last.pt"),
        "contract_version": "maskfree150.v1",
        "count": 3,
        "dataset": "acdc",
        "epoch": 149,
        "population": "image-only observation units",
        "protocol": "spatial_predictive",
        "reason": None,
        "split": "dev",
        "splits": ["dev"],
        "unit": "observation unit",
    }
    assert report["method_to_candidate"]["E5_evidence_plus_challenge"] == {
        "u1": "u1-e5",
        "u2": "u2-e5",
        "u3": "u3-e5",
    }
    # E5 has per-unit NLLs [1, 3] for p1 and [2] for p2: patient macro is 2.
    e5 = report["verification"]["method_summaries"]["E5_evidence_plus_challenge"]
    assert e5["patient_macro_normalized_nll"] == 2.0
    assert e5["raw_pooled_normalized_nll"] == 2.0
    bootstrap = report["verification"]["paired_patient_bootstrap"][
        "E5_evidence_plus_challenge_minus_E1_cuts_inspired_control"
    ]
    assert bootstrap["available"] is True
    assert bootstrap["count"] == 2
    assert bootstrap["mean_difference"] == 0.5
    matched = report["coverage"]["matched_coverage"]
    assert matched["support_pixels"] == 11
    assert matched["quality"]["value"] is None
    assert matched["quality"]["available"] is False
    assert report["coverage"]["valid_fg_collapse"]["student_audited"] is False
    assert report["verification"]["diagnostics"]["temporal"]["available"] is False
    assert "spatial_predictive" in report["verification"]["diagnostics"]["temporal"]["reason"]
    assert report["verification"]["diagnostics"]["stability"]["available"] is False
    assert "independent repeated-annotation" in report["verification"]["diagnostics"]["stability"]["reason"]
    assert report["verification"]["diagnostics"]["ranking_agreement"]["available"] is True

    with Path(paths["csv"]).open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    required = {
        "dataset",
        "split",
        "protocol",
        "epoch",
        "checkpoint",
        "population",
        "count",
        "available",
        "unit",
        "contract_version",
    }
    assert required.issubset(csv_rows[0])
    assert any(row["name"] == "verification.method.E5_evidence_plus_challenge.patient_macro_normalized_nll" for row in csv_rows)
    assert "Predictive NLL" in Path(paths["md"]).read_text(encoding="utf-8")


def test_mixed_split_reports_keep_train_and_test_provenance_separate(tmp_path: Path) -> None:
    """An all-split aggregate must not inherit the first row's split label."""
    train_rows = _split_verification_rows(
        "train", e5_sum=4.0, e1_sum=8.0, matched_no_audit=4.0, matched_audited=3.0
    )
    test_rows = _split_verification_rows(
        "test", e5_sum=8.0, e1_sum=4.0, matched_no_audit=1.0, matched_audited=2.0
    )
    paths = write_image_only_reports(
        tmp_path,
        {"dataset": "acdc", "resolved_protocol": "spatial_predictive"},
        [],
        [*train_rows, *test_rows],
        [],
        [],
        epoch=149,
        checkpoint=tmp_path / "last.pt",
    )
    overall = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert overall["provenance"]["split"] == "mixed"
    assert overall["provenance"]["splits"] == ["test", "train"]
    summary = overall["verification"]["method_summaries"]["E5_evidence_plus_challenge"]
    assert summary["provenance"]["split"] == "mixed"
    e5_rows = [
        row
        for row in overall["metric_rows"]
        if row["name"] == "verification.method.E5_evidence_plus_challenge.patient_macro_normalized_nll"
    ]
    assert len(e5_rows) == 1
    assert e5_rows[0]["split"] == "mixed"

    split_reports = overall["split_reports"]
    assert set(split_reports) == {"train", "test"}
    train = json.loads(Path(split_reports["train"]["json"]).read_text(encoding="utf-8"))
    test = json.loads(Path(split_reports["test"]["json"]).read_text(encoding="utf-8"))
    assert train["provenance"]["split"] == "train"
    assert test["provenance"]["split"] == "test"
    assert train["verification"]["method_summaries"]["E5_evidence_plus_challenge"][
        "patient_macro_normalized_nll"
    ] == 1.0
    assert test["verification"]["method_summaries"]["E5_evidence_plus_challenge"][
        "patient_macro_normalized_nll"
    ] == 2.0
    assert train["coverage"]["matched_coverage"]["quality"]["student_audited"][
        "normalized_nll"
    ] == 3.0
    assert test["coverage"]["matched_coverage"]["quality"]["student_audited"][
        "normalized_nll"
    ] == 2.0


def test_repeated_annotation_stability_is_reported_separately(tmp_path: Path) -> None:
    rows = _verification_rows()
    agreements = (
        {"E5_evidence_plus_challenge": (0.8, 0.9), "E1_cuts_inspired_control": (0.5, None),
         "student_no_audit": (0.7, 0.8), "student_audited": (0.9, 0.95)},
        {"E5_evidence_plus_challenge": (0.6, 0.7), "E1_cuts_inspired_control": (0.5, None),
         "student_no_audit": (0.8, 0.85), "student_audited": (0.8, 0.9)},
        {"E5_evidence_plus_challenge": (0.4, 0.5), "E1_cuts_inspired_control": (0.5, None),
         "student_no_audit": (0.6, 0.65), "student_audited": (0.7, 0.75)},
    )
    for row, values in zip(rows, agreements):
        row["repeated_annotation_stability"] = {
            "available": True,
            "protocol": "fit_noise_sigma_0.05",
            "scope": "pre_freeze_image_only",
            "methods": {
                method: {
                    "agreement": agreement,
                    "valid_agreement": valid,
                    "common_valid_count": 3,
                    "count": 4,
                    "available": True,
                }
                for method, (agreement, valid) in values.items()
            },
        }
    paths = write_image_only_reports(
        tmp_path,
        {"dataset": "acdc", "resolved_protocol": "spatial_predictive"},
        [],
        rows,
        [],
        [],
        epoch=149,
        checkpoint=tmp_path / "last.pt",
    )
    report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    diagnostics = report["verification"]["diagnostics"]
    stability = diagnostics["stability"]
    assert stability["available"] is True
    assert stability["protocol"] == "fit_noise_sigma_0.05"
    assert stability["scope"] == "pre_freeze_image_only"
    e5 = stability["methods"]["E5_evidence_plus_challenge"]
    # p1 has unit agreements [.8, .6] -> .7, p2 has [.4] -> patient macro .55.
    assert e5["agreement"] == 0.55
    assert e5["valid_agreement"] == 0.65
    assert e5["common_valid_count"] == 9
    assert e5["count"] == 12
    assert diagnostics["ranking_agreement"]["available"] is True
    assert diagnostics["stability"]["ranking_agreement"]["available"] is True
    stability_rows = [
        row
        for row in report["metric_rows"]
        if row["name"] == "verification.stability.E5_evidence_plus_challenge.agreement"
    ]
    assert len(stability_rows) == 1
    assert stability_rows[0]["value"] == 0.55
    assert stability_rows[0]["available"] is True
