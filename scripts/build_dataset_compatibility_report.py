"""Build an evidence-backed compatibility decision report from audit CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_report(
    cmr_multi_audit: str | Path,
    cmr_multi_zt: str | Path,
    cmrxmotion_audit: str | Path,
    output: str | Path,
) -> Path:
    multi = _read(Path(cmr_multi_audit))
    zt = _read(Path(cmr_multi_zt))
    motion = _read(Path(cmrxmotion_audit))
    multi_accepted = sum(row.get("status") == "accepted" for row in zt)
    multi_review = len(zt) - multi_accepted
    motion_labeled = sum(row.get("has_mask", "").lower() == "true" for row in motion)
    motion_missing = len(motion) - motion_labeled
    motion_affine = sum(row.get("affine_equal", "").lower() == "false" for row in motion)
    multi_status = "USE_FOR_TRAINING" if multi and multi_review == 0 else "USE_WITH_RESTRICTIONS"
    motion_status = "USE_FOR_TRAINING" if motion_labeled and not motion_missing and not motion_affine else "USE_WITH_RESTRICTIONS"
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset compatibility",
        "",
        "This report is generated from the local audit CSVs. It preserves the",
        "existing [B,3,H,W] spatial 2.5-D contract and does not alter source data.",
        "",
        "| Criterion | ACDC | M&Ms | CMR-MULTI | CMRxMotion |",
        "|---|---|---|---|---|",
        f"| SAX anatomy compatible | existing contract | existing contract | evidence required: {len(multi)} cases | evidence from audit |",
        "| LV/RV/MYO compatible | verified current schema | verified current mapping | raw 0/1/2/3 remapped | raw 0/1/2/3 remapped |",
        f"| GT quality sufficient | current training source | labeled source | {multi_accepted}/{len(zt)} Z/T accepted | {motion_labeled}/{len(motion)} labeled |",
        "| Input convertible to current 2.5-D | yes | yes | yes, Z/T then spatial Z | yes, native 3-D |",
        "| Spatial context meaningful | yes | yes | only within accepted Z/T cases | adjacent native Z slices |",
        "| Z/affine risk | current pipeline | current pipeline | flattened-axis spacing not used | affine mismatches flagged |",
        f"| Missing/uncertain data | current policy | current policy | manual-review cases: {multi_review} | missing masks: {motion_missing}; affine flags: {motion_affine} |",
        "| Subject leakage controllable | subject split | subject split | case/subject manifests | subject groups all acquisitions/phases |",
        "| Decision | USE_FOR_TRAINING | USE_WITH_RESTRICTIONS | "
        f"{multi_status} | {motion_status} |",
        "",
        "## Dataset-specific restrictions",
        "",
        "- CMR-MULTI: exclude cases whose Z/T inference is not accepted; do not",
        "  train on temporal context or invent flattened-axis spacing.",
        "- CMRxMotion: supervised training uses labeled cases only. Affine",
        "  mismatches require native aligned-array handling plus visual QC; strict",
        "  mode excludes them.",
        "- Mixed training must use subject/dataset-balanced sampling and the same",
        "  model, loss, augmentation, and validation contract as the baseline.",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmr-multi-audit", default="reports/cmr_multi_audit.csv")
    parser.add_argument("--cmr-multi-zt", default="reports/cmr_multi_zt_inference.csv")
    parser.add_argument("--cmrxmotion-audit", default="reports/cmrxmotion_audit.csv")
    parser.add_argument("--output", default="reports/dataset_compatibility.md")
    args = parser.parse_args()
    print(build_report(
        args.cmr_multi_audit,
        args.cmr_multi_zt,
        args.cmrxmotion_audit,
        args.output,
    ))


if __name__ == "__main__":
    main()
