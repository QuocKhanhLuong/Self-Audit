from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_dataset_compatibility_report import build_report


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_compatibility_report_records_audit_evidence_and_restrictions(tmp_path: Path) -> None:
    multi = tmp_path / "multi.csv"
    zt = tmp_path / "zt.csv"
    motion = tmp_path / "motion.csv"
    _write_csv(multi, ["case_id"], [{"case_id": "C001"}])
    _write_csv(
        zt,
        ["case_id", "status"],
        [{"case_id": "C001", "status": "manual_review"}],
    )
    _write_csv(
        motion,
        ["has_mask", "affine_equal"],
        [{"has_mask": "True", "affine_equal": "False"}, {"has_mask": "False", "affine_equal": ""}],
    )
    output = build_report(multi, zt, motion, tmp_path / "compatibility.md")
    text = output.read_text(encoding="utf-8")
    assert "101/105" not in text
    assert "manual-review cases: 1" in text
    assert "missing masks: 1; affine flags: 1" in text
    assert "USE_WITH_RESTRICTIONS" in text
