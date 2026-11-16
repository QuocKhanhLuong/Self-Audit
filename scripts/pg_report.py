#!/usr/bin/env python3
"""Build a truthful Candidate PG report from epoch CSV/diagnostic artifacts.

This is a report reader, not an evaluator: absent runs remain ``NOT_RUN`` and
no metric is inferred from configuration, fixtures, or a stopped process.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = dict(raw)
            for key, value in list(row.items()):
                if value == "":
                    row[key] = None
                else:
                    try:
                        row[key] = float(value) if any(token in key for token in ("dice", "hd95", "assd", "loss", "lr", "seconds")) else int(value) if key in {"epoch", "update"} else value
                    except (ValueError, TypeError):
                        pass
            rows.append(row)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _read_patient_csv(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(dict(row) for row in csv.DictReader(stream))
    return rows


def _best_val(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from self_audit.candidate_pg.logging import select_best_epoch
    try:
        best = select_best_epoch(rows, split="VAL18", path="student_only", metric="fg_dice")
    except ValueError:
        return {"status": "NOT_RUN_NO_VAL_STUDENT_ROWS"}
    return {"status": "SELECTED", "epoch": int(best.get("epoch", 0)),
            "fg_dice": float(best["fg_dice"]), "split": "VAL18", "path": "student_only"}


def _plot(rows: list[dict[str, Any]], diag: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "NOT_RUN_MATPLOTLIB", "reason": str(exc)}
    if not rows and not diag:
        return {"status": "NOT_RUN_NO_PLOT_INPUT"}
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    if rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        groups = {}
        for row in rows:
            if row.get("fg_dice") is None or row.get("epoch") is None:
                continue
            key = f"{row.get('split','?')}/{row.get('path','?')}"
            groups.setdefault(key, [[], []]); groups[key][0].append(row["epoch"]); groups[key][1].append(row["fg_dice"])
        for key, (x, y) in sorted(groups.items()):
            order = sorted(zip(x, y)); ax.plot([a for a, _ in order], [b for _, b in order], label=key)
        ax.set(xlabel="epoch", ylabel="FG Dice", title="Candidate PG learning curves"); ax.legend(fontsize=7)
        fig.tight_layout(); path = out / "learning_curves.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
    if diag:
        fig, ax = plt.subplots(figsize=(7, 4))
        x, y = [], []
        for index, row in enumerate(diag):
            value = row.get("loss_total", row.get("agreement"))
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                x.append(row.get("update", index)); y.append(value)
        if y: ax.plot(x, y); ax.set(xlabel="update", ylabel="diagnostic", title="Candidate PG diagnostics")
        fig.tight_layout(); path = out / "diagnostics.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
    return {"status": "PASS" if paths else "NOT_RUN_NO_PLOT_VALUES", "files": paths}


def build_report(*, metrics_csv: str | Path | None = None,
                 metrics_jsonl: str | Path | None = None,
                 diagnostics_jsonl: str | Path | None = None,
                 paired_json: str | Path | None = None,
                 metrics_dir: str | Path | None = None,
                 out_dir: str | Path = "reports/pg-k8-20261116/report") -> dict[str, Any]:
    root = Path(out_dir); root.mkdir(parents=True, exist_ok=True)
    source_root = Path(metrics_dir) if metrics_dir is not None else None
    if source_root is not None:
        metrics_csv = metrics_csv or source_root / "epochs.csv"
        metrics_jsonl = metrics_jsonl or source_root / "epochs.jsonl"
        diagnostics_jsonl = diagnostics_jsonl or source_root / "diag.jsonl"
    rows = _read_csv(Path(metrics_csv)) if metrics_csv else []
    jsonl_rows = _read_jsonl(Path(metrics_jsonl)) if metrics_jsonl else []
    if not rows:
        rows = jsonl_rows
    elif jsonl_rows:
        if len(rows) != len(jsonl_rows):
            raise ValueError("epochs.csv and epochs.jsonl row counts differ")
        csv_keys = [(row.get("epoch"), row.get("split"), row.get("path")) for row in rows]
        json_keys = [(row.get("epoch"), row.get("split"), row.get("path")) for row in jsonl_rows]
        if csv_keys != json_keys:
            raise ValueError("epochs.csv and epochs.jsonl keys differ")
    diag = _read_jsonl(Path(diagnostics_jsonl)) if diagnostics_jsonl else []
    patient_paths = sorted(source_root.glob("patients_epoch*.csv")) if source_root is not None and source_root.is_dir() else []
    patient_rows = _read_patient_csv(patient_paths)
    paired: Any = {"status": "NOT_RUN_NO_PAIRED_COMPARISONS"}
    if paired_json and Path(paired_json).is_file():
        try: paired = json.loads(Path(paired_json).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc: paired = {"status": "NOT_RUN_INVALID_PAIRED_JSON", "reason": str(exc)}
    report = {
        "schema": "self_audit.candidate_pg.report.v1",
        "evidence_status": "SOFTWARE_ONLY" if rows else "NOT_RUN",
        "real_results": "NOT_RUN",
        "selection": _best_val(rows),
        "rows": len(rows), "diagnostic_rows": len(diag), "patient_rows": len(patient_rows),
        "source": {"epochs_csv": str(metrics_csv) if metrics_csv else None,
                   "epochs_jsonl": str(metrics_jsonl) if metrics_jsonl else None,
                   "patient_csv": [str(path) for path in patient_paths]},
        "paired": paired, "plots": _plot(rows, diag, root / "plots"),
        "shuffle_diagnostics": {"A3": {"status": "NOT_RUN"}, "A4": {"status": "NOT_RUN"}},
    }
    (root / "PG_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    readme = ["# Candidate PG report", "", f"Evidence status: **{report['evidence_status']}**", "", "Real S1/S2/S3/A1/A2/A3/A4 results: **NOT_RUN**.", ""]
    selection = report["selection"]
    readme.append(f"Selected epoch: {selection.get('epoch', 'NOT_RUN')} (VAL student-only; TEST is descriptive).")
    readme.append("")
    readme.append("Missing inputs remain NOT_RUN; this report does not fabricate patient, GPU, or training evidence.")
    (root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--metrics-jsonl", type=Path)
    parser.add_argument("--diagnostics-jsonl", type=Path)
    parser.add_argument("--paired-json", type=Path)
    parser.add_argument("--metrics-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(metrics_csv=args.metrics_csv, metrics_jsonl=args.metrics_jsonl,
                                  diagnostics_jsonl=args.diagnostics_jsonl, paired_json=args.paired_json,
                                  metrics_dir=args.metrics_dir,
                                  out_dir=args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
