#!/usr/bin/env python3
"""Build the locked A-aug comparison from existing results and TEST ledgers."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
from typing import Mapping, Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fuse_compare_results import paired_statistics

CLASSES = ("RV", "MYO", "LV")

def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict): raise ValueError(f"{path} must contain a JSON object")
    return value

def _check_transaction(tx: Mapping[str, Any]) -> None:
    if tx.get("schema") != "self_audit.candidate_fuse.evaluation_transaction.v1":
        raise ValueError("TEST transaction schema mismatch")
    if tx.get("status") != "COMMITTED" or tx.get("role") != "test":
        raise ValueError("TEST transaction must be committed and role=test")
    metrics = tx.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {"teacher", "student", "fusion"}:
        raise ValueError("TEST transaction must contain exactly teacher/student/fusion")
    patient_sets=[set(metrics[mode].get("per_patient",{})) for mode in metrics]
    if any(ids != patient_sets[0] for ids in patient_sets[1:]):
        raise ValueError("TEST mode patient IDs differ")

def _split(result: Mapping[str, Any], tx: Mapping[str, Any] | None = None) -> dict:
    if result.get("status") != "COMPLETED":
        raise ValueError("RESULT status must be COMPLETED")
    validation = result.get("validation")
    if not isinstance(validation, Mapping) or set(validation) != {"teacher", "student", "fusion"}:
        raise ValueError("RESULT must contain all three VAL modes")
    patient_sets=[set(validation[mode].get("per_patient",{})) for mode in validation]
    if any(ids != patient_sets[0] for ids in patient_sets[1:]):
        raise ValueError("VAL mode patient IDs differ")
    splits = {"VAL18": {"models": {m: validation[m]["per_patient"] for m in validation}}}
    if tx is not None:
        _check_transaction(tx)
        splits["TEST50"] = {"models": {m: tx["metrics"][m]["per_patient"] for m in tx["metrics"]}}
    return splits

def _summary(splits: Mapping[str, Any]) -> dict:
    result={}
    for split_name, split in splits.items():
        result[split_name]={}
        for mode, patients in split["models"].items():
            classes={}
            for cls in CLASSES:
                dice=[float(row["per_class"][cls]["dice"]) for row in patients.values()
                      if row["per_class"][cls].get("dice") is not None]
                hd=[float(row["per_class"][cls]["hd95_mm"]) for row in patients.values()
                    if row["per_class"][cls].get("hd95_mm") is not None]
                classes[cls]={"dice_patient_macro":sum(dice)/len(dice) if dice else None,
                              "hd95_mm_patient_macro":sum(hd)/len(hd) if hd else None,
                              "dice_patients":len(dice),"hd95_patients":len(hd)}
            fgs=[float(row["fg_dice"]) for row in patients.values() if row.get("fg_dice") is not None]
            result[split_name][mode]={"fg_dice_patient_macro":sum(fgs)/len(fgs) if fgs else None,
                "patient_count":len(fgs),"per_class":classes}
    return result

def build_report(old_result: Mapping[str, Any], new_result: Mapping[str, Any],
                 old_tx: Mapping[str, Any], new_tx: Mapping[str, Any], phase0: Mapping[str, Any]) -> dict:
    """Pair by patient; primary inference is A-aug student minus old A student VAL."""
    old_splits, new_splits = _split(old_result, old_tx), _split(new_result, new_tx)
    old_test_ids=set(old_splits["TEST50"]["models"]["student"])
    new_test_ids=set(new_splits["TEST50"]["models"]["student"])
    if old_test_ids != new_test_ids:
        raise ValueError("old/new TEST transaction patient IDs differ")
    val_old, val_new = old_splits["VAL18"]["models"], new_splits["VAL18"]["models"]
    paired = paired_statistics(
        {pid: float(row["fg_dice"]) for pid, row in val_new["student"].items()},
        {pid: float(row["fg_dice"]) for pid, row in val_old["student"].items()},
        bootstrap_resamples=10_000, seed=42)
    return {"schema": "self_audit.fuse_aug_report.v1", "status": "COMPLETED",
        "primary": {"comparison": "A-aug student minus A student", "split": "VAL18",
                    "metric": "patient_macro_fg_dice", "statistics": paired},
        "old_A": _summary(old_splits), "A_aug": _summary(new_splits),
        "phase0": dict(phase0), "test_interpretation":
        "TEST opening 2, after TEST was opened; descriptive only, never used for selection.",
        "per_mode_metrics": ["patient-macro FG/RV/MYO/LV Dice", "patient-macro RV/MYO/LV HD95 mm"],
        "test_descriptive_only": True}

def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"

def render_readme(report: Mapping[str, Any]) -> str:
    """Render concise reviewer tables from the exact report summaries."""
    header="| Split | Model | FG Dice | RV Dice | MYO Dice | LV Dice | RV HD95 mm | MYO HD95 mm | LV HD95 mm |"
    divider="|---|---|---:|---:|---:|---:|---:|---:|---:|"
    rows=["# A-aug comparison", "",
        "Primary comparison: student A-aug minus student A on paired VAL18 patients.",
        "TEST opening 2: TEST lần 2, sau khi TEST đã mở. TEST50 is descriptive only and was not used for selection.",
        "", "## VAL and TEST comparisons", "", header, divider]
    for cohort,key in (("Old A","old_A"),("A-aug","A_aug")):
        for split,models in report[key].items():
            for model,metrics in models.items():
                c=metrics["per_class"]
                vals=[metrics.get("fg_dice_patient_macro"),c["RV"].get("dice_patient_macro"),
                      c["MYO"].get("dice_patient_macro"),c["LV"].get("dice_patient_macro"),
                      c["RV"].get("hd95_mm_patient_macro"),c["MYO"].get("hd95_mm_patient_macro"),
                      c["LV"].get("hd95_mm_patient_macro")]
                rows.append("| " + " | ".join([f"{cohort} {split}",model,*map(_fmt,vals)]) + " |")
    rows.extend(["", "## Phase 0 L8", "", header, divider])
    phase0=report.get("phase0",{}).get("metrics",{}).get("L8_TRAIN_UNAUGMENTED",{})
    for model,metrics in phase0.items():
        if not isinstance(metrics,Mapping) or "per_class" not in metrics: continue
        c=metrics["per_class"]
        vals=[metrics.get("fg_dice"),c["RV"].get("dice"),c["MYO"].get("dice"),c["LV"].get("dice"),
              c["RV"].get("hd95_mm"),c["MYO"].get("hd95_mm"),c["LV"].get("hd95_mm")]
        rows.append("| Phase0 L8 | " + " | ".join([model,*map(_fmt,vals)]) + " |")
    stats=report["primary"]["statistics"]
    ci=stats["patient_bootstrap_mean_difference_95ci"]
    rows.extend(["", "## Primary paired VAL result", "",
        f"Student A-aug minus A: mean {stats['mean_difference']:.6f}; 95% patient-bootstrap CI [{ci[0]:.6f}, {ci[1]:.6f}]; paired two-sided Wilcoxon p={stats['wilcoxon_two_sided']['p_value']:.6g}; n={stats['n_paired_patients']}; 10,000 resamples, seed 42.",
        "", "## Profile and campaign timing", ""])
    profiles=report.get("profiles",{})
    for label,key in (("Matched old A (legacy)","matched_old_A_profile"),("A-aug","A_aug")):
        profile=profiles.get(key)
        if isinstance(profile,Mapping):
            rows.append(f"- {label}: T {profile['T'].get('seconds_per_update_including_load_prepare_backward_step','NA')} s/update; J {profile['J'].get('seconds_per_update_including_load_prepare_backward_step','NA')} s/update.")
        else: rows.append(f"- {label}: NOT_AVAILABLE")
    campaign=report.get("timing",{}).get("campaign_seconds")
    rows.append(f"- A-aug campaign elapsed: {_fmt(campaign)} seconds." if campaign is not None else "- A-aug campaign elapsed: NOT_AVAILABLE.")
    rows.append("")
    return "\n".join(rows)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-run",type=Path); p.add_argument("--new-run",type=Path)
    p.add_argument("--old-result",type=Path); p.add_argument("--new-result",type=Path)
    p.add_argument("--old-test",type=Path); p.add_argument("--new-test",type=Path)
    p.add_argument("--phase0",required=True,type=Path); p.add_argument("--out",required=True,type=Path)
    a=p.parse_args(argv)
    if a.old_run or a.new_run:
        if not (a.old_run and a.new_run): p.error("--old-run and --new-run must be supplied together")
        a.old_result=a.old_run/"RESULT.json"; a.old_test=a.old_run/"TEST_ONCE"/"TEST_TRANSACTION.json"
        a.new_result=a.new_run/"RESULT.json"; a.new_test=a.new_run/"TEST_ONCE"/"TEST_TRANSACTION.json"
    if not all((a.old_result,a.new_result,a.old_test,a.new_test)): p.error("supply run directories or all four result/transaction paths")
    report=build_report(_read(a.old_result),_read(a.new_result),_read(a.old_test),_read(a.new_test),_read(a.phase0))
    legacy_profile=a.new_run/"PROFILE_TJ_LEGACY.json" if a.new_run else None
    new_profile=a.new_run/"PROFILE_TJ.json" if a.new_run else None
    flow_path=a.new_run/"AUG_FLOW.json" if a.new_run else None
    report["profiles"]={
        "matched_old_A_profile":_read(legacy_profile) if legacy_profile and legacy_profile.is_file() else "NOT_AVAILABLE",
        "A_aug":_read(new_profile) if new_profile and new_profile.is_file() else "NOT_AVAILABLE"}
    flow=_read(flow_path) if flow_path and flow_path.is_file() else {}
    campaign_seconds=flow.get("campaign_seconds")
    if campaign_seconds is not None and (not isinstance(campaign_seconds,(int,float)) or not 0 <= float(campaign_seconds) < float("inf")):
        raise ValueError("AUG_FLOW campaign_seconds must be finite and nonnegative")
    report["timing"]={"campaign_seconds":campaign_seconds}
    inputs=[a.old_result,a.new_result,a.old_test,a.new_test,a.phase0]
    for path in (legacy_profile,new_profile,flow_path):
        if path and path.is_file(): inputs.append(path)
    report["input_sha256"]={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    report["raw_receipts"]={"old_RESULT":_read(a.old_result),"new_RESULT":_read(a.new_result),
        "old_TEST_TRANSACTION":_read(a.old_test),"new_TEST_TRANSACTION":_read(a.new_test),
        "phase0":_read(a.phase0)}
    if legacy_profile and legacy_profile.is_file():
        report["raw_receipts"]["matched_old_A_PROFILE_TJ"]=_read(legacy_profile)
    if new_profile and new_profile.is_file():
        report["raw_receipts"]["A_aug_PROFILE_TJ"]=_read(new_profile)
    if flow_path and flow_path.is_file():
        report["raw_receipts"]["AUG_FLOW"]=flow
    if legacy_profile and legacy_profile.is_file():
        report["raw_receipts"]["matched_old_A_PROFILE_TJ"]=_read(legacy_profile)
    if new_profile and new_profile.is_file():
        report["raw_receipts"]["A_aug_PROFILE_TJ"]=_read(new_profile)
    if flow_path and flow_path.is_file():
        report["raw_receipts"]["AUG_FLOW"]=flow
    out=a.out/"REPORT.json" if (a.old_run or a.out.suffix=="") else a.out
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    if a.old_run or a.out.suffix=="":
        (out.parent/"README.md").write_text(render_readme(report),encoding="utf-8")
    print(json.dumps({"status":report["status"],"primary":report["primary"]},indent=2))
if __name__=="__main__": main()
