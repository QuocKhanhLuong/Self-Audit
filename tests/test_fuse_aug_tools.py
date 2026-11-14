"""Software-only contract tests for A-aug comparison/report inputs."""
from __future__ import annotations

import sys
import json
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fuse_aug_report import build_report, render_readme
from fuse_aug_diagnose import _pins, run as diagnose
from fuse_aug_profile import run as profile
import fuse_aug_profile as profile_module


def _patients(offset=0):
    rows={}
    for i in range(4):
        value=.7+offset+i*.01
        rows[f"p{i}"]={"fg_dice":value,"per_class":{
            c:{"dice":value,"hd95_mm":float(i+1),"frames_with_reference":2}
            for c in ("RV","MYO","LV")}}
    return rows


def _result(offset=0):
    return {"status":"COMPLETED","validation":{
        mode:{"per_patient":_patients(offset+(j*.001))}
        for j,mode in enumerate(("teacher","student","fusion"))}}


def _transaction(offset=0):
    return {"schema":"self_audit.candidate_fuse.evaluation_transaction.v1",
        "status":"COMMITTED","role":"test","attempt":1,"metrics":{
        mode:{"per_patient":_patients(offset+j*.001)}
        for j,mode in enumerate(("teacher","student","fusion"))}}


def test_report_pairs_student_val_and_summarizes_all_modes():
    report=build_report(_result(.01),_result(.02),_transaction(),_transaction(.01),
                        {"status":"COMPLETED","metrics":{"teacher":{},"student":{},"fusion":{}}})
    primary=report["primary"]
    assert primary["comparison"]=="A-aug student minus A student"
    assert primary["statistics"]["n_paired_patients"]==4
    assert primary["statistics"]["bootstrap_resamples"]==10_000
    assert primary["statistics"]["bootstrap_seed"]==42
    assert set(report["old_A"])=={"VAL18","TEST50"}
    assert set(report["A_aug"]["VAL18"])=={"teacher","student","fusion"}
    assert "hd95_mm_patient_macro" in report["A_aug"]["TEST50"]["fusion"]["per_class"]["RV"]
    assert report["test_descriptive_only"] is True
    assert "opening 2" in report["test_interpretation"]
    readme=render_readme(report)
    assert "| Old A VAL18 | teacher |" in readme
    assert "| A-aug TEST50 | fusion |" in readme
    assert "| Phase0 L8 |" in readme or "## Phase 0 L8" in readme
    assert "TEST lần 2" in readme


def test_report_rejects_uncommitted_or_malformed_test():
    tx=_transaction(); tx["status"]="FAILED"
    with pytest.raises(ValueError,match="committed"):
        build_report(_result(),_result(),tx,_transaction(),{})
    tx=_transaction(); tx["metrics"].pop("fusion")
    with pytest.raises(ValueError,match="exactly"):
        build_report(_result(),_result(),tx,_transaction(),{})


def test_report_rejects_unpaired_val_patients():
    result=_result(.02)
    for row in result["validation"].values(): row["per_patient"].pop("p3")
    with pytest.raises(ValueError,match="patient ID sets differ"):
        build_report(_result(),result,_transaction(),_transaction(),{})


def test_actual_committed_old_result_and_test_receipt_old_vs_old():
    root=Path(__file__).resolve().parents[1]/"reports/fuse-semi-20261113/track-A"
    result_path=root/"RESULT.json"; tx_path=root/"TEST_ONCE/TEST_TRANSACTION.json"
    phase0_path=Path(__file__).resolve().parents[1]/"reports/fuse-acdc-k8-aug-20261114/PHASE0_L8.json"
    if not (result_path.is_file() and tx_path.is_file() and phase0_path.is_file()):
        pytest.skip("historical committed A receipts are not in this checkout")
    result=json.loads(result_path.read_text()); tx=json.loads(tx_path.read_text())
    phase0=json.loads(phase0_path.read_text())
    report=build_report(result,result,tx,tx,phase0)
    assert report["primary"]["statistics"]["mean_difference"]==0.0
    assert report["primary"]["statistics"]["patient_bootstrap_mean_difference_95ci"]==[0.0,0.0]


def _synthetic_a_config(tmp_path):
    from self_audit.candidate_fuse.config import default_config, normalize_config, SEMI_SCHEMA
    from self_audit.candidate_fuse.data import write_synthetic_manifests
    manifests=write_synthetic_manifests(tmp_path/"data",seed=42)
    cfg=default_config(synthetic=True)
    cfg["schema"]=SEMI_SCHEMA
    cfg["experiment"]={"track":"A","dataset":"ACDC"}
    cfg["augmentation"]={"policy":"inplane_strong_v1"}
    for role,prefix in (("train","train"),("development","val"),("test","test")):
        cfg["data"][f"{prefix}_manifest"]=manifests[role]["path"]
        cfg["data"][f"{prefix}_manifest_sha256"]=manifests[role]["sha256"]
    cfg=normalize_config(cfg)
    config_path=tmp_path/"a_aug.synthetic.json"
    config_path.write_text(json.dumps(cfg,sort_keys=True,indent=2)+"\n")
    return cfg,config_path,manifests


def test_diagnose_loads_real_format_sidecars_and_evaluates_synthetic_exports(tmp_path):
    from self_audit.candidate_fuse.teacher import build_teacher
    from self_audit.candidate_fuse.student import FuseStudent
    from self_audit.candidate_fuse.export import export_model
    cfg,config_path,manifests=_synthetic_a_config(tmp_path)
    teacher,_=build_teacher(cfg,synthetic=True)
    student=FuseStudent(cfg,teacher_channels=teacher.feature_channels,teacher_phases=teacher.feature_phases)
    exports=tmp_path/"exports"; exports.mkdir()
    for mode in ("teacher","student","fusion"):
        export_model(student if mode!="teacher" else None,teacher,cfg,exports/f"{mode}.pt",mode=mode)
    pins=_pins(exports)
    for mode,pin in pins.items():
        sidecar=json.loads((exports/f"{mode}.pt.manifest.json").read_text())
        assert pin["sha256"]==sidecar["artifact_sha256"]
        assert sidecar["artifact_bytes"]==(exports/f"{mode}.pt").stat().st_size
    result=diagnose(config_path,Path(manifests["train"]["path"]),manifests["train"]["sha256"],
        Path(manifests["development"]["path"]),manifests["development"]["sha256"],
        exports,tmp_path/"diagnosis.json",device="cpu",synthetic=True)
    assert result["evidence_status"]=="SOFTWARE_ONLY"
    assert result["test_opened"] is False
    assert set(result["metrics"])=={"L8_TRAIN_UNAUGMENTED","VAL18"}
    assert set(result["metrics"]["VAL18"])=={"teacher","student","fusion"}


@pytest.mark.parametrize("field,value",[("artifact_bytes",-1),("mode","fusion")])
def test_diagnose_rejects_sidecar_identity_mismatch(tmp_path,field,value):
    from self_audit.candidate_fuse.teacher import build_teacher
    from self_audit.candidate_fuse.student import FuseStudent
    from self_audit.candidate_fuse.export import export_model
    cfg,_,_=_synthetic_a_config(tmp_path)
    teacher,_=build_teacher(cfg,synthetic=True)
    student=FuseStudent(cfg,teacher_channels=teacher.feature_channels,teacher_phases=teacher.feature_phases)
    exports=tmp_path/"exports"; exports.mkdir()
    for mode in ("teacher","student","fusion"):
        export_model(student if mode!="teacher" else None,teacher,cfg,exports/f"{mode}.pt",mode=mode)
    sidecar=exports/"student.pt.manifest.json"
    original=json.loads(sidecar.read_text()); modified=dict(original); modified[field]=value
    sidecar.write_text(json.dumps(modified))
    with pytest.raises(ValueError,match="SHA/mode"):
        _pins(exports)


def test_profile_runs_matched_twenty_step_controls_on_cpu_synthetic_data(tmp_path,monkeypatch):
    from self_audit.candidate_fuse.config import normalize_config
    _,config_path,_=_synthetic_a_config(tmp_path)
    old=json.loads(config_path.read_text()); old.pop("augmentation")
    old=normalize_config(old)
    old_config=tmp_path/"a_old.synthetic.json"
    old_config.write_text(json.dumps(old,sort_keys=True,indent=2)+"\n")
    monkeypatch.setattr(profile_module,"_reset_peak",lambda *_:None)
    monkeypatch.setattr(profile_module,"_sync",lambda *_:None)
    monkeypatch.setattr(profile_module,"_peak_bytes",lambda *_:123456)
    result=profile(config_path,tmp_path/"profile_aug",device="cpu",synthetic=True)
    legacy=profile(old_config,tmp_path/"profile_old",device="cpu",legacy=True,synthetic=True)
    assert result["evidence_status"]=="SOFTWARE_ONLY"
    assert result["T"]["optimizer_steps"]==20
    assert result["T"]["batch_sizes"]==[2]*20
    assert result["T"]["schedule_phases"][:10]==["T-a"]*10
    assert result["T"]["schedule_phases"][10:]==["T-b"]*10
    assert result["T"]["keys"][0]==result["T"]["keys"][10]
    assert result["J"]["optimizer_steps"]==20
    assert result["timing_scope"].startswith("wall time includes per-update data loading")
    assert result["T"]["keys"]==legacy["T"]["keys"]
    assert result["J"]["pairs"]==legacy["J"]["pairs"]
    assert result["T"]["peak_vram_bytes"]==legacy["T"]["peak_vram_bytes"]==123456
    assert (tmp_path/"profile_aug"/"PROFILE_TJ.json").is_file()
    assert (tmp_path/"profile_old"/"PROFILE_TJ_LEGACY.json").is_file()
