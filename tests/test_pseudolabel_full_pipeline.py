import subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_full_pipeline_dry_run_builds_all_stages(tmp_path):
    out=tmp_path/"run"
    cp=subprocess.run([
        sys.executable,str(ROOT/"scripts/run_full_pipeline_v3.py"),
        "--dataset","acdc","--root",str(tmp_path/"data"),
        "--split-manifest",str(tmp_path/"split.json"),
        "--out",str(out),"--device","cpu","--teacher-epochs","2",
        "--train-student","--student-epochs","3","--dry-run"
    ],capture_output=True,text=True,check=True)
    text=cp.stdout
    assert "[DRYRUN] teacher:" in text
    assert "train_pseudolabel_v3.py" in text
    assert "[DRYRUN] eval:" in text
    assert "evaluate_pseudolabel_frozen.py" in text
    assert "[DRYRUN] student:" in text
    assert "train_student_v3.py" in text
    assert (out/"pipeline.log").exists()
