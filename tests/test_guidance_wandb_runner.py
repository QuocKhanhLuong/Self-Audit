"""The tracker observes existing logs without changing the scientific pipeline."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "guidance_tracker", Path(__file__).parents[1] / "scripts/run_stable_guidance_wandb.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_global_step_continues_across_epochs(tmp_path):
    (tmp_path / "PREFLIGHT_123.json").write_text(json.dumps({
        "checks": {"native_cine_inventory": {"train_steps_per_epoch": 2553}}}))
    p = module.Progress(tmp_path)
    assert p.observe("[STAGE] train python train.py")["pipeline/stage"] == "train"
    result = p.observe('[GUIDANCE_METRIC] {"epoch": 2, "step": 25, "feature_loss": 0.2, "semantic_loss": 0.0}')
    assert result["train/global_step"] == 2578
    assert result["train/semantic_loss"] == 0.0
    assert "train/dice" not in result
    result = p.observe("[STUDENT] {'epoch': 2, 'steps': 2553, 'feature_loss': 0.1, 'peak_allocated_bytes': None}")
    assert result["epoch/epoch"] == 2
    assert result["epoch/feature_loss"] == 0.1


def test_bank_and_export_progress_are_separate(tmp_path):
    p = module.Progress(tmp_path)
    r = p.observe("[BANK] patient001 SUPPORT_NOT_ACCURACY classes=[100, 2, 3, 4] residual=1e-07")
    assert r["bank/completed_patients"] == 1
    assert r["bank/class_3_pixels"] == 4
    for stage in ("export_val", "export_test"):
        p.observe(f"[STAGE] {stage} python export.py")
        assert p.observe("[EXPORT] patient001 class_pixels=[1, 2, 3, 4]")[f"{stage}/completed_patients"] == 1
    assert p.observe("unstructured warning") == {}


def test_inventory_and_preflight_are_visible(tmp_path):
    p = module.Progress(tmp_path)
    assert p.observe("[PREFLIGHT/DATA] patient001 split=train")["preflight/inventoried_patients"] == 1
    assert p.observe("[PREFLIGHT] PASS_EXECUTION_ONLY_QUALITY_UNVERIFIED report=/x")["preflight/status"].startswith("PASS_")
    assert p.observe("[SSL] frame=5/20 tiles=4 frozen_features=[10, 48, 48, 64]")["teacher/frame"] == 5
