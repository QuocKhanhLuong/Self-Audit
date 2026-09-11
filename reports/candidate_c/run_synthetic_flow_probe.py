"""Coordinator-only bounded CPU fixture probe; not a training recipe or efficacy experiment."""
from pathlib import Path
import importlib.util, json, os, subprocess, sys, tempfile, time
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("runtime_probe_helpers", ROOT / "tests/test_runtime_pipeline_smoke.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
work = Path(tempfile.mkdtemp(prefix="candidate-c-full-flow-"))
acdc = work / "acdc_flow"
acdc.mkdir()
original_dump = yaml.safe_dump

def candidate_dump(value, *args, **kwargs):
    if isinstance(value, dict) and "model" in value:
        value["model"]["window_mode"] = "candidate_c"
        if "training" in value:
            value["training"]["rollout"]["predicted_history_exposure"] = True
    return original_dump(value, *args, **kwargs)

yaml.safe_dump = candidate_dump
started = time.time()
try:
    h.test_bounded_pipeline_e2e_acdc_to_mnms(acdc)
finally:
    yaml.safe_dump = original_dump
print("ACDC Candidate C: three intervals + best/last + calibration + export + frozen synthetic M&Ms PASS", flush=True)
acdc_cfg = yaml.safe_load((acdc / "full_pipeline_smoke.yaml").read_text())
acdc_report = json.loads((acdc / "reports/pipeline_report.json").read_text())
external = json.loads((acdc / "reports/external_mnms.json").read_text())
assert external["window_mode"] == "candidate_c"
native = work / "native_flow"
native.mkdir()
for split, seed in [("training", 323), ("validation", 423)]:
    h._create_synthetic_mnms_dataset(native / "data", split=split, seed=seed)
# Keep patient-level splits disjoint. Images and masks are renamed as pairs.
for folder in ["volumes", "masks"]:
    for old, new in [("m001", "m003"), ("m002", "m004")]:
        p = native / "data/validation" / folder / (old + "_t00.npy")
        p.rename(p.with_name(new + "_t00.npy"))
cfg = yaml.safe_load((ROOT / "configs/self_audit_full_mnms.yaml").read_text())
cfg["dataset"].update(data_root=str(native / "data"), image_size=32, depth_axis=2)
cfg["dataset"]["dataloader"].update(num_workers=0, pin_memory=False, persistent_workers=False)
cfg["model"] = acdc_cfg["model"]
cfg["training"] = acdc_cfg["training"]
cfg["checkpoint"].update(output_dir=str(native / "weights"), save_best=True, save_last=True, best_selection_min_epoch=2)
cfg["logging"]["report_dir"] = str(native / "reports")
cfg["logging"]["wandb"]["enabled"] = False
cfg["calibration"]["enabled"] = False
cfg["diagnostics"]["evaluate_headroom"] = False
cfg["diagnostics"]["evaluate_decomposition"] = False
config_path = native / "mnms_candidate_c.yaml"
config_path.write_text(original_dump(cfg))
cmd = [sys.executable, "scripts/train_self_audit.py", "--config", str(config_path), "--device", "cpu", "--no_tqdm", "--max_val_batches", "1"]
r = subprocess.run(cmd, cwd=ROOT, env=h._get_subprocess_env(), capture_output=True, text=True, timeout=180)
(native / "command.log").write_text(r.stdout + r.stderr)
assert r.returncode == 0, r.stderr + r.stdout
report = json.loads((native / "reports/pipeline_report.json").read_text())
assert not report["completed"] and len(report["epochs"]) == 3
assert report["incomplete_reason"] == "max_val_batches_restricted"
assert (native / "weights/last.pt").is_file()
assert not (native / "weights/best.pt").exists()
print("Native M&Ms Candidate C: three intervals + last.pt + partial-validation best.pt refusal PASS", flush=True)
receipt = {"status": "PASS", "workspace": str(work), "elapsed_seconds": time.time()-started, "mode": "candidate_c", "predicted_history_exposure": True, "acdc_epochs": 3, "native_mnms_epochs": 3, "native_mnms_calibration": "disabled", "native_run_completed": False, "native_best_pt": "correctly absent: partial validation", "external_mode": external["window_mode"], "source_data": "synthetic only; distinct native cohorts", "limitation": "No GPU/real-data/efficacy claim; predicted history and solver eligibility may be sparse.", "acdc_epoch_exposure": [{k:v for k,v in e.items() if "evidence" in k or "candidate_c" in k or "history" in k} for e in acdc_report["epochs"]], "native_epoch_exposure": [{k:v for k,v in e.items() if "evidence" in k or "candidate_c" in k or "history" in k} for e in report["epochs"]]}
(ROOT / "reports/candidate_c/synthetic_flow_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
