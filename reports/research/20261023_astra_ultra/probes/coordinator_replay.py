"""Rerun bounded research probes in an isolated source copy; never touch worker receipts."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
REPORT = Path("reports/research/20261023_astra_ultra")
PROBES = ["A_source_probes.py", "B_counterexamples.py", "C_recompute.py", "C_historical_recompute.py", "F_adversarial_probe.py"]
OUT = ROOT / REPORT / "evidence/coordinator_replay"
OUT.mkdir(parents=True, exist_ok=True)
common = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True).strip()
common = str((ROOT / common).resolve())
receipt = {"scope": "CPU manufactured and tracked-count replay only; no patient data or GPU", "python": sys.version, "executable": sys.executable, "source_files": {}, "commands": []}
with tempfile.TemporaryDirectory(prefix="self-audit-research-replay-") as tmp:
    work = Path(tmp)
    for source in sorted((ROOT / "self_audit").glob("*.py")) + [ROOT / "train.py"]:
        rel = source.relative_to(ROOT)
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        receipt["source_files"][str(rel)] = hashlib.sha256(source.read_bytes()).hexdigest()
    probe_dir = work / REPORT / "probes"
    probe_dir.mkdir(parents=True)
    (work / REPORT / "evidence").mkdir()
    env = {**os.environ, "GIT_DIR": common, "PYTHONDONTWRITEBYTECODE": "1"}
    for name in PROBES:
        original = ROOT / REPORT / "probes" / name
        script = probe_dir / name
        script.write_bytes(original.read_bytes())
        start = time.monotonic()
        run = subprocess.run([sys.executable, str(script)], cwd=work, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (OUT / (name + ".log")).write_text(run.stdout)
        receipt["commands"].append({"probe": name, "sha256": hashlib.sha256(script.read_bytes()).hexdigest(), "exit_code": run.returncode, "seconds": time.monotonic()-start, "log": name+".log"})
        print(name, "exit", run.returncode, flush=True)
    for result in (work / REPORT / "evidence").glob("*.json"):
        shutil.copyfile(result, OUT / result.name)
receipt["status"] = "PASS" if all(c["exit_code"] == 0 for c in receipt["commands"]) else "FAIL"
(OUT / "RECEIPT.json").write_text(json.dumps(receipt, indent=2)+"\n")
print(receipt["status"])
raise SystemExit(0 if receipt["status"] == "PASS" else 1)
