#!/usr/bin/env python3
"""Bounded real-data gate; never starts the 150-epoch experiment.

Uses the active interpreter/environment, one child at a time, and read-only
resource sampling. All outputs are fresh; existing evidence is never replaced.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from self_audit_maskfree.config import load_config
from self_audit_maskfree.resources import compute_resource_deltas, resource_snapshot


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("/root/Self-Audit/data"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--total-epochs", type=int, choices=(50, 150), default=150)
    p.add_argument("--acdc-data-root", type=Path, default=None)
    p.add_argument("--mnms-data-root", type=Path, default=None)
    p.add_argument("--prefetch-batches", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--data-cache-bytes", type=int, default=64 * 1024 * 1024)
    p.add_argument("--batch-size", type=int, choices=(8, 16, 32, 64), default=8)
    p.add_argument("--candidate-workers", type=int, choices=(0, 2, 4, 8), default=0)
    p.add_argument("--candidate-chunk-size", type=int, choices=range(1, 9), default=8)
    p.add_argument("--prefetch-max-bytes", type=int, default=32 * 1024 * 1024)
    p.add_argument("--diagnostic", action="store_true", help="1 warmup + 3 instrumented batches and cProfile")
    p.add_argument("--reference-report", type=Path, help="existing matched ordinary report; strict comparator unchanged")
    p.add_argument("--print-plan", action="store_true", help="validate configs and print commands without execution")
    return p


def configs_and_commands(args):
    output = args.output.resolve()
    roots = {
        "acdc": getattr(args, "acdc_data_root", None) or args.data_root / "ACDC",
        "mnms": getattr(args, "mnms_data_root", None) or args.data_root / "MnM/extracted/M&M",
    }
    total_epochs = getattr(args, "total_epochs", 150)
    configs, commands = {}, []
    for dataset, root in roots.items():
        config = load_config(ROOT / f"configs/maskfree_{dataset}_150.yaml").replace(
            data_root=str(root.resolve()), output_dir=str(output / "preflight_workspace"),
            run_id=f"rental-{dataset}-gate", image_size=224, batch_size=args.batch_size,
            accumulation_steps=1, total_epochs=total_epochs, amp=False, device="cuda", audit_device="cpu",
            allow_cpu=False, max_steps=None, max_epochs=None, resume=None,
            timing_mode="diagnostic" if args.diagnostic else "production", logging_mode="buffered",
            data_cache_bytes=args.data_cache_bytes, prefetch_batches=args.prefetch_batches,
            candidate_workers=args.candidate_workers, candidate_chunk_size=args.candidate_chunk_size,
            prefetch_max_bytes=args.prefetch_max_bytes)
        configs[dataset] = config
        commands.append([sys.executable, str(ROOT / "scripts/train_maskfree.py"), "--config",
                         str(output / f"{dataset}.json"), "--preflight"])
    profile = [sys.executable, str(ROOT / "scripts/profile_maskfree.py"), "--config", str(output / "acdc.json"),
               "--output", str(output / "profile"), "--kind", "optimized", "--device", "cuda",
               "--audit-device", "cpu", "--image-size", "224", "--batch-size", str(args.batch_size), "--warmup-batches", "1" if args.diagnostic else "5",
               "--measured-batches", "3" if args.diagnostic else "30", "--timing-mode",
               "instrumented" if args.diagnostic else "ordinary"]
    profile.extend(["--total-epochs", str(total_epochs)])
    if args.diagnostic:
        profile.extend(["--cprofile", "--torch-trace"])
    if args.reference_report:
        profile.extend(["--reference-report", str(args.reference_report.resolve())])
    commands.append(profile)
    return configs, commands


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    configs, commands = configs_and_commands(args)
    if args.print_plan:
        print(json.dumps({"commands": commands, "configs": {k: v.to_dict() for k, v in configs.items()}}, indent=2))
        return 0
    if args.output.exists():
        raise ValueError("output must be fresh; refusing to overwrite benchmark evidence")
    for config in configs.values():
        if not Path(config.data_root).is_dir():
            raise FileNotFoundError(config.data_root)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This real-data gate requires CUDA; no CPU fallback")
    # Respect CUDA_VISIBLE_DEVICES; do not choose another GPU or terminate jobs.
    initial = resource_snapshot(data_roots=[c.data_root for c in configs.values()], include_gpu=True)
    selected_uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    if not selected_uuid or not initial.get("gpu", {}).get("compute_apps_available"):
        raise RuntimeError("Cannot establish selected GPU identity/competing compute processes; inspect resource diagnostics first")
    competing = [app for app in initial.get("gpu", {}).get("competing_apps", [])
                 if str(app.get("gpu_uuid", "")).removeprefix("GPU-").lower() == selected_uuid.removeprefix("GPU-").lower() and app.get("pid") != os.getpid()]
    if competing:
        raise RuntimeError(f"Selected GPU has compute processes; schedule this gate alone: {competing}")
    args.output.mkdir(parents=True)
    for name, config in configs.items():
        (args.output / f"{name}.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    (args.output / "plan.json").write_text(json.dumps({"commands": commands, "initial": initial}, indent=2) + "\n")
    # One advisory lock prevents two copies of this harness on the same host.
    import fcntl
    with open(Path(tempfile.gettempdir()) / f"maskfree-benchmark-{selected_uuid}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, command in enumerate(commands):
            records = []
            with (args.output / f"command-{index}.log").open("w") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while child.poll() is None:
                        snap = resource_snapshot(pid=child.pid, data_roots=[c.data_root for c in configs.values()],
                                                 report_dir=args.output, include_gpu=True, include_children=True)
                        records.append(snap)
                        if len(records) >= 600:
                            raise RuntimeError("bounded resource sampling limit reached; child gate exceeded 30 minutes")
                        try:
                            child.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)  # only this harness-owned process group
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                    payload = {"command": command, "returncode": child.returncode, "samples": records,
                               "deltas": [compute_resource_deltas(a, b) for a, b in zip(records, records[1:])]}
                    (args.output / f"resources-{index}.json").write_text(json.dumps(payload, indent=2) + "\n")
            if child.returncode:
                print(f"Gate failed; inspect {args.output / f'command-{index}.log'}", file=sys.stderr)
                return child.returncode
    print(f"Bounded gates completed. Review {args.output / 'profile/profile_report.json'}; full training was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
