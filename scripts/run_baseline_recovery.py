#!/usr/bin/env python3
"""Baseline recovery runner, historical lineage verifier, and gate orchestrator.

Ordered executable workflow:
1. verify historical source/checkpoint/config/label-lineage and payload hash before reuse;
   fail closed on missing files/claims.
2. Predeclare gated protocol before run (no TEST hyperparameter selection, preregistered step 10000).
3. Bounded profile and tiny-fit diagnostic commands (actual latency/memory/throughput, not simulated).
4. Baseline training via only train.py with baseline_recovery engine configuration.
5. Evaluate baseline gate (FG Dice >= historic - 0.02, all foreground noncollapse).
6. Block shared training unless baseline gate evidence is valid and tied to source/weights/data.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.data import write_json
from self_audit.model import digest
from self_audit.recovery_gates import (
    HISTORICAL_COMMIT,
    HISTORICAL_SOURCE_SHA256,
    HISTORICAL_MANIFEST_SHA256,
    HISTORICAL_PAYLOAD_SHA256,
    HISTORICAL_DATA_SHA256,
    HISTORICAL_DINO_WEIGHTS_SHA256,
    HISTORICAL_ACDC_LIDS,
    HISTORIC_FG_DICE,
    HISTORIC_ALL_DICE,
    BASELINE_TOLERANCE_FG,
    CERTIFICATE_SCHEMA,
    PREREGISTERED_PROTOCOL_SCHEMA,
    RecoveryFailureType,
    RecoveryGateError,
    MissingEvidenceError,
    ImplementationVerificationError,
    ConvergenceError,
    EmpiricalGateFailureError,
    verify_historical_lineage,
    verify_baseline_protocol_config,
    evaluate_baseline_gate,
    build_baseline_gate_certificate,
    verify_baseline_gate_certificate,
    block_shared_training_unless_baseline_gated,
    run_tiny_fit_diagnostic,
)


def run_bounded_profile(config_path, manifest_path, manifest_sha256, out_dir, *,
                        model_config_path=None, device="cpu", profile_steps=4, synthetic=False):
    """Profile the canonical trainer's initial warmup; no second training implementation."""
    from self_audit import engine
    if type(profile_steps) is not int or not 1 <= profile_steps <= 8:
        raise ValueError("profile_steps must be an integer in 1..8")
    out = Path(out_dir).resolve()
    if out.exists(): raise ValueError("Profile output must be new")
    cfg = json.loads(Path(config_path).read_text())
    if model_config_path: cfg["model"] = json.loads(Path(model_config_path).read_text())
    cfg = engine.validate_config(cfg)
    original_steps, original_warmup = cfg["steps"], cfg["warmup_steps"]
    cfg["steps"] = profile_steps
    cfg["warmup_steps"] = min(original_warmup,profile_steps)
    br = cfg.get("baseline_recovery") or {}
    cfg["baseline_recovery"] = {**br,"enabled":True,"purpose":"profile"}
    # Keep the warmup gate enabled. A diagnostic ending before it is never a quality certificate.
    out.mkdir(parents=True)
    is_cuda = device.startswith("cuda")
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    stamps = []
    original_ema = engine.ema_update
    def timed_ema(*args, **kwargs):
        result = original_ema(*args, **kwargs)
        if is_cuda: torch.cuda.synchronize(device)
        stamps.append(time.perf_counter())
        return result
    started = time.perf_counter()
    engine.ema_update = timed_ema
    try:
        report = engine.train(cfg,manifest_path,manifest_sha256,out/"run",device=device,synthetic=synthetic)
    finally:
        engine.ema_update = original_ema
    if is_cuda: torch.cuda.synchronize(device)
    cold_total = time.perf_counter()-started
    rows = [json.loads(s) for s in (out/"run/steps.jsonl").read_text().splitlines()]
    times = np.diff([started,*stamps]).tolist()
    marginal = times[1:]
    optimized = sum(report["slices_optimized"].values())
    result = {"scope":"bounded diagnostic; NOT patient quality or convergence evidence",
              "synthetic":synthetic,"device":device,"is_cuda":is_cuda,"profile_steps":profile_steps,
              "recipe_steps":original_steps,"recipe_warmup_steps":original_warmup,
              "effective_warmup_steps":cfg["warmup_steps"],"cold_total_seconds":cold_total,
              "cold_first_commit_seconds":times[0],"commit_latencies_seconds":times,
              "marginal_step_seconds":float(np.mean(marginal)) if marginal else None,
              "optimized_slices_per_cold_second":optimized/cold_total,
              "slices_sampled":report["slices_sampled"],"slices_optimized":report["slices_optimized"],
              "optimizer_calls":report["optimizer_calls"],"ema_updates":report["ema_updates"],
              "peak_vram_allocated_mib":torch.cuda.max_memory_allocated(device)/2**20 if is_cuda else None,
              "peak_vram_reserved_mib":torch.cuda.max_memory_reserved(device)/2**20 if is_cuda else None,
              "pseudo_phase_measured":any(r.get("pseudo_optimizer_updates",0)>0 for r in rows),
              "limitation":"An initial warmup profile does not measure post-warmup pseudo-training throughput",
              "source_sha256":engine.source_digest(),"data_sha256":manifest_sha256,
              "effective_config_sha256":digest(out/"run/CONFIG.json")}
    write_json(out/"PROFILE_REPORT.json",result)
    return result


def predeclare_recovery_protocol(out_path, *, protocol_type="baseline", target_steps=10000,
                                 warmup_steps=1000, seed=42, tolerance=BASELINE_TOLERANCE_FG,
                                 config_path=None, model_config_path=None, train_manifest_path=None,
                                 evaluation_manifest_path=None, synthetic=False):
    from self_audit.engine import validate_config, source_digest
    from self_audit.recovery_gates import HISTORICAL_TEST_MANIFEST_SHA256
    if any(p is None for p in (config_path,train_manifest_path,evaluation_manifest_path)):
        raise MissingEvidenceError("Predeclare requires resolved config, TRAIN and evaluation manifests")
    if Path(out_path).exists(): raise ValueError("Protocol output must be new; do not replace a locked protocol")
    cfg = json.loads(Path(config_path).read_text())
    if model_config_path: cfg["model"] = json.loads(Path(model_config_path).read_text())
    cfg = validate_config(cfg)
    if not synthetic:
        verify_baseline_protocol_config(cfg)
        if protocol_type == "baseline" and digest(train_manifest_path) != HISTORICAL_DATA_SHA256:
            raise ImplementationVerificationError("Baseline TRAIN must match the locked L8/U72 manifest")
        if protocol_type == "baseline" and digest(evaluation_manifest_path) != HISTORICAL_TEST_MANIFEST_SHA256:
            raise ImplementationVerificationError("Baseline comparison must use the locked TEST50 development manifest")
    if (target_steps,warmup_steps,seed) != (cfg["steps"],cfg["warmup_steps"],cfg["seed"]):
        raise ImplementationVerificationError("Protocol schedule must match resolved config")
    if tolerance != BASELINE_TOLERANCE_FG:
        raise ImplementationVerificationError("Baseline tolerance is locked")
    proto = {"schema":PREREGISTERED_PROTOCOL_SCHEMA,"protocol_type":protocol_type,
             "synthetic":synthetic,"created_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
             "executed_source_sha256":source_digest(),"tinyfit_source_sha256":source_digest(),
             "config":cfg,"target_steps":target_steps,"warmup_steps":warmup_steps,"seed":seed,
             "train_manifest_sha256":digest(train_manifest_path),
             "evaluation_manifest_sha256":digest(evaluation_manifest_path),"tolerance":tolerance,
             "checkpoint_selection":"final step only; never best TEST score",
             "historical_reference":{"commit":HISTORICAL_COMMIT,"foreground":HISTORIC_FG_DICE,
                                     "all_class_including_BG":HISTORIC_ALL_DICE},
             "shared_development_thresholds":{"foreground_per_domain":.75,"each_foreground_class":.60}}
    write_json(out_path,proto)
    return proto


def dispatch_baseline_training(config_path, data_manifest, data_sha256, out_dir, *,
                               model_config=None, device="cuda", synthetic=False,
                               allow_full_run=False, protocol_path=None, protocol_sha256=None):
    from self_audit.engine import validate_config, source_digest
    if not protocol_path or not protocol_sha256 or digest(protocol_path) != protocol_sha256:
        raise MissingEvidenceError("Training requires a SHA-pinned preregistered protocol")
    proto = json.loads(Path(protocol_path).read_text())
    cfg = json.loads(Path(config_path).read_text())
    if model_config: cfg["model"] = json.loads(Path(model_config).read_text())
    cfg = validate_config(cfg)
    if (proto.get("config") != cfg or proto.get("executed_source_sha256") != source_digest()
            or proto.get("train_manifest_sha256") != data_sha256 or digest(data_manifest) != data_sha256):
        raise ImplementationVerificationError("Training source/config/data differs from preregistration")
    cmd = [sys.executable,str(ROOT/"train.py"),"--config",str(Path(config_path).resolve()),
           "--data",str(Path(data_manifest).resolve()),"--data-sha256",data_sha256,
           "--out",str(Path(out_dir).resolve()),"--device",device]
    if model_config: cmd += ["--model-config",str(Path(model_config).resolve())]
    if synthetic: cmd.append("--synthetic")
    if allow_full_run: cmd.append("--allow-full-run")
    return subprocess.run(cmd,cwd=ROOT).returncode



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. Verify history
    vh = sub.add_parser("verify-history", help="Verify historical lineage and payload hashes")
    vh.add_argument("--manifest", default="reports/window_pilot_test50_20261024/baseline/CHECKPOINT_MANIFEST.json")
    vh.add_argument("--payload", help="Optional path to weights.pt")
    vh.add_argument("--source-digest", help="Optional executed source digest to verify")
    vh.add_argument("--require-payload", action="store_true", help="Fail closed if physical weights.pt missing")
    vh.add_argument("--out", help="Output JSON receipt path")

    # 2. Profile
    pr = sub.add_parser("profile", help="Run short bounded CPU/CUDA execution profile")
    pr.add_argument("--config", default="configs/baseline_recovery/baseline_l8_10k.json")
    pr.add_argument("--model-config", help="Optional path to model override config JSON")
    pr.add_argument("--data", required=True)
    pr.add_argument("--data-sha256", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--device", default="cpu")
    pr.add_argument("--steps", type=int, default=4)
    pr.add_argument("--synthetic", action="store_true")

    # 3. Tiny fit diagnostic
    tf = sub.add_parser("tiny-fit", help="Run 2-patient diagnostic fit gate")
    tf.add_argument("--data", required=True)
    tf.add_argument("--data-sha256", required=True)
    tf.add_argument("--model-config", help="Path to model override config JSON with real pinned asset paths")
    tf.add_argument("--out", required=True)
    tf.add_argument("--device", default="cpu")
    tf.add_argument("--max-steps", type=int, default=200)
    tf.add_argument("--target-dice", type=float, default=0.95)
    tf.add_argument("--diagnostic-lr-factor", type=float, default=1.0)
    tf.add_argument("--synthetic", action="store_true")

    # 4. Predeclare
    pd = sub.add_parser("predeclare", help="Predeclare baseline recovery protocol")
    pd.add_argument("--out", required=True)
    pd.add_argument("--protocol-type", default="baseline", choices=["baseline", "shared"])
    pd.add_argument("--config", required=True)
    pd.add_argument("--model-config")
    pd.add_argument("--train-manifest", required=True)
    pd.add_argument("--evaluation-manifest", required=True)
    pd.add_argument("--synthetic", action="store_true")
    pd.add_argument("--steps", type=int, default=10000)
    pd.add_argument("--warmup-steps", type=int, default=1000)
    pd.add_argument("--seed", type=int, default=42)

    # 5. Gate check
    gc = sub.add_parser("gate-check", help="Evaluate baseline gate and build physical evidence bundle certificate")
    gc.add_argument("--checkpoint-manifest", required=True)
    gc.add_argument("--checkpoint-payload", help="Path to weights.pt (optional if alongside manifest)")
    gc.add_argument("--train-manifest", required=True)
    gc.add_argument("--train-config", required=True)
    gc.add_argument("--protocol", required=True)
    gc.add_argument("--frozen-predictions", required=True)
    gc.add_argument("--metrics", required=True)
    gc.add_argument("--evaluation-manifest", required=True)
    gc.add_argument("--tinyfit-receipt", required=True, help="Mandatory path to TINY_FIT_REPORT.json")
    gc.add_argument("--synthetic", action="store_true")
    gc.add_argument("--out", required=True, help="Output certificate JSON path")

    # 6. Verify shared prerequisite gate
    sg = sub.add_parser("check-shared-prereq", help="Verify baseline certificate before shared training")
    sg.add_argument("--certificate", required=True)
    sg.add_argument("--expected-cert-sha", help="Expected certificate SHA-256")
    sg.add_argument("--shared-manifest", help="Shared training data manifest path")
    sg.add_argument("--shared-config", help="Shared training configuration JSON path")

    # 7. Train baseline
    tb = sub.add_parser("train-baseline", help="Dispatch baseline training strictly via train.py")
    tb.add_argument("--config", default="configs/baseline_recovery/baseline_l8_10k.json")
    tb.add_argument("--model-config", help="Path to model override config JSON")
    tb.add_argument("--data", required=True)
    tb.add_argument("--data-sha256", required=True)
    tb.add_argument("--out", required=True)
    tb.add_argument("--protocol", required=True)
    tb.add_argument("--protocol-sha256", required=True)
    tb.add_argument("--device", default="cuda")
    tb.add_argument("--allow-full-run", action="store_true")
    tb.add_argument("--synthetic", action="store_true")

    args = parser.parse_args(argv)

    try:
        if args.command == "verify-history":
            res = verify_historical_lineage(
                args.manifest,
                payload_path=args.payload,
                executed_source_digest=args.source_digest,
                require_payload_exists=args.require_payload,
            )
            if args.out:
                write_json(args.out, res)
            print(json.dumps({"status": "PASS", **res}, indent=2))
            return 0

        elif args.command == "profile":
            res = run_bounded_profile(
                args.config, args.data, args.data_sha256, args.out,
                model_config_path=args.model_config,
                device=args.device, profile_steps=args.steps, synthetic=args.synthetic
            )
            print(json.dumps({"status": "PASS", **res}, indent=2))
            return 0

        elif args.command == "tiny-fit":
            res = run_tiny_fit_diagnostic(
                args.data, args.data_sha256, args.out,
                model_config_path=args.model_config,
                device=args.device, max_steps=args.max_steps,
                target_fg_dice=args.target_dice,
                diagnostic_lr_factor=args.diagnostic_lr_factor,
                synthetic=args.synthetic
            )
            print(json.dumps({"status": "PASS", **res}, indent=2))
            return 0

        elif args.command == "predeclare":
            res = predeclare_recovery_protocol(
                args.out, protocol_type=args.protocol_type,
                target_steps=args.steps, warmup_steps=args.warmup_steps, seed=args.seed,
                config_path=args.config, model_config_path=args.model_config,
                train_manifest_path=args.train_manifest, evaluation_manifest_path=args.evaluation_manifest,
                synthetic=args.synthetic
            )
            print(json.dumps({"status": "PREDECLARED", "path": args.out}, indent=2))
            return 0

        elif args.command == "gate-check":
            cert = build_baseline_gate_certificate(
                checkpoint_manifest_path=args.checkpoint_manifest,
                checkpoint_payload_path=args.checkpoint_payload,
                train_manifest_path=args.train_manifest,
                train_config_path=args.train_config,
                protocol_path=args.protocol,
                frozen_predictions_path=args.frozen_predictions,
                metrics_path=args.metrics,
                evaluation_manifest_path=args.evaluation_manifest,
                tinyfit_receipt_path=args.tinyfit_receipt,
                synthetic=args.synthetic,
                out_path=args.out,
            )
            print(json.dumps({"status": "GATE_PASSED", "certificate": args.out}, indent=2))
            return 0

        elif args.command == "check-shared-prereq":
            shared_cfg = json.loads(Path(args.shared_config).read_text()) if args.shared_config else None
            cert = block_shared_training_unless_baseline_gated(
                args.certificate,
                expected_cert_sha=args.expected_cert_sha,
                shared_manifest_path=args.shared_manifest,
                shared_config=shared_cfg,
            )
            print(json.dumps({"status": "SHARED_UNBLOCKED", "certificate_status": cert.get("status")}, indent=2))
            return 0

        elif args.command == "train-baseline":
            return dispatch_baseline_training(
                args.config, args.data, args.data_sha256, args.out,
                model_config=args.model_config,
                device=args.device, synthetic=args.synthetic,
                allow_full_run=args.allow_full_run,
                protocol_path=args.protocol, protocol_sha256=args.protocol_sha256,
            )

    except RecoveryGateError as exc:
        print(json.dumps({
            "status": "FAIL",
            "failure_classification": exc.failure_type.value,
            "error": str(exc),
            "details": exc.details,
        }, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
