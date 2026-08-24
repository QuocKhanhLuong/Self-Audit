#!/usr/bin/env python3
"""Standalone CLI tool to visualize predictions, transitions, and patient volume QA.

Usage examples:
    # 1. Visualize slice predictions from a Phase A, B, or C checkpoint on validation set
    python scripts/visualize_predictions.py --checkpoint weights/self_audit/phase_c_joint.pt --config configs/self_audit_joint.yaml --num_samples 6 --output_dir reports/visualizations

    # 2. Visualize full multi-slice QA grid for a specific patient volume
    python scripts/visualize_predictions.py --checkpoint weights/self_audit/phase_c_joint.pt --config configs/self_audit_joint.yaml --patient patient001 --output_dir reports/visualizations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from self_audit.evaluation.visualizer import (
    plot_patient_volume_qa,
    plot_phase_a_samples,
    plot_phase_c_audit_trace,
    save_figure,
)
from self_audit.evaluation.volume_inference import infer_patient_volume
from self_audit.training._utils import (
    build_data_loader,
    build_model_from_config,
    build_patient_dataset,
    load_checkpoint,
    load_config,
    move_batch,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Audit Prediction Visualizer & Image Exporter")
    parser.add_argument("--checkpoint", default="weights/self_audit/best.pt", help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/self_audit_joint.yaml", help="Path to config YAML")
    parser.add_argument("--data_root", default=None, help="Dataset root directory override")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to visualize")
    parser.add_argument("--patient", default=None, help="Specific patient ID to visualize volume QA (e.g. patient001)")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of 2D slices to visualize")
    parser.add_argument("--output_dir", default="reports/visualizations", help="Output directory for saved images")
    parser.add_argument("--device", default=None, help="Compute device (cuda / cpu)")
    parser.add_argument("--tau_accept", type=float, default=0.0, help="Self-audit acceptance threshold")
    parser.add_argument("--t_max", type=int, default=3, help="Maximum refinement turns")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    device = resolve_device(args.device or config.get("device"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Visualizer] Loading model from {args.checkpoint} on {device}...")
    model = build_model_from_config(config, device)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_file():
        load_checkpoint(ckpt_path, model, map_location=device, strict=False)
        print(f"[Visualizer] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[Visualizer Warning] Checkpoint not found at {ckpt_path}. Using initialized weights.")

    model.eval()

    # Case 1: Specific patient volume QA
    if args.patient:
        print(f"[Visualizer] Generating Volume QA for patient {args.patient}...")
        dataset = build_patient_dataset(config, split=args.split, train=False)
        found_records = [r for r in getattr(dataset, "records", []) if args.patient.lower() in str(r.case_id).lower() or args.patient.lower() in str(r.patient_id).lower()]
        if not found_records:
            print(f"[Visualizer Error] No records found matching patient '{args.patient}' in split '{args.split}'.")
            return
        rec = found_records[0]
        from self_audit.data.common import load_array, to_depth_first
        vol, _ = load_array(rec.image_path)
        gt, _ = load_array(rec.mask_path)
        d_axis = config.get("depth_axis", 2 if rec.source_format == "npy" else None)
        vol_zhw = to_depth_first(vol, depth_axis=d_axis)
        gt_zhw = to_depth_first(gt, depth_axis=d_axis)

        res = infer_patient_volume(
            model,
            vol_zhw,
            image_size=int(config.get("image_size", 256)),
            depth_axis=0,  # already ZHW
            mode="self_audit",
            tau_accept=float(args.tau_accept),
            t_max=int(args.t_max),
            device=device,
        )
        fig = plot_patient_volume_qa(
            vol_zhw,
            gt_zhw,
            res.prediction.numpy(),
            initial_pred_volume=res.initial_prediction.numpy(),
            patient_id=rec.case_id,
        )
        out_file = out_dir / f"volume_qa_{rec.case_id}.png"
        saved = save_figure(fig, out_file)
        print(f"[Visualizer Success] Saved Volume QA figure to: {saved}")
        return

    # Case 2: Slice samples from split loader
    print(f"[Visualizer] Extracting {args.num_samples} sample slices from split '{args.split}'...")
    dataset = build_patient_dataset(config, split=args.split, train=False)
    loader = build_data_loader(dataset, config, device=device, train=False, batch_size=args.num_samples)

    sample_imgs, sample_msks, sample_inits, sample_finals, sample_cids = [], [], [], [], []
    sample_candidates, sample_halts = [], []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model.infer(batch["image"], mode="self_audit", tau_accept=float(args.tau_accept), t_max=int(args.t_max))
            init_log = out.get("initial_logits", out.get("a0_logits", out.get("logits")))
            fin_log = out.get("logits")
            cands = out.get("transition_candidates", [])
            halts = out.get("halt_turn")
            for idx in range(batch["image"].shape[0]):
                sample_imgs.append(batch["image"][idx].cpu())
                sample_msks.append(batch["mask"][idx].cpu())
                sample_inits.append(init_log[idx].argmax(dim=0).cpu())
                sample_finals.append(fin_log[idx].argmax(dim=0).cpu())
                cids = batch.get("case_id", [f"Case_{len(sample_imgs)}"])
                sample_cids.append(cids[idx] if isinstance(cids, list) else f"Case_{len(sample_imgs)}")
                if halts is not None and torch.is_tensor(halts):
                    sample_halts.append(int(halts[idx].cpu()))
                if len(sample_imgs) >= args.num_samples:
                    break
            if cands:
                sample_candidates.append([c.argmax(dim=1).cpu() for c in cands])
            if len(sample_imgs) >= args.num_samples:
                break

    if not sample_imgs:
        print("[Visualizer Warning] No samples extracted from dataset.")
        return

    # Save 2D multi-turn inspection figure
    fig = plot_phase_c_audit_trace(
        sample_imgs[:args.num_samples],
        sample_msks[:args.num_samples],
        sample_inits[:args.num_samples],
        sample_finals[:args.num_samples],
        transition_candidates=sample_candidates[0] if sample_candidates else None,
        halted_turns=sample_halts[:args.num_samples] if sample_halts else None,
        tau_accept=float(args.tau_accept),
        case_ids=sample_cids[:args.num_samples],
        title=f"Self-Audit Refinement Inspection ({args.split} split)",
    )
    out_file = out_dir / f"self_audit_{args.split}_samples.png"
    saved = save_figure(fig, out_file)
    print(f"[Visualizer Success] Saved Slice Inspection figure to: {saved}")


if __name__ == "__main__":
    main()
