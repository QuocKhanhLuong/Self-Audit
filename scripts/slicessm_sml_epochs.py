#!/usr/bin/env python3
"""Per-epoch report of the SliceSSM S/M/L width ablation, with parameters, GFLOPs and inference time.

Reads each arm's ``arms/A2/epoch_full_metrics.jsonl`` (val10: Dice, Jaccard, HD95 and ASD per epoch, native mm and SSL4MIS voxel units) and ``epochs.jsonl`` (train loss and
train FG Dice).  Missing epochs (a run still in progress) are left empty, so the script can be re-run while the arms train.  Model cost is measured here with one common
protocol for all sizes: the deployed model, batch 1, input 1x1x10x256x232 (a typical native ACDC volume), fp32, 10 warm-up and 30 timed forward passes on GPU and on CPU
(4 threads).  GPU timings are noisy while other jobs share the card; re-run on an idle GPU for final numbers.
Outputs: ``<out>.md`` (tables), ``<out>_per_epoch.csv`` (every epoch, every size) and ``<out>.png`` (val Dice / HD95 curves).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

WIDTHS = {"S": (16, 32, 64, 112, 128), "M": (24, 48, 96, 160, 192), "L": (40, 72, 144, 240, 288)}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def per_epoch(arm_dir: Path) -> dict[int, dict]:
    a2 = arm_dir / "arms" / "A2"
    rows: dict[int, dict] = {}
    for r in read_jsonl(a2 / "epochs.jsonl"):
        e = rows.setdefault(int(r["epoch"]), {})
        if r["split"] == "train":
            e["train_loss"], e["train_fg_dice"] = r.get("loss_total"), r.get("fg_dice")
        elif r["split"] == "val":
            e["val_fg_dice_logger"] = r.get("fg_dice")
    for r in read_jsonl(a2 / "epoch_full_metrics.jsonl"):
        v = r.get("val")
        if v:
            e = rows.setdefault(int(r["epoch"]), {})
            nat, vox = v["native_mm"]["mean"], v["ssl4mis_voxels"]["mean"]
            e.update({"val_dice": nat["dice"], "val_jaccard": nat["jaccard"], "val_hd95_mm": nat["hd95"], "val_asd_mm": nat["asd"], "val_hd95_vox": vox["hd95"], "val_asd_vox": vox["asd"]})
    return rows


def cost(channels: tuple[int, ...], device_gpu: bool) -> dict:
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from self_audit.candidate_pg.slicessm_student import SliceSSMStudent, export_slicessm_student

    torch.set_num_threads(4)
    full = SliceSSMStudent(None, channels=channels)
    import tempfile
    path = Path(tempfile.mkdtemp()) / "m.pt"
    deployed_params = export_slicessm_student(full, path)["parameters_deployed"]
    from self_audit.candidate_pg.slicessm_student import load_slicessm_export

    model = load_slicessm_export(path).eval()
    x, geo = torch.rand(1, 1, 10, 256, 232), {"shape": (10, 256, 232), "spacing_zyx": [10.0, 1.4, 1.4]}
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x, geo)

    def timed(dev: str, runs: int, warm: int) -> float:
        m, xi = model.to(dev), x.to(dev)
        with torch.no_grad():
            for _ in range(warm):
                m(xi, geo)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                m(xi, geo)
            if dev == "cuda":
                torch.cuda.synchronize()
        return (time.perf_counter() - t0) / runs

    out = {"train_params": sum(p.numel() for p in full.parameters()), "deployed_params": deployed_params, "gflops": fc.get_total_flops() / 1e9,
           "cpu_s": timed("cpu", 5, 2), "gpu_s": timed("cuda", 30, 10) if device_gpu else None}
    model.to("cpu")
    return out


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s", type=Path, required=True)
    ap.add_argument("--m", type=Path, required=True)
    ap.add_argument("--l", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="path prefix, e.g. reports/.../SLICESSM_SML_EPOCHS")
    a = ap.parse_args()
    runs = {"S": a.s, "M": a.m, "L": a.l}
    data = {k: per_epoch(d) for k, d in runs.items()}
    selected = {}
    for k, d in runs.items():
        p = d / "FINAL_RESULT.json"
        selected[k] = json.loads(p.read_text())["selected_epoch"] if p.is_file() else None
    costs = {k: cost(WIDTHS[k], torch.cuda.is_available()) for k in runs}
    epochs = sorted({e for k in data for e in data[k]})
    fmt = lambda v, n=4: "" if v is None else f"{v:.{n}f}"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["train_loss", "train_fg_dice", "val_dice", "val_jaccard", "val_hd95_mm", "val_asd_mm", "val_hd95_vox", "val_asd_vox"]
    with a.out.with_name(a.out.name + "_per_epoch.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "size", *cols])
        for e in epochs:
            for k in runs:
                w.writerow([e, k, *[data[k].get(e, {}).get(c, "") for c in cols]])
    L = ["# SliceSSM width ablation S / M / L: per-epoch report", "",
         "Same recipe for all sizes (UniMatch, no teacher, 150 epochs, seed 42); only the widths differ (S 16-32-64-112-128, M = the existing 24-48-96-160-192 model, L 40-72-144-240-288). Metrics are val10 (10 patients, one seed), "
         "native units (HD95/ASD in mm) unless marked `vox` (SSL4MIS protocol, voxels). `*` marks the val10-selected epoch. Empty cells: epoch not yet reached.", "",
         "## Model size and inference time (deployed model, batch 1, 1x1x10x256x232, fp32)", "",
         "| | S | M | L |", "|---|---|---|---|",
         "| parameters, train graph | " + " | ".join(f"{costs[k]['train_params'] / 1e6:.2f} M" for k in runs) + " |",
         "| parameters, deployed | " + " | ".join(f"{costs[k]['deployed_params'] / 1e6:.2f} M" for k in runs) + " |",
         "| GFLOPs (multiply-add = 2) | " + " | ".join(f"{costs[k]['gflops']:.1f}" for k in runs) + " |",
         "| inference time, GPU | " + " | ".join(f"{costs[k]['gpu_s'] * 1000:.1f} ms" if costs[k]["gpu_s"] else "n/a" for k in runs) + " |",
         "| inference time, CPU (4 threads) | " + " | ".join(f"{costs[k]['cpu_s'] * 1000:.0f} ms" for k in runs) + " |",
         "| selected epoch | " + " | ".join(str(selected[k]) if selected[k] else "running" for k in runs) + " |", "",
         "## Per epoch (val10)", "",
         "| epoch | loss S | loss M | loss L | Dice S | Dice M | Dice L | Jaccard S | Jaccard M | Jaccard L | HD95 mm S | HD95 mm M | HD95 mm L | ASD mm S | ASD mm M | ASD mm L |",
         "|---|" + "---|" * 15]
    for e in epochs:
        def cell(key, k, n=4):
            v = data[k].get(e, {}).get(key)
            s = fmt(v, n)
            return s + "*" if s and selected[k] == e and key == "val_dice" else s
        L.append(f"| {e} | " + " | ".join(cell("train_loss", k, 3) for k in runs) + " | " + " | ".join(cell("val_dice", k) for k in runs) + " | " + " | ".join(cell("val_jaccard", k) for k in runs)
                 + " | " + " | ".join(cell("val_hd95_mm", k, 2) for k in runs) + " | " + " | ".join(cell("val_asd_mm", k, 2) for k in runs) + " |")
    a.out.with_suffix(".md").write_text("\n".join(L) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        for k in runs:
            es = [e for e in epochs if "val_dice" in data[k].get(e, {})]
            ax[0].plot(es, [data[k][e]["val_dice"] for e in es], label=f"{k} ({costs[k]['deployed_params'] / 1e6:.1f} M)")
            ax[1].plot(es, [data[k][e]["val_hd95_mm"] for e in es], label=k)
        ax[0].set_ylim(0.8, 0.95); ax[0].set_title("val10 Dice"); ax[1].set_ylim(0, 15); ax[1].set_title("val10 HD95 (mm)")
        for x in ax:
            x.set_xlabel("epoch"); x.legend()
        fig.tight_layout(); fig.savefig(a.out.with_suffix(".png"), dpi=130)
    except Exception as exc:  # noqa: BLE001 - the figure is optional
        print("figure skipped:", exc)
    print(f"wrote {a.out.with_suffix('.md')} ({len(epochs)} epochs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
