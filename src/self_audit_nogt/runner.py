"""Bounded, sequential image-only experiments. References live in another process."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import resource
import time

import nibabel as nib
import numpy as np
import torch

from .core import SmallAnnotator, cache_loss, canonicalize, cluster_image, direct_loss, name_partition
from .data import ReadFirewall, crop, native_volume, prepare_images, sha256


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False))


def arrays(cache):
    return (np.load(cache / "images.npy", mmap_mode="r"),
            np.load(cache / "supports.npy", mmap_mode="r"),
            json.loads((cache / "manifest.json").read_text()))


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "self_audit_nogt").glob("*.py"))
    paths += [root / "self_audit_maskfree/ontology.py", root / "self_audit_maskfree/contracts.py"]
    return {str(p.relative_to(root)): sha256(p) for p in paths}


def infer(model, images, indices, batch_size=4):
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            x = torch.from_numpy(np.array(images[indices[start:start + batch_size]], copy=True))[:, None]
            result.extend(model(x).argmax(1).numpy().astype(np.uint8))
    return result


def export_predictions(cache, out, model=None, all_partitions=None, step=0):
    """Names and exports before evaluator. Native output contains UNKNOWN, not draft."""
    images, supports, manifest = arrays(cache)
    indices = [u["index"] for u in manifest["units"] if u["split"] == "dev"]
    units = [manifest["units"][i] for i in indices]
    t0 = time.perf_counter()
    predictions = ([all_partitions[i] for i in indices] if model is None else infer(model, images, indices))
    forward_seconds = time.perf_counter() - t0
    target = np.load(cache / "partitions.npy", mmap_mode="r")
    named, anonymous, rows = {}, {}, []
    rules = Counter()
    agreements, agreement_weights = [], []
    naming_seconds = 0.0
    for unit, prediction in zip(units, predictions):
        i, box = unit["index"], unit["box"]
        x = crop(images[i], box)
        p = canonicalize(crop(prediction, box), x)
        agreements.append(float((p == crop(target[i], box)).mean()))
        agreement_weights.append(p.size)
        start = time.perf_counter()
        result = name_partition(x, p, unit["volume"] + "_" + str(unit["z"]))
        naming_seconds += time.perf_counter() - start
        top, left, h, w = box
        n = np.full(images[i].shape, 255, np.uint8)
        a = np.zeros(images[i].shape, np.uint8)
        n[top:top + h, left:left + w] = result["named"]
        a[top:top + h, left:left + w] = p
        named.setdefault(unit["volume"], []).append(n)
        anonymous.setdefault(unit["volume"], []).append(a)
        trace = result["trace"]
        rules.update(trace["rules_fired"])
        rows.append({"unit": unit["volume"] + "_" + str(unit["z"]), "patient": unit["patient"],
                     "pixels": int(p.size), "unknown_pixels": int((result["named"] == 255).sum()),
                     "valid_per_class": [int((result["named"] == k).sum()) for k in range(4)],
                     "unresolved": result["unresolved"], "trace": trace})
    dest = out / f"step_{step:04d}"
    dest.mkdir(parents=True, exist_ok=False)
    start_export = time.perf_counter()
    for vol in manifest["volumes"]:
        if vol["id"] not in named:
            continue
        n = native_volume(np.stack(named[vol["id"]]), vol)
        a = native_volume(np.stack(anonymous[vol["id"]]), vol)
        np.savez_compressed(dest / (vol["id"] + ".npz"), named=n, partition=a,
                            affine=np.array(vol["affine"]), native_shape=np.array(vol["native_shape"]))
        # Final artifact has an explicit unknown ID plus a separate validity map.
        if step in (0, 1024):
            header = nib.load(str(cache / "images_only" / vol["filename"])).header.copy()
            header.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(n, np.array(vol["affine"]), header), dest / (vol["id"] + "_named.nii.gz"))
            nib.save(nib.Nifti1Image((n != 255).astype(np.uint8), np.array(vol["affine"]), header),
                     dest / (vol["id"] + "_valid.nii.gz"))
    totals = np.array([r["valid_per_class"] for r in rows]).sum(0).tolist()
    total_pixels = sum(r["pixels"] for r in rows)
    summary = {"step": step, "units": len(rows), "rules_fired": dict(rules), "valid_per_class": totals,
               "valid_non_bg_fraction": sum(totals[1:]) / total_pixels,
               "unknown_fraction": sum(r["unknown_pixels"] for r in rows) / total_pixels,
               "unresolved_unit_fraction": sum(r["unresolved"] for r in rows) / len(rows),
               "r2_merges": sum(len(r["trace"].get("merges", [])) for r in rows),
               "anonymous_agreement": float(np.average(agreements, weights=agreement_weights)),
               "forward_seconds": forward_seconds, "naming_seconds": naming_seconds,
               "native_export_seconds": time.perf_counter() - start_export}
    write_json(dest / "ontology_rows.json", rows)
    write_json(dest / "image_only_summary.json", summary)
    return summary


def freeze(out, cache, config, checkpoint=None):
    manifest = {"schema": "nogt.freeze.v1", "time_unix": time.time(), "sources": source_hashes(),
                "config": config, "input_manifest_sha256": sha256(cache / "manifest.json"),
                "split_sha256": sha256(cache / "patient_split.json"),
                "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
                "selection_rule": "fixed update, no reference metrics available",
                "mapping": {"0": "BG", "1": "RV", "2": "MYO", "3": "LV", "255": "UNKNOWN"},
                "predictions": {str(p.relative_to(out)): sha256(p) for p in sorted(out.glob("step_*/*.npz"))}}
    write_json(out / "FROZEN.json", manifest)


def benchmark(cache, model=None, repeats=32):
    images, _, manifest = arrays(cache)
    dev = [u for u in manifest["units"] if u["split"] == "dev"]
    # Same predeclared evenly spaced development states, three warmup calls.
    chosen = [dev[i] for i in np.linspace(0, len(dev) - 1, repeats, dtype=int)]
    values, parts = [], []
    for j, unit in enumerate(chosen[:3] + chosen):
        x = crop(images[unit["index"]], unit["box"])
        t0 = time.perf_counter()
        if model is None:
            p = cluster_image(x)
        else:
            with torch.no_grad():
                z = model(torch.from_numpy(np.array(images[unit["index"]], copy=True))[None, None])
            p = canonicalize(crop(z.argmax(1)[0].numpy().astype(np.uint8), unit["box"]), x)
        t1 = time.perf_counter()
        name_partition(x, p)
        t2 = time.perf_counter()
        if j >= 3:
            parts.append(t1 - t0)
            values.append(t2 - t0)
    return {"warmup": 3, "batch": 1, "repeats": repeats, "device": "cpu",
            "stage": "preloaded image to named output; disk/preparation/native export separately timed",
            "end_to_end_seconds": values, "partition_forward_seconds": parts,
            "p50": float(np.median(values)), "p90": float(np.quantile(values, .9)),
            "partition_forward_p50": float(np.median(parts))}


def anchor(cache, out, config):
    out.mkdir(parents=True, exist_ok=False)
    images, _, manifest = arrays(cache)
    partitions = np.zeros(images.shape, dtype=np.uint8)
    times = []
    t0 = time.perf_counter()
    for u in manifest["units"]:
        i = u["index"]
        top, left, h, w = u["box"]
        start = time.perf_counter()
        partitions[i, top:top + h, left:left + w] = cluster_image(crop(images[i], u["box"]))
        times.append(time.perf_counter() - start)
        if (i + 1) % 400 == 0:
            print(json.dumps({"stage": "anchor", "units": i + 1}), flush=True)
    np.save(cache / "partitions.npy", partitions)
    generation = time.perf_counter() - t0
    summary = export_predictions(cache, out, all_partitions=partitions)
    timing = {"generation_seconds": generation, "units": len(times), "seconds_per_unit": times,
              "latency": benchmark(cache), "export_summary": summary,
              "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(out / "timing.json", timing)
    freeze(out, cache, config)
    print(json.dumps({"stage": "anchor_done", "summary": summary, "generation_seconds": generation}), flush=True)


def train(cache, out, mode, seed, config, updates=None):
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = SmallAnnotator()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    images, supports, manifest = arrays(cache)
    targets = np.load(cache / "partitions.npy", mmap_mode="r")
    indices = np.array([u["index"] for u in manifest["units"] if u["split"] == "train"])
    updates = config["updates"] if updates is None else updates
    checkpoints = set(config["checkpoints"]) if updates == config["updates"] else {updates}
    summary = []
    if 0 in checkpoints:
        summary.append(export_predictions(cache, out, model=model, step=0))
    train_seconds = 0.0
    t0 = time.perf_counter()
    rows = []
    grad_evidence = {}
    for step in range(1, updates + 1):
        if time.perf_counter() - t0 > 2700:
            raise RuntimeError("Predeclared 45-minute arm cap exceeded; INCOMPLETE")
        start = time.perf_counter()
        chosen = rng.choice(indices, size=config["batch_size"], replace=False)
        x = torch.from_numpy(np.array(images[chosen], copy=True))[:, None]
        mask = torch.from_numpy(np.array(supports[chosen], copy=True))
        model.train()
        logits = model(x)
        if mode == "cache":
            loss = cache_loss(logits, torch.from_numpy(np.array(targets[chosen], copy=True)), mask)
        else:
            loss = direct_loss(logits, x, mask)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1:
            grad_evidence = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite gradient")
        optimizer.step()
        train_seconds += time.perf_counter() - start
        rows.append({"step": step, "examples": step * config["batch_size"], "loss": float(loss.detach()),
                     "train_seconds": train_seconds, "elapsed_seconds": time.perf_counter() - t0,
                     "sample_index_hash": hashlib.sha256(chosen.tobytes()).hexdigest()})
        if step in checkpoints:
            s = export_predictions(cache, out, model=model, step=step)
            summary.append(s)
            print(json.dumps({"mode": mode, "seed": seed, "step": step,
                              "loss": rows[-1]["loss"], "agreement": s["anonymous_agreement"],
                              "valid_non_bg": s["valid_non_bg_fraction"], "train_seconds": train_seconds}), flush=True)
    torch.save({"model": model.state_dict(), "seed": seed, "mode": mode, "updates": updates,
                "pretraining": None, "checkpoint_rule": "fixed last update"}, out / "final.pt")
    with (out / "curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "timing.json", {"mode": mode, "seed": seed, "updates": updates,
                 "examples": updates * config["batch_size"], "train_seconds": train_seconds,
                 "train_validation_export_seconds": time.perf_counter() - t0,
                 "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 "parameters": sum(p.numel() for p in model.parameters()), "initial_gradients": grad_evidence,
                 "checkpoints": summary, "latency": benchmark(cache, model)})
    freeze(out, cache, {**config, "mode": mode, "seed": seed, "actual_updates": updates}, out / "final.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "anchor", "train"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--mode", choices=["cache", "direct"], default="cache")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    firewall = ReadFirewall()
    t0 = time.perf_counter()
    try:
        with firewall:
            if args.stage == "prepare":
                args.out.mkdir(parents=True, exist_ok=True)
                m = prepare_images(args.raw, args.split, args.cache, config["resolution"])
                write_json(args.out / "preparation.json", {"seconds": time.perf_counter() - t0,
                           "volumes": len(m["volumes"]), "slices": len(m["units"]),
                           "train_slices": sum(u["split"] == "train" for u in m["units"]),
                           "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            elif args.stage == "anchor":
                anchor(args.cache, args.out, config)
            else:
                train(args.cache, args.out, args.mode, args.seed, config, args.updates)
    finally:
        args.out.mkdir(parents=True, exist_ok=True)
        firewall.save(args.out / "file_reads.json")


if __name__ == "__main__":
    main()
