#!/usr/bin/env python3
"""Actual resolver interventions; no manual reference is opened."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, median_filter
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from self_audit_nogt.core import name_partition
from self_audit_nogt.data import ReadFirewall, crop


def summary(result):
    return {"rules": result["trace"]["rules_fired"], "assignment": result["trace"]["assignment"],
            "enclosure_table": result["trace"].get("enclosure_table", {}),
            "unknown_fraction": float((result["named"] == 255).mean()),
            "valid_class_pixels": [int((result["named"] == k).sum()) for k in range(4)]}


def open_gap(partition, encloser, enclosed):
    """Fixed image-only horizontal cut from an enclosed group to image exterior.

    Only pixels belonging to the encloser are changed; every cut pixel gets the
    enclosed anonymous ID. Four-neighbour path is explicit, not diagonal erosion.
    """
    y, x = np.argwhere(partition == enclosed)[len(np.argwhere(partition == enclosed)) // 2]
    result = partition.copy()
    segment = result[y, x:]
    segment[segment == encloser] = enclosed
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    guard = ReadFirewall()
    with guard:
        yy, xx = np.mgrid[:96, :96]
        r = (yy - 48)**2 + (xx - 48)**2
        closed = np.zeros((96, 96), np.uint8)
        closed[r < 16**2] = 1
        closed[(r >= 16**2) & (r <= 22**2)] = 2
        image = np.take(np.array([.1, .8, .4], np.float32), closed)
        opened = closed.copy()
        opened[48, 64:72] = 0
        variants = {"closed": closed, "gap": opened,
                    "permuted": np.take(np.array([4, 7, 2], np.uint8), closed),
                    "missing_myo": np.where(closed == 2, 0, closed), "empty": np.zeros_like(closed),
                    "displaced": np.roll(closed, 12, axis=0), "smoothed": median_filter(closed, size=7)}
        for name, operation in [("dilated", binary_dilation), ("eroded", binary_erosion)]:
            candidate = closed.copy()
            candidate[candidate == 2] = 0
            candidate[operation(closed == 2, iterations=2)] = 2
            variants[name] = candidate
        synthetic = {k: summary(name_partition(image, v)) for k, v in variants.items()}
        synthetic["partition_overlap"] = float((closed == opened).mean())
        ring0, ring1 = closed == 2, opened == 2
        synthetic["ring_dice"] = float(2 * (ring0 & ring1).sum() / (ring0.sum() + ring1.sum()))
        images = np.load(args.cache / "images.npy", mmap_mode="r")
        partitions = np.load(args.cache / "partitions.npy", mmap_mode="r")
        manifest = json.loads((args.cache / "manifest.json").read_text())
        real = []
        eligible = 0
        for unit in manifest["units"]:
            if unit["split"] != "dev":
                continue
            x = crop(images[unit["index"]], unit["box"])
            partition = crop(partitions[unit["index"]], unit["box"])
            before = name_partition(x, partition)
            if "R3_enclosure" not in before["trace"]["rules_fired"]:
                continue
            eligible += 1
            mapping = before["trace"]["assignment"]
            myo = next(int(g) for g, role in mapping.items() if role == "MYO")
            lv = next(int(g) for g, role in mapping.items() if role == "LV")
            changed = open_gap(partition, myo, lv)
            after = name_partition(x, changed)
            real.append({"unit": unit["volume"] + "_" + str(unit["z"]),
                         "partition_agreement": float((partition == changed).mean()),
                         "named_agreement": float((before["named"] == after["named"]).mean()),
                         "before": summary(before), "after": summary(after)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"synthetic": synthetic, "real_eligible_enclosure_units": eligible,
                                   "real_interventions": real, "gt_used": False,
                                   "interpretation": "topological sensitivity, not proof of MRI correctness or global root cause"}, indent=2))
    guard.save(args.out.with_name("ontology_file_reads.json"))
    print(json.dumps({"real_enclosure_eligible": eligible,
                      "r3_lost": sum("R3_enclosure" not in r["after"]["rules"] for r in real),
                      "synthetic_ring_dice": synthetic["ring_dice"]}))


if __name__ == "__main__":
    main()
