"""Qualitative comparison on three datasets: input, ground truth and one column per model (real slices only)."""
from __future__ import annotations

import json
from pathlib import Path

from svg_kit import Figure, INK, MUTED, RULE, WHITE

LEGEND = {"cardiac": [("RV", "#E53935"), ("MYO", "#43A047"), ("LV", "#1E88E5")], "la": [("LA", "#8E24AA")]}


def build(root: Path, order=("acdc", "mnms", "la")) -> Figure:
    metas = [json.loads((root / s / "META.json").read_text()) for s in order]
    labels = [m["label"] for m in metas[0]["models"]]
    for m in metas:
        if [x["label"] for x in m["models"]] != labels:
            raise ValueError("all datasets must show the same models in the same order")
    cols = ["Input", "Ground truth"] + labels
    gap, left, right = 8, 112, 24
    size = int((1008 - left - right - gap * (len(cols) - 1)) // len(cols))
    row_h = size + 30
    top = 40
    height = top + row_h * len(metas) + 54
    f = Figure("fig5_qualitative", height, "Qualitative comparison",
               "Input slice, ground truth and predictions of each model on ACDC, M&Ms and LA 2018.")
    x_of = lambda j: left + j * (size + gap)  # noqa: E731
    for j, c in enumerate(cols):
        f.text(x_of(j) + size / 2, 20, c, 14, weight="bold")
    for i, m in enumerate(metas):
        y = top + i * row_h
        f.lines(left / 2 + 4, y + size / 2, [m["name"].split(" (")[0], m.get("label_ratio", "10% labels")], 14, weight="bold")
        files = ["in.png", "gt.png"] + [f"pred_{k}.png" for k in range(len(labels))]
        for j, fn in enumerate(files):
            f.image(f"r{i}c{j}", x_of(j), y, size, root / [s for s in order][i] / fn)
        for k, mod in enumerate(m["models"]):
            f.text(x_of(2 + k) + size / 2, y + size + 14, f"Dice {100 * mod['patient_dice_3d']:.1f}", 14, color=MUTED)
    # legend (squares are plain rect nodes so the geometry check sees them)
    y = top + row_h * len(metas) + 8
    x = left
    seen = set()
    for m in metas:
        for name, color in LEGEND["la" if m["name"].startswith("LA") else "cardiac"]:
            if name in seen:
                continue
            seen.add(name)
            f.rect(f"lg_{name}", x, y, 16, 16, fill=color, stroke=color, radius=2, width=1)
            f.text(x + 24, y + 8, name, 14, anchor="start")
            x += 24 + 14 * (len(name) + 1) + 24
    return f


def main() -> int:
    import sys
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/paper-figures-20261124/qualitative")
    print(build(root).export(Path("reports/paper-figures-20261124/final")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
