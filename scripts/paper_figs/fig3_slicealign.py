#!/usr/bin/env python3
"""Build the publication-clean SliceAlign schematic (Fig. 3).

The implementation is intentionally declarative: the pairwise offset head and
the slice-local warp/fuse path are laid out independently, so no route has to
cross a tensor block or borrow a port from another slice.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from svg_kit import (  # noqa: E402
    ALIGN,
    BLUE,
    ENC,
    Figure,
    GREY,
    INK,
    LIGHT,
    MUTED,
    ORANGE,
    PINK,
    RED,
    RULE,
    TEAL,
    WHITE,
)


def _panel(fig: Figure, sid: str, x: int, y: int, w: int, h: int, title: str, subtitle: str) -> None:
    """A paper panel drawn below edges and nodes, with an unambiguous header."""
    fig.rect(sid, x, y, w, h, fill=WHITE, stroke=RULE, radius=10, layer="bg", width=1.2)
    fig.add(f'<path d="M{x + 20} {y + 48} H{x + w - 20}" stroke="{RULE}" stroke-width="1"/>', "bg")
    fig.text(x + 24, y + 22, title, 18, weight="bold", anchor="start")
    fig.text(x + 24, y + 40, subtitle, 14, color=MUTED, anchor="start")


def _cube_port(cube, side: str):
    """Front-face ports for the oblique cube (the wire meets a shaded face)."""
    cy = cube.y + cube.h / 2
    if side == "left":
        return cube.left
    if side == "right":
        # Use the outer edge of the shaded right face so the wire never starts
        # underneath the cube's isometric side.
        return cube.right
    if side == "top":
        return cube.x + cube.w / 2, cube.y
    if side == "bottom":
        return cube.x + cube.w / 2, cube.y + cube.h
    raise ValueError(side)


def build() -> Figure:
    fig = Figure(
        "fig3_slicealign",
        820,
        "3  SliceAlign: align neighbours before fusion",
        "Stage-3 adjacent-slice offset estimation, bidirectional warping, and zero-initialized residual fusion.",
    )
    fig.heading(
        "3  SliceAlign: align neighbours before fusion",
        "Stage-3 adjacent-slice offset estimation and residual fusion",
    )
    fig.rule(78)

    # ──────────────────────────────────────────────────────────────────────
    # A. Offset head: two adjacent slices share one reduction and offset head.
    # ──────────────────────────────────────────────────────────────────────
    _panel(
        fig,
        "panel_a",
        24,
        98,
        960,
        245,
        "A  Shared adjacent-slice offset head",
        "Offsets are predicted once for each pair (i, i+1) in stage-3 cells.",
    )

    fi = fig.cube("a_fi", 52, 195, 62, 62, fill=ENC, label="$f_i$", size=18)
    fip = fig.cube("a_fip", 52, 276, 62, 62, fill=ENC, label="$f_{i+1}$", size=18)
    fig.text(52, 166, "adjacent features", 14, color=MUTED, anchor="start")

    reduce = fig.box(
        "a_reduce",
        176,
        206,
        112,
        108,
        ["1×1×1", "reduce", "C → 32"],
        fill=LIGHT,
        stroke=BLUE,
        size=16,
    )
    # The two wires fan into distinct points on the left face.
    fig.edge(
        [_cube_port(fi, "right"), (150, _cube_port(fi, "right")[1]), (150, 234), (176, 234)],
        color=BLUE,
        source="a_fi",
        target="a_reduce",
    )
    fig.edge(
        [_cube_port(fip, "right"), (150, _cube_port(fip, "right")[1]), (150, 286), (176, 286)],
        color=BLUE,
        source="a_fip",
        target="a_reduce",
    )

    concat = fig.box(
        "a_concat",
        326,
        226,
        112,
        68,
        ["concat", "pair (i, i+1)"],
        fill="#E8F1F7",
        stroke=BLUE,
        size=16,
    )
    conv = fig.box(
        "a_conv",
        476,
        216,
        112,
        88,
        ["1×3×3", "conv", "+ GELU"],
        fill="#E8F1F7",
        stroke=BLUE,
        size=16,
    )
    conv0 = fig.box(
        "a_conv0",
        628,
        216,
        112,
        88,
        ["1×3×3", "conv", "zero-init"],
        fill="#F5E7DE",
        stroke=ORANGE,
        size=16,
    )
    tanh = fig.box(
        "a_tanh",
        784,
        232,
        82,
        56,
        ["4 tanh"],
        fill=ALIGN,
        stroke=ORANGE,
        size=16,
        weight="bold",
    )
    phi = fig.cube("a_phi", 906, 228, 58, 64, fill=ALIGN, label="$\\phi_i$", size=16)

    # Straight, face-to-face hand-offs: every endpoint is on a visible block face.
    fig.connect(_cube_port(reduce, "right"), concat.left, color=BLUE, source="a_reduce", target="a_concat")
    fig.connect(concat.right, conv.left, color=BLUE, source="a_concat", target="a_conv")
    fig.connect(conv.right, conv0.left, color=BLUE, source="a_conv", target="a_conv0")
    fig.connect(conv0.right, tanh.left, color=ORANGE, source="a_conv0", target="a_tanh")
    fig.connect(tanh.right, _cube_port(phi, "left"), color=ORANGE, source="a_tanh", target="a_phi")
    fig.text(918, 309, "(dy, dx)", 14, color=RED, anchor="start")

    # A compact mathematical contract sits beneath the chain rather than on a wire.
    fig.text(
        504,
        326,
        "pair 64 ch → hidden 32 ch → offsets 2 ch (dy, dx)   |   final conv is zero-initialized",
        14,
        color=MUTED,
    )

    # ──────────────────────────────────────────────────────────────────────
    # B. Slice-local alignment: both neighbours are warped onto slice i.
    # ──────────────────────────────────────────────────────────────────────
    _panel(
        fig,
        "panel_b",
        24,
        366,
        960,
        430,
        "B  Bidirectional warp and residual fusion at slice i",
        "The same pairwise flow is used in opposite directions; missing neighbours are zero-filled.",
    )

    # Feature inputs are arranged in code order: NEXT, target, PREV.
    next_f = fig.cube("b_next", 48, 466, 64, 64, fill=ENC, label="$f_{i+1}$", size=17)
    target_f = fig.cube("b_target", 48, 566, 64, 64, fill=ENC, label="$f_i$", size=18)
    prev_f = fig.cube("b_prev", 48, 666, 64, 64, fill=ENC, label="$f_{i-1}$", size=17)

    warp_next = fig.box(
        "b_warp_next",
        206,
        467,
        136,
        62,
        ["warp $f_{i+1}$", "with $+\\phi_i$"],
        fill="#F5E7DE",
        stroke=ORANGE,
        size=16,
    )
    warp_prev = fig.box(
        "b_warp_prev",
        206,
        667,
        136,
        62,
        ["warp $f_{i-1}$", "with $-\\phi_{i-1}$"],
        fill="#F5E7DE",
        stroke=ORANGE,
        size=16,
    )

    # Face-to-face inputs to the two warp operators.
    fig.connect(_cube_port(next_f, "right"), warp_next.left, color=ORANGE, source="b_next", target="b_warp_next")
    fig.connect(_cube_port(prev_f, "right"), warp_prev.left, color=ORANGE, source="b_prev", target="b_warp_prev")

    concat_b = fig.box(
        "b_concat",
        430,
        538,
        136,
        118,
        ["concat", "[NEXT, PREV]"],
        fill="#E8F1F7",
        stroke=BLUE,
        size=16,
    )
    fuse = fig.box(
        "b_fuse",
        636,
        557,
        126,
        80,
        ["1×1×1", "fuse", "zero-init"],
        fill=ALIGN,
        stroke=ORANGE,
        size=16,
    )
    plus = fig.circle("b_plus", 810, 597, r=17, label="+", fill=WHITE, stroke=TEAL, size=22)
    out = fig.cube("b_out", 858, 565, 72, 64, fill="#D4E9E4", label="$\\tilde f_i$", size=16)

    # Pairwise outputs enter separate points on concat; the route bends in the
    # open gutter, never across a node.
    fig.edge(
        [warp_next.right, (388, warp_next.right[1]), (388, 568), (430, 568)],
        color=ORANGE,
        source="b_warp_next",
        target="b_concat",
    )
    fig.edge(
        [warp_prev.right, (388, warp_prev.right[1]), (388, 626), (430, 626)],
        color=ORANGE,
        source="b_warp_prev",
        target="b_concat",
    )
    fig.connect(concat_b.right, fuse.left, color=BLUE, source="b_concat", target="b_fuse")
    fig.connect(fuse.right, plus.left, color=TEAL, source="b_fuse", target="b_plus")
    fig.connect(plus.right, _cube_port(out, "left"), color=TEAL, source="b_plus", target="b_out")

    # Residual f_i route: source on the top face, along a dedicated corridor,
    # then into the top of the add node. It does not cross either warp path.
    target_left = _cube_port(target_f, "left")
    fig.edge(
        [target_left, (32, target_left[1]), (32, 438), (810, 438), (810, 580)],
        color=TEAL,
        dashed=True,
        source="b_target",
        target="b_plus",
    )

    # Notes are outside the active routes and remain legible at column width.
    fig.add(f'<path d="M48 746 H960" stroke="{RULE}" stroke-width="1"/>', "bg")
    fig.text(
        504,
        758,
        "warp convention: sample p + flow   •   bilinear sampler uses zero outside the image",
        14,
        color=MUTED,
    )
    fig.text(
        504,
        778,
        "first/last slice: missing neighbour = 0   •   padded pair validity mask gates both warped features",
        14,
        color=MUTED,
    )
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/paper-figures-20261124/final"))
    args = parser.parse_args()
    result = build().export(args.out)
    print(result)
