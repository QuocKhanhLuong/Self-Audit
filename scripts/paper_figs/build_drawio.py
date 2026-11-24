#!/usr/bin/env python3
"""Generate the four SAS-UNet model figures as a clean, grid-aligned draw.io file and check them.

    python scripts/paper_figs/build_drawio.py --slices reports/paper-figures-20261124/slices \
        --out reports/paper-figures-20261124/sas_unet_figures.drawio --preview reports/paper-figures-20261124/preview

Pages: Fig1 Overview, Fig2 PhysZSSM, Fig3 SliceAlign, Fig4 ShiftRecovery.  When ``--slices`` has no PNG files the image
cells hold a neutral grey placeholder (never a fake slice); run ``export_fig_slices.py`` first for the real ones.
Exit status is non-zero when a layout rule fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drawio_kit import Page, check_page, preview, write_drawio  # noqa: E402

ENC, DEC = "#BBD7EE", "#8CCFC0"
BLUE_CUBE, ORANGE, AMBER = "#1F77B4", "#F28C28", "#FFC107"
UP, DOWN, CONV = "#5E9C2F", "#C62828", "#4A90C8"


def R(s, y):  # right port
    return (s.x + s.w, y)


def L(s, y):  # left port
    return (s.x, y)


def B(s, x):  # bottom port
    return (x, s.y + s.h)


def T(s, x):  # top port
    return (x, s.y)


def stack(p, prefix, x0, y0, pngs, size=60, step=10):
    """Offset pile of slices, k=0 (back, upper-left) .. last (front, lower-right)."""
    out = []
    for k, f in enumerate(pngs):
        out.append(p.image(f"{prefix}{k}", x0 + step * k, y0 + step * k, size, size, f, kind="stack"))
    return out


def lvl_cube(p, sid, x, port_y, f, back, front=None):
    """Two-tone in-plane block whose front face has side ``f``; ``port_y`` crosses the front face."""
    y = ((port_y - 10 - f // 2) // 10) * 10
    return p.cube(sid, x, y, f + 10, f + 10, BLUE_CUBE, front=back, depth=10)


# ----------------------------------------------------------------------------------------------------------------------
def fig1(sl: Path) -> Page:
    p = Page("Fig1 Overview", 1420, 880)
    ins = [sl / f"in_{k}.png" for k in range(5)]
    outs = [sl / f"out_{k}.png" for k in range(5)]
    ys = [140, 240, 340, 440, 540]  # port rows
    # stacks: input (right port of the front slice), output (left port of the back slice)
    sin = stack(p, "in", 10, 70, ins)
    sout = stack(p, "out", 1290, 110, outs)
    p.text("t_in", 10, 190, 110, 20, r"\(D\times H\times W\)", 13)
    p.text("t_out", 1290, 230, 130, 20, r"\(K=4\) classes", 13)
    # encoder / bottleneck
    e0 = lvl_cube(p, "e0", 160, ys[0], 60, ENC)
    e1 = lvl_cube(p, "e1", 220, ys[1], 50, ENC)
    e2 = lvl_cube(p, "e2", 270, ys[2], 40, ENC)
    e3 = lvl_cube(p, "e3", 340, ys[3], 30, ENC)
    sa = p.cube("sa", 420, 400, 90, 70, ORANGE, "Slice<br>Align")
    pz1 = p.cube("pz1", 560, 400, 90, 70, AMBER, "PhysZ-<br>SSM ×1", font="#000000")
    e4 = lvl_cube(p, "e4", 660, ys[4], 20, ENC)
    pz2 = p.cube("pz2", 740, 500, 130, 70, AMBER, "PhysZSSM ×2", font="#000000")
    # decoder
    d3 = lvl_cube(p, "d3", 910, ys[3], 30, DEC)
    d2 = lvl_cube(p, "d2", 970, ys[2], 40, DEC)
    d1 = lvl_cube(p, "d1", 1040, ys[1], 50, DEC)
    d0 = lvl_cube(p, "d0", 1120, ys[0], 60, DEC)
    # step icons sit on the corner of the L-shaped connector
    ic = {}
    for name, cx, cy, lab, col in (("i0", 190, ys[1], "↓", DOWN), ("i1", 240, ys[2], "↓", DOWN), ("i2", 290, ys[3], "↓", DOWN),
                                   ("i3", 600, ys[4], "↓", DOWN), ("i4", 920, ys[4], "↑", UP), ("i5", 990, ys[3], "↑", UP),
                                   ("i6", 1060, ys[2], "↑", UP), ("i7", 1150, ys[1], "↑", UP)):
        ic[name] = p.icon(name, cx - 10, cy - 10, lab, col)
    cc = p.icon("conv", 1230, 130, "C", CONV)
    # data stream
    p.edge("in4", "e0", [R(p.get("in4"), 140), L(e0, 140)])
    chain = [("e0", "i0", 190), ("e1", "i1", 240), ("e2", "i2", 290)]
    for a, i, x in chain:
        s = p.get(a)
        p.edge(a, i, [B(s, x), (x, p.get(i).y)])
    p.edge("i0", "e1", [R(ic["i0"], ys[1]), L(e1, ys[1])])
    p.edge("i1", "e2", [R(ic["i1"], ys[2]), L(e2, ys[2])])
    p.edge("i2", "e3", [R(ic["i2"], ys[3]), L(e3, ys[3])])
    p.edge("e3", "sa", [R(e3, ys[3]), L(sa, ys[3])])
    p.edge("sa", "pz1", [R(sa, ys[3]), L(pz1, ys[3])])
    p.edge("pz1", "i3", [B(pz1, 600), (600, ic["i3"].y)])
    p.edge("i3", "e4", [R(ic["i3"], ys[4]), L(e4, ys[4])])
    p.edge("e4", "pz2", [R(e4, ys[4]), L(pz2, ys[4])])
    p.edge("pz2", "i4", [R(pz2, ys[4]), L(ic["i4"], ys[4])])
    p.edge("i4", "d3", [T(ic["i4"], 920) if False else (920, ic["i4"].y), B(d3, 920)])
    p.edge("d3", "i5", [R(d3, ys[3]), L(ic["i5"], ys[3])])
    p.edge("i5", "d2", [(990, ic["i5"].y), B(d2, 990)])
    p.edge("d2", "i6", [R(d2, ys[2]), L(ic["i6"], ys[2])])
    p.edge("i6", "d1", [(1060, ic["i6"].y), B(d1, 1060)])
    p.edge("d1", "i7", [R(d1, ys[1]), L(ic["i7"], ys[1])])
    p.edge("i7", "d0", [(1150, ic["i7"].y), B(d0, 1150)])
    p.edge("d0", "conv", [R(d0, ys[0]), L(cc, ys[0])])
    p.edge("conv", "out0", [R(cc, ys[0]), L(p.get("out0"), ys[0])])
    # skips (dashed)
    p.edge("e0", "d0", [R(e0, ys[0]), L(d0, ys[0])], "skip")
    p.edge("e1", "d1", [R(e1, ys[1]), L(d1, ys[1])], "skip")
    p.edge("e2", "d2", [R(e2, ys[2]), L(d2, ys[2])], "skip")
    p.edge("pz1", "d3", [R(pz1, ys[3]), L(d3, ys[3])], "skip")
    # tensor sizes above the skip lines
    widths = {"s0": 170, "s1": 160, "s2": 170, "s3": 160, "s4": 170}
    for sid, x, y, lab in (("s0", 590, 100, r"\([24,\,D,\,H,\,W]\)"), ("s1", 580, 200, r"\([48,\,D,\,H/2,\,W/2]\)"),
                           ("s2", 560, 300, r"\([96,\,D,\,H/4,\,W/4]\)"), ("s3", 700, 400, r"\([160,\,D,\,H/8,\,W/8]\)"),
                           ("s4", 720, 460, r"\([192,\,D,\,H/16,\,W/16]\)")):
        p.text(sid, x, y, widths[sid], 20, lab, 12)
    # train-only heads
    a2 = p.box("aux2", 960, 270, 60, 20, "aux. head", "#F5F5F5", "#757575", size=12, dashed=True, bold=False)
    a1 = p.box("aux1", 1040, 160, 60, 20, "aux. head", "#F5F5F5", "#757575", size=12, dashed=True, bold=False)
    sdf = p.box("sdf", 1120, 60, 60, 20, "SDF head", "#F5F5F5", "#757575", size=12, dashed=True, bold=False)
    p.edge("d2", "aux2", [T(d2, 1000), (1000, a2.y + a2.h)], "train")
    p.edge("d1", "aux1", [T(d1, 1070), (1070, a1.y + a1.h)], "train")
    p.edge("d0", "sdf", [T(d0, 1150), (1150, sdf.y + sdf.h)], "train")
    # spacing / offset streams (red)
    sp = p.box("sp", 740, 600, 170, 40, r"slice spacing \(s_z\) (mm)", "#FFFFFF", "#D32F2F", "#D32F2F", 14)
    p.edge("sp", "pz2", [T(sp, 800), B(pz2, 800)], "spacing")
    p.edge("sp", "pz1", [L(sp, 620), (620, 620), (620, pz1.y + pz1.h)][0:1] + [(620, 620), (620, pz1.y + pz1.h)], "spacing")
    sr = p.box("sr", 440, 510, 110, 40, r"shift recovery<br>\(\mathcal{L}_{\mathrm{sr}}\)", "#F5F5F5", "#757575", size=14, dashed=True)
    p.edge("sa", "sr", [B(sa, 460), T(sr, 460)], "spacing")
    p.text("phi", 470, 480, 30, 20, r"\(\boldsymbol{\phi}\)", 13, "#D32F2F")
    # legend
    p.box("lg", 10, 580, 520, 280, "", "#FFFFFF", "#000000", kind="legend", stroke_w=1.2, bold=False)
    rows = (("up", UP, "↑", "Transposed conv, stride \\(1{\\times}2{\\times}2\\)"), ("dn", DOWN, "↓", "Strided conv, stride \\(1{\\times}2{\\times}2\\)"),
            ("cv", CONV, "C", "Conv \\(1\\times1\\times1\\)"))
    for k, (n, col, lab, txt) in enumerate(rows):
        y = 600 + 40 * k
        p.icon("lg_" + n, 30, y, lab, col)
        p.text("lgt_" + n, 60, y, 230, 20, txt, 12, align="left")
    p.cube("lg_enc", 30, 720, 30, 30, BLUE_CUBE, front=ENC, depth=10)
    p.text("lgt_enc", 70, 720, 220, 20, "Encoder block, \\(1{\\times}3{\\times}3\\) convs", 12, align="left")
    p.cube("lg_dec", 30, 770, 30, 30, BLUE_CUBE, front=DEC, depth=10)
    p.text("lgt_dec", 70, 770, 220, 20, "Decoder block, \\(1{\\times}3{\\times}3\\) convs", 12, align="left")
    lines = (("data", "Latent feature stream"), ("skip", "Skip connection"), ("spacing", "Spacing / offset stream"), ("train", "Training only"))
    for k, (kind, txt) in enumerate(lines):
        y = 610 + 40 * k
        p.edge(None, None, [(310, y), (360, y)], kind, eid=f"lgl_{kind}")
        p.text("lgtl_" + kind, 370, y - 10, 150, 20, txt, 12, align="left")
    p.cube("lg_m1", 310, 770, 30, 30, ORANGE, depth=10)
    p.cube("lg_m2", 360, 770, 30, 30, AMBER, depth=10)
    p.text("lgt_mod", 400, 770, 120, 20, "Slice-axis modules", 12, align="left")
    p.text("lgt_note", 20, 810, 500, 40, r"Stage-3 encoder block has one extra \(3{\times}1{\times}1\) conv. Face size = in-plane resolution; cube depth = \(D\) slices (constant: no down-sampling along \(z\))", 11, "#424242", align="left")
    return p


# ----------------------------------------------------------------------------------------------------------------------
def fig2() -> Page:
    p = Page("Fig2 PhysZSSM", 880, 520)
    # input tensor and tokens
    for k in range(5):
        p.shape(__import__("drawio_kit").Shape(f"ten{k}", "stack", 20 + 10 * k, 230 + 10 * k, 60, 60,
                                               "rounded=0;whiteSpace=wrap;html=1;strokeColor=#000000;strokeWidth=1;fillColor=#FFFFFF;"))
        p.shape(__import__("drawio_kit").Shape(f"tend{k}", "overlay", 40 + 10 * k, 250 + 10 * k, 10, 10,
                                               "rounded=0;html=1;strokeColor=none;fillColor=#D32F2F;"))
    p.text("t_ten", 10, 350, 120, 20, r"\([C,D,H,W]\)", 13)
    for k in range(5):
        p.shape(__import__("drawio_kit").Shape(f"tok{k}", "stack", 170, 230 + 20 * k, 20, 20,
                                               "rounded=0;whiteSpace=wrap;html=1;strokeColor=#000000;strokeWidth=1;fillColor=#FFFFFF;"))
    p.text("t_tok", 150, 200, 60, 20, r"\(D\) tokens", 12)
    p.text("t_x", 200, 260, 50, 30, r"\(\mathbf{x}_{h,w}\)", 13)
    p.text("t_per", 120, 270, 50, 20, r"per\((h,w)\)", 11)
    p.edge("ten4", "tok3", [(120, 300), (170, 300)], eid="e_per")
    # regions
    p.region("r_step", 320, 60, 400, 110, "#F9D6DE", "#F48FB1")
    p.text("t_step", 330, 70, 170, 20, "Spacing-scaled step", 13, bold=True, align="left")
    p.region("r_scan", 240, 190, 460, 250, "#BCD7EC", "#2196F3")
    p.text("t_scan", 250, 410, 170, 20, "Bidirectional selective scan", 13, bold=True, align="left")
    # main flow
    ln = p.box("ln", 260, 240, 40, 120, "LayerNorm", BLUE_CUBE, "#000000", "#FFFFFF", 14, bold=True, rounded=False, vertical=True)
    l1 = p.box("l1", 330, 240, 40, 120, "Linear", BLUE_CUBE, "#000000", "#FFFFFF", 14, bold=True, rounded=False, vertical=True)
    fw = p.box("fw", 420, 230, 80, 50, "Forward<br>scan", "#C62828", "#000000", "#FFFFFF", 14, bold=True, rounded=False)
    bw = p.box("bw", 420, 320, 80, 50, "Backward<br>scan", "#C62828", "#000000", "#FFFFFF", 14, bold=True, rounded=False)
    pl = p.box("pl", 530, 290, 20, 20, "+", "#FFFFFF", "#000000", size=14, rounded=True, bold=True)
    pl.style = pl.style.replace("rounded=1", "ellipse;rounded=0")
    gt = p.box("gt", 590, 290, 20, 20, "⊙", "#FFFFFF", "#000000", size=14, bold=True)
    gt.style = gt.style.replace("rounded=1", "ellipse;rounded=0")
    l2 = p.box("l2", 640, 240, 40, 120, "Linear", BLUE_CUBE, "#000000", "#FFFFFF", 14, bold=True, rounded=False, vertical=True)
    pf = p.box("pf", 720, 290, 20, 20, "+", "#FFFFFF", "#000000", size=14, bold=True)
    pf.style = pf.style.replace("rounded=1", "ellipse;rounded=0")
    p.text("t_out", 780, 280, 60, 40, r"\(\tilde{\mathbf{x}}_{h,w}\)", 13)
    p.edge("tok3", "ln", [(190, 300), (260, 300)], eid="e_x")
    p.edge("ln", "l1", [(300, 300), (330, 300)])
    p.edge("l1", "fw", [(370, 300), (390, 300), (390, 250), (420, 250)], eid="e_u1")
    p.edge("l1", "bw", [(370, 300), (390, 300), (390, 350), (420, 350)], eid="e_u2")
    p.text("t_u", 400, 290, 20, 20, r"\(\mathbf{u}\)", 13)
    p.edge("fw", "pl", [(500, 250), (540, 250), (540, 290)])
    p.edge("bw", "pl", [(500, 350), (540, 350), (540, 310)])
    p.edge("pl", "gt", [(550, 300), (590, 300)])
    p.edge("gt", "l2", [(610, 300), (640, 300)])
    p.edge("l2", "pf", [(680, 300), (720, 300)])
    p.edge("pf", "t_out", [(740, 300), (780, 300)])
    p.edge("l1", "gt", [(350, 360), (350, 390), (600, 390), (600, 310)], eid="e_gate")
    p.text("t_gate", 430, 400, 130, 20, r"gate \(\mathbf{g}\) (SiLU)", 12)
    p.edge("tok4", "pf", [(180, 330), (180, 470), (730, 470), (730, 310)], eid="e_res")
    p.text("t_res", 400, 480, 80, 20, "residual", 12)
    # spacing-scaled step
    sf = p.box("sf", 370, 110, 130, 40, r"\(\mathrm{softplus}(\mathrm{Linear}(\mathbf{u}))\)", "#F9A3BD", "#000000", size=12, bold=False)
    m1 = p.box("m1", 540, 120, 20, 20, "×", "#FFFFFF", "#000000", size=14, bold=True)
    m1.style = m1.style.replace("rounded=1", "ellipse;rounded=0")
    m2 = p.box("m2", 600, 120, 20, 20, "×", "#FFFFFF", "#000000", size=14, bold=True)
    m2.style = m2.style.replace("rounded=1", "ellipse;rounded=0")
    p.edge("l1", "sf", [(350, 240), (350, 130), (370, 130)], eid="e_dt")
    p.edge("sf", "m1", [(500, 130), (540, 130)])
    p.edge("m1", "m2", [(560, 130), (600, 130)], "spacing")
    p.text("t_sz", 500, 20, 100, 20, r"\(s_z/s_{\mathrm{ref}}\)", 13, "#D32F2F")
    p.text("t_mask", 610, 20, 110, 20, r"slice mask \(m_t\)", 13, "#D32F2F")
    p.edge("t_sz", "m1", [(550, 40), (550, 120)], "spacing", eid="e_sz")
    p.edge("t_mask", "m2", [(610, 40), (610, 120)], "spacing", eid="e_mask")
    p.edge("m2", "fw", [(610, 140), (610, 170), (460, 170), (460, 230)], "spacing", eid="e_d1", owner_of="t_dt")
    p.edge("m2", "bw", [(610, 140), (610, 170), (520, 170), (520, 300), (440, 300), (440, 320)], "spacing", eid="e_d2", owner_of="t_dt")
    p.text("t_dt", 630, 150, 40, 20, r"\(\Delta_t\)", 13, "#D32F2F")
    return p


# ----------------------------------------------------------------------------------------------------------------------
def fig3() -> Page:
    from drawio_kit import Shape
    p = Page("Fig3 SliceAlign", 1060, 500)
    tile = "rounded=0;whiteSpace=wrap;html=1;strokeColor=#000000;strokeWidth=1;fillColor=#FFFFFF;"
    ring = "ellipse;html=1;fillColor=none;strokeColor=#D32F2F;strokeWidth=2.5;"
    tiles = {}
    for sid, y, lab in (("fm", 150, r"\(\mathbf{f}_{i-1}\)"), ("fi", 250, r"\(\mathbf{f}_{i}\)"), ("fp", 350, r"\(\mathbf{f}_{i+1}\)")):
        t = p.shape(Shape(sid, "box", 110, y, 50, 50, tile))
        p.shape(Shape(sid + "_r", "overlay", 120, y + 10, 30, 20, ring))
        p.text(sid + "_t", 110, y + 30, 50, 20, lab, 12, kind="overlay")
        tiles[sid] = t
    # regions
    p.region("outer", 150, 130, 790, 300, "none", "#000000", dashed=False, width=2)
    p.region("r_off", 180, 140, 370, 280, "#F9D6DE", "#F48FB1")
    p.region("r_warp", 590, 140, 340, 280, "#BCD7EC", "#2196F3")
    p.text("t_off", 180, 10, 370, 20, "Offset estimation", 14, bold=True)
    p.text("t_warp", 590, 10, 340, 20, "Warping and fusion", 14, bold=True)
    p.edge(None, None, [(180, 40), (550, 40)], eid="a_off")
    p.edge(None, None, [(590, 40), (930, 40)], eid="a_warp")
    vb = dict(fill=BLUE_CUBE, stroke="#000000", font="#FFFFFF", size=14, bold=True, rounded=False, vertical=True)
    cv = p.box("cv", 200, 190, 50, 170, r"Conv \(1\times1\times1\) (shared)", **vb)
    cc = p.box("cc", 270, 190, 50, 170, r"Concat \((i,\,i{+}1)\)", "#7CB342", "#000000", "#FFFFFF", 14, bold=True, rounded=False, vertical=True)
    cb = p.box("cb", 340, 190, 50, 170, r"ConvBlock \(1\times3\times3\)", **vb)
    cz = p.box("cz", 410, 190, 50, 170, r"Conv \(1\times3\times3\) (zero-init)", **vb)
    th = p.box("th", 480, 190, 50, 170, r"\(4\tanh(\cdot)\)", "#C62828", "#000000", "#FFFFFF", 14, bold=True, rounded=False, vertical=True)
    w1 = p.box("w1", 610, 190, 80, 50, r"Warp<br>\(\mathbf{f}_{i-1}\)", "#C62828", "#000000", "#FFFFFF", 14, bold=True, rounded=False)
    w2 = p.box("w2", 610, 310, 80, 50, r"Warp<br>\(\mathbf{f}_{i+1}\)", "#C62828", "#000000", "#FFFFFF", 14, bold=True, rounded=False)
    cat = p.box("cat", 730, 190, 50, 170, "Concat", "#7CB342", "#000000", "#FFFFFF", 14, bold=True, rounded=False, vertical=True)
    cz2 = p.box("cz2", 810, 190, 50, 170, r"Conv \(1\times1\times1\) (zero-init)", **vb)
    pl = p.box("pl", 890, 260, 20, 20, "+", "#FFFFFF", "#000000", size=14, bold=True)
    pl.style = pl.style.replace("rounded=1", "ellipse;rounded=0")
    out = p.shape(Shape("fo", "box", 960, 250, 50, 50, tile))
    p.shape(Shape("fo_r", "overlay", 970, 260, 30, 20, ring))
    p.text("fo_t", 960, 280, 50, 20, r"\(\tilde{\mathbf{f}}_{i}\)", 12, kind="overlay")
    # offset path
    p.edge("fi", "cv", [(160, 270), (200, 270)])
    p.edge("fp", "cv", [(160, 370), (170, 370), (170, 330), (200, 330)])
    for a, b in (("cv", "cc"), ("cc", "cb"), ("cb", "cz"), ("cz", "th")):
        p.edge(a, b, [(p.get(a).x + 50, 270), (p.get(b).x, 270)])
    p.edge("th", "w1", [(530, 270), (570, 270), (570, 210), (610, 210)], "spacing", eid="e_p1", owner_of="t_p1")
    p.edge("th", "w2", [(530, 270), (570, 270), (570, 330), (610, 330)], "spacing", eid="e_p2", owner_of="t_p2")
    p.text("t_p1", 540, 170, 60, 20, r"\(-\boldsymbol{\phi}_{i-1}\)", 12, "#D32F2F")
    p.text("t_p2", 540, 350, 60, 20, r"\(\boldsymbol{\phi}_{i}\)", 12, "#D32F2F")
    # feature routes
    p.edge("fm", "w1", [(140, 150), (140, 100), (650, 100), (650, 190)], eid="e_fm")
    p.edge("fp", "w2", [(140, 400), (140, 460), (650, 460), (650, 360)], eid="e_fp")
    p.edge("w1", "cat", [(690, 210), (730, 210)])
    p.edge("w2", "cat", [(690, 330), (730, 330)])
    p.edge("cat", "cz2", [(780, 270), (810, 270)])
    p.edge("cz2", "pl", [(860, 270), (890, 270)])
    p.edge("pl", "fo", [(910, 270), (960, 270)])
    p.edge("fi", "pl", [(110, 270), (20, 270), (20, 80), (900, 80), (900, 260)], eid="e_res")
    p.text("t_res", 400, 50, 80, 20, "residual", 12)
    return p


# ----------------------------------------------------------------------------------------------------------------------
def fig4(sl: Path) -> Page:
    p = Page("Fig4 ShiftRecovery", 960, 480)
    cl = [sl / f"in_{k}.png" for k in range(5)]
    sh = [sl / f"shift_{k}.png" for k in range(5)]
    stack(p, "cl", 20, 60, cl)
    stack(p, "sh", 60, 270, sh)
    p.text("t_cl", 10, 30, 180, 20, r"unlabeled stack \(\mathbf{X}\)", 13)
    p.text("t_sh", 10, 430, 330, 20, r"\(\mathcal{T}_{\mathbf{s}}(\mathbf{X})\), \(s_i\sim\mathcal{U}\{-8,\dots,8\}\) px", 13)
    p.text("t_ps", 100, 200, 100, 30, "per-slice<br>shift", 12, "#D32F2F")
    p.edge("cl4", "sh0", [(90, 160), (90, 270)], "spacing", eid="e_shift", owner_of="t_ps")
    e1 = p.cube("e1", 260, 100, 140, 70, ORANGE, "Encoder E0–E3<br>+ offset head")
    e2 = p.cube("e2", 260, 310, 140, 70, ORANGE, "Encoder E0–E3<br>+ offset head")
    p.edge("cl4", "e1", [(120, 130), (260, 130)])
    p.edge("sh4", "e2", [(160, 340), (260, 340)])
    p.edge("e1", "e2", [(330, 170), (330, 310)], "train", eid="e_shared", both=True)
    p.text("t_shared", 350, 230, 110, 20, "shared weights", 12)
    b1 = p.box("b1", 460, 110, 80, 40, r"\(\mathrm{sg}(\boldsymbol{\phi}_i)\)", "#FFFFFF", "#000000", size=14, bold=False)
    b2 = p.box("b2", 460, 320, 80, 40, r"\(\tilde{\boldsymbol{\phi}}_i\)", "#FFFFFF", "#000000", size=14, bold=False)
    p.text("t_sg", 450, 70, 100, 20, "stop-gradient", 12)
    p.edge("e1", "b1", [(400, 130), (460, 130)], "spacing")
    p.edge("e2", "b2", [(400, 340), (460, 340)], "spacing")
    loss = p.box("loss", 620, 210, 300, 80, r"\(\mathcal{L}_{\mathrm{sr}}=\mathrm{Smooth}\text{-}\ell_1\big(\tilde{\boldsymbol{\phi}}_i-\mathrm{sg}(\boldsymbol{\phi}_i),\)<br>\((s_{i+1}-s_i)/8\,\big)\)",
                 "#F9D6DE", "#F48FB1", size=14)
    p.edge("b1", "loss", [(540, 130), (680, 130), (680, 210)], "spacing")
    p.edge("b2", "loss", [(540, 340), (680, 340), (680, 290)], "spacing")
    p.edge("sh4", "loss", [(130, 370), (130, 410), (860, 410), (860, 290)], "spacing", eid="e_known", owner_of="t_known")
    p.text("t_known", 480, 420, 100, 20, r"known \(s_i\)", 13, "#D32F2F")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slices", type=Path, default=Path("reports/paper-figures-20261124/slices"))
    ap.add_argument("--out", type=Path, default=Path("reports/paper-figures-20261124/sas_unet_figures.drawio"))
    ap.add_argument("--preview", type=Path, default=None, help="directory for matplotlib layout previews (not the draw.io render)")
    args = ap.parse_args()
    from drawio_kit import write_drawio
    pages = [fig1(args.slices), fig2(), fig3(), fig4(args.slices)]
    ok = True
    for pg in pages:
        bad = check_page(pg)
        print(f"{pg.name}: {len(pg.shapes)} shapes, {len(pg.edges)} edges, {getattr(pg, 'crossings', 0)} crossings, {len(bad)} violations")
        for b in bad:
            print("   ", b)
        ok &= not bad
        if args.preview:
            args.preview.mkdir(parents=True, exist_ok=True)
            preview(pg, args.preview / (pg.name.split()[0].lower() + ".png"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_drawio(pages, args.out)
    print("wrote", args.out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
