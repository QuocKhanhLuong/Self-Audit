#!/usr/bin/env python3
"""Build the PhysZSSM detail figure as a publication-sized SVG.

The layout follows the implementation in ``candidate_pg/slicessm_student.py``:
tokens are scanned independently at every in-plane cell, with physical spacing
and the padded-slice mask applied to both scan directions.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from svg_kit import (  # noqa: E402
    Figure, ENC, GOLD, INK, LIGHT, MUTED, PINK, RED, RULE, SSM, TEAL, WHITE,
    BLUE,
)


def build() -> Figure:
    f = Figure(
        "fig2_physzssm",
        800,
        "2  PhysZSSM: physically stepped slice context",
        "A bidirectional, spacing-aware selective state-space scan over z tokens at each in-plane location.",
    )
    f.heading(
        "2  PhysZSSM: physically stepped slice context",
        "Bidirectional selective scan over z tokens; no down-sampling along the slice axis",
    )
    f.rule(78)

    # Row 1: tokenisation at one in-plane location.
    f.rect("token_panel", 24, 102, 960, 164, fill="#F8FBFD", stroke="#C8DCE9", radius=8, width=1.4, layer="bg")
    f.text(48, 122, "Tokenise each (h, w) column", 18, anchor="start", weight="bold")
    inp = f.cube("input", 50, 163, 84, 74, fill=ENC)
    f.text(92, 138, "$[C,D,H,W]$", 15, weight="bold")
    tokens = f.rect("tokens", 190, 163, 112, 74, fill=LIGHT, stroke=BLUE, radius=4)
    # Four aligned bars are a compact visual cue for the D-token column.
    f.add('<g fill="#B9D7EA" stroke="#3678A3" stroke-width="1.1">'
          '<rect x="202" y="174" width="16" height="10" rx="2"/>'
          '<rect x="202" y="188" width="16" height="10" rx="2"/>'
          '<rect x="202" y="202" width="16" height="10" rx="2"/>'
          '<rect x="202" y="216" width="16" height="10" rx="2"/></g>')
    f.lines(259, 191, ["reshape", "$D$ tokens", "per $(h,w)$"], size=14, weight="bold")
    norm = f.box("norm", 340, 163, 104, 74, ["LayerNorm"], fill=LIGHT, stroke=BLUE, size=14)
    proj = f.box("proj", 480, 163, 122, 74, ["Linear", "$u_t,\\,g_t$"], fill=LIGHT, stroke=BLUE, size=15)
    f.connect(inp.right, tokens.left, color=BLUE, source="input", target="tokens")
    f.connect(tokens.right, norm.left, color=BLUE, source="tokens", target="norm")
    f.connect(norm.right, proj.left, color=BLUE, source="norm", target="proj")

    # Row 2: context block. u explicitly passes SiLU before both scans; g is
    # transformed separately and gates their summed output.
    f.rect("context_panel", 24, 294, 960, 278, fill="#FDF9F8", stroke="#E4C7C8", radius=8, width=1.4, layer="bg")
    f.text(48, 314, "Bidirectional z context", 18, anchor="start", weight="bold")
    f.text(48, 338, "Each (h,w) cell is scanned independently; forward and backward parameters are not shared.", 15, anchor="start", color=MUTED)

    ubox = f.box("u_silu", 145, 400, 125, 64, ["SiLU($u_t$)", "$\\times m_t$"], fill=SSM, stroke=GOLD, size=14, weight="bold")
    fw = f.box("forward", 340, 355, 150, 64, ["forward scan", "$A_{\\to},B_{\\to},C_{\\to},\\Delta_{\\to}$"], fill=PINK, stroke=RED, size=14, weight="bold")
    bw = f.box("backward", 340, 455, 150, 64, ["backward scan", "$A_{\\leftarrow},B_{\\leftarrow},C_{\\leftarrow},\\Delta_{\\leftarrow}$"], fill=PINK, stroke=RED, size=14, weight="bold")
    f.edge([(490, 237), (490, 280), (36, 280), (36, 380), (145, 380), (145, 432)], color=BLUE, source="proj", target="u_silu")
    f.edge([(555, 237), (555, 280), (760, 280), (760, 350), (657.5, 350), (657.5, 360)], color=GOLD, source="proj", target="gate")
    f.edge([(270, 418), (300, 418), (300, 387), (340, 387)], color=BLUE, source="u_silu", target="forward")
    f.edge([(270, 446), (300, 446), (300, 487), (340, 487)], color=BLUE, source="u_silu", target="backward")

    gate = f.box("gate", 590, 360, 135, 46, ["SiLU($g_t$)"], fill=SSM, stroke=GOLD, size=15, weight="bold")
    add = f.circle("add", 560, 432, r=18, label="+", fill=WHITE, stroke=RED, size=20)
    f.edge([(490, 387), (520, 387), (560, 387), (560, 414)], color=BLUE, source="forward", target="add")
    f.edge([(490, 487), (520, 487), (560, 487), (560, 450)], color=BLUE, source="backward", target="add")
    mult = f.circle("mult", 660, 432, r=18, label="×", fill=WHITE, stroke=GOLD, size=20)
    f.connect(add.right, mult.left, color=BLUE, source="add", target="mult")
    f.edge([(657.5, 406), (657.5, 410), (660, 410), (660, 414)], color=GOLD, source="gate", target="mult")
    outproj = f.box("outproj", 740, 393, 92, 78, ["out Linear", "$\\times m_t$"], fill=LIGHT, stroke=BLUE, size=14)
    f.connect(mult.right, outproj.left, color=BLUE, source="mult", target="outproj")
    plus = f.circle("residual_add", 876, 432, r=18, label="+", fill=WHITE, stroke=TEAL, size=20)
    f.connect(outproj.right, plus.left, color=BLUE, source="outproj", target="residual_add")
    output = f.cube("output", 900, 372, 54, 120, fill=ENC)
    f.text(927, 308, "output", 15, weight="bold")
    f.text(927, 326, "$[C,D,H,W]$", 14, weight="bold")
    f.connect(plus.right, output.left, color=TEAL, source="residual_add", target="output")

    # Residual x is carried around the block and added after out projection.
    f.edge([(92, 237), (30, 237), (30, 540), (876, 540), (876, 450)], color=TEAL, source="input", target="residual_add")
    f.text(540, 558, "residual $x$", 15, color=TEAL, weight="bold")

    # Formula panel is a background layer so arrows and labels remain visible.
    f.rect("formula_panel", 24, 600, 960, 176, fill="#FFF9F1", stroke="#E4CFA9", radius=8, width=1.4, layer="bg")
    f.text(48, 623, "Physically stepped, masked recurrence", 18, anchor="start", weight="bold", color=INK)
    f.text(48, 647, "$u_t \\leftarrow \\operatorname{SiLU}(u_t) m_t$; $\\Delta_t$ carries the same mask.", 15, anchor="start", color=MUTED)
    step1 = f.box("step_softplus", 55, 668, 270, 44, ["$\\operatorname{softplus}(W_{dt}\\,\\delta(u_t)+b)$"], fill=SSM, stroke=GOLD, size=14, weight="bold")
    step2 = f.box("step_spacing", 365, 668, 220, 44, ["$\\times (s_z / 10\\,\\mathrm{mm})$"], fill=SSM, stroke=GOLD, size=14, weight="bold")
    step3 = f.box("step_mask", 625, 668, 120, 44, ["$\\times m_t$"], fill=SSM, stroke=GOLD, size=14, weight="bold")
    step4 = f.box("step_delta", 785, 668, 135, 44, ["$\\Delta_t$"], fill="#F9E8C8", stroke=GOLD, size=16, weight="bold")
    f.connect(step1.right, step2.left, color=GOLD, source="step_softplus", target="step_spacing")
    f.connect(step2.right, step3.left, color=GOLD, source="step_spacing", target="step_mask")
    f.connect(step3.right, step4.left, color=GOLD, source="step_mask", target="step_delta")
    f.text(55, 741, "$h_t=\\exp(\\Delta_t A)h_{t-1}+\\Delta_t u_t B_t$", 16, anchor="start", weight="bold", color=INK)
    f.text(500, 741, "$y_t=\\langle h_t,C_t\\rangle+D u_t$", 16, anchor="start", weight="bold", color=INK)
    f.text(55, 764, "Padded slices keep the residual $x$; only the scan update and $\\Delta_t$ are masked.", 15, anchor="start", color=MUTED)
    return f


def main() -> int:
    out = Path("reports/paper-figures-20261124/final")
    print(build().export(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
