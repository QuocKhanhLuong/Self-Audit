"""Training-only per-slice shift recovery, using existing patient011 PNGs."""
from pathlib import Path

from svg_kit import (Figure, BLUE, ENC, ORANGE, ALIGN, RED, PINK, GREY,
                     MUTED, LIGHT, WHITE, RULE)


def build(slices: Path) -> Figure:
    f = Figure("fig4_shiftrecovery", 776, "Slice-shift recovery auxiliary",
               "Shared stage-0-to-3 encoder and offset head estimate clean and shifted slice flows; a detached clean flow and known relative shifts define a Smooth-L1 training target.")
    f.heading("4  Shift recovery from known perturbations", "Training only  ·  Relative-shift supervision without segmentation labels")

    # Five real slices in two aligned filmstrips; no image generation or inference.
    f.text(24, 94, "Clean view X", 18, weight="bold", anchor="start")
    f.text(24, 350, r"Shifted view $\mathcal{T}_{s}(X)$", 18, weight="bold", anchor="start")
    for prefix, y, kind in [("clean", 120, "in"), ("shifted", 376, "shift")]:
        f.rect(prefix, 24, y-8, 328, 80, fill=WHITE, stroke=RULE)
        for k in range(5):
            f.image(f"{prefix}_{k}", 32 + 64*k, y, 56, slices / f"{kind}_{k}.png")
    f.text(188, 210, "ACDC patient011 · base → apex", 14, color=MUTED)
    f.text(188, 466, "Same slices; independent (dy, dx) shifts", 14, color=MUTED)

    ec = f.box("clean_encoder", 400, 112, 204, 80, ["Encoder E0–E3", "+ offset head"], fill=ALIGN, stroke=ORANGE, size=18, weight="bold")
    es = f.box("shift_encoder", 400, 368, 204, 80, ["Encoder E0–E3", "+ offset head"], fill=ALIGN, stroke=ORANGE, size=18, weight="bold")
    f.edge([(352, 152), ec.left], source="clean", target=ec.id)
    f.edge([(352, 408), es.left], source="shifted", target=es.id)

    clean = f.box("detach", 652, 112, 132, 80, [r"$\mathrm{sg}(\phi_i)$", "Detached"], fill=LIGHT, stroke=GREY, size=18)
    moved = f.box("moved_flow", 652, 368, 132, 80, [r"$\tilde{\phi}_i$", "Gradient"], fill=WHITE, stroke=ORANGE, size=18)
    f.edge([ec.right, clean.left], color=GREY, source=ec.id, target=clean.id)
    f.edge([es.right, moved.left], color=ORANGE, source=es.id, target=moved.id)
    f.text(502, 472, "Gradient to encoder + offset head", 14, color=ORANGE)

    # Shared parameters are a relation, never an extra network module.
    f.edge([(502, 192), (502, 368)], color=GREY, dashed=True, arrow=False)
    f.rect("shared_label", 424, 260, 156, 36, fill=WHITE, stroke="none", layer="text")
    f.text(502, 278, "Shared weights", 16, color=GREY)

    inject = f.box("sample_shifts", 48, 248, 280, 56,
                   ["Per-slice integer shifts", r"$s_i\sim\mathcal{U}\{-8,\ldots,8\}^{2}$ px"],
                   fill=PINK, stroke=RED, size=16)
    f.edge([(332, 192), (332, 228), (280, 228), (280, 248)], color=RED, source="clean", target=inject.id)
    f.edge([inject.bottom, (188, 320), (336, 320), (336, 368)], color=RED, source=inject.id, target="shifted")

    # Combine flows beside their rows; the residual is a difference, not a target flow.
    delta = f.box("difference", 836, 244, 148, 64,
                  ["Flow change", r"$\tilde{\phi}_i-\mathrm{sg}(\phi_i)$"], fill=LIGHT, stroke=BLUE, size=17)
    f.edge([clean.right, (910, 152), delta.top], color=GREY, source=clean.id, target=delta.id)
    f.edge([moved.right, (910, 408), delta.bottom], color=ORANGE, source=moved.id, target=delta.id)

    # Explicit known-shift target. Keep the long route outside the filmstrip/encoder.
    target = f.box("target", 48, 524, 280, 96,
                   ["Known relative-shift target", r"$t_i=(s_{i+1}-s_i)/8$"], fill=PINK, stroke=RED, size=18)
    f.edge([inject.left, (12, 276), (12, 572), target.left], color=RED, source=inject.id, target=target.id)
    f.text(188, 642, "Pixels ÷ stage-3 stride → feature cells", 14, color=MUTED)

    loss = f.box("loss", 400, 524, 584, 96,
                 [r"$\mathcal{L}_{\mathrm{sr}}=\mathrm{SmoothL1}(\tilde{\phi}_i-\mathrm{sg}(\phi_i),\ t_i)$",
                  r"Total objective adds $0.1\,\mathcal{L}_{\mathrm{sr}}$"],
                 fill=PINK, stroke=RED, size=22)
    f.edge([target.right, loss.left], color=RED, source=target.id, target=loss.id)
    f.edge([delta.right, (996, 276), (996, 496), (916, 496), (916, 524)], source=delta.id, target=loss.id)

    f.rule(660)
    f.text(24, 688, "Offsets are two-component (dy, dx) fields on the H/8 × W/8 grid; targets broadcast spatially.", 15, anchor="start")
    f.text(24, 718, "Smooth-L1 is averaged over flow elements (valid adjacent pairs when a slice mask is supplied).", 15, anchor="start")
    f.text(24, 748, "This auxiliary uses the encoder and offset predictor only; it is absent at inference.", 15, anchor="start")
    return f
