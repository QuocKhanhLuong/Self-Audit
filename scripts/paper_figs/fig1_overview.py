"""SASNet overview: the canonical SliceSSMStudent architecture."""
from pathlib import Path

from svg_kit import (Figure, BLUE, ENC, DEC, TEAL, ORANGE, ALIGN, GOLD,
                     SSM, GREY, MUTED, LIGHT, WHITE, RULE, RED)


def build(slices: Path) -> Figure:
    f = Figure("fig1_overview", 992, "SASNet architecture",
               "Five native-depth encoder stages, SliceAlign, physically stepped z scans, skip-connected decoder and training-only auxiliary heads.")
    f.heading("1  SASNet: slice-aware segmentation", "SliceSSMStudent  ·  In-plane hierarchy with constant slice depth D")

    f.text(204, 84, "Input cine slices · patient011", 16, weight="bold")
    f.text(804, 84, "SASNet predictions · same slices", 16, weight="bold")
    f.rect("input", 24, 92, 360, 80, fill=WHITE, stroke=RULE)
    f.rect("output", 624, 92, 360, 80, fill=WHITE, stroke=RULE)
    for k in range(5):
        f.image(f"in{k}", 32 + 70*k, 100, 64, slices / f"in_{k}.png")
        f.image(f"out{k}", 632 + 70*k, 100, 64, slices / f"out_{k}.png")
    f.text(204, 190, r"$X:\ 1\times D\times H\times W$", 16, color=MUTED)
    f.text(632, 190, "BG / RV / MYO / LV", 14, color=MUTED, anchor="start")

    ey = [264, 384, 504, 624]
    ex = [120, 152, 184, 216]
    dx = [772, 744, 716, 688]
    widths, heights, channels = [88, 80, 72, 64], [64, 56, 48, 40], [24, 48, 96, 160]
    enc, dec = [], []
    for i, (x, y, w, h, c) in enumerate(zip(ex, ey, widths, heights, channels)):
        enc.append(f.cube(f"enc{i}", x, y, w, h, fill=ENC, label=str(c)))
        dec.append(f.cube(f"dec{i}", dx[i], y, w, h, fill=DEC, label=str(c)))
        f.text(x - 28, y + h/2, f"E{i}", 16, color=BLUE, weight="bold")
    f.edge([(100, 172), (100, 228), (180, 228), enc[0].top], source="input", target="enc0")

    for i in range(3):
        a, b = enc[i], enc[i+1]
        mid = (a.bottom[1] + b.top[1])/2
        f.edge([a.bottom, (a.bottom[0], mid), (b.top[0], mid), b.top], source=a.id, target=b.id)
        a, b = dec[i+1], dec[i]
        mid = (a.top[1] + b.bottom[1])/2
        f.edge([a.top, (a.top[0], mid), (b.bottom[0], mid), b.bottom], color=TEAL, source=a.id, target=b.id)
    for i in range(3):
        f.edge([enc[i].right, dec[i].left], dashed=True, source=enc[i].id, target=dec[i].id)
        scale = "H,W" if i == 0 else f"H/{2**i},W/{2**i}"
        f.text(480, enc[i].right[1]-22, f"$[{channels[i]},D,{scale}]$", 16, color=MUTED)
    f.text(500, 338, "Skip features concatenate with upsampled decoder features", 14, color=MUTED)

    align = f.box("align", 332, 620, 112, 48, "SliceAlign", fill=ALIGN, stroke=ORANGE, size=17, weight="bold")
    pz3 = f.box("pz3", 488, 620, 144, 48, "PhysZSSM ×1", fill=SSM, stroke=GOLD, size=17, weight="bold")
    f.edge([enc[3].right, align.left], source=enc[3].id, target=align.id)
    f.edge([align.right, pz3.left], source=align.id, target=pz3.id)
    f.edge([pz3.right, dec[3].left], dashed=True, source=pz3.id, target=dec[3].id)
    f.text(520, 596, r"$[160,D,H/8,W/8]$", 16, color=MUTED)

    e4 = f.cube("enc4", 488, 768, 56, 32, fill=ENC, label="192", size=16)
    f.text(460, 784, "E4", 16, color=BLUE, weight="bold")
    pz4 = f.box("pz4", 616, 760, 144, 48, "PhysZSSM ×2", fill=SSM, stroke=GOLD, size=17, weight="bold")
    f.edge([pz3.bottom, (560, 720), (532, 720), e4.top], source=pz3.id, target=e4.id)
    f.edge([e4.right, pz4.left], source=e4.id, target=pz4.id)
    f.edge([pz4.right, (792, 784), (792, 692), (720, 692), dec[3].bottom], color=TEAL, source=pz4.id, target=dec[3].id)
    f.text(624, 832, r"$[192,D,H/16,W/16]$", 16, color=MUTED)

    spacing = f.box("spacing", 584, 704, 140, 32, r"$s_z$ (mm)", fill=WHITE, stroke=RED, color=RED, size=16)
    f.edge([(608, 704), (608, 668)], color=RED, source=spacing.id, target=pz3.id)
    f.edge([(688, 736), (688, 760)], color=RED, source=spacing.id, target=pz4.id)
    sr = f.box("shift", 340, 712, 132, 60, ["Shift recovery", r"$0.1\,\mathcal{L}_{\mathrm{sr}}$"], fill=LIGHT, stroke=GREY, dashed=True, size=16)
    f.edge([align.bottom, (388, 692), (406, 692), sr.top], color=GREY, dashed=True, source=align.id, target=sr.id)
    f.text(434, 686, r"$\phi$", 17, color=ORANGE)
    f.text(406, 812, "Training only · Fig. 4", 14, color=GREY)

    seg = f.box("segmentation", 744, 212, 176, 32, "1×1×1 segmentation", fill=LIGHT, stroke=TEAL, size=15)
    f.edge([dec[0].top, seg.bottom], color=TEAL, source=dec[0].id, target=seg.id)
    f.edge([seg.top, (832, 172)], color=TEAL, source=seg.id, target="output")
    f.text(938, 256, "Train only", 14, color=GREY)
    sdf = f.box("sdf", 900, 272, 84, 48, ["SDF", "tanh"], fill=WHITE, stroke=GREY, dashed=True, size=15)
    aux1 = f.box("aux_half", 876, 388, 108, 48, ["Aux head", "H/2"], fill=WHITE, stroke=GREY, dashed=True, size=15)
    aux2 = f.box("aux_quarter", 852, 504, 132, 48, ["Aux head", "H/4"], fill=WHITE, stroke=GREY, dashed=True, size=15)
    for a, b in [(dec[0], sdf), (dec[1], aux1), (dec[2], aux2)]:
        f.edge([a.right, b.left], color=GREY, dashed=True, source=a.id, target=b.id)

    # Compact legend and exact convolution contract, isolated below the graph.
    f.rule(860)
    f.rect("legend_enc", 24, 878, 20, 16, fill=ENC, stroke=BLUE, radius=0)
    f.text(54, 886, "Encoder", 15, anchor="start")
    f.rect("legend_dec", 170, 878, 20, 16, fill=DEC, stroke=TEAL, radius=0)
    f.text(200, 886, "Decoder", 15, anchor="start")
    f.edge([(326, 886), (368, 886)], dashed=True)
    f.text(380, 886, "Skip / concat", 15, anchor="start")
    f.edge([(534, 886), (576, 886)], color=GREY, dashed=True)
    f.text(588, 886, "Training only", 15, anchor="start")
    f.text(24, 916, "Down: strided conv (1,2,2).  Up: transposed conv (1,2,2).  D is never downsampled.", 15, anchor="start")
    f.text(24, 942, "Encoder: two 1×3×3 convs per stage; E3 includes one extra 3×1×1 conv between them.", 15, anchor="start")
    f.text(24, 968, "Decoder: 1×3×3 conv after each concat. Cubes show feature tensors; numbers are channels.", 15, anchor="start")
    return f
