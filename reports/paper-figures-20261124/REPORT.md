# SAS-UNet model figures: status (2026-11-24)

Branch `claude/slicessm-student-20261122`. Figures only: no training, no change of model or training code, no evaluation re-run.

## What is done

- `scripts/paper_figs/export_fig_slices.py` (Phase 1, needs the GPU/data machine). Not run yet; see "Phase 1" below.
- `scripts/paper_figs/build_drawio.py` + `drawio_kit.py` generate the four pages (Fig1 Overview, Fig2 PhysZSSM, Fig3 SliceAlign, Fig4 ShiftRecovery) as
  a grid-aligned `.drawio` and check them. `scripts/paper_figs/check_drawio_xml.py` re-checks the finished file.
- `sas_unet_figures.drawio` is the generated file. **Its 20 image cells hold a neutral grey placeholder, not slices**, because no real slice PNGs exist yet.
  After Phase 1 run: `python scripts/paper_figs/build_drawio.py --slices reports/paper-figures-20261124/slices` and the real images are embedded (base64).
- `preview/fig{1..4}.png` are matplotlib previews of the layout (raw LaTeX text, no draw.io fonts). They are not draw.io renders.

## Not done (and why)

- **draw.io was not run** (the user asked for code only, no render). No `fig{1..4}.pdf` or 2x PNG exports exist; export them in draw.io with
  `drawio -x -f pdf --crop -p <i> -o fig<i+1>.pdf sas_unet_figures.drawio` after the real slices are embedded.
  The rule "look at every render" therefore has not been met by draw.io; I checked geometry with the built-in checker and the matplotlib preview only.
- Math rendering in the PNG export is unchecked. `math="1"` is set on every page.

## Phase 1: how to run (GPU machine)

```
python scripts/paper_figs/export_fig_slices.py --config <run>/config.json --run <S0slicessm seed-42 run dir with resume_A2.pt> \
    --out reports/paper-figures-20261124/slices
```
Checkpoint: SliceSSM `S0slicessm`, seed 42 (user's choice), `best_student` of `resume_A2.pt`. Patient rule: test20 patient whose ED 3D mean Dice
(RV/MYO/LV) is closest to the test20 median (ties: lowest id); the script aborts if the chosen Dice is not within 1 Dice point of the median.
ED = the smaller `frame_index` of the patient's two labelled frames (assumption; both indices go to META.json). It writes `in_k.png`, `out_k.png`,
`shift_k.png` (k=0..4, base to apex, 400x400) and `META.json` with patient id, frame, slice indices, crop box, Dice, median, checkpoint path, sha256 and git hash.
Patient id, Dice and median are therefore **not known yet**; this report will need them added from META.json.

## Page checks (built-in checker, `build_drawio.py`)

| Page | Shapes | Edges | Crossings (limit 2) | Rule violations | Render rounds |
|---|---:|---:|---:|---:|---:|
| Fig1 Overview | 74 | 36 | 1 | 0 | 0 (no draw.io) |
| Fig2 PhysZSSM | 41 | 20 | 1 | 0 | 0 |
| Fig3 SliceAlign | 30 | 18 | 0 | 0 | 0 |
| Fig4 ShiftRecovery | 21 | 9 | 0 | 0 | 0 |

The crossings are drawn with `jumpStyle=arc`: Fig1 red spacing line to PhysZ-SSM x1 crosses the data edge into the last encoder block; Fig2 the Delta line to the backward scan crosses the forward-scan output.
Rules checked per page: all vertex and waypoint coordinates on the 10 px grid; every edge orthogonal with explicit waypoints and exit/entry constraints; no segment through a block or label (10 px clearance for labels);
at least 20 px between blocks. Output of `check_drawio_xml.py` on the final file:

```
Fig1 Overview: 74 vertices, 36 edges, 10 embedded images, gridSize=10, math=1, 0 problems
Fig2 PhysZSSM: 41 vertices, 20 edges, 0 embedded images, gridSize=10, math=1, 0 problems
Fig3 SliceAlign: 30 vertices, 18 edges, 0 embedded images, gridSize=10, math=1, 0 problems
Fig4 ShiftRecovery: 21 vertices, 9 edges, 10 embedded images, gridSize=10, math=1, 0 problems
```

## Contradiction between the drawing and the code (not fixed in the figure)

`aniso_student.py:90`: the stage-3 encoder block is `conv 1x3x3 stride (1,2,2)` + `conv 3x1x1` + `conv 1x3x3`. It contains a slice-axis `3x1x1` convolution,
but Fig1's legend calls all encoder blocks "In-plane encoder block" and the paper text said inter-slice information is exchanged only by SliceAlign and PhysZSSM.
Per the task rules I did not change the architecture drawing. Options: (a) rename the legend entry (e.g. "Encoder block (1x3x3 conv; one 3x1x1 conv at stage 3)"), or (b) leave the figure and say it in the caption.
The paper draft (`3_Methodology.tex`) now states the exception.

Verified against the code: channels 24/48/96/160/192 (2,531,734 deployed parameters), strides (1,2,2), PhysZSSM x1 at stage 3 and x2 at stage 4, SliceAlign before PhysZSSM at stage 3,
stage-3 skip taken after PhysZSSM, aux heads on the H/4 and H/2 decoders, SDF head on the full-resolution decoder, Delta scaled by s_z/10 mm with padded slices at zero.
