# Task: redraw the 4 SAS-UNet (SliceSSM) figures so they look publication-clean

Worktree: this one (branch claude/slicessm-student-20261122). Read first: reports/paper-figures-20261124/REPORT.md,
scripts/paper_figs/{drawio_kit,build_drawio,check_drawio_xml}.py, reports/paper-figures-20261124/preview/*.png,
self_audit/candidate_pg/slicessm_student.py and aniso_student.py (ground truth for the architecture),
the `sas_unet_figures (1).drawio` file in ~/Downloads (original style reference), real slices in reports/paper-figures-20261124/slices/ (META.json).

## Problem (user feedback, screenshots of the current draw.io output)
- 3D tensor blocks look ugly: stacked flat rectangles with a diagonal red-square marker, offsets/depth inconsistent, edges not meeting block faces.
- Arrows and boxes are misaligned (arrows end short of / inside boxes, labels off-centre, the "per (h,w)" label sits on the arrow, tokens column does not line up with the cube).
- Fig3 (SliceAlign) has loose, crossing residual loops, tiny icons with red ellipses, uneven spacing.
Result must look like a polished paper figure (think nnU-Net / UNETR / Mamba-paper figures): consistent isometric cubes (single oblique projection, same depth vector everywhere, three shaded faces), consistent grid, equal gaps, arrows start/end exactly on block faces, no overlaps, text centred.

## Hard constraints
- Figures only. No training, no change to model/training code, no re-run of reported evaluation.
- Architecture content must equal the code (SliceSSMStudent): channels 24/48/96/160/192, in-plane stride (1,2,2) down, no z down-sampling, encoder blocks 1x3x3 convs except one extra 3x1x1 conv in the stage-3 encoder block, SliceAlign at stage 3 (before PhysZSSM), PhysZSSM x1 at stage 3 and x2 at stage 4, skips from stage 3 take features after PhysZSSM, aux heads at H/4 and H/2, SDF head full-res (training only), shift-recovery aux (training only, +-8 px, Smooth-L1, target (s_{i+1}-s_i)/8, clean flow detached, weight 0.1). Do not add/remove modules; if the code contradicts a figure element, stop and report.
- Keep the real ACDC slices (patient011) in Fig1 and Fig4.
- Pick the drawing method that gives the cleanest geometry (recommended: generate SVG programmatically with exact coordinates, then export vector PDF + 2x PNG with cairosvg/inkscape/rsvg-convert, install via pip/uv if missing; math labels via mathtext or pre-rendered glyphs). Also keep an editable `.drawio` produced by the existing drawio_kit if feasible, but the PDF/PNG look is the priority and must NOT depend on draw.io rendering.
- Self-check by actually rasterizing and LOOKING at every PNG (Read the image), fix until: no overlap, no arrow ending in empty space, uniform spacing, readable at column width (about 7 inch wide, text >= 7 pt after scaling).
- Output to reports/paper-figures-20261124/final/: fig1_overview, fig2_physzssm, fig3_slicealign, fig4_shiftrecovery as .pdf, .png (2x) and .svg, plus build script under scripts/paper_figs/. Do not commit raw patient volumes.
- Commit script + outputs on this branch (message ends with: Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>). Do not push.
- Update REPORT.md with what changed and any contradictions found.
