# SAS-UNet paper figures: SVG-native redraw (2026-11-25)

The current deliverables are in [`final/`](final/). The original draw.io file,
preview PNGs, slice assets, and earlier report entries are preserved below.
The publication exports are generated directly from SVG with librsvg; draw.io
is not part of their rendering pipeline.

## Deliverables and rebuild

| Figure | Vector PDF | SVG source | 2× PNG |
|---|---|---|---|
| 1 · Overview | [PDF](final/fig1_overview.pdf) | [SVG](final/fig1_overview.svg) | [PNG](final/fig1_overview.png) |
| 2 · PhysZSSM | [PDF](final/fig2_physzssm.pdf) | [SVG](final/fig2_physzssm.svg) | [PNG](final/fig2_physzssm.png) |
| 3 · SliceAlign | [PDF](final/fig3_slicealign.pdf) | [SVG](final/fig3_slicealign.svg) | [PNG](final/fig3_slicealign.png) |
| 4 · Shift recovery | [PDF](final/fig4_shiftrecovery.pdf) | [SVG](final/fig4_shiftrecovery.svg) | [PNG](final/fig4_shiftrecovery.png) |

```bash
python scripts/paper_figs/build_final.py --review-dir /tmp/sas-unet-figure-review
```

Requirements: Python with `matplotlib`, `Pillow`, and `PyMuPDF`, plus
`rsvg-convert` from librsvg on PATH. No dependency installation was needed.
The build script accesses only source text and the existing figure PNGs;
it does not import a model, load a checkpoint, run inference, or read a patient volume.

Each SVG uses a 1008-unit-wide canvas and an explicit width of **7 inches**;
PDF width is **504 pt**. PNGs are **2016 pixels wide**, exactly 2× the SVG
viewBox (288 dpi at 7 inches). The smallest label is **7 pt** after print
scaling. DejaVu Sans and mathtext glyphs are vector outlines, so PDF/SVG
appearance does not depend on locally installed fonts or remote MathJax.
Only the real MRI thumbnails are raster images. Accessible SVG titles,
descriptions, and label text accompany the outlines.

[`final/sas_unet_figures.drawio`](final/sas_unet_figures.drawio) is an editable
four-page companion generated with the existing `drawio_kit` through a new
adapter. Labels, math, boxes, edges, and images remain separate native cells.
The companion uses a 1-unit grid; the historical 10-pixel-grid checker applies
to the preserved legacy file. Draw.io itself is unavailable here, so its native
cube projection and MathJax rendering have not been visually certified.
**The SVG/PDF/PNG exports are the visual source of truth.**

## Design and fidelity

The diagram-design skill informed the architecture layout, connector routing,
and render-and-look review. The requested scientific figure style, exact
architecture, four-figure scope, and 7-inch print format take precedence over
the skill's generic HTML, font, and node-count defaults. All original modules
are retained; compound operations are expanded only to show their existing
implementation.

- **Fig. 1:** one oblique projection, with depth vector `(16, -12)` for every
  tensor, three separately shaded faces, and connectors attached to the visible
  silhouette. The input and output piles became aligned filmstrips that show
  all five existing patient011 slices. Encoder channels remain
  `24/48/96/160/192`; the stage-3 skip leaves PhysZSSM, not the earlier encoder.
  Strides, the extra stage-3 `3×1×1` convolution, decoder concatenation, auxiliary
  heads, full-resolution SDF head, and training-only shift loss are explicit.
- **Fig. 2:** tokenization, pre-normalization, projected input/gate branches,
  both scan directions, masked update, residual, and physical-step equations
  are separated into readable regions. The forward and backward scans have
  independent parameters. The canonical model has no optional token z-gate.
- **Fig. 3:** adjacent-pair offset estimation and per-slice fusion occupy two
  panels. Proper mathematical subscripts replace raw text syntax. The diagram
  shows the shared reduction, `64→32→2` offset-head channels, zero initialization,
  `4 tanh`, signed neighbour warps, `[NEXT, PREV]` concatenation, and one clear
  residual path. The tiny red ellipse icons are removed.
- **Fig. 4:** the existing clean/shifted patient011 thumbnails remain unchanged.
  Shared parameters, clean-flow detachment, two-component integer shifts in
  `[-8,8]`, target `(s[i+1]-s[i])/8`, Smooth-L1, and weight `0.1` are explicit.

The requested `~/Downloads/sas_unet_figures (1).drawio` was not present.
The available `sas_unet_figures_clean.drawio` was inspected for labels and
palette alongside the worktree source. Its duplicate cell IDs prevent the
skill's strict extractor from accepting it; the original file was not changed.

## Architecture consistency and scope

An independent read of `slicessm_student.py` and `aniso_student.py` found **no
contradiction between CODEX_TASK.md's hard architecture constraints and the
implementation**. The previously reported extra stage-3 convolution is
included explicitly. All model/config/training/evaluation sources are unchanged;
no reported evaluation was rerun, no real-data training occurred, and no raw
patient volume or checkpoint is added. Existing slice provenance in `META.json`
is retained; patient scores are not recomputed or used as new evidence.

## Verification

The build checks measured glyph bounds, label collisions, visible-face ports,
orthogonal routes, node intersections, shared strokes, crossings, SVG structure,
PNG dimensions, and vector PDF content. It also hashes all existing slice assets
before and after generation. `final/BUILD_RECEIPT.json` records those checks,
the inputs, architecture-source hashes, drawing-script hashes, and output hashes.
Manual raster review and its final results are recorded in the completion note
below; automated geometry checks alone do not establish visual quality.

Repository-wide commands from AGENTS.md were also attempted:

- `python scripts/check_repo.py`: exits 1 with `NON_Q3D_TESTS_REMAIN` for seven
  existing PG/SliceSSM tests. Those files are unchanged by this figure delivery.
- `python -m pytest -q`: stops during collection because `medpy` is unavailable
  in the active Python environment; no test pass is claimed.
- `python scripts/smoke_q3d.py --out /tmp/self-audit-q3d-smoke`: completed with
  `COMPLETED_WITH_SYNTHETIC_GATE_OVERRIDE`. The measured tokenizer gate remained
  `NOT_READY`/failed (missing dead-code fraction and 200-update usage history).
  This synthetic check performed a CPU teacher update, which goes beyond the
  task's strict “no training” wording; it was invoked from the earlier repository
  verification instructions. Its artifacts are confined to `/tmp`, are not
  committed, and provide neither patient nor GPU evidence. No further training
  or evaluation commands were run.

---

# Historical status (2026-11-24; preserved)

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

## Phase 1 result (run on vast-gpu, 2026-11-24)

- Run: `slicessm-screen-20261122/S0slicessm` (seed 42), `resume_A2.pt` best_student, epoch 94, val10 fg Dice 0.9070.
- Checkpoint sha256: `b1da2a9a8723d7ecf041f44e73176e639a5c7d6908096e32157cc24376107ff6`.
- Split test20, ED frame = smaller frame_index (0; patient frames [0, 1]).
- Selected patient: **patient011**, ED 3D mean Dice 0.9255 vs test20 median 0.9256 (|diff| 0.0001, n = 20).
- Slice indices (base to apex): 0, 2, 4, 6, 8; crop box (y0, x0, side) = (64, 24, 104).
- Injected shifts (dy, dx px, seed 0): (-1,-5), (4,8), (-6,-8), (8,2), (-2,-3).
- 15 PNGs plus META.json are in `slices/`; `sas_unet_figures.drawio` now embeds them (Fig1 and Fig4: 10 images each); checker 0 violations.
- draw.io PDF/PNG export was not run (no render requested).

## Legend fix (2026-11-25)

`SliceSSMStudent` inherits `self.e3` from `AnisoStudent` (`aniso_student.py:90`): stage-3 encoder = conv 1x3x3 stride (1,2,2), conv 3x1x1, conv 1x3x3. Legend of Fig 1 now reads "Encoder block / Decoder block, 1x3x3 convs" and the note states the extra 3x1x1 conv at stage 3. No block was added or removed; checker 0 violations.

## Final format (2026-11-25)

Model name in all figures and the paper is **SASNet**. In-figure titles removed (captions in LaTeX); canvas auto-cropped to content (viewBox top offset, build receipt records `crop_top`). Fig 2 re-routed: residual starts on the input cube's left face, SiLU(u) entered from the top, SiLU(g) gap enlarged. Fig 1 label "SASNet predictions". Files: `final/fig{1..4}_*.{pdf,svg,png}` (7 in wide, vector text outlines, minimum 7 pt, PNG 2x), also copied to the paper project `figs/`. Real slices: patient011 (test20 ED, seed 42; used for illustration only).

## Qualitative figure (pipeline ready, baselines pending)

`scripts/paper_figs/export_qualitative.py --spec <spec.json>` (GPU) writes, per dataset, `in.png`, `gt.png`, `pred_k.png` and `META.json` into `qualitative/<slug>/`; `fig5_qualitative.py` builds the figure (rows = ACDC / M&Ms-1 / LA 2018, columns = input, ground truth, one per model; 3D Dice of the shown patient under each prediction). Selection rule (fixed): reference model (first in the spec) ED 3D Dice closest to the split median, ties lowest id; middle GT-foreground slice; GT-box + 15 % square crop. Template: `qualitative_spec_template.json`.

Dry run with SASNet only (vast-gpu, test split, forward passes only): ACDC patient011 slice 4 (Dice 0.9255, median 0.9256, n=20); M&Ms-1 A2L1N6 slice 7 (0.8738 vs 0.8735, n=136); LA 2018 UPT6DX9IQY9JAZ7HJKA7 slice 52 (0.9190 vs 0.9192, n=20). When baseline checkpoints exist, add them to the spec and re-run: the reference patient does not change as long as SASNet stays first.

## draw.io file: duplicate cell ids fixed (2026-11-25)

The earlier `sas_unet_figures.drawio` had auto-generated edge ids (`e2`, `e3`, `e4`) that collided with the encoder cube ids `e2`..`e4` (Fig 1 and Fig 4). draw.io resolves edge source/target by id, so edges could attach to the wrong cell and look mis-aligned. Edge ids are now `edge_<n>`, `drawio_kit.py` refuses colliding ids, and `check_drawio_xml.py` reports duplicate ids and dangling edge source/target. New file: `sasnet_figures.drawio` (native draw.io cells, real slices, SASNet naming), 0 problems. Not rendered by draw.io (not installed here).
