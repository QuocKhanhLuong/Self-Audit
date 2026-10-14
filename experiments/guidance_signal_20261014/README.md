# World-left-v2 attribution: isolated code and execution-host handoff

**Status: opt-in experimental code, packaged for Git publication.** See [publication scope](../../reports/guidance_research_continuation_20261014/PUBLICATION.md). Base repository `b937e3b746608725e43d3bc3cef8861666d152dc`. The current task implements and checks software locally; a separate Codex on the 4070 runs original-artifact diagnostics. No remote connection, real-data pilot or new full training run is triggered by this package.

Use a separate clean checkout/worktree of the Git commit containing this directory. For the historical ZIP overlay, use its recorded base and copy the experimental/report directories as described in that archive. The hash-pinned published `reports/stable_guidance_world_v2_20261014/` must remain unchanged. Importing top-level production `src` is deliberately avoided: `runtime.py` loads the eleven verified executed-source modules under a private package name. The earlier experimental changes are not assumed merged into production.

Use the existing original-run Python/CineMA environment. The commands below use `python`; under the local RTK convention prepend `rtk proxy`. Do not install/download alternate weights, datasets or a segmentation-finetuned encoder. Local software tests require NumPy, SciPy, PyTorch, pytest and nibabel for evaluator checks; stage generation additionally requires the original CineMA dependencies and its already present weights/config. Runtime checks pin their recorded provenance.

## Files and what they do

| Entry point | Purpose | Reference access / optimization |
|---|---|---|
| `preflight` | CPU tests and optional original identity checks | No real data inference; synthetic CPU tests only |
| `prepare` | Replay original TRAIN readout/graph; save raw votes, M/h/Z and source provenance | No references or optimizer |
| `stagewise GENERATE` | Original new checkpoint plus frozen teacher, locked VAL18, ten stages | No references or optimizer |
| `stagewise SCORE` | Verify all freezes, then score original VAL36 reference identities | Evaluation only; no fitting |
| `pilot --write-plan-only` | Bind original artifacts, selected TRAIN IDs and fixed steps | No optimizer |
| `pilot --plan … --expected-plan-sha …` | Explicit matched Q/Z technical continuation | TRAIN only; bounded steps; no evaluation |
| `stagewise_table` | Recompute available stage table from published count records | No patient arrays or inference |

The new executed-source allowlist supports the **world-left-v2 original run**. It rejects an old d5 bank/checkpoint or sanitized report copies. It does not silently replay the old run with new geometry code. Original-run evidence still missing from the publication must be measured on its own exact source path; old/new marginal counts cannot supply joint-region errors.

## 1. Local/offline preflight

From the repository root, choose a fresh output directory outside all original/frozen artifacts:

```bash
python -B -m experiments.guidance_signal_20261014.preflight \
  --output /absolute/fresh/diagnostics/software_preflight
```

Expected success: exit0, `PREFLIGHT.json` status `PASS_SOFTWARE_ONLY`, `pytest.log` and `junit.xml`. Tests include tiny synthetic CPU optimizer steps and manufactured arrays, not original-run reproduction or anatomy evidence. Offline environment flags are set and CUDA is disabled for these tests. Missing dependencies or test failures must remain failures; do not substitute a fake teacher into a real command.

## 2. Original artifact identity gate on the execution host

Fill these paths from existing run records; the placeholders are intentionally not guessed. Do not edit the original recipe to relocate its paths. If original split paths are no longer available, restore the verified original paths/mounts or stop and record the mismatch.

```bash
SA_RUN='/absolute/original/stable_guidance_world_v2_acdc_4070_20261014_024155'
SA_DIAG='/absolute/fresh_parent/world_v2_signal_diagnostics'
SA_IMAGES='/absolute/original/acdc_image_root'
SA_TEACHER='/absolute/existing/cinema.safetensors'
SA_TEACHER_CONFIG='/absolute/existing/original_cinema_config.yaml'
SA_REFS='/absolute/original/acdc_reference_root'
```

`SA_DIAG` must be external to the original run, image, reference and frozen bank roots. Create only the parent; each command's leaf output must be new. Original identities are independently pinned:

| Artifact | SHA256 |
|---|---|
| `RECIPE.json` | `bed149d73bb0527c1dbcea8ff72e9b28e093149974dcb0c5893fd43d53a6092b` |
| `bank/FROZEN.json` | `04fb2d30df6c0e0ed235dcf851a491895661b7fd4763e6b89ce95ca6811accae` |
| `student/FROZEN.json` | `85340cf5f4502d974638edd08eaca030d4642c530b5bf1680d2c21b18e09971e` |
| `student/last.pt` | `90bc16bb9f89963b452355376673e3ba9aa24b38870e4595237d75e402ce873e` |
| launcher patch | `ecbcd639fbaa010ac3f7fca9ac5ba3272278806ce92eecc8171c4833f70815ee` |

```bash
python -B -m experiments.guidance_signal_20261014.preflight \
  --bank "$SA_RUN/bank" --checkpoint "$SA_RUN/student" \
  --recipe "$SA_RUN/RECIPE.json" \
  --expected-bank-sha 04fb2d30df6c0e0ed235dcf851a491895661b7fd4763e6b89ce95ca6811accae \
  --expected-checkpoint-sha 90bc16bb9f89963b452355376673e3ba9aa24b38870e4595237d75e402ce873e \
  --output "$SA_DIAG/original_preflight"
```

Expected success: `PASS_SOFTWARE_AND_ORIGINAL_IDENTITIES`. A verified original freeze is different from a sanitized publication copy. Never rewrite hashes/placeholders to obtain a pass. The original `io.verify` checks all declared payload hashes, so a long read here is not training.

## 3. Prepare exact evidence, without labels

The initial technical subset is **patient001, patient002, patient013, patient017**, selected before new measurements: lexicographically first two TRAIN patients plus first two with existing new-bank hard RV support. Selection uses TRAIN bank metadata only and is intentionally enriched for RV opportunity, not a random population estimate. It does not assert those RV labels are correct. The same four IDs are used by both pilot arms. For the complete 80-patient TRAIN scalar audit, make another fresh output and omit `--patients`; do not change this technical subset after seeing VAL.

```bash
python -B -m experiments.guidance_signal_20261014.prepare \
  --bank "$SA_RUN/bank" --recipe "$SA_RUN/RECIPE.json" \
  --expected-bank-sha 04fb2d30df6c0e0ed235dcf851a491895661b7fd4763e6b89ce95ca6811accae \
  --patients patient001 patient002 patient013 patient017 \
  --output "$SA_DIAG/signals_train4"
```

The replay preserves original float32 vote accumulation and graph construction. It solves the linear joint mass equations, without normalizing `sV` as a class distribution or inverting entropy/reliability. Fixed tolerances are R0 max error `2e-7`, Q/reliability max error `3e-6`, and zero hard-label disagreements; the exact original solve is checked separately. The linear residual error target is `1e-10`; Z requires `h>1e-6` and a quotient truncation-error estimate at most `1e-3`. These are numerical gates fixed before new reference scoring, not semantic guarantees or rigorous floating-point interval bounds.

The executed teacher already returns float16 features; the bank save cast alone does not prove a loss of precision. Actual replay mismatch still stops the process. Record the numerical environment and verify teacher/cache identity; do not increase tolerances to fit the old result.

`FROZEN.json` is written only after reproduction succeeds. Failed replay leaves `FAILED.json` and no valid signal freeze. Invalid Z is NaN with a separate valid mask; mixed evidence remains soft, zero support remains unsupported. NPZ files and detailed source ledgers are **private patient-derived artifacts**. Only reviewed scalar summaries may later be shared.

## 4. Close the stagewise gap before deciding on target modification

Generate all ten stages on the full locked VAL18, with the frozen original teacher and new student. This is bounded inference, not a training run. It can take substantial time on the original cines; retain progress and failure receipts. The command contains no reference argument and does not accept TEST.

```bash
python -B -m experiments.guidance_signal_20261014.stagewise GENERATE \
  --run "$SA_RUN" --root "$SA_IMAGES" \
  --teacher-checkpoint "$SA_TEACHER" --teacher-config "$SA_TEACHER_CONFIG" \
  --expected-checkpoint-sha256 90bc16bb9f89963b452355376673e3ba9aa24b38870e4595237d75e402ce873e \
  --device cuda --out "$SA_DIAG/stages_frozen"
```

By default, it writes private hard stage arrays plus safe scalar fidelity/gate records. `--private-arrays` additionally persists soft probabilities locally for later joint analysis; never publish that folder. R0/Q raw argmax and gated results are separate; tied argmax uses the declared lowest-class convention and is not evidence of BG. The diagnostic R0 gate reuses Q-derived reliability and is explicitly labeled as such. Common-grid labels use a fixed nearest-floor mapping to native coordinates; references are not resized. The projection control bilinearly transports common-grid **probabilities**, then applies unchanged student gates. Native gated labels must equal the pinned `predict_cine` result.

Only after `stagewise_complete` freeze verifies all18 ×10 payloads, score independently:

```bash
python -B -m experiments.guidance_signal_20261014.stagewise SCORE \
  --generated "$SA_DIAG/stages_frozen" \
  --reference-root "$SA_REFS" --allow-verified-acdc-frame-grid \
  --out "$SA_DIAG/stage_scores"
```

The recovery flag activates the original exact frame/cine-equality check; it does not authorize arbitrary GT resizing. All36 original reference hashes/indices must match. Retain aligned/recovery and actual VAL coded/uncoded strata, all18 patients, undefined precision, and classwise failure counts. Old/new joint-region errors remain UNKNOWN without verified old prediction arrays; the current runner does not manufacture that comparison. FIDELITY_SCALARS covers each patient's full cine, not only the two scored frames. Reference-informed development is disclosed; these results cannot be called pristine validation or label-free research-wide selection.

## 5. Conditional 64-step paired technical pilot

Do not execute if the preceding frozen-source check shows that Z chiefly strengthens wrong names, if identity/reproduction fails, or if meaningful original weighted support is excluded. Plan creation itself does not optimize:

```bash
python -B -m experiments.guidance_signal_20261014.pilot \
  --bank "$SA_RUN/bank" --signals "$SA_DIAG/signals_train4" \
  --checkpoint "$SA_RUN/student/last.pt" \
  --expected-checkpoint-sha 90bc16bb9f89963b452355376673e3ba9aa24b38870e4595237d75e402ce873e \
  --plan "$SA_DIAG/pilot_plan.json" --write-plan-only \
  --patient-ids patient001 patient002 patient013 patient017 --steps 64 --seed 42
```

Record the printed `plan_sha256` independently before execution. This is the **canonical JSON digest**, not the raw formatted file digest. The run refuses a changed patient list, seed or budget against that digest.

```bash
SA_PLAN_SHA='paste_the_recorded_canonical_plan_sha256'
python -B -m experiments.guidance_signal_20261014.pilot \
  --bank "$SA_RUN/bank" --signals "$SA_DIAG/signals_train4" \
  --checkpoint "$SA_RUN/student/last.pt" \
  --expected-checkpoint-sha 90bc16bb9f89963b452355376673e3ba9aa24b38870e4595237d75e402ce873e \
  --plan "$SA_DIAG/pilot_plan.json" --expected-plan-sha "$SA_PLAN_SHA" \
  --device cuda --output "$SA_DIAG/pilot_Q_Z_64"
```

Exactly64 steps **per arm**, same original checkpoint, same fresh AdamW initialization and schedule; no original optimizer-state resume. No epoch expansion. The CLI hard limit is128. Preserve original Q activation, permissions, detached weights and denominator; change only target Q→Z on common numerically valid support. Excluded weight divided by the original clamped denominator must be at most `1e-6` for every selected patient and scheduled batch, checked before optimization. Full original Q CE is logged separately; the matched masked-Q comparator is not a falsely claimed reproduction of the untouched full-Q objective.

Outputs include per-step/fixed-batch CE/H/KL, separate semantic channels and target strata, feature/semantic gradient norms, before/after common/native diagnostics, order/initialization hashes, timing/memory and private final weights. These weights have a private technical-pilot role and are not automatically exportable by the original student evaluator. Anatomical comparison of the two trained arms needs a separately reviewed frozen export adapter/protocol; this pilot alone cannot justify an accuracy claim or a full50-epoch job.

## Stop conditions and handoff receipt

On any nonzero exit, retain the command, exit code and error; do not delete a failed receipt or reseal it as success. No references in TRAIN preparation, permission or threshold fitting. No fallback laterality, class permutation, source relabeling, BG filling, TEST selection, graph/backbone/context changes or automatic push.

Return to the research review with: exact code/source/patch hashes; original identity receipt; signal reproduction and per-class positive-evidence scalars; frozen stage and reference identities; frame confusion counts; class/geometry-stratified fidelity; and, only if actually run, paired technical-pilot invariants and traces. Label each as executed, historical, NOT RUN or failed. Keep patient arrays, masks, weights, source ledgers and credentials on the execution host. A software pass is **not** an anatomical certificate.
