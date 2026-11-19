# Agent prompt — CBKD-SA lean: fix the two pain points of Codebook KD · 2026-11-18 (rev 4, single seed)

> **Repo copy (2026-11-19).** This file is the **method spec** for CBKD-SA. For the Modal run that replaces A4, scope, compute, arms, paths and order are set by `CBKD_SA_A4_MODAL_PROMPT_20261119.md` in this folder, which wins on any conflict (only A4-SA is run; the old vast paths below no longer exist).
>
> Self-contained. Extends campaign `pg-k8-20261116`.
> Status: **proposed, not implemented, not trained.** rev 4 supersedes rev 1–3. Constraint from user: **one seed (42), minimal arms, no time for extras.**

---

## 0. Context

- Repo `QuocKhanhLuong/Self-Audit` (local `/Users/alvinluong/Self-Audit`), branch `main`, GPU `vast-gpu`.
- Old runs `/root/mount_data/runs/pg-k8-20261116/{S1,S2,S3,A1,A2,A3,A4}`; new runs `/root/mount_data/runs/cbkd-sa-20261118/<stage-or-arm>`.
- Unchanged from `pg-k8-20261116` §0/§2/§8: ACDC TRAIN80/VAL18/TEST50, K8 = 16 ED/ES masks, U = other TRAIN cine frames, student UNETR++-lite (16,32,64,128,256) from S1, frozen MedNeXt-L teacher S2 (region + SDF heads), seed 42, optimizer, 8000 updates (80/epoch), `inplane_strong_v1`, student-only VAL18 FG selection, TEST50 monitoring only, per-epoch train/val/test logs, one console line per epoch.
- CBKD-v1 (`aux_code_branch`) keeps exact behaviour. New variants: `cbkd_sa`, `pm_raw`.

## 1. The two pain points of CBKD-v1

**P1 — No signal from U reaches the deployed decoder.**
v1 has only code-level terms on a train-only branch (stage3 → `P_S` → VQ → `H_S^Q`); Pred-Match is off (`qpred = 0`). On U, gradients touch stem→stage3 only through that branch; the decoder learns from 8 patients alone. Evidence: A3 train FG 0.957 vs val 0.696 / test 0.653. The source paper (arXiv 2609.37243) reports that the code term alone ≈ no KD; Pred-Match carries the gain.

**P2 — The codes cannot carry myocardium geometry (representation ceiling).**
`H_T^Q` decodes each code into the *pooled class fractions* of one stage3 cell. A stage3 cell is 8 px ≈ 11 mm in-plane (at ≈ 1.37 mm spacing) and 2 slices (10–20 mm) in z, while a normal LV wall is ≈ 5–6 mm. Almost every MYO cell is mixed, and fractions do not say where the wall lies inside the cell. No loss change can transfer MYO geometry through such codes.

P1 explains A3's failure. P2 limits what CBKD can add even when PG supplies U signal (A4), and it contradicts the stated goal "preserve medically important information".

## 1b. Code-level findings on v1 (read from `self_audit/candidate_pg`, commit fd12e91; logs not yet inspected)

- **The assignment KL was saturated and carried zero gradient.** `assign_kl` computes `log(clamp_min(ρ_S, 1e-12))` on probabilities from `softmax(−‖z−C‖²/τ)` with τ = 1 on raw (non-normalized) features. When the student's probability on the teacher's code falls below 1e-12, the term equals −log(1e-12) = 27.63 and `clamp_min` blocks the gradient. Two terms (L + U) give 55.26, matching the reported median `loss_cbkd` ≈ 54.6 and the flat A3/A4 loss for all 8000 updates.
- **Student-side code collapse from the first logged row.** In `training_step.py` the diag fields `code_perplexity` / `code_dead` come from the **student's** assignments on clean U (perplexity 1.0, 127/128 dead), not from the S3 codebook. Every student cell maps to one code; with no assign gradient and commitment pulling all `z_S` toward that one code, the collapse never breaks.
- **`code_coverage` = 1.0 is from a different source** (`usage_ema` of the frozen S3 model, never updated in the arm), so it does not contradict `code_dead`; the diag mixes two sources under similar names.
- **`codes_per_class = 0` in `S3_RECEIPT.json` is a reporting artifact**: the receipt calls `codebook.metrics(last_assignments, last_valid_mask)` without labels, so `usage_metrics` returns its default zeros. The receipt perplexity (68.9) is from the last labeled batch only, not the whole TRAIN set.
- **Consequence:** in A3/A4, CBKD-v1 transferred no codebook knowledge. Its only live gradients were `qtask` (soft CE on pooled GT, flowing through the STE into stage3 like an auxiliary stage3 head) and commitment toward a single code. The A2 → A4 gap therefore cannot be attributed to codebook KD until explained.

Required guards for any new code path (tests in §9 are extended):
- No `clamp`-then-`log` on probabilities: use `log_softmax` of logits.
- l2-normalized `z` and `C` (cosine), with τ calibrated from data, so distances are bounded.
- Log student code-usage perplexity from the first CBKD update; a value of 1.0 at the first row must be visible in the console line once (single warning, not per step).
- Diag field names state their source (`student_code_perplexity`, `s3_usage_ema_coverage`).

## 2. Fix (one method, two parts)

**Part A (P2) — shape-atom codebook.** Replace the pooled-fraction decoder with an atom decoder `D_A`: each code decodes into the **local patch of its cell at native resolution** (2 × 8 × 8 voxels: 4 class logits + 3 SDF channels). Codes become local anatomical configurations (wall segment, LV–MYO edge, RV insertion, apex cap, background).

**Part B (P1) — full-resolution Pred-Match to the atom-reconstructed teacher.** On clean weak U and on L_K, the student's full-res prediction matches the teacher prediction *re-assembled from its codes* through `D_A`. This sends U signal into the whole student (encoder + decoder), and the target is the teacher filtered through the atom dictionary (the paper found the codebook-branch prediction a better target than the raw teacher prediction).

Everything else from rev 2/3 (greedy selection, quotas, code-level weak→strong CE) is **dropped** for this campaign: it is not needed to test P1/P2 and each piece adds a confound that one seed cannot resolve.

```
S3-SA (offline, teacher frozen)
teacher L3 → align(1,8,8) → P_T → VQ(C 128×64, cosine) → D_A → patch 2×8×8 (classes + SDF)
   targets: L_K GT patches; U = teacher full-res probs/SDF (confidence-weighted)

Student step (added on top of the shared step)
clean weak U, L → teacher → k*_T, margin m_T → assemble D_A(k*_T) → q_T^full   (detached)
clean weak U, L → student-only → p_S (full-res) → L_pm = ‖p_S − q_T^full‖² · m_T (U) / ·1 (L)
L_K             → stage3 → P_S → hardVQ(C)+STE → frozen D_A → patch → L_atom vs GT patch (+0.25 commit)
```

## 3. Phase 0 — pain-point measurements, no training (`reports/cbkd-sa-20261118/PHASE0.md`)

O1 and O2 use **ground truth only** and decide whether P2 is real; they take minutes.

- **O1 · Pooled-code oracle (ceiling of v1's representation).** For every VAL18 volume (native geometry): pool the one-hot GT to the stage3 grid (1,8,8) as class fractions (valid-aware), upsample back to native resolution (trilinear), argmax. Report FG/RV/MYO/LV Dice and HD95 vs GT, and the **mixed-cell fraction** (no class > 0.7) for all cells and for cells containing MYO. This is the best any pooled-fraction code could ever do, even with a perfect student.
- **O2 · Shape-atom oracle (ceiling of Part A).** On TRAIN L_K GT patches (2×8×8 one-hot), run k-means with K = 128; on VAL18, replace every cell's GT patch by its nearest centroid, argmax, reassemble. Same metrics as O1.
- **O3 · Raw teacher (S2) on VAL18/TEST50:** FG/RV/MYO/LV, HD95 — the student ceiling reference.
- **O4 · Old S3 codebook:** `q_T` VAL FG on the pooled grid, perplexity, dead codes.
- **O5 · All 4 old arms at their VAL-selected epoch** (train/val/test; FG, RV, MYO, LV, HD95).
- **O6 · Switch-bug check:** diff resolved configs A1..A4; confirm shared forwards are identical when `cbkd.enabled` flips; compare A2 vs A4 `diag.jsonl`: mean `a_v`, coverage per class, `τ_c`, `loss_pg`.

Reading: if O1 MYO Dice is clearly below O2 MYO Dice and O3 teacher MYO Dice, P2 is confirmed quantitatively. All items are reported; none gates the next phase.

## 4. Phase 1 — S3-SA (offline, teacher frozen; `candidate_pg/shape_atoms.py`)

- Tap teacher level3 (phase 2,8,8), align to (1,8,8) with `align_feature`, `P_T` 1×1×1 256→64, codebook `C` 128×64 with l2-normalized (cosine) lookup, commitment 0.25, usage-entropy 0.01, dead-code restart every 500 updates.
- `D_A`: per-cell MLP 64 → 256 → (2·8·8·(4+3)). Class logits + SDF (tanh, normalized by 5 mm, q3d convention).
- Losses on TRAIN only: L_K GT patches (CE + Dice; L1 SDF from GT); U teacher full-res probabilities (soft CE, weight = teacher max-prob) and teacher SDF (L1 on voxels with teacher max-prob ≥ 0.9).
- 4000 updates. Freeze `P_T`, `C`, `D_A`.
- `τ*` = median over TRAIN cells of (2nd-nearest − nearest cosine distance). `m_T = 1 − exp(−(d2 − d1)/τ*)`, detached, 1 on labeled cells.
- Offline VAL18 report: teacher reconstructed from codes (`assemble(D_A(k*_T))`) vs raw teacher vs O2 oracle — FG/RV/MYO/LV, HD95; perplexity, dead codes; margin distribution for cells containing a class boundary vs interior cells.

## 5. Phase 2 — student terms (`cbkd.variant = "cbkd_sa"`)

- `L_pm = Σ m_T·valid·‖p_S − q_T^full‖² / Σ m_T·valid` (sum over classes), on clean weak U (no CutMix) and on L_K (weight 1). `p_S` = student-only softmax at native resolution. Gates receive no gradient.
- `L_atom = segloss(D_A(hardVQ_C(P_S(stage3))), GT patch)` on L_K + `0.25·‖z_S − sg(e)‖²` on L_K and clean weak U. `D_A` is the **frozen teacher atom decoder**, so the student's codes must mean the teacher's local anatomy; no new student head.
- `loss += λ_cb(upd)·(L_pm + L_atom)`, `λ_cb = 0.5`, 800-update labeled-only warmup, same ramp as v1. Fixed; no tuning.
- Control variant `pm_raw`: `L_pm` with target = **raw teacher** full-res softmax, uniform weight; no codes, no `L_atom`.

## 6. Phase 3 — arms (seed 42 only)

| Arm | PG | CBKD | Question |
|---|---|---|---|
| A1 | off | off | reuse |
| A2 | on | off | reuse |
| A3 | off | v1 | reuse |
| A4 | on | v1 | reuse |
| **A3-SA** | off | cbkd_sa | Does fixing P1+P2 make CBKD useful on its own? |
| **A4-SA** | on | cbkd_sa | Does it improve the full model over A4? |
| **A3-raw** | off | pm_raw | Is the atom dictionary better than plain KD from the raw teacher? |

Same init, teacher, artifacts, optimizer, updates, augmentation, selection rule; resolved configs differ only in `pg.enabled`, `cbkd.enabled`, `cbkd.variant`. Verify by config diff and artifact hashes. Run order if GPU is short: A3-SA → A4-SA → A3-raw.

## 7. Single-seed analysis rules

- Primary metrics: **MYO Dice and HD95** (P2 is about MYO geometry), then FG.
- Per-patient paired tests on TEST50 (Wilcoxon + 10,000 bootstrap of the mean difference, 95 % CI); also report on VAL18. Comparisons: A3-SA vs A1, A3-SA vs A3, A3-SA vs A3-raw, A4-SA vs A4, A4-SA vs A2.
- A difference is called **supported** only if the bootstrap CI excludes 0 on TEST50 and the sign agrees on VAL18; otherwise **inconclusive**. State in the report that seed variance is not estimated.

## 8. Hard constraints

1. Inspect first: `candidate_pg/{engine,student_pg,codebook,reliability,views}.py`, v1 CBKD code, S3 outputs, q3d SDF utilities. Write `reports/cbkd-sa-20261118/INSPECTION.md` before editing.
2. v1 behaviour and all existing suites unchanged.
3. Label budget: supervised terms use L_K masks only; U gives images and detached teacher outputs; S3-SA, k-means oracle fitting and `τ*` use TRAIN only (O1/O2 *evaluation* on VAL18 uses VAL GT for measurement only, never for training or selection).
4. Export strips `atom_branch.`, `P_S.`, `fusions.`, `mimic_heads.`; assert no teacher/codebook/`D_A` keys; cold-load test.
5. Logging per old §8; add columns `loss_pm, loss_atom`; `diag.jsonl` adds mean `m_T` (boundary vs interior cells), student code-usage perplexity, and (A4-SA) disagreement rate between `q_T^full` and the PG target.
6. TEST50 never used for any decision.
7. Commit + push after implementation/tests and after the report; report the remote SHA.
8. Optimization protocol (from the PG pipeline audit of 2026-11-18): log the global grad norm and per-term grad norms (sup / PG / CBKD) every 50 updates; persist every CBKD sub-loss; do not let CBKD-SA share one global clip with the segmentation loss (clip per loss group and log the factor); start CBKD-SA terms at their own warmup, not at the same update as PG.

## 9. Tests (pytest, `tmp_path`)

1. `pool_fraction` / `upsample_argmax` (O1) and `patch_split` / `assemble_full_res` (O2, Part A): exact round trip on a toy volume with known answer; pad cells excluded.
2. `code_margin`: 0 at d1 = d2, → 1 as the gap grows; detached.
3. `pred_match_full`: zero when `p_S == q_T^full`; weights respected; no grad to target.
4. `vq_hard_ste_cos`: exact codewords; grad reaches `z_S`; `C`, `D_A` unchanged after a step.
5. Gates get no grad from `L_pm` or `L_atom`.
6. Resolved configs differ only in the fields in §6.
7. v1 outputs bit-identical before/after on a fixed seed and toy batch.
8. Export clean; exported logits equal in-memory student.
9. **Scale test:** with realistic feature scale (‖z‖ ≈ 10–30, random codebook), every CBKD-SA term has finite, non-zero gradient w.r.t. `z_S` and student parameters; reproduce the v1 saturation (gradient exactly 0 at 27.63) as a regression test of the old function.
10. Student code-usage perplexity > 1 at initialization on a toy batch with l2-normalized features.

## 10. Non-goals

Code selection (greedy/quota/leave-one-out), code-level weak→strong CE, stage2 codebooks, cine temporal distillation, coupling `m_T` into PG, extra seeds, λ/τ tuning, M&Ms.

## 11. Report (`reports/cbkd-sa-20261118/README.md`)

Separate measurements / interpretation / deviations.
- Phase 0: O1 vs O2 vs O3 table (FG/RV/MYO/LV, HD95, mixed-cell fraction); O4–O6.
- S3-SA: code-reconstructed teacher vs raw teacher vs O2 oracle on VAL18; perplexity; margin boundary vs interior.
- All 7 arms at VAL-selected epoch: train/val/test FG, RV, MYO, LV, HD95, ASSD; learning curves.
- Paired tests per §7 with verdicts: P1 fixed? (A3-SA vs A3/A1, train–val gap); P2 fixed? (A3-SA vs A3-raw on MYO/HD95; S3-SA reconstruction vs O1); full model (A4-SA vs A4).
