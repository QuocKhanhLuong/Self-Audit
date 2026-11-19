# Agent prompt: W2S-Match + gated Codebook KD (2026-11-19, rev 1, single seed)

> Self-contained. Read it all before touching the repo.
> Status: **proposed, not implemented, not trained.** User constraints: **one seed (42), minimal arms, no time for calibration studies.** Target: final **student-only** inference Dice of about 0.89–0.90 on the official ACDC test set.

---

## 0. Context

- Repo `QuocKhanhLuong/Self-Audit` (local `/Users/alvinluong/Self-Audit`), branch `main` at `dc102c9` (contains the analysed `fd12e91` plus 12 PG memory / recovery fixes). GPU: Modal A100-40GB (see Compute below).
- Package: `self_audit/candidate_pg` (`teacher_training.py`, `training_step.py`, `engine.py`, `student_pg.py`, `teacher_mednext.py`, `reliability.py`, `losses.py`, `codebook.py`, `views.py`, `cinema_init.py`, `evaluation.py`, `export.py`) and `self_audit/candidate_fuse` (student base `FuseStudent`).
- Base config: `configs/candidate_pg/pg_k8_base.json` (resolved copy: `reports/pg-server-20261116/PG_K8_CONFIG.json`).
- Old runs (`/root/mount_data/runs/pg-k8-20261116/...` on the old vast server) are gone; only the Drive bundle `pg_k8_A1-A4_bundle.zip` remains. New runs: `/root/mount_data/runs/w2s-match-20261119/<stage-or-arm>`. Report: `reports/w2s-match-20261119/`.
- Related: `reports/agent-prompts-20261119/CBKD_V2_AGENT_PROMPT_20261118.md` (CBKD-SA rev 4). Design notes live outside the repo; this prompt wins on any conflict.

### Decisions already made (do not reopen)

| Item | Decision |
|---|---|
| Split | ACDC training 1–100 split 80 train / 20 val as in `splits/acdc_patient_split_seed42.json`; official test = patients 101–150 (TEST50). Duplicates are **not** removed or moved. |
| Labels | K8: 8 TRAIN patients, ED/ES masks (16 volumes). Reuse the existing K8 list of campaign `pg-k8` unless §3 inspection shows it is not a subset of the seed42 TRAIN80. |
| Unlabeled U | Same as `pg-k8`: the remaining TRAIN cine frames. |
| Student | UNETR++-lite `PGStudent` (channels 16/32/64/128/256, no z down-sampling), encoder initialized from the existing CineMA S1 artifact. 3D. |
| Teacher W | MedNeXt-L 3D, trained with the repo teacher stage (`run_teacher_stage`, `method: unimatch_v1`, SDF head and SDF consistency kept). **Only change: fixed threshold 0.95** via `teacher.semi.threshold.clip = [0.95, 0.95]`. |
| Codebook | CBKD-SA (shape atoms, SDF channels kept) fitted on W, used inside the W2S step with an arbiter gate (§2.3). |
| Thresholds | τ = 0.95, τ_v = 0.99 (raised to 0.999 by the saturation check §5), stability = all visible views agree and ≥ 2 visible, α_max = 0.5. |
| Stability | Computed within the step (no EMA). |

### Compute: Modal A100-40GB, total budget about USD 30 (vast server is gone)

- Launcher `scripts/modal_app.py` (functions `pull_drive`, `cpu`, `gpu`). One volume `self-audit-mount` is mounted at `/root/mount_data`, so old paths keep working. Layout: `raw/ACDC/{training,testing}`, `bundles/pg-k8-20261118/`, `prepared/...`, `runs/w2s-match-20261119/`, `hf/` (CineMA download cache).
- Raw ACDC comes from Google Drive (`pull_drive`, done by the user). Old prepared manifests and run artifacts on vast are **not** available. Regenerate prepared data from `raw/ACDC` with the repo's own preparation code; old SHA locks will not match, so write new locks into the new configs and report both.
- Look for the CineMA S1 encoder artifact in `bundles/pg-k8-20261118/` (unzipped). If it is not there, rerun S1 (10,000 updates, unchanged config) before any student arm and report the cost.
- Cost model: about USD 2.5 per hour for the `gpu` function (A100-40GB + 4 CPU + 24 GB RAM). Before each long stage run a 200-update timing probe, extrapolate, and keep `reports/w2s-match-20261119/BUDGET.md` (spent, projected, remaining). If the projected total for the remaining stages exceeds the remaining credit, **stop and report options before starting the stage**; never silently shorten a stage or drop an arm.
- Evaluation cost: VAL20 every epoch (VAL18 is a subset, free). **TEST50 only once per arm, on the VAL20-selected checkpoint** (deviation from `pg-k8`, which monitored TEST50 every epoch; it was never used for decisions).
- Checkpoint every 500 updates into the volume; every training command must resume from its last checkpoint. Launch long stages with `modal run --detach scripts/modal_app.py::gpu --cmd "..."`. Run tests and data preparation on the `cpu` function.

## 1. Goal and arms

| Arm | What | Depends on | Purpose |
|---|---|---|---|
| **W** | Teacher stage, threshold fixed 0.95 | – | Weak supervisor; "weak" reference |
| **CB** | CBKD-SA codebook fit on frozen W | W | Codebook for S2 |
| **S2** | Student, W2S arbiter targets + gated codebook KD | W, CB, S1 | Main method |
| **S0** | Student, UniMatch self-training (own pseudo-labels), no teacher, no codebook | S1 | Baseline |
| S2-arb | S2 without codebook terms | W, S1 | Optional ablation / fallback (§6) |

GPU order if only one GPU: W → saturation check → CB → S2 → S0 → S2-arb (if time). S0 does not depend on W; start it whenever a GPU is free.

All student arms share: same S1 encoder artifact, optimizer, 8000 updates (80/epoch), `inplane_strong_v1`, CutMix (`full_depth: true`, area 0.25–0.5, `align_px: 16`), FP `dropout3d` 0.5 on bottleneck and all skips, batch L=1 / U=2 (A, B), λ_u schedule (800 labeled-only warmup, existing ramp), same global gradient clip as `pg-k8`, VAL20 student-only FG selection. Resolved configs of S0, S2, S2-arb may differ **only** in `pg.target_mode` and `cbkd.*`; verify by config diff.

## 2. Method specification

### 2.1 Unlabeled loss (all student arms)

Reuse `_pg_loss` in `training_step.py` (per-voxel weighted CE + weighted Dice on a packet `{y, weights, valid}`). Streams s1, s2, fp with weights 0.25 / 0.25 / 0.5. The arms differ only in how the packet `(y, w)` is built:

- `pg.target_mode = "self"` (S0): y = argmax of the student's own clean weak prediction (stop-grad), w = 1 where its max-prob ≥ 0.95, else 0. Mixed by the same CutMix boxes as the strong views.
- `pg.target_mode = "w2s_arbiter"` (S2, S2-arb): (y, w) from §2.2.

Do not use the class-adaptive threshold for students: fixed 0.95 (same `clip = [0.95, 0.95]` trick if the code path goes through `ClassAdaptiveThreshold`).

### 2.2 W2S arbiter (new, `candidate_pg/w2s_arbiter.py`)

Inputs per view v ∈ {s1, s2, fp}, all detached: teacher clean-weak region softmax p_T, student clean-weak softmax p_S (both mixed by the view's CutMix box; identity for fp), stability flag st, schedule α(t), valid mask. Let y_T, c_T = argmax / max of p_T, and y_S, c_S likewise. Rules are applied **in order; the first match wins**:

1. y_T = y_S and max(c_T, c_S) ≥ τ → (y_T, 1)
2. y_T ≠ y_S and c_T ≥ τ_v → (y_T, 1)  *teacher veto*
3. y_T ≠ y_S and c_S ≥ τ and st → (y_S, α(t))  *student override*
4. y_T ≠ y_S and c_T ≥ τ → (y_T, 1 − α(t))  *teacher fallback*
5. otherwise → w = 0

Also return the integer rule map r ∈ {1..5} (0 for invalid/pad voxels, w = 0 there).

**Stability, source-aligned.** For a voxel of source image A at location x, the visible views are fp (always), s1 if x ∉ M1, s2 if x ∉ M2. For a voxel of source B at x: s1 if x ∈ M1, s2 if x ∈ M2. st = true iff the student argmax of every visible view equals y_S of that source voxel **and** at least 2 views are visible. Use the current step's strong / FP predictions (detached). Map st into each view's mixed coordinates with the same boxes.

**α(t):** 0 until 20 % of total updates, linear to 0.5 at 50 %, constant afterwards.

### 2.3 Gated codebook KD (S2 only)

Codebook artifacts from arm CB (§4): frozen P_T, C, D_A, τ*.

- Teacher side, per step on clean weak A (stop-grad): L3 of W (phase (2,8,8)) → align to (1,8,8) → P_T → l2-norm → nearest code k*_T, margin m_T = 1 − exp(−(d2 − d1)/τ*) → D_A → assemble atom patches → q_T at native resolution (class probabilities).
- **L_pm** on U only: the student clean-weak forward of A keeps gradient (p_S^A). Gate g = 1 where the arbiter rule map of the fp view (identity coordinates of A) is 1, 2 or 4, else 0. L_pm = Σ g·m_T·valid·‖p_S^A − q_T‖² / Σ g·m_T·valid (sum over classes; m_T upsampled nearest to full resolution). **No L_pm on labeled data** (deviation from rev 4: GT is available there, and matching a weaker teacher on labeled data can only cost accuracy).
- **L_atom** on L_K: stage3 → P_S (128 → 64) → l2-norm → hard VQ with C (STE) → frozen D_A → patch; CE + Dice on class logits and L1 on SDF against GT patches; plus 0.25 commitment on L_K and clean weak U.
- Loss: L = L_sup + λ_u·L_u + λ_cb·(L_pm + L_atom), λ_cb = 0.5. λ_cb onset at update 1200 with a 400-update linear ramp (staggered from the λ_u onset at 800 and the α onset at 1600).

Guards (from the CBKD-v1 audit): no clamp-then-log on probabilities (use `log_softmax`), l2-normalized z and C, log student code-usage perplexity from the first CBKD update, diag field names state their source.

## 3. Phase 0: inspection, no training (`reports/w2s-match-20261119/INSPECTION.md`)

Inspect before editing and record:

1. Prepared manifests (`/root/mount_data/prepared/ACDC/fuse-k8-20261113/*.json`, `configs/candidate_fuse/semi_k8.json`): are TRAIN/VAL equal to the seed42 80/20 lists? How was VAL18 derived? Is K8 ⊂ TRAIN80? Which frames form U?
2. If the prepared VAL has 18 patients, build VAL20 from the seed42 list (add the missing patients from the raw data with the same preprocessing). Keep logging VAL18 as well for comparability with old runs.
3. How `_pg_loss` packets are built today (fusion target, `reliability.voxel_weights`), and where the class-adaptive threshold enters the student path.
4. Whether `ClassAdaptiveThreshold` with clip `[0.95, 0.95]` returns exactly 0.95 for all classes (unit check).
5. Availability and hashes of the S1 CineMA encoder artifact; whether W must be retrained (yes: threshold changed; old S2 teacher is not reused).
6. Teacher L3 tap and the SDF output (`public_sdf`, `foreground3`) used by CB.

Stop and report if K8 is not a subset of TRAIN80 or if any VAL20/TEST50 patient appears in TRAIN. A missing S1 artifact is not a stop: rerun S1 as described under Compute.

## 4. Phase 1: implementation and tests

- `w2s_arbiter.py` (§2.2), `pg.target_mode ∈ {"fusion" (existing default), "self", "w2s_arbiter"}` wired into `student_training_step`.
- CBKD-SA as specified in rev 4 §4 (codebook fit) and §2.3 here (student terms, gate). New variant `cbkd.variant = "cbkd_sa_gated"`. CBKD-v1 behaviour unchanged.
- Logging every 50 updates (`diag.jsonl`) and per epoch (`epochs.csv`): per-rule voxel fractions overall and per class, override share of foreground disagreement voxels, foreground coverage vs the K8 labeled prior, α, λ_u, λ_cb, τ_v in use, mean m_T on gated voxels, student code perplexity, `loss_sup / loss_u / loss_pm / loss_atom`, global grad norm and clip factor, per-term grad norms (sup / u / cbkd via `autograd.grad` on the 50-update rows only).
- Watchdogs (console WARN once each, do not stop): override share > 30 % of FG disagreement voxels; FG coverage drift beyond ±10 % relative to the K8 prior; override share < 0.5 % for 500 updates after α onset (veto saturation).

Tests (pytest, `tmp_path`), all must pass before any training:

1. Arbiter: each rule fires on a hand-built voxel; first-match order; rule 1 with both confidences < τ gives w = 0; pad voxels w = 0, r = 0; outputs carry no grad.
2. α schedule values at 0 %, 20 %, 35 %, 50 %, 100 %.
3. Source-aligned stability on a toy CutMix with known boxes: visible-view counts 1/2/3 for A and 0/1/2 for B; A voxel inside both boxes is never stable.
4. Gate: L_pm is exactly 0 when every voxel is rule 3 or 5; gradient reaches student parameters only through p_S^A; no gradient to W, P_T, C, D_A.
5. `target_mode = "self"` builds y from the student's own clean weak prediction with fixed 0.95.
6. Fixed-threshold clip returns 0.95 for all four classes after many EMA updates.
7. CBKD-SA scale test (rev 4 test 9) and code-usage perplexity > 1 at init (rev 4 test 10).
8. Config diff of S0 / S2 / S2-arb shows only the intended fields.
9. Export: student-only checkpoint has no teacher / codebook / `P_S` / `D_A` keys; cold-load logits equal in-memory logits.
10. Existing suites unchanged; CBKD-v1 and `fusion` target bit-identical on a fixed seed and toy batch.

Commit and push after tests pass; record the remote SHA.

## 5. Phase 2: teacher W and saturation check

- Train W with the base teacher config, only `teacher.semi.threshold.clip = [0.95, 0.95]`, 8000 updates. Select on VAL20 FG Dice; freeze. Report W on VAL20 / VAL18 / TEST50 (FG, RV, MYO, LV, HD95).
- **Saturation check (minutes, no training):** run frozen W on a fixed random subset of U (≥ 200 volumes). Report the fraction of predicted-foreground voxels with c_T ≥ 0.99, per class, and a c_T histogram. If the overall fraction > 0.90, set τ_v = 0.999 for S2 / S2-arb; else keep 0.99. Record the decision in the report; no other tuning.
- Stop and report if W VAL20 FG < 0.60 (pipeline problem, not a weak teacher).

## 6. Phase 3: codebook CB (rev 4 §4, on W)

- Fit P_T, C (128 × 64, cosine), D_A (patch 2 × 8 × 8 × (4 class + 3 SDF)) on TRAIN only, 4000 updates; targets: K8 GT patches, and W full-res probabilities / SDF on U (confidence-weighted; SDF only where W max-prob ≥ 0.9). Freeze. Compute τ*.
- Report on VAL20: code-reconstructed W (q_T) vs raw W (FG / MYO Dice, HD95), perplexity, dead codes.
- **Fallback rule:** if q_T FG Dice on VAL20 is more than 0.05 below raw W, or perplexity < 10, do not run S2 with codebook; run **S2-arb as the main arm** and report why.

## 7. Phase 4: student arms

Run S2 (or S2-arb per §6), then S0, then S2-arb if time remains. 8000 updates each, seed 42, VAL20 student-only FG selection, per-epoch train / VAL20 / VAL18 logs, TEST50 once on the selected checkpoint, one console line per epoch including rule fractions and λ / α.

## 8. Hard constraints

1. Inspect first (§3); write `INSPECTION.md` before editing.
2. Supervised terms use K8 masks only. U provides images and detached teacher / student outputs only. Codebook fitting and τ* use TRAIN only.
3. TEST50 is never used for any decision (selection, τ_v, fallback). Decisions use VAL20 or the saturation check on U.
4. No tuning of τ, τ_v (except the §5 rule), α, λ, clip norm, or learning rates. No extra seeds.
5. Same global gradient clip for every student arm (do not change it); log the clip factor. (Deviation from rev 4 constraint 8, kept simple for a fair single-seed comparison.)
6. Export student only; inference uses S alone.
7. Commit + push after implementation / tests and after the report; report SHAs.
8. Do not modify old run directories.

## 9. Non-goals

SDF head on the student, EMA stability, per-class thresholds, calibration studies on VAL, removing or moving duplicate patients, changing teacher or student architecture, M&Ms, extra seeds, code selection / quotas from older CBKD revisions.

## 10. Report (`reports/w2s-match-20261119/README.md`)

Separate **measurements**, **interpretation**, **deviations**.

- Phase 0 findings; split / K8 / U facts.
- W: metrics, threshold trace (`tau_c` should be constant 0.95), saturation check and the τ_v used.
- CB: q_T vs raw W, perplexity, fallback decision.
- All arms at their VAL20-selected epoch: VAL20, VAL18, TEST50 and **TEST49 (TEST50 without patient118, an exact-voxel duplicate of TRAIN patient055)**: FG, RV, MYO, LV Dice, HD95; learning curves.
- Paired per-patient comparisons on TEST50: S2 vs S0, S2 vs W, S2 vs S2-arb (if run): Wilcoxon and 10,000-bootstrap 95 % CI of the mean difference. Call a difference **supported** only if the CI excludes 0 on TEST50 and the sign agrees on VAL20; otherwise **inconclusive**. State that seed variance is not estimated.
- PGR = (S2 − W) / (ceiling − W) with ceiling = 0.908 (Track A full-label student, different pipeline; label it as a reference only).
- Rule-fraction curves per class over training; watchdog events.
