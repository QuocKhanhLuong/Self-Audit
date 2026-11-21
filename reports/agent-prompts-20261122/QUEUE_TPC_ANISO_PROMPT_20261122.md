# Agent prompt: re-order the GPU queue, and do the CPU-side work for TPC-Gate and AnisoStudent (2026-11-22, rev 2: fixed 150 epochs)

> Self-contained. Read it all before acting. Machine: the vast RTX 5000 Ada 32 GB (tmux session `w2s`). Repo branch `claude/w2s-ssl4mis-20261120`; pull and record the SHA.
> Background (read first): `reports/m3hl-teacher-20261121/RUNNING.md`, `reports/m3hl-teacher-20261121/INSPECTION.md`, `reports/w2s-ssl4mis-20261120/INSPECTION.md`, `reports/agent-prompts-20261121/M3HL_TEACHER_AGENT_PROMPT_20261121.md`.
> User decisions (2026-11-22, rev 2):
> - **Every student training from now on runs a fixed 150 epochs** (at the current 80 updates/epoch, that is 12,000 updates; derive the update count from the config's `updates_per_epoch`, do not hard-code it).
> - **Restart A4-SA at 150 epochs.** The LR schedule is cosine over the total updates, so the running 100-epoch job cannot be extended.
> - The arms already finished at 100 epochs (S0, S1kd, S2-m3hl-arb) are **kept, not rerun**. Every report must label each arm with its epoch budget and state that 100- and 150-epoch arms are not directly comparable.
> - After A4-SA (150), look at it, then move to the new student. CPU-side work must not compete with the GPU job.

---

## 0. State at the time of writing (verify each number from the run artifacts; do not trust this table)

| Item | test20 Dice | Note |
|---|---|---|
| W (MedNeXt-L teacher) | 0.860 | val10 0.836 |
| S0 (student UniMatch, no teacher) | 0.871 | student > W, the weak-teacher regime holds |
| S1kd (naive KD from W) | 0.871 | no gain over S0 |
| S2-main (W2S + CBKD-SA, W teacher) | dropped | CB fallback: code-reconstructed W is 0.089 Dice below raw W |
| A4-SA (PG + CBKD-SA, CB from W) | running, about epoch 22/100 | uses the **same CB that failed the fallback** |
| T-M3HL (official eval) | 0.8847 | val10 0.8885 |
| S2-m3hl-arb (W2S arbiter, M3HL teacher) | 0.8877 | +0.3 over T-M3HL, not yet tested for significance or metric parity |
| S1-kd-m3hl, CB-out, S2-m3hl | queued | |

Two student trainings at once caused OOM, so student stages run one at a time. Keep it that way.

## 1. Part Q: GPU queue (150-epoch rule)

### 1.1 Global epoch change

- Add a single config field, `train.epochs = 150`, that sets the total student updates to `150 × updates_per_epoch`, for every student arm built from now on (A4SA, S1kd-m3hl, S2-m3hl, S0-unetrpp-150, S0-aniso). The configs of finished arms stay untouched.
- Schedules defined as a **fraction** of training (cosine LR, the α(t) ramp at 20 % → 50 %) scale automatically. Onsets defined in **absolute updates** (labeled-only warm-up / λ_u onset at 800, CBKD onset at 1200 with a 400-update ramp, α onset) stay at their absolute values. Do not rescale them. List every schedule field and whether it is relative or absolute in the run report.
- Selection is still the val10-best epoch, now over 150 epochs. Also evaluate and report the **final (epoch 150) checkpoint** on val10 and test20.
- Teacher and codebook stages (W, T-M3HL, CB, M3CB, S1, S1-ANISO) are not student trainings: their budgets do not change.
- Run a config-diff test: a new 150-epoch config differs from its 100-epoch counterpart only in the epoch/update fields.

### 1.2 Restart A4-SA

1. Stop the running A4-SA cleanly: let it write its current checkpoint, then end the process.
2. Rename its run directory to `A4SA_e100_partial` and keep every log and CSV. Do not delete anything; do not resume from it.
3. Start A4-SA from scratch at 150 epochs. Use the same CB artifact, the same W, the same S1 init and the same seed 42.

### 1.3 Queue order (one student training at a time)

1. `A4SA` (150) → `A4SAFINAL`.
2. `S1KDM3 S1KDM3FINAL` at 150 (required: separates the arbiter effect from simply having the M3HL teacher). Note: its counterpart S2-m3hl-arb ran at 100 epochs, so this pair has unequal schedules. Report it, and if time remains at the end of the queue, add an optional `S2ARBM3-150` rerun.
3. `M3CB`, then `S2M3 S2M3FINAL` at 150 only if CB-out passes the existing fallback rule. Record the CB-out reconstruction numbers either way.
4. `S1-ANISO`, then the two-arm architecture screening at 150 epochs (§4.3).

Rough GPU cost: each 150-epoch student is about 1.5× the old 100-epoch runtime. Write the projected end time of every queue item into RUNNING.md.

Implement the queue as a small resumable script (or extend the existing pipelines' `--steps`), so that each item starts automatically when the previous one has finished. Write the order and the commands to `reports/m3hl-teacher-20261121/RUNNING.md`. Ask the user only if a step fails twice.

## 2. Part C1: metric parity and significance of S2-m3hl-arb vs T-M3HL (CPU, now)

- Evaluate T-M3HL predictions and the S2-m3hl-arb checkpoint (its val10-selected one) on test20 with **the same evaluators**: (a) the SSL4MIS-compatible evaluator (voxel HD95, per volume, medpy, as in `test_ACDC.py`); (b) our native 3D metric (HD95 in mm). For T-M3HL, use its slice-wise predictions resampled to the native grid exactly as the student is evaluated.
- Per patient (20 patients; ED and ES averaged per patient), compute the paired difference in mean Dice and HD95. Report Wilcoxon signed-rank and a 10,000-sample bootstrap 95 % CI. Also report the val10 sign.
- Verdict: "supported" only if the CI excludes 0 and val10 has the same sign; otherwise "inconclusive".
- Output: `reports/m3hl-teacher-20261121/PARITY_S2ARB_VS_TM3HL.md`.

## 3. Part C2: offline TPC-Gate analysis on the frozen M3HL teacher (CPU, or GPU only if ≥ 6 GB free; inference only, `no_grad`, batch 1)

Goal: decide **before any new training** whether through-plane coherence of a slice-wise teacher localizes its errors better than its confidence. All thresholds and kill criteria below are pre-registered: do not change them after seeing results.

### 3.1 Data use

- Teacher predictions (softmax) for val10 (20 volumes) and for U63 (126 ED/ES volumes).
- val10 is the selection set (choose the quantile q of §3.3 here only).
- U63 ground truth exists in ACDC. Use it **only as a diagnostic** to get stable statistics (126 volumes). Never use it for selection or training. State this in the report.
- test20 is not used in Part C2.

### 3.2 Signals per voxel (teacher argmax ŷ_T, max-prob c_T)

- **Alignment.** Per volume, compute per-slice centroids of the predicted foreground (generic) and of the LV blood pool (cardiac), fit a quadratic in z to each, and translate slices onto the fitted path. Compute every signal twice, once with generic alignment and once with LV alignment.
- **S_flip(v):** fraction of the adjacent slices (z ± 1 after alignment) whose label at the same in-plane position differs from ŷ_T(v). Report it overall and restricted to the ±2 px boundary band.
- **S_area(z, c):** |A_c(z) − mean(A_c(z−1), A_c(z+1))| / max(that mean, ε), where A_c(z) is the area of class c in slice z; at the end slices use the one available neighbour. Assign it to every voxel of class c in slice z.
- **S_topo(z) ∈ {0,1}.** Generic version: more than one connected component per class, or holes in a class. Cardiac version: generic plus "MYO ring not closed around LV" and "RV touching LV without MYO in between".
- **S_tta(v):** vote disagreement over 8 TTA variants (flips, rot90, gamma). Secondary signal.
- **TPC score:** r(v) = max(within-volume rank of S_flip(v), within-volume rank of S_area(z, c(v)), S_topo(z)). Compute a generic TPC and a cardiac TPC.
- **Baselines:** 1 − c_T, entropy, margin, and slice position |z − z_mid| normalized to [0, 1].

### 3.3 Experiments and pre-registered kill criteria

| ID | Measurement | Kill criterion |
|---|---|---|
| E0 | Error map of T-M3HL: error rate and Dice by relative slice position (base / mid / apex terciles) and by class; fraction of erroneous voxels with c_T ≥ 0.99 | TPC-Gate has little to work with if < 30 % of the errors lie at c_T ≥ 0.99 **and** errors are not concentrated by slice position |
| E1 | AUROC and AUPRC for "teacher voxel is wrong", for every signal, the TPC scores and the baselines. Report overall, in the boundary band, and inside the c_T ≥ 0.99 subset. Patient-level bootstrap CI | Kill if ΔAUROC(best TPC − best baseline) < 0.05, or AUROC inside c_T ≥ 0.99 < 0.65 |
| E2 | Oracle headroom: choose q ∈ {0.80, 0.90, 0.95} on val10; replace the teacher label by GT at flagged voxels; compute pseudo-label Dice and HD95 (mm) | Kill if gain < 0.5 Dice **and** < 0.5 mm HD95 |
| E3 | Student feasibility, with **existing** checkpoints S0 and S2-m3hl-arb: at flagged voxels where student and teacher disagree, the fraction where the student is right (val10 and U63) | Kill override if < 55 % (then TPC can still be used as drop-only) |

Output: `reports/m3hl-teacher-20261121/TPC_OFFLINE.md` (tables + one figure per experiment), and the code in `scripts/tpc_offline.py` and `self_audit/candidate_pg/tpc_signals.py`, with unit tests on toy volumes (known answers for S_flip, S_area, S_topo and the alignment). End with a GO / NO-GO line per experiment.

## 4. Part C3: AnisoStudent (CPU now, GPU after the queue reaches it)

### 4.1 Architecture (`self_audit/candidate_pg/aniso_student.py`)

Same input and output contract as the current `PGStudent` (native volumes, 4 classes, geometry), so every existing training path works by switching `student.arch = "aniso"`. No z down-sampling anywhere.

| Stage | Resolution | Blocks | Channels |
|---|---|---|---|
| E0 | H | 2 × conv 1×3×3 | 1→24→24 |
| E1 | H/2 | 2 × conv 1×3×3 | 24→48→48 |
| E2 | H/4 | 2 × conv 1×3×3 | 48→96→96 |
| E3 | H/8 | conv 1×3×3, conv 3×1×1 (first z mixing), conv 1×3×3 | 96→160 |
| E4 | H/16 | 2 × conv 1×3×3, then a Z-Transformer: 2 layers of self-attention along z only for every (h, w), dim 192, 4 heads, MLP ratio 2, relative slice-position encoding | 160→192 |
| Decoder | | 4 stages: transposed conv 1×2×2 + conv 1×3×3 after skip concat; deep supervision at D2, D3 | mirror |
| Heads | | seg (4) 1×1×1; SDF (3, tanh, train-only) | |

Norm/activation: as in `PGStudent`. Expose the stage-3 feature for the code branch, so CBKD variants still work.

### 4.2 Tests and measurements (CPU)

- Parameter count ≤ 4.0 M (print the per-stage breakdown) and FLOPs for a 16 × 224 × 224 input (fvcore or ptflops). Write both into the report.
- Shape tests for D ∈ {6, 10, 18} with padding; the Z-Transformer is masked on padded slices; gradient flows to all stages; export/cold-load test; `student.arch = "unetrpp_lite"` stays bit-identical.
- **S1-ANISO config:** the same CineMA → student distillation as the existing S1, but distilling only into E0–E2 (in-plane stages, matching CineMA's per-slice processing), with the same S1 budget. It is a GPU step in the queue (§1).

### 4.3 Screening (GPU, queued after §1 items 1–2)

- Two arms, identical except for the architecture: `S0-unetrpp-150` (UNETR++-lite, existing S1 init; a new 150-epoch run, because the finished S0 used 100 epochs and the screening needs equal schedules) and `S0-aniso` (S1-ANISO init, 150 epochs). Both use UniMatch self-training without a teacher, seed 42.
- Pre-registered decision on **val10 only**: keep AnisoStudent if its HD95 (mm) ≤ the UNETR++-lite HD95 and its Dice ≥ UNETR++-lite − 0.5. Report also an inter-slice consistency metric: mean Dice between adjacent aligned slices, compared to the GT's own value.
- Do not open test20 for the screening.
- Output: `reports/m3hl-teacher-20261121/ANISO_SCREEN.md`.

## 5. Hard constraints

1. The only job you may stop is the 100-epoch A4-SA (§1.2), and its artifacts must be kept. Never run two student trainings at once.
2. CPU-side work must not starve the GPU job: cap CPU threads (for example `OMP_NUM_THREADS=4`), and avoid heavy I/O on the run disk.
3. No decision uses test20; U63 GT is diagnostic-only; val10 is the only selection set.
4. Seed 42; no tuning beyond the pre-registered choices (q in §3.3, the screening rule in §4.3).
5. Pull/rebase before every push, never force-push; commit after each part with tests passing; report the SHAs.
6. Never delete runs or data; back up new checkpoints as the existing pipelines do.

## 6. Report back to the user

Queue order and ETA; PARITY verdict; the E0–E3 GO/NO-GO table; AnisoStudent params/FLOPs and test status; and, when available, the A4-SA, S1-kd-m3hl and CB-out results.
