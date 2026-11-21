# Agent prompt: M3HL as the teacher for W2S-Match + CBKD-SA (SSL4MIS split) (2026-11-21, rev 1)

> Self-contained. Read it all before touching anything.
> User decision (2026-11-21, after the advisor suggested a SOTA semi-supervised teacher): the teacher is **M3HL** (MICCAI 2025, "Mutual Mask Mix with High-Low Level Feature Consistency", Liu et al., code `PHPJava666/M3HL`). One seed. Minimal arms.
> This prompt **extends** `reports/agent-prompts-20261120/W2S_SSL4MIS_SPLIT_AGENT_PROMPT_20261120.md` (split, metrics, pipeline mechanics, budget rules, hard constraints). Where the two prompts conflict, this one wins.

---

## 0. Context

- Repo `QuocKhanhLuong/Self-Audit`, branch **`claude/w2s-ssl4mis-20261120`**. It already contains W2S-Match (`self_audit/candidate_pg/w2s_*.py`), CBKD-SA (`shape_atoms.py`) and the SSL4MIS prompt. Pull it and record the SHA.
- Compute: the RTX 4070 server is primary (GPU 1, tmux `LQK`, workspace `/home/linhdang/workspace/quockhanh_workspace/SpecMamba`). Modal A100 (`scripts/modal_app.py`) is secondary and budget-limited. Every script reads its root from `SELF_AUDIT_ROOT`.
- New campaign: `m3hl-teacher-20261121`. Runs go to `$SELF_AUDIT_ROOT/runs/m3hl-teacher-20261121/<stage>`. The report goes to `reports/m3hl-teacher-20261121/`.
- Already on this branch from the SSL4MIS campaign (commits `cfdea1b`..`3e9d6c5`; reuse, do not rewrite): `configs/candidate_pg/ssl4mis_split_lock.json` (SSL4MIS commit `06df6047`; L7 = patient021, 038, 049, 050, 058, 099, 100; U63; val10; test20), `configs/candidate_pg/ssl4mis_lists/`, `scripts/ssl4mis_prepare_acdc.py`, `self_audit/candidate_pg/ssl4mis_data.py`, `evaluation_ssl4mis_compat.py`, `target_mode = "teacher"`, `scripts/w2s_saturation.py`, `scripts/w2s_ssl4mis_pipeline.py`, and `reports/w2s-ssl4mis-20261120/INSPECTION.md` (read it: S1 leakage, duplicates, SSL4MIS code reading, min-max normalization per 3D volume).
- **Priority (user decision 2026-11-21):** M3HL is now the main teacher. If the MedNeXt-L "W-weak" stage of the SSL4MIS campaign has **not started**, do not start it; it is deferred (it may run later as the weak-teacher regime). If it is already running or finished, do not stop or delete it. S0 (student UniMatch, no teacher) is shared by both campaigns: reuse it if it is finished on the SSL4MIS split, otherwise run it here.

### Facts about M3HL (read from the repo at commit `19151b3`; verify them, do not trust blindly)

- 2D U-Net (`networks/unet.py`, `UNet_HL`, built by `M3HL_net(in_chns=1, class_num=4)`), plus an EMA copy during training.
- `code/ACDC_M3HL_train.py`: 30,000 iterations, batch 24 (12 labeled), lr 0.01, patch 256×256, seed 1337, `--labelnum 7`. `patients_to_slices` maps 7 patients to the first 136 slices of `train_slices.list`. A single `self_train` stage, no separate pre-train stage. The best checkpoint is chosen by val mean Dice of `model`, not of `ema_model`, and saved as `unet_best_model.pth`.
- Data: the SSL4MIS h5 format, `data/ACDC/{train_slices.list, val.list, test.list}`, `data/slices/*.h5`, `data/*.h5`. **The repo ships neither the lists nor the data.**
- `code/test_ACDC.py`: slice-wise; each slice is resized to 256×256 with `zoom(order=0)`; the network output (`out_main[0]` if it is a tuple) goes through argmax and is resized back. `calculate_metric_percase` uses `medpy` `dc` and `hd95` **without voxel spacing**, so HD95 is in voxels, per volume.
- Published ACDC 7-labeled mean Dice: 90.47 (2D, SSL4MIS 70/10/20).

## 1. Goal and arms

Train M3HL as the teacher with its official recipe on the exact SSL4MIS split. Then train the 3D student (UNETR++-lite 16/32/64/128/256, S1 CineMA init) with W2S-Match targets from the M3HL teacher, and with CBKD-SA rebuilt to fit **from teacher outputs** (§3.3). An M3HL U-Net has no MedNeXt-like L3 feature on the student grid and has no SDF head.

| Arm | What | Depends on | Required |
|---|---|---|---|
| **T-M3HL** | Reproduce M3HL with the official code on the SSL4MIS lists, 7 labeled. Frozen afterwards. | split | yes |
| **S1-kd-m3hl** | Student, naive KD: UniMatch with M3HL pseudo-labels (fixed 0.95), no arbiter, no codebook | T-M3HL, S1 | yes |
| **S2-m3hl-arb** | Student, W2S arbiter with M3HL as teacher, no codebook | T-M3HL, S1 | yes |
| **CB-out** | Output-space shape-atom codebook fitted on frozen M3HL outputs (§3.3) | T-M3HL | yes, if the budget allows S2-m3hl |
| **S2-m3hl** | Student, arbiter + gated CBKD-SA (`cbkd_sa_out`) | T-M3HL, CB-out, S1 | yes, if the budget allows; otherwise report |
| S0 | Student UniMatch, no teacher | S1 | reuse if finished, else run |

Run order on one GPU: T-M3HL → saturation check → S2-m3hl-arb → S1-kd-m3hl → CB-out → S2-m3hl → (S0 if needed). T-M3HL is a light 2D job; if the 4070 is busy it may run on Modal.

The resolved student configs may differ only in `pg.target_mode`, `pg.teacher_source` and `cbkd.*`. Prove this with a config diff.

## 2. Phase 0: inspection, no training (`reports/m3hl-teacher-20261121/INSPECTION.md`)

1. Clone `PHPJava666/M3HL` into `external/M3HL` at pinned commit `19151b3`, the same way `external/MedNeXt` is pinned. Do not vendor its files into `self_audit/`. Record its license.
2. Confirm the facts in §0 from the code, line by line. Report any difference.
3. **SSL4MIS data in h5 form.** The lists and lock already exist (SSL4MIS commit `06df6047`). Build the h5 files from our raw ACDC with SSL4MIS's own preprocessing script (`code/dataloaders/acdc_data_processing.py` or its equivalent at that commit). Do not download third-party preprocessed h5 archives.
   - Check: the patient sets of `train_slices.list` / `val.list` / `test.list` equal the lock; the first 136 slices of `train_slices.list` belong exactly to the 7 L7 patients.
4. **Intensity contract.** Record how SSL4MIS preprocessing normalizes intensity, and how our 3D student pipeline does. The teacher must get its own normalization at inference (§3.1).
5. List running jobs (`tmux ls`, `nvidia-smi`, `modal app list`). Report them and ask before stopping anything.

Stop and report if the h5 build cannot reproduce the lists' slice counts, or if L7 ≠ the first 136 slices.

## 3. Phase 1: implementation and tests (CPU)

### 3.1 Teacher wrapper (`self_audit/candidate_pg/teacher_m3hl.py`)

- Load the frozen M3HL `model` checkpoint selected by val (`unet_best_model.pth`). Use `eval()` and `requires_grad_(False)`.
- `forward_3d(volume, geometry)` takes the student's weak view `[B,1,D,H,W]` in our normalization and returns class probabilities `[B,4,D,H,W]` on the same grid. Do the same as the official test path: per slice, undo our normalization and apply M3HL's; resize to 256×256; run the network (take `out[0]` if the output is a tuple); softmax; resize back with **bilinear on probabilities** (deviation from `order=0` on labels, needed because the arbiter and KD need probabilities; record it); renormalize.
- It must work online inside the student step, on the clean weak view, so CutMix mixing and the arbiter work exactly as with the MedNeXt teacher. It returns no features.
- Add `pg.teacher_source ∈ {"mednext", "m3hl"}`. The existing MedNeXt path must stay bit-identical.

### 3.2 Student targets

- `target_mode = "teacher"` (S1-kd-m3hl) and `"w2s_arbiter"` (S2-m3hl-arb, S2-m3hl) consume p_T from the wrapper. τ = 0.95. α schedule and stability rule unchanged.
- τ_v: run the saturation check of the W2S prompt §5 on the M3HL teacher (fraction of FG voxels with c_T ≥ 0.99 on ≥ 200 U volumes). If it is > 0.90, use τ_v = 0.999; otherwise 0.99. Record the result. No other tuning.

### 3.3 CBKD-SA from teacher outputs (`cbkd.variant = "cbkd_sa_out"`)

The atom decoder D_A, codebook C (128 × 64, cosine), margin m_T, τ*, the gated L_pm and L_atom stay exactly as in `shape_atoms.py`. Only the **teacher-side encoder** changes:

- Teacher target map per frame: M3HL probabilities (4 channels) + an SDF (3 channels: RV, MYO, LV), computed from the M3HL argmax with the repo's existing SDF utility and normalized the same way as the MedNeXt SDF. Use a valid mask.
- `E_T`: a small conv encoder that maps the 7-channel map to one 64-d vector per student stage-3 cell (kernel = stride = cell size (1,8,8), then 1×1×1 layers). This replaces `P_T` applied to teacher features. Everything after `E_T` is unchanged.
- Fit on TRAIN only (L7 GT maps, and M3HL maps on U63), 4000 updates. Dead-code restart, usage entropy, τ* = median(d2 − d1) as in rev 4. Freeze.
- Report on val10: Dice of q_T (code-reconstructed) vs raw M3HL; perplexity; dead codes; m_T at boundary cells vs interior cells.
- **Fallback rule (unchanged):** if q_T FG Dice on val is more than 0.05 below raw M3HL, or perplexity < 10, skip S2-m3hl and report it.
- Student side unchanged: `P_S` from student stage 3, hard VQ into C (STE), frozen D_A, L_atom on L7, and L_pm on U gated by arbiter rules {1, 2, 4} and weighted by m_T.

### 3.4 Tests (pytest, `tmp_path`); all must pass before any GPU run

1. The M3HL wrapper on a toy U-Net: per-slice resize round trip keeps shape and geometry; output sums to 1 over classes; no gradient; normalization conversion matches a hand-computed value.
2. `teacher_source = "mednext"` gives outputs bit-identical to before on a fixed toy batch.
3. `cbkd_sa_out`: E_T output shape per cell; codebook fit loss finite; L_pm gradient reaches only student parameters; the scale test and the perplexity > 1 at init test from rev 4 pass for the new encoder.
4. The h5 builder reproduces slice counts and the L7 = first-136-slices check on a toy list.
5. Config diff of the student arms shows only the allowed fields.
6. All existing suites pass.

Commit and push; record the SHA.

## 4. Phase 2: teacher, probe, budget

- **T-M3HL:** run `external/M3HL/code/ACDC_M3HL_train.py` on our h5 data with the **official defaults** (30k iterations, batch 24, labeled 12, lr 0.01, seed 1337, `--labelnum 7`). Keep the official seed 1337 for fidelity to the published recipe; it is a deviation from seed 42, so record it. Then evaluate with `external/M3HL/code/test_ACDC.py` (official metric: per-volume Dice and HD95 in voxels) and with our native 3D metric (HD95 in mm) on val10 and test20.
- **Reproduction check:** compare the test20 mean Dice with the published 90.47. Report the gap. Continue whatever the gap is (do not tune M3HL). If the gap is larger than 3 points, flag it at the top of the report.
- Timing probe: 200 updates of one student arm with the online M3HL wrapper, and of CB-out. Update `BUDGET.md`. If the required arms do not fit, stop and report options; never shorten training silently.

## 5. Phase 3: launch

Launch detached on the 4070 (tmux window in `LQK`) through a resumable pipeline script, `scripts/run_pipeline_m3hl_teacher.sh`: a `DONE.json` marker per stage, resume from checkpoints every 500 updates, and a budget guard. Write `RUNNING.md`. Stop the session once the first epoch line of the first student arm is visible.

## 6. Phase 4 (later, when the user asks): report (`reports/m3hl-teacher-20261121/README.md`)

Keep measurements, interpretation and deviations in separate sections.

- Phase 0 facts, the M3HL commit, and how the h5 build was verified.
- T-M3HL: official-metric and native-metric results on val10 and test20, and the gap to the published 90.47.
- Saturation check and the τ_v used. CB-out reconstruction report and the fallback decision.
- All arms (plus S0 and, if available, the MedNeXt-teacher arms of the SSL4MIS campaign) on val10 and test20: per-class Dice, HD95 in mm and in voxels (official), #params, FLOPs, latency.
- **Regime analysis:** arbiter rule fractions over training for the M3HL teacher (and for the MedNeXt teacher if those runs exist); override share on FG disagreement voxels. Offline H-W1 on val10: P(student correct | disagree, stable, confident) vs P(M3HL correct | disagree).
- Paired per-patient tests on test20: S2-m3hl-arb vs S1-kd-m3hl, S2-m3hl vs S2-m3hl-arb, best student vs T-M3HL. Use Wilcoxon and a 10,000-sample bootstrap 95 % CI. Call a difference "supported" only if the CI excludes 0 and val has the same sign. State that n = 20 and that seed variance is not estimated.
- PGR of the best student relative to T-M3HL. Use the full-label ceiling only if such a run exists; otherwise mark PGR as unavailable.

## 7. Hard constraints

1. Inspect first; write `INSPECTION.md` before editing code.
2. Supervised terms use L7 masks only. U63 gives images and detached teacher outputs only. CB-out and τ* use L7 ∪ U63 only.
3. val10 is the only selection signal (M3HL official selection included); test20 is opened once per arm and never used for any decision.
4. M3HL runs with official defaults; no tuning of M3HL or of τ, τ_v (except the saturation rule), α, λ, learning rates, clip norms or update counts.
5. Export the student only. The cold-load test must show no M3HL, codebook, E_T, `P_S` or `D_A` keys in the export.
6. Never modify raw data, other campaigns' run directories or old bundles. Never stop jobs or delete data without the user's OK. Never commit weights or h5 data.
7. Commit and push after Phase 1 and after the report; report the SHAs.

## 8. Non-goals

Porting M3HL to 3D or to a larger backbone; re-running the other published baselines; tuning M3HL; full cine frames as U; official TEST50; 3-labeled setting; extra seeds; changing the student architecture; retraining S1.
