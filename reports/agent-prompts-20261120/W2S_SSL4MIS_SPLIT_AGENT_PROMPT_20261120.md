# Agent prompt: move W2S-Match to the SSL4MIS ACDC split and rerun from stage S2 (2026-11-20, rev 1)

> Self-contained. Read it all before touching the repo.
> Scope decided by the user: switch every new run to the **SSL4MIS ACDC split**, keep the existing S1 (CineMA → student encoder) artifact, and **rerun the pipeline from S2 (teacher) onward**. One seed (42). Minimal arms.
> Background docs, in `reports/agent-prompts-20261119/` (this prompt wins on any conflict): `W2S_MATCH_AGENT_PROMPT_20261119.md` (method spec, §2), `CBKD_V2_AGENT_PROMPT_20261118.md` (CBKD-SA rev 4, codebook fit §4), `CBKD_SA_A4_MODAL_PROMPT_20261119.md` (Modal launcher, pipeline script pattern).

---

## 0. Context

- Repo `QuocKhanhLuong/Self-Audit`. **Base branch `claude/w2s-ssl4mis-20261120`** (this prompt's branch, created from `claude/cbkd-sa-a4-modal-20261119` @ `2101edd`). `main` (`dc102c9`) does **not** contain the W2S / CBKD-SA code. Work on this branch; record the SHA.
- Already implemented on this branch (verify with tests, do not rewrite): `self_audit/candidate_pg/w2s_arbiter.py`, `w2s_step.py`, `w2s_config.py` (arms W, S0, S2, S2arb; selection on VAL20; eval splits VAL20/VAL18/TEST50 — these must change to the SSL4MIS splits), `w2s_checks.py`, `shape_atoms.py` (CBKD-SA), `tests/test_candidate_pg_w2s.py`, `tests/test_candidate_pg_cbkd_sa*.py`, pipeline helpers `scripts/cbkd_sa_*.py`, `scripts/run_pipeline_cbkd_sa_a4.sh`, the S1/S2 import scripts from the 4070 server (`scripts/cbkd_sa_import_s1.py`, commit `2101edd`).
- Compute: two places. Write every script so the root comes from env `SELF_AUDIT_ROOT`:
  - **RTX 4070 server = primary compute**: GPU 1, tmux session `LQK`, workspace `/home/linhdang/workspace/quockhanh_workspace/SpecMamba`. ACDC raw data and the completed S1 artifact are already there.
  - Modal A100-40GB = secondary: launcher **`scripts/modal_app.py`** (`pull_drive`, `cpu`, `gpu`), volume `self-audit-mount` at `/root/mount_data`. About USD 2.5/h. Remaining credit: ask the user. Measured A100 cost (`reports/cbkd-sa-a4-20261119/BUDGET.md`): S2 teacher about 6.0 h, S3-SA about 2.1 h, one CBKD-SA student arm about 8.7 h. The full campaign does **not** fit about USD 30 on Modal; plan Modal for at most one or two stages.
- Runs that are probably still in flight on the **old 80/20/50 split**: `cbkd-sa-a4-20261119` (Modal and/or server) and `w2s-match-20261119`. Their results stay as secondary evidence only; none of them feed the new campaign.
- New campaign: `w2s-ssl4mis-20261120`. Runs go to `$SELF_AUDIT_ROOT/runs/w2s-ssl4mis-20261120/<stage>`. The report goes to `reports/w2s-ssl4mis-20261120/`.

### Why the split changes

The paper compares against BCP, ABD, M3HL, UniMatch, AD-MT and ARCO. Most of them report ACDC on the SSL4MIS split: 70 train / 10 val / 20 test, all taken from the 100 ACDC training patients, with 3 or 7 labeled patients (5 % / 10 %). Using the same split and label count lets the paper quote their published numbers.

## 1. Protocol (new; replaces 80/20/50 and K8)

| Item | Decision |
|---|---|
| Split | The exact SSL4MIS lists from `HiLab-git/SSL4MIS`, `data/ACDC/`: `train.list` / `train_slices.list`, `val.list`, `test.list`. Pin the commit hash used. |
| Labeled | **7 patients (10 %)**, the patients that the SSL4MIS `patients_to_slices(dataset="ACDC", patiens_num=7)` rule selects from the head of `train_slices.list` (136 slices). Derive the patient IDs and their ED/ES frames from the list and write them to the lock file. Do not choose them yourself. |
| Unlabeled U (main) | **ED/ES frames only** of the remaining 63 train patients, the same pool the baselines use. Full cine frames are **not** used in this campaign. |
| Selection | Best checkpoint on **val (10 patients, 20 volumes)**, student-only FG Dice. |
| Test | **test (20 patients, 40 volumes)**, evaluated **once per arm**, on the val-selected checkpoint. Never used for any decision. |
| Official TEST50 (101–150) | Not in this campaign. |
| Seed | 42 only. |

Write `configs/candidate_pg/ssl4mis_split_lock.json`. It holds: the SSL4MIS commit; the sha256 of each list file; patient IDs and volume names for L7, U63, val10 and test20; and a check that the four sets are pairwise disjoint and together cover patients 001–100.

## 2. Pipeline and arms

```
S1  CineMA → student encoder        REUSE existing artifact (verify hash; do not retrain)
S2  teacher W: MedNeXt-L, repo teacher stage (unimatch_v1, SDF head kept), τ fixed 0.95, L7 + U63(ED/ES)
    └─ saturation check on U → τ_v ∈ {0.99, 0.999}  (rule unchanged from the W2S prompt §5)
S3  CB: CBKD-SA codebook fit on frozen W (rev 4 §4), TRAIN = L7 + U63 only
S4  student arms (UNETR++-lite 16/32/64/128/256, S1 init), 8000 updates each:
      S2-main  W2S arbiter + gated CBKD-SA      (main method; target_mode=w2s_arbiter, cbkd.variant=cbkd_sa_gated)
      S0       UniMatch self-training, no teacher (target_mode=self)
      S1-kd    naive KD: UniMatch with teacher pseudo-labels, τ 0.95, no arbiter, no codebook (target_mode=teacher)
      S2-arb   arbiter only, no codebook           (optional: run only if the budget allows)
      SUP7     student supervised on L7 only        (lower bound; cheap, run if budget allows)
```

Naming note: the student arm "S2" of the old W2S prompt is called **S2-main** here, so it does not clash with pipeline stage S2 (teacher). If `target_mode = "teacher"` does not exist yet, add it: y is the teacher's clean-weak argmax and w = 1 where the teacher's max-prob ≥ 0.95, mixed by the same CutMix boxes.

Run order on one GPU: S2 → saturation → S3 → S2-main → S0 → S1-kd → (S2-arb, SUP7). S0 and SUP7 do not depend on W; start them whenever a GPU is free.

Every other setting is unchanged from the W2S prompt §1–§2: optimizer, 8000 updates, augmentation, CutMix, FP, batch sizes, λ_u schedule, α(t), λ_cb onset 1200, global clip, guards. The resolved configs of S2-main, S0, S1-kd and S2-arb may differ **only** in `pg.target_mode` and `cbkd.*`. Prove it with a config diff.

## 3. Evaluation (two metric sets; both are required)

1. **Native metrics (ours):** 3D Dice per class (RV, MYO, LV) and the mean, HD95 in **mm** with the true voxel spacing, per volume, averaged over the 40 test volumes.
2. **SSL4MIS-compatible metrics:** first open the SSL4MIS ACDC test script (`code/test_2D_fully.py` or the equivalent at the pinned commit) and record exactly how it computes the metrics: per-volume or per-slice, resizing, the `medpy` calls, whether `voxelspacing` is passed to `hd95`, and how empty predictions are handled. Then implement `evaluation_ssl4mis_compat.py`, which applies the **same** computation to our student's predictions after resampling them to the original image grid. Unit-test it against a hand-built volume where you know the answer. If HD95 there is in voxels, label it "HD95 (voxels, SSL4MIS protocol)" everywhere.

Also report the student's #params, FLOPs for one fixed 3D input, and inference latency (GPU, and CPU if available).

## 4. Phase 0: inspection, no training (`reports/w2s-ssl4mis-20261120/INSPECTION.md`)

1. Check out `claude/w2s-ssl4mis-20261120`. List what of the W2S prompt is already implemented and tested: `w2s_arbiter.py`, `target_mode` values, `cbkd_sa_gated`, watchdogs, tests 1–10. Implement only what is missing; do not rewrite working code.
2. List every running job on the old split: `modal app list`, and `tmux ls` plus `nvidia-smi` on the server. **Report them and ask the user** whether to stop them before launching anything new. Do not stop them on your own.
3. Fetch the SSL4MIS lists at a pinned commit and build the lock (§1). Map each volume name (`patientXXX_frameYY`) to our raw NIfTI files. Stop and report if any name does not map.
4. **S1 leakage check:** find which images the S1 CineMA distillation used (its config or receipt). If S1 touched images of val10 or test20 patients (image-only, no labels), do not retrain. Record the overlap per patient in the report and flag it as a disclosed, label-free exposure.
5. Re-prepare data for L7 / U63 / val10 / test20 with the repo's own preparation code (same preprocessing as `pg-k8`). Write new manifest hashes into the new configs.
6. Duplicate check: exact-voxel duplicates between train (L7 ∪ U63) and val10 ∪ test20. Report any; do not move patients.
7. Inspect the SSL4MIS evaluation code (§3) and record its exact behaviour.

Stop and report if the lists cannot be fetched, a volume does not map, or the sets overlap.

## 5. Phase 1: implementation and tests (CPU)

- Split lock, data preparation for the new sets, `target_mode = "teacher"` if missing, `evaluation_ssl4mis_compat.py`.
- A pipeline script `scripts/run_pipeline_w2s_ssl4mis.sh`. It reads `SELF_AUDIT_ROOT` and `MAX_HOURS`, writes a `DONE.json` marker per stage (config hash + artifact hashes), skips finished stages, resumes an unfinished stage from its last checkpoint (checkpoint every 500 updates), and has a budget guard (stops with `ABORTED_BUDGET` at 1.3× the projected stage time).
- Tests (pytest, `tmp_path`), all passing before any GPU run:
  1. The lock reproduces the SSL4MIS patient sets; disjointness; L7 = what `patients_to_slices` gives.
  2. The U sampler yields only ED/ES frames of U63; no val or test patient is ever loaded during training.
  3. The SSL4MIS-compatible evaluator matches hand-computed values (Dice and HD95, both the voxel and mm variants).
  4. `target_mode = "teacher"` builds the packet from the teacher only, with fixed 0.95.
  5. Config diff of the student arms shows only the intended fields.
  6. Existing suites pass unchanged; the old split configs still load.
- Commit, push, record the SHA.

## 6. Phase 2: timing probe and budget

- Run 200 updates of each stage type (S2, S3, one student arm) and extrapolate (×1.15 for evaluation and I/O).
- Write `BUDGET.md`: the cost rate, projected hours per stage, the total, and the credit left.
- Measure s/update on the 4070 too; it is the primary GPU.
- Required: S2, S3, S2-main, S0, S1-kd. Optional (only if remaining credit ≥ 1.3× their cost): S2-arb, SUP7. Splitting stages across Modal and the 4070 server is fine (for example, S0 and SUP7 on the server). Record which runs where.
- If the required stages do not fit, **stop and report options**. Never shorten updates or drop a required arm silently.

## 7. Phase 3: launch

- Launch the pipeline detached: the server's tmux window `LQK` (primary), or Modal `modal run --detach scripts/modal_app.py::gpu --cmd "..."`. Write `RUNNING.md` with the app IDs or tmux windows and the monitoring commands.
- End the session once the first epoch line of S2 is visible. Do not poll for hours.

## 8. Phase 4 (later session, when the user asks): report (`reports/w2s-ssl4mis-20261120/README.md`)

Keep **measurements**, **interpretation** and **deviations** in separate sections.

- Phase 0 facts: SSL4MIS commit, L7 IDs, S1 overlap, duplicates, how the SSL4MIS evaluator behaves.
- W: val and test metrics, τ trace, the saturation check and the τ_v used. CB: q_T vs raw W, perplexity, the fallback decision (W2S prompt §6 rule).
- All arms at their val-selected checkpoint: val10 and test20, both metric sets, per class, plus #params, FLOPs and latency.
- Paired per-patient comparisons on test20: S2-main vs S0, S2-main vs S1-kd, S2-main vs W. Use Wilcoxon and a 10,000-sample bootstrap 95 % CI. Call a difference "supported" only if the CI excludes 0 and val has the same sign. State that n = 20 patients and that seed variance is not estimated.
- PGR = (S2-main − W) / (ceiling − W). There is no ceiling run in this campaign; leave PGR blank or mark the reference used.
- A draft row block for the paper's main table (our arms only, SSL4MIS-compatible metrics). Leave baseline numbers empty: the author fills them from the papers.

## 9. Hard constraints

1. Inspect first; write `INSPECTION.md` before editing code.
2. Supervised terms use L7 masks only. U63 gives images and detached model outputs only. Codebook fitting and τ* use L7 ∪ U63 only.
3. val10 is the only selection signal; test20 is opened once per arm. Neither is used for τ_v, fallback or any other decision.
4. No tuning of τ, τ_v (except the saturation rule), α, λ, learning rate, clip norm or update counts. No extra seeds.
5. Export the student only; cold-load test; no teacher, codebook, `P_S` or `D_A` keys in the export.
6. Never write to raw data, old run directories or the old bundle. Never stop running jobs or delete volume data without the user's OK.
7. Commit and push after Phase 1 and after the report; report the SHAs.

## 10. Non-goals

Official TEST50; full cine frames as U (a later ablation); M&Ms; re-running published baselines (ARCO included; a separate prompt); 3-labeled (5 %) setting; extra seeds; architecture changes; retraining S1; calibration studies.
