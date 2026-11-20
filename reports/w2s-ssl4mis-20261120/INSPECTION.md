# INSPECTION (Phase 0): W2S-Match on the SSL4MIS ACDC split

Date 2026-11-20. Host: RTX 4070 server (read-only inspection; no training, no code edits yet). Branch claude/w2s-ssl4mis-20261120 @ 651a31c, checked out as a separate git worktree (Self-Audit-ssl4mis) so the running old-split job is not disturbed.

## 1. Already implemented on this branch (from the Modal session; verified by reading, tests not yet run here)
- w2s_arbiter.py (5 rules, alpha schedule, source-aligned stability), w2s_step.py (target_mode in fusion, self, w2s_arbiter; cbkd_sa_gated terms), w2s_config.py (arms W, S0, S2, S2arb), w2s_checks.py (saturation rule, watchdogs), shape_atoms.py, tests/test_candidate_pg_w2s.py (20 tests), cbkd_sa pipeline scripts, S1/S2 import scripts.
- Missing for this campaign: target_mode=teacher; engine support for W2S arms (engine only knows A1-A4, variant cbkd_sa, requires S2/S3 receipts for every arm); watchdogs and saturation check are not called by any runner; the data loader, evaluation and selection are hard-coded to 80/18/50 and VAL18; SSL4MIS split lock, data preparation and evaluator; pipeline script. A server-side branch (claude/server4070-20261119, 52523b4) already has engine wiring for gated CBKD-SA / S0 without teacher / watchdog and diag logging, a saturation script and S1-link verification; these will be ported, the VAL20 parts are obsolete.

## 2. Jobs still running on the old split (to be decided by the user)
- RTX 4070 server, GPU 1, tmux LQK window a4sa: A4-SA stage S2 (MedNeXt-L teacher, old 80/20/50 split), epoch 77/100, about 275 s/epoch, expected done about 16:10. Uses 11.4 GB of 12 GB. Pipeline is limited to stages S1,S2 and stops after S2.
- GPU 0 (RTX 5070 Ti) jobs belong to other users; not touched.
- Modal: modal app list shows no apps (nothing running). Remaining Modal credit unknown.
- No w2s-match-20261119 run was ever launched.

## 3. SSL4MIS split lock (HiLab-git/SSL4MIS commit 06df6047a59aba9988ced8331998b5957ecb356b, data/ACDC)
- train.list 140 volumes (70 patients x frame01/frame02), val.list 20 volumes (10 patients), test.list 40 volumes (20 patients). train_slices.list 1312 slices.
- Patient sets are pairwise disjoint and cover patients 001-100 exactly.
- L7: the first 136 lines of train_slices.list cover exactly 7 patients, each with both frames: ['patient021', 'patient038', 'patient049', 'patient050', 'patient058', 'patient099', 'patient100'].
- U63 = the other 63 train patients (ED/ES frames only = 126 volumes). val10 = ['patient002', 'patient019', 'patient028', 'patient030', 'patient039', 'patient045', 'patient078', 'patient082', 'patient085', 'patient087']. test20 = ['patient001', 'patient007', 'patient008', 'patient011', 'patient013', 'patient022', 'patient024', 'patient033', 'patient052', 'patient059', 'patient064', 'patient065', 'patient066', 'patient068', 'patient075', 'patient080', 'patient081', 'patient083', 'patient084', 'patient093'].
- List sha256 (first 16 hex): train.list 97a399f7bdbfe96b, train_slices.list 1b21b2a69d1b28b4, val.list 61b935c936d8e12f, test.list 98161bdfaad072bf.

## 4. Name mapping
- SSL4MIS names are positional: frame01 and frame02 for every patient. Raw ACDC ED is frame 1 for 99/100 patients (patient090: ED 4, ES 11); ES ranges 6-16. So frame01 -> ED and frame02 -> ES (by Info.cfg) is an assumption, not a literal file-name mapping. Needs the user to confirm.

## 5. S1 leakage check
- S1 trained on all cine frames (2148 frames) of the old TRAIN80 patients, image only. Overlap with the new sets: val10 6/10 patients seen (patient002, 019, 039, 045, 078, 085), test20 19/20 seen (all except patient065), L7 7/7 seen (expected, they are train). Label-free exposure only; S1 is not retrained, per the prompt. To be disclosed in the report.

## 6. Duplicates
- Exact-voxel duplicate groups among the 100 training patients (full 4D array hash): patient074 == patient076, both in the train set (U63). No duplicate crosses train and val10/test20.

## 7. SSL4MIS code reading (commit 06df6047, files code/test_2D_fully.py, utils/metrics.py, dataloaders/*, train_acdc_unet_semi_seg.sh)
- Data: 2D slices. Each ED/ES frame volume is min-max normalised over the whole 3D volume ((x-min)/(max-min)), then stored as one h5 per slice. train_slices.list holds 1312 slice ids (1-based slice index) over 140 volumes; I checked that the slice count of every one of the 140 volumes equals the raw ACDC depth (0 mismatches), so the lists map onto our raw volumes patient by patient.
- Labeled set: `patients_to_slices("ACDC", 7) = 136`; the labeled set is `train_slices.list[:136]`, the unlabeled set is the remaining slices (index 136 onward, 1176 slices of the other 63 patients). The first 136 lines are exactly the 14 volumes of the 7 patients, so "7 patients" is also whole volumes here.
- Names: frame01 / frame02 are positional (first and second annotated frame). In ACDC ED comes before ES for all 100 patients, so frame01 = ED and frame02 = ES whether read positionally or by Info.cfg. The earlier open point 2 is closed.
- Test script (test_2D_fully.py): per volume; every slice is resized to 256x256 with scipy zoom order=0 (both image going in and prediction coming back), argmax, then the volume is assembled. Per class (RV=1, MYO=2, LV=3) it calls medpy.metric.binary dc, asd and hd95 on the binary masks with NO voxelspacing argument, so HD95 and ASD are in voxel units on the (slice, H, W) index grid. Results are averaged over the 40 test volumes per class, then over the 3 classes. There is no handling of empty prediction or empty label (medpy raises); published code bases built on it (BCP etc.) return 0 for such cases, so the compatible evaluator must record how it treats them.
- Model selection in their training scripts uses val.list (20 volumes) with the same slice-wise 256x256 pipeline; test.list is evaluated with the best-on-val checkpoint.
- Consequence for us: our student is 3D on native volumes. The compatible metric set applies the identical medpy computation (binary per class, voxel units, no spacing, mean over classes and volumes) to our native-grid predictions; our own native metric set (mm, true spacing) stays the primary one. U sampling in this campaign is by ED/ES volumes of U63 (126 volumes), not by 2D slices.

## 8. Decisions taken by the user (2026-11-20 14:2x)
- The old-split A4-SA stage S2 on the 4070 was stopped at epoch 78/100 and its weights (S2/teacher_latest.pt, 947 MB) were deleted; logs and CSVs of that run were kept. Probe artifacts under runs/probe-a4sa were not touched.
- Open points that remain: Modal credit (not stated; plan on the 4070 only unless told otherwise).

## 8b. Former open points (superseded)
1. Stop or keep the old-split S2 job on the 4070 (it blocks the GPU for about 2.3 h more).
2. Confirm frame01 -> ED, frame02 -> ES.
3. Modal credit left, if any Modal stage is wanted.
