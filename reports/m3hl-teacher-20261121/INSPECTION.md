# M3HL teacher: Phase 0 inspection (2026-11-21)

Branch `claude/w2s-ssl4mis-20261120`, base SHA `3e32dc5`. No training was run for this campaign. See "State" at the end.

## 1. M3HL source (`external/M3HL`, commit `19151b3`, git-ignored like `external/MedNeXt`)
- **License: none.** The repo has no LICENSE file (only README.md and `code/`). Not vendored, not copied into `self_audit/`; imported from `external/` at the pinned commit only. Weights are never committed.
- Facts in prompt §0 checked in code, all confirmed:
  - 2D `UNet_HL` (feature chns 16/32/64/128/256), `M3HL_net(in_chns=1, class_num=4)`, EMA copy (`update_model_ema(.., 0.99)`).
  - Defaults: `max_iterations=30000`, `batch_size=24`, `labeled_bs=12`, `base_lr=0.01`, patch 256, `seed=1337`, `labelnum=7`; `patients_to_slices` 7 -> 136.
  - Best checkpoint = val mean Dice of `model` (not EMA), every 200 iterations, saved as `unet_best_model.pth` (state dict).
  - Test: per slice `zoom(order=0)` to 256, `out_main[0]` if tuple, argmax, zoom back; medpy `dc/hd95/asd`, no voxel spacing; case dropped to 0 when prediction has no voxel of the class.
- **Differences / traps found (not in the prompt):**
  1. `--gpu` default is `'6'` and the script sets `CUDA_VISIBLE_DEVICES=args.gpu` itself. On the 4070 pass `--gpu 1` (with `CUDA_DEVICE_ORDER=PCI_BUS_ID`), otherwise it fails or lands on another user's GPU.
  2. `net_factory`/`M3HL_net` call `.cuda()` unconditionally: GPU only; the wrapper imports `UNet_HL` directly and runs on CPU too.
  3. `ulab_a/ulab_b` (labels of unlabeled slices) are read, but only used in `unl_label`/`l_label`, which feed TensorBoard images; the loss uses `plab_*` and labeled labels only. So the official recipe never trains on U ground truth.
  4. Unused imports (`cv2`, `imageio`, `asyncore`, `matplotlib`) need installing/old python: `asyncore` is removed in Python 3.12, use <=3.11 on the server.
  5. Test script writes predictions under `./model/M3HL/...` relative to cwd and `shutil.rmtree`s them.

## 2. SSL4MIS h5 data (built from our raw ACDC, `scripts/m3hl_build_h5.py`)
- Source: raw per-frame NIfTI (`patientNNN_frameYY.nii` + `_gt.nii`), ED/ES resolved via `Info.cfg`; `frame01` = ED, `frame02` = ES. Preprocessing = SSL4MIS `acdc_data_processing.py` @06df6047: SimpleITK array (z,y,x), per-3D-volume min-max to [0,1].
- Built locally on the Mac from the raw volume copy, not downloaded from third parties: 140 train + 20 val + 40 test volumes, 1312 train slice files; `H5_RECEIPT.json` status READY.
- Checks (all pass, also as pytest `test_real_lists_pass_lock_check`): patient sets of the lists equal the lock; `train_slices.list` = 1312 unique slices covering exactly `train.list`; **first 136 slices belong exactly to the 7 L7 patients** (021, 038, 049, 050, 058, 099, 100) and L7 has exactly 136 slices.
- Found: the pinned `train_slices.list` uses **1-based** slice indices (`_slice_1` ... `_slice_N`). The builder writes slice file `i` = `volume[i-1]` accordingly (a 0-based build mismatched the list in 140/140 cases, caught by the receipt check).
- U63 label arrays are written as zeros (point 3 above makes this safe and structural).
- 14 frames with real labels in train (L7 x 2), 20 val, 40 test; the h5 tree is not committed.

## 3. Intensity contract
| | normalization |
|---|---|
| SSL4MIS / M3HL | per 3D volume (one frame), min-max to [0,1] on raw intensity |
| our 3D student pipeline | global cine percentiles 0.5/99.5 over the whole 4D cine, clip to [0,1] |
- The student's views are percentile-clipped, so the wrapper (`teacher_m3hl.py`) applies per-volume min-max to the view. This equals the raw min-max up to the clipped outliers (about 0.5% of voxels at each end); it is a **deviation** and cannot be removed without storing raw frame extrema. To quantify it, after T-M3HL is trained compare val10 Dice from the official h5 input vs from our normalized input via the wrapper.
- Other recorded deviation: bilinear resize of probabilities back to native size (official code resizes label maps with order 0).

## 4. Running jobs
- 4070 server: **unreachable** (SSH to 100.64.167.8:22 times out; Tailscale on the Mac is offline). tmux/nvidia-smi not inspectable.
- Modal: nothing running. W (MedNeXt teacher) finished earlier (val10 0.836, test20 0.860); S0 / S1kd / CB were killed at epoch 70 / 62 / update 2000 when the workspace hit its spend limit (`Workspace ... has exceeded its spend limit`). No job was stopped by me for this campaign. Their checkpoints are backed up on the Mac (`~/w2s-ssl4mis-backup-20261121`).

## 5. State
Phase 0: done. Phase 1: wrapper + h5 builder + tests done (7 tests); `cbkd_sa_out`, `pg.teacher_source` config/engine wiring, the config diff and the pipeline script are **not done**. Phase 2/3: **blocked**, no GPU (Modal spend limit, 4070 unreachable).
