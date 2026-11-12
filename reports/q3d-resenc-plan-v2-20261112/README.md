# Corrected-geometry Q3D campaign v2

This is the immutable deployment plan and a timestamped launch snapshot, not a completed campaign result. Runtime source commit: `9b40989a58a72d7058c9f139e00e8230924bf1a9`. The running server checkout stays pinned to that commit; later report-only commits do not change the process.

`PLAN.json` SHA-256: `be8cd9689584389e655007b6a6fec8f3b647dc6e61f51713f3dfe41d26bb9ec4`. The exact runtime plan is `/root/mount_data/runs/q3d-resenc-pixdim-20261112/campaign-v2/PLAN.json`. All 15 stages are READY: 3 T, 3 Q, 3 P, 6 J. Recipe paths intentionally refer to the server deployment. Earlier plans remain unchanged.

Authorized order: **V1 K8 → V1 K5 → V1 K10 → K8 V1-u → K8 C1 → K8 C0**. The K8 teacher-supervised comparison reuses the same T artifact and does not add a fourth T job. The selector proof is bound to the historical K8 reference and original source manifest hashes, revalidated before execution, and does not rewrite the corrected manifests. L5 ⊂ L8 ⊂ L10; split, masks and voxel arrays remain unchanged.

The T-only GPU profile passed 20 updates on real K8 L data, including its largest labeled volume. It measures T-a only: 0.4827517451602034 seconds/update, peak allocated 2,998,038,016 bytes (2.792 GiB), reserved 3,326,083,072 bytes (3.098 GiB), BF16 autocast with FP32 optimizer parameters on NVIDIA RTX PRO 4000 Blackwell. Temporary profile outputs were removed and the main run has independent fresh initialization. See `../q3d-resenc-pixdim-20261112/GPU_PROFILE_T.json`.

The supervisor-managed `run_ordered.py` starts production K8 T/Q/P, then profiles J for 20 updates using those genuine SHA-verified artifacts (including the largest image-only U volume), and only then starts K8 J and the remaining order above. Profile failure, OOM, tokenizer gate failure or physiology gate failure stops the queue; there is no automatic retry or threshold change. This dependency ordering is necessary because a real J profile requires completed T/Q/P. Full campaign timing is NOT_RUN: T-b, Q, P, evaluation and J were not measured by the T-a profile.

Live log window: `ssh_tmux:q3d-training`. Supervisor service: `q3d-campaign`. Combined log: `/root/mount_data/runs/q3d-resenc-pixdim-20261112/launch.log`. The log streams each optimizer update; DEV selection runs every 400 updates. `LAUNCH_RECEIPT.json` records actual optimizer updates and GPU/process observations, not just successful process creation. It is a historical snapshot, not live state.

Software evidence, with default OMP environment: `627 passed, 2 skipped in 187.07s (0:03:07)`; check_repo PASS; all-arm smoke PASS. Complete stdout is in `../q3d-resenc-pixdim-20261112/final-software/`. These checks are SOFTWARE_ONLY; the separate T profile and production optimizer records are real GPU evidence, not efficacy evidence. Tracked Git files remained clean after tests; unrelated pre-existing untracked files were preserved.

Independent read-only reviews found no blocker in external selector validation, shared-stage readiness and the declared launch order. Baseline A/A-3D sources/configs were not modified. MYO S4 winner fraction remains an explicit OPEN risk (0.825916936% on L8 GT); no coverage or weighting threshold was changed after review.

At launch, remaining stages were NOT_RUN: T completion, Q, P, J GPU profile, K8 J, K5/K10, K8 ablations and final evaluation. Do not infer their completion from this receipt. No test-set efficacy result is reported.
