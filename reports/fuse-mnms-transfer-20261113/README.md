# M&Ms transfer B preparation — 2026-11-13

Status at 2026-11-13 15:42 UTC: software integration accepted after one independent Claude review; **Track B queue installed and WAITING for Track A**. `LAUNCH_RECEIPT.json` verifies the production checkout, queue, services and visible log window. Target patient/GPU training and TEST are **NOT_RUN**.

The predecessor is the running ACDC semi K8 experiment A. Its pinned checkout, recipe, manifests and processes are unchanged. `TRACK_A_SNAPSHOT.json` is a time-stamped observation, not a final result.

## Locked plan

| Stage | Budget | Labels and selection |
|---|---:|---|
| Source teacher zero-shot | VAL34 only | selected full-label ACDC80 teacher, report only |
| Target teacher T | 2,000 optimizer steps; batch2; 4,000 exposures; 250 L8 epochs | target L8 only; T-b groups throughout; VAL34 selection |
| Joint J | 100 epochs ×175 TRAIN patients =17,500 paired steps | one L8 frame + one image-only U frame per step |
| Deploy | student VAL FG ≥ fusion VAL FG −0.01 | same fixed rule as ACDC |
| TEST | one transaction, four frozen comparison artifacts | source teacher, adapted teacher, student, fusion; no TEST selection |

The user explicitly chose100 target J epochs, so the target uses more J updates than ACDC's8,000. Labels used are **ACDC80 source plus8 target patients**. B is a transfer experiment; A is semi K8; the full-label run uses all TRAIN80 labels. B0 is optional and NOT_RUN.

The original345-case M&Ms open-release split is175/34/136. The prior geometry amendment and L8 lock are reused by SHA. Native images are read from original NIfTI or ZIP into a bounded RAM cache; no crop, resampling, new normalized disk arrays or non-L8 TRAIN mask access is allowed. Target TEST member payloads remain closed during preparation.

## Operations

A separate queue now waits for A's completed one-shot TEST, GPU availability, SHA verification and disk headroom. It will profile 20 T and 20 J optimizer steps, with largest permitted L8 T volume and largest TRAIN image-only U volume. OOM/profile failure stops without architecture or crop changes. A separate checkout, output directory and monitor isolate the target run.

The production checkout is `/root/Self-Audit-fuse-transfer-20261113`, pinned to
`553b53f259abaec64384c7a51e88db6ed1f4e568`; the new output root is
`/root/mount_data/runs/fuse-mnms-transfer-20261113`. Services
`fuse-mnms-transfer`, `fuse-mnms-watchdog` and `fuse-q3d-cleanup` are running.
The queue has zero automatic training retries. A's campaign PID remained 582872
before and after installation; the only GPU process at verification was A's J
child. A had completed 18 J epochs and reached update 1,500 of 8,000.

The visible Orca window **FUSE M&Ms | queued after A** attaches to tmux
`fuse-mnms-transfer`; it shows the waiting state and switches to per-class metrics
when B starts. Separate Luna alarms check B and archive completion every 10 minutes;
the existing A monitor is unchanged. IDs and precheck receipts are in the launch
receipt. No target ETA is inferred before the GPU profile.

The user authorized archiving the stopped historical Q3D run to private Google Drive and removing its server copy only after full remote SHA256 verification. Original ACDC/M&Ms data and FUSE A/full-label runs are excluded. At the launch observation, seven 2 GiB parts (14 GiB) had passed full remote download and SHA256 comparison; the complete archive was not yet verified and no source file had been deleted. The guarded cleanup service waits for complete verification and stopped producer/old writers. Final archive receipts will be added by the completion monitor. [Private archive folder](https://drive.google.com/drive/folders/1-EPd8W7iBvnn2RFxGm-LvG4CuSFL26Gk).

## Evidence

See PLAN.json for source teacher, geometry and L8 pins. Focused test reports are SOFTWARE_ONLY. Full pytest: 964 passed, 2 skipped, 2 warnings, 3 subtests passed (OMP_NUM_THREADS unset). The isolated full-suite audit including all staged new files passed and left git status empty. check_repo, legacy FUSE/Q3D smoke and four-mode transfer smoke passed; all four transfer cold loads were bitwise equal. Verbatim outputs are in validation/. REVIEW.json records the independent Claude verdict and remaining low-priority notes. MANIFESTS.json pins all three runtime manifests; CPU_PREFLIGHT.json verifies original-source hashes and largest native L/U frame loading without TEST voxel decoding. Target GPU profile, target training, target TEST and B0 remain NOT_RUN. No target efficacy or completion claim is made here.


Integration corrections include stable source-teacher provenance and the zero-shot
comparison alias, checkpoint/VAL ordering and resume compatibility, profile call
signatures, TEST archive verification before its ledger, complete source/geometry
crosslinks, and exact integral-label conversion for original float NIfTI masks.
No frozen A/A3D/Q3D source or Q3D config changed.

The source-integrity audit verified 183 unique TRAIN files (175 images and eight
L8 mask files), 68 VAL files and one complete TEST archive. TRAIN contains 16
labeled ED/ES frames and 4,602 image-only U frames. The prepared manifest SHAs
remain unchanged after these loader corrections. The temporary preparation
checkout had macOS AppleDouble metadata from its initial transfer; the receipt
records those files honestly. Production deployment uses a fresh Git checkout.

The archive uploader is a single uninterrupted tar stream with ordered 2 GiB
parts. Each part is downloaded from Drive in full and SHA256-compared before
being accepted; the source inventory must remain unchanged. A failed earlier
single-file upload is preserved as a failed attempt and is not an accepted
archive. The guarded cleanup helper defaults to a plan; server execution is
allowed only after the complete verification receipt and remote parts agree.
