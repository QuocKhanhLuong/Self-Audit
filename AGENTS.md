# Current source of truth

Read ACTIVE_PIPELINE.json and docs/METHOD.md. The only trainer is train.py,
backed by self_audit/candidate_a/. Candidate A uses a declared dense TRAIN label
budget; U images are currently unused. Do not claim semi-supervised improvement
or no-GT training. All teacher/K1/measurement target labels count in that budget.
No U/VAL/TEST mask may influence training, target generation, label-subset
selection or gates. Development metrics are read-only and do not select a
checkpoint or stop training.

Teacher and student receive the same MRI context. Freeze teacher outputs for a
whole student rollout. K1 trials use the same decoder state and quantized student
values. Teacher discrepancy only proposes trial IDs. The student acts on the
whole feasible bank plus STOP. The decoder receives code, coordinates and public
geometry only. Count R0, IDs, finite value levels, padding and packet metadata.

An epoch covers every annotated TRAIN frame/slice once. Keep patient splits,
label budget, sampling/augmentation seeds and code budgets matched across arms.
Log per-class Dice/F1/precision/recall with TP/FP/FN and absent-class status.
Physical distances are 2D in-plane; three-slice context does not make them 3D.
ACDC/M&Ms source ontologies must be explicitly normalized before training.

Do not revive old joint_guard/EMA or frozen-Q/CineMA trainers. The previous tree
is recoverable at f01add12524cf73bb15675ecfeadc1014e420210; frozen-Q history remains
at archive/pre-semisupervised-20261020 (924265f7df604d3dca84dda49578b1a0cf28d821).
Historical checkpoints require their pinned source; never silently reinterpret
old configs or checkpoint semantics as the current epoch protocol.

Run python scripts/check_repo.py and python -m pytest -q before delivery. Report
actual tests, source/weight hashes, file paths and missing evidence. Manufactured
fixtures are not patient/GPU evidence. Do not run full patient training,
download/accept licenses, change active jobs, or delete local datasets.
Use isolated worktrees. Never git clean -fdx around user data/checkpoints.
