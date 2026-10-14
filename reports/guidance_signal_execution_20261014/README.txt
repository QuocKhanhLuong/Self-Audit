Q/Z pilot execution and fixed VAL evaluation, 14 October 2026

Start with summary.json, MAIN_METRICS.csv and CLASS_METRICS.csv.
The completed run is a paired continuation pilot: Q64 and Z64 optimizer steps
from the same original 50-epoch world-left-v2 checkpoint, fresh AdamW, seed 42,
batch size 8 and identical sample schedule. It is not a full retraining run.
Four TRAIN patients were selected; the fixed 64-step budget visited three.

Primary native gated results (percent except AUROC):
Model      Dice     IoU   Precision  Recall   AUROC    Coverage
Baseline   5.0685  2.9018   17.5510    3.3227  0.7900    0.3668
Q          1.1840  0.6132   17.2478    0.6878  0.8573    0.1155
Z         11.4078  6.6095   24.5228   21.2128  0.8816   62.0358

Z improves gated Dice by 6.34 percentage points over the baseline, but absolute
segmentation quality remains poor. Z class Dice: RV 0.13%, MYO 23.84%, LV 10.25%.
RV is nearly absent. Ungated argmax Dice is 17.19%, 21.06%, 21.50% for baseline,
Q, Z: Q/Z are close under this diagnostic. The gated gap is strongly affected
by prediction coverage. Different Q/Z target entropies also prevent ranking
the methods by their raw training cross-entropy values alone.
This small single-seed pilot is insufficient to establish generalization.

Evaluation uses the original locked VAL18, 36 labeled frames, original native
probability/margin gates (0.7/0.2) and class permissions. All predictions for all
models were frozen before references were read. TEST was not scored. VAL has
previously been exposed during development; it is not pristine independent
validation. No threshold search or checkpoint selection was performed here.
Metrics average frames within each patient, then patients, then RV/MYO/LV.
UNKNOWN=255 remains a false negative for foreground references. AUROC uses
ungated float32 probabilities on all native voxels, one-vs-rest with tie ranks.
Precision uses zero_division=0 for all three fixed foreground classes. The
older report excluded undefined precision values, so its 29.08% baseline
precision is not directly comparable to the 17.55% shown here.

Evidence directories:
- original_attempt/: original preflight failures, successful 86-test preflight,
  TRAIN signal replay and the support guard failure before any optimizer step.
  STATUS.training_started refers to process launch, not completed updates.
- baseline_stagewise/: ten-stage diagnostics for the original checkpoint only.
  Its Q/R0 stage names describe signals, not the subsequently trained Q/Z arms.
- training_r2/: user-approved excluded-weight tolerance amendment 1e-6 -> 2e-6,
  failed initial numerical test receipt, successful 94-test preflight, plan,
  schedule, full per-step Q/Z traces, training logs and frozen scalar receipts.
  Maximum excluded fraction was 1.3718879278030748e-6. All other model, loss,
  support, schedule and initialization settings stayed fixed. This is a
  disclosed TRAIN-based protocol revision, not original-protocol reproduction.
- evaluation_failed_backend/: retained failure before reference access, caused
  by one baseline threshold-boundary pixel changing under deterministic flags.
- evaluation/: successful evaluation using archived exporter backend flags.
  Baseline labels match the original export exactly for all 18 full cines;
  original baseline metric counts also match. Checkpoints remained unchanged.
  Twelve independent AUROC checks against sklearn on one labeled frame had
  maximum absolute error 1.11e-16. The evaluator's 20 synthetic tests passed.
- executed_source/: executed isolated pilot, evaluator, tests and dashboards.
  Production pilot/preflight fixes are merged separately into experiments/.
  The published evaluator test removes its host-specific worktree import;
  this directory retains the exact executed test version for provenance.

W&B training:
https://wandb.ai/nthne-hanoi-university-of-science-and-technology/self-audit-stable-guidance/runs/t5tmy3bj
W&B evaluation:
https://wandb.ai/nthne-hanoi-university-of-science-and-technology/self-audit-stable-guidance/runs/zfgxvxqg
Local W&B text exports reflect their capture time. FINAL_METRICS.json records
the final evaluation metrics also added to the completed training run online.

Publication provenance:
MANIFEST.json records original/published SHA-256 and byte counts for copied
files. PUBLICATION.json lists omitted files, symlink aliases and placeholders.
Machine paths, host/email and GPU UUIDs are replaced with placeholders; CSV
line endings are normalized to LF. Numeric measurements and artifact digests
are preserved. Sanitized freezes are report copies, not executable artifacts:
their embedded digests refer to originals retained on the execution machine.
Python evidence uses .py.txt so repository pytest will not collect snapshots.
Remove only the trailing .txt to reconstruct the executed source layout.

Images, masks, prediction/probability arrays, source ledgers, model weights,
binary W&B event stores and runtime caches remain private on the machine.
The previous hash-pinned stable_guidance_world_v2_20261014 bundle is unchanged.

Before publication, the merged pilot and portable evaluator tests passed together:
114 passed in 98.74s. See publication_checks/ for the log, JUnit and source hashes.
