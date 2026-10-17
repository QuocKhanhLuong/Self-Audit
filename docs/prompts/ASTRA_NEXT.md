# Astra: continue only the canonical student

Read ACTIVE_PIPELINE.json, docs/CURRENT.md and docs/RUNBOOK.md before editing.
The sole current trainer is train.py, backed by experiments/layer_guided_student_v1.
Do not restore small GuidanceStudent, historical full-pipeline wrappers, or stale
instructions from the archive. Do not add a second default trainer.

First run scripts/check_repo.py and the current tests. Verify official local
DINOv3 weights and input-bank provenance. A coarse-only pilot is execution evidence,
not evidence that native detail or self-learning has been completed.

Next implement native-detail training as a NEW policy/version, not a silent
extension of frozen_historical_q_reliability_v1. Preserve the existing reference
and the coarse-only baseline. Write the policy, observation operator, valid
support, sources of supervision and active gradient paths before training.

Mandatory requirements:

1. Preserve the existing exact D/U decomposition; identify teacher observation
   resolution separately from student stride 4. Merely resizing a 48x48 target
   does not introduce native anatomical information.
2. Demonstrate the current zero-detail invariant. Add a positive test showing
   a declared native, within-cell-varying image signal produces a finite nonzero
   detail gradient and changes detail weights. Keep a constant/replicated-target
   negative control; do not claim upsampled labels supply independent fine detail.
3. Use one image-derived evidence mechanism first. Specify which assumptions
   connect its cost to anatomy. A descending image energy is not a Dice guarantee.
   Keep padding invalid; do not manufacture background for unsupported pixels.
4. Test proposal quality and student-to-proposal transfer separately. Stop-gradient
   the candidate target and track coordinate transforms under augmentations.
5. Keep source/label provenance, class/source loss contributions, class permissions,
   target entropy/KL and native/coarse diagnostics explicit. Do not substitute
   total loss decrease for RV learning. Never create named ground truth from
   confidence, pixel coordinates or unverified metadata.
6. Do not project gradients then pass them through AdamW while claiming the
   additive-step guard theorem. A guarded update must check actual displacement
   and rollback optimizer state where applicable.
7. Compare the new policy with the same pretrained initialization, same bank,
   patient schedule and compute accounting. Version full-Q versus Z separately.
   Ensure a bounded pilot actually visits all patients declared in its design.
8. Keep the canonical CLI fail-closed: --describe must identify actual active
   architecture/phase; unsupported phases must error; official-weight or data
   failures never fall back to random initialization or the old model.

Deliver code, tests, a new schema/recipe and updated CURRENT/RUNBOOK in the same
change. Report actual commands, exit codes, hashes and missing capabilities.
Use a separate worktree; do not erase frozen data, kill jobs, fabricate receipts,
read references in training or tune on official TEST. Do not launch a full run,
merge or claim GPU/patient success without the corresponding explicit task and
execution evidence. Source-level cleanup must not weaken provenance checks.
