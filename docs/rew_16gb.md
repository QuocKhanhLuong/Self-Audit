# Supervised Read–Evaluate–Write: experimental 16 GB test recipe

## Status and scope

Implementation prepared 17 September 2026 against main
`96c32b10fc7b8e09b48822e10ae9eb6cc149e253`. This is an **opt-in supervised
experiment**, not a replacement of the frozen baseline or the mask-free
package. No legacy production model, loss, loader, calibration, or launcher is
edited. Candidate C is not enabled in this first experiment. Its replay records,
checkpoints, and calibration do not certify REW.

The implementation reuses the existing ConvNeXt-Tiny wrapper, FPN, initial head,
annotation loss, ACDC discovery and 2.5-D preprocessing. It adds an explicitly
separate model/loss/runner inside `src/self_audit`, rather than making old
checkpoints silently load into a different model.

## What actually runs

1. Encoder/FPN and initial head generate A0.
2. One auditor evaluates A0/A0: CC/WW state correctness, exact zero quality
   change. This is not fabricated transition history.
3. One shared writer tries three reads: fixed local K8; four local anchors plus
   four learned near ellipse points; four anchors plus four learned wider points.
   The near/wide radii are initialized differently but are trainable; their
   ranges overlap and diversity is not guaranteed. There are no separate experts.
4. Each read uses a pair-message MLP over query/source content, annotation
   probabilities and relative position. There is **no QK dot-product attention**
   in this reader. Audit feedback conditions sampling geometry, not its content
   projection. Each turn uses one writer block; this is not legacy depth 1/2/3.
5. The auditor estimates CC/FIX/REGRESS/WW for each proposed read–edit pair.
   On each non-overlapping 32x32 output tile, select the largest predicted
   `mean(pFIX - 2*pREGRESS)`, requiring improvement over KEEP by 0.01. KEEP
   is exact identity; ties retain it. Logits of different candidates are NOT
   averaged. Partial edge tiles have their actual pixel-count denominator.
6. The assembled whole-image candidate is independently audited again. It is
   retained only when predicted slice-Dice change exceeds tau=0 and hard labels
   actually change. Otherwise retain the previous state and halt that sample.
   All candidate readers/writers and the auditor are shared across <=3 turns.

The local four labels concern correctness BEFORE/AFTER, not anatomy classes:
CC=correct/correct; FIX=wrong/correct; REGRESS=correct/wrong; WW=wrong/wrong.
An unchanged hard label cannot be FIX or REGRESS. Different hard labels cannot
be CC. These admissibility constraints use predictions only. All model inputs
are explicitly logits; the implementation never guesses logits vs probabilities
by checking whether their numeric values happen to lie between zero and one.

Anatomy classes remain BG=0, RV=1, MYO=2, LV=3. Local FIX-minus-REGRESS utility
is **not delta Dice**. The global head learns a separate foreground slice-Dice
change target. Validation reports resized-grid 3D class Dice, so this metric-space
difference is explicit. No calibrated confidence or non-degradation guarantee
is claimed.

## Learning from the first step

Training is joint from the first update, with a detached hard accept/reject gate.
The actor receives final-retained segmentation loss, 0.5*A0 loss and
0.5*mean(all actual proposal segmentation losses). Even a rejected read's writer
and coordinate generator can learn. The annotation loss is existing Dice+CE.

Auditor loss contains A0 state supervision, every attempted natural candidate,
the actual assembled candidate, 0.2*paired local read-advantage MSE, and
0.25*synthetic-pair loss. Synthetic pairs are same-patient training-only partial
repair/label disturbance, evaluated in both directions. Their outcomes are
measured against GT, not assigned by the edit's name. No synthetic pair is fed
into official rollout history.

The paired advantage target compares each adaptive read's measured
`FIX - harm_weight*REGRESS` with the local-only read on the SAME input state and
writer weights. Its gradients train the auditor. Geometry learns from the GT
segmentation losses on all reads, not from backpropagating the auditor's score.
There is **no implemented learned read-policy distillation or reinforcement
learning stage**. Such extensions remain research proposals.

Auditor inputs detach features/predictions. Its loss cannot train the actor;
feedback and selection detach the auditor output. GT is passed only to loss
construction AFTER model forward. `ReadEvaluateWriteNet.forward(image)` accepts
no GT argument. A random auditor is not assumed reliable at initialization.
Zero acceptance at the start is possible; check nonzero proposal loss and
geometry/auditor gradients rather than forcing acceptance.

## Hardware recipe

`configs/self_audit_rew_16gb.yaml` is conservative, not a measured memory gate:

| Setting | Value |
| --- | --- |
| Physical batch / accumulation | 1 / 8 (8 samples per complete optimizer group) |
| Image / shared channels | 256x256 / 96 |
| Encoder | ImageNet ConvNeXt-Tiny, no silent fallback |
| Read candidates / samples per read | 3 / 8 |
| Maximum turns / output tile | 3 / 32x32 |
| Precision | CUDA BF16 autocast; sampling coordinates use float32 |
| Memory tradeoff | Activation checkpointing of each read, non-reentrant |
| DataLoader workers / prefetch | 2 / 1 batch per worker |
| Cache | 1 raw volume per worker, same baseline preprocessing |
| Parent Torch threads / worker threads | 4 / 1 |
| Epoch horizon / warmup | 50 / 5 |
| Encoder / actor / auditor base LR | 3e-5 / 3e-4 / 3e-4 |

RAM and GPU VRAM are different budgets. Data workers use host RAM; the learned
auditor runs on the model's CUDA device (not the old FP64 mask-free audit path).
K8 does not make three proposals free. Checkpointing recomputes work to save
activations. No runtime speedup or 16 GB full-run memory guarantee is made.
Unsupported BF16 or missing CUDA stops explicitly; there is no automatic batch,
precision, device, or pretrained-weight fallback.

## Run on the rental machine

Use an isolated checkout of the delivered branch; do not switch a live training
checkout underneath a running process. Worktree paths below may be changed.

```bash
git -C /root/Self-Audit fetch origin feat/rew-supervised-16gb
git -C /root/Self-Audit worktree add --detach \
  /root/Self-Audit-rew origin/feat/rew-supervised-16gb
cd /root/Self-Audit-rew
python -m pip install -r requirements-rew.txt
```

Keep your existing CUDA-compatible PyTorch environment. The first pretrained
encoder construction requires reachable ImageNet weights or a populated timm
cache. No weights are shipped in this delivery.

Bounded GPU smoke: eight microbatches (one update with this recipe), plus at
most two validation batches. It never starts all 50 epochs.

```bash
SMOKE="runs/rew/smoke_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python scripts/train_rew.py \
  --config configs/self_audit_rew_16gb.yaml \
  --data-root /root/Self-Audit/data/ACDC \
  --patient-manifest splits/acdc_patient_split_seed42.json \
  --output "$SMOKE" --max-batches 8 --max-val-batches 2
cat "$SMOKE/run_status.json"
tail -n 1 "$SMOKE/train_metrics.jsonl"
```

The existing patient manifest stores ED/ES volume names; raw ACDC uses frame
names. `--patient-manifest` preserves its train/validation PATIENT membership
while retaining actual frame IDs. It does not invent an ED/ES mapping. Missing
patients, overlapping memberships, unknown paired patients and official-test
conflicts fail closed. Image and GT content for train/val are hashed. Official
test filenames may be inventoried, but their arrays are not evaluated here.

For intentionally new development splits only, `--split-policy train80val20`
can be used instead of a manifest; it reserves official test data and records
new membership. Do not use this to silently change a baseline comparison.

Full 50-epoch TEST EXPERIMENT, after reviewing the GPU smoke:

```bash
RUN="runs/rew/acdc50_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python scripts/train_rew.py \
  --config configs/self_audit_rew_16gb.yaml \
  --data-root /root/Self-Audit/data/ACDC \
  --patient-manifest splits/acdc_patient_split_seed42.json \
  --output "$RUN"
```

Use a fresh directory; do not resume `bounded.pt`, old Self-Audit checkpoints or
mask-free runs. The runner honors configured epochs, rather than a 150-epoch
completion rule. Full runs save `last.pt` after each completed epoch, including
optimizer/scaler/RNG and durable log offsets. To continue an interrupted full
run, use the same config/root/manifest/output and add `--resume "$RUN/last.pt"`.
Resume fails on source/config/data identity mismatch, missing checkpointed logs,
or a bounded/non-resumable checkpoint. It restores completed-epoch state, not
uncheckpointed work. CUDA bitwise replay is not certified.

## Reading the test result

`run_status.json`: completed_bounded vs completed, configured/completed epochs,
step count, setup/full-invocation wall times and measured CUDA peak allocated/
reserved memory. Process-tree RSS is a sampled sum and can double-count shared
pages; it is not a cgroup peak or a certified total RAM bound. Startup peaks and
future epoch peaks may differ. CUDA fields are null on CPU.

`train_metrics.jsonl`: proposal/state/auditor losses; geometry and auditor
update gradient norms; actual attempts/accepts; pixel choice counts
[KEEP,local,near,wide]; local target counts [CC,FIX,REGRESS,WW]. Batch wall seconds
include input wait/transfer; compute_seconds is recorded separately. No claimed
accuracy follows from nonzero loss or gradients.

`validation.jsonl`: A0/final resized-grid 3D class Dice, first per volume, then
mean per patient/class and foreground macro. Both-empty classes are excluded;
no available foreground yields null. Bounded validation does not report
partial-volume Dice. Validation is report-only: no best-checkpoint selection,
threshold sweep or external tuning. Final/native export, clinical volumetry,
external M&Ms evaluation and calibrated risk control are not implemented by
this new runner; legacy evaluators must not be presented as supporting this
checkpoint without an explicit adapter.

## Executed evidence and limits

CPU tests: **41 passed, 1 CUDA skipped**, 11.83 seconds, one pre-existing warning.
Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
python -m pytest -q tests/test_rew.py tests/test_self_audit_core.py --disable-warnings
```

Checks include all 64 correctness triples, KEEP identity/ties, actual tile
candidate selection, final re-audit, actor/auditor gradient separation, geometry
learning despite rejection, checkpointed-vs-normal gradients, native synthetic
NIfTI and 0/2 spawned data workers, patient-manifest mapping, partial validation,
bounded CLI, occupied-output rejection, and epoch resume. Resume matched model
weights and Torch RNG bitwise in the tested CPU fixture. Runner tests use an
explicit small injected encoder, never a claimed pretrained ConvNeXt run.

A separate legacy Candidate-C test failed at `test_candidate_c.py:348` on BOTH
the untouched baseline and the modified tree in this environment. It expects
only one replay row to settle; both settle here. It was not fixed, excluded
silently, or counted as passing. Both raw failure logs are retained. This is not
a claim that all legacy or repository CI tests pass.

No real MRI corpus, trained checkpoint, CUDA device, timm model instantiation,
ImageNet weight download, clinical quality, novelty or sustained speedup was
verified here. Local timm installation was unavailable due network resolution.
The GPU smoke on the user's machine remains required.
