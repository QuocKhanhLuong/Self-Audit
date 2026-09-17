# Implementation audit by Astra

IMPLEMENTED in the isolated experimental namespace. Existing production methods are unmodified. The model is intentionally small and has one optional refinement; it is not a full reproduction of the repository's ConvNeXt actor.

| Boundary | Actual implementation | Evidence |
|---|---|---|
| Annotation | `[B,3,H,W]` → `/4` features → shared QKV dynamic read → `[B,4,H,W]` logits | `annotation.py`, shape/causal tests |
| Window | `[B,h,w,8,2]`; condition = four actor probabilities + one detached q map | Source + fixed-coordinate intervention |
| Audit observation | Explicit floating 3-channel image and four-class logits; cloned/detached from AnnotationState | Signature, raw-target rejection, alias and gradient tests |
| Anchor | Registered Sobel buffers plus intensity/local average; no trainable parameters | Frozen buffer/parameter test |
| Intrinsic score | Hard predicted-region feature residual + boundary penalty; no GT/corruption labels | `intrinsic_risk` and permutation counterexample |
| Spatial q | Small convolution head distills intrinsic risk | RF MSE; no actor gradients |
| Value | 25 detached features for C=16; MLP regresses net RF contrast | No auditor-lite full forward before decision |
| Decision | Positive values, stable top floor(Bρ); batch cap | Controlled both-action and tie tests |
| Skip | Exact A0 logits; q never invoked | Hook-based identity test |
| Twins | One shared pre-update A0; identity skip vs one guided read | Equality, no mutation and no-grad tests |
| Replay | Detached image/logits/features/value/step only; uniform RNG; age expiry | Deterministic sampling/clone tests |
| Actor objective | Retained per-example CE + foreground soft Dice | GT appears only in actor loss |
| Slow updates | RF/Trigger every four actor steps, separately counted | Trainer counters and firewall checks |
| Evaluation oracle | Decorated no-grad, used after output generation only | No returned gradient; GT permutation integration test |

`AuditObservation` is an interface firewall, not a cryptographic provenance system: arbitrary Python callers could deliberately wrap a one-hot GT tensor as logits. The guarantee is the implemented call graph, detached experience construction and tests showing GT permutation leaves pre-update RF experiences/updates identical while actor updates differ. No claim is made that tensor shapes alone prove origin. Dataset masks are loaded for supervised actor training; neither their values nor derived error masks enter RF APIs.

`forward` computes cheap detached trigger features/value for all control policies, including no-audit, for common diagnostics. Thus B0 has zero full audit calls but retains this small common overhead; the present timing is not an optimized annotation-only deployment latency. All geometry/value work and RF setup/probes must be counted before a speed claim.

## Defects caught before real ACDC runs

Union's first actor implementation ignored prior logits except for their shape, used the wrong condition-channel count, omitted actor probabilities from the condition, detached returned coordinates, and initially had mutually incompatible guidance shape checks. A later partial worker edit loosened shape checking but did not repair the recurrence/conditioning contract. The worker then repeatedly aborted writes, leaving a zero-byte Auditor. This was **not** a usable delivery.

Astra preserved the partial artifact, confirmed worker process exit, fenced its dispatch, corrected the five actor issues and implemented Auditor/Trigger/replay/model/loss modules. No experiment used the defective partial actor. Root's first complete suite passed 14 tests.

Astra also caught an evaluation-schedule defect in the new harness: holding a single global step constant would make periodic audit always-on/off across an entire validation pass. It now advances eligible batch events; a dedicated test verifies 50% occupancy for an even fixture. This fix preceded real ACDC execution. No threshold or scientific setting was changed in response to validation quality.

## Important limitations

Hidden metadata rejection does not rule out visible corruption artifacts. No corruption ranking is trained in this baseline. The class-permutation invariant fixed score cannot know anatomical semantics. Replay expiry limits stale targets but does not remove actor/q nonstationarity. The B=8 cap is batch-dependent and has a floor-rounding limitation at B=1. No soft gate, tile composition, multi-candidate search, acceptance critic, historical repair or smoothing is silently added.

Hardware evidence is CPU only for new code. The observed occupied 4060 Ti is not a CUDA validation. AMP, activation checkpointing, distributed execution, full training resume, 4090 throughput, peak training VRAM and M&Ms generalization are NOT TESTED.

## Reproduction and scope

Tested environment: Python 3.11.16, Torch 2.14.0, macOS 26.2 arm64, CPU. NumPy, PyYAML and nibabel are required by the harness; SciPy is used by the evaluation-only extent diagnostic. Existing repository dependencies still apply. No package/global configuration was changed for the research.

From the isolated worktree:

```sh
PYTHONPATH=src /Users/alvinluong/miniforge3/bin/python -m pytest tests/test_event_audit.py -q
/Users/alvinluong/miniforge3/bin/python scripts/train_event_audit.py --config configs/self_audit_event_acdc.yaml --data-root /tmp/astra_event_acdc_training --patient-manifest /tmp/astra_event_acdc_training/acdc_patient_split_seed42.json --output /tmp/event_acdc_fresh_run
/Users/alvinluong/miniforge3/bin/python reports/astra_event_self_audit_20260917T175414Z/evidence/analyze_results.py
```

Use a new output path; the training command refuses to overwrite completed experiments and enforces CPU/short-run limits. Exact data-root and manifest paths also appear in `evidence/acdc_data_identity.json`. The supplied configuration reproduces the original 21-run protocol, not the separate frozen-head diagnostic. Run `scripts/check_event_audit.py --help` for the frozen intervention interface; `--fit-frozen-trigger` is explicitly a separate diagnostic. The actual config and all implementation source hashes in the real-run receipt match the final implementation (the diagnostic script later received a comment clarification only).

Model `.pt` checkpoints remain locally available in the experiment evidence folders but are ignored by Git. Raw ACDC volumes stay in `/tmp` and are not repository artifacts. Reports include deidentified patient IDs, metric records and file hashes, not medical images. Unrelated dirty-work diff snapshots were retained privately outside Git; their hashes and byte-identical final checks are in `evidence/active_checkout_preservation.json`.

The publication allowlist consists only of `src/self_audit_event/`, its new config, two new scripts, one new test file and this new report directory. No existing tracked model, training or config path is modified. Experimental code is a bounded research baseline, not a drop-in production or variable-batch inference API.
