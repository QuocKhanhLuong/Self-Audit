# External M&Ms evaluator review after the joint-from-epoch-1 change

Scope: `scripts/evaluate_external_mnms.py` and the evaluation helpers it calls.
Baseline: `main` at `bb639ec118af72dac0d5e24339a5e325e2d6d163`
("Add joint-from-start curriculum for native Candidate C runs").

Read-only with respect to production: no script, config, or `src/` file was
modified. The one file added by this review is
`tests/test_external_mnms_joint_lineage.py`. The native runner
(`scripts/run_acdc_mnms_candidate_c.sh`), `src/self_audit/data/mnms.py`,
native-side `_utils.py` validation, and `README.md` are another worker's scope
and are referenced here only where they touch the external path.

This is revision 2. It corrects three things stated too strongly in revision 1:
the characterisation of declared label mappings and depth axes (§3.1, §3.2), a
blanket "no backward" claim about the inference path (§1.2), and the standing of
`source_training.config` (§3.4). §2 and §5 now cite committed tests and exact
paths instead of scratchpad shorthand.

`src/self_audit/training/_utils.py` and `src/self_audit/data/mnms.py` are being
edited concurrently by the native worker, so line numbers into those two files
were re-checked at the time of writing and may drift; the named symbols are the
stable reference.

## Evidence status

All evidence below is CPU software evidence, of two kinds kept separate on
purpose:

1. **Committed regression tests** — `tests/test_external_mnms_joint_lineage.py`,
   9 tests, 4.70 s. Frozen tiny synthetic checkpoints stamped with the **real**
   `UnifiedConfig.to_dict()` tree parsed from the actual repository configs,
   driven end to end through `run_external_evaluation`.
2. **A one-off external run against a genuine CLI-produced checkpoint** — the
   alias-committed `best.pt` left behind by an earlier joint-from-start CLI
   smoke at
   `/var/folders/zk/7qfmzgnd1c9fqnzv4lfsm8t00000gn/T/self-audit-joint-from-start-8e20c8nq/acdc/weights/best.pt`
   (§2.4). That artifact was produced by `scripts/train_self_audit.py`, not by a
   bare `torch.save`.

No metric value anywhere in this report is a performance claim. The weights are
randomly initialised or trained for one tiny epoch, and the M&Ms cohort is
synthetic noise with block masks.

**Explicitly unknown on this host** (nothing here is evidence about any of them):

* **Real data.** `ls preprocessed_data` → *No such file or directory*. No real
  ACDC or M&Ms volume was read.
* **`timm` / real ConvNeXt-Tiny.** `timm` is not installed, so every model built
  here reports `encoder_backend: "fallback_synthetic"`
  (`src/self_audit/models/encoder.py:109`). Nothing here exercises the real
  backbone or its `state_dict` keys.
* **GPU (RTX 4070) execution.** Everything ran on CPU. No CUDA, AMP/bfloat16, or
  device-placement behaviour was exercised.

## 1. Independent trace: frozen ACDC `best.pt` to external M&Ms

### 1.1 Model reconstruction and Candidate C setting

`run_external_evaluation` (`scripts/evaluate_external_mnms.py:229`) resolves the
execution mechanism before it builds anything:

* Checkpoint mode declarations are gathered from five locations
  (`scripts/evaluate_external_mnms.py:253-281`) and any disagreement is a hard
  error.
* Candidate C solver settings are gathered from five locations
  (`scripts/evaluate_external_mnms.py:283-313`) and compared key by key through
  `_mechanism_values_equal`.
* CLI / eval-config / checkpoint conflicts are rejected pairwise
  (`scripts/evaluate_external_mnms.py:344-356`).
* A legacy checkpoint with no mechanism metadata may only be evaluated under
  `current` (`scripts/evaluate_external_mnms.py:358-371`), and a `candidate_c*`
  mode with no checkpoint solver settings is refused
  (`scripts/evaluate_external_mnms.py:408-412`).
* Absent-only inheritance: an *explicit* `candidate_c: null` in the eval config
  means "the documented defaults", not absence
  (`scripts/evaluate_external_mnms.py:373-375`, `414-419`).

After binding, the source config and the live module are both verified
unconditionally — `verify_model_config(model, flat)` and
`verify_model_config(model, ckpt_payload["config"])`
(`scripts/evaluate_external_mnms.py:446-448`), routed through
`_verify_execution_mechanism` (`src/self_audit/provenance.py:560-622`).
`verify_bound_state` brackets inference on both sides
(`scripts/evaluate_external_mnms.py:451`, `:489`).

Verdict: correct and fail-closed. No defect found.

### 1.2 tau, calibration, and source lineage

* **No weight update and no threshold sweep.** There is no optimizer, no
  parameter gradient, and no threshold search anywhere on this path.
  `src/self_audit/evaluation/cohort.py` and
  `src/self_audit/evaluation/volume_inference.py` contain no optimizer,
  `linspace`, or sweep construct; the only `model.train()` calls
  (`volume_inference.py:551`, `:672`) are `finally`-block restorations of the
  caller's prior mode.

  **Correction to revision 1**, which said "no `backward`" without
  qualification. That phrasing was wrong as written. Under `window_mode:
  candidate_c` the solver *intentionally* runs autograd during inference:
  `src/self_audit/models/self_audit_net.py:330` opens
  `torch.inference_mode(False), torch.enable_grad()`, `:338` clones the frozen
  factual sampling coordinates and marks them `requires_grad_(True)`, and
  `:449-453` calls `torch.autograd.grad(total, variables, allow_unused=True)`,
  counted as `coordinate_backward`. That differentiation is with respect to
  **sampling coordinates**, not network weights: the code's own comment notes it
  uses `autograd.grad` rather than `.backward()` precisely so no `.grad` buffer
  in the process is written or cleared, and the network parameters are not among
  `variables`. The accurate statement is therefore: **the external path performs
  no optimizer step and computes no gradient with respect to model weights; the
  Candidate C solver does compute coordinate gradients by design, as part of the
  frozen forward mechanism.**

* **tau is fixed and never derived from M&Ms**: CLI `--tau-accept`, else
  `config.audit.tau_accept`, else `0.0`
  (`scripts/evaluate_external_mnms.py:455-467`), with the origin stamped into
  the report as `tau_accept_source`. The shipped protocol config fixes
  `tau_accept: 0.0` (`configs/self_audit_acdc_to_mnms.yaml:34`), so the
  documented external run transfers **no calibrated threshold at all** — see
  §3.3.

* **Checkpoint identity**: `_inspect_checkpoint_dataset`
  (`scripts/evaluate_external_mnms.py:144-226`) collects every declared training
  dataset, rejects contradictions, rejects anything containing `mnm`, and
  rejects any non-`acdc` identity. A checkpoint with no identity at all
  downgrades the report to `uncertified_historical_checkpoint` rather than
  claiming `independent_external_evaluation`.

* **`weights_only=True`** with no pickle fallback on every checkpoint read:
  `scripts/evaluate_external_mnms.py:135`,
  `load_checkpoint` (`src/self_audit/training/_utils.py:1550`), and the
  stale-alias sibling read inside `_require_committed_best_alias`
  (deliberately fail-closed, `_utils.py:1723`).

* **Public-alias integrity**: `_require_committed_best_alias`
  (`src/self_audit/training/_utils.py:1679`) refuses a `best.pt` whose
  sibling `last.pt` commits a different sha256, and refuses one whose sibling
  records `best_reference=None`. Exercised for real in §2.4.

### 1.3 Split handling and BG/RV/MYO/LV mapping

* Only `test`/`testing` is accepted (`scripts/evaluate_external_mnms.py:85-87`).
* The two-class `mnm_binary` derivative is rejected twice — in
  `_normalize_external_config` (`scripts/evaluate_external_mnms.py:90-92`) and
  again in `validate_dataset_splits`
  (`src/self_audit/training/_utils.py:592-595`).
* `num_classes != 4` is rejected (`scripts/evaluate_external_mnms.py:102-103`).
* The mapping `{0:0, 1:3, 2:2, 3:1}` is the module default
  (`scripts/evaluate_external_mnms.py:57`), matches `DEFAULT_MNMS_TO_ACDC`
  (`src/self_audit/data/mnms.py:25`), and is the value the shipped protocol
  config declares. `MNMSClassMapping` (`src/self_audit/data/mnms.py:80-93`)
  requires raw keys exactly `0..3`, `0 → 0`, and a bijection onto `0..3`.

The mapping composes correctly end to end. With synthetic raw labels 1/2/3 the
report's ACDC indices and class names agree: raw 1 (LV) → ACDC 3 = `LV`,
raw 2 (MYO) → ACDC 2 = `MYO`, raw 3 (RV) → ACDC 1 = `RV`.
`test_real_config_lineage_evaluates_externally` asserts the report's
`label_mapping`, and
`test_protocol_config_declares_the_documented_mapping_and_depth_axis` asserts
the shipped config still declares it.

### 1.4 Depth axis

`depth_axis` flows to `build_patient_dataset` and to `_stored_grid`
(`scripts/evaluate_external_mnms.py:495`); `validate_depth_axis`
(`src/self_audit/training/_utils.py:254-261`) accepts `0|1|2|None`. The shipped
protocol config declares `depth_axis: 2`
(`configs/self_audit_acdc_to_mnms.yaml:21`), matching the documented
preprocessed M&Ms `[H, W, Z]` convention. See §3.2.

## 2. Joint-from-start lineage compatibility

### 2.1 What the new configs actually put in a checkpoint

The trainer stamps `UnifiedConfig.to_dict()` into the checkpoint
(`src/self_audit/training/unified_trainer.py:2809`, `:2898`) — the **nested**
schema-1 tree (`src/self_audit/training/unified_config.py:773-793`). Two
structural facts make the new configs work with the external evaluator:

* `config["dataset"]` is a mapping carrying `name: "acdc"`, one of the three
  keys `_inspect_checkpoint_dataset` reads from a config `dataset` mapping
  (`scripts/evaluate_external_mnms.py:178-181`). No competing identity key is
  emitted, so the "contradictory identities" branch cannot trip.
* `config["model"]` always carries both `window_mode` and a complete
  `candidate_c` mapping, because `ModelConfig.candidate_c` is a dataclass with a
  default factory (`unified_config.py:316`) and `CandidateCSettings`' fields are
  exactly `CANDIDATE_C_KEYS` (`unified_config.py:284-293` vs. `_utils.py:86-98`).
  So `validate_candidate_c_settings` on the checkpoint's own declaration can
  never fail on an unknown key.

### 2.2 Committed regression tests

`tests/test_external_mnms_joint_lineage.py` replaces revision 1's scratchpad
smokes. Each test parses the **actual repository config**, shrinks only the
model size (`shared_channels: 16`, `window_k: 4`, `max_turns: 2`,
`pretrained_encoder: false`, `fallback: true`), saves a frozen tiny checkpoint
carrying the real nested `to_dict()` tree, and drives `run_external_evaluation`
against a synthetic one-case two-slice M&Ms `testing` cohort. Only lineage,
provenance and refusal behaviour are asserted; no metric value is asserted
anywhere.

| test | covers |
|---|---|
| `test_real_config_lineage_evaluates_externally[staged_acdc_current]` | `configs/self_audit_full.yaml`, `window_mode: current` |
| `…[joint_from_start_acdc_current]` | `configs/self_audit_joint_from_start.yaml`, `window_mode: current` |
| `…[joint_from_start_acdc_candidate_c]` | same config switched to `candidate_c` as the native runner does; mode **and** full solver mapping inherited from the checkpoint and recorded verbatim |
| `test_cli_window_mode_conflicting_with_joint_checkpoint_is_refused` | `--window-mode current` against a `candidate_c` checkpoint |
| `test_eval_config_candidate_c_conflicting_with_joint_checkpoint_is_refused` | eval-config `rho_feature_pixels: 2.0` against the frozen source's `1.0` |
| `test_native_mnms_joint_checkpoint_is_refused_as_external_evidence` | `configs/self_audit_joint_from_start_mnms.yaml` |
| `test_committed_best_alias_binds_and_a_stale_alias_is_refused` | real commit protocol, then an interrupted-publication alias |
| `test_protocol_config_declares_the_documented_mapping_and_depth_axis` | shipped protocol pins mapping, `depth_axis: 2`, `split: testing`, `tau_accept: 0.0` |
| `test_external_evaluation_never_mutates_the_frozen_checkpoint` | checkpoint bytes unchanged after evaluation |

```
$ PYTHONPATH=.:src python -m pytest tests/test_external_mnms_joint_lineage.py -q
9 passed in 4.70s
```

**Conclusion: the joint-from-start config lineage is compatible with the
external evaluator, for both `current` and `candidate_c`, and this is now pinned
by tests.** No production change is required for it to run.

### 2.3 Pre-existing tests still pass

```
$ PYTHONPATH=.:src python -m pytest tests/test_external_mnms_evaluator.py \
      tests/test_mnms_external_flow.py -q
8 passed in 0.72s

$ PYTHONPATH=.:src python -m pytest tests/test_candidate_c_mnms.py -q
25 passed in 2.85s
```

`PYTHONPATH=.:src` is required; the repository has no `conftest.py` or
`pyproject.toml`, so a bare `pytest tests/test_mnms_external_flow.py` fails at
collection with `ModuleNotFoundError: No module named 'self_audit'`.

The gap those suites left — every one of them hand-builds the checkpoint config
as a *flat* dict such as
`{"model": {...}, "dataset": {"name": "acdc"}, "training_dataset": "acdc"}`
(`tests/test_candidate_c_mnms.py:363-386`, `:451-470`, `:494-513`), never the
nested tree a real run writes, and never references a repository config file —
is what §2.2 now closes.

### 2.4 The alias-commit question, closed two ways

Revision 1 listed "does a real committed `best.pt` bind?" as an unknown, on the
grounds that its synthetic checkpoints were written by a bare `save_checkpoint`
and therefore tested only **nested-config binding**, not the selected-best /
`best_reference` / alias commit protocol. That caveat about those artifacts was
right; the conclusion that settling it required real training was not. Tiny
training suffices, and it had already happened.

**(a) Portable, committed.**
`test_committed_best_alias_binds_and_a_stale_alias_is_refused` builds the alias
the way a run does — an immutable `selected_best/` snapshot, a hash-verified
reference from `make_best_reference`, that reference recorded inside `last.pt`,
then `publish_best_alias` — and asserts the external report's
`checkpoint_binding.checkpoint_sha256` equals the committed
`best_reference.sha256`. It then overwrites the alias with other bytes and
asserts the evaluation refuses with `the public best alias has sha256`.

**(b) Against a genuine CLI-produced artifact.** An earlier joint-from-start CLI
smoke left a complete run tree at

```
/var/folders/zk/7qfmzgnd1c9fqnzv4lfsm8t00000gn/T/self-audit-joint-from-start-8e20c8nq/
├── results.json                      # both datasets PASS, 2 optimizer steps each
├── acdc/weights/best.pt              # the public alias
├── acdc/weights/last.pt              # carries best_reference
└── acdc/weights/selected_best/epoch_1_2cd3c9ff908a.pt
```

That checkpoint's own config is a real joint-from-start tree —
`dataset.name: "acdc"`, `model.window_mode: "candidate_c"`, a full
`model.candidate_c` block, `checkpoint.best_selection_min_epoch: 0`, and a
single-interval `joint_self_audit` schedule with `trainable: "all"` and
`rollout: "threshold_gate"` — and `last.pt` carries

```json
"best_reference": {"schema_version": 1,
                   "path": "selected_best/epoch_1_2cd3c9ff908a.pt",
                   "sha256": "9dfe196854338c9a0b78c81e6f73728eded3008a684bda4f782e8ca0d6f092c6",
                   "epoch": 1, "metric": 0.09090264141559601}
```

Running the external evaluator against that alias plus a synthetic M&Ms
`testing` cohort (`configs/self_audit_acdc_to_mnms.yaml` with `image_size` and
the model size reduced to match that tiny run):

```
training_dataset          = acdc
evidence_class            = independent_external_evaluation
split                     = testing
window_mode               = candidate_c
tau_accept, source        = 0.0, config.audit.tau_accept
label_mapping             = {0:0, 1:3, 2:2, 3:1}
binding.checkpoint_sha256 = 9dfe196854338c9a0b78c81e6f73728eded3008a684bda4f782e8ca0d6f092c6
binding.epoch             = 1
encoder_backend           = fallback_synthetic
```

The bound sha256 equals the sha256 `last.pt` commits, so the evaluated weights
are demonstrably the committed selection. Copying `last.pt` over the alias in a
throwaway directory reproduced the refusal
(`Refusing to bind …: the public best alias has sha256 ef23e4c0… but …/last.pt
commits selected-best sha256 9dfe1968…`).

That temp tree is machine-local and will be reaped; (a) is the durable form.

## 3. Protocol constraints and remaining validation limits

Revision 1 filed the first two of these as severity-ranked findings. That
framing was wrong and is withdrawn. Neither is a defect, and neither is evidence
that target-data leakage or a threshold/mapping sweep occurred. They are
recorded here as **limits on what the evaluator validates**, so a reader knows
which parts of the protocol rest on the config file being correct rather than on
a runtime check.

### 3.1 The declared label mapping is input-format configuration, not a tuning knob

`_normalize_external_config` takes `external_test.raw_to_acdc` from the
evaluation config (`scripts/evaluate_external_mnms.py:94-99`) and validates its
*form* — keys `0..3`, `0 → 0`, bijection onto `0..3`
(`src/self_audit/data/mnms.py:80-93`). It does not require the canonical
`{0:0, 1:3, 2:2, 3:1}`.

That is the correct interface. The mapping declares **how the stored arrays
encode classes**, which is a property of the data on disk, not of the model or
the protocol. A non-canonical mapping is legitimate in real situations — most
obviously a cohort whose masks have already been remapped to ACDC indices, for
which the identity mapping `{0:0, 1:1, 2:2, 3:3}` is the *correct* declaration
and the canonical one would be wrong. Refusing every non-canonical mapping would
break those inputs, so **no global mapping lock is proposed here, and none is
authorized.**

Revision 1 also quoted a ~3.8x spread in synthetic macro-Dice across mapping
choices. That comparison is withdrawn: it measured random weights on noise, a
mis-declared mapping is a decoding error rather than a tuning gain, and
presenting it as a headline number was sensational and unsupported.

The genuine, narrow limit: the evaluator does not cross-check the declared
mapping against any independent record of how the cohort's arrays are encoded,
so a *mis-declared* mapping produces a silently wrong decode rather than a
refusal. What keeps the shipped protocol correct is that
`configs/self_audit_acdc_to_mnms.yaml` declares the documented values and the
report echoes what was applied in `label_mapping`. Both are now pinned:
`test_protocol_config_declares_the_documented_mapping_and_depth_axis` fails if
the shipped declaration is edited, and
`test_real_config_lineage_evaluates_externally` fails if the report stops
recording the mapping it applied. If anything further is wanted, the
non-interface-breaking option is a recorded `label_mapping_is_canonical` boolean
in the report payload — a disclosure, not a restriction. Not implemented; no
production edits under this assignment.

### 3.2 Absent `depth_axis` uses the documented heuristic

`validate_depth_axis` returns `None` for an absent key
(`src/self_audit/training/_utils.py:257-258`), and `infer_depth_axis` then
applies its documented rule — a single axis of size ≤ 64, else `argmin(shape)`
(`src/self_audit/data/common.py:147-154`). This is stated behaviour, not an
accident, and for real preprocessed M&Ms shapes (`[H, W, Z]`, `Z ≪ H`) it
resolves to axis 2.

`depth_axis: 0` is likewise not "wrong" in the abstract — it is the correct
declaration for an array genuinely stored `[Z, H, W]`. Like the label mapping,
the axis describes the file layout.

The limit to record: for a cohort whose layout is ambiguous, an absent
`depth_axis` is resolved by the documented heuristic rather than refused, and a
mis-declared axis is not detectable from the array alone. The shipped protocol
config declares `depth_axis: 2` explicitly
(`configs/self_audit_acdc_to_mnms.yaml:21`), which is correct for the documented
preprocessed layout and is now pinned by
`test_protocol_config_declares_the_documented_mapping_and_depth_axis`. Revision
1's suggested "require `depth_axis` in `_normalize_external_config`" patch is
withdrawn: it would be interface-breaking for callers whose data legitimately
relies on the documented inference.

### 3.3 tau provenance names the mechanism, not a calibration artifact

`tau_accept_source` records `cli:--tau_accept` / `config.audit.tau_accept` /
`default:0.0` (`scripts/evaluate_external_mnms.py:455-467`). It does not bind
the value to any ACDC calibration artifact, and it does not need to: the
documented protocol fixes `tau_accept: 0.0`
(`configs/self_audit_acdc_to_mnms.yaml:34`) and passes `--tau-accept 0.0`
literally. Recorded only so a report is never later read as evidence that the
external tau was the source run's *calibrated* tau. It is not; it is a fixed
`0.0`.

### 3.4 `source_training.config` is descriptive metadata, and checkpoint paths moved

`configs/self_audit_acdc_to_mnms.yaml:5-7` declares
`source_training.config: configs/self_audit_full.yaml`.
`grep -rn source_training` across `*.py`, `*.sh`, `*.yaml`, `*.ps1` matches only
that declaration: **no code reads it.** It is descriptive, ignored metadata — a
stale documentation lead after the joint-from-start change, not an enforced
lineage claim and not a violated one. Revision 1 overstated this by saying the
protocol config "asserts" a lineage that is "wrong".

Runtime lineage authority is unaffected and rests entirely on the checkpoint's
own embedded config: `_inspect_checkpoint_dataset` reads `config.dataset.name`
from the artifact, and `verify_model_config` runs against both the evaluation
config and the checkpoint's config. A joint-from-start checkpoint is identified
correctly regardless of what `source_training` says, which §2.2 and §2.4 both
demonstrate.

Separately, the external evaluator has **no** epoch or stage assumption of its
own: best selection is entirely config-driven through
`checkpoint.best_selection_min_epoch`
(`src/self_audit/training/unified_trainer.py:2775`, validated against the gated
interval at `src/self_audit/training/unified_config.py:897-907`), and the new
configs set it to `0`. No literal `120` gate survives in runtime code — only
stale module docstrings at `src/self_audit/training/unified_trainer.py:6` and
`:9` (native scope, flagged not owned).

What an operator needs to know is that an ACDC `best.pt` now lands in one of
three places depending on what produced it:

| producer | ACDC `best.pt` location |
|---|---|
| `configs/self_audit_full.yaml` (staged) | `weights/self_audit_full/` |
| `configs/self_audit_joint_from_start.yaml` | `weights/self_audit_joint_from_start/` |
| `scripts/run_acdc_mnms_candidate_c.sh:202` | `runs/$ACDC_RUN/weights/` |

The two `evaluate_external_mnms.py` invocations in `README.md` and the one in
`docs.md` hard-code only the first (line numbers omitted deliberately: the
native worker is editing `README.md` concurrently).
**Those files are the native worker's scope; no edit is proposed here.** The
external command form this review can vouch for is: point `--checkpoint` at the
frozen ACDC checkpoint produced by *that run's own* `checkpoint.output_dir`, and
pass the protocol's fixed `--tau-accept 0.0`, which is a fixed threshold and not
a calibrated one:

```bash
PYTHONPATH=.:src python scripts/evaluate_external_mnms.py \
  --config configs/self_audit_acdc_to_mnms.yaml \
  --checkpoint <that run's checkpoint.output_dir>/best.pt \
  --data-root preprocessed_data/mnm \
  --split testing \
  --tau-accept 0.0 \
  --device cuda \
  --output reports/external_mnms.json
```

### 3.5 Hypotheses tested and falsified

Recorded so they are not re-investigated:

* **An empty M&Ms test cohort is not silently validated.** The `explicit_split`
  branch of `validate_dataset_splits`
  (`src/self_audit/training/_utils.py:723`) lacks the native branch's explicit
  zero-record check (`_utils.py:641`), but `discover_mnms_records`
  raises first: `FileNotFoundError: No paired M&Ms image/mask volumes found
  under ...`. Confirmed against an empty `testing/volumes` + `testing/masks`
  pair.
* **The fallback encoder is not laundered as ConvNeXt-Tiny in provenance.**
  `ConvNeXtTinyEncoder.name` is `"convnext_tiny"` regardless of backend
  (`src/self_audit/models/encoder.py:87`), and the protocol config permits
  fallback (it declares no `encoder_allow_fallback`, defaulted to `True` at
  `src/self_audit/training/_utils.py:364`) while training configs set
  `fallback: false`. But `model_identity` records
  `encoder_backend: "fallback_synthetic"`, which appears in every report
  produced here, so the artifact discloses it.
* **Passing a nested training config as `--config` is rejected**, though the
  message is poor: `_normalize_external_config` stringifies the `dataset`
  mapping (`scripts/evaluate_external_mnms.py:76-77`) and reports the whole dict
  repr as "not mnms". Cosmetic only; the refusal is correct.

## 4. Remaining unknowns

Settled since revision 1: the committed-alias question (§2.4).

Still open, none of them testable on this host:

1. Whether a real ACDC `best.pt` (timm ConvNeXt-Tiny, `shared_channels: 96`)
   loads into the external evaluator's model. `load_state_dict(strict=True)`
   makes an architecture mismatch loud, but this was exercised only against the
   fallback encoder. **timm unknown.**
2. Whether the real preprocessed M&Ms `testing` tree satisfies
   `discover_mnms_records`' directory conventions
   (`src/self_audit/data/mnms.py:117-129` and the NIfTI paths). Only the
   `volumes/` + `masks/` `.npy` layout was exercised. **Real-data unknown.**
3. Whether real M&Ms raw masks contain only labels `0..3`.
   `MNMSClassMapping.apply` (`src/self_audit/data/mnms.py:99-105`) refuses an
   unmapped raw label, so an out-of-range label is a hard failure, not a silent
   remap. `scripts/inspect_mnms_mask.py` is the tool to confirm before a real
   run. **Real-data unknown.**
4. GPU behaviour: CUDA device placement, bfloat16 AMP, and the memory footprint
   of a full-size external evaluation. **RTX 4070 unknown.**
5. Any real-data or real-performance claim whatsoever. None is made here.

## 5. Reproduction

Durable, committed:

```bash
cd /Users/alvinluong/Self-Audit
PYTHONPATH=.:src python -m pytest tests/test_external_mnms_joint_lineage.py -q
PYTHONPATH=.:src python -m pytest tests/test_external_mnms_evaluator.py \
    tests/test_mnms_external_flow.py tests/test_candidate_c_mnms.py -q
```

One-off, machine-local while the temp tree survives (§2.4b): the genuine
CLI-produced alias is
`/var/folders/zk/7qfmzgnd1c9fqnzv4lfsm8t00000gn/T/self-audit-joint-from-start-8e20c8nq/acdc/weights/best.pt`,
its committed snapshot is
`.../acdc/weights/selected_best/epoch_1_2cd3c9ff908a.pt`, and the run manifest
is `.../results.json`. Evaluate it with `configs/self_audit_acdc_to_mnms.yaml`
patched to `image_size: 32` and
`model.{shared_channels: 16, window_k: 4, max_turns: 2}` to match that tiny run,
plus any synthetic four-class M&Ms `testing` cohort — the same construction
`_populate_mnms_testing` performs in
`tests/test_external_mnms_joint_lineage.py`.

Files added by this review: `tests/test_external_mnms_joint_lineage.py` and this
report. Nothing else in the repository was modified.
