# Candidate C — config and baseline interfaces (Worker 5)

Base revision: `9a612485b7ead1273082d7dd52879f8e9e7a3e83`. Contract:
`reports/candidate_c/architecture_contract.md`.

Scope of this report: the configuration schema, the model-builder boundary, the
canonical ACDC YAML, and the checkpoint lineage of the execution mode. Nothing
here measures, or claims anything about, whether Candidate C works. No
experiment was run and no model was trained.

## 1. Owned changes

| File | Change |
|---|---|
| `src/self_audit/training/_utils.py` | `WINDOW_MODES` / `DEFAULT_WINDOW_MODE` / `CANDIDATE_C_SOLVER_MODES` / `CANDIDATE_C_DEFAULTS`; `validate_window_mode`; `validate_candidate_c_settings`; `window_mode` + `candidate_c` added to `MODEL_ARCHITECTURE_KEYS`; `filter_model_config` normalizes instead of dropping them; `build_model_from_config` forwards them to `build_self_audit_net` |
| `src/self_audit/training/unified_config.py` | `CandidateCSettings` dataclass; `ModelConfig.window_mode` + `ModelConfig.candidate_c` + `runs_candidate_c_solver`; `RolloutConfig.predicted_history_exposure` + `predicted_history_weight`; `to_legacy_model_config` carries the mode; `ResolvedExecutionConfig.window_mode` / `.candidate_c_settings` and flat-config normalization |
| `configs/self_audit_full.yaml` | documented `model.window_mode: "current"`, the full `model.candidate_c` block, and the two `training.rollout` exposure fields |
| `tests/test_candidate_c_config.py` | new, 68 tests |
| `reports/candidate_c/baseline_interfaces.md` | this report |

No other file was touched. No commit, no push, no spawn.

## 2. The switch

`model.window_mode` is the single activation switch. Exactly six values:

```
current | feature_only | free_offsets | candidate_c | candidate_c_no_fix | direct_rollback
```

`current` is the default and reproduces the historical network. There is no
`enabled` or `enforce_fix` field — per the contract, the mode alone selects
activation and constraint. `model.candidate_c` is always present in the
serialized config so the lineage records the exact solver settings that were in
force. The solver runs only in `candidate_c` / `candidate_c_no_fix`;
`direct_rollback` also consumes the predicted REGRESS threshold and minimum mass.

### CLI examples

The canonical entrypoint is `scripts/train_self_audit.py`. Baseline, unchanged
behaviour (the canonical file already says `current`):

```sh
python scripts/train_self_audit.py --config configs/self_audit_full.yaml
```

Candidate C needs no code change and no new switch on the command line — only a
config whose `model.window_mode` differs. **No such file exists in the repo
today**; create one by copying the canonical file and editing that one line:

```sh
cp configs/self_audit_full.yaml configs/self_audit_candidate_c.yaml
# then edit configs/self_audit_candidate_c.yaml and change exactly:
#   model:
#     window_mode: "candidate_c"     # was "current"
python scripts/train_self_audit.py --config configs/self_audit_candidate_c.yaml
```

Nothing else in the copy needs to change; `model.candidate_c` may be left at the
documented defaults. Whether a separate ablation config should be committed to
`configs/` is the coordinator's call, not this stream's.

Programmatic check of what a config will actually build, before spending a GPU
hour on it:

```sh
PYTHONPATH=src python - <<'PY'
from self_audit.training.unified_config import resolve_downstream_config
r = resolve_downstream_config("configs/self_audit_full.yaml")
print(r.window_mode)            # current
print(r.candidate_c_settings)   # complete validated mapping
PY
```

### Config examples

Model section as it now reads in `configs/self_audit_full.yaml`:

```yaml
model:
  encoder_name: "convnext_tiny"
  pretrained_encoder: true
  fallback: false
  shared_channels: 96
  num_classes: 4
  window_k: 8
  max_turns: 3
  window_mode: "current"
  candidate_c:
    rho_feature_pixels: 1.0
    lam: 1.0
    lr: null            # null selects the normalized one-feature-pixel proposal rule
    fix_threshold: 0.5
    regress_threshold: 0.5
    margin_fraction: 0.5
    min_regress_mass: 0.001
    replay_atol: 0.00001
    replay_rtol: 0.00001
    max_backtracks: 2
```

Rollout section:

```yaml
  rollout:
    tau: 0.0
    max_turns: 3
    predicted_history_exposure: false
    predicted_history_weight: 0.1
```

The C defaults are exactly the contract values. `max_backtracks` is validated in
`[0, 2]`: the contract budgets at most two candidate checks in total, so a YAML
cannot quietly raise the budget.

## 3. Strict validation

Everything is rejected rather than coerced:

* unknown `model.*` key — still `Unknown keys in section 'model'`;
* unknown `model.candidate_c.*` key — `Unsupported model.candidate_c key(s): ...`
  (this is how `enabled: true` fails);
* unknown `training.rollout.*` key — `Unknown keys in section 'training.rollout'`;
* unknown `window_mode` value — error naming the six legal modes;
* NaN / ±Inf anywhere in `candidate_c` or in `predicted_history_weight`;
* `rho_feature_pixels <= 0`, `lr <= 0` (when not null), `lam < 0`,
  `min_regress_mass` outside `[0, 1]`, `replay_atol/rtol < 0`, `fix_threshold` /
  `regress_threshold` / `margin_fraction` outside `[0, 1]`,
  `max_backtracks` non-integer or `> 2`, `predicted_history_weight < 0`;
* `predicted_history_exposure` must be a real bool — `1` and `"yes"` are errors;
* `predicted_history_weight: true` is a type error, not `1.0`.

The historical required keys of `model` and `training.rollout` are still
required; only the new fields are optional.

## 4. The option is not silently filtered

`filter_model_config` previously returned `{key: config[key] for key in
MODEL_ARCHITECTURE_KEYS if key in config}`, so any new model key would have been
dropped on the floor. `window_mode` and `candidate_c` are now in
`MODEL_ARCHITECTURE_KEYS`, validated in `filter_model_config`, defaulted in
`build_model_from_config`, and passed to `build_self_audit_net`.

The core worker owns `CandidateCConfig` and the `SelfAuditNet` constructor. The
schema does not import the model package (no import cycle): it hands over a
complete, validated plain mapping and the model package converts it.

There is no compatibility scaffolding for an intermediate working tree: the core
constructor is implemented, so `build_model_from_config` validates the two keys
and passes them straight to `build_self_audit_net`. A config asking for
`candidate_c` therefore either builds a `candidate_c` network or raises; it can
never quietly build the baseline.

**Schema/runtime bound alignment**, checked against the landed
`models/self_audit_net.py`:

| Field | Model package | Schema | Note |
|---|---|---|---|
| `margin_fraction` | `[0, 1]` | `[0, 1]` | aligned (schema was tightened to match) |
| `fix_threshold`, `regress_threshold`, `min_regress_mass` | `[0, 1]` | `[0, 1]` | aligned |
| `rho_feature_pixels` | finite > 0 | finite > 0 | aligned |
| `max_backtracks` | int in `[0, 2]` | int in `[0, 2]` | aligned; total candidate checks |
| `window_mode` | lowercased/stripped then matched | exact match | schema stricter; no silent case coercion |

Every direction is schema-stricter-or-equal, so the schema never admits a value
the runtime would later refuse. `test_schema_rejects_everything_the_model_
package_rejects` asserts this.

## 5. Old configs still load

* Canonical unified configs without the new keys parse and default to
  `window_mode: current`, the contract C defaults, `predicted_history_exposure:
  false`, `predicted_history_weight: 0.1`.
* Historical flat (non-`schema_version: 1`) configs resolve through
  `ResolvedExecutionConfig`; the resolved model config is normalized to state
  `window_mode: "current"` explicitly rather than leaving it unset, and an
  invalid mode in a flat config is still an error.
* `tests/test_unified_config_contract.py` (24 tests, pre-existing) passes
  unchanged.

This section is about reading **config files**. It says nothing about resuming
an old **checkpoint**, which is fail-closed — see §8a.

## 6. Serialization roundtrip

`UnifiedConfig.to_dict()` → `parse_unified_config()` → `to_dict()` is exactly
equal, with every knob retained, for the canonical file and for each of the six
modes with a non-default `candidate_c` value. `model` serializes to exactly the
nine documented keys; `candidate_c` serializes as a complete mapping (never a
partial one that could later drift against a changed default).

## 7. Baseline preservation

Asserted in `test_canonical_baseline_recipe_is_unchanged`: `convnext_tiny`,
`pretrained_encoder: true`, `fallback: false`, `shared_channels: 96`,
`window_k: 8`, `max_turns: 3`, `stage_weights (0.5, 0.7, 0.8, 1.0)`, total 130
epochs, intervals `[0,100) annotation_bootstrap`, `[100,120) auditor_training`,
`[120,130) joint_self_audit`, `window_mode: current`,
`predicted_history_exposure: false`. No A0 strength, learning rate, batch size
or schedule boundary was touched.

## 8. Checkpoint / provenance binding of the execution mode

The authoritative binding is `provenance.resolve_model_identity` /
`provenance.verify_model_config`, owned by the runtime worker. **That work has
landed** in the working tree: `resolve_model_identity` reports `window_mode`,
`candidate_c_settings` and `candidate_c_signature`, and folds the mode and the
solver signature into the signed identity itself rather than appending a field
after hashing, so the signature is not mode-blind. `verify_model_config` then
refuses a mode mismatch, an unreportable mode, and a per-setting solver
mismatch.

This stream owns only the input side of that check, and deliberately adds no
second implementation:

* `UnifiedConfig.to_dict()` writes `model.window_mode` and the complete
  `model.candidate_c` mapping into the config saved in the checkpoint payload;
* `to_legacy_model_config()` puts both into the `model` section that
  `verify_model_config` is handed through `_checkpoint_binding_from_payload`;
* `_checkpoint_binding_from_payload` calls the centralized verifier and nothing
  else — an earlier local `_bind_execution_mode` helper was removed in this
  revision, because a second checker that could pass while the signature stayed
  mode-blind is worse than no local checker at all.

`tests/test_candidate_c_config.py` §6 exercises the real mechanism: it asserts
the identity reports the mode (failing, not skipping, if that regresses), drives
`verify_model_config` with a real tiny net to get both a mode mismatch and a
`candidate_c.lam` solver mismatch, and checks the matching cases bind.

There is no path by which an external ACDC→M&Ms run can retune the solver: the
settings live in `model.candidate_c` in the training config, they are serialized
into the checkpoint config and into the model identity signature, and the
centralized verifier refuses a checkpoint whose declared mode or solver settings
differ from the live model. No external-label-driven adjustment of any C field
exists.

### 8a. Old checkpoints are fail-closed, and that is not the same claim as §5

Two distinct statements, easy to conflate:

* **Old config *files* still load.** A YAML without `model.window_mode`,
  `model.candidate_c`, or the two `training.rollout` fields parses and gets the
  documented defaults (§5). Nothing about reading a config changed.
* **Old saved checkpoints do not resume.** A `last.pt` produced before this
  change is refused, and is **not** migrated. There is no silent upgrade path
  and none should be added.

Three independent gates in `resume_from_checkpoint` produce that refusal:

1. `compare_execution_configs(saved_config, current_config)` walks the whole
   resolved config and raises on any key present on one side only —
   `Config mismatch on resume: missing key 'model.candidate_c' in saved config`.
   A saved config that does carry the keys but names a different mode is refused
   the same way, at `model.window_mode`.
2. `compute_config_signature` digests the full resolved config, so the added
   keys change the signature even where a diff would be waved through.
3. The source content signature gate: the checkpoint records the source it was
   produced with, and `_utils.py` / `unified_config.py` / the model package all
   changed, so a pre-change checkpoint fails
   `Source signature mismatch on resume` regardless of its config.

Calibration artifacts inherit the same fate: a threshold calibrated under the
old source and config is bound to that lineage and cannot be re-attached to a
run under the new one. The correct action for an interrupted pre-change run is
to restart it under the new config, not to hand-edit a checkpoint.

`test_old_checkpoints_fail_closed_on_the_changed_schema` asserts gates 1 and 2's
config comparison directly. Gate 3 is not re-tested here; it is pre-existing
behaviour owned elsewhere.

## 9. Schema contract for the M&Ms native YAML

The M&Ms worker owns `configs/self_audit_full_mnms.yaml`. It is validated by the
same `parse_unified_config`, so the matching contract is:

* `model.window_mode` — optional, defaults to `"current"`; one of the six values
  above. Document it explicitly in the file, as the ACDC file now does.
* `model.candidate_c` — optional mapping; the ten keys listed in §2, no others.
  Omitting it, or any subset of it, yields the same documented defaults.
* `training.rollout.predicted_history_exposure` (bool, default `false`) and
  `training.rollout.predicted_history_weight` (float ≥ 0, default `0.1`) —
  optional.
* Nothing else changes: unknown keys anywhere still fail.

An M&Ms config that wants Candidate C sets `model.window_mode: candidate_c` and
nothing else; there is no dataset-specific code path in the switch.

## 10. Trainer contract for the rollout fields

Consumed by the training worker, not by this stream:

* `config.training.rollout.predicted_history_exposure: bool` — default `False`.
  When `True`, the bootstrap interval adds the bounded auxiliary official
  `self_audit` accepted-history rollout annotation loss described in the
  contract; the primary zero-history objective, the A0 strength and the
  130-epoch recipe stay as they are.
* `config.training.rollout.predicted_history_weight: float ≥ 0` — default `0.1`.
  The weight on that auxiliary term only.

If the training worker wants different names, the contract routes that through
the coordinator before this stream edits them.

## 11. Test evidence

Environment: `/private/tmp/self-audit-torch241/bin/python`, Python 3.10.21,
torch 2.4.1, CPU only. Three focused files, one combined run:

```
$ PYTHONPATH=src /private/tmp/self-audit-torch241/bin/python -m pytest \
    tests/test_candidate_c_config.py \
    tests/test_unified_config_contract.py \
    tests/test_checkpoint_binding.py -q
147 passed in 169.32s (0:02:49)
```

That is 68 + 24 + 55 = 147: 68 new tests in `tests/test_candidate_c_config.py`,
plus the 24 pre-existing config-contract tests and the 55 pre-existing
checkpoint-binding tests, both of which cover code this stream touched and both
of which pass unchanged. No full suite was run.

Compile/import smoke, all clean: `self_audit.training.unified_config`,
`self_audit.training._utils`, `self_audit.training.unified_trainer`,
`self_audit.training.finetune_joint`, `self_audit.models.self_audit_net`.

What the 68 tests cover, by section of the file:

1. the exact six-mode surface; every mode parsing from YAML; the contract C
   defaults; strict rejection of unknown keys, non-finite values and
   out-of-range values (13 parametrized cases); partial and null `candidate_c`;
   the historical `model` keys still required and unknown ones still refused;
2. a unified config carrying none of the new keys, and a historical flat config,
   both resolving to `current` with the documented defaults;
3. `filter_model_config` not dropping the switch, and every mode arriving at the
   constructor with the canonical baseline kwargs intact;
4. the two rollout exposure fields, including six strict-validation cases and
   the unchanged `tau` / `max_turns` requirement;
5. canonical and per-mode serialization roundtrips, and the frozen baseline
   recipe;
6. the checkpoint config carrying the mode and every solver setting; an
   execution-mode mismatch and a `candidate_c.lam` solver mismatch each refused
   by the centralized `provenance.verify_model_config` against real tiny nets;
   and old saved configs failing closed on resume;
7. each of the six modes driven through `build_model_from_config` into a real
   (tiny, untrained, `pretrained_encoder: False`) `SelfAuditNet` whose
   `window_mode` and `candidate_c` are read back, plus a cross-check that the
   schema refuses everything `resolve_candidate_c_config` refuses.

## 12. Limits

* **The tests prove delivery, not behaviour.** They show each mode arrives at
  the constructor, is stored on the model, and binds through the centralized
  verifier; they say nothing about what the network then computes. That is the
  core and runtime-review streams' evidence.
* No GPU, no dataset, no training, no ACDC or M&Ms run. No Dice number, no
  effectiveness or novelty claim follows from any of this.
* `provenance.py` was not edited by this stream. Its execution-mode binding had
  landed in the working tree when this revision ran and §6 of the test file
  depends on it; if that work is reverted or reshaped, those tests fail loudly
  rather than falling back to a local check.
* All of this sits on uncommitted working-tree edits from several streams. If
  `models/self_audit_net.py` or `provenance.py` change again, re-run
  `tests/test_candidate_c_config.py` — the schema/runtime bound table in §4 and
  the verifier tests in §6 are the parts most likely to drift.
* Gate 3 of §8a (source signature) is pre-existing behaviour owned elsewhere and
  was read, not re-tested.
* Whether a committed `configs/self_audit_candidate_c.yaml` ablation config
  should exist is the coordinator's call; this stream created no new config file.

## Final integration amendment
Root aligned `min_regress_mass` validation with the runtime probability range and added the upper-bound rejection case. Worker test counts above are historical; final integration evidence is authoritative. Free offsets now have matched maximum per-axis reach 0.92 (= 0.25 + 0.55 + 0.12), while the parameterization differs.
