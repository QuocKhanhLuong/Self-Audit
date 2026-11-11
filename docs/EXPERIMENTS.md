# Component ablations for A and A-3D

The A matrix remains the 2D/2.5D reference. A-3D has its own hash-pinned
recipes and never reuses an A checkpoint. The primary A-3D comparison is one
seed with four student runs. It is deliberately small enough to answer the
main contribution question before spending compute on secondary sweeps:

The planner defaults to seed 42. For `--track candidate_a3d`, omitting
`--arms` selects `full`, `teacher_free`, `fixed_queries` and
`global_only_queries`; the command below spells them out for clarity. Existing
plans retain their recorded job lists, so create a new plan to use these defaults.

```bash
DATASET=acdc
DATA=/prepared/$DATASET
MODEL=/private/mednext3d.json
PLAN=/runs/${DATASET}-a3d-main-s42

python scripts/candidate_a_ablations.py plan --track candidate_a3d \
  --dataset "$DATASET" --train "$DATA/train.json" \
  --development "$DATA/development.json" --model-config "$MODEL" \
  --out "$PLAN" --seeds 42 \
  --arms full teacher_free fixed_queries global_only_queries \
  --epochs 100 --pretrain-epochs 100 --warmup-epochs 10
python scripts/candidate_a_ablations.py run \
  --plan "$PLAN/PLAN.json" --execute --device cuda
```

Use `DATASET=mnms` and a new `PLAN` directory for original M&Ms. This creates
six jobs: shared `s42_pretrain` and `s42_warmup`, followed by
`s42_full`, `s42_teacher_free`, `s42_fixed_queries`, and
`s42_global_only_queries`. The four student runs share the same labeled TRAIN
cohort, seed, schedule, initialization, bank and packet budget. Read the
student metrics from each `$PLAN/runs/s42_*/metrics.csv` and compare development
metrics only after the frozen prediction artifacts are written.

The four primary arms isolate these effects:

| Arm | What it removes or changes | Interpretation |
|---|---|---|
| `full` | Complete A-3D method | Reference |
| `teacher_free` | No teacher proposals or teacher checkpoint | Contribution of teacher-guided query discovery; TRAIN-label utility supervision remains |
| `fixed_queries` | Learned K1 policy disabled; fixed feasible IDs are used | Contribution of adaptive query selection/K1 as a package |
| `global_only_queries` | `local_queries=false` | Contribution of local typed query representation |

This compact set does not establish every original research claim. In
particular, it is not a U-Net-versus-new-student comparison and it does not
test conventional feature-matching KD. Those require separately implemented,
matched controls. The 21-arm catalogue below remains available as an explicit
secondary sweep; it is not the primary ablation run.
Report these as single-seed comparisons; they do not estimate variation across
training seeds.

For the expanded label-budget campaign, see [FEWSHOT.md](FEWSHOT.md). It uses
the nested `k=2,5,8,10` TRAIN-patient sweep at seed 42 and all 21 A-3D arms;
the four `full` runs are prioritized before interpreting the remaining arms.
This campaign is separate from the compact four-arm contribution comparison
above.

For a plan-only validation, omit `--execute`. Every output directory must be
new; the runner refuses to overwrite a previous or partial experiment.

The A-3D reference factor is `local_queries=true`; its global-only control is
`local_queries=false`. Other named controls keep the same student, query bank,
epoch schedule, label budget, seed and serialized-bit budget within each seed
block: teacher-free, teacher-frozen, REGION-only, BOUNDARY-only, fixed/random
selection, no exploration, no bit penalty, no measurement supervision, R0-only,
value precision and decoder capacity. A teacher backend comparison may include
RefineFormer3D only after an independently hash-pinned adapter and checkpoint
have been verified. The config must otherwise fail closed instead of inventing
an adapter.

For a real-data `teacher_free`-only plan, `--model-config` may be omitted: this
control explicitly does not instantiate or fall back to a teacher. Any plan
containing a teacher arm still requires the local, verified backbone/checkpoint
config; the placeholder MedNeXt fields in `full.json` are never a random-weight
or unverified-pretraining fallback.

`configs/ablations/candidate_a.json` defines factor overrides on one canonical
recipe. The planner materializes exact standalone configs and pins hashes, so
there are no silently drifting copies of the same full configuration.

| Group | Arms |
|---|---|
| Teacher | full, teacher_free, teacher_frozen, teacher_region_proposals, teacher_boundary_proposals |
| K1 | fixed_queries, random_queries, no_exploration, no_bit_penalty |
| Measurement bank | region_only, boundary_only |
| Measurement supervision | no_region_supervision, no_boundary_supervision, no_measurement_supervision |
| Code | r0_only, values_4bit, values_16bit, budget_768bit |
| Capacity | decoder_half_width |

The teacher-free learned arm retains TRAIN-label utility supervision and student/
random proposals. Fixed/random query controls disable K1 utility learning. Teacher
branch arms change proposal sources, not the entire teacher architecture. The
capacity arm measures decoder capacity, not a new architecture. Removing the
mandatory code-only information boundary or replacing code with unrestricted
floats is not treated as a valid Candidate A ablation.

Within each seed block, share the exact label budget, data split, epoch traversal,
augmentation seed, initialization for common-shaped modules and teacher warm-up
checkpoint. Full uses both query types, 8-bit value precision and 1024 packet bits.
Other arms retain these settings unless named as the changed factor. Report actual
bytes/query counts and measured time: equal maximum budgets do not imply equal
realized rate or compute. Four-bit levels still occupy one byte in the reference
packet format; that arm changes precision, not the per-value storage byte count.

```bash
python scripts/candidate_a_ablations.py plan --dataset acdc \
  --train /prepared/acdc/train.json --development /prepared/acdc/development.json \
  --model-config /private/backbone.json --out /runs/acdc-plan \
  --seeds 42 --epochs 100 --pretrain-epochs 100 --warmup-epochs 10
python scripts/candidate_a_ablations.py run --plan /runs/acdc-plan/PLAN.json
# Starts actual training only when explicitly requested:
python scripts/candidate_a_ablations.py run --plan /runs/acdc-plan/PLAN.json --execute --device cuda
```

Use `--dataset mnms` with original M&Ms prepared manifests to generate the same
matrix. For the compact A-3D comparison, pass
`--arms full teacher_free fixed_queries global_only_queries` explicitly. Plans
never choose data splits or label budgets. Changing the label budget requires a
separate prepared manifest. Seeds here are training seeds; they do not silently
resample the labeled cohort. Schedules are reference experiment settings, not
validated convergence recipes. No automatic test-set sweep or model selection
is performed. No interrupted experiment is silently resumed or overwritten.

For secondary analysis, `configs/ablations/candidate_a3d.json` retains 21
explicit arms: the 18 A arms that do not change the budget (including decoder
capacity), two native-3D budget controls (2048-bit and 8192-bit, replacing A's
invalid 768-bit arm), and `global_only_queries`. The primary server run selects
only the four arms listed above with `--arms`; the remaining arms are diagnostic
follow-ups. The reference uses
`local_queries=true`, a 4096-bit packet budget, and a 3-D bank with 2x2x2 REGION
and BOUNDARY grids. All budgets include the fixed affine-bearing packet header
and R0; a budget that cannot fit R0 is rejected during planning. The optional
TRAIN-GT code-capacity diagnostic runs separately with
`scripts/diagnose_candidate_a3d.py`; it uses quantized TRAIN labels as privileged
values and is reported only as a representation diagnostic, never as deployment
or patient efficacy.

The A reference uses a 1024-bit packet budget. A-3D uses 4096 bits; its budget
arms are 2048 and 8192 bits. Actual bytes, query counts and measured time remain
the reported quantities, since equal maximum budgets do not imply equal rates.
