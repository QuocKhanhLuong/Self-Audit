# Component ablations for A and A-3D

The A matrix remains the 2D/2.5D reference. A-3D has its own hash-pinned
recipes and never reuses an A checkpoint. Use `--track candidate_a3d` with the
planner:

```bash
python scripts/candidate_a_ablations.py plan --track candidate_a3d \
  --dataset acdc --train /prepared/acdc/train.json \
  --development /prepared/acdc/development.json \
  --model-config /private/mednext3d.json --out /runs/a3d-plan \
  --seeds 42 43 44
python scripts/candidate_a_ablations.py run \
  --plan /runs/a3d-plan/PLAN.json --execute --device cuda
```

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
  --seeds 42 43 44 --epochs 100 --pretrain-epochs 100 --warmup-epochs 10
python scripts/candidate_a_ablations.py run --plan /runs/acdc-plan/PLAN.json
# Starts actual training only when explicitly requested:
python scripts/candidate_a_ablations.py run --plan /runs/acdc-plan/PLAN.json --execute --device cuda
```

Use `--dataset mnms` with original M&Ms prepared manifests to generate the same
matrix. Use `--arms full teacher_free fixed_queries` for a selected subset. Plans
never choose data splits or label budgets. Changing the label budget requires a
separate prepared manifest. Seeds here are training seeds; they do not silently
resample the labeled cohort. Schedules are reference experiment settings, not
validated convergence recipes. No automatic test-set sweep or model selection
is performed. No interrupted experiment is silently resumed or overwritten.

For A-3D, `configs/ablations/candidate_a3d.json` materializes 21 arms: the 18
A arms that do not change the budget (including decoder capacity), two
native-3D budget controls (2048-bit and 8192-bit, replacing A's invalid 768-bit
arm), and `global_only_queries`. The reference uses
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
