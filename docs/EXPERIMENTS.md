# Component ablations

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
