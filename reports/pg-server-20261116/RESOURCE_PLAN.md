# PG server resource plan

## Current evidence

No GPU, SSH session, dataset download, or campaign was used for this delivery.
The real S1 resource profile is therefore **NOT_RUN**; GPU model, VRAM, seconds
per step, and thread recommendation remain unknown. The owned probe has three
focused synthetic CPU checks (frame selection, TRAIN-only guard, and bounded
warmup/measurement schedule); these validate script logic only.

The profile reads only the configured hash-pinned real TRAIN manifest and
image-only U frames. It chooses a largest native frame by voxel count, keeps the
student input in native geometry, uses the existing CineMA preprocessing,
feature backprojection, valid cosine loss and AdamW settings, and tests fresh
seed-42 students with PyTorch thread counts 2, 4, and 8. Each case uses three
warmups and five synchronized CUDA updates. It does not run S1 evaluation,
read VAL18/TEST50, tune batch size, or profile S2/S3/arms. The eventual campaign
must initialize a fresh seed-42 student and must not reuse the probe model.

## Commands

Run the bounded probe on the intended CUDA host after the locked TRAIN manifest
and the two official CineMA files are already available locally:

```bash
python scripts/pg_resource_profile.py \
  --config configs/candidate_pg/pg_k8_base.json \
  --device cuda:0 \
  --out /tmp/pg-s1-resource-profile.json
```

Then start the S1 campaign separately with a fresh model initialization:

```bash
python scripts/pg_campaign.py \
  --config configs/candidate_pg/pg_k8_base.json \
  --out /path/to/pg-run \
  --device cuda:0 \
  --stage S1
```

The JSON receipt records GPU name/total memory, peak allocated and reserved
memory, per-step samples and mean, chosen TRAIN patient/frame/shape, and the
recommended thread count from the fastest measured case. It explicitly records
S2, S3, arms, and batch-size optimization as not measured.
