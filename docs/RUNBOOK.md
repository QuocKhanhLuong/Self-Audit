# Runbook: DINOv3 student only

## 1. Use a fresh current worktree

On a machine already running jobs, do not reset its checkout. Fetch the new
main and add a separate worktree at a path that does not exist:

```bash
git fetch origin main
git worktree add --detach ../Self-Audit-current origin/main
cd ../Self-Audit-current
python train.py --describe
python scripts/check_repo.py
```

Leave datasets and original bank/run folders at their existing absolute paths.
No `git clean`, `rm -rf`, checkpoint deletion or process termination is needed.

## 2. Environment and approved artifacts

Use a Python environment with a matching PyTorch/torchvision pair and the
correct CUDA support. `requirements.txt` excludes PyTorch to avoid replacing
a working GPU build. Install it after choosing the PyTorch environment:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Also install dependencies of the reviewed local official DINOv3 checkout.
Record Python, pip freeze, PyTorch, CUDA and GPU information in a fresh receipt
directory. The source is loaded locally; there is no automatic weight download.

Set `BANK` to the ORIGINAL frozen TRAIN bank directory (FROZEN.json and patient
NPZs), not a sanitized report copy. Pin its externally verified SHA256 as
`BANK_SHA`. The existing batch-12 cardinal bank was recorded with digest
`f5a7b51328cc877ebdc4e70cbf7f90bebfe6da672597082beb5ae270f3e98665`;
verify the actual host artifact, never rewrite a manifest to match this example.

`APPROVAL` is an externally reviewed JSON receipt with fields:

```json
{
  "schema": "layer_guided.backbone_approval.v1",
  "approved": true,
  "model": "dinov3_convnext_tiny",
  "lineage": "official_self_supervised_dinov3",
  "repo": "/absolute/path/to/clean/official/dinov3",
  "weights": "/absolute/path/to/approved/weights.pth",
  "weights_sha256": "REPLACE_WITH_REVIEWED_64_HEX_SHA256",
  "source_commit": "REPLACE_WITH_REVIEWED_40_HEX_GIT_COMMIT",
  "review_record": "Record of independent source and weight provenance review"
}
```

Pin the reviewed receipt digest as `APPROVAL_SHA`. Hashing an unreviewed file
is not provenance review. The loader requires a clean official source checkout,
known commit and approved local weights. Missing approval is an error; nobody
should fabricate approval or accept model terms on the user's behalf.

## 3. Bounded new-model pilot

```bash
set -o pipefail
python train.py --phase coarse \
  --bank "$BANK" --bank-manifest-sha256 "$BANK_SHA" \
  --approval "$APPROVAL" --approval-sha256 "$APPROVAL_SHA" \
  --out "$NEW_RUN_DIR" --device cuda 2>&1 | tee "$NEW_LOG_FILE"
```

Do not create NEW_RUN_DIR first. Default is configs/pilot.json, capped at 32
optimizer updates, batch size 1. It may visit only part of the TRAIN cohort.
Inspect STARTUP.json: class LayerGuidedStudent, official dinov3_convnext_tiny,
pretrained_loaded=true, expected hashes and actual parameter count. REPORT.json
must report successful completion, actual updates and semantic updates. No
successful REPORT means incomplete, even if an earlier checkpoint exists.

Check loss/gradient finiteness, actual versus planned patients, checkpoint
reload and peak memory. Semantic CE has a target-entropy floor: inspect KL too.
These checks establish execution, not anatomy or generalization.

## 4. Explicit full budget

After the pilot, largest relevant patient shapes and budget have been reviewed:

```bash
python train.py --phase coarse --allow-full-run \
  --recipe configs/full50.json \
  --bank "$BANK" --bank-manifest-sha256 "$BANK_SHA" \
  --approval "$APPROVAL" --approval-sha256 "$APPROVAL_SHA" \
  --out "$NEW_FULL_RUN_DIR" --device cuda
```

This is still coarse-only training, not native self-learning. Do not change
resolution, target policy, activation, weights, architecture or optimizer under
an existing recipe identity to recover from OOM or poor Dice. Version the change.

## 5. Evaluation and research

Checkpoint/prediction APIs live in training_v1/checkpoint.py. For raw NIfTI,
read through self_audit_guidance.io.read_cine first to match training normalization.
Preserve TZYX versus XYZT, affine and UNKNOWN=255. Keep dense and gated outputs
separate. This cleanup does not add an automatic reference-label evaluation.
Freeze predictions before permitted reference scoring; never use TEST to choose
settings. Use docs/prompts/ASTRA_NEXT.md for the next native-detail implementation.
