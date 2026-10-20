# ACDC v2 execution report — incomplete, no patient outcomes

Source: `637f38ced217fe1db2347a87924695533056d8f0` (fetched origin/main,
2026-10-20 UTC). Isolated worktree: `/root/Self-Audit-acdc-v2-run2`.
Training source SHA256:
`3cca8cf4925ab7e98515ee3d731e2d4d0336f72129238d9efdc457c2af9735cf`.
No trainer/model/loss policy was changed. This is semi-supervised training;
audit annotations count toward the same dense TRAIN budget.

## Measured preflight

- NVIDIA RTX PRO 4000 Blackwell, 24,467 MiB total VRAM; driver 595.71.05.
- Python 3.12.15; PyTorch 2.14.1+cu132; CUDA available. Real CUDA matrix
  multiplication succeeded. This is not a patient or decoder-JVP pilot.
- `python scripts/check_repo.py`: PASS.
- `python -m pytest -q`: 43 passed, one PyTorch TorchScript deprecation warning.
- Installed missing requirements using `python -m pip install -r requirements.txt`.
  Full local installation log: `/tmp/self-audit-v2-dependencies.log`.
- No visible training process or allocated GPU process at preflight. GPU utilization
  was 100% despite 2 MiB allocated and no visible compute process; exclusivity is
  not established. Existing tmux sessions/jobs were left running.

## Data evidence and unresolved integrity

Resolved Drive location: `KhanhDrive:Dataset/ACDC`, an extracted hierarchy with
`training/`, `testing/`, and `MANDATORY_CITATION.md` (no archive to unzip).
Destination: `/workspace/self-audit-v2/data/raw/ACDC`.
Transfer uses `--immutable`; existing data are not overwritten. Drive returned
HTTP 403 `RATE_LIMIT_EXCEEDED` for the shared project quota during inspection.
Download progress continues in `/workspace/self-audit-v2/logs/acdc-download.log`.
The remote `testing/` listing/transfer includes `_gt` filenames. Their provenance
and authorization are unverified: no TEST mask arrays are opened or used; the
folder remains inference-only. Do not treat this as a certified no-GT official
release solely from its name.

The inventory receipt is an explicitly partial snapshot, not a complete dataset
certification. Do not prepare caches from this snapshot.

The locked repository split declares 80 TRAIN and 20 VAL, with SHA256
`bc42574017276a6defc668bf533330fd41a64a5e90a75ea1b8958fc9e469265a`.
No split was changed. References in documentation to a reviewed VAL18 split
must not be silently substituted for this actual VAL20 file. Duplicate content
must be reviewed explicitly if the completed image inventory discovers it.

`scripts/verify_acdc_download.py` checks TRAIN/VAL ID coverage, cine SHA256,
image shapes, finite/nonsingular affines, duplicate cine bytes, and exact ED/ES
image-to-cine voxel equality. It opens zero reference mask arrays. It also checks
expected official TEST folder IDs, but does not certify TEST image geometry or
HD95 geometry. A successful `rclone check` is required separately for transport
integrity. `prepare_acdc.py` remains responsible for selected-label alignment and
mask ontology validation. No U mask has been used for selection or training.

## Execution status

**Zero patient training runs, zero scored checkpoints, no patient metrics.**
Official pretrained weights were not found in local workspace/root/tmp searches.
No HF token/token file was present. Weights SHA256 and checkpoint SHA256 are
missing, not zero or fabricated values. The official model source is
https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m .
An approved native PyTorch checkpoint and pinned official source checkout are
still required by `configs/backbone.example.json`. No license was accepted,
no gated access bypassed, and no random fallback used.

The 81 planned configurations cover 4/8/16 patients × subset seeds 42/123/2026 ×
nine methods. Label membership is recorded in `configs/LABEL_SUBSETS.json` using
only locked TRAIN IDs and deterministic hash ranking; subsets are nested.
`EXPERIMENT_STATUS.csv` lists all runs as NOT_RUN and puts L8 first. There is one
fixed training initialization seed (42), not three training seeds. All result
CSVs are header-only until real frozen-checkpoint outcomes exist. No mean/std,
accuracy, training GPU-hours, VRAM profile, acceptance rates, failure overlays,
or medical conclusion can be calculated yet. No attempted patient run failed;
the runs were blocked before launch by missing weights/unverified data.

## Batch and pilot requirements

The current canonical trainer samples one native slice for each supervised,
audit, and U input per transaction. It has no configurable batch size; adding
`batch_size` to its exact config fails validation. Therefore batch 16/24/32/48
has **not** been implemented or benchmarked. Native image sizes differ between
patients; padding/shape bucketing and valid-pixel handling require an explicit
matched protocol and gradient/JVP verification before any large-batch claim.
Do not silently resize images, change denominators, or describe this as a
24-GB optimized batch experiment.

After full data verification and approved weight provisioning, prepare all nine
caches with `prepare_acdc.py`, preserving selected TRAIN labels only; prepare
VAL references through the separately authorized `--evaluation` path into
`development.json`. Record actual labeled frame/slice counts and all manifest
hashes. Run the fixed 32-step head pilot, then a decoder JVP/memory pilot. A
32-step run cannot cover all 72 U patients in L8 at one patient per step; record
that coverage limitation and run a separately declared coverage pilot if needed.
Then execute matched L8 baselines and ablations, freeze/checkpoint-hash before
scoring, and keep official TEST inference-only. Physical HD95 is not currently
implemented by the repository evaluator; no HD95 may be reported without further
certified geometry and implementation checks.

## Local receipts and monitoring

New window `self-audit-v2` in the existing `ssh_tmux` session tails transfer logs:
`tmux select-window -t ssh_tmux:self-audit-v2`.
Raw data, large logs and checkpoints stay outside Git. The compact evidence in
`logs/` contains no credentials, images or masks. The figures directory has no
patient figures because no frozen predictions exist.

ACDC required citation: O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al.,
“Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation
and Diagnosis: Is the Problem Solved?”, IEEE TMI 37(11), 2514–2525 (2018),
doi:10.1109/TMI.2018.2837502.

Data continuation: `/workspace/self-audit-v2/finish-data.sh` is running in tmux
window `acdc-prepare`, waiting for the original transfer. It requires successful
remote `rclone check` and the local image inventory before preparing caches,
starting with L8 seeds 42/123/2026. Only L8_S42 enables the authorized separate
development manifest. It fails closed on copy/check/inventory/preparation errors
or insufficient disk, preserving existing outputs. It launches no training.
Continuation log: `/workspace/self-audit-v2/logs/data-preparation.log`. These are
pending background actions, not completed data preparation evidence.
