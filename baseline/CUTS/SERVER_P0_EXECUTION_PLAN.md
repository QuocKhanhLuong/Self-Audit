# CUTS Cardiac Baseline — Scientific P0 Server Execution Plan

## 1. Objective and boundary

This is the server-side runbook for the remaining Scientific P0 work with real ACDC/M&Ms image roots. Its purpose is to validate the environment, data contract, image-only CUTS pipeline, and target-resolution feasibility before authorizing a scientific run.

P0 ends at:

```text
image-only MRI
→ CUTS encoder
→ dense latent map
→ PHATE + K-means K=10
→ raw anonymous partition
```

This plan must not implement or run:

- `cardiac_adapter_v1`, BG/RV/MYO/LV/VOID, or semantic mapping;
- Dice, HD95, ASSD, GT evaluation, or oracle mapping;
- diffusion, K=4, CommonStudent, FreeMask auditor, or predictive challenge;
- AMP, persistent PatchSampler RNG, pooling/downsampling, or encoder changes;
- full 200-epoch scientific training or segmentation results.

Scientific P0 means “ready for scientific runs,” not “segmentation results are available.”

## 2. Required inputs

Before any server operation, confirm the following.

| Input | Requirement |
| --- | --- |
| CUTS repository | `Azios1010/CUTS`, branch `main`, containing the pushed P0 commits. |
| FreeMask repository | `QuocKhanhLuong/Self-Audit` at SHA `96c32b10fc7b8e09b48822e10ae9eb6cc149e253`; use only `src/self_audit_maskfree/`. |
| ACDC/M&Ms data | Readable image roots; do not mount a reference-mask tree into the generation environment. |
| Conda | Ability to create a dedicated `cuts-cardiac` environment; do not alter `specmamba`. |
| Scratch/output space | A location for manifests, smoke artifacts, small checkpoints, latents, and raw partitions; never commit large artifacts. |
| GPU | CUDA-capable device; record model, driver, and VRAM. |

If an input is absent or ambiguous in a way that changes cohort membership or scientific behavior, stop at the relevant gate. Do not replace it with a local fixture or a CUTS-only split.

## 3. Repository audit and source integrity

### 3.1 Mandatory commands

```bash
git fetch origin
git branch --show-current
git rev-parse HEAD
git status --short --branch
git log --oneline -5
git diff --name-only origin/main...HEAD
```

At minimum, the checkout must contain:

```text
b975d41 Add CUTS cardiac P0 benchmark shell
6ee7e85 Add CUTS cardiac server environment candidate
```

### 3.2 Protected CUTS core audit

Verify that no local patch or unreviewed difference exists in:

```text
src/model/CUTS_model.py
src/data_utils/patch_sampler.py
src/utils/losses.py
src/utils/scheduler.py
```

```bash
git diff --exit-code HEAD -- \
  src/model/CUTS_model.py \
  src/data_utils/patch_sampler.py \
  src/utils/losses.py \
  src/utils/scheduler.py
```

If the worktree is dirty or protected core differs unexpectedly, stop. Do not automatically commit, reset, rebase, or stash someone else’s work.

### 3.3 P0 route audit

Check that the primary generation route does not call GT/oracle paths:

```bash
rg -n -i 'label_hint_seg|point_hint_seg|guided_relabel|dice|hungarian|diffusion' \
  src/cardiac_benchmark scripts_cardiac
```

The words `mask` or `label` can occur in a firewall or test fixture in order to reject prohibited inputs. They must not create a production path that reads GT.

## 4. Create the dedicated CUTS environment

### 4.1 Candidate environment

Use the committed candidate:

```text
benchmark/environments/cuts-cardiac-candidate.yaml
```

```bash
conda env create -f benchmark/environments/cuts-cardiac-candidate.yaml
conda activate cuts-cardiac
```

Candidate basis:

```text
Python 3.10
PyTorch 2.1.2 / torchvision 0.16.2 / torchaudio 2.1.2
pytorch-cuda 12.1
NumPy 1.26.4
SciPy, scikit-learn, scikit-image
nibabel, SimpleITK, PyYAML, tqdm, matplotlib, pandas, h5py, tabulate
pip: phate, sewar
```

Do not install by default: `monai`, `einops`, `timm`, `pycocotools`, `iopath`, `albumentations`, `wandb`, `mamba-ssm`, `causal-conv1d`, diffusion packages, or any SpecMamba dependency not imported by CUTS P0.

### 4.2 Audit the resolved environment

After Conda resolves the environment, record installed—not requested—versions:

```bash
python - <<'PY'
import os, platform, torch, numpy, scipy, sklearn, skimage, nibabel, yaml, phate, sewar
import SimpleITK as sitk
print('OS=', platform.platform())
print('Python=', platform.python_version())
print('PyTorch=', torch.__version__)
print('CUDA=', torch.version.cuda)
print('cuDNN=', torch.backends.cudnn.version())
print('GPU=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print('NumPy=', numpy.__version__)
print('SciPy=', scipy.__version__)
print('scikit-learn=', sklearn.__version__)
print('scikit-image=', skimage.__version__)
print('PHATE=', phate.__version__)
print('nibabel=', nibabel.__version__)
print('SimpleITK=', sitk.Version())
print('PyYAML=', yaml.__version__)
print('sewar=', getattr(sewar, '__version__', 'installed'))
print('torch_threads=', torch.get_num_threads())
print('torch_interop_threads=', torch.get_num_interop_threads())
print('OMP_NUM_THREADS=', os.environ.get('OMP_NUM_THREADS'))
print('MKL_NUM_THREADS=', os.environ.get('MKL_NUM_THREADS'))
print('OPENBLAS_NUM_THREADS=', os.environ.get('OPENBLAS_NUM_THREADS'))
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c 'import numpy as n; n.show_config()'
```

Python 3.10/Torch 2.1/CUDA 12.1 differences from the historical CUTS README stack are runtime-compatibility deviations, not scientific method changes, only if the parity gates below pass.

## 5. Compatibility validation and environment freeze

### 5.1 Required checks before freezing

```bash
python -m unittest tests/cardiac/test_p0.py -v
python scripts_cardiac/p0_local_smoke.py \
  --output /scratch/cuts_p0/fixture_smoke
```

Expected:

```text
all unittest checks PASS
gt_opened=false
metrics_written=false
```

Review these gates in the output:

- four same-resolution 5x5 convolutions, with no pooling/downsampling;
- synthetic forward/backward;
- one-step CUTS training-algebra parity;
- PatchSampler parity under seed 42 and preserved per-call reseeding;
- dev `no_grad`, `eval`, and BatchNorm isolation;
- checkpoint/profile mismatch rejection;
- inference latent parity;
- mock-manifest rejection when `scientific_run=true`;
- GT-schema firewall;
- PHATE differential against the official helper;
- seed-1 reproducibility, seed-2 exception-only retry, and no seed 3;
- raw-bundle mutation rejection.

If any test fails, do not freeze the environment and do not run real-data smoke. Investigate the cause; do not loosen tolerances or change protected CUTS core solely to pass a modern runtime.

### 5.2 Freeze artifacts

Only after all compatibility tests pass:

```bash
conda env export --no-builds > benchmark/environments/cuts-cardiac-server.yaml
conda list --explicit > benchmark/environments/cuts-cardiac-explicit.txt
pip freeze > benchmark/environments/cuts-cardiac-pip.txt
sha256sum benchmark/environments/cuts-cardiac-{server.yaml,explicit.txt,pip.txt}
```

Create `benchmark/environments/cuts-cardiac-server.json` containing:

```text
repository SHA
OS, hostname, and runtime class
GPU, driver, and VRAM
CUDA/cuDNN
Python/PyTorch/torchvision/torchaudio
NumPy/SciPy/sklearn/skimage/PHATE/nibabel/SimpleITK/PyYAML/sewar
BLAS implementation and OMP/MKL/OpenBLAS/thread settings
lock-file hashes
P0 test commands, timestamps, return codes, and results
```

Do not call these files frozen if the compatibility tests or fixture smoke failed.

## 6. Image-only data audit

### 6.1 Inspect ACDC and M&Ms

For each root, inspect only images and headers:

```bash
find /data/ACDC -type f \( -name '*.nii' -o -name '*.nii.gz' -o -name '*.npy' -o -name '*.npz' \) | head -100
find /data/MnMs -type f \( -name '*.nii' -o -name '*.nii.gz' -o -name '*.npy' -o -name '*.npz' \) | head -100
```

Required audit record:

| Item | Record |
| --- | --- |
| Actual root | Absolute path, read access, filesystem/mount class. |
| Format | NIfTI/NPY/NPZ; raw or preprocessed. |
| Patient identity | ID rule, aliases, repeat scans. |
| Directory grouping | `training`/`dev`/`test`/`testing`, or no official split. |
| Cine/frame structure | Rank, frame axis, acquired frames, ED/ES-only limitation if applicable. |
| Geometry | Native shape, depth axis, spacing, affine/orientation if available. |
| Image provenance | Source checksum and relative source path. |
| Dataset-global risks | Duplicate source/frame, mixed roots, or unknown crop provenance. |

Do not use the mask tree, annotations, `Info.cfg`, supervised CSVs, GT-derived crops, or label-driven ED/ES selection to decide cohort or sample membership.

### 6.2 Fail-closed conditions

Stop manifest materialization if:

- patient aliases cannot be resolved safely;
- roots mix raw/preprocessed sources with unclear provenance;
- frame inventory is not determinable;
- images cannot be read image-only;
- a patient could cross splits;
- a manifest record includes a GT path/key or cannot resolve to a real image.

## 7. P0.1b — materialize shared scientific manifests

### 7.1 Contract authority

Use only the pinned FreeMask implementation:

```text
Self-Audit SHA: 96c32b10fc7b8e09b48822e10ae9eb6cc149e253
src/self_audit_maskfree/data/discovery.py::assign_splits
src/self_audit_maskfree/data/discovery.py::discover_dataset
src/self_audit_maskfree/data/geometry.py::read_slice_stack
split_seed = 42
```

Do not use `splits/acdc_patient_split_seed42.json` as the final scientific manifest.

### 7.2 Materialize exactly one manifest per dataset

```bash
python scripts_cardiac/build_shared_manifest.py \
  --freemask-root /workspace/Self-Audit \
  --image-root /data/ACDC \
  --dataset acdc \
  --output /benchmark/manifests/acdc_shared_seed42.json

python scripts_cardiac/build_shared_manifest.py \
  --freemask-root /workspace/Self-Audit \
  --image-root /data/MnMs \
  --dataset mnms \
  --output /benchmark/manifests/mnms_shared_seed42.json
```

Each manifest is a scientific runtime artifact outside source control, or within a managed benchmark artifact store. Both FreeMask and CUTS must consume the same manifest path/hash. CUTS must never independently rediscover or split the cohort.

### 7.3 Post-materialization validation

For each manifest, require:

```text
manifest_kind = scientific
frozen = true
split_seed = 42
manifest hash exists and rehashes identically
image roots exist
all source paths exist
no train/dev/test patient overlap
all records for one patient share one split
no fixture/mock marker
no GT path, annotation path, label/mask/foreground-hint key
scientific_run=true accepts the manifest
```

Run the builder a second time, into a scratch output, against the same root/config. Membership and manifest hash must be identical. If not, stop and investigate filesystem ordering, source checksums, headers, or split policy.

### 7.4 Required manifest report

```text
dataset
root
raw/preprocessed status
manifest path
manifest SHA-256
FreeMask source SHA
split_seed
train/dev/test patient counts
train/dev/test sample counts
frame/cine limitation
geometry limitation
```

## 8. Scientific runtime guards and provenance audit

Before real execution, verify the implementation enforces:

```text
scientific_run=true
→ manifest_kind=scientific
→ pinned FreeMask SHA
→ frozen manifest hash
→ image roots and source files exist
→ mock fixtures are rejected
```

Every checkpoint, latent, and raw partition must retain distinct RNG fields:

```text
split_seed = 42
benchmark_seed = 42
model_init_seed = 42
training_rng_seed = 42
loader_seed_policy = deterministic derivation from 42
PatchSampler constructor seed = 42; official per-call reseeding preserved
clustering_seed = 1
clustering_retry_seed = 2
actual_clustering_seed_used
retry_occurred
```

Do not replace this contract with a generic `seed: 42`. Also record CUTS SHA, FreeMask SHA, manifest hash, image checksum, checkpoint hash, latent hash, raw partition hash, config hash, and environment-lock hash.

## 9. Target-resolution resource preflight

### 9.1 Preconditions

Run only after frozen environment and real manifest are available. Read the actual resolution from the manifest and frozen FreeMask data contract; 224×224 is a template, not an assumption.

### 9.2 Matrix

Run independently:

```text
CUTS-2D: [1,H,W]
CUTS-2.5D: [3,H,W]
```

At physical batch sizes:

```text
1, 2, 4, 8, 16
```

First use synthetic tensors at the actual H/W to measure GPU memory. Then use a very small train-only image set to validate image I/O and normalization. Never use test records, GT, or semantic outputs.

### 9.3 Required measurements

```text
peak allocated VRAM
peak reserved VRAM
host RAM / RSS
forward time
backward time
optimizer step time
images/second
CPU SSIM overhead
largest stable physical batch
```

If batch 16 OOMs, preserve the evidence and record the largest stable physical batch. This is a training-profile deviation because CUTS uses BatchNorm; do not add pooling/downsampling and do not claim gradient accumulation is equivalent to a larger physical batch.

## 10. Real-image latent / PHATE / K10 smoke

### 10.1 Sample policy

Use a small, fixed set of train records only, for example one to three records after the manifest has frozen. Log the sample IDs before execution. Do not use dev, test, or GT.

### 10.2 Pipeline

```text
real image-only MRI
→ CUTS image loader
→ frozen/synthetic-compatible CUTS forward
→ latent [L,H,W]
→ flatten [H×W,L]
→ PHATE seed 1
→ K-means K=10 seed 1
→ raw anonymous [H,W]
```

Keep the official PHATE parameters:

```text
n_components=3
knn=100
n_landmark=500
t=2
verbose=false
n_jobs=1 initially
```

Retry policy:

```text
seed 1 success → no retry
seed 1 exception → seed 2 exactly once
seed 2 exception → failure record; no seed 3
```

### 10.3 Measurements and acceptance

Record per sample:

```text
latent export time/file size/hash
PHATE wall time
PHATE peak CPU RAM
K-means wall time
raw partition file size/hash
actual clustering seed/retry/exception
```

Repeat at least one fixed sample in a fresh process. With the same checkpoint/image/config/environment/seed, the raw partition hash must be identical. Do not canonicalize or relabel raw IDs to manufacture reproducibility.

## 11. Bounded real Stage-1 smoke

### 11.1 Scope

Do not run 200 epochs. Run only a few optimizer steps or one tiny bounded epoch using a frozen manifest.

```text
train: small set of train records/patients
dev: small set of dev records/patients
test: zero opened records
GT: zero opened files
```

Create an explicit engineering-smoke configuration outside the scientific configuration, with a small step/epoch budget and `fixture_or_smoke_only` in its artifact identity. Its checkpoint must never be treated as a scientific checkpoint.

### 11.2 Assertions

| Check | PASS condition |
| --- | --- |
| Train isolation | Optimizer sees only train-split sample IDs. |
| Dev isolation | `model.eval()`, `torch.no_grad()`, no optimizer step, no BatchNorm running-stat mutation. |
| Test isolation | Access log contains no test sample or source path. |
| GT firewall | Open log contains no reference or annotation path. |
| Loss | MSE plus NT-Xent, `0.999*recon + 0.001*contrastive`, remains finite. |
| Checkpoint | A checkpoint is written with manifest/config/environment/RNG hashes. |
| Loader | CUTS-2D receives central slice only; CUTS-2.5D receives `[z-1,z,z+1]`, clipped/replicated at edges. |

Do not interpret bounded-smoke loss as a metric or use it to select methods.

## 12. Freeze and handoff criteria

Mark `Scientific P0 ready = YES` only when all of the following pass:

```text
P0.1a protocol/schema
P0.1b ACDC scientific shared manifest
P0.1b M&Ms scientific shared manifest
dedicated cuts-cardiac environment frozen
server P0 tests and fixture smoke
real CUTS-2D and CUTS-2.5D loader smoke
target-resolution preflight
bounded real Stage-1 smoke
real latent export and PHATE/K10 smoke
raw partition fresh-process reproducibility
provenance/freeze/evaluator-skeleton validation
GT firewall, test isolation, and dev isolation
protected CUTS core unchanged
```

If one dataset fails manifest materialization, overall Scientific P0 is `NO`. Per-dataset status may be reported, but the benchmark cannot be called ready.

## 13. Run-log template

Store one record for every gate:

```text
date/time:
operator:
server hostname:
repository SHA:
FreeMask SHA:
environment lock hash:
dataset:
manifest path/hash:
command:
input sample IDs:
GT mounted/opened: yes/no
test records opened: count
result: PASS / FAIL / DEFERRED
artifact paths/hashes:
failure reason and stack trace, if any:
```

## 14. Required final server report

The final report must include:

```text
Repository: branch, HEAD, git status, protected-core modified?
CUTS scientific environment: exact versions, GPU, driver, BLAS/thread settings
Environment parity: test command, pass/fail counts, PHATE/PatchSampler/latent/raw-hash results
Environment freeze: YAML, explicit lock, pip freeze, JSON record, SHA-256 hashes
ACDC: root, raw/preprocessed, manifest/hash, split/sample counts
M&Ms: root, raw/preprocessed, manifest/hash, split/sample counts
Target-resolution preflight: H×W, 2D/2.5D max batch, VRAM, throughput
Real PHATE/K10: sample count, runtime, RAM, reproducibility, retries
Real Stage-1 smoke: train/dev count, optimizer steps, test/GT opens, checkpoint
P0.1a, P0.1b ACDC, P0.1b M&Ms, P0.2–P0.8 statuses
Scientific P0 ready: YES / NO
Remaining blockers
```

Do not proceed to P1 automatically.
