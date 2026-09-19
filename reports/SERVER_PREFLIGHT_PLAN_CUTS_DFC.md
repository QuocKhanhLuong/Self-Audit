# CUTS + DFC Blackwell Server Preflight Plan

Actionable chronological plan for an NVIDIA Blackwell server. Do not assume exact model, driver or CUDA. Preflight output is not production output.

## Server Gate 0 — Pull / context sync

    git fetch origin --prune
    git checkout main
    git pull --ff-only origin main
    git status --short
    git rev-parse HEAD
    git rev-parse origin/main
    git log -1 --oneline
    python scripts/validate_cardiac_benchmark_freeze.py

Require clean HEAD equal to origin/main, freeze ID cardiac-benchmark-v1-79a716b74dd67a76 and payload 79a716b74dd67a76c5a13f0ba4d8876ab28ba6e29d819a4dbcfad75780382b3c. Read handoff/state and stop on mismatch.

## Server Gate 1 — Blackwell environment

    nvidia-smi
    python --version
    python -m pip freeze
    python -c "import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.device_count(),torch.cuda.get_arch_list()); [print(i,torch.cuda.get_device_name(i),torch.cuda.get_device_capability(i),torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())]"
    python -c "import torch; assert torch.cuda.is_available(); a=torch.randn((1024,1024),device='cuda'); b=a@a; torch.cuda.synchronize(); print(float(b[0,0]),torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())"

Persist driver, GPU model, capability, PyTorch, CUDA, Python and pip receipt. Visible GPU is insufficient; allocation/matmul/synchronize/read must pass. Failure is ENVIRONMENT/COMPATIBILITY, not algorithm failure.

## Server Gate 2 — Dependency smoke

    python -c "import torch,numpy; print(torch.__version__,numpy.__version__)"
    python -c "import sklearn,phate; print(sklearn.__version__,phate.__version__)"
    python -m compileall src/shared_benchmark
    python -m compileall baseline/CUTS/src/cardiac_benchmark
    python -m compileall baseline/DFC/src/cardiac_benchmark
    python -m pytest -q --basetemp .pytest-shared tests/shared_benchmark
    python -m pytest -q --basetemp .pytest-cuts baseline/CUTS/tests/cardiac
    python -m pytest -q --basetemp .pytest-dfc baseline/DFC/tests/cardiac

Use fresh counts/times; close DFC 8x8 regression before approval.

## Server Gate 3 — Physical GT isolation

Use image-only mounts such as /data/cardiac/acdc_images_only and /data/cardiac/mnms_images_only. Do not mount GT into baseline process/container.

    findmnt -T /data/cardiac/acdc_images_only
    findmnt -T /data/cardiac/mnms_images_only
    python -c "from pathlib import Path; assert not Path('/data/cardiac/GT').exists(); print('GT mount absent')"

Adapt explicit forbidden path to provider layout; filename filtering is insufficient.

## Server Gate 4 — Real manifests

Use existing builder, never hand-write:

    python scripts/prepare_shared_benchmark_manifest.py --image-root /data/cardiac/acdc_images_only --dataset acdc --output /runs/cardiac_benchmark/manifests/acdc.json
    python scripts/prepare_shared_benchmark_manifest.py --image-root /data/cardiac/mnms_images_only --dataset mnms --output /runs/cardiac_benchmark/manifests/mnms.json

For each record patient/split/frame/sample counts, manifest hash, source manifest logical hash, shared-grid hash and root identity. Validate source hashes/firewall; do not run GT metrics.

## Server Gate 5 — CUTS checkpoint

    test -f /checkpoints/cuts_final_epoch.pt
    sha256sum /checkpoints/cuts_final_epoch.pt

Verify dataset, mode, 200 epochs, final_epoch, manifest/grid/config lineage and random-init S0 lineage. Loading alone is insufficient.

## Server Gate 6 — One-sample preflight

Use current-head CLI options and deterministic list; replace paths with Gate 4/5 values.

    python scripts/run_cuts_scientific.py --manifest /runs/cardiac_benchmark/manifests/acdc.json --image-root /data/cardiac/acdc_images_only --output-root /runs/cardiac_benchmark/preflight/cuts/acdc --checkpoint /checkpoints/cuts_final_epoch.pt --split test --mode 2d --sample-list /runs/cardiac_benchmark/lists/acdc-one.txt --device cuda --num-workers 1 --apply-adapter --semantic-root /runs/cardiac_benchmark/preflight/cuts/acdc-semantic
    python scripts/run_dfc_scientific.py --manifest /runs/cardiac_benchmark/manifests/acdc.json --image-root /data/cardiac/acdc_images_only --output-root /runs/cardiac_benchmark/preflight/dfc/acdc --split test --sample-list /runs/cardiac_benchmark/lists/acdc-one.txt --device cuda --apply-adapter --semantic-root /runs/cardiac_benchmark/preflight/dfc/acdc-semantic

Verify RAW_COMPLETE, raw hash, provenance, runtime/environment/GPU memory, optional adapter and SEMANTIC_COMPLETE.

## Server Gate 7 — Resume test

Rerun Gate-6 commands. CUTS must safely prove changed checkpoint identity cannot reuse old raw preflight artifacts; do not alter production checkpoint.

## Server Gate 8 — Ten-sample smoke

Use manifest order or deterministic list, never manual random selection. Run --limit 10 for CUTS-2D and DFC MinL3. Inspect decoding/boundaries/NaN/Inf/PHATE/DFC/OOM, artifacts, resume, semantic failures, runtimes and VRAM. No full run.

## Server Gate 9 — 50–100 sample preflight

Run deterministic --limit 50 or --limit 100 separately for CUTS-2D and DFC MinL3/dataset. Record success/failure, raw/adapter/total seconds/sample, median and p90/p95 where practical, peak VRAM, environment and retry/resume evidence.

## Server Gate 10 — DFC runtime budget

From real measurements and declared full workload/sample mix/device/concurrency/extrapolation, compute T_base. Set T_budget = 1.25 * T_base and require T_budget <= 72 hours. Never fabricate runtime.

## Server Gate 11 — CUTS cost

Separate Stage-1 training wall time/peak VRAM from generation/PHATE/KMeans seconds/sample/peak VRAM. Keep 200 scientific epochs; do not force 150 or compare only inference against another method full training.

## Server Gate 12 — Full-run decision

Only mark READY_FOR_FULL_RUN after repository/freeze, CUDA kernel, server tests, image-only roots, GT isolation, manifests, checkpoint, 1-sample, resume, 10-sample, 50–100 sample, runtime/VRAM and DFC budget gates pass. Otherwise BLOCKED with exact blocker.

## Output directory plan

    /runs/cardiac_benchmark/
      preflight/{cuts,dfc}/{acdc,mnms}/
      production/{cuts,dfc}/{acdc,mnms}/

Do not reuse preflight artifacts in production unless explicitly permitted; prefer fresh production roots.

