# Runtime optimization, 2026-10-24

Status: queued after the active A1 arm. These are input-equivalence and CPU cost results, not a measured whole-training GPU speedup or patient accuracy result.

The immutable TRAIN mask cache reduces repeated full-volume ontology checks and foreground scans. The one-step prefetch worker uses a cloned NumPy RNG, prepares only TRAIN inputs/augmentation noise, and hands the live RNG back at each canonical role boundary. It keeps a separate label-read set and never runs a model or touches CUDA. Real L8 CPU input, metadata, augmentation and final RNG hashes match across canonical, cached and prefetched sampling. Timings in the CPU receipts must not be read as end-to-end GPU timings.

The boundary optimizer preserves completed A0/A1 and the original batch24, crop256, 10k updates, 1k warmup and sample exposure. It measures complete CUDA transactions at several thread counts, and enables a cache/prefetch mode only after full model, EMA, optimizer and RNG state parity against canonical training at threads8. Small gains below 3% retain the existing runtime. The paired A3 cost check rejects a regression. Failed cases remain in the result journal. The old queue stays available until a checked continuation child is ready; failure before handoff resumes it.

A5 extraction separately benchmarks 1/2/4/8/12/16 patient workers using the existing canonical frozen-A0 bank builder at slice batch 1. It retains canonical global patient-balanced selection and copies selected vectors without renormalization. All benchmark banks must match the serial selected vectors, occupancy, confidences and endpoint metadata. No external teacher assets or A1 targets are accepted by A5. The full bank still covers 80 TRAIN patients and 2148 timepoints.

The canonical package and train.py source hash are unchanged. Runtime helpers have separate hashes in PLAN.json and enabled training runs write RUNTIME_PERFORMANCE.json; frozen checkpoint receipts include its hash. No AMP, encoder freezing, shortened training, changed objective or new VAL/TEST-based selection is introduced. Validation/test scoring follows the predeclared seven-arm protocol after all training, with full native predictions and BG/RV/MYO/LV metrics.

Actual checks: check_repo PASS; pytest 164 passed, 1 existing FutureWarning (20.67s). GPU timings, state parity, A5 profiles, remaining training and final metrics are still pending. See PLAN.json for hashes and server artifact paths.
