# Unified pipeline: Astra review

Reviewer: Astra Medium. Implementation: AGY through Orca, run `run_ddf8d83f7287`, reviewed in bounded waves. Base `main`: `f1dcae326dffad1cd7e2cf668425e83df4bae936` (refetched before final verification).

## Decision

GREEN for the unified software pipeline. Astra independently reproduced 611 passed, 0 failed (6 warnings) across the full repository suite in 156.22 seconds after AGY corrections. Approved for commit/push to main. This report records software evidence, not a trained-model performance claim.

## Implementation

The canonical entrypoint resolves one strict configuration and constructs one `UnifiedTrainer`, one model, one global epoch timeline and, when enabled, one W&B run. It saves `best.pt`/`last.pt`. Calibration binds selected best weights and verifies lineage; final diagnostics and frozen Proposal-1 bank export use the resolved model/data contract. Incomplete smoke runs cannot certify calibration or completion. Historical interfaces are available under explicit legacy entrypoints.

Global epoch indices are zero-based; report epoch counts are one-based.

| Epoch indices | Trainable modules | Encoder / annotation / auditor base LR | Annotation / audit weights | Rollout and transition population | Batch / augmentation |
|---|---|---|---|---|---|
| 0–99 | Encoder, annotation | 3e-5 / 3e-4 / 0 | 1 / 0; A0–A3 weights 0.5, 0.7, 0.8, 1 | Propagate without audit; no transition population | 4 / yes |
| 100–119 | Auditor | 0 / 0 / 3e-4 | 0 / 1 | Frozen annotation evaluation; adjacent and synthetic transitions | 4 / no |
| 120–129 | All | 1e-6 / 1e-5 / 1e-5 | 1 / 1 | Threshold gate at tau=0; active attempted transitions | 2 / yes |

AdamW, weight decay 1e-4, gradient clip 3, accumulation 1. Optimizer/scheduler reset at 100 and 120, warmup-cosine schedule with 5-epoch warmup and minimum ratio 0 within each interval. Live model weights carry across both boundaries. Best selection begins at global index 120 and maximizes final foreground macro Dice. Calibration sweeps 81 thresholds from -0.02 to 0.02 on the source validation cohort only. Audit stop-gradient and reject→HALT are retained. Canonical configs require pretrained ConvNeXt-Tiny and disallow synthetic fallback.

## Experiment identities and commands

### A: ACDC train/validation

```bash
python scripts/train_self_audit.py --config configs/self_audit_full.yaml --device cuda
```

Add `--wandb` to enable the single run (configuration defaults to logging disabled). Resume uses `--resume weights/self_audit_full/last.pt`; strict source/config/cohort and completed-epoch checks apply.

### B: frozen ACDC checkpoint → external M&Ms test

```bash
python scripts/evaluate_external_mnms.py \
  --config configs/self_audit_acdc_to_mnms.yaml \
  --checkpoint weights/self_audit_full/best.pt \
  --data-root preprocessed_data/mnm --split testing \
  --tau-accept 0.0 --device cuda --output reports/external_mnms.json
```

This command fixes tau before opening external labels. It does not optimize, select checkpoints, or calibrate on M&Ms. Labels support evaluation and the explicitly non-deployable oracle diagnostic only. Incompatible or contradictory checkpoint dataset identities fail; historical checkpoints with no declared training identity remain explicitly uncertified. Tau supplied manually must be fixed from source-only evidence, not selected after viewing external results.

### C: M&Ms-native supervised training

```bash
python scripts/train_self_audit.py --config configs/self_audit_full_mnms.yaml --device cuda
```

Supported through real `MNMSDataset` factory dispatch. Requires paired 3D image/mask data in patient-disjoint `train`/`training` and `val`/`validation` directories; optional `test`/`testing` is separately checked if present. Canonical storage is `[H,W,Z]`, depth axis 2; native validation/calibration uses M&Ms validation only and is a separate experiment/output directory. Filenames must preserve patient identity; arbitrary renaming does not provide authoritative identity. Raw 4D release data needs explicit paired-3D preparation. Missing native splits fail; there is no automatic ACDC substitution or split invention.

Both M&Ms routes enforce BG/RV/MYO/LV model classes and explicit raw mapping `{0:0, 1:3, 2:2, 3:1}`. Unknown, fractional and nonfinite labels fail before integer conversion; `mnm_binary`/`mnms_binary` paths are rejected. No ACDC/M&Ms training union is introduced.

### Frozen Proposal-1 bank

```bash
python scripts/export_transition_bank.py --config configs/self_audit_full.yaml \
  --checkpoint weights/self_audit_full/best.pt \
  --output reports/transition_bank.json --device cuda
```

The default bank is GT-free on-policy generation at configured tau=0; reference masks are used separately for evaluation. It is not silently presented as a calibrated-tau bank. Synthetic counterfactual export remains explicit and provenance-tagged.

## Independently reproduced evidence

- Final integrated gate: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider tests --junitxml=/private/tmp/self-audit-final-tests.xml` → **611 passed, 0 failed, 6 warnings, 156.22s**. Warnings concern test-only scalar conversion, test-only scheduler stepping, and MPS pin-memory support. No assertions were skipped to obtain GREEN.
- Initial integrated wave-3 gate: 604 passed, 7 failed. Review identified a real undefined `audit_cfg` in the audit CLI plus outdated test adapter stubs/documentation assertions; sent back to AGY rather than approving worker-only results.
- Resume probes matched complete shuffled-epoch model state digests at boundaries 1, 100 and 120, including changed loader cardinality at the final boundary. Malformed RNG and altered sampling/cohort metadata were rejected.
- Canonical Bash wrapper: actual ACDC-shaped fixture, one optimizer step, default output/report paths and explicit no-W&B override respected; incomplete status and calibration firewall held.
- Native M&Ms canonical CLI: actual disjoint training/validation fixture, one optimizer step and real checkpoint/report creation passed.
- Unmocked complete CPU curriculum: three synthetic epochs through all states; `best.pt`, `last.pt`, lineage-verified calibration, diagnostics, and frozen bank all passed. Bank contained two on-policy trajectories; reject→HALT observed.
- Frozen external M&Ms CLI consumed that exact ACDC best checkpoint and actual `[32,32,2]` four-class data, executed all four comparison modes and preserved checkpoint SHA-256. External phase metadata was correctly marked unknown.
- Complete-smoke evidence was written under the local temporary directory `self-audit-complete-review-az07i6_z`; no fixture weights or synthetic scores are publication evidence.

## Compatibility and limitations

This changes the canonical interface: old A/B/C flags require the legacy entrypoint. The default internal curriculum preserves the historical single-process Python recipe, including carrying live weights and resetting optimizers. The historical standalone multi-process shell workflow reloads stage checkpoints and is explicitly retained for that reproduction route. The refactor does not establish numerical equivalence to a completed historical 130-epoch training run.

No real ACDC/M&Ms data or CUDA device was available locally, no long retraining was run, and PowerShell execution was unavailable. Synthetic fixtures explicitly disable pretraining and use a small fallback model; this does not weaken production defaults or verify pretrained CUDA performance. Real dataset preparation and full training remain operational prerequisites. No Proposal 2/3 changes are included.
