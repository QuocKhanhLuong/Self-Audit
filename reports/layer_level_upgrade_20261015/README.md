# Layer/function-level upgrade: published reference and Astra handoff

Start with [the exact Astra prompt](../../docs/prompts/ASTRA_LAYER_UPGRADE_EXACT.md) and [implementation contract](../../docs/research/layer_level_upgrade_20261015/EXACT_SPEC.md). This bundle preserves the complete prototype previously delivered as `SelfAudit_LayerLevel_Upgrade_20261015.zip` and adds instructions for faithful reimplementation, not an unreviewed replacement of the production student.

## Contents

- [reference/THEORY.md](reference/THEORY.md): propositions, proofs, counterexamples and limitations.
- [reference/layer_upgrade/](reference/layer_upgrade/): six Python modules covering model, input/projection layers, losses, selected-parameter update guard and image-energy refinement.
- [reference/ASTRA_INTEGRATION.md](reference/ASTRA_INTEGRATION.md): original integration discussion, retained verbatim.
- [reference/tests/test_operators.py.txt](reference/tests/test_operators.py.txt): original 47-test suite; the transport suffix prevents root pytest from collecting an archival copy.
- Original manifest, CPU receipts and probe scripts; [publication verification](PUBLICATION.json) records the new delivery separately.

All 19 reference file contents are unchanged from the supplied archive. The original `pushed_to_git=false` and no-push statements describe the earlier handoff; they are historical, not the publication status of this branch. The original manifest still uses the original test filename, which the materializer restores. No images, masks, model weights, credentials, or third-party source files are included.

## Reproduce safely

From the repository root, use a compatible environment and a NEW directory outside the checkout:

```bash
python scripts/materialize_layer_upgrade_reference.py \
  --out /tmp/self-audit-layer-reference-unique --run-checks
```

The materializer verifies the pinned manifest and every payload before copying; it refuses an existing or in-repository destination. Results go to the new `results/` directory, not into this reference. Do not run the original probe scripts directly in the archived folder because they write JSON files.

Publication checks reran 47 CPU tests successfully and the random full-size model forward/backward at 32,890,558 parameters. They did not load official DINOv3 weights, read patient arrays, run ACDC training or measure generalization. Historical receipts remain separate from the new verification record.

## What Astra should do

First reproduce the exact functions in a new opt-in `experiments/layer_guided_student_v1/` implementation and demonstrate output/gradient/side-effect parity. Then add explicit model/config/loader and geometry contracts without changing the reference mathematics. Future deviations, real-data training, push or main merge require their own disclosed scope. Do not substitute a generic U-Net, an AdamW gradient hook or an unvalidated learned refiner and call it the same implementation.

This branch adds only the reference, documents and a verification/copy helper. Production `src/`, active checkpoints, bank artifacts, existing experiment code and previous reports remain untouched.
