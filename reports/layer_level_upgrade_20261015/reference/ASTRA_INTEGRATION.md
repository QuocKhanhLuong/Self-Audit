# Astra: integrate the layer-level upgrade as a controlled research experiment

Continue Self-Audit. Read `THEORY.md`, `README.md`, all `layer_upgrade/*.py`, and the original frozen run evidence before editing anything. This package supplies implemented operators and proofs; do not replace it with another literature survey or an unqualified recommendation to train longer.

## First verify scope

Refresh repository HEAD. The package was checked against Self-Audit `main@4d58f3aebe4f205b58b3184c8971b1d838109d82`. It does not imply no later changes exist. Preserve historical artifacts. The actual instantiated smoke model was a RANDOM torchvision ConvNeXt-Tiny with a new decoder, 32,890,558 parameters; official DINOv3 checkpoints were not loaded or tested. Obtain only approved backbone weights, record verified checkpoint/source hashes, and disclose pretraining lineage.

Implement in a separate opt-in experimental path. Do not push, merge or launch a full run without explicit permission. The user is asking to upgrade the architecture mathematically, not to silently rewrite baseline provenance.

## Inspect and independently challenge the proofs

- Verify block-average D, repeat U, DU=I, zero-mean detail and direct gradient separation. Explicitly test that softmax and pooling DO NOT commute. Check odd-shape padding, cropping and teacher-grid transformations.
- Preserve the input adapter's repeated-center function at initialization while allowing difference-convolution weights to receive gradients. Do not use an untracked MRI normalization or invent metadata.
- Verify parameter-space gradient projection, actual finite-step loss checks and full rollback on rejection. Do NOT feed a projected raw gradient through AdamW and claim protected finite loss. If integrating AdamW, implement actual-displacement checks plus restoration of both model and optimizer state in a separately tested variant.
- Verify frozen-energy gradient, weighted-degree Lipschitz bound and mirror-descent inequality. Preserve frozen unaries/anchors/edges within a solve. Test a case where energy falls but anatomical labels become wrong. The certificate is for proxy energy, not correctness.
- The conditional correction theorem assumes the image unary favors the TRUE class with enough margin. Do not pretend this assumption is observable without a validated model of the evidence.

## Make it compatible before changing research factors

Add an explicit new-model constructor/loader. The old tiny GuidanceStudent checkpoint cannot initialize this full architecture. Verify raw native DINO ConvNeXt stage shapes; bypass its optional resized intermediate grid. Require explicit approved local backbone weights, never an accidental random fallback.

Keep teacher runtime separate from student inference. No teacher feature concatenation into the forward path that is unavailable at deployment. Cache teacher features with physical/image coordinate provenance. Transport evidence mass and support consistently, not integer labels or unvalidated confidence scores.

Retain actual source distinctions: fixed anatomical teacher targets, conditional Z, partial semantic sets, image-derived affinities, and EMA predictions are not interchangeable truth sources. Use named/set losses only where the evidence justifies the ontology. Missing orientation does not authorize RV/LV labels or permit assuming a side.

## Minimal experiment sequence

1. Build the approved strong encoder and ordinary residual decoder; verify native forward/backward and peak memory before training. Then compare the split bounded-detail head with a matched conventional head, same backbone, sources, schedule and compute accounting.
2. Add a selected-decoder-block guard as a separate factor. Define class/source witness batches from TRAIN before running and sample all selected patients. Report each guard loss, actual update, rejections/stalls, source correctness at evaluation and overhead. A protected wrong RV source is not improvement.
3. Add the image-energy refiner only after its image unaries are explicitly justified. First use it as a detached proposal generator, not an always-trusted pseudo-label updater. Freeze proposed outputs before evaluation and report edit-specific benefit/harm, not only reduced energy.
4. Keep representation self-learning active on samples without named evidence using the provided masked-feature interface; ensure the student really sees a masked view and the target/mask are correctly aligned. Do not create an oversized predictor that bypasses segmentation features.

Do not stack all three changes in the first pilot. A 64-step run on a few patients is a mechanism check, not a generalization test. No semantic threshold tuning against the exposed VAL and no TEST-fitting. Record patient coverage, optimizer steps and initialization. If no real data/checkpoint is accessible, run the exact local tests and report the missing integration checks honestly; do not invent patient improvements.

## Deliver

- Opt-in architecture code and artifact-compatible recipe/loader changes.
- Independent theorem review with any corrected assumptions.
- Geometry/gradient/optimizer regression receipts.
- Matched small-run report ONLY if authorized and inputs exist.
- A concise Vietnamese decision brief separating operator validity, execution correctness, anatomical evidence and remaining research hypotheses.

The central claim to test is: a strong dense student benefits from an explicit contract for what a coarse teacher may constrain, which grounded class proxies an update may sacrifice, and when image-derived corrections may become supervision. Generic pyramids, gradient projection and mirror descent are prior art, not automatically novel contributions.
