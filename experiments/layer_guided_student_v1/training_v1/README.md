# Current training policy

Canonical command: repository-root `python train.py --phase coarse`.
See [CURRENT](../../../docs/CURRENT.md) and [RUNBOOK](../../../docs/RUNBOOK.md).

Implementation preserved from c4973553a743648a04c603c6b4634240933b8b80:
LayerGuidedStudent with approved official DINOv3 ConvNeXt-Tiny; full historical
Q/reliability, positive mass/support transport, direct coarse CE and full-view
stride-16 feature cosine. Ordinary AdamW; no native-detail objective; guard and
refiner off. Exact split detail remains zero. This is not the completed
native-detail or self-learning method. Zero support is not background supervision.

The internal JSON recipes are retained for API/tests. Use root configs/pilot.json
or configs/full50.json from the canonical command; full budgets require explicit
--allow-full-run. Checkpoint source hashes are preserved; old small-CNN checkpoints
cannot load as the new architecture. Exact optimizer resume is not implemented.

Tests using manufactured banks/encoders are software evidence, not official-weight,
GPU or patient results. The next version must explicitly introduce and test a
native-detail policy; never enable it silently under this schema.
