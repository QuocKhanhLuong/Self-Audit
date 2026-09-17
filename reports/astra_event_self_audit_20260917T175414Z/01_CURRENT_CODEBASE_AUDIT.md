# Repository and actual code audit

IMPLEMENTED provenance: fetched canonical origin/main **874c0772c9b00a0fbc5e781ea8e457794160f654**. The user's active local main checkout was **605e62b5e11ce1fc792068813e9ec2b829cefd4f** with unrelated dirty/untracked work; it was not switched or reset. New isolated worktree `/Users/alvinluong/Self-Audit-event`, branch `feat/event-triggered-reference-free-audit`, starts at the fetched main. Prompt-era REW commit 9719e536365d2510f51483cde332fa5f5eeb14ec was not assumed to be HEAD. See evidence/provenance.json and before-diff/status receipts.

No `.codegraph/` exists in this checkout. Semble was used first for source discovery, followed by direct source reads. Documentation was not treated as source truth. Union source findings are separately reviewed; root directly inspected the dynamic operator, REW model/loss and data/training adapter.

## Four existing directions

Canonical supervised Self-Audit (`src/self_audit/`) starts from A0, proposes recurrent annotation corrections, audits transitions and retains/rejects edits. Its audit training targets depend on segmentation GT. The dynamic-window operator is a reusable leaf, not intrinsically tied to those GT-derived targets.

Candidate C is historical counterfactual restitution of an already accepted transition. Its replay, coordinate optimization and preservation/re-audit constraints concern historical repair. Its results do not validate present-time audit invocation or RF utility.

Current supervised REW (`src/self_audit/models/read_evaluate_write.py`) is a shared-writer, three-read candidate search with learned supervised critic, regional KEEP/proposal selection and a final assembled-output audit. This is a useful supervised control. It is not the requested new method.

The mask-free namespace has a different training/inference contract. No modules or results from it certify the new supervised-actor/GT-free-auditor direction.

## Actual REW trace

`ReadEvaluateWriteNet.forward(image)` accepts `[B,3,H,W]`, no GT. Encoder/FPN produces features; initial head gives `[B,4,H,W]`. `OutcomeAuditor.state(features,initial)` is an identity transition. For identical hard predictions the code permits only CC/WW and sets global change to zero. Its learned probability is not true correctness at initialization.

`SharedReadWriter.forward` combines image features, annotation probabilities and entropy. Three proposals use the same pair-message/writer parameters: local has eight fixed compass offsets; near/wide each have four fixed local anchors plus four learned ellipse points. Each read samples feature/probability evidence and writes a gated logit residual. There are no QK dot products in this REW operator. Audit feedback enters its geometry only and is detached.

`OutcomeAuditor.forward` detaches features and both logit inputs, then predicts CC/FIX/REGRESS/WW and signed slice-Dice proxy change. `select_tiles` computes no-grad mean pFIX−λ·pREGRESS, compares with KEEP=min_gain, and gathers exact previous/proposal logits per tile. That local utility is not ΔDice. A separate audit examines the assembled output; detached global change and a hard-label-change check control acceptance. Rejected samples stop later turns.

`compute_rew_losses` uses GT only after forward: retained segmentation loss, initial loss, **all proposals'** segmentation loss, state/transition GT outcomes, read-advantage targets and GT-dependent synthetic repairs. Therefore rejection does not remove proposal segmentation gradients. This loss is explicitly ineligible for the RF Auditor/Trigger. State target A0→A0 has only CC/WW. WW permits wrong→different-wrong; four correctness states discard class-specific and probability information.

The user-reported early B32 fit, 100% KEEP and nonzero gradients are context, not newly reproduced measurements here. The occupied remote training job is observed directly, but this report does not reinterpret its incomplete run as a scientific result.

## Reusable DynamicWindow leaf

`src/self_audit/models/dynamic_window.py::DynamicWindowAttention` uses learned Q/K/V 1×1 projections. Per-query K coordinates are generated from feature/condition maps, turn/iteration embeddings and bounded center/radius/orientation/residual offsets. Bilinear sampling provides keys and values; softmax of query–key dot products aggregates evidence. With a coordinate override, the generator is bypassed exactly; no GT is consulted. This is the right reusable mechanism for the new baseline. Geometry alone is not novel.

The operator returns coordinates `[B,h,w,K,2]` plus attention/geometry metadata. It can clamp points at borders and duplicate effective samples; interventions must measure support utility, not just nonzero geometry gradients. Two vectorized `grid_sample` calls implement a read. No REW candidate search or GT outcome code is reused.

## Data and remote compute

`vast-gpu` resolves to the authorized SSH instance. `/root/Self-Audit` also had HEAD 874c0772c9b00a0fbc5e781ea8e457794160f654. The observed RTX 4060 Ti (16,380 MiB) was occupied by PID17042 running the existing REW B32 job, at approximately 7,870 MiB and 100% utilization. No new GPU process was launched and no active checkout/config was changed.

ACDC is `/root/Self-Audit/data/ACDC`, patient split `/root/Self-Audit/splits/acdc_patient_split_seed42.json`, SHA256 **bc42574017276a6defc668bf533330fd41a64a5e90a75ea1b8958fc9e469265a**. Only annotated train-folder frames/GT pairs for the existing train+validation patients were copied to `/tmp/astra_event_acdc_training`. No official test image/mask contents were opened/copied. Transfer receipt records 400 NIfTI files and archive hash; the first transfer failed on an incorrect manifest key and was corrected before extraction.

The unchanged data adapter uses 2.5D neighboring slices, depth axis 2, volume percentile clipping/z-score, bilinear image resize and nearest-neighbor label resize. The new wrapper requires the existing patient manifest and refuses missing, duplicated, overlapping or unassigned patients. It found 1,526 train slices and 376 validation slices from 80/20 patients. Full-volume evaluation is enforced; there is no foreground-only slice filtering.

Remote host RAM/quota and GPU memory are separate resources. The observed cgroup quota was approximately 30.72 CPUs and host-memory limit about 85 GiB; host CPU percentage cannot select worker count correctly. This turn does not tune remote batch size/workers/checkpointing or infer late-training VRAM from a first batch. Local bounded CPU runs use explicit B8/accum1 and no loader workers.

ACDC attribution required by its dataset: O. Bernard, A. Lalande, C. Zotti, F. Cervenansky et al., *Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?*, IEEE TMI 37(11):2514–2525, 2018, [doi:10.1109/TMI.2018.2837502](https://doi.org/10.1109/TMI.2018.2837502).

Final remote recheck: SSH returned **Connection refused** after local experiments had completed. Current remote training/GPU state is UNKNOWN; the earlier occupied-GPU observation is a timestamped session observation, not a current availability claim. No remote process was stopped or altered. See `evidence/remote_recheck_failure.json`.
