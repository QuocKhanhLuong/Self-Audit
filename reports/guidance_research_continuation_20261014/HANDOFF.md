# Historical local handoff to the execution-host Codex

> This records the pre-publication handoff. The subsequent user request authorizes Git publication of code/docs/scalar evidence; see [PUBLICATION.md](PUBLICATION.md) for the current entry point. The ZIP and its manifest remain a separate historical local snapshot, not a checksum of the current Git tree.

Base: `b937e3b746608725e43d3bc3cef8861666d152dc`, branch analyzed `QuocKhanhLuong/guidance-results-research-20261014`. Everything added in this continuation is **local/uncommitted/unpublished**. No new files can be obtained by simply pulling remote main. Use the accompanying local ZIP/patch or an explicitly authorized later publication.

The archive contains only the experimental code, tests, English runbook and scalar/source research evidence. It excludes patient arrays, reference masks, model weights, caches and credentials. `HANDOFF_MANIFEST.json` records each included file's SHA256; the archive's SHA256 is recorded separately in `HANDOFF_RECEIPT.json`. The report-only manifest is not a replacement for original run freezes.

Apply the overlay only to a separate checkout at the pinned base; do not overwrite an existing experiment directory. Read [README.md](../../experiments/guidance_signal_20261014/README.md) before any on-host command. The independent code-only [EXPERIMENTAL_CODE.patch](EXPERIMENTAL_CODE.patch) was checked using `git apply --check` in an empty temporary directory, exit0.

Completed locally: source/provenance and published-metric audit; mathematical checks; isolated Q/h/Z reconstruction implementation; frozen stagewise CLI; strictly bounded paired Q/Z technical-pilot CLI; **85 passing offline CPU tests**, exit0. A transient in-progress API integration failure is preserved and superseded by `preflight_local_02`. Independent Astra review supplied defects and math checks; provider capacity prevented its final post-fix signoff. Final integration acceptance is the coordinator's.

Execution-host order:

1. Run software and original-identity preflight using the actual new run, not sanitized report copies.
2. Reproduce TRAIN R0/Q/reliability/hard labels, persist exact source evidence and positive-class diagnostics. A mismatch stops the pilot.
3. Generate all stages on locked VAL18 with unchanged recipe/teacher; freeze before scoring the original VAL36 references. Keep raw RV separate from accepted RV. VAL is exposed development data; TEST is excluded.
4. Only if those results justify target de-neutralization, create/pin the four-TRAIN-patient, seed42, **64-step-per-arm** plan. Both arms use the same original checkpoint, fresh optimizer and common valid support; only Q→Z changes. No automatic full run.

Still NOT RUN here: original private-bank replay, real checkpoint gradients/probabilities, CUDA deterministic behavior, new stagewise reference scores, trained-arm anatomy comparison and jointly aligned old/new-region errors. Stagewise generation pins the original **new** checkpoint only; pilot output is intentionally private/nonexportable by the original evaluator. Do not claim these remaining comparisons were completed.

Scientific decision: **NO-GO for another50-epoch run**. The evidence shows proxy-fitting progress, strong class/support selection effects and baseline native-grid sensitivity. It does not yet localize the first anatomical error or show that Z contains trustworthy anatomical names.
