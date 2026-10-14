# Code and documentation publication

The user subsequently requested **push code with documentation to Git**. This commit packages the isolated diagnostics, regression tests, runbook and scalar/source research evidence. The development snapshot was based on `b937e3b746608725e43d3bc3cef8861666d152dc`; the exact publication commit is the Git commit containing this file. Remote availability is verified separately after the push, never inferred from a local commit.

## Start here

- [Execution-host runbook](../../experiments/guidance_signal_20261014/README.md): preflight, original identity gate, TRAIN evidence replay, frozen VAL stagewise diagnosis and conditional bounded pilot.
- [Updated failure attribution](UPDATED_FAILURE_ATTRIBUTION.md) and [next experiment decision](NEXT_EXPERIMENT_DECISION.md).
- [Target signal audit](TARGET_SIGNAL_AUDIT.md), [problem formulation](PROBLEM_FORMULATION.md) and [theory/prior-art proofs](THEORY_PRIOR_ART.md).
- [85-test final software receipt](preflight_local_02/PREFLIGHT.json), [JUnit](preflight_local_02/junit.xml), [test log](preflight_local_02/pytest.log), [code-review limits](CODE_REVIEW.md).

From the repository root, run software preflight in a **fresh** output directory outside frozen artifacts:

```bash
python -B -m experiments.guidance_signal_20261014.preflight --output /absolute/fresh/software_preflight
```

The experimental code imports the hash-pinned published executed-source snapshot under a private package name. Top-level production `src/`, previous run artifacts, configurations, patient splits and thresholds are unchanged. No new full run, GPU measurement or anatomical result is implied by publication. The existing85 tests passed locally before publication and their Python source hashes were rechecked against the final code. Only publication documentation was changed afterward.

The independent Astra code-review follow-up ended with a provider capacity error after delivering defects and mathematical checks. The coordinator verified fixes and final integration; there is no fabricated independent post-fix signoff.

## Historical receipt semantics

Statements such as `LOCAL_UNCOMMITTED_UNPUBLISHED`, `not_uploaded`, or 'no push' in the research/hand-off/execution receipts describe their **creation time before this publication request**. They are preserved as history rather than rewritten to invent a remote event. This publication notice supersedes their current distribution instructions, not their experimental results. The historical HANDOFF_MANIFEST and HANDOFF_RECEIPT describe the original local ZIP bytes; they are not a current Git-tree integrity manifest. The ZIP itself remains local and is excluded from this commit.

Included: code, tests, English docs, scalar CSV/JSON, text mathematical/source checks, the pre-publication code patch, and actual local CPU test receipts (including the superseded in-progress failure). Excluded: patient arrays, masks, weights, credentials, bytecode/caches, and the redundant local ZIP. The reported patient IDs are the already published pseudonymous benchmark IDs.

**Execution status remains unchanged:** original private-bank replay, new-checkpoint real gradients/probabilities, CUDA behavior, new stagewise reference scores and matched trained-arm anatomical scoring are NOT RUN here. A separate Codex on the4070 executes the original-artifact gates. Official TEST stays outside tuning; VAL is already exposed development data. **NO-GO for another50-epoch run** until the discriminating diagnostics justify it.

Publication formatting checks preserve the tested source bytes and historical CSV/report bytes. The raw default whitespace check flags RFC-style CSV CRLF, extra final blank lines, and the verbatim unified-diff context markers. The source/docs check passes with `core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof`, excluding only `GEOMETRY_ATTRIBUTION_SOURCE_DIFF.txt` (an archived source diff). No data or tested code was rewritten merely to remove these formatting differences.
