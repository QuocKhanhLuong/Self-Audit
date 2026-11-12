# Review disposition — migration to ResEnc

Claude reviews were read-only against concurrent working-tree snapshots. Their raw reports are preserved here; a finding is not a final-state assertion. Final verification is recorded in README.md after the code stops changing. No reviewer trained on patients.

| Review | Findings addressed in the implementation | Remaining evidence boundary |
| --- | --- | --- |
| Teacher | Real constructor requires a loaded encoder; only official SHA receives official HF revision; duplicate encoder aliases must agree; exact high-side padding without a hidden minimum | Strict load and parity do not establish segmentation efficacy |
| Config | Exact pinned SHA/URL, dependency pins and phase/stride lock; encoder aliases classified as loaded; portable checkpoint/receipt configuration | Table D7 overlap declaration is not an independent patient audit |
| Engine | Shared T checkpoint contains teacher and T metrics only; sealed selected quality; optimizer warmup enters source hash; unique parameter groups | T/J patient training NOT_RUN |
| Cleanup | Feature-forward signature, removed recipes, same-class synthetic fixture, export cold-load isolation, fixture contracts | All synthetic metrics remain software evidence |
| Preflight | Empirical ERF determines the phase, skip lattice only informational; receipt or actual probe validated; no declared-phase fallback PASS | Raw header verification, L coverage/physiology, CUDA VRAM NOT_RUN |
| Campaign | Explicit six-run DAG, true labeled-ID nesting, schema/K readiness, receipt type checks, teacher quality beside matching C0 | Missing or legacy manifest inputs remain NOT_RUN; no new patient cohort fabricated |

The external ERF probe was corrected to reproduce the delivered preprocessing (logical center, full stride padding, replicate mode), with all prior variants preserved as NONCALIBRATION. Final mean offsets satisfy the predeclared point-estimate rule. Its near-threshold Y value is reported without altering that rule.
