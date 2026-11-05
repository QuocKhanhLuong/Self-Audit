# Expanded teacher-student research council

Start with the [Vietnamese synthesis](SYNTHESIS_VI.md). This report explores executable region programs, partition-distribution decoders, structured solvers, cardiac dynamics, and alternative teacher interfaces. The methods are research hypotheses, not implemented model improvements or validated novelty.

The 2026-11-05 council used six separate Astra executions, preserved 16 initial ideas, screened 53 distinct discovery candidates and inspected the methods of six selected papers plus archived teacher metric sources. All six research assignments completed and their worker terminals were released. Task success means the dossiers and peer work were delivered, not that the scientific hypotheses succeeded.

| Task | Outcome | Initial ideas | Final dossier | Peer record |
|---|---|---|---|---|
| geometry | SUCCEEDED | [Initial](geometry/initial.md) | [Final](geometry/final.md) | [Review](geometry/peer_review.md) |
| dynamics | SUCCEEDED | [Initial](dynamics/initial.md) | [Final](dynamics/final.md) | [Review](dynamics/peer_review.md) |
| transport | SUCCEEDED | [Initial](transport/initial.md) | [Final](transport/final.md) | [Review](transport/peer_review.md) |
| compilation | SUCCEEDED | [Initial](compilation/initial.md) | [Final](compilation/final.md) | [Review](compilation/peer_review.md) |
| architecture | SUCCEEDED | [Initial](architecture/initial.md) | [Final](architecture/final.md) | [Review](architecture/peer_review.md) |
| teacher | SUCCEEDED | [Initial](teacher/initial.md) | [Final](teacher/final.md) | [Review](teacher/peer_review.md) |

## Evidence and publication scope

- [Source inventory](RETRIEVAL_LEDGER.json) contains citation metadata. [Reading receipt](SOURCE_READING_RECEIPT.json) identifies the inspected methods and hashes of the original extracted texts. Discovery is not source verification.
- [Council provenance](COUNCIL_PROVENANCE.json) records the requested/effective models, dispatch identities and completion states. Geometry's final replies reached the coordinator after two peers had completed; no further peer confirmation is claimed.
- [Research validation](VALIDATION.json) is the unchanged historical receipt from before Git publication. Its `repo_changes_by_this_task=false` applies to the research turn. [Publication validation](PUBLICATION_VALIDATION.json) records checks of this Git bundle.
- [Publication lineage](PUBLICATION_LINEAGE.json) pins unchanged imports and identifies derivatives. [Archive inventory](RESEARCH_ARCHIVE_MANIFEST.json) hashes the original local archive, including files intentionally not distributed here. [Bundle manifest](MANIFEST.json) inventories only the published files.

The authored dossiers, synthetic probe, citations, original cited figure crop and research receipts are included. Raw paper PDFs, extracted full texts, retrieval transcripts, the PDF review screenshot and the machine-specific packaging script remain in local OpenResearch storage. Historical absolute paths inside unchanged reports refer to that original research session; use the relative dossier links above and the direct paper/DOI links for portable access. The teacher XML sources cited in the dossiers are not part of this Git bundle.

No training pipeline, weights, datasets or OpenResearch experiment/job was changed. Patient/GPU training and latency evaluation remain NOT_RUN; novelty remains unresolved. The active pipeline is still the semi-supervised v2 specified by the repository's ACTIVE_PIPELINE.json and docs/METHOD.md.

## Reproduce the synthetic checks

From the repository root, with NumPy installed:

```sh
python reports/teacher-student-expanded-research-20261105/probe_structured_claims.py
```

The script regenerates PROBE_RESULTS.json beside itself. It checks exact partition-law marginals, a coarse-partition representation limit and noncommutativity of projection with a graph solve. These are FIXTURE_ONLY mathematical checks, not patient segmentation evidence. Floating-point values can differ slightly across numerical libraries.
