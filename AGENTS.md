<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->


<!-- SEMBLE_START -->
## Semble Code Search

A `semble` MCP server is available with two tools:
- `mcp__semble__search` — search the codebase with a natural-language or code query.
- `mcp__semble__find_related` — find code similar to a specific file and line.

Use `mcp__semble__search` to find where something is implemented — instead of using Grep or Glob to discover files. After semble returns the file and line, navigate there directly and read that file. Do not grep for the same content again.

Pass `content="docs"` to the MCP search tool for documentation and prose, `content="config"` for config files, or `content="all"` for everything. On the CLI, use `--content docs`, `--content config`, or `--content all` instead.

For CLI fallback or sub-agents without MCP access, use:

```bash
semble search "authentication flow" ./my-project --max-snippet-lines 10
semble search "deployment guide" ./my-project --content docs
semble search "database host port" ./my-project --content config
semble find-related src/auth.py 42 ./my-project
semble search "save model to disk" ./my-project --top-k 10
```

### Workflow

1. Call `mcp__semble__search` with a query describing what the code does or its name. The tool returns results with 10 lines of context each (function/class signature + first body lines, enough to confirm the location).
2. Navigate directly to the top result's file and line. Read only the function or class at that location.
3. Make the edit. Do not re-search or grep for the same content.
4. Set the MCP search tool's `content` field to `docs`, `config`, or `all` when searching beyond code.
5. Optionally use `mcp__semble__find_related` with `file_path`, `line`, and the same `content` selection to discover similar code elsewhere.
6. Use Grep only when you need every occurrence of a literal string across the whole repo (e.g., all callers of a renamed function).
<!-- SEMBLE_END -->

@/Users/alvinluong/.codex/RTK.md

--- project-doc ---

# Repository source of truth

Read `ACTIVE_PIPELINE.json` and the complete `docs/CANDIDATE_Q3D_SPEC.md`
before changing the active pipeline. Candidate Q3D is active; Candidate A and
Candidate A-3D remain explicit comparison baselines. `train.py` and
`evaluate.py` are the only canonical top-level entrypoints.

Implement Q3D under `self_audit/candidate_q3d/` and
`configs/candidate_q3d/`. Candidate A-3D may be imported as a dependency, but
files under `self_audit/candidate_a/` and
`self_audit/candidate_a3d/` are frozen baseline sources and must not be
changed. Keep config, checkpoint, geometry, and evaluation contracts separate
for all three tracks.

Q3D uses native `[B,1,D,H,W]` volumes with declared native geometry. Do not
crop or resample. The four stages are T (teacher), Q (tokenizer), P
(physiology cache), and J (student joint). The joint teacher, teacher adapter,
and codebook are frozen; deployment contains the student only. The tokenizer
perplexity/dead-code/class-coverage gate is mandatory. A failed gate stops a
real run. An explicitly synthetic smoke may continue through P/J using the
engine's smoke-only override, but it must report the measured gate result and
the override; the override must be unreachable for real configs.

Real training must fail closed when manifests, pretrained teacher source and
hash, pretraining dataset, or overlap provenance are missing. Do not use U,
development, or test masks as training targets. No patient training, dataset
downloads, dataset deletion, or active-job changes are authorized for this
delivery. Synthetic smoke results are software evidence only and must be
reported separately from real-data and GPU evidence.

Run `python scripts/check_repo.py`, `python -m pytest -q`, and
`python scripts/smoke_q3d.py --out /tmp/self-audit-q3d-smoke` for the
software-only verification workflow. Do not claim patient efficacy, SOTA, or
novelty from synthetic checks or unverified weights.
