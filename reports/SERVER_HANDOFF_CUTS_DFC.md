# CUTS + DFC Server Handoff

## Scope

Only CUTS and DFC. Out of scope: FreeMask, PiCIE, STEGO, MedCLIP-SAMv2, shared split-policy changes.

## Audited repository state

- audited HEAD / origin/main: e737f2dac7dddb318b241ffc8ef1d3805aa9b5e1
- freeze: cardiac-benchmark-v1-79a716b74dd67a76
- scientific payload: 79a716b74dd67a76c5a13f0ba4d8876ab28ba6e29d819a4dbcfad75780382b3c
- shared-grid hash: 7c9d33fed0facbbabe736a5216bc599b65f1b465e3e26a3d19c624cf9be57949
- test counts: shared 52/52 (13.58 s), CUTS 14/14 (11.99 s), DFC 11/11 (10.18 s)
- local verdicts: CUTS READY LOCAL; DFC READY LOCAL.

Real image-only ACDC/M&Ms manifests/firewall are deferred, not PASS.

## Frozen scientific protocol

CUTS: primary CUTS-2D, CUTS-2.5D sensitivity, Stage-1 200 epochs, final-epoch checkpoint, PHATE, KMeans K=10, no GT.

DFC: DFC-Direct-2D-Default-MinL3, per-image fresh model/optimizer/BN, maxIter 1000, MinL3 semantics, no GT.

## Production guarantees

Effective config identity is computed; supplied config hash only compares. CUTS raw cache binds checkpoint content SHA-256 before resume. Raw seals/verifies before cardiac_adapter_v1; semantic binds raw hash. Semantic failure keeps raw valid and CLI exits non-zero. Receipts persist timing, memory and environment observationally.

## Server-only blockers

Real image-only roots, physical GT isolation, Blackwell compatibility, CUTS checkpoint/provenance, real decoding/CUDA/runtime, and DFC T_base/T_budget.

## Mandatory session start

Every new server session must:

1. Read this handoff.
2. Verify Git SHA and worktree.
3. Validate freeze.
4. Verify environment and real CUDA kernel.
5. Verify server preflight state.

Then read SERVER_PREFLIGHT_PLAN_CUTS_DFC.md and SERVER_PREFLIGHT_STATE_CUTS_DFC.md.
