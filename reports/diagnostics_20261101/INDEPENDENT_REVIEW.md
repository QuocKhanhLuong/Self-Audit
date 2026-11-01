# Independent review: context/gradient diagnostic integration

## Scope and evidence boundary

Compared the design brief with the image-only prediction, frozen scoring authorization, feature bridge, gradient arms, and TRAIN-only P fit paths. This review is bound to the source bytes below as inspected on 2026-11-01; the worktree is shared and those hashes can change after this review.

| File | SHA-256 |
|---|---|
| `docs/diagnostics_design/CODEX_PROMPT_SELF_AUDIT_DIAGNOSTICS.md` | `e63ed513c0ec0bef3f1e98ee4573312eb2bdcb43863a7fce73edb2a1e2a61489` |
| `self_audit/diagnostic_protocol.py` | `0612aa1e0552c69a9103d0ff5a42e2f72f65025691489a3654485ef1a1602bf3` |
| `self_audit/diagnostic_models.py` | `ceb38aba2cb62b4587035ec2c9e8429f53a8d94363b7db8c9add9cde5f1686b7` |
| `self_audit/gradient_bridge.py` | `ce68e066f31e362ced75284296119c45fb1f6a08d0ae6ba5093a72a2e94e2a56` |
| `self_audit/bridge_alignment.py` | `497d835023e77fa8215b9aa4b6e5d0ac07674e22f0a951cdc87b7c31aa75f4b5` |
| `scripts/run_context_diagnostic.py` | `8679bcddd963be2bc55d8ab84df66fa874ec4c07e8af03013f0c3eba2cb13947` |
| `self_audit/gradient_bridge_runtime.py` | `9bac26468c661c04db6b715d9795d1c7d97d26a4588eecc3a367908173cafc72` |
| `self_audit/gradient_bridge_metrics.py` | `d25abdd58a73788eacf01c63a49d038433169e0116fe9163d5e152e85c1e2b98` |

## Current defects

### 1. Feature support ignores compact-model right/bottom padding

[`feature_support_from_valid`](../../self_audit/diagnostic_models.py) (lines 112-122) area-resizes the native validity mask directly to the bottleneck shape. `CompactStudentTap.extract` pads the input before encoding (line 73), and `CompactStudent._pad_geometry` pads each spatial dimension to the encoder stride (8 for the four-stage fixture; lines 275-285). The support calculation therefore stretches a native all-valid mask across padded feature cells instead of padding it with invalid zeros first. `predict_stage` passes its native validity mask to this helper (runtime lines 997-1019), so D1-D5 corrections and D3 spatial shuffling can use cells that are not fully supported by the native image.

Bounded CPU witness: for `valid=torch.ones((1,33,35), bool)` and a `[1,1,5,5]` bottleneck (40x40 padded input at stride 8), the current helper marks 25 cells supported. Padding the mask to 40x40 with zeros before area reduction marks only 16 fully native cells (4x4). This follows the helper's documented “fully supported” policy and the already-correct pad-before-reduce pattern in runtime `_fit_valid_support` (lines 719-729). The D0 odd-geometry parity test does not exercise this support mask.

## Reviewed without an additional finding

- `authorize_score` verifies frozen A and B predictions and their lock linkage before writing the exposure record. The context scorer's pre-authorization checks inspect image/prediction payloads; reference-mask payloads are loaded later in the authorized scoring loop. No pre-freeze GT payload read was reproduced in the reviewed path.
- Historical teacher loading uses the isolated historical package alias, verifies its source digest and strict checkpoint payload identity, and the bridge uses the pinned native suffix. `compute_gradient_arms` obtains D1/D2 and crossed D4/D5 VJPs from the common KD objective via autograd; D3 is a support-restricted spatial shuffle and reuses the D2 norm/fallback policy. I found no additional defect in those paths in this fixture review.
- P fitting selects patients with the configured seed and the current runtime passes that same seed into `BridgeAlignmentConfig`, which records it in the artifact budget.
- The paired gate compares D2 against both D0 and D1 and has separate GO, NO_GO, INCONCLUSIVE, and BLOCKED outcomes; D3-D5 and scorer-only native losses do not gate.

## Verification and limits

Ran `/Users/alvinluong/miniforge3/bin/python -m pytest -q tests/test_diagnostic_protocol.py tests/test_diagnostic_models.py tests/test_bridge_alignment.py tests/test_gradient_bridge.py tests/test_gradient_bridge_metrics.py`: **34 passed**. Also ran two bounded synthetic CPU witnesses described above. No patient assets, VAL masks, SSH, server jobs, training, or external assets were accessed; there is no end-to-end patient evidence or scientific/clinical conclusion here.
