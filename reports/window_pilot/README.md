# Window pilot verification — 2026-10-23

Implementation base: `8960b6baac4c6600553d900c303f192c2fcfa4f0`.
All receipts refer to runtime source SHA256 `3c7fa85f00d54a6e087af3cd6c7045351fb72d532f1069bf09d95f9cd72359b7`.

- `hypergradient_cpu.json`: cold/warm actual NativeStudent + TinyEncoder center/radius hypergradients match central finite differences; maximum relative error 5.174e-5. Virtual versus actual optimizer maximum absolute parameter error 2.776e-17.
- `baseline_parity.json`: disabled-window final student, EMA teacher and optimizer tensors match the original base exactly after three manufactured steps. Image-generation seed11 and training seed42 are distinct.
- `matched_smoke_cpu.json`: numerical gate ran before three steps per baseline/fixed/meta arm. Identical role schedules, no out-of-budget label reads, three EMA updates each; meta has two geometry updates after one warmup.
- `verification.json`: 81 tests passed; check_repo passed. One existing TorchScript deprecation warning. Receipts include runtime hashes and limitations.

No pretrained weights or patient data were used. No CUDA/long training was run. The finite-difference gate includes a cold virtual step that worsens its manufactured outer Dice loss; negative diagnostics are retained. Numerical correctness is not evidence that the method improves cardiac Dice. Raw manufactured smoke outputs are temporary; the reusable scripts reproduce fresh outputs.

See `docs/WINDOW_PILOT.md` for the method and three opt-in bounded recipes.
