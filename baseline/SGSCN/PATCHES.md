# Official-code-derived changes

- Scope all model parameters in an immutable configuration rather than global
  argparse. Preserve tensor operations, architecture, loss math, update order.
- Change hard-coded .cuda() coordinates to the input tensor device; same numeric
  arrays and arithmetic. Preserve signed BN-logit context density and epsilon.
- Decode image bytes from a hash-verified image allowlist into OpenCV BGR.
- Add explicit seeds, environment receipts and repeat diagnostics; no best seed.
- Persist final integer argmax, trajectory and raw seal outside input directories.
- Detect nonfinite loss/gradient/output and fail the sample; do NOT repair it.
- Do not run arbitrary upstream top-level code, create input/result, or emit
  randomly colored predictions as the raw artifact.

Architecture and stopping differences between paper and source are unresolved.
Executable reference profiles are labeled accordingly, not promoted to paper.
