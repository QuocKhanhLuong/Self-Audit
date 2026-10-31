# B8 reproduction: launched, not completed

This is the historical launch receipt. See [current handoff](STATUS_20261031.md)
for the completed training, verified local payloads, tests and Drive blocker.

Full patient training began at 2026-10-31 01:28:34 UTC (08:28:34 Vietnam).
The first real training row was observed. The 1,000-step warm-up checkpoint
was subsequently written and pseudo-learning began. This is the one authorized
new reproduction; no accuracy result is available at launch.

- Training source: `c06c4c9ae08610e047f1e960712f551690de4067` on
  `QuocKhanhLuong/b8-resumable-20261031`, based on historical `0f54455b`.
- Runner source: `2a1bf35d5313101b42332456ed73aea058ece31a`.
- Plan SHA256: `239c6023c14d058fa667b62ae53fca09a523d2dd739b496a09b1431fff166125`.
- Server run: `/workspace/ema_full_B8_repro_20261031T012656Z_c06c4c9`.
- tmux: `b8-repro-20261031`.
- Drive folder: [ema_full_B8_repro_20261031T012656Z_c06c4c9](https://drive.google.com/drive/folders/101_EUKoSXVDaM-GoWZVzrx-Q1r6RJQ78).

The Drive write/readback probe passed SHA256 and size verification. The folder
does not yet contain final trained weights at this launch receipt. The runner
automatically uploads the full resume checkpoint, raw-student weights and
provenance, reads every uploaded payload byte back, and only then evaluates.
An upload failure retains local weights and records `UPLOAD_PENDING`.

Prelaunch verification: historical patched source passed repository checks and
94 tests on the server. An actual labeled-TRAIN supervised smoke updated encoder,
decoder, detail, fuse and head. An original-vs-patched CPU fixture matched weights,
optimizer, RNG, sampling and losses exactly. A bounded manufactured CUDA resume
completed with correct optimizer-state placement; it was not a claim of bitwise
historical CUDA reproduction. Receipts and the immutable plan are in `server_launch/`.

Monitor without changing the run:

```bash
ssh vast-gpu 'cat /workspace/ema_full_B8_repro_20261031T012656Z_c06c4c9/STATUS.json'
ssh vast-gpu 'tail -n 1 /workspace/ema_full_B8_repro_20261031T012656Z_c06c4c9/training/steps.jsonl'
ssh -t vast-gpu 'tmux attach -t b8-repro-20261031'
```

The later four-arm compact pilot needs a new, explicit teacher identity and
new cache. Historical missing-payload hashes must remain unchanged. A separate
strict deterministic runtime option is being verified for all matched compact
arms, because independent CUDA runs previously differed before the resume
boundary. Teacher runtime/recipe is unchanged. The option uses strict PyTorch
deterministic algorithms, deterministic cuDNN, benchmark disabled and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; it makes no quality or speed claim and must
pass actual CUDA replay before patient compact training.

Installed PyTorch documents CUDA bilinear backward as unsupported under strict
deterministic execution. The opt-in compact path therefore retains the native
bilinear forward and uses ATen's deterministic CPU bilinear derivative, copying
the resulting gradient back to the original device. Architecture, weights,
losses, data and update rules are unchanged. All compact arms pay this runtime
cost, which must be measured. Default training and no-grad inference retain the
native path. CPU forward/backward parity and numerical gradcheck passed; CUDA
forward parity, derivative comparison and exact replay remain mandatory gates.
The backend dispatch is scoped by the training runtime context and restored on
exit; `compact_model.py` remains byte-identical to the existing model definition
so older standalone exports retain their required model-source hash. The
canonical runner uses a separate process per training segment.

At launch: teacher `RUNNING`; remote final weights, quality, compact CUDA replay,
and four-arm patient pilot `NOT_RUN`. No empirical GO/NO_GO has been assigned.
