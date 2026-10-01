# DSS-US independent native implementation

No unlicensed upstream source is copied or run. The pinned reference is for
factual evidence only; DINO is a separate Apache-2.0 backbone dependency.

Phase 1 enumerates 11 CAMUS paper profiles with immutable config indices.
All full profiles currently return BLOCKED_PROTOCOL, before input/model/output
access. Missing scientific details cannot be filled with guessed defaults.

Implemented independent primitives: ImageNet input conversion; published
feature/SSD/MI/positional affinity mathematics; normalized Laplacian; explicitly
parameterized spectral K-means; dataset-level segment clustering; dual feature
combination; pinned/hash-bound last-attention DINO key extraction.

Not yet executable as a full paper producer: exact row preprocessing/morphology/
embedding settings, end-to-end CAMUS orchestration and CRF correspondence.
Step I per-image segment Dice and incomplete Step II matching remain gated.
These are explicit evidence dependencies, not alternate implementations.

~~~
python baseline/DSS_US/scripts/run_native.py --config baseline/DSS_US/config/native/step1_ours_comb.yaml --check-protocol
python -m pytest baseline/DSS_US/tests
~~~

A return code of 2 is the expected unresolved-protocol refusal. Tests use
synthetic parameter choices, a fake hook fixture and the pinned licensed DINO
architecture with a random-weight checkpoint fixture. They are not CAMUS
reproduction, pretrained-backbone validation or clinical inference.

DINO source is pinned by scripts/pin_native_references.py. A real checkpoint must
be separately supplied and SHA256-verified; automatic weight downloads are
disabled. GT is never input to any producer/feature-fitting function.
See SOURCE_LEDGER.md and REIMPLEMENTATION.md for remaining dependencies.
