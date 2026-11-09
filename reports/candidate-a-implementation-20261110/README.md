# Candidate A implementation validation

This receipt covers software contracts and manufactured CPU data only. It is not a cardiac benchmark result. See [VALIDATION.json](VALIDATION.json) for exact source and synthetic weight hashes, and [the implementation guide](../../docs/CANDIDATE_A.md) for train/evaluate/export commands.

87 tests passed; the existing native JVP test emitted one PyTorch deprecation warning. The source inventory matches the synthetic checkpoint. No patient training, patient evaluation or GPU verification was run. No compatible pretrained teacher weights were available locally.
