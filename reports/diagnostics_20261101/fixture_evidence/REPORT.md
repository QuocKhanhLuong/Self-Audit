# Context and Gradient Bridge Diagnostic

- Status: FIXTURE_ONLY
- Diagnostic A score: FIXTURE_ONLY
- Diagnostic B score: FIXTURE_ONLY
- A lock SHA-256: 5beedfb9dabe84b102f834275ccb2ae68e7174119595e06751d5bb9638076161
- A prediction freeze SHA-256: 5a265ceb4a50bb904e5140f0f82acd96b977f0f2f5c3db3f955563e10b748fa3
- B lock SHA-256: 6c56d84cc012aca563e89eb69d88a674120e01ac8f8861962af5f0984a4b76c5
- B prediction freeze SHA-256: a706724b84f4a97663f913ed413df525a5d08521f611262e8906c161ee89fb18
- Synthetic fixture: True
- Context results: [A/RESULT.json](A/RESULT.json)
- Gradient results and paired bootstrap/LOO: [B/SCORES_B.json](B/SCORES_B.json)

## Evidence limits

- The development cohort is exposed descriptive evidence, not confirmation.
- Synthetic fixture metrics are never emitted as patient Dice tables.
- Patient, CUDA, clinical, and production success must be reported only from actually completed runs.
