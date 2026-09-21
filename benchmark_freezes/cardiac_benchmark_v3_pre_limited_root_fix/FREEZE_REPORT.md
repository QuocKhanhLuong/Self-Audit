# Cardiac benchmark v3 freeze (Self-Audit ACDC protocol)

- freeze ID: `cardiac-benchmark-v3-261d4ae936e817b5`
- scientific payload SHA-256: `261d4ae936e817b50945a612574f6fa9f8c1f5882db35ce827269f46bf920860`
- repository commit: `817f2ba1427506732ce6bb30bc5980e955710533`
- shared manifest SHA-256: `a334cc20bfab90d93fa05579a9c508e419297f20647dbae655156b00669935aa`
- shared grid SHA-256: `57858ddf831decee0ce40c0fcc66f68b794e9c94b8f1eebe0a3a146502e535bf`
- counts: train=1526 / dev=376 / test=0

- normalization: Self-Audit volume 0.5/99.5 clip + population z-score; no CUTS/DFC-specific intensity transform

This is a code/static image-only freeze. Runtime training, DFC optimization, generation/PHATE, and real-data benchmarking are NOT_RUN. M&Ms remains deferred.
