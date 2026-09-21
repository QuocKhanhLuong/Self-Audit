# Cardiac benchmark v3 freeze (Self-Audit ACDC protocol)

- freeze ID: `cardiac-benchmark-v3-3549ca2a4564c44c`
- scientific payload SHA-256: `3549ca2a4564c44cd8bd1d1859bcd20809e216ad5d0bffe78e46cffb0ccee8b2`
- repository commit: `3483f92cdf398e3af9d6e4f025cef5b12bec4d9f`
- shared manifest SHA-256: `a334cc20bfab90d93fa05579a9c508e419297f20647dbae655156b00669935aa`
- shared grid SHA-256: `57858ddf831decee0ce40c0fcc66f68b794e9c94b8f1eebe0a3a146502e535bf`
- counts: train=1526 / dev=376 / test=0

- normalization: Self-Audit volume 0.5/99.5 clip + population z-score; no CUTS/DFC-specific intensity transform

This is a code/static image-only freeze. Runtime training, DFC optimization, generation/PHATE, and real-data benchmarking are NOT_RUN. M&Ms remains deferred.
