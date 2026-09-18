# Cardiac benchmark v1 freeze

- freeze ID: `cardiac-benchmark-v1-56969c44eba76815`
- scientific payload SHA-256: `56969c44eba76815c911ad9050049c4799f5e5b2cb31c907e9a2fb8f7d427a30`
- repository commit: `f0d61da5ae27832fec75c792d6f07d67dd8c2d50`
- adapter spec SHA-256: `34b1faeb7b40f77c2d4d9789a6e1a342e891fcb8ff8db4edd30957b2ae9d114a`
- synthetic fixture SHA-256: `cdf03cd9f2d312a5156a8ae444b116a1e2231eff8d05f575804a2b398fe6a823`
- shared grid SHA-256: `7c9d33fed0facbbabe736a5216bc599b65f1b465e3e26a3d19c624cf9be57949`

This regenerated pre-data freeze is image-only by contract. ACDC and M&Ms scientific manifests and physical GT isolation remain deferred. The global fixture invariant is enforced before output.

## CUTS repair validation

This replacement binds the corrected CUTS integration. CUTS-2D now selects the
central plane before percentile normalization; CUTS-2.5D retains explicit
three-plane stack normalization. Checkpoint, latent, and raw-partition
provenance use the shared manifest logical SHA, shared-grid hash, canonical
source-image SHA-256, PHATE/KMeans configuration, and raw dtype/shape/hash.
The fixture-only CUTS smoke now materializes the current shared manifest schema
and completes successfully.

- shared benchmark tests: 38 passed;
- CUTS cardiac tests: 11 passed;
- DFC cardiac tests: 8 passed;
- both compileall checks: passed;
- CUTS P0 local smoke: passed;
- freeze validator: passed;
- copied bound-artifact mutation: rejected.

Real ACDC/M&Ms manifests and physical GT isolation remain deferred. No GT was
accessed and no metrics were run.
