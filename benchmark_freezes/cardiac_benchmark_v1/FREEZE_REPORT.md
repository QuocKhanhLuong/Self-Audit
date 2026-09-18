# Cardiac benchmark v1 freeze

- freeze ID: `cardiac-benchmark-v1-2200633e7f727d37`
- scientific payload SHA-256: `2200633e7f727d379058c11713308a4fd1dace76e8c573cb7ce4d23dba326391`
- repository commit: `94944e35ebdde5ce89ec2ad98ac792008e6c5d5b`
- adapter spec SHA-256: `34b1faeb7b40f77c2d4d9789a6e1a342e891fcb8ff8db4edd30957b2ae9d114a`
- synthetic fixture SHA-256: `cdf03cd9f2d312a5156a8ae444b116a1e2231eff8d05f575804a2b398fe6a823`
- shared grid SHA-256: `7c9d33fed0facbbabe736a5216bc599b65f1b465e3e26a3d19c624cf9be57949`

This regenerated pre-data freeze is image-only by contract. ACDC and M&Ms scientific manifests and physical GT isolation remain deferred. The global fixture invariant is enforced before output.

## Validation evidence

- all 14 fixture IDs are unique and canonically ordered;
- every expected semantic/validity pixel satisfies `validity == (semantic != VOID)`;
- the corrected high-K fixture is `BVMLLMB` / `1011111` at the affected row;
- shared benchmark tests: 9 passed;
- CUTS cardiac P0 tests: 8 passed;
- DFC cardiac tests: 8 passed;
- `compileall src/shared_benchmark`: passed;
- freeze validator: passed;
- mutation of a copied bound contract artifact: rejected.

This replaces the invalid pre-freeze payload `6feede910fdcb9c9`; no scientific
manifest or real GT was accessed.

The artifact set is intentionally uncommitted.  The repository receipt records
the working tree state observed when this replacement was generated; commit
review is required before treating it as an authoritative scientific freeze.
