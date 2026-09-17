# DFC P0 shared contract

This wrapper consumes a FreeMask-compatible, image-only manifest under policy SHA `96c32b10fc7b8e09b48822e10ae9eb6cc149e253`, never generates an independent split. `split_seed=42`; a patient may appear in exactly one of `train`, `dev`, or `test`.

Each record binds canonical `sample_id`, dataset, patient/study identity, split, frame, central slice, `[z-1,z,z+1]` context indices, source identity/path/hash, stored/native shape and axes, orientation, affine validity, spacing, declared shared grid, and transform. Primary 2D uses only the central source plane: context is provenance only. Source values are image-only; GT/ROI/mask/scribble/metric/reference/checkpoint fields are rejected.

Fixtures have `mock=true`, `fixture=true`, a synthetic-grid declaration, and a separate fixture freeze namespace. A scientific run fails closed unless a non-mock manifest has a valid canonical hash, existing source roots/files with matching hashes, verified shared FreeMask/DFC identity, and scientific qualification metadata. P0.1b remains deferred until real artifacts establish those facts.
