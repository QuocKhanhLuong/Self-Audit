# W2.3 interim reviewer notes — required before approval

1. Native draft strict_physical accepts one affine, even malformed 1x1, and then emits
   native-mm scores without validated transform chain / spacing-affine consistency. This
   does not fail closed. Minimal authorized alternative: strict native physical path
   raises/deferred pending real transform provenance; unverified path suppresses native-mm
   metrics and explicitly reports their unavailability. Do not grow a geometry subsystem.
   Remove newly added `verified` boolean API (there was no legacy caller needing it).
   Test one-affine, malformed-affine, inconsistent-spacing attempted bypasses. Shape/spacing
   alone is not proof; simply stamping DEFERRED beside unchanged numeric keys is insufficient.

2. Physical spacing boundary: finite/positive/dimensional checks before empty-mask returns.
   Unknown spacing uses unit pixel grid. Keep aggregation policy unchanged in this task.

3. Actual default DataLoader collates multiple cases with different source H/W, known and
   unknown spacing; per-sample testing alone is insufficient.

4. Preprocessor draft skip check should never silently overwrite an existing image because
   its paired mask is missing (or vice versa) without --no-skip. Fail on partial pair.
   Validate existing image AND mask shapes when skipping; preserve historical metadata.
   When processing a subset/new cases into an existing output directory, do not drop old
   metadata entries for retained arrays. Preserve old entries or fail explicitly before
   writes when safe merging is ambiguous. No historical dataset execution in this task.
   Global metadata target_size must not be rewritten to a new value for retained old arrays
   when current raw input contains only NEW cases (the per-case skip check never runs).
   Validate existing metadata and retained pairs/grid before writing any new arrays;
   reject malformed existing metadata rather than resetting it to an empty dictionary.

These are operational/measurement fixes, not new architecture or scientific validation.

5. Existing test_anisotropic_two_resize_spacing_fixture manually calculates its second
   spacing; it does not test production metadata across the actual preprocessing -> loader
   chain. Add tiny synthetic NIfTI preprocessing -> saved metadata -> ACDC loader resize
   integration with anisotropic non-square input and check effective spacing after both
   actual operations. Keep this synthetic (no raw dataset claim). Remove added trailing
   whitespace reported by git diff --check at preprocess_acdc.py lines77/109/154/162/199.
