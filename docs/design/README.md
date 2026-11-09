# Portable design sources

This directory records the exact local design inputs used for the A/A-3D
implementation review. `A_CURRENT.tex` is the six-page source document whose
pages `A1.tex`, `A2.tex`, and `A3.tex` define one method A; the other pages are
not separate implementations. `A3D_SPEC.md` is an unchanged snapshot of the
original A-3D implementation/review brief; it is not a claim that every item
was already implemented. The current implementation guide is
`docs/CANDIDATE_A3D.md`, which is updated against the actual code and marks
unfinished checks explicitly.

The source files are snapshots, not experimental results. Their source hashes
were computed on 10 November 2026:

| file | SHA-256 |
|---|---|
| `A_CURRENT.tex` | `8ed515a9f8fafca0da20f104ef0edd5a50077ca205686beaeda856d00218cce2` |
| `A1.tex` | `c8d2f462fd9dc872d6da7b24765f9b0b26a942b6fbd712adff32d1d86cc0201a` |
| `A2.tex` | `3b232d160dc40be80edf50bb9bee5a0ef56849b2342b15b18f9b795762d6085f` |
| `A3.tex` | `d904e2a8934827e423c6465b1bd8147f5ca3cca7bb4dd352427aa0abd87bee86` |
| `A3D_SPEC.md` | `8ae7746896797199dd1ac384934d91b581f72e4e6864615f49556e9ac1ec790b` |

The implementation may evolve after this snapshot; when it does, update the
hash table and the corresponding track docs. A source/design snapshot is not a
checkpoint, a benchmark result or proof of novelty.
