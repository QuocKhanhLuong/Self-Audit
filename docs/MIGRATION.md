# Migration to the Candidate A family

The complete pre-cleanup tree remains in Git at
`f01add12524cf73bb15675ecfeadc1014e420210`. No history was rewritten. Reproduce it
in a separate worktree; do not mix its checkpoints with the epoch-based runtime.

Removed active code: native joint_guard/EMA trainers, JVP/audit control modules,
their experiment planners and mechanism scripts, obsolete config variants,
error-only historical training entrypoints, and tests specific to those removed
algorithms. Shared pretrained-backbone loading, geometry/data validation and
segmentation losses remain because Candidate A uses them. Repeated ablation
recipes are replaced by a base recipe plus explicit factor overrides.

Current schemas are `self_audit.candidate_a.v2` for native 2D/2.5D and
`self_audit.candidate_a3d.v1` for native full-volume 3D. In both tracks,
`epochs` replaces arbitrary update counts. An A epoch covers annotated
frame/slices; an A-3D epoch covers annotated full-volume frames. Track-specific
geometry and checkpoint namespaces are not interchangeable. Native v2 state
import remains an explicit checked initialization path, not an active native-v2
trainer. Source-pinned earlier Candidate A checkpoints must use their recorded
source; this cleanup does not silently waive provenance checks.

Older frozen-Q/CineMA history remains at `archive/pre-semisupervised-20261020`,
commit `924265f7df604d3dca84dda49578b1a0cf28d821`. Historical reports are retained
as historical evidence, not current recipes. Untracked data, checkpoints, reports,
user files and active jobs are outside this cleanup.
