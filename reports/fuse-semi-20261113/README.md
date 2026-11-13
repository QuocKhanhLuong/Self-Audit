# FUSE semi K8 delivery and full-label archive

Track A uses the exact locked ACDC L8: patient007, patient010, patient011,
patient024, patient026, patient041, patient077, patient099. Only their 16 ED/ES
TRAIN masks are exposed; U contains 2,132 frames over all 80 TRAIN patients.
VAL18 and TEST50 manifests, native geometry and all image arrays are unchanged.
The real CPU access audit blocked 144 disallowed mask paths and measured zero
non-L8 TRAIN mask reads. This is data-contract evidence, not model efficacy.

The user explicitly chose 8,000 T frame exposures: 500 L8 epochs with physical
batch 2; J remains 100 epochs/8,000 pairs. The architecture, losses, T-a fraction,
LR groups, dropout schedule and VAL deploy margin 0.01 remain unchanged.
Final Track A VAL/TEST comparison results are NOT_RUN until its campaign completes.

M&Ms Track B and B0 are BLOCKED. All 209 extracted headers have unknown spatial
units, including all 34 staging validation patients. C8J7L5 also fails the audit
orthogonality tolerance 1e-4 (maximum off-diagonal Gram magnitude 0.00011516).
This is a small orientation inconsistency, not proof of corrupt voxel data.
Official complete train/val/test membership and TEST data are not established.
No repair, unit assumption, target L8 selection, class mapping claim or B/B0
training was performed. See mnms/ for every patient's header and findings.

The completed full-label run remains a separate all-TRAIN80-label result.
Its fusion TEST50 FG is 90.825664%; selected on VAL at J epoch 83. The full-label
paired patient analysis uses 10,000 bootstrap resamples seed 42 and two-sided
Wilcoxon; it is exploratory after TEST opening. TEST fusion-minus-teacher mean
is +0.20095 percentage points, 95% CI [-0.04569,+0.41640], Wilcoxon p=0.002049.
Fusion-minus-student mean is +0.05553 pp, 95% CI [-0.55183,+0.58594], p=0.050605.
The tests address different distribution properties; the mean CIs include 0.
These data do not establish that fusion has a reliably larger mean FG Dice.

The fusion checkpoint, matching sidecar and results/provenance bundle are saved
in [Google Drive](https://drive.google.com/drive/folders/1a5Uurg6A-fmofPLMyF-QVqQJIcTqM8B5).
Drive and local/server SHA256 match. Existing server artifacts remain intact.
See DRIVE_RECEIPT.json for IDs, byte counts and verification.

Software checks and launch receipts are appended after verification. The initial
full suite exposed two stale tests expecting Q3D as the default; both failures
were reproduced on unchanged main 5ec4bff. Tests now assert the already-active FUSE
default while preserving explicit Q3D/A/A3D registration checks; baseline source
files were not modified. The repository-copy cleanliness check exercises the
full staged source, including all new tests.

Software acceptance: `python scripts/check_repo.py` PASS; `python -m pytest -q`
reported **874 passed, 2 skipped, 3 subtests passed in 248.99s** with the default
OMP environment. The full-suite repository-copy cleanliness test passed. Focused
new tests:26 passed,3 subtests passed. CPU smoke T→J→VAL3→export passed, with
bitwise cold-load for teacher/student/fusion on three native shapes. These are
SOFTWARE_ONLY results; optional unavailable external-source tests were skipped.
Raw output is retained in pytest.txt, check_repo.txt and SOFTWARE_ONLY_SMOKE.json.

Track A launched on 2026-11-13 at 12:22:28 UTC (19:22 Vietnam) from source commit
`32e528e39b79493b8e2251bd884d7a2ed4583748`, in a new pinned server checkout
`/root/Self-Audit-fuse-semi-20261113` and run directory
`/root/mount_data/runs/fuse-acdc-k8-20261113`. Real GPU T updates and per-epoch
VAL class metrics have been observed. This is RUNNING evidence, not a final
K8 efficacy result. T→J→VAL/deploy/export→one TEST transaction is managed by
`fuse-k8-campaign`; automatic retries are disabled, including on OOM.

The visible Orca window is **FUSE K8 | epoch + VAL per-class**, attached to
tmux `fuse-k8-training`. The server watchdog checks every 420 seconds. Orca alarm
`6e159a89-7381-4c3f-9394-f354f1d32a76` checks every 10 minutes; healthy runs skip
agent notifications, and completion/failure triggers evidence collection and
one report. No monitor pulls or changes the pinned training source. See
LAUNCH_RECEIPT.json for the verified process/GPU snapshot and alarm precheck.
