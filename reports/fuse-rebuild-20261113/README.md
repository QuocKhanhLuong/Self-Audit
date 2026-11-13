# Candidate FUSE implementation and launch

Uses labels of all TRAIN80 patients; not a few-label result.

Implementation: isolated `fuse-rebuild` branch from 50a7441939b1d9d05611ff2dc34a2a9b2f6f5a28. Parallel Luna Fast workers, one independent Claude reviewer. Frozen A/A3D/Q3D modules and Q3D configs unchanged.

SOFTWARE_ONLY: 66 focused tests passed in 21.36s; check_repo PASS; native CPU smoke T/J/VAL3/export with all three cold-loads bitwise equal. Separate-process CPU campaign completed T/J/VAL3/EXPORT/TEST_ONCE with the TEST transaction COMMITTED. Exact output: pytest.txt, check_repo.txt, smoke.txt, CPU_CAMPAIGN.md. Full historical repository pytest NOT_RUN under the latest user amendment.

Independent Claude review: PASS for logic/formulas; five Low residuals documented in CLAUDE_REVIEW.md. No blocking findings. Resume after a crash near best-artifact selection requires verifying the selected artifact against the last committed validation; automatic retries are disabled.

PATIENT_DATA_INTEGRITY_CPU: all 98 TRAIN/VAL cached cine volumes equal normalization of the original NIfTI bit for bit (CACHE_RAW_EQUALITY.json). This is data integrity evidence, not efficacy.

P0 server inspection is read-only and historical. GPU profile, switch receipt and launch evidence will be added separately. Patient efficacy, complete VAL/TEST outcomes, deployment decision and inference timings: NOT_RUN at implementation commit.
