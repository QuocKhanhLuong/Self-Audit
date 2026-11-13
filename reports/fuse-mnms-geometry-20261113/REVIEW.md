# Independent read-only review

Reviewer: independent Luna subagent `mnms_audit_review`; no source edits and
no tests run by the reviewer. Coordinator ran and preserved software checks.

Final disposition: no remaining findings after fixes and v2 receipt refresh.

Reviewed polar-factor math, corner-displacement bound, known-unit conversion,
explicit unknown-unit policy, handedness, XYZ→ZYX geometry, existing canonical
validator; source/header identity hashes, real open-release partition layout;
L8 selection and lock-before-mask order, path and source SHA guards, mapping.

Resolved findings:
- Bind each patient ID to its source image path and exact filename.
- Pin the actual Table III URL and PDF SHA instead of accepting any document.
- Require all 345 effective geometries before marking GEOMETRY_READY.
- Preserve finite blocked-row receipts for invalid cached geometry; snapshot
  acquisition intentionally fails closed for corrupt/unreadable source headers.
- Pin geometry and per-patient report hashes in SUMMARY; L8 checks both hashes
  and their per-patient canonical geometry agreement before mask access.
- Reject a selected mask path that resolves to another patient, including an
  in-root symlink, before hashing or loading it.
- Refresh executed-source hashes and real receipts after integration changes.

Independent final receipt check found:
- EXECUTION code hashes match the current files. Coordinator separately
  checked those exact hashes on the isolated server checkout.
- SUMMARY hashes match geometry_manifest, per_patient and source_snapshot.
- L8 lock SHA ab239da264645bf82e246b8db6bb55dc5f3314260ef2dcccadc6c9e49bd579f4
  and mask receipt SHA ad3adf4fb0951ed38e6706737341219a530736c701252122a947169b093ead04 verify.
- Selected/read patient IDs match; outside-L8 mask reads are zero.
- Geometry counts are 175/34/136, with no blocked IDs or patient exclusions.

Evidence limits: metadata approximation and dataset-level mm interpretation,
not recovered DICOM geometry. No patient efficacy or target TEST result follows
from these checks. Target runtime manifests and B/B0 execution remain NOT_RUN.
