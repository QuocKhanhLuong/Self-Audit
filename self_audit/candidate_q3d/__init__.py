"""Candidate Q3D public schema identifiers.

The 3-D A and A-3D packages retain their own schemas and checkpoint namespaces.
"""

SCHEMA = "self_audit.candidate_q3d.v1"
CHECKPOINT_SCHEMA = "self_audit.candidate_q3d.resume.v2"
EXPORT_SCHEMA = "q3d.student_export.v1"

__all__ = ["SCHEMA", "CHECKPOINT_SCHEMA", "EXPORT_SCHEMA"]
