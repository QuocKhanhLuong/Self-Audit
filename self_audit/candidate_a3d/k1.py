"""Shared K1 rollout for native 3-D.

There is one algorithm for 2-D and 3-D.  The shared implementation validates
the dimensional geometry and delegates ROI/point alignment, quantization and
packet cost to the bank/codec passed by the caller.
"""

from ..candidate_a.k1 import (
    STOP_ID,
    CandidateTrace,
    K1Config,
    RolloutResult,
    infer_student,
    student_rollout,
)

__all__ = [
    "STOP_ID", "CandidateTrace", "K1Config", "RolloutResult",
    "student_rollout", "infer_student",
]
