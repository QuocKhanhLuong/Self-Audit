"""CRF boundary: never silently substitute a different library/parameterization."""
from shared_benchmark.native_protocol import ProtocolBlocked


def refine(*args, **kwargs):
    raise ProtocolBlocked(["Paper-profile CRF recipe/backend correspondence unresolved; no substitute applied"])
