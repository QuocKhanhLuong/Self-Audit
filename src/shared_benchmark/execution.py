"""Public execution-layer import surface.

The implementation lives in :mod:`shared_benchmark.artifacts`; this small
module gives runners and downstream tooling a semantically named entry point
without coupling them to either baseline package.
"""
from .artifacts import *  # noqa: F401,F403
