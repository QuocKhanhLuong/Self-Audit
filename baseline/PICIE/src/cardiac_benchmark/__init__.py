"""S1 SSL out-of-domain cardiac benchmark shell for PICIE.

PICIE uses a ResNet-18 backbone initialized from scratch.
The anonymous partition is produced without cardiac semantic labels.
"""

from .manifest import MANIFEST_SCHEMA_VERSION, load_manifest, require_scientific_manifest

__all__ = ["MANIFEST_SCHEMA_VERSION", "load_manifest", "require_scientific_manifest"]
