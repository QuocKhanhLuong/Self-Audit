"""Method-neutral image-only benchmark foundation.

This package owns neither a splitter nor semantic labels.  It projects the
authoritative FreeMask image-only discovery result into the common frozen
contract consumed by anonymous-partition baselines.
"""

from .firewall import FirewallError, validate_image_only_manifest
from .manifest import (
    MANIFEST_SCHEMA_VERSION,
    SharedManifestError,
    build_shared_manifest,
    canonical_json_bytes,
    load_shared_manifest,
    manifest_hash,
    validate_manifest,
    validate_scientific_manifest,
    write_shared_manifest,
)
from .spatial import (
    SPATIAL_CONTRACT_VERSION,
    SharedSpatialError,
    build_grid_spec,
    grid_hash,
    load_pinned_grid_spec,
    read_context_stack,
    resize_values_to_grid,
)

__all__ = [
    "FirewallError",
    "MANIFEST_SCHEMA_VERSION",
    "SPATIAL_CONTRACT_VERSION",
    "SharedManifestError",
    "SharedSpatialError",
    "build_grid_spec",
    "build_shared_manifest",
    "canonical_json_bytes",
    "grid_hash",
    "load_pinned_grid_spec",
    "load_shared_manifest",
    "manifest_hash",
    "read_context_stack",
    "resize_values_to_grid",
    "validate_image_only_manifest",
    "validate_manifest",
    "validate_scientific_manifest",
    "write_shared_manifest",
]
