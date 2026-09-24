"""Method-neutral image-only benchmark foundation.

This package owns neither a splitter nor semantic labels.  It projects a
checked authoritative image-only discovery receipt into the common frozen
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
    SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION,
    SELF_AUDIT_NORMALIZATION_VERSION,
    SELF_AUDIT_SPATIAL_CONTRACT_VERSION,
    SPATIAL_CONTRACT_VERSION,
    SharedSpatialError,
    build_grid_spec,
    grid_hash,
    load_pinned_grid_spec,
    load_self_audit_compat_224_grid_spec,
    load_self_audit_grid_spec,
    read_context_stack,
    read_self_audit_context_stack,
    normalize_self_audit_volume,
    resize_values_to_grid,
)

__all__ = [
    "FirewallError",
    "MANIFEST_SCHEMA_VERSION",
    "SELF_AUDIT_COMPAT_224_SPATIAL_CONTRACT_VERSION",
    "SELF_AUDIT_NORMALIZATION_VERSION",
    "SELF_AUDIT_SPATIAL_CONTRACT_VERSION",
    "SPATIAL_CONTRACT_VERSION",
    "SharedManifestError",
    "SharedSpatialError",
    "build_grid_spec",
    "build_shared_manifest",
    "canonical_json_bytes",
    "grid_hash",
    "load_pinned_grid_spec",
    "load_self_audit_compat_224_grid_spec",
    "load_self_audit_grid_spec",
    "load_shared_manifest",
    "manifest_hash",
    "read_context_stack",
    "read_self_audit_context_stack",
    "normalize_self_audit_volume",
    "resize_values_to_grid",
    "validate_image_only_manifest",
    "validate_manifest",
    "validate_scientific_manifest",
    "write_shared_manifest",
]
