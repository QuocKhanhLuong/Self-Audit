"""Image-only ACDC / M&Ms data layer for the mask-free pipeline (W3).

Public surface, as frozen in ``reports/maskfree150/architecture_contract.md``:

* :func:`discover_dataset` -- image-only manifest with splits, geometry,
  duplicate collapsing, resolved protocol, readiness and limitations;
* :class:`ImageOnlyDataset` -- frozen unit list yielding ``TrainingUnit``;
* :func:`load_verification_unit` -- sealed ``O_verify`` behind a freeze receipt;
* :func:`load_full_input` -- deployment-only whole-image context.

Nothing in this package opens a segmentation mask or an annotation sidecar; see
``firewall.py`` for the enforced path rules and their limitation.
"""
from __future__ import annotations

from .dataset import (
    FreezeReceiptError,
    ImageOnlyDataset,
    UnitNotFoundError,
    VerificationAccessError,
    build_training_unit,
    load_full_input,
    load_verification_unit,
    validate_freeze_receipt,
)
from .discovery import (
    SCHEMA_VERSION,
    DataRootError,
    MixedStudyGeometryError,
    ProtocolUnavailableError,
    discover_dataset,
    load_manifest,
    save_manifest,
)
from .firewall import MaskAccessError, assert_image_only, forbidden_reason, is_image_only_path
from .geometry import GeometryError, inverse_transform_record, masked_resize, read_geometry
from .partition import (
    BLOCK_SIZE,
    GUARD_BAND,
    PARTITION_SPEC_VERSION,
    PartitionError,
    RolePartition,
    build_partition,
    partition_identity,
)
from .prefetch import (
    DEFAULT_PREFETCH_BATCHES,
    DEFAULT_PREFETCH_MAX_BYTES,
    MAX_ALLOWED_PREFETCH_BATCHES,
    PrefetchBatchIterator,
    iter_batches,
)

__all__ = [
    "BLOCK_SIZE",
    "DEFAULT_PREFETCH_BATCHES",
    "DEFAULT_PREFETCH_MAX_BYTES",
    "DataRootError",
    "FreezeReceiptError",
    "GUARD_BAND",
    "GeometryError",
    "ImageOnlyDataset",
    "MAX_ALLOWED_PREFETCH_BATCHES",
    "MaskAccessError",
    "MixedStudyGeometryError",
    "PARTITION_SPEC_VERSION",
    "PartitionError",
    "PrefetchBatchIterator",
    "ProtocolUnavailableError",
    "RolePartition",
    "SCHEMA_VERSION",
    "UnitNotFoundError",
    "VerificationAccessError",
    "assert_image_only",
    "build_partition",
    "build_training_unit",
    "discover_dataset",
    "forbidden_reason",
    "inverse_transform_record",
    "is_image_only_path",
    "iter_batches",
    "load_full_input",
    "load_manifest",
    "load_verification_unit",
    "masked_resize",
    "partition_identity",
    "read_geometry",
    "save_manifest",
    "validate_freeze_receipt",
]
