"""Opt-in native-volume EMA pilot.

Importing this package has no side effects and never touches the historical
``self_audit`` trainer.  Use :func:`validate_volume_config` before constructing
models or data loaders.
"""

from .config import SCHEMA, config_sha256, default_volume_config, load_volume_config, validate_volume_config
from .engine import VolumeTrainer, train_volume
from .checkpoint import load_checkpoint, load_volume_checkpoint, save_checkpoint, save_volume_checkpoint, source_sha256
from .evaluation import evaluate_frozen, export_predictions, export_student
from .legacy_compat import current_execution_source_sha256, load_legacy_checkpoint, verify_legacy_manifest

__all__ = [
    "SCHEMA", "VolumeTrainer", "config_sha256", "default_volume_config", "evaluate_frozen",
    "export_predictions", "export_student", "load_checkpoint", "load_volume_checkpoint", "load_volume_config",
    "save_checkpoint", "save_volume_checkpoint", "source_sha256", "current_execution_source_sha256",
    "load_legacy_checkpoint", "verify_legacy_manifest", "train_volume", "validate_volume_config",
]
