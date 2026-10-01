"""Compatibility wrapper for checkpoints serialized under ``picie.backbone``."""

from modules import backbone as _backbone

for _name, _value in vars(_backbone).items():
    if _name.startswith("__") and _name not in {"__all__"}:
        continue
    globals()[_name] = _value

__all__ = getattr(_backbone, "__all__", [])
