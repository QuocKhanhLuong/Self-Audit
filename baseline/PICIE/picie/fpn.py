"""Compatibility wrapper for checkpoints serialized under ``picie.fpn``."""

from modules import fpn as _fpn

for _name, _value in vars(_fpn).items():
    if _name.startswith("__") and _name not in {"__all__"}:
        continue
    globals()[_name] = _value

__all__ = getattr(_fpn, "__all__", [])
