"""Explicit native-volume recipe; 2-D settings cannot silently become 3-D.

The optional ``semi_supervised`` block is A-3D-specific.  It is normalized
here, after the shared strict recipe validation, so existing labeled-only
configs remain valid while new configs cannot hide misspelled consistency
controls in an unvalidated nested mapping.
"""

import math
from copy import deepcopy

from self_audit.candidate_a.config import validate_training_fields
from self_audit.candidate_a.k1 import K1Config

from . import SCHEMA

_SEMI_DEFAULTS = {
    "enabled": False,
    "ema_decay": 0.99,
    "teacher_weight": 1.0,
    "student_weight": 1.0,
    "rampup_epochs": 10,
    "confidence_threshold": 0.7,
    "background_weight": 0.1,
    "steps_per_epoch": 0,
    "intensity_gamma": [0.7, 1.5],
    "intensity_noise_std": 0.03,
}

# Public immutable-by-convention reference for trainer code.  Callers should
# copy it before changing values; ``validate_config`` always returns a fresh
# nested mapping.
DEFAULT_SEMI_SUPERVISED_CONFIG = deepcopy(_SEMI_DEFAULTS)


def _validate_semi_supervised(value):
    if value is None:
        value = _SEMI_DEFAULTS
    if not isinstance(value, dict) or set(value) != set(_SEMI_DEFAULTS):
        raise ValueError("semi_supervised requires the exact declared fields")
    out = deepcopy(value)
    if type(out["enabled"]) is not bool:
        raise ValueError("semi_supervised.enabled must be boolean")
    for key in ("ema_decay", "teacher_weight", "student_weight",
                "confidence_threshold", "background_weight",
                "intensity_noise_std"):
        item = out[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"semi_supervised.{key} must be finite numeric")
    if not 0.0 < float(out["ema_decay"]) < 1.0:
        raise ValueError("semi_supervised.ema_decay must be in (0,1)")
    if float(out["teacher_weight"]) < 0.0 or float(out["student_weight"]) < 0.0:
        raise ValueError("semi_supervised consistency weights must be nonnegative")
    if out["enabled"] and float(out["teacher_weight"]) <= 0.0:
        raise ValueError("enabled semi-supervision requires teacher_weight > 0")
    if out["enabled"] and float(out["student_weight"]) <= 0.0:
        raise ValueError("enabled semi-supervision requires student_weight > 0")
    if type(out["rampup_epochs"]) is not int or out["rampup_epochs"] < 0:
        raise ValueError("semi_supervised.rampup_epochs must be a nonnegative integer")
    if not 0.0 <= float(out["confidence_threshold"]) <= 1.0:
        raise ValueError("semi_supervised.confidence_threshold must be in [0,1]")
    if not 0.0 <= float(out["background_weight"]) <= 1.0:
        raise ValueError("semi_supervised.background_weight must be in [0,1]")
    if out["enabled"] and float(out["background_weight"]) <= 0.0:
        raise ValueError("enabled semi-supervision requires background_weight > 0")
    if type(out["steps_per_epoch"]) is not int or out["steps_per_epoch"] < 0:
        raise ValueError("semi_supervised.steps_per_epoch must be a nonnegative integer")
    gamma = out["intensity_gamma"]
    if (not isinstance(gamma, (list, tuple)) or len(gamma) != 2
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(item) or float(item) <= 0.0 for item in gamma)
            or float(gamma[0]) > float(gamma[1])):
        raise ValueError("semi_supervised.intensity_gamma must be two positive ordered values")
    out["intensity_gamma"] = [float(gamma[0]), float(gamma[1])]
    if float(out["intensity_noise_std"]) < 0.0:
        raise ValueError("semi_supervised.intensity_noise_std must be nonnegative")
    return out


def validate_config(config):
    raw = deepcopy(config)
    has_semi = "semi_supervised" in raw
    semi = raw.pop("semi_supervised", None)
    c = validate_training_fields(raw, schema=SCHEMA, method="candidate_a3d",
        protocol="native_3d", teacher_kinds=("none", "candidate_a3d"), augmentations=("none",))
    # Keep legacy config payloads byte-for-byte compatible.  New semisupervised
    # recipes must opt in explicitly and carry the complete validated block.
    if has_semi:
        c["semi_supervised"] = _validate_semi_supervised(semi)
    if c["model"]["backbone"] not in {"mednext_l_k3", "tiny_test3d"}:
        raise ValueError("A-3D supports mednext_l_k3 or synthetic-only tiny_test3d; other teachers require a verified adapter")
    from .bank import BankConfig3D, QueryBank3D
    from .codec import MeasurementCodec3D
    from .model import StudentConfig3D
    bank = QueryBank3D(BankConfig3D(**c["bank"]))
    if tuple(bank.config.classes) != (1, 2, 3):
        raise ValueError("A-3D requires native RV/MYO/LV classes")
    codec = MeasurementCodec3D(bank, **c["codec"])
    StudentConfig3D(**c["student"])
    rollout = K1Config(**c["rollout"])
    base = tuple(rollout.baseline_ids)
    if len(set(base)) != len(base) or any(type(i) is not int or not 0 <= i < len(bank) for i in base):
        raise ValueError("R0 must contain fixed unique 3-D bank IDs")
    if codec.cost(base) > rollout.bit_budget:
        raise ValueError("R0 plus 3-D metadata exceeds the packet bit budget")
    return c
