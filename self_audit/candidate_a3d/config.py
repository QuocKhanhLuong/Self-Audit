"""Explicit native-volume recipe; 2-D settings cannot silently become 3-D."""

from self_audit.candidate_a.config import validate_training_fields
from self_audit.candidate_a.k1 import K1Config
from . import SCHEMA


def validate_config(config):
    c = validate_training_fields(config, schema=SCHEMA, method="candidate_a3d",
        protocol="native_3d", teacher_kinds=("none", "candidate_a3d"), augmentations=("none",))
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
