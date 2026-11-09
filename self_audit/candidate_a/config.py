"""Versioned, explicit recipe for the typed-measurement experiment."""
from copy import deepcopy
import math
from . import SCHEMA, CLASS_MAPPING


def validate_training_fields(config, *, schema=SCHEMA, method="candidate_a",
                             protocol="native_2d_25d",
                             teacher_kinds=("none", "native_v2", "candidate_a"),
                             augmentations=("aligned_flip", "none")):
    """Shared training/provenance fields; each track validates its own tensors."""
    c = deepcopy(config)
    fields = {"schema", "method", "stage", "epochs", "seed", "threads", "student_lr",
              "teacher_lr", "weight_decay", "teacher_update_every", "freeze_teacher",
              "use_teacher", "lambda_distance", "model", "teacher_init", "bank",
              "codec", "student", "rollout", "loss", "class_mapping", "augmentation",
              "dataset", "validation_every", "checkpoint_every_epochs"}
    if set(c) != fields or c["schema"] != schema or c["method"] != method:
        raise ValueError(f"expected exact {schema} config; unknown/missing fields rejected")
    if c["stage"] not in {"teacher_pretrain", "teacher_warmup", "joint"}:
        raise ValueError("invalid candidate-A stage")
    if c["class_mapping"] != CLASS_MAPPING:
        raise ValueError("required class mapping: BG=0 RV=1 MYO=2 LV=3")
    for k in ("epochs", "seed", "threads", "teacher_update_every"):
        if type(c[k]) is not int or c[k] < (0 if k == "seed" else 1):
            raise ValueError(f"invalid {k}")
    if set(c["dataset"]) != {"name", "protocol"} or c["dataset"]["name"] not in {"acdc", "mnms"}:
        raise ValueError("dataset.name must be acdc or mnms")
    if c["dataset"]["protocol"] != protocol:
        raise ValueError(f"{method} requires protocol {protocol}")
    if type(c["validation_every"]) is not int or c["validation_every"] < 0:
        raise ValueError("validation_every must be a non-negative integer")
    if type(c["checkpoint_every_epochs"]) is not int or c["checkpoint_every_epochs"] < 0:
        raise ValueError("checkpoint_every_epochs must be a non-negative integer")
    for k in ("student_lr", "teacher_lr", "weight_decay", "lambda_distance"):
        if isinstance(c[k], bool) or not isinstance(c[k], (int, float)) or not math.isfinite(c[k]) or c[k] < 0:
            raise ValueError(f"invalid {k}")
    if min(c["student_lr"], c["teacher_lr"]) <= 0:
        raise ValueError("learning rates must be positive")
    for k in ("freeze_teacher", "use_teacher"):
        if type(c[k]) is not bool:
            raise ValueError(f"invalid {k}")
    if c["stage"] != "joint" and (not c["use_teacher"] or c["freeze_teacher"]):
        raise ValueError("teacher training stages require an unfrozen teacher")
    if c["augmentation"] not in augmentations:
        raise ValueError(f"{method} supports augmentation {augmentations}")
    if set(c["teacher_init"]) != {"kind", "path", "sha256"} or c["teacher_init"]["kind"] not in teacher_kinds:
        raise ValueError("explicit teacher initialization provenance required")
    if c["teacher_init"]["kind"] != "none" and (not c["teacher_init"]["path"] or len(c["teacher_init"]["sha256"]) != 64):
        raise ValueError("teacher checkpoint path and manifest SHA256 required")
    if set(c["model"]) != {"backbone", "source_repo", "source_commit", "weights", "weights_sha256"}:
        raise ValueError("native teacher backbone provenance fields required")
    if set(c["codec"]) != {"value_bits"}:
        raise ValueError("codec requires value_bits")
    if set(c["loss"]) != {"region", "boundary", "policy"}:
        raise ValueError("separate region/boundary/policy weights required")
    for value in c["loss"].values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("loss weights must be finite and nonnegative")
    return c


def validate_config(config):
    c = validate_training_fields(config)
    # Dataclass constructors reject unknown nested fields; their own validators
    # establish the finite bank, architecture and action/budget contracts.
    from .bank import BankConfig, QueryBank
    from .codec import MeasurementCodec
    from .model import StudentConfig
    from .k1 import K1Config
    bank = QueryBank(BankConfig(**c["bank"]))
    if bank.config.classes != (1, 2, 3):
        raise ValueError("candidate A uses the native RV/MYO/LV foreground ontology")
    codec = MeasurementCodec(bank, **c["codec"])
    StudentConfig(**c["student"])
    rollout = K1Config(**c["rollout"])
    if type(rollout.bit_budget) is not int or rollout.bit_budget <= 0:
        raise ValueError("a finite positive complete packet bit budget is mandatory")
    base = tuple(rollout.baseline_ids)
    if len(set(base)) != len(base) or any(type(j) is not int or not 0 <= j < len(bank) for j in base):
        raise ValueError("R0 must be fixed unique bank indices")
    if codec.cost(base) > rollout.bit_budget:
        raise ValueError("R0 plus packet metadata exceeds bit budget")
    return c
