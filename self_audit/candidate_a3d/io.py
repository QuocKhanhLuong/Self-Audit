"""3-D factories using the shared source-pinned checkpoint implementation."""

from self_audit.candidate_a import io as shared
from . import CHECKPOINT_SCHEMA, EXPORT_SCHEMA

source_inventory = shared.source_inventory


def build_student(config):
    from .bank import BankConfig3D, QueryBank3D
    from .codec import MeasurementCodec3D
    from .model import CandidateA3DStudent, StudentConfig3D
    bank = QueryBank3D(BankConfig3D(**config["bank"]))
    codec = MeasurementCodec3D(bank, **config["codec"])
    model = CandidateA3DStudent(bank, StudentConfig3D(**config["student"]))
    return model, bank, codec


def read_checkpoint(folder, expected_sha256):
    payload, meta = shared.read_checkpoint(folder, expected_sha256)
    if meta["schema"] not in {CHECKPOINT_SCHEMA, EXPORT_SCHEMA}:
        raise ValueError("A-3D requires its own checkpoint; 2D weights are not interchangeable")
    return payload, meta


def save_checkpoint(folder, **kwargs):
    return shared.save_checkpoint(folder, checkpoint_schema=CHECKPOINT_SCHEMA,
                                  model_class="CandidateA3DStudent", **kwargs)


def load_student(folder, expected_sha256, device="cpu"):
    result = shared.load_student(folder, expected_sha256, device)
    if result[-1]["schema"] not in {CHECKPOINT_SCHEMA, EXPORT_SCHEMA}:
        raise ValueError("not an A-3D student checkpoint")
    return result


def export_student(checkpoint, checkpoint_sha256, out):
    read_checkpoint(checkpoint, checkpoint_sha256)
    return shared.export_student(checkpoint, checkpoint_sha256, out)
