"""Method-agnostic frozen native protocol contracts. No producer/evaluator imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ProtocolBlocked(ValueError):
    def __init__(self, reasons, status="BLOCKED_PROTOCOL"):
        self.reasons = list(reasons)
        self.status = status
        super().__init__(status + ": " + "; ".join(self.reasons))


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def value_hash(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


REQUIRED_FIELD_GROUPS = ("paper_unspecified", "implementation_conventions", "required_data")


def unresolved_required_fields(config: dict) -> list[tuple[str, str]]:
    """Declared required fields (``{"value": ...}`` entries) that still have no value."""
    missing = []
    for group in REQUIRED_FIELD_GROUPS:
        for name, entry in sorted(config.get(group, {}).items()):
            if not isinstance(entry, dict) or entry.get("value") is None:
                missing.append((group, name))
    return missing


def load_lock(path: Path, *, purpose="producer") -> dict:
    """A scientific edit cannot bypass execution gates by changing a status flag."""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if purpose not in {"producer", "native_track_b"} or config.get("schema") != "medical-native.protocol.v1":
        raise ProtocolBlocked(["unknown execution purpose/protocol schema"])
    index_path = path.parent / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get(path.name) != value_hash(config):
        raise ProtocolBlocked(["config differs from frozen index"])
    if purpose == "native_track_b":
        spec_path = path.parents[2] / "evaluation/track_b/spec.json"
        if not spec_path.is_file():
            raise ProtocolBlocked(["frozen native evaluator specification missing"])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if config.get("native_evaluator_spec_sha256") != value_hash(spec):
            raise ProtocolBlocked(["native evaluator specification differs from frozen protocol"])
    if purpose not in config.get("gates", {}):
        raise ProtocolBlocked(["execution purpose has no explicit gate"])
    blockers = list(config["gates"][purpose])
    if purpose == "producer":
        blockers += [f"{group} value required: {name}" for group, name in unresolved_required_fields(config)]
    if blockers:
        raise ProtocolBlocked(blockers)
    if purpose == "producer" and not config.get("scientific"):
        raise ProtocolBlocked(["no executable scientific settings"])
    return config


def native_status(*, producer_complete: bool, track_b_status: str,
                  track_a_status: str, all_required_data: bool = True,
                  protocol_resolved: bool = True, verification_reliable: bool = True,
                  protocol_faithful: bool = True, same_seed_equal: bool | None = None,
                  stochastic_behavior_documented: bool = False) -> dict:
    """Track A absence and documented stochastic variation do not change reproduction."""
    if not verification_reliable or not protocol_faithful:
        reproduction = "BLOCKED_REPRODUCIBILITY"
    elif not protocol_resolved:
        reproduction = "BLOCKED_PROTOCOL"
    elif not all_required_data:
        reproduction = "PARTIAL" if producer_complete else "BLOCKED_DATA"
    elif producer_complete and track_b_status == "COMPLETE":
        reproduction = "COMPLETE"
    else:
        reproduction = "PARTIAL" if producer_complete else "PENDING"
    return {
        "NATIVE_REPRODUCTION_STATUS": reproduction,
        "NATIVE_TRACK_A_STATUS": track_a_status,
        "NATIVE_TRACK_B_STATUS": track_b_status,
        "repeat_run_diagnostics": {
            "same_seed_equal": same_seed_equal,
            "stochastic_behavior_documented": stochastic_behavior_documented,
            "verification_reliable": verification_reliable,
        },
    }
