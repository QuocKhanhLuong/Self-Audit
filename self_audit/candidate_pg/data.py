"""Candidate PG data adapters and patient-balanced resumable sampler."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
import random
from typing import Any, Iterator, Mapping

from ..candidate_fuse.data import Frame, FuseData, open_fuse_data
from .config import (
    K8_SOURCE_SHA256,
    SPLIT_LOCK_SHA256,
    TEST_MANIFEST_SHA256,
    TRAIN_MANIFEST_SHA256,
    VAL_MANIFEST_SHA256,
)


FrameKey = tuple[str, int]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest_ref(config: Mapping[str, Any], role: str) -> tuple[str | None, str | None]:
    data = config.get("data", {})
    refs = data.get("manifests") if isinstance(data, Mapping) else None
    if isinstance(refs, Mapping) and role in refs:
        value = refs[role]
        if isinstance(value, Mapping):
            return value.get("path"), value.get("sha256")
        if isinstance(value, (str, Path)):
            return str(value), None
    # Compatibility with the FUSE-style six manifest fields.  This adapter is
    # intentionally read-only and never invents a path or SHA.
    aliases = {
        "train": ("train_manifest", "train_manifest_sha256"),
        "val": ("val_manifest", "val_manifest_sha256"),
        "test": ("test_manifest", "test_manifest_sha256"),
    }
    path_key, sha_key = aliases[role]
    return data.get(path_key), data.get(sha_key)


def _resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    value = Path(path).expanduser()
    return value if value.is_absolute() else (_repo_root() / value).resolve()


@dataclass(frozen=True)
class PGData:
    """Three role-specific FUSE views used by the PG engine."""

    train: FuseData
    val: FuseData
    test: FuseData

    @property
    def development(self) -> FuseData:
        return self.val

    def __getitem__(self, role: str) -> FuseData:
        if role in {"val", "development"}:
            return self.val
        if role in {"train", "test"}:
            return getattr(self, role)
        raise KeyError(role)

    def __iter__(self) -> Iterator[str]:
        return iter(("train", "val", "test"))

    def keys(self) -> tuple[str, str, str]:
        return ("train", "val", "test")


def open_pg_data(config: Mapping[str, Any]) -> PGData:
    """Open the exact Track-A K8 manifests through the FUSE loader.

    The TRAIN manifest is the semi-K8 v2 contract, whose U records contain no
    mask references.  This function never calls ``load`` while opening data;
    therefore a TRAIN U frame (and in particular a non-L mask) cannot be read
    as a side effect of constructing the PG bundle.
    """

    if not isinstance(config, Mapping):
        raise TypeError("PG config must be a mapping")
    data_cfg = config.get("data")
    if not isinstance(data_cfg, Mapping):
        raise ValueError("PG config is missing data")
    if data_cfg.get("split_lock") != "configs/candidate_fuse/semi_k8.json":
        raise ValueError("PG must reuse the exact FUSE semi-K8 split lock")
    if data_cfg.get("split_lock_sha256") not in {None, SPLIT_LOCK_SHA256}:
        raise ValueError("PG split lock SHA256 does not match configs/candidate_fuse/semi_k8.json")
    if data_cfg.get("source_k8_sha256") not in {None, K8_SOURCE_SHA256}:
        raise ValueError("PG source K8 SHA256 does not match the reviewed lock")
    if data_cfg.get("u_partner_policy") != "different_patient" or data_cfg.get("u_sampling") != "patient_balanced":
        raise ValueError("PG requires patient-balanced different-patient U sampling")
    if data_cfg.get("native_geometry") is not True:
        raise ValueError("PG requires native geometry")

    runtime = config.get("runtime", {})
    synthetic = bool(runtime.get("synthetic", False)) if isinstance(runtime, Mapping) else False
    expected_real_sha = {
        "train": TRAIN_MANIFEST_SHA256,
        "val": VAL_MANIFEST_SHA256,
        "test": TEST_MANIFEST_SHA256,
    }
    views: dict[str, FuseData] = {}
    for role, fuse_role in (("train", "train"), ("val", "development"), ("test", "test")):
        path_value, sha_value = _manifest_ref(config, role)
        path = _resolve_path(path_value)
        if path is None or sha_value is None:
            raise ValueError(f"PG {role} manifest path and SHA256 are required")
        if not synthetic and sha_value != expected_real_sha[role]:
            raise ValueError(f"PG {role} manifest SHA256 is not the locked Track-A receipt")
        if not path.is_file():
            raise FileNotFoundError(f"PG {role} manifest does not exist: {path}")
        views[role] = open_fuse_data(path, sha_value, role=fuse_role, synthetic=synthetic)

    train = views["train"]
    if not isinstance(train, FuseData):
        raise TypeError("PG requires the ACDC FuseData TRAIN adapter")
    # Real runs must expose the reviewed K8 budget.  Tiny synthetic fixtures
    # are allowed to use fewer patients for software-only tests.
    if not synthetic and len(getattr(train, "_labeled_ids", ())) != 8:
        raise ValueError("real PG TRAIN manifest must contain exactly locked L8")
    if set(train.labeled_keys()) & set(train.unlabeled_keys()):
        raise ValueError("PG L/U frame identity overlap")
    return PGData(train=train, val=views["val"], test=views["test"])


class PGSampler:
    """Checkpointable one-L/two-U, patient-balanced sampler.

    Each epoch contains one update per U patient.  The first U patient is a
    shuffled permutation and the second is the same permutation cyclically
    shifted by one, guaranteeing different patients while using every patient
    exactly once as an anchor.  Frame choice within each patient is random but
    checkpointed.  L frames are independently shuffled and cycled as needed.
    """

    SCHEMA = "self_audit.candidate_pg.sampler.v1"

    def __init__(self, data: PGData | FuseData, *, seed: int = 42, steps_per_epoch: int | None = None):
        train = data.train if isinstance(data, PGData) else data
        if not isinstance(train, FuseData) or train.role != "train":
            raise ValueError("PGSampler requires TRAIN FuseData or PGData")
        if type(seed) is not int:
            raise TypeError("PGSampler seed must be an integer")
        self.data = train
        self.seed = seed
        self.py_rng = random.Random(seed)
        self.epoch_rng = random.Random(seed ^ 0x5047_5352)
        self.lkeys = tuple(train.labeled_keys())
        self.ukeys = tuple(train.unlabeled_keys())
        self._u_by_patient: dict[str, tuple[FrameKey, ...]] = {}
        for key in self.ukeys:
            self._u_by_patient.setdefault(key[0], tuple())
        for pid in tuple(self._u_by_patient):
            self._u_by_patient[pid] = tuple(key for key in self.ukeys if key[0] == pid)
        self._patients = tuple(sorted(self._u_by_patient))
        if len(self._patients) < 2:
            raise ValueError("PGSampler needs at least two U patients for different-patient pairs")
        if any(not self._u_by_patient[pid] for pid in self._patients):
            raise ValueError("every U patient must have at least one image-only frame")
        self.steps_per_epoch = int(steps_per_epoch or len(self._patients))
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        self.epoch = 0
        self.update = 0
        self._l_stream: list[FrameKey] = []
        self._l_cursor = 0
        self._anchor_patients: list[str] = []
        self._partner_patients: list[str] = []
        self._u_cursor = 0
        self._new_epoch()

    @property
    def joint_epoch_size(self) -> int:
        return self.steps_per_epoch

    @property
    def epoch_size(self) -> int:
        return self.steps_per_epoch

    def _new_epoch(self) -> None:
        self._l_stream = list(self.lkeys)
        self.py_rng.shuffle(self._l_stream)
        self._l_cursor = 0
        self._anchor_patients = list(self._patients)
        self.epoch_rng.shuffle(self._anchor_patients)
        # A one-place cyclic shift is a deterministic derangement for N >= 2.
        self._partner_patients = self._anchor_patients[1:] + self._anchor_patients[:1]
        self._u_cursor = 0

    def _next_labeled(self) -> FrameKey:
        if not self._l_stream or self._l_cursor >= len(self._l_stream):
            self._l_stream = list(self.lkeys)
            self.py_rng.shuffle(self._l_stream)
            self._l_cursor = 0
        key = self._l_stream[self._l_cursor]
        self._l_cursor += 1
        return key

    def _sample_frame(self, pid: str) -> FrameKey:
        frames = self._u_by_patient[pid]
        return frames[self.epoch_rng.randrange(len(frames))]

    def next_step(self) -> tuple[FrameKey, FrameKey, FrameKey]:
        if self._u_cursor >= self.steps_per_epoch:
            self.epoch += 1
            self._new_epoch()
        anchor = self._anchor_patients[self._u_cursor % len(self._anchor_patients)]
        partner = self._partner_patients[self._u_cursor % len(self._partner_patients)]
        if anchor == partner:  # defensive if a future policy changes the shift
            raise RuntimeError("PG sampler produced same-patient U pair")
        sample = (self._next_labeled(), self._sample_frame(anchor), self._sample_frame(partner))
        self._u_cursor += 1
        self.update += 1
        return sample

    def next_pair(self) -> tuple[FrameKey, FrameKey, FrameKey]:
        """Alias retained for engine code that calls the FUSE sampler API."""

        return self.next_step()

    def next_batch(self) -> tuple[FrameKey, FrameKey, FrameKey]:
        return self.next_step()

    def next(self) -> tuple[FrameKey, FrameKey, FrameKey]:
        return self.next_step()

    def __next__(self) -> tuple[FrameKey, FrameKey, FrameKey]:
        return self.next_step()

    def __iter__(self) -> "PGSampler":
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "seed": self.seed,
            "epoch": self.epoch,
            "update": self.update,
            "py_rng": self.py_rng.getstate(),
            "epoch_rng": self.epoch_rng.getstate(),
            "l_stream": list(self._l_stream),
            "l_cursor": self._l_cursor,
            "anchor_patients": list(self._anchor_patients),
            "partner_patients": list(self._partner_patients),
            "u_cursor": self._u_cursor,
            "steps_per_epoch": self.steps_per_epoch,
        }

    @staticmethod
    def _tuples(values: Any, label: str) -> list[tuple[str, int]]:
        if not isinstance(values, list):
            raise ValueError(f"sampler {label} must be a list")
        out: list[tuple[str, int]] = []
        for value in values:
            if not isinstance(value, (list, tuple)) or len(value) != 2 or not isinstance(value[0], str) or type(value[1]) is not int:
                raise ValueError(f"sampler {label} contains an invalid frame key")
            out.append((value[0], value[1]))
        return out

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != self.SCHEMA:
            raise ValueError("invalid PG sampler checkpoint schema")
        if int(state.get("seed")) != self.seed or int(state.get("steps_per_epoch")) != self.steps_per_epoch:
            raise ValueError("PG sampler checkpoint seed/epoch size mismatch")
        l_stream = self._tuples(state.get("l_stream"), "l_stream")
        if len(l_stream) != len(self.lkeys) or set(l_stream) != set(self.lkeys) or len(set(l_stream)) != len(l_stream):
            raise ValueError("PG sampler checkpoint labeled permutation differs from dataset")
        anchors = list(state.get("anchor_patients", [])); partners = list(state.get("partner_patients", []))
        if anchors != list(self._patients) and set(anchors) != set(self._patients):
            raise ValueError("PG sampler checkpoint anchor patients differ from dataset")
        if len(anchors) != len(self._patients) or len(set(anchors)) != len(anchors) or set(anchors) != set(self._patients):
            raise ValueError("PG sampler checkpoint anchor permutation is invalid")
        if len(partners) != len(anchors) or len(set(partners)) != len(partners) or set(partners) != set(self._patients):
            raise ValueError("PG sampler checkpoint partner permutation is invalid")
        if any(a == b for a, b in zip(anchors, partners)):
            raise ValueError("PG sampler checkpoint contains same-patient U pair")
        l_cursor = int(state.get("l_cursor")); u_cursor = int(state.get("u_cursor"))
        if not 0 <= l_cursor <= len(l_stream) or not 0 <= u_cursor <= self.steps_per_epoch:
            raise ValueError("PG sampler checkpoint cursor is invalid")
        self.py_rng.setstate(state["py_rng"])
        self.epoch_rng.setstate(state["epoch_rng"])
        self.epoch = int(state.get("epoch", 0)); self.update = int(state.get("update", 0))
        self._l_stream = l_stream; self._l_cursor = l_cursor
        self._anchor_patients = anchors; self._partner_patients = partners; self._u_cursor = u_cursor


__all__ = ["Frame", "FrameKey", "FuseData", "PGData", "PGSampler", "open_pg_data"]
