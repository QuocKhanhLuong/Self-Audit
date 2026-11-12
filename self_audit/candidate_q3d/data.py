"""Protocol-isolated Q-3D views and deterministic 1L+1U sampling."""

from __future__ import annotations

import copy
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..candidate_a3d.data import CandidateA3DData
from ..candidate_a3d.geometry import canonical_geometry


class Q3DData:
    """Keep student supervision and optional all-TRAIN teacher labels explicit.

    ``student_data`` is the manifest used by the student: K labeled TRAIN
    patients and image-only U.  Protocol A reuses that label pool for the
    teacher.  Protocol B requires a separate CandidateA3DData manifest whose
    labeled set covers every TRAIN patient; its labels are accessed only via
    ``load_teacher_labeled_frame``.
    """

    def __init__(
        self,
        student_data: CandidateA3DData,
        *,
        protocol: str,
        teacher_data: CandidateA3DData | None = None,
    ) -> None:
        if not isinstance(student_data, CandidateA3DData):
            raise TypeError("Q3DData requires an existing CandidateA3DData student view")
        if protocol not in {"A", "B"}:
            raise ValueError("Q-3D protocol must be A or B")
        all_student_ids = set(student_data.lids + student_data.uids)
        if protocol == "A":
            if teacher_data is None:
                teacher_data = student_data
            if not isinstance(teacher_data, CandidateA3DData):
                raise TypeError("protocol A teacher data must be CandidateA3DData")
            if (set(teacher_data.lids + teacher_data.uids) != all_student_ids
                    or set(teacher_data.lids) != set(student_data.lids)):
                raise ValueError("protocol A teacher and student must use the same K labeled patients")
        else:
            if not isinstance(teacher_data, CandidateA3DData):
                raise ValueError("protocol B requires a separate all-TRAIN labeled teacher_data view")
            if (set(teacher_data.lids) != all_student_ids or teacher_data.uids
                    or set(teacher_data.lids + teacher_data.uids) != all_student_ids):
                raise ValueError("protocol B teacher must have GT for every TRAIN patient and no U pool")
        if set(teacher_data.lids + teacher_data.uids) != all_student_ids:
            raise ValueError("teacher and student views must contain the same TRAIN patient cohort")
        student_records = {record["id"]: record for record in student_data.labeled + student_data.unlabeled}
        teacher_records = {record["id"]: record for record in teacher_data.labeled + teacher_data.unlabeled}
        for patient in all_student_ids:
            if student_records[patient]["image_sha256"] != teacher_records[patient]["image_sha256"]:
                raise ValueError(f"teacher and student native cine bytes differ for TRAIN patient {patient!r}")
            if canonical_geometry(student_data.geometry_for(patient)) != canonical_geometry(
                teacher_data.geometry_for(patient)
            ):
                raise ValueError(f"teacher and student native geometry differs for TRAIN patient {patient!r}")
        if student_data.manifest.get("dataset") != teacher_data.manifest.get("dataset"):
            raise ValueError("teacher and student views must declare the same dataset identity")
        self.student_data = student_data
        self.teacher_data = teacher_data
        self.protocol = protocol
        self.labeled_patients = tuple(student_data.lids)
        self.unlabeled_patients = tuple(student_data.uids)
        self.teacher_patients = tuple(teacher_data.lids)
        self.train_patients = tuple(sorted(all_student_ids))
        self.label_budget = len(self.labeled_patients)
        self.unlabeled_image_reads = student_data.unlabeled_image_reads
        self.label_reads = student_data.label_reads

    def labeled_frame_keys(self) -> list[tuple[int, int]]:
        """Return student-supervised full-volume frame references."""
        return self.student_data.labeled_frame_keys()

    def labeled_image_frame_keys(self) -> list[tuple[int, int]]:
        """Return every L cine frame key from images without opening masks."""
        keys: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.student_data.labeled):
            image = self.student_data.base.array(record["image"], record["image_sha256"])
            geometry = self.student_data.geometry_for(record["id"])
            if image.ndim != 4 or tuple(image.shape[1:]) != tuple(geometry["shape"]):
                raise ValueError("Q-stage labeled cine must match native full-volume geometry")
            keys.extend((record_index, frame) for frame in range(image.shape[0]))
        if not keys:
            raise ValueError("Q-stage labeled manifest has no native image frames")
        return keys

    def unlabeled_frame_keys(self) -> list[tuple[int, int]]:
        """Return image-only student U frame references, with no mask access."""
        return self.student_data.unlabeled_frame_keys()

    def teacher_frame_keys(self) -> list[tuple[int, int]]:
        """Return the protocol-bound teacher GT frames (K in A, all TRAIN in B)."""
        return self.teacher_data.labeled_frame_keys()

    def labeled_patient_for_key(self, key: tuple[int, int]) -> str:
        record_index, _ = self._checked_key(key, len(self.student_data.labeled), "labeled")
        return self.student_data.lids[record_index]

    def unlabeled_patient_for_key(self, key: tuple[int, int]) -> str:
        record_index, _ = self._checked_key(key, len(self.student_data.unlabeled), "unlabeled")
        return self.student_data.uids[record_index]

    def teacher_patient_for_key(self, key: tuple[int, int]) -> str:
        record_index, _ = self._checked_key(key, len(self.teacher_data.labeled), "teacher")
        return self.teacher_data.lids[record_index]

    @staticmethod
    def _checked_key(key: tuple[int, int], record_count: int, name: str) -> tuple[int, int]:
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            raise ValueError(f"Q-3D {name} key must be (record_index, frame)")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in key):
            raise ValueError(f"Q-3D {name} key values must be integers")
        record_index, frame = key
        if not 0 <= record_index < record_count or frame < 0:
            raise IndexError(f"Q-3D {name} key is outside the declared dataset")
        return record_index, frame

    def load_labeled_frame(self, key: tuple[int, int]):
        return self.student_data.load_labeled_frame(key)

    def load_labeled_image_frame(
        self, key: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        """Load one labeled TRAIN cine frame without opening its mask.

        Q-stage tokenizer learning uses labeled images as image-only teacher
        inputs. The mask path is deliberately never passed to the cache reader.
        """
        record_index, frame = self._checked_key(key, len(self.student_data.labeled), "labeled")
        record = self.student_data.labeled[record_index]
        cine = self.student_data.base.array(record["image"], record["image_sha256"])
        geometry = self.student_data.geometry_for(record["id"])
        if cine.ndim != 4 or tuple(cine.shape[1:]) != tuple(geometry["shape"]):
            raise ValueError("Q-stage labeled cine must match native full-volume geometry")
        if not 0 <= frame < cine.shape[0]:
            raise IndexError("Q-stage labeled image frame is outside its native cine")
        image = torch.from_numpy(np.array(cine[frame], dtype=np.float32, copy=True))[None, None]
        if not torch.isfinite(image).all() or bool((image < 0).any()) or bool((image > 1).any()):
            raise ValueError("Q-stage labeled image must be finite and normalized to [0,1]")
        metadata = {"patient": record["id"], "frame": frame}
        return image, geometry, metadata

    def load_unlabeled_frame(self, key: tuple[int, int]):
        image, geometry, metadata = self.student_data.load_unlabeled_frame(key)
        if set(metadata) != {"patient", "frame"}:
            raise ValueError("Q-3D U metadata may contain only patient and frame identifiers")
        if metadata["patient"] != self.unlabeled_patient_for_key(key):
            raise ValueError("Q-3D U frame patient differs from its image-only key")
        return image, geometry, metadata

    def load_teacher_labeled_frame(self, key: tuple[int, int]):
        return self.teacher_data.load_labeled_frame(key)

    def identity(self, *, include_unlabeled_frames: bool = True) -> dict[str, Any]:
        """Return a stable, JSON-safe identity used by sampler checkpoints."""
        student_l = self._metadata_labeled_frame_keys(self.student_data)
        # C0 is an L-only control.  Its identity records the frozen manifest and
        # declared TRAIN patient inventory without opening any image-only U cine.
        student_u = self.student_data.unlabeled_frame_keys() if include_unlabeled_frames else []
        teacher_l = self._metadata_labeled_frame_keys(self.teacher_data)
        return {
            "protocol": self.protocol,
            "train_patients": list(self.train_patients),
            "student_labeled_patients": sorted(self.labeled_patients),
            "student_unlabeled_patients": sorted(self.unlabeled_patients),
            "teacher_labeled_patients": sorted(self.teacher_patients),
            "student_manifest_sha256": self._manifest_sha256(self.student_data),
            "teacher_manifest_sha256": self._manifest_sha256(self.teacher_data),
            "ordered_student_labeled_frame_keys": [list(key) for key in student_l],
            "ordered_student_unlabeled_frame_keys": [list(key) for key in student_u],
            "ordered_teacher_labeled_frame_keys": [list(key) for key in teacher_l],
        }

    @staticmethod
    def _metadata_labeled_frame_keys(data: CandidateA3DData) -> list[tuple[int, int]]:
        """Bind resume identity to declared label frames without loading GT arrays."""
        keys: list[tuple[int, int]] = []
        for record_index, record in enumerate(data.labeled):
            cine = data.base.array(record["image"], record["image_sha256"])
            geometry = data.geometry_for(record["id"])
            if cine.ndim != 4 or tuple(cine.shape[1:]) != tuple(geometry["shape"]):
                raise ValueError("labeled cine must match native full-volume geometry")
            masks = record.get("masks")
            if not isinstance(masks, Mapping) or not masks:
                raise ValueError("labeled patient must declare annotated frame indices")
            if any(not str(frame_key).isdigit() for frame_key in masks):
                raise ValueError("annotated frame keys must be integer frame indices")
            for frame_key in sorted(masks, key=int):
                frame = int(frame_key)
                if not 0 <= frame < cine.shape[0]:
                    raise IndexError("annotated image frame is outside its native cine")
                keys.append((record_index, frame))
        if not keys:
            raise ValueError("labeled manifest has no declared annotated frame indices")
        return keys

    @staticmethod
    def _manifest_sha256(data: CandidateA3DData) -> str:
        return hashlib.sha256(Path(data.path).read_bytes()).hexdigest()

    def cache_source_identity(self) -> dict[str, Any]:
        """Bind physiology rows to the exact protocol and image-only U cohort."""
        records = {record["id"]: record for record in self.student_data.unlabeled}
        return {
            "protocol": self.protocol,
            "dataset": self.student_data.manifest.get("dataset"),
            "student_manifest_sha256": self._manifest_sha256(self.student_data),
            "unlabeled_patients": [
                {"id": patient, "image_sha256": records[patient]["image_sha256"]}
                for patient in self.unlabeled_patients
            ],
        }


class Q3DJointSampler:
    """Infinite deterministic sampler: one L and one U patient per update.

    L sampling retains its historical random stream exactly. U patients are
    visited once per shuffled cycle; a separate frame RNG samples within each
    patient so frame counts cannot change future patient order. All streams,
    the current U permutation, and its cursor are serialized for exact resume.
    """

    STATE_SCHEMA = "self_audit.candidate_q3d.joint_sampler.v3"
    LEGACY_STATE_SCHEMA = "self_audit.candidate_q3d.joint_sampler.v2"

    def __init__(self, data: Q3DData, *, seed: int = 42,
                 require_unlabeled: bool = True) -> None:
        if not isinstance(data, Q3DData):
            raise TypeError("Q3DJointSampler requires Q3DData")
        if type(seed) is not int or seed < 0:
            raise ValueError("sampler seed must be a nonnegative integer")
        if type(require_unlabeled) is not bool:
            raise TypeError("require_unlabeled must be a bool")
        if not data.labeled_patients:
            raise ValueError("Q-3D sampling requires at least one labeled L patient")
        if require_unlabeled and not data.unlabeled_patients:
            raise ValueError("Q-3D paired sampling requires image-only U patients")
        self.data = data
        self.seed = seed
        self.require_unlabeled = require_unlabeled
        # Keep the L stream identical between C0 (L-only) and paired arms.
        # Adding/removing U must not change which supervised examples C0 sees.
        self._labeled_rng = random.Random(seed)
        # U patient permutations and per-patient frame choices are independent
        # streams: changing cine frame counts cannot perturb future patients.
        self._unlabeled_permutation_rng = random.Random(seed + 1)
        # Frame choices use seed + 2, independent of permutation advancement.
        self._unlabeled_frame_rng = random.Random(seed + 2)
        self.updates = 0
        self._unlabeled_permutation: list[str] = []
        self._unlabeled_cursor = 0
        self._unlabeled_cycle_count = 0
        self._labeled_by_patient: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._unlabeled_by_patient: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for key in data.labeled_frame_keys():
            self._labeled_by_patient[data.labeled_patient_for_key(key)].append(key)
        if require_unlabeled:
            for key in data.unlabeled_frame_keys():
                self._unlabeled_by_patient[data.unlabeled_patient_for_key(key)].append(key)
        missing_l = set(data.labeled_patients) - set(self._labeled_by_patient)
        missing_u = (set(data.unlabeled_patients) - set(self._unlabeled_by_patient)
                     if require_unlabeled else set())
        if missing_l or missing_u:
            raise ValueError("every Q-3D L and U patient must have at least one native frame")
        self._identity = data.identity(include_unlabeled_frames=require_unlabeled)
        if require_unlabeled:
            self._start_next_unlabeled_cycle()

    @property
    def completed_unlabeled_cycles(self) -> int:
        """Number of complete U permutations already consumed."""
        return self._unlabeled_cycle_count

    def _start_next_unlabeled_cycle(self) -> None:
        self._unlabeled_permutation = list(self.data.unlabeled_patients)
        self._unlabeled_permutation_rng.shuffle(self._unlabeled_permutation)
        self._unlabeled_cursor = 0

    def sample_update(self) -> dict[str, Any]:
        labeled_patient = self._labeled_rng.choice(self.data.labeled_patients)
        labeled_key = self._labeled_rng.choice(self._labeled_by_patient[labeled_patient])
        record = {
            "protocol": self.data.protocol,
            "logical_update": self.updates,
            "labeled_patient": labeled_patient,
            "labeled_key": labeled_key,
        }
        if self.require_unlabeled:
            unlabeled_patient = self._unlabeled_permutation[self._unlabeled_cursor]
            unlabeled_key = self._unlabeled_frame_rng.choice(
                self._unlabeled_by_patient[unlabeled_patient]
            )
            cycle_position = self._unlabeled_cursor + 1
            cycle_size = len(self._unlabeled_permutation)
            record["unlabeled_patient"] = unlabeled_patient
            record["unlabeled_key"] = unlabeled_key
            record["unlabeled_cycle"] = self._unlabeled_cycle_count + 1
            record["unlabeled_cycle_position"] = cycle_position
            record["unlabeled_cycle_complete"] = cycle_position == cycle_size
            self._unlabeled_cursor += 1
            if self._unlabeled_cursor == cycle_size:
                self._unlabeled_cycle_count += 1
                self._start_next_unlabeled_cycle()
        self.updates += 1
        return record

    def __iter__(self):
        while True:
            yield self.sample_update()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.STATE_SCHEMA,
            "seed": self.seed,
            "updates": self.updates,
            "require_unlabeled": self.require_unlabeled,
            "unlabeled_permutation": list(self._unlabeled_permutation),
            "unlabeled_cursor": self._unlabeled_cursor,
            "unlabeled_cycle_count": self._unlabeled_cycle_count,
            "labeled_rng_state": self._labeled_rng.getstate(),
            "unlabeled_permutation_rng_state": self._unlabeled_permutation_rng.getstate(),
            "unlabeled_frame_rng_state": self._unlabeled_frame_rng.getstate(),
            "data_identity": copy.deepcopy(self._identity),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("Q-3D sampler checkpoint state must be a mapping")
        schema = state.get("schema")
        if schema == self.LEGACY_STATE_SCHEMA:
            raise ValueError(
                "Q-3D sampler v2 state cannot resume with the cycle-based v3 sampler; "
                "start a fresh joint stage"
            )
        if schema != self.STATE_SCHEMA:
            raise ValueError("Q-3D sampler checkpoint schema mismatch; expected v3")
        if type(state.get("seed")) is not int or state.get("seed") != self.seed:
            raise ValueError("Q-3D sampler seed differs from checkpoint")
        if state.get("require_unlabeled") is not self.require_unlabeled:
            raise ValueError("Q-3D sampler L/U mode differs from checkpoint")
        if state.get("data_identity") != self._identity:
            raise ValueError("Q-3D sampler checkpoint belongs to another protocol or label subset")
        updates = state.get("updates")
        if type(updates) is not int or updates < 0:
            raise ValueError("Q-3D sampler update count is invalid")

        permutation = state.get("unlabeled_permutation")
        cursor = state.get("unlabeled_cursor")
        cycle_count = state.get("unlabeled_cycle_count")
        if (not isinstance(permutation, list)
                or any(type(patient) is not str for patient in permutation)):
            raise ValueError("Q-3D sampler unlabeled permutation is invalid")
        if type(cursor) is not int or type(cycle_count) is not int:
            raise ValueError("Q-3D sampler cursor or cycle count is invalid")
        if cursor < 0 or cycle_count < 0:
            raise ValueError("Q-3D sampler cursor or cycle count is invalid")
        if self.require_unlabeled:
            expected_patients = tuple(self.data.unlabeled_patients)
            if (len(permutation) != len(expected_patients)
                    or set(permutation) != set(expected_patients)):
                raise ValueError(
                    "Q-3D sampler unlabeled permutation is not a full patient permutation"
                )
            if cursor >= len(permutation):
                raise ValueError("Q-3D sampler unlabeled cursor is outside the current permutation")
            if updates != cycle_count * len(permutation) + cursor:
                raise ValueError("Q-3D sampler cursor and cycle count disagree with update count")
        elif permutation or cursor != 0 or cycle_count != 0:
            raise ValueError("Q-3D L-only sampler state must not contain a U permutation or cycles")

        try:
            labeled_rng = random.Random()
            permutation_rng = random.Random()
            frame_rng = random.Random()
            labeled_rng.setstate(state["labeled_rng_state"])
            permutation_rng.setstate(state["unlabeled_permutation_rng_state"])
            frame_rng.setstate(state["unlabeled_frame_rng_state"])
        except (KeyError, IndexError, OverflowError, TypeError, ValueError) as exc:
            raise ValueError("Q-3D sampler RNG state is invalid") from exc

        # Commit only after every field has passed validation, so a malformed
        # checkpoint cannot leave a partially restored sampler behind.
        self._labeled_rng = labeled_rng
        self._unlabeled_permutation_rng = permutation_rng
        self._unlabeled_frame_rng = frame_rng
        self._unlabeled_permutation = list(permutation)
        self._unlabeled_cursor = cursor
        self._unlabeled_cycle_count = cycle_count
        self.updates = updates


__all__ = ["Q3DData", "Q3DJointSampler"]
