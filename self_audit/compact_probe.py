"""Independent checkpointed probe generator for cross-patient transfer utility.

Owned by Trainer worker per CPTU_IMPLEMENTATION_CONTRACT.md:
  - Samples image-only TRAIN-U records excluding ALL proposal patient IDs in the transaction.
  - Separate NumPy Generator initialized with probe_seed (default 42017).
  - Historical L/U/audit RNG consumption remains strictly unchanged.
  - Reuses same spatial/intensity crop geometry and cached teacher targets.
  - Never accesses labels, masks, or non-TRAIN records.
  - Missing probe (no eligible unlabeled patients) yields None (zero decision in controller).
  - Optional same_batch_utility control reuses already sampled proposal tensors.
  - Checkpointed state for exact resume.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "CompactProbeSampler",
    "build_same_batch_probe",
]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_schedule(metadata: Mapping[str, Any]) -> List[Dict[str, Any]]:
    samples = metadata.get("samples")
    if not isinstance(samples, list):
        samples = [metadata]
    return [
        {
            "patient": str(sample.get("patient", "")),
            "frame": int(sample.get("frame", 0)),
            "slice": int(sample.get("slice", 0)),
            "crop_sha256": _canonical_sha256(sample.get("crop")),
        }
        for sample in samples
    ]


class CompactProbeSampler:
    """Independent generator for TRAIN-U cross-patient utility probes.

    Maintains a dedicated NumPy Generator independent of the proposal and audit
    RNG schedules. Samples image-only TRAIN-U slices from patients strictly
    disjoint from all proposal patients in the current transaction.
    """

    def __init__(
        self,
        data: Any,
        cache: Any,
        *,
        seed: int = 42017,
        batch_size: int = 8,
        crop_size: int = 0,
        generator: Optional[np.random.Generator] = None,
    ) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError(f"probe_seed must be a non-negative integer, got {seed!r}")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError(f"probe_batch_size must be a positive integer, got {batch_size!r}")
        if type(crop_size) is not int or crop_size < 0:
            raise ValueError(f"crop_size must be a non-negative integer, got {crop_size!r}")

        self.data = data
        self.cache = cache
        self.seed = seed
        self.batch_size = batch_size
        self.crop_size = crop_size

        if generator is not None:
            if not isinstance(generator, np.random.Generator):
                raise TypeError("generator must be a numpy.random.Generator instance")
            self.rng = generator
        else:
            self.rng = np.random.default_rng(seed)

        self.samples_drawn = 0
        self.probes_created = 0
        self.seen_patients: Set[str] = set()
        self.total_sampling_seconds = 0.0
        self.total_cache_seconds = 0.0

    def get_state(self) -> Dict[str, Any]:
        """Serialize complete generator state and coverage counters for checkpointing."""
        return {
            "seed": int(self.seed),
            "batch_size": int(self.batch_size),
            "crop_size": int(self.crop_size),
            "bit_generator_state": copy.deepcopy(self.rng.bit_generator.state),
            "samples_drawn": int(self.samples_drawn),
            "probes_created": int(self.probes_created),
            "seen_patients": sorted(self.seen_patients),
            "total_sampling_seconds": float(self.total_sampling_seconds),
            "total_cache_seconds": float(self.total_cache_seconds),
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore generator state and counters from checkpoint, strictly validating all fields."""
        if not isinstance(state, Mapping):
            raise TypeError("probe sampler state must be a mapping")

        # 1. Verify seed
        if "seed" not in state or state["seed"] != self.seed:
            raise ValueError(
                f"probe sampler seed mismatch: state seed={state.get('seed')!r}, expected={self.seed!r}"
            )

        # 2. Verify batch_size
        if "batch_size" not in state or state["batch_size"] != self.batch_size:
            raise ValueError(
                f"probe sampler batch_size mismatch: state batch_size={state.get('batch_size')!r}, expected={self.batch_size!r}"
            )

        # 3. Verify crop_size
        if "crop_size" not in state or state["crop_size"] != self.crop_size:
            raise ValueError(
                f"probe sampler crop_size mismatch: state crop_size={state.get('crop_size')!r}, expected={self.crop_size!r}"
            )

        # 4. Verify counters
        if "samples_drawn" not in state or type(state["samples_drawn"]) is not int or state["samples_drawn"] < 0:
            raise ValueError(f"invalid samples_drawn in probe state: {state.get('samples_drawn')!r}")
        if "probes_created" not in state or type(state["probes_created"]) is not int or state["probes_created"] < 0:
            raise ValueError(f"invalid probes_created in probe state: {state.get('probes_created')!r}")

        # 5. Verify patient IDs
        raw_patients = state.get("seen_patients")
        if not isinstance(raw_patients, list) or not all(isinstance(pid, str) for pid in raw_patients):
            raise ValueError(f"seen_patients must be a list of strings, got {raw_patients!r}")
        valid_uids = {str(r["id"]) for r in self.data.unlabeled} if hasattr(self.data, "unlabeled") else set()
        for pid in raw_patients:
            if valid_uids and pid not in valid_uids:
                raise ValueError(f"seen_patients contains unknown patient ID: {pid!r}")

        # 6. Verify and restore BitGenerator state
        bg_state = state.get("bit_generator_state")
        if not isinstance(bg_state, Mapping) or "bit_generator" not in bg_state:
            raise ValueError("probe sampler state missing valid bit_generator_state mapping")
        self.rng.bit_generator.state = copy.deepcopy(bg_state)

        # Commit validated fields
        self.samples_drawn = state["samples_drawn"]
        self.probes_created = state["probes_created"]
        self.seen_patients = set(raw_patients)
        self.total_sampling_seconds = float(state.get("total_sampling_seconds", 0.0))
        self.total_cache_seconds = float(state.get("total_cache_seconds", 0.0))

    def sample_probe(
        self,
        proposal_patient_ids: Union[str, Sequence[str], Set[str]],
        *,
        device: str = "cpu",
        temperature: float = 1.0,
        coefficient: float = 1.0,
        confidence: float = 0.7,
    ) -> Optional[Dict[str, Any]]:
        """Sample one utility probe from TRAIN-U excluding all proposal patient IDs.

        Args:
            proposal_patient_ids: Single patient ID or collection of IDs to exclude.
            device: Target torch device for detached tensors.
            temperature: Distillation temperature.
            coefficient: Loss weighting coefficient.
            confidence: Raw teacher confidence threshold.

        Returns:
            Mapping with detached tensors 'x', 'targets', 'valid', floats
            'temperature', 'coefficient', 'confidence', and JSON-safe 'metadata'.
            Returns None if no eligible unlabeled patients remain.
        """
        if isinstance(proposal_patient_ids, str):
            excluded = {proposal_patient_ids}
        else:
            excluded = set(proposal_patient_ids)

        if not hasattr(self.data, "unlabeled"):
            raise ValueError("data provider must have 'unlabeled' attribute for probe sampling")

        # Find eligible TRAIN-U patients strictly excluding every proposal patient
        eligible_indices = [
            i for i, r in enumerate(self.data.unlabeled) if str(r["id"]) not in excluded
        ]

        if not eligible_indices:
            # Contract: No eligible probe => zero reason. Missing probe is a zero decision.
            return None

        # Sample one eligible unlabeled patient index using probe generator
        chosen_patient_idx = eligible_indices[int(self.rng.integers(len(eligible_indices)))]
        chosen_pid = str(self.data.unlabeled[chosen_patient_idx]["id"])

        # Sample probe batch using the same image-only sampling and crop geometry, timing sampling
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        t_sample_start = time.monotonic()
        X_probe, probe_meta = self.data.sample_batch(
            chosen_patient_idx,
            self.rng,
            self.batch_size,
            labeled=False,
            crop_size=self.crop_size,
        )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        probe_sampling_seconds = time.monotonic() - t_sample_start

        # Retrieve cached teacher targets aligned with probe_meta, timing cache lookup
        if self.cache is None:
            raise ValueError("teacher cache is required to sample utility probe targets")
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        t_cache_start = time.monotonic()
        teacher_targets, valid_mask = self.cache.sample_batch(probe_meta, device=device)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        probe_cache_seconds = time.monotonic() - t_cache_start

        # Update counters and accumulated timings
        self.samples_drawn += len(X_probe)
        self.probes_created += 1
        self.seen_patients.add(chosen_pid)
        self.total_sampling_seconds += probe_sampling_seconds
        self.total_cache_seconds += probe_cache_seconds

        metadata = {
            "patient": chosen_pid,
            "samples": _sample_schedule(probe_meta),
            "batch_size": int(len(X_probe)),
            "excluded_proposal_patients": sorted(excluded),
            "same_batch": False,
            "timings": {
                "probe_sampling_seconds": float(probe_sampling_seconds),
                "probe_cache_seconds": float(probe_cache_seconds),
            },
        }

        return {
            "x": X_probe.to(device).detach(),
            "targets": teacher_targets.to(device).detach(),
            "valid": valid_mask.to(device).detach(),
            "temperature": float(temperature),
            "coefficient": float(coefficient),
            "confidence": float(confidence),
            "metadata": metadata,
            "timings": {
                "probe_sampling_seconds": float(probe_sampling_seconds),
                "probe_cache_seconds": float(probe_cache_seconds),
            },
        }


def build_same_batch_probe(
    x: Tensor,
    targets: Tensor,
    valid: Tensor,
    *,
    temperature: float = 1.0,
    coefficient: float = 1.0,
    confidence: float = 0.7,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Package proposal tensors as utility probe for same_batch_utility control."""
    meta = dict(metadata) if metadata is not None else {}
    meta["same_batch"] = True
    meta["timings"] = {
        "probe_sampling_seconds": 0.0,
        "probe_cache_seconds": 0.0,
    }
    return {
        "x": x.detach(),
        "targets": targets.detach(),
        "valid": valid.detach(),
        "temperature": float(temperature),
        "coefficient": float(coefficient),
        "confidence": float(confidence),
        "metadata": meta,
        "timings": {
            "probe_sampling_seconds": 0.0,
            "probe_cache_seconds": 0.0,
        },
    }
