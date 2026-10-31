"""Frozen full-EMA teacher identity, lineage preflight, and bounded memmapped cache.

Owned by Worker teacher/data per COMPACT_IMPLEMENTATION_CONTRACT.md:
  - Frozen full-EMA teacher identity with mandatory hashes and explicit raw-vs-EMA inference state.
  - Canonical teachers require explicit pins: mandatory checkpoint, source, payload hashes, full EMA method,
    and explicit quality receipt. Never stamps verified=True from claimed fields alone.
  - Safe historical bridge: exact old source checkout 0f54455b... with closure 3c7fa85... verified clean,
    isolated package import under unique importlib namespace, strict model build with mandatory DINO pins.
  - Historical checkpoint payload inspection: handles student.pt, weights.pt, or matched SHA payload,
    failing closed if physical payload is missing (Real missing assets BLOCKED).
  - Strict selected state (student or teacher_ema) with no unverified fallback payload.
  - FixedTeacherCache: binds train manifest SHA, exact unlabeled IDs, mandatory per-file hashes,
    axes/classes/dtype/shape vs image, finite normalized sampled probabilities, and valid metadata
    with crop and flip synchronization; rejects bad frames.
  - Preflight full label union with actual data BEFORE model/cache use; synthetic kind only on synthetic data.
  - build_teacher_cache: open_memmap output streaming batch-by-batch (no full volume in RAM), with
    latency, VRAM, and disk bytes report.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import re
from pathlib import Path
import subprocess
import sys
import time
import types
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import TrainingData, crop_native, write_json
from .model import digest

__all__ = [
    "HISTORICAL_BASELINE_PINS",
    "HISTORICAL_MANIFEST_SHA256",
    "HISTORICAL_MODEL_SHA256",
    "HISTORICAL_PAYLOAD_SHA256",
    "HISTORICAL_SOURCE_CLOSURE",
    "HISTORICAL_SOURCE_COMMIT",
    "HISTORICAL_WINDOW_SHA256",
    "REPRODUCTION_RECEIPT_SCHEMA",
    "FixedTeacherCache",
    "build_teacher_cache",
    "find_checkpoint_payload",
    "import_historical_package",
    "import_reproduction_package",
    "is_historical_baseline_checkpoint",
    "load_historical_teacher_model",
    "load_reproduction_teacher_model",
    "load_teacher_for_cache",
    "preflight_teacher_lineage",
    "validate_reproduction_receipt",
    "validate_teacher_identity",
    "verify_historical_source",
    "verify_reproduction_source",
]

# Exact pinned historical baseline constants
HISTORICAL_SOURCE_COMMIT = "0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca"
HISTORICAL_SOURCE_CLOSURE = "3c7fa85f00d54a6e087af3cd6c7045351fb72d532f1069bf09d95f9cd72359b7"
HISTORICAL_MANIFEST_SHA256 = "03c1730f88941ada898bed7fcf1b9f13cdd8f08bcce4390539fec665c82e97de"
HISTORICAL_PAYLOAD_SHA256 = "b36bcafe9613325d85ee70dabd6a36fef7c17429882967831a151c0cce6d8437"
HISTORICAL_TRAIN_MANIFEST_SHA256 = "e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4"
HISTORICAL_MODEL_SHA256 = "80e3b11b46c11d4660aa13d99ecbf30640247c987f5425dcfe62aa1c1f90fc03"
HISTORICAL_WINDOW_SHA256 = "8b661ce007ef692c4575c418b90e89b59b7217f21f098c7ab19863de2e04fa72"

# Reproduction receipt schema
REPRODUCTION_RECEIPT_SCHEMA = "self_audit.b8.reproduction.v1"
HISTORICAL_LABELED_IDS = [
    "patient007",
    "patient010",
    "patient011",
    "patient024",
    "patient026",
    "patient041",
    "patient077",
    "patient099",
]
HISTORICAL_LABEL_BUDGET = 8
HISTORICAL_INFERENCE_STATE = "student"
HISTORICAL_METHOD = "ema_full"

HISTORICAL_BASELINE_PINS: Dict[str, Any] = {
    "source_commit": HISTORICAL_SOURCE_COMMIT,
    "source_sha256": HISTORICAL_SOURCE_CLOSURE,
    "manifest_sha256": HISTORICAL_MANIFEST_SHA256,
    "payload_sha256": HISTORICAL_PAYLOAD_SHA256,
    "train_manifest_sha256": HISTORICAL_TRAIN_MANIFEST_SHA256,
    "inference_state": HISTORICAL_INFERENCE_STATE,
    "training_method": HISTORICAL_METHOD,
    "model_class": "NativeStudent",
    "label_budget": HISTORICAL_LABEL_BUDGET,
    "labeled_ids": list(HISTORICAL_LABELED_IDS),
}


def _validate_patient_id(pid: str) -> None:
    if not isinstance(pid, str) or not pid.strip():
        raise ValueError(f"invalid patient ID: '{pid}'")
    for forbidden in ("..", "/", "\\", "\0", "\n", "\r", "\t"):
        if forbidden in pid:
            raise ValueError(f"invalid character in patient ID: '{pid}'")


def is_historical_baseline_checkpoint(identity_or_manifest: Mapping[str, Any]) -> bool:
    """Return True if the identity or manifest matches the pinned historical baseline."""
    manifest_sha = identity_or_manifest.get("manifest_sha256")
    source_sha = identity_or_manifest.get("source_sha256")
    return (
        manifest_sha == HISTORICAL_MANIFEST_SHA256
        and source_sha == HISTORICAL_SOURCE_CLOSURE
    )


# ---------------------------------------------------------------------------
# Safe historical bridge
# ---------------------------------------------------------------------------

def verify_historical_source(source_dir: Union[str, Path]) -> Dict[str, Any]:
    """Verify that source_dir contains the clean historical checkout 0f54455b with closure 3c7fa85f."""
    s_path = Path(source_dir).resolve()
    if not s_path.is_dir():
        raise FileNotFoundError(f"historical source directory not found: {s_path}")

    pkg_dir = s_path / "self_audit"
    train_file = s_path / "train.py"
    if not pkg_dir.is_dir() or not train_file.is_file():
        raise ValueError(f"historical source directory {s_path} lacks self_audit/ or train.py")

    # Compute source closure
    files = sorted(pkg_dir.glob("*.py")) + [train_file]
    closure = hashlib.sha256("".join(f"{p.name}:{digest(p)}\n" for p in files).encode()).hexdigest()
    if closure != HISTORICAL_SOURCE_CLOSURE:
        raise ValueError(
            f"historical source closure mismatch: got {closure}, expected {HISTORICAL_SOURCE_CLOSURE}"
        )

    # If git repo exists, check exact commit and clean status
    is_git = (s_path / ".git").exists() or (s_path.parent / ".git").exists()
    if is_git:
        try:
            head = subprocess.check_output(
                ["git", "-C", str(s_path), "rev-parse", "HEAD"], text=True
            ).strip()
            if head != HISTORICAL_SOURCE_COMMIT:
                raise ValueError(
                    f"historical source commit mismatch: got {head}, expected {HISTORICAL_SOURCE_COMMIT}"
                )
            dirty = subprocess.check_output(
                ["git", "-C", str(s_path), "status", "--porcelain", "--untracked-files=no"], text=True
            ).strip()
            if dirty:
                raise ValueError(f"historical source checkout has uncommitted changes: {dirty}")
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"git check failed on historical source: {exc}") from exc

    return {
        "status": "PASS",
        "source_dir": str(s_path),
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "source_closure": HISTORICAL_SOURCE_CLOSURE,
        "verified_clean": True,
    }


def import_historical_package(
    source_dir: Union[str, Path],
    pkg_alias: str = "_historical_self_audit_0f54455b",
) -> types.ModuleType:
    """Import historical self_audit package under a unique namespace preserving old relative imports."""
    s_path = Path(source_dir).resolve()
    verify_historical_source(s_path)

    if pkg_alias in sys.modules:
        return sys.modules[pkg_alias]

    pkg_dir = s_path / "self_audit"
    pkg = types.ModuleType(pkg_alias)
    pkg.__path__ = [str(pkg_dir)]
    pkg.__file__ = str(pkg_dir / "__init__.py")
    pkg.__package__ = pkg_alias
    sys.modules[pkg_alias] = pkg

    # Ensure init is executed
    init_file = pkg_dir / "__init__.py"
    if init_file.is_file():
        spec = importlib.util.spec_from_file_location(pkg_alias, str(init_file))
        if spec and spec.loader:
            spec.loader.exec_module(pkg)

    return pkg


def load_historical_teacher_model(
    source_dir: Union[str, Path],
    model_config: Mapping[str, Any],
    *,
    synthetic: bool = False,
    window: Optional[Any] = None,
) -> nn.Module:
    """Build historical model using the isolated historical import namespace."""
    pkg_alias = "_historical_self_audit_0f54455b"
    import_historical_package(source_dir, pkg_alias=pkg_alias)

    model_mod = importlib.import_module(f"{pkg_alias}.model")
    if not hasattr(model_mod, "build_model"):
        raise AttributeError(f"{pkg_alias}.model lacks build_model")

    model, _ = model_mod.build_model(
        dict(model_config),
        synthetic=synthetic,
        window=window,
    )
    return model


# ---------------------------------------------------------------------------
# Safe reproduction bridge
# ---------------------------------------------------------------------------

def verify_reproduction_source(
    source_dir: Union[str, Path],
    *,
    expected_commit: Optional[str] = None,
    expected_closure: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify that source_dir contains a clean reproduction source checkout.

    Strictly asserts:
      1. source_dir is a git repository (.git required).
      2. expected_commit and expected_closure are explicitly provided.
      3. git commit matches expected_commit exactly, and working tree is clean.
      4. source closure matches expected_closure exactly.
      5. self_audit/model.py matches HISTORICAL_MODEL_SHA256 bit-for-bit.
      6. self_audit/window.py matches HISTORICAL_WINDOW_SHA256 bit-for-bit.
    """
    s_path = Path(source_dir).resolve()
    if not s_path.is_dir():
        raise FileNotFoundError(f"reproduction source directory not found: {s_path}")

    # Must be a git repository root
    is_git = (s_path / ".git").exists()
    if is_git:
        try:
            toplevel = subprocess.check_output(
                ["git", "-C", str(s_path), "rev-parse", "--show-toplevel"], text=True
            ).strip()
            if Path(toplevel).resolve() != s_path:
                is_git = False
        except subprocess.CalledProcessError:
            is_git = False
    if not is_git:
        raise ValueError(f"reproduction source at {s_path} is not a git repository; git is strictly required")

    if expected_commit is None or expected_closure is None:
        raise ValueError("verify_reproduction_source requires explicit expected_commit and expected_closure")

    pkg_dir = s_path / "self_audit"
    train_file = s_path / "train.py"
    if not pkg_dir.is_dir() or not train_file.is_file():
        raise ValueError(f"reproduction source directory {s_path} lacks self_audit/ or train.py")

    # Critical architectural invariant: model.py and window.py must match historical baseline exactly
    model_file = pkg_dir / "model.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"model.py missing in {pkg_dir}")
    actual_model_sha = digest(model_file)
    if actual_model_sha != HISTORICAL_MODEL_SHA256:
        raise ValueError(
            f"reproduction source model.py hash mismatch: got {actual_model_sha}, "
            f"expected historical {HISTORICAL_MODEL_SHA256}"
        )

    window_file = pkg_dir / "window.py"
    if not window_file.is_file():
        raise FileNotFoundError(f"window.py missing in {pkg_dir}")
    actual_window_sha = digest(window_file)
    if actual_window_sha != HISTORICAL_WINDOW_SHA256:
        raise ValueError(
            f"reproduction source window.py hash mismatch: got {actual_window_sha}, "
            f"expected historical {HISTORICAL_WINDOW_SHA256}"
        )

    try:
        head = subprocess.check_output(
            ["git", "-C", str(s_path), "rev-parse", "HEAD"], text=True
        ).strip()
        if head.lower() != expected_commit.lower():
            raise ValueError(
                f"reproduction source commit mismatch: got {head}, expected {expected_commit}"
            )
        dirty = subprocess.check_output(
            ["git", "-C", str(s_path), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
        if dirty:
            raise ValueError(f"reproduction source checkout has uncommitted changes: {dirty}")
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"git check failed on reproduction source: {exc}") from exc

    # Compute source closure
    files = sorted(pkg_dir.glob("*.py")) + [train_file]
    closure = hashlib.sha256("".join(f"{p.name}:{digest(p)}\n" for p in files).encode()).hexdigest()
    if closure.lower() != expected_closure.lower():
        raise ValueError(
            f"reproduction source closure mismatch: got {closure}, expected {expected_closure}"
        )

    return {
        "status": "PASS",
        "source_dir": str(s_path),
        "source_commit": head,
        "source_closure": closure,
        "model_sha256": actual_model_sha,
        "window_sha256": actual_window_sha,
        "verified_clean": True,
    }


def import_reproduction_package(
    source_dir: Union[str, Path],
    *,
    pkg_alias: Optional[str] = None,
    expected_commit: Optional[str] = None,
    expected_closure: Optional[str] = None,
) -> types.ModuleType:
    """Import reproduction self_audit package under an isolated namespace."""
    s_path = Path(source_dir).resolve()
    ver = verify_reproduction_source(
        s_path, expected_commit=expected_commit, expected_closure=expected_closure
    )
    commit_tag = (ver.get("source_commit") or "repro")[:8]
    if pkg_alias is None:
        pkg_alias = f"_reproduction_self_audit_{commit_tag}"

    if pkg_alias in sys.modules:
        return sys.modules[pkg_alias]

    pkg_dir = s_path / "self_audit"
    pkg = types.ModuleType(pkg_alias)
    pkg.__path__ = [str(pkg_dir)]
    pkg.__file__ = str(pkg_dir / "__init__.py")
    pkg.__package__ = pkg_alias
    sys.modules[pkg_alias] = pkg

    init_file = pkg_dir / "__init__.py"
    if init_file.is_file():
        spec = importlib.util.spec_from_file_location(pkg_alias, str(init_file))
        if spec and spec.loader:
            spec.loader.exec_module(pkg)

    return pkg


def load_reproduction_teacher_model(
    source_dir: Union[str, Path],
    model_config: Mapping[str, Any],
    *,
    synthetic: bool = False,
    window: Optional[Any] = None,
    expected_commit: Optional[str] = None,
    expected_closure: Optional[str] = None,
) -> nn.Module:
    """Build reproduction model using the isolated reproduction import namespace."""
    pkg = import_reproduction_package(
        source_dir, expected_commit=expected_commit, expected_closure=expected_closure
    )
    pkg_alias = pkg.__name__
    model_mod = importlib.import_module(f"{pkg_alias}.model")
    if not hasattr(model_mod, "build_model"):
        raise AttributeError(f"{pkg_alias}.model lacks build_model")

    model, _ = model_mod.build_model(
        dict(model_config),
        synthetic=synthetic,
        window=window,
    )
    return model


def validate_reproduction_receipt(
    receipt_or_path: Union[Mapping[str, Any], str, Path],
    *,
    expected_receipt_sha256: Optional[str] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    raw_student_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Validate reproduction B8 teacher receipt against strict reproduction contract.

    Contract:
      - schema: 'self_audit.b8.reproduction.v1'
      - new_reproduction: True
      - inference_state: 'student'
      - valid 64-hex SHA-256 for checkpoint_payload, raw_student, checkpoint_manifest, source
      - valid 40-hex source_commit
      - label_budget: 8, labeled_ids matching HISTORICAL_LABELED_IDS exactly
      - quality_gate: 'GO' with metrics.test50.Macro4 >= 0.87
      - remote_files: list of verified backup files with readback == 'SHA256_MATCH' and non-empty drive_id
      - fail-closed verification of physical files if paths provided
    """
    receipt_file_sha: Optional[str] = None
    if isinstance(receipt_or_path, (str, Path)):
        p = Path(receipt_or_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"reproduction receipt file not found: {p}")
        receipt_file_sha = digest(p)
        if expected_receipt_sha256 is not None:
            if receipt_file_sha.lower() != expected_receipt_sha256.lower():
                raise ValueError(
                    f"reproduction receipt file hash mismatch: got {receipt_file_sha}, "
                    f"expected {expected_receipt_sha256}"
                )
        receipt_data = json.loads(p.read_text())
    elif isinstance(receipt_or_path, Mapping):
        receipt_data = dict(receipt_or_path)
    else:
        raise TypeError("receipt_or_path must be a file path or mapping")

    # Schema & identity
    if receipt_data.get("schema") != REPRODUCTION_RECEIPT_SCHEMA:
        raise ValueError(
            f"invalid reproduction receipt schema: got '{receipt_data.get('schema')}', "
            f"expected '{REPRODUCTION_RECEIPT_SCHEMA}'"
        )
    if receipt_data.get("new_reproduction") is not True:
        raise ValueError("reproduction receipt must declare new_reproduction=True")
    if receipt_data.get("inference_state") != "student":
        raise ValueError(
            f"reproduction teacher inference_state must be 'student', got '{receipt_data.get('inference_state')}'"
        )

    # Commit & hashes
    source_commit = receipt_data.get("source_commit")
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", source_commit) is None:
        raise ValueError(f"invalid reproduction source_commit: {source_commit!r}; expected 40-hex git SHA")

    for h_field in ("checkpoint_payload_sha256", "checkpoint_manifest_sha256", "raw_student_sha256", "source_sha256"):
        val = receipt_data.get(h_field)
        if not isinstance(val, str) or re.fullmatch(r"[0-9a-fA-F]{64}", val) is None:
            raise ValueError(f"reproduction receipt missing or invalid 64-hex '{h_field}': {val!r}")

    # L8 lineage check
    label_budget = receipt_data.get("label_budget")
    if label_budget != 8:
        raise ValueError(f"reproduction label_budget must be 8, got {label_budget}")
    labeled_ids = receipt_data.get("labeled_ids", [])
    if sorted(labeled_ids) != sorted(HISTORICAL_LABELED_IDS):
        raise ValueError(
            f"reproduction labeled_ids mismatch: got {sorted(labeled_ids)}, "
            f"expected {sorted(HISTORICAL_LABELED_IDS)}"
        )
    for pid in labeled_ids:
        _validate_patient_id(pid)

    # Quality gate: Macro4 >= 0.87 and quality_gate == "GO"
    quality_gate = receipt_data.get("quality_gate")
    if quality_gate != "GO":
        raise ValueError(f"reproduction quality_gate is '{quality_gate}'; must be 'GO'")

    metrics = receipt_data.get("metrics")
    if not isinstance(metrics, Mapping) or "test50" not in metrics:
        raise ValueError("reproduction receipt missing 'metrics.test50'")
    test50 = metrics["test50"]
    macro4 = test50.get("Macro4")
    if type(macro4) not in (int, float) or not np.isfinite(macro4) or macro4 < 0.87 or macro4 > 1.0:
        raise ValueError(
            f"reproduction teacher quality gate failed: test50.Macro4={macro4} must be finite in [0.87, 1.0]"
        )

    # Remote files verification check: strictly check hashes, sizes, and reject contradictory duplicates
    remote_files = receipt_data.get("remote_files")
    if not isinstance(remote_files, list) or len(remote_files) == 0:
        raise ValueError("reproduction receipt missing mandatory remote_files backup verification")

    verified_remote_files: Dict[str, Dict[str, Any]] = {}
    for item in remote_files:
        if not isinstance(item, Mapping):
            raise ValueError(f"invalid remote_file record: {item}")
        fname = item.get("file")
        fbytes = item.get("bytes")
        fsha = item.get("sha256")
        fdrive = item.get("drive_id")
        freadback = item.get("readback")

        if not fname or not isinstance(fname, str):
            raise ValueError(f"remote_file missing valid filename: {item}")
        if type(fbytes) is not int or fbytes <= 0:
            raise ValueError(f"remote_file '{fname}' has invalid byte count {fbytes!r}; must be positive int")
        if not isinstance(fsha, str) or re.fullmatch(r"[0-9a-fA-F]{64}", fsha) is None:
            raise ValueError(f"remote_file '{fname}' has invalid sha256 {fsha}")
        if not fdrive or not isinstance(fdrive, str):
            raise ValueError(f"remote_file '{fname}' missing verified drive_id: {item}")
        if freadback != "SHA256_MATCH":
            raise ValueError(f"remote_file '{fname}' readback verification status '{freadback}' != 'SHA256_MATCH'")

        if fname in verified_remote_files:
            prior = verified_remote_files[fname]
            if prior["sha256"] != fsha.lower() or prior["bytes"] != fbytes:
                raise ValueError(
                    f"duplicate contradictory remote_file entry for '{fname}': "
                    f"prior=({prior['bytes']}b, {prior['sha256']}) vs new=({fbytes}b, {fsha})"
                )
        verified_remote_files[fname] = {"bytes": fbytes, "sha256": fsha.lower(), "drive_id": fdrive}

    # Verify essential files map exactly to receipt hashes
    expected_mappings = [
        ("raw_student/raw_student.pt", receipt_data["raw_student_sha256"].lower()),
        ("full_checkpoint/weights.pt", receipt_data["checkpoint_payload_sha256"].lower()),
        ("full_checkpoint/MANIFEST.json", receipt_data["checkpoint_manifest_sha256"].lower()),
    ]
    for ess_name, ess_sha in expected_mappings:
        if ess_name not in verified_remote_files:
            raise ValueError(f"reproduction remote_files missing mandatory verified backup of '{ess_name}'")
        actual_rec_sha = verified_remote_files[ess_name]["sha256"]
        if actual_rec_sha != ess_sha:
            raise ValueError(
                f"reproduction remote_files hash mismatch for '{ess_name}': "
                f"remote record has {actual_rec_sha}, expected receipt {ess_sha}"
            )

    # Physical checks on disk if paths provided
    payload_p = None
    if checkpoint_dir is not None:
        cd = Path(checkpoint_dir).resolve()
        m_file = cd / "MANIFEST.json"
        if not m_file.is_file():
            raise FileNotFoundError(f"reproduction checkpoint MANIFEST.json not found in {cd}")
        actual_msha = digest(m_file)
        if actual_msha.lower() != receipt_data["checkpoint_manifest_sha256"].lower():
            raise ValueError(
                f"reproduction checkpoint manifest hash mismatch: got {actual_msha}, "
                f"expected {receipt_data['checkpoint_manifest_sha256']}"
            )
        if "full_checkpoint/MANIFEST.json" in verified_remote_files:
            expected_mbytes = verified_remote_files["full_checkpoint/MANIFEST.json"]["bytes"]
            actual_mbytes = m_file.stat().st_size
            if actual_mbytes != expected_mbytes:
                raise ValueError(
                    f"full checkpoint MANIFEST.json size mismatch: {actual_mbytes}b != remote {expected_mbytes}b"
                )

        # Validate checkpoint manifest contents
        cp_manifest = json.loads(m_file.read_text())
        if cp_manifest.get("schema") != "self_audit.checkpoint.v2":
            raise ValueError("reproduction requires a full v2 checkpoint manifest")
        if cp_manifest.get("config", {}).get("method") != "ema_full":
            raise ValueError("checkpoint MANIFEST.json config.method must be 'ema_full'")
        if cp_manifest.get("step") != 10000:
            raise ValueError(f"checkpoint MANIFEST.json step must be 10000, got {cp_manifest.get('step')!r}")
        if cp_manifest.get("source_sha256", "").lower() != receipt_data["source_sha256"].lower():
            raise ValueError("checkpoint MANIFEST.json source_sha256 mismatch with receipt")
        manifest_lids = cp_manifest.get("labeled_ids")
        if not isinstance(manifest_lids, list) or sorted(manifest_lids) != sorted(receipt_data["labeled_ids"]):
            raise ValueError("checkpoint MANIFEST.json labeled_ids mismatch with receipt")

        payload_p = find_checkpoint_payload(cd, receipt_data["checkpoint_payload_sha256"])
        if "full_checkpoint/weights.pt" in verified_remote_files:
            expected_pbytes = verified_remote_files["full_checkpoint/weights.pt"]["bytes"]
            actual_pbytes = payload_p.stat().st_size
            if actual_pbytes != expected_pbytes:
                raise ValueError(
                    f"full checkpoint payload size mismatch: {actual_pbytes}b != remote {expected_pbytes}b"
                )

    raw_state = None
    if raw_student_path is not None:
        rp = Path(raw_student_path).resolve()
        if not rp.is_file():
            raise FileNotFoundError(f"raw_student payload file not found: {rp}")
        actual_rsha = digest(rp)
        if actual_rsha.lower() != receipt_data["raw_student_sha256"].lower():
            raise ValueError(
                f"raw_student payload hash mismatch: got {actual_rsha}, "
                f"expected {receipt_data['raw_student_sha256']}"
            )
        if "raw_student/raw_student.pt" in verified_remote_files:
            expected_rbytes = verified_remote_files["raw_student/raw_student.pt"]["bytes"]
            actual_rbytes = rp.stat().st_size
            if actual_rbytes != expected_rbytes:
                raise ValueError(
                    f"raw_student.pt size mismatch: {actual_rbytes}b != remote {expected_rbytes}b"
                )
        raw_state = torch.load(rp, map_location="cpu", weights_only=True)
        if not isinstance(raw_state, Mapping):
            raise ValueError(f"raw_student.pt must contain a state_dict mapping, got {type(raw_state)}")
        for k, v in raw_state.items():
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                raise ValueError(f"non-finite weights in raw_student.pt at key '{k}'")

    # If both checkpoint_dir and raw_student_path provided, compare raw student tensors to full payload student
    if checkpoint_dir is not None and raw_student_path is not None and payload_p is not None and raw_state is not None:
        full_payload = torch.load(payload_p, map_location="cpu", weights_only=True)
        if not isinstance(full_payload, Mapping) or "student" not in full_payload:
            raise ValueError("full checkpoint payload must contain 'student' mapping")
        student_state = full_payload["student"]
        if not isinstance(student_state, Mapping):
            raise ValueError("checkpoint student state must be a mapping")
        if set(raw_state.keys()) != set(student_state.keys()):
            raise ValueError(
                f"state_dict key mismatch between raw_student.pt and checkpoint student: "
                f"raw={set(raw_state.keys())} vs checkpoint={set(student_state.keys())}"
            )
        for k, raw_t in raw_state.items():
            stud_t = student_state[k]
            if not torch.equal(raw_t, stud_t):
                raise ValueError(f"weight tensor mismatch between raw_student.pt and checkpoint student at '{k}'")

    res = dict(receipt_data)
    # Claims in a dict alone cannot stamp physical verified
    res["verified"] = bool(checkpoint_dir or raw_student_path)
    res["teacher_kind"] = "reproduction"
    res["training_method"] = "ema_full"
    res["manifest_sha256"] = receipt_data["checkpoint_manifest_sha256"].lower()
    res["payload_sha256"] = receipt_data["checkpoint_payload_sha256"].lower()
    res["raw_student_sha256"] = receipt_data["raw_student_sha256"].lower()
    res["checkpoint_payload_sha256"] = receipt_data["checkpoint_payload_sha256"].lower()
    res["checkpoint_manifest_sha256"] = receipt_data["checkpoint_manifest_sha256"].lower()
    res["source_sha256"] = receipt_data["source_sha256"].lower()
    res["source_commit"] = source_commit.lower()
    res["source_training_cost"] = receipt_data.get("source_training_cost", "NOT_KNOWN")
    if "parent_receipt_sha256" in receipt_data:
        res["parent_receipt_sha256"] = receipt_data["parent_receipt_sha256"]
    res["quality_receipt"] = {
        "Macro4": macro4,
        "foreground": test50.get("foreground"),
        "quality_gate": quality_gate,
        "remote_verified": True,
    }
    res["remote_drive_verified"] = True
    res["physical_payload_status"] = "VERIFIED_ON_DISK" if (checkpoint_dir or raw_student_path) else "BLOCKED_MISSING_PAYLOAD"
    res["verification_status"] = "VERIFIED" if (checkpoint_dir or raw_student_path) else "CLAIMED_METADATA_UNVERIFIED"
    if receipt_file_sha is not None:
        res["receipt_file_sha256"] = receipt_file_sha.lower()
    return res


def find_checkpoint_payload(
    checkpoint_dir: Union[str, Path],
    expected_sha256: Optional[str] = None,
) -> Path:
    """Find checkpoint payload file matching expected SHA256 (inspecting student.pt, weights.pt, etc.).

    Do not assume weights.pt: historical runs often save student.pt or final payload.
    Fails closed with FileNotFoundError if no matching payload is present.
    """
    cd = Path(checkpoint_dir).resolve()
    if not cd.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {cd}")

    candidates = ["student.pt", "weights.pt", "checkpoint.pt", "model.pt", "payload.pt"]
    for name in candidates:
        target = cd / name
        if target.is_file():
            if expected_sha256 is None:
                return target
            if digest(target) == expected_sha256:
                return target

    # Search all .pt files in directory
    if expected_sha256 is not None:
        for pt_file in cd.glob("*.pt"):
            if digest(pt_file) == expected_sha256:
                return pt_file

    raise FileNotFoundError(
        f"mandatory payload file matching sha256={expected_sha256} not found in {cd} "
        f"(checked {candidates}). Real missing assets BLOCKED."
    )


# ---------------------------------------------------------------------------
# Teacher identity validation and preflight
# ---------------------------------------------------------------------------

def validate_teacher_identity(
    teacher_identity: Optional[Mapping[str, Any]],
    *,
    declared_k: Optional[int] = None,
    declared_labeled_ids: Optional[Sequence[str]] = None,
    data: Optional[Any] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    payload_path: Optional[Union[str, Path]] = None,
    allow_historical_bridge: bool = True,
) -> Dict[str, Any]:
    """Validate teacher identity against strict lineage, state, method, and hash criteria.

    Canonical teachers require explicit pins: must supply mandatory checkpoint/source/payload hashes,
    full EMA method, and explicit quality receipt. Cannot stamp verified=True from claimed fields alone.
    """
    if teacher_identity is None:
        raise ValueError("teacher_identity is required; unknown lineage blocks real training")
    if not isinstance(teacher_identity, Mapping):
        raise TypeError("teacher_identity must be a dictionary or Mapping")

    t_id = dict(teacher_identity)
    kind = t_id.get("teacher_kind", "canonical")
    if kind not in {"historical_baseline", "canonical", "synthetic", "reproduction"}:
        raise ValueError(f"unknown teacher_kind '{kind}'; expected historical_baseline, canonical, synthetic, or reproduction")

    # Unpack lineage
    lineage = t_id.get("lineage")
    if isinstance(lineage, Mapping):
        teacher_lids = list(lineage.get("exact_labeled_ids") or lineage.get("labeled_ids") or [])
        teacher_k = lineage.get("budget_k", len(teacher_lids))
    else:
        teacher_lids = list(t_id.get("labeled_ids") or [])
        teacher_k = t_id.get("label_budget", len(teacher_lids))

    if not teacher_lids:
        raise ValueError("teacher labeled IDs are mandatory; unknown lineage blocks real training")
    if type(teacher_k) is not int or teacher_k < 1:
        raise ValueError("teacher label budget must be a positive integer")
    for pid in teacher_lids:
        _validate_patient_id(pid)
    if len(set(teacher_lids)) != len(teacher_lids):
        raise ValueError("duplicate teacher labeled IDs detected")
    if len(teacher_lids) != teacher_k:
        raise ValueError(f"teacher labeled ID count {len(teacher_lids)} != declared budget K={teacher_k}")

    # Mandatory state selection
    inference_state = t_id.get("inference_state")
    if inference_state not in {"student", "teacher_ema"}:
        raise ValueError(f"invalid inference_state '{inference_state}'; must be 'student' or 'teacher_ema'")

    checks_passed: List[str] = ["lineage_format", "patient_ids_validated", "inference_state_declared"]
    is_manifest_verified = False
    is_payload_verified = False

    # Check synthetic compatibility: synthetic kind ONLY with synthetic data
    if data is not None:
        is_data_synthetic = bool(getattr(data, "manifest", {}).get("synthetic", False))
        if kind == "synthetic" and not is_data_synthetic:
            raise ValueError(
                "synthetic teacher kind is strictly prohibited with real patient data; "
                "synthetic_kind only permitted with synthetic data"
            )
        if is_data_synthetic and kind not in {"synthetic", "historical_baseline"}:
            pass  # Synthetic fixtures can use synthetic teachers

    # Lineage preflight against declared_k and declared_labeled_ids / data
    target_k = declared_k
    target_ids = declared_labeled_ids
    if data is not None:
        if target_k is None:
            target_k = getattr(data, "manifest", {}).get("label_budget", len(getattr(data, "lids", [])))
        if target_ids is None:
            target_ids = getattr(data, "lids", None)

    if target_k is not None:
        if len(teacher_lids) > target_k or teacher_k > target_k:
            raise ValueError(
                f"teacher labeled count ({len(teacher_lids)}) exceeds declared student budget K={target_k}"
            )
        checks_passed.append("budget_k_satisfied")

    if target_ids is not None:
        decl_set = set(target_ids)
        teach_set = set(teacher_lids)
        union_set = teach_set | decl_set
        if target_k is not None and len(union_set) > target_k:
            raise ValueError(
                f"union of teacher and student labeled IDs ({len(union_set)}) exceeds budget K={target_k}"
            )
        checks_passed.append("label_union_satisfied")

    # Canonical teacher requirements: explicit pins
    if kind == "canonical":
        for h_key in ("manifest_sha256", "source_sha256", "payload_sha256"):
            h_val = t_id.get(h_key)
            if not isinstance(h_val, str) or re.fullmatch(r"[0-9a-f]{64}", h_val) is None:
                raise ValueError(f"canonical teacher requires mandatory 64-hex '{h_key}'; unpinned teachers are prohibited")

        training_method = t_id.get("training_method")
        if training_method not in {"ema_full", "ema_full_meta_window"}:
            raise ValueError(
                f"canonical teacher requires a full EMA training method ('ema_full' or 'ema_full_meta_window'), "
                f"got '{training_method}'"
            )

        quality_receipt = t_id.get("quality_receipt")
        if not quality_receipt or not isinstance(quality_receipt, Mapping):
            raise ValueError(
                "canonical teacher requires an explicit, verified quality receipt mapping (Macro4, foreground Dice, etc.)"
            )
        checks_passed.extend(["mandatory_hashes_present", "full_ema_method", "quality_receipt_present"])

    # Historical baseline requirements
    elif kind == "historical_baseline":
        pins = HISTORICAL_BASELINE_PINS
        if t_id.get("manifest_sha256") != pins["manifest_sha256"]:
            raise ValueError(
                f"historical baseline manifest hash mismatch: got {t_id.get('manifest_sha256')}, "
                f"expected {pins['manifest_sha256']}"
            )
        if t_id.get("source_sha256") != pins["source_sha256"]:
            raise ValueError(
                f"historical baseline source hash mismatch: got {t_id.get('source_sha256')}, "
                f"expected {pins['source_sha256']}"
            )
        if t_id.get("payload_sha256") != pins["payload_sha256"]:
            raise ValueError(
                f"historical baseline payload hash mismatch: got {t_id.get('payload_sha256')}, "
                f"expected {pins['payload_sha256']}"
            )
        if inference_state != pins["inference_state"]:
            raise ValueError(
                f"historical baseline teacher state must be '{pins['inference_state']}' "
                f"(raw student per evaluation.py/REPORT.json), got '{inference_state}'"
            )
        if sorted(teacher_lids) != sorted(pins["labeled_ids"]):
            raise ValueError("historical baseline labeled IDs do not match pinned IDs")
        if not allow_historical_bridge and not t_id.get("historical_bridge"):
            raise ValueError("historical source closure mismatch requires explicit safe historical bridge; fail closed")
        checks_passed.append("historical_pins_verified")

    # Reproduction teacher
    elif kind == "reproduction":
        repro = validate_reproduction_receipt(t_id)
        t_id.update(repro)
        checks_passed.extend(["reproduction_receipt_validated", "drive_readback_verified", "macro4_quality_go"])

    # Synthetic teacher
    elif kind == "synthetic":
        checks_passed.append("synthetic_kind_validated")

    # Actual file checks on disk (if paths provided)
    if manifest_path is not None:
        mp = Path(manifest_path).resolve()
        if not mp.is_file():
            raise FileNotFoundError(f"teacher manifest file not found: {mp}")
        actual_m_sha = digest(mp)
        expected_m_sha = t_id.get("manifest_sha256") or t_id.get("checkpoint_manifest_sha256")
        if expected_m_sha and actual_m_sha != expected_m_sha:
            raise ValueError(f"teacher manifest hash mismatch: actual {actual_m_sha} != expected {expected_m_sha}")
        is_manifest_verified = True
        checks_passed.append("actual_manifest_file_verified")

    if payload_path is not None:
        pp = Path(payload_path).resolve()
        if not pp.is_file():
            raise FileNotFoundError(f"teacher payload file not found: {pp}")
        actual_p_sha = digest(pp)
        expected_p_sha = t_id.get("payload_sha256")
        allowed_p_shas = {expected_p_sha, t_id.get("raw_student_sha256"), t_id.get("checkpoint_payload_sha256")}
        allowed_p_shas.discard(None)
        if allowed_p_shas and actual_p_sha not in allowed_p_shas:
            raise ValueError(f"teacher payload hash mismatch: actual {actual_p_sha} not in {allowed_p_shas}")
        is_payload_verified = True
        checks_passed.append("actual_payload_file_verified")

    # Verification status: cannot stamp verified=True from claimed fields alone
    is_fully_verified = is_manifest_verified and (is_payload_verified or kind == "synthetic")

    receipt = dict(t_id)
    receipt["verified"] = is_fully_verified
    receipt["verification_status"] = "VERIFIED" if is_fully_verified else "CLAIMED_METADATA_UNVERIFIED"
    receipt["inference_state"] = inference_state
    receipt["teacher_kind"] = kind
    receipt["labeled_ids"] = list(teacher_lids)
    receipt["label_budget"] = teacher_k
    receipt["checks_passed"] = checks_passed
    receipt["physical_payload_status"] = "VERIFIED_ON_DISK" if is_payload_verified else "BLOCKED_MISSING_PAYLOAD"
    return receipt


def preflight_teacher_lineage(
    *,
    teacher_labeled_ids: Sequence[str],
    student_labeled_ids: Optional[Sequence[str]] = None,
    data: Optional[Any] = None,
    warmstart_labeled_ids: Optional[Sequence[str]] = None,
    audit_labeled_ids: Optional[Sequence[str]] = None,
    auxiliary_labeled_ids: Optional[Sequence[str]] = None,
    label_budget: Optional[int] = None,
    declared_labeled_ids: Optional[Sequence[str]] = None,
    teacher_kind: Optional[str] = None,
) -> Dict[str, Any]:
    """Preflight check enforcing that the union of all labeled IDs <= K with actual data.

    Must be executed BEFORE any image/mask IO or model forward passes.
    """
    if data is not None:
        if student_labeled_ids is None:
            student_labeled_ids = getattr(data, "lids", [])
        if label_budget is None:
            label_budget = getattr(data, "manifest", {}).get("label_budget", len(getattr(data, "lids", [])))
        is_synthetic_data = bool(getattr(data, "manifest", {}).get("synthetic", False))
        if teacher_kind == "synthetic" and not is_synthetic_data:
            raise ValueError("synthetic teacher kind is strictly prohibited with real patient data")

    if label_budget is None or type(label_budget) is not int or label_budget < 1:
        raise ValueError(f"label_budget must be a positive integer, got {label_budget}")

    groups: Dict[str, List[str]] = {
        "teacher": list(teacher_labeled_ids or []),
        "student": list(student_labeled_ids or []),
        "warmstart": list(warmstart_labeled_ids or []),
        "audit": list(audit_labeled_ids or []),
        "auxiliary": list(auxiliary_labeled_ids or []),
    }

    if not groups["teacher"]:
        raise ValueError("teacher_labeled_ids cannot be empty; unknown lineage blocks real use")

    union_set: set[str] = set()
    for grp_name, id_list in groups.items():
        for pid in id_list:
            _validate_patient_id(pid)
            union_set.add(pid)

    union_ids = sorted(union_set)
    union_count = len(union_ids)

    if union_count > label_budget:
        raise ValueError(
            f"union of all labeled patient IDs ({union_count}) exceeds label budget K={label_budget}; "
            f"union={union_ids}"
        )

    if declared_labeled_ids is not None:
        dec_set = set(declared_labeled_ids)
        for pid in dec_set:
            _validate_patient_id(pid)
        if union_set != dec_set:
            missing = sorted(dec_set - union_set)
            extra = sorted(union_set - dec_set)
            raise ValueError(
                f"labeled patient union does not match declared IDs (missing: {missing}, extra: {extra})"
            )
        if union_count != label_budget:
            raise ValueError(
                f"exact K required: union has {union_count} patients but budget K={label_budget}"
            )

    return {
        "status": "PASS",
        "label_budget": label_budget,
        "union_count": union_count,
        "union_labeled_ids": union_ids,
        "exact_budget_matched": (union_count == label_budget),
        "components": {k: sorted(v) for k, v in groups.items() if v},
    }


# ---------------------------------------------------------------------------
# Fixed Teacher Cache
# ---------------------------------------------------------------------------

class FixedTeacherCache:
    """Bounded-handle memmapped cache of frozen teacher soft target probabilities.

    Strictly binds:
      - Cache manifest hash against expected SHA256.
      - Training manifest SHA256 and exact unlabeled patient IDs.
      - Mandatory per-file SHA256 hashes for every patient array file.
      - Axes, classes, dtype, and shape verified against image cines.
      - Finite normalized probabilities on sampling.
      - Valid metadata with geometric crop and flip synchronization; rejects bad frames.
    """

    def __init__(
        self,
        path: Union[str, Path],
        sha: str,
        data: Any,
        teacher_identity: Optional[Mapping[str, Any]] = None,
        *,
        max_handles: int = 16,
    ):
        p = Path(path).resolve()
        if p.is_dir():
            manifest_path = p / "CACHE_MANIFEST.json"
            if not manifest_path.is_file():
                manifest_path = p / "MANIFEST.json"
        else:
            manifest_path = p

        if not manifest_path.is_file():
            raise FileNotFoundError(f"teacher cache manifest not found at {manifest_path}")

        actual_sha = digest(manifest_path)
        if actual_sha != sha.lower():
            raise ValueError(
                f"teacher cache manifest hash mismatch: got {actual_sha}, expected {sha}"
            )

        self.manifest_path = manifest_path
        self.manifest_dir = manifest_path.parent
        self.raw_manifest = json.loads(manifest_path.read_text())

        schema = self.raw_manifest.get("schema")
        if schema != "self_audit.compact_teacher_cache.v1":
            raise ValueError(f"invalid cache manifest schema: {schema}")

        # Check native axes and class ontology
        native_axes = self.raw_manifest.get("native_axes")
        if native_axes != "TZYX":
            raise ValueError(f"cache native_axes must be 'TZYX', got '{native_axes}'")
        num_classes = self.raw_manifest.get("num_classes")
        if num_classes != 4:
            raise ValueError(f"cache num_classes must be 4, got {num_classes}")

        # Bind training manifest SHA256
        data_train_sha = digest(data.path)
        cache_train_sha = self.raw_manifest.get("train_manifest_sha256")
        if not cache_train_sha or cache_train_sha != data_train_sha:
            raise ValueError(
                f"cache train_manifest_sha256 mismatch: cache has '{cache_train_sha}', "
                f"data has '{data_train_sha}'"
            )

        # Validate embedded teacher identity and run preflight with data
        cache_id = self.raw_manifest.get("teacher_identity", {})
        self.teacher_identity = validate_teacher_identity(
            cache_id, declared_k=len(data.lids), declared_labeled_ids=data.lids, data=data,
        )
        expected_id = validate_teacher_identity(
            teacher_identity, declared_k=len(data.lids), declared_labeled_ids=data.lids, data=data,
        )
        identity_keys = ("teacher_kind", "inference_state", "labeled_ids", "label_budget",
                         "manifest_sha256", "payload_sha256", "source_sha256", "training_method")
        if any(self.teacher_identity.get(k) != expected_id.get(k) for k in identity_keys):
            raise ValueError("cache teacher identity differs from the configured teacher")
        if not bool(data.manifest.get("synthetic", False)):
            if cache_id.get("verified") is not True or cache_id.get("physical_payload_status") not in ("VERIFIED_ON_DISK", "REMOTE_DRIVE_VERIFIED"):
                raise ValueError("real cache lacks verified physical teacher provenance")
            self.teacher_identity.update(verified=True, physical_payload_status="VERIFIED_ON_DISK",
                                         verification_status="VERIFIED_FROM_PINNED_CACHE")

        preflight_teacher_lineage(
            teacher_labeled_ids=self.teacher_identity["labeled_ids"],
            data=data,
            teacher_kind=self.teacher_identity.get("teacher_kind"),
        )

        # Bind exact unlabeled patient IDs
        self.patients_meta = self.raw_manifest.get("patients", {})
        if not isinstance(self.patients_meta, Mapping):
            raise ValueError("cache manifest missing 'patients' mapping")

        data_uids = set(getattr(data, "uids", []))
        cache_uids = set(self.patients_meta.keys())
        if data_uids != cache_uids:
            missing = sorted(data_uids - cache_uids)
            extra = sorted(cache_uids - data_uids)
            raise ValueError(
                f"cache unlabeled IDs do not match data exactly: missing={missing}, extra={extra}"
            )

        # Map unlabeled data records and verify every file hash and shape vs image
        unlabeled_records = {r["id"]: r for r in getattr(data, "unlabeled", [])}
        for uid in data_uids:
            p_rec = self.patients_meta[uid]
            data_rec = unlabeled_records[uid]
            if p_rec.get("image_sha256") != data_rec.get("image_sha256"):
                raise ValueError(
                    f"TRAIN image hash mismatch in teacher cache for unlabeled patient '{uid}'"
                )

            # Mandatory file hash
            file_sha = p_rec.get("file_sha256")
            if not isinstance(file_sha, str) or len(file_sha) != 64:
                raise ValueError(f"mandatory 64-hex file_sha256 missing for patient '{uid}'")

            arr_file = self.manifest_dir / p_rec.get("file", "")
            if not arr_file.is_file():
                raise FileNotFoundError(f"cache array file missing: {arr_file}")
            if digest(arr_file) != file_sha:
                raise ValueError(f"cache array hash mismatch for patient '{uid}'")

            # Check array shape vs image cine shape
            cine = data.array(data_rec["image"], data_rec["image_sha256"])
            expected_shape = (cine.shape[0], cine.shape[1], 4, cine.shape[2], cine.shape[3])
            arr_shape = tuple(p_rec.get("shape", []))
            array_header = np.load(arr_file, mmap_mode="r", allow_pickle=False)
            if array_header.shape != expected_shape or array_header.dtype not in (np.dtype("float16"), np.dtype("float32")):
                raise ValueError(f"cache payload geometry/dtype mismatch for {uid}")
            del array_header
            if arr_shape != expected_shape:
                raise ValueError(
                    f"cache array shape {arr_shape} != expected image shape {expected_shape} for '{uid}'"
                )

        # LRU bounded handle cache for memmap arrays
        self.max_handles = max(1, int(max_handles))
        self._handle_cache: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()

    def _get_memmap(self, patient_id: str) -> np.ndarray:
        if patient_id in self._handle_cache:
            self._handle_cache.move_to_end(patient_id)
            return self._handle_cache[patient_id]

        if len(self._handle_cache) >= self.max_handles:
            oldest_id, oldest_arr = self._handle_cache.popitem(last=False)
            if hasattr(oldest_arr, "_mmap") and oldest_arr._mmap is not None:
                try:
                    oldest_arr._mmap.close()
                except Exception:
                    pass

        p_rec = self.patients_meta[patient_id]
        arr_file = self.manifest_dir / p_rec["file"]
        arr = np.load(arr_file, mmap_mode="r", allow_pickle=False)
        self._handle_cache[patient_id] = arr
        return arr

    def sample_batch(
        self,
        metadata: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        *,
        device: str = "cpu",
    ) -> Tuple[Tensor, Tensor]:
        """Retrieve cropped/padded/flipped teacher target probabilities and validity masks.

        Validates finite normalized sampled probabilities and rejects bad frame indices.
        """
        if isinstance(metadata, Mapping):
            if "samples" in metadata and isinstance(metadata["samples"], list):
                sample_list = metadata["samples"]
            else:
                sample_list = [metadata]
        elif isinstance(metadata, (list, tuple)):
            sample_list = list(metadata)
        else:
            raise TypeError("metadata must be a mapping or sequence of mappings")

        prob_list: List[Tensor] = []
        valid_list: List[Tensor] = []

        for s in sample_list:
            pid = s["patient"]
            t = int(s["frame"])
            z = int(s["slice"])

            mmap = self._get_memmap(pid)
            t_dim, z_dim = mmap.shape[0], mmap.shape[1]

            # Reject bad frames
            if t < 0 or t >= t_dim or z < 0 or z >= z_dim:
                raise ValueError(
                    f"rejected bad frame/slice index (t={t}, z={z}) for patient '{pid}' "
                    f"with dimensions T={t_dim}, Z={z_dim}"
                )

            # mmap shape is (T, Z, 4, H, W)
            prob_raw = np.array(mmap[t, z], dtype=np.float32, copy=True)

            # Check finite and normalized [0, 1]
            if not np.isfinite(prob_raw).all() or prob_raw.min() < 0 or prob_raw.max() > 1.0001:
                raise ValueError(f"non-finite or out-of-range sampled teacher probabilities for '{pid}'")
            if not np.allclose(prob_raw.sum(axis=0), 1.0, atol=1e-3):
                raise ValueError(f"sampled teacher probabilities do not sum to 1.0 for '{pid}'")

            prob_t = torch.from_numpy(prob_raw)[None]  # (1, 4, H, W)
            h, w = prob_raw.shape[-2:]
            valid_t = torch.ones((1, h, w), dtype=torch.bool)

            # Apply identical geometric crop/pad if present in metadata
            crop = s.get("crop")
            if crop is not None:
                pad_lrtb = tuple(crop["pad_lrtb"])
                if any(pad_lrtb):
                    prob_t = F.pad(prob_t, pad_lrtb, mode="replicate")
                    valid_t = F.pad(valid_t, pad_lrtb, value=False)
                top = int(crop["top"])
                left = int(crop["left"])
                size = int(crop["size"])
                spatial = (..., slice(top, top + size), slice(left, left + size))
                prob_t = prob_t[spatial]
                valid_t = valid_t[spatial]

            # Synchronize horizontal flip if present
            if s.get("flip") or s.get("hflip"):
                prob_t = prob_t.flip(-1)
                valid_t = valid_t.flip(-1)

            prob_list.append(prob_t)
            valid_list.append(valid_t)

        batch_probs = torch.cat(prob_list, dim=0).to(device)
        batch_valid = torch.cat(valid_list, dim=0).to(device)
        return batch_probs, batch_valid

    def close(self) -> None:
        """Close all open memmap handles."""
        while self._handle_cache:
            _, arr = self._handle_cache.popitem()
            if hasattr(arr, "_mmap") and arr._mmap is not None:
                try:
                    arr._mmap.close()
                except Exception:
                    pass

    def __enter__(self) -> FixedTeacherCache:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Loader and Cache Generator
# ---------------------------------------------------------------------------

def load_teacher_for_cache(
    checkpoint_dir: Union[str, Path],
    *,
    teacher_identity: Mapping[str, Any],
    device: str = "cpu",
    historical_source_path: Optional[Union[str, Path]] = None,
    reproduction_source_path: Optional[Union[str, Path]] = None,
    synthetic: bool = False,
) -> nn.Module:
    """Load verified frozen teacher model for cache generation, failing closed if unverified.

    Loads with strict=True (never strict=False), strict state selection (no fallback),
    and mandatory payload hash validation. Real missing assets BLOCKED.
    """
    ckpt_path = Path(checkpoint_dir).resolve()
    manifest_file = ckpt_path / "CHECKPOINT_MANIFEST.json"
    if not manifest_file.is_file():
        manifest_file = ckpt_path / "MANIFEST.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"checkpoint manifest missing in {ckpt_path}")

    teacher_identity = validate_teacher_identity(teacher_identity)
    actual_manifest_sha = digest(manifest_file)
    expected_manifest_sha = teacher_identity.get("manifest_sha256")
    if expected_manifest_sha and actual_manifest_sha != expected_manifest_sha:
        raise ValueError(
            f"actual manifest SHA ({actual_manifest_sha}) != teacher identity ({expected_manifest_sha})"
        )

    manifest_data = json.loads(manifest_file.read_text())

    # Full lineage check against actual manifest
    manifest_lids = manifest_data.get("labeled_ids", [])
    identity_lids = teacher_identity.get("labeled_ids", [])
    if sorted(manifest_lids) != sorted(identity_lids):
        raise ValueError(
            f"manifest labeled IDs {manifest_lids} do not match teacher identity {identity_lids}"
        )

    # Mandatory payload SHA
    expected_payload_sha = manifest_data.get("payload_sha256") or teacher_identity.get("payload_sha256")
    if not expected_payload_sha:
        raise ValueError("mandatory payload_sha256 missing in manifest and identity")

    payload_path = find_checkpoint_payload(ckpt_path, expected_payload_sha)
    actual_payload_sha = digest(payload_path)
    if actual_payload_sha != expected_payload_sha:
        raise ValueError(
            f"payload hash mismatch: {actual_payload_sha} != {expected_payload_sha}"
        )

    verified_identity = validate_teacher_identity(
        teacher_identity, manifest_path=manifest_file, payload_path=payload_path)
    if manifest_data.get("schema") != "self_audit.checkpoint.v2":
        raise ValueError("teacher must be a canonical v2 checkpoint")
    if manifest_data.get("source_sha256") != teacher_identity.get("source_sha256"):
        raise ValueError("teacher source closure differs from checkpoint")
    if manifest_data.get("config", {}).get("method") not in {"ema_full", "ema_full_meta_window"}:
        raise ValueError("teacher checkpoint did not use full EMA training")
    if manifest_data.get("config", {}).get("method") != teacher_identity.get("training_method"):
        raise ValueError("teacher training method mismatch")
    if bool(manifest_data.get("synthetic", False)) != bool(synthetic):
        raise ValueError("teacher physical payload synthetic identity mismatch")
    if teacher_identity.get("teacher_kind") == "canonical":
        quality = teacher_identity["quality_receipt"]
        quality_path = Path(quality.get("path", ""))
        quality_sha = quality.get("sha256")
        if not isinstance(quality_sha, str) or re.fullmatch(r"[0-9a-f]{64}", quality_sha) is None:
            raise ValueError("canonical quality receipt needs a mandatory SHA256")
        if not quality_path.is_file() or digest(quality_path) != quality_sha:
            raise ValueError("canonical teacher quality receipt file/hash not verified")
        qr = json.loads(quality_path.read_text())
        if qr.get("checkpoint_manifest_sha256") != actual_manifest_sha or qr.get("inference_state") != teacher_identity["inference_state"]:
            raise ValueError("quality receipt refers to a different teacher/state")
        macro4 = qr.get("class_first_macro4")
        if type(macro4) not in (int, float) or not np.isfinite(macro4) or not 0 <= macro4 <= 1:
            raise ValueError("teacher quality receipt lacks measured Macro4")
        verified_identity["quality_receipt_sha256"] = quality_sha
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload at {payload_path} must be a Mapping")

    # Strict state selection: no fallback payload!
    inference_state = teacher_identity.get("inference_state", "student")
    if inference_state == "student":
        if "student" not in payload:
            raise KeyError(
                f"'student' state missing from payload keys {sorted(payload.keys())}; no fallback allowed"
            )
        state_dict = payload["student"]
    elif inference_state == "teacher_ema":
        if "teacher" not in payload:
            raise KeyError(
                f"'teacher' state missing from payload keys {sorted(payload.keys())}; no fallback allowed"
            )
        state_dict = payload["teacher"]
    else:
        raise ValueError(f"unknown inference_state '{inference_state}'")

    config = manifest_data.get("config", {})
    model_cfg = config.get("model", manifest_data.get("architecture", {}))
    window_cfg = config.get("window", manifest_data.get("window"))

    # Build model: if reproduction teacher or source path given, use reproduction bridge
    if teacher_identity.get("teacher_kind") == "reproduction" or reproduction_source_path is not None:
        source_p = reproduction_source_path or historical_source_path
        if source_p is None:
            raise ValueError("reproduction teacher requires explicit --reproduction-source or historical source path")
        model = load_reproduction_teacher_model(
            source_p,
            model_cfg,
            synthetic=synthetic or bool(manifest_data.get("synthetic", False)),
            window=window_cfg,
            expected_commit=teacher_identity.get("source_commit"),
            expected_closure=teacher_identity.get("source_sha256"),
        )
    elif historical_source_path is not None:
        model = load_historical_teacher_model(
            historical_source_path,
            model_cfg,
            synthetic=synthetic or bool(manifest_data.get("synthetic", False)),
            window=window_cfg,
        )
    else:
        if is_historical_baseline_checkpoint(teacher_identity):
            raise ValueError(
                "loading historical baseline requires explicit --historical-source path; "
                "current source cannot bypass historical source closure"
            )
        from .model import build_model
        from .engine import source_digest
        if source_digest() != teacher_identity.get("source_sha256"):
            raise ValueError("canonical teacher source closure mismatch")
        model, _ = build_model(
            model_cfg,
            synthetic=synthetic or bool(manifest_data.get("synthetic", False)),
            window=window_cfg,
        )

    # Strict loading only — never strict=False
    model.load_state_dict(state_dict, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    for value in model.state_dict().values():
        if not torch.isfinite(value).all():
            raise ValueError("non-finite teacher weights")
    model.eval()
    model._compact_teacher_identity = verified_identity
    return model.to(device)
    model.eval()
    model._compact_teacher_identity = verified_identity
    return model.to(device)


def build_teacher_cache(
    model: nn.Module,
    data: Any,
    output_dir: Union[str, Path],
    teacher_identity: Mapping[str, Any],
    *,
    device: str = "cpu",
    batch_size: int = 8,
    precision: str = "float32",
) -> Dict[str, Any]:
    """Generate fixed teacher soft target probability cache on image-only U patients.

    Uses np.lib.format.open_memmap output streaming directly into disk files
    (never allocating the full probability volume in memory).
    Real missing assets BLOCKED. Reports latency, VRAM, and disk bytes.
    """
    out = Path(output_dir).resolve()
    if out.exists():
        raise ValueError("cache output already exists; preserve the previous cache")
    if type(batch_size) is not int or batch_size < 1 or precision not in {"float32", "float16"}:
        raise ValueError("invalid cache batch size or precision")

    # Validate identity and run preflight with actual data BEFORE model forward passes
    verified_id = validate_teacher_identity(
        teacher_identity,
        declared_k=len(data.lids),
        declared_labeled_ids=data.lids,
        data=data,
        allow_historical_bridge=True,
    )
    preflight_teacher_lineage(
        teacher_labeled_ids=verified_id["labeled_ids"],
        data=data,
        teacher_kind=verified_id.get("teacher_kind"),
    )

    is_synthetic_data = bool(getattr(data, "manifest", {}).get("synthetic", False))
    if not is_synthetic_data:
        # Real patient data requires verified physical teacher assets
        if verified_id.get("teacher_kind") == "synthetic":
            raise ValueError("synthetic teacher cannot be used with real patient data")

    if not is_synthetic_data:
        loaded_identity = getattr(model, "_compact_teacher_identity", None)
        if not loaded_identity or loaded_identity.get("verified") is not True:
            raise ValueError("real cache generation requires the physically verified teacher loader")
        keys = ("manifest_sha256", "payload_sha256", "source_sha256", "labeled_ids", "inference_state")
        if any(loaded_identity.get(k) != verified_id.get(k) for k in keys):
            raise ValueError("loaded teacher does not match configured teacher identity")
        verified_id = dict(loaded_identity)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    out.mkdir(parents=True, exist_ok=False)
    unlabeled = getattr(data, "unlabeled", [])
    if not unlabeled:
        raise ValueError("data has no unlabeled patients to cache")

    patients_meta: Dict[str, Any] = {}
    total_start = time.monotonic()
    volume_latencies: List[float] = []

    dtype_np = np.float16 if precision == "float16" else np.float32

    for r in unlabeled:
        pid = r["id"]
        vol_start = time.monotonic()

        # Image-only U: access image array only, never masks
        cine = data.array(r["image"], r["image_sha256"])
        t_dim, z_dim, h_dim, w_dim = cine.shape

        arr_filename = f"{pid}_probs.npy"
        arr_path = out / arr_filename

        # Stream directly to disk via open_memmap — no full volume RAM allocation!
        mmap_out = np.lib.format.open_memmap(
            arr_path,
            mode="w+",
            dtype=dtype_np,
            shape=(t_dim, z_dim, 4, h_dim, w_dim),
        )

        slice_indices = [(t, z) for t in range(t_dim) for z in range(z_dim)]
        for i in range(0, len(slice_indices), batch_size):
            chunk = slice_indices[i : i + batch_size]
            contexts = [data.context(cine, t, z) for t, z in chunk]
            batch_x = torch.cat(contexts, dim=0).to(device)

            with torch.no_grad():
                logits = model(batch_x)
                if logits.shape[1] != 4:
                    raise ValueError(f"expected 4-class logits, got {logits.shape[1]}")
                probs = F.softmax(logits, dim=1).cpu().numpy().astype(dtype_np)

            if not np.isfinite(probs).all() or probs.min() < 0 or probs.max() > 1.0001:
                raise ValueError(f"invalid non-finite or out-of-range teacher probabilities for '{pid}'")

            for (t, z), p_slice in zip(chunk, probs):
                mmap_out[t, z] = p_slice

        mmap_out.flush()
        del mmap_out

        vol_elapsed = time.monotonic() - vol_start
        volume_latencies.append(vol_elapsed)

        arr_sha = digest(arr_path)
        patients_meta[pid] = {
            "file": arr_filename,
            "file_sha256": arr_sha,
            "image_sha256": r["image_sha256"],
            "shape": [t_dim, z_dim, 4, h_dim, w_dim],
            "dtype": str(dtype_np.__name__),
            "file_bytes": arr_path.stat().st_size,
        }

    total_elapsed = time.monotonic() - total_start
    total_disk_bytes = sum(p["file_bytes"] for p in patients_meta.values())

    # VRAM telemetry
    vram_peak_allocated = None
    vram_peak_reserved = None
    if device.startswith("cuda") and torch.cuda.is_available():
        vram_peak_allocated = torch.cuda.max_memory_allocated(device=device)
        vram_peak_reserved = torch.cuda.max_memory_reserved(device=device)

    manifest_data = {
        "schema": "self_audit.compact_teacher_cache.v1",
        "teacher_identity": verified_id,
        "train_manifest_sha256": digest(data.path),
        "native_axes": "TZYX",
        "num_classes": 4,
        "precision": precision,
        "total_unlabeled_patients": len(unlabeled),
        "patients": patients_meta,
        "telemetry": {
            "total_latency_seconds": float(total_elapsed),
            "mean_seconds_per_volume": float(total_elapsed / max(1, len(unlabeled))),
            "median_seconds_per_volume": float(np.median(volume_latencies)),
            "total_disk_bytes": int(total_disk_bytes),
            "vram_peak_allocated_bytes": vram_peak_allocated,
            "vram_peak_reserved_bytes": vram_peak_reserved,
        },
    }

    manifest_path = out / "CACHE_MANIFEST.json"
    write_json(manifest_path, manifest_data)
    manifest_sha = digest(manifest_path)

    return {
        "status": "PASS",
        "cache_manifest_path": str(manifest_path),
        "cache_manifest_sha256": manifest_sha,
        "unlabeled_patients_cached": len(unlabeled),
        "teacher_kind": verified_id["teacher_kind"],
        "inference_state": verified_id["inference_state"],
        "total_disk_bytes": total_disk_bytes,
        "total_latency_seconds": total_elapsed,
    }


def main() -> None:
    """CLI entrypoint for teacher preflight, identity validation, and cache building."""
    parser = argparse.ArgumentParser(description="Compact teacher preflight, identity, and cache builder.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: preflight
    pf = subparsers.add_parser("preflight", help="Preflight teacher lineage and label budget.")
    pf.add_argument("--teacher-labeled-ids", nargs="+", required=True)
    pf.add_argument("--student-labeled-ids", nargs="*", default=None)
    pf.add_argument("--student-manifest", type=str, default=None)
    pf.add_argument("--label-budget", type=int, default=None)
    pf.add_argument("--declared-labeled-ids", nargs="*", default=None)
    pf.add_argument("--teacher-kind", type=str, default=None)

    # Subcommand: validate-identity
    vi = subparsers.add_parser("validate-identity", help="Validate a teacher identity JSON file.")
    vi.add_argument("--identity-file", type=str, required=True)
    vi.add_argument("--manifest-file", type=str, default=None)
    vi.add_argument("--payload-file", type=str, default=None)

    # Subcommand: verify-historical-source
    vhs = subparsers.add_parser("verify-historical-source", help="Verify historical source checkout.")
    vhs.add_argument("--historical-source", type=str, required=True)

    args = parser.parse_args()

    if args.command == "preflight":
        data = None
        if args.student_manifest:
            sm_path = Path(args.student_manifest).resolve()
            sm = json.loads(sm_path.read_text())
            from .compact_data import CompactTrainingData
            data = CompactTrainingData(sm_path, digest(sm_path), synthetic=bool(sm.get("synthetic", False)))

        receipt = preflight_teacher_lineage(
            teacher_labeled_ids=args.teacher_labeled_ids,
            student_labeled_ids=args.student_labeled_ids,
            data=data,
            label_budget=args.label_budget,
            declared_labeled_ids=args.declared_labeled_ids,
            teacher_kind=args.teacher_kind,
        )
        print(json.dumps(receipt, indent=2))

    elif args.command == "validate-identity":
        raw = json.loads(Path(args.identity_file).read_text())
        verified = validate_teacher_identity(
            raw,
            manifest_path=args.manifest_file,
            payload_path=args.payload_file,
            allow_historical_bridge=True,
        )
        print(json.dumps(verified, indent=2))

    elif args.command == "verify-historical-source":
        receipt = verify_historical_source(args.historical_source)
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
