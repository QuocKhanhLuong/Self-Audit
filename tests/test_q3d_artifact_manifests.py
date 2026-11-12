from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from self_audit.candidate_q3d.artifacts import (
    file_sha256,
    load_stage_manifest,
    stage_manifest_file_ref,
    write_stage_manifest,
)


IDENTITY = {
    "dataset": "acdc",
    "protocol": "A",
    "k": 5,
    "seed": 42,
    "train_manifest_sha256": hashlib.sha256(b"training cohort").hexdigest(),
}
T_SOURCE = {
    "teacher_architecture": "nnfoundation_resenc_l",
    "teacher_source_commit": "pinned",
    "encoder_strides": [[1, 1, 1], [2, 2, 2], [1, 2, 2], [1, 2, 2]],
    "phase_offsets": {"encoder_stride8": [0.0, 0.0, 0.0]},
}
Q_SOURCE = {
    "teacher_architecture": "nnfoundation_resenc_l",
    "feature_source": "encoder_stride8",
    "feature_stride": 8,
    "encoder_strides": [[1, 1, 1], [2, 2, 2], [1, 2, 2], [1, 2, 2]],
    "phase_offsets": {"encoder_stride8": [0.0, 0.0, 0.0]},
}
P_SOURCE = {"physiology_cache": "TRAIN_U", "feature_source": "encoder_stride8"}


def _file(path: Path, contents: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def _t_manifest(tmp_path: Path) -> tuple[Path, dict]:
    checkpoint = _file(tmp_path / "T.pt", b"teacher snapshot")
    manifest_path = tmp_path / "T.artifact.json"
    manifest = write_stage_manifest(
        manifest_path,
        stage="T",
        checkpoint_path=checkpoint,
        identity=IDENTITY,
        source_config=T_SOURCE,
        dependencies={},
        content_hashes={"teacher_state_sha256": hashlib.sha256(b"teacher state").hexdigest()},
    )
    return manifest_path, manifest


def _tqp_manifests(tmp_path: Path) -> tuple[Path, Path, Path]:
    t_path, _ = _t_manifest(tmp_path)
    t_ref = stage_manifest_file_ref(t_path)
    q_path = tmp_path / "Q.artifact.json"
    write_stage_manifest(
        q_path,
        stage="Q",
        checkpoint_path=_file(tmp_path / "Q.pt", b"frozen tokenizer snapshot"),
        identity=IDENTITY,
        source_config=Q_SOURCE,
        dependencies={"T": t_ref},
        content_hashes={
            "teacher_state_sha256": hashlib.sha256(b"teacher state").hexdigest(),
            "codebook_sha256": hashlib.sha256(b"frozen C0").hexdigest(),
            "sampling_phase_sha256": hashlib.sha256(b"measured sampling phase").hexdigest(),
        },
    )
    p_path = tmp_path / "P.artifact.json"
    write_stage_manifest(
        p_path,
        stage="P",
        checkpoint_path=_file(tmp_path / "P.pt", b"physiology snapshot"),
        identity=IDENTITY,
        source_config=P_SOURCE,
        dependencies={"T": t_ref, "Q": stage_manifest_file_ref(q_path)},
        content_hashes={
            "teacher_state_sha256": hashlib.sha256(b"teacher state").hexdigest(),
            "codebook_sha256": hashlib.sha256(b"frozen C0").hexdigest(),
            "cache_sha256": hashlib.sha256(b"canonical physiology cache").hexdigest(),
        },
    )
    return t_path, q_path, p_path


def test_stage_manifests_pin_checkpoint_bytes_and_source_identity(tmp_path: Path) -> None:
    t_path, q_path, p_path = _tqp_manifests(tmp_path)
    t_manifest = load_stage_manifest(t_path, identity=IDENTITY, source_config=T_SOURCE)
    q_manifest = load_stage_manifest(q_path, identity=IDENTITY, source_config=Q_SOURCE)
    p_manifest = load_stage_manifest(p_path, identity=IDENTITY, source_config=P_SOURCE)

    assert t_manifest["checkpoint"]["sha256"] == file_sha256(tmp_path / "T.pt")
    assert q_manifest["dependencies"]["T"] == stage_manifest_file_ref(t_path)
    assert p_manifest["dependencies"]["Q"] == stage_manifest_file_ref(q_path)
    assert p_manifest["content_hashes"]["cache_sha256"] == hashlib.sha256(
        b"canonical physiology cache",
    ).hexdigest()
    assert len(p_manifest["payload_sha256"]) == 64


def test_loading_a_manifest_rejects_checkpoint_byte_changes(tmp_path: Path) -> None:
    t_path, _, _ = _tqp_manifests(tmp_path)
    (tmp_path / "T.pt").write_bytes(b"changed teacher snapshot")
    with pytest.raises(ValueError, match="checkpoint file SHA-256 mismatch"):
        load_stage_manifest(t_path, identity=IDENTITY)


def test_loading_a_later_stage_recursively_rejects_changed_parent_manifest(tmp_path: Path) -> None:
    _, _, p_path = _tqp_manifests(tmp_path)
    q_path = tmp_path / "Q.artifact.json"
    q_path.write_text(q_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="dependency manifest file SHA-256 mismatch"):
        load_stage_manifest(p_path, identity=IDENTITY)


def test_manifest_loader_rejects_cohort_source_or_payload_sha_mismatch(tmp_path: Path) -> None:
    t_path, _, _ = _tqp_manifests(tmp_path)
    other_identity = {**IDENTITY, "protocol": "B"}
    with pytest.raises(ValueError, match="identity mismatch"):
        load_stage_manifest(t_path, identity=other_identity)
    with pytest.raises(ValueError, match="source_config differs"):
        load_stage_manifest(t_path, identity=IDENTITY, source_config={"teacher_architecture": "other"})

    raw = json.loads(t_path.read_text(encoding="utf-8"))
    raw["content_hashes"]["teacher_state_sha256"] = hashlib.sha256(b"edited state").hexdigest()
    t_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical payload SHA-256 mismatch"):
        load_stage_manifest(t_path, identity=IDENTITY)


def test_writer_requires_correct_stage_dependencies_and_is_immutable(tmp_path: Path) -> None:
    t_path, t_manifest = _t_manifest(tmp_path)
    q_checkpoint = _file(tmp_path / "Q.pt", b"tokenizer")
    with pytest.raises(ValueError, match="exactly the matching T"):
        write_stage_manifest(
            tmp_path / "bad-Q.artifact.json",
            stage="Q",
            checkpoint_path=q_checkpoint,
            identity=IDENTITY,
            source_config=Q_SOURCE,
            dependencies={},
            content_hashes={
                "teacher_state_sha256": t_manifest["content_hashes"]["teacher_state_sha256"],
                "codebook_sha256": hashlib.sha256(b"C0").hexdigest(),
                "sampling_phase_sha256": hashlib.sha256(b"measured sampling phase").hexdigest(),
            },
        )

    with pytest.raises(ValueError, match="sampling_phase_sha256"):
        write_stage_manifest(
            tmp_path / "Q-missing-phase.artifact.json",
            stage="Q",
            checkpoint_path=q_checkpoint,
            identity=IDENTITY,
            source_config=Q_SOURCE,
            dependencies={"T": stage_manifest_file_ref(t_path)},
            content_hashes={
                "teacher_state_sha256": t_manifest["content_hashes"]["teacher_state_sha256"],
                "codebook_sha256": hashlib.sha256(b"C0").hexdigest(),
            },
        )

    with pytest.raises(FileExistsError, match="immutable"):
        write_stage_manifest(
            t_path,
            stage="T",
            checkpoint_path=tmp_path / "T.pt",
            identity=IDENTITY,
            source_config=T_SOURCE,
            dependencies={},
            content_hashes={"teacher_state_sha256": hashlib.sha256(b"teacher state").hexdigest()},
        )


def test_manifest_reader_rejects_wrong_dependency_stage_and_sha(tmp_path: Path) -> None:
    t_path, _ = _t_manifest(tmp_path)
    t_ref = stage_manifest_file_ref(t_path)
    q_path = tmp_path / "Q.artifact.json"
    write_stage_manifest(
        q_path,
        stage="Q",
        checkpoint_path=_file(tmp_path / "Q.pt", b"tokenizer"),
        identity=IDENTITY,
        source_config=Q_SOURCE,
        dependencies={"T": t_ref},
        content_hashes={
            "teacher_state_sha256": hashlib.sha256(b"teacher state").hexdigest(),
            "codebook_sha256": hashlib.sha256(b"C0").hexdigest(),
            "sampling_phase_sha256": hashlib.sha256(b"measured sampling phase").hexdigest(),
        },
    )
    raw = json.loads(q_path.read_text(encoding="utf-8"))
    raw["dependencies"]["T"]["sha256"] = hashlib.sha256(b"not the T manifest").hexdigest()
    # Reseal the edited payload to reach recursive file-reference validation.
    from self_audit.candidate_q3d.artifacts import _canonical_sha256

    raw["payload_sha256"] = _canonical_sha256({key: value for key, value in raw.items() if key != "payload_sha256"})
    q_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="dependency manifest file SHA-256 mismatch"):
        load_stage_manifest(q_path, identity=IDENTITY)
