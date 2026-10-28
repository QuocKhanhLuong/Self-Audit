import hashlib
import json
from pathlib import Path

import pytest
import torch

from self_audit.volume.legacy_compat import load_legacy_checkpoint, verify_legacy_manifest


def _canonical_map():
    path = Path("reports/teacher_memory_full_20261024/completed_20261025/ASSET_LINEAGE.json")
    return json.loads(path.read_text())["canonical_runtime_file_sha256"]


@pytest.fixture
def synthetic_current_map(monkeypatch):
    """Unit trust root only; this does not certify compatibility with old source."""
    import self_audit.volume.legacy_compat as compat
    current = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in _canonical_map()}
    monkeypatch.setattr(compat, "_trusted_canonical_file_map", lambda: current)
    return current


def test_current_checkout_rejects_immutable_historical_source_map():
    import self_audit.volume.legacy_compat as compat
    with pytest.raises(ValueError, match="canonical module differs"):
        compat._verify_canonical_files(_canonical_map())


def _model_map():
    return {"name": "canonical", "commit": "deadbeef"}


def _base_manifest(payload_sha, *, model=None):
    return {
        "schema": "self_audit.checkpoint.v2",
        "source_sha256": "a" * 64,
        "data_sha256": "b" * 64,
        "payload_sha256": payload_sha,
        "backbone": model or _model_map(),
    }


def test_legacy_compat_requires_all_external_pins(tmp_path):
    folder = tmp_path / "legacy"
    folder.mkdir()
    payload = b"payload"
    folder.joinpath("weights.pt").write_bytes(payload)
    metadata = _base_manifest(hashlib.sha256(payload).hexdigest())
    folder.joinpath("MANIFEST.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="explicit source_sha256"):
        verify_legacy_manifest(folder)
    with pytest.raises(ValueError, match="explicit data_sha256"):
        verify_legacy_manifest(folder, expected_source_sha256="a" * 64)


def test_legacy_compat_checks_external_canonical_files_not_manifest_claims(tmp_path, synthetic_current_map):
    folder = tmp_path / "legacy"
    folder.mkdir()
    payload = b"payload"
    folder.joinpath("weights.pt").write_bytes(payload)
    metadata = _base_manifest(hashlib.sha256(payload).hexdigest())
    folder.joinpath("MANIFEST.json").write_text(json.dumps(metadata, sort_keys=True))
    accepted = verify_legacy_manifest(
        folder,
        expected_source_sha256="a" * 64,
        expected_data_sha256="b" * 64,
        expected_payload_sha256=metadata["payload_sha256"],
        expected_model=_model_map(),
        expected_canonical_modules=synthetic_current_map,
    )
    assert accepted["schema"] == "self_audit.checkpoint.v2"
    with pytest.raises(ValueError, match="incomplete"):
        verify_legacy_manifest(
            folder,
            expected_source_sha256="a" * 64,
            expected_data_sha256="b" * 64,
            expected_payload_sha256=metadata["payload_sha256"],
            expected_model=_model_map(),
            expected_canonical_modules={"self_audit/model.py": synthetic_current_map["self_audit/model.py"]},
        )


def test_legacy_loader_returns_verified_student_teacher_and_optimizer(tmp_path, synthetic_current_map):
    from self_audit.model import build_model

    student, _ = build_model({"backbone": "tiny_test"}, synthetic=True)
    teacher, _ = build_model({"backbone": "tiny_test"}, synthetic=True)
    body = [p for n, p in student.named_parameters() if not n.startswith("head.")]
    head = [p for n, p in student.named_parameters() if n.startswith("head.")]
    optimizers = [torch.optim.AdamW(body, lr=3e-4, weight_decay=1e-4),
                  torch.optim.SGD(head, lr=3e-3, momentum=0.0, weight_decay=0.0)]
    payload = {"student": student.state_dict(), "teacher": teacher.state_dict(),
               "optimizers": [o.state_dict() for o in optimizers]}
    torch.save(payload, tmp_path / "weights.pt")
    payload_sha = hashlib.sha256((tmp_path / "weights.pt").read_bytes()).hexdigest()
    manifest = _base_manifest(payload_sha, model={"backbone": "tiny_test"})
    manifest.update({"config": {"model": {"backbone": "tiny_test"}, "window": {"mode": "none"},
                                 "body_lr": 3e-4, "head_lr": 3e-3, "weight_decay": 1e-4},
                     "synthetic": True})
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    loaded_student, loaded_teacher, loaded_optimizers, loaded_manifest = load_legacy_checkpoint(
        tmp_path, manifest_sha256=manifest_sha, source_sha256="a" * 64,
        data_sha256="b" * 64, payload_sha256=payload_sha,
        expected_model={"backbone": "tiny_test"}, expected_canonical_modules=synthetic_current_map,
        return_optimizers=True,
    )
    assert loaded_manifest["schema"] == "self_audit.checkpoint.v2"
    assert loaded_student.training is False and loaded_teacher.training is False
    assert len(loaded_optimizers) == 2
    assert all(not p.requires_grad for p in loaded_teacher.parameters())
    assert all(torch.equal(a, b) for a, b in zip(student.parameters(), loaded_student.parameters(), strict=True))


def test_legacy_loader_rejects_canonical_source_change(tmp_path, monkeypatch):
    # The loader compares the pinned external map to actual files.  Simulate a
    # changed canonical model source without editing shared canonical modules.
    import self_audit.volume.legacy_compat as compat

    original = compat._file_sha256
    target = "self_audit/model.py"

    def altered(path):
        digest = original(path)
        if str(path).endswith(target):
            return "0" * 64
        return digest

    monkeypatch.setattr(compat, "_file_sha256", altered)
    with pytest.raises(ValueError, match="canonical module differs"):
        compat._verify_canonical_files(_canonical_map())
