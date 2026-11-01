import json

import pytest

from self_audit import diagnostic_protocol as p


@pytest.fixture
def round_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "source_snapshot", lambda: {"git_head": "fixture", "files": {"code.py": "abc"}})
    image = tmp_path / "image.npy"
    image.write_bytes(b"synthetic image bytes")
    assets = {"image": {"path": str(image), "sha256": p.sha256(image)}}
    directory = tmp_path / "round"
    return directory, assets, image


def freeze_a(setup):
    directory, assets, image = setup
    p.create_lock(directory, "A", {"fixture": True, "lambdas": [0, .5, 1]}, assets)
    output = directory / "a.npy"
    output.write_bytes(b"fixture predictions A")
    p.freeze_predictions(directory, "A", [output], {"fixture": True})
    return output


def freeze_b(setup):
    directory, assets, image = setup
    p.create_lock(directory, "B", {"fixture": True, "rho": .01}, assets)
    output = directory / "b.npy"
    output.write_bytes(b"fixture predictions B")
    p.freeze_predictions(directory, "B", [output], {"fixture": True})
    return output


def test_round_requires_both_predictions_before_any_gt(round_setup):
    directory, assets, _ = round_setup
    freeze_a(round_setup)
    with pytest.raises(FileNotFoundError):
        p.authorize_score(directory, "A")
    freeze_b(round_setup)
    result = p.authorize_score(directory, "A")
    assert result["both_protocols_locked_before_first_GT"]
    assert p.authorize_score(directory, "B") == result
    with pytest.raises(ValueError, match="VAL already exposed"):
        p.create_lock(directory, "B", {"new": True}, assets)


def test_blocked_b_does_not_block_a_but_cannot_backdate_b(round_setup):
    directory, assets, _ = round_setup
    freeze_a(round_setup)
    exposed = p.authorize_score(directory, "A", "Historical adapter payload unavailable")
    assert not exposed["both_protocols_locked_before_first_GT"]
    with pytest.raises(ValueError, match="already exposed"):
        p.create_lock(directory, "B", {"late": True}, assets)
    with pytest.raises(ValueError, match="only independent A"):
        p.authorize_score(directory, "B", "missing")


def test_no_lock_b_before_a_predictions(round_setup):
    directory, assets, _ = round_setup
    with pytest.raises(FileNotFoundError):
        p.create_lock(directory, "B", {"fixture": True}, assets)


def test_hash_source_and_prediction_tampering_fail_before_scoring(round_setup, monkeypatch):
    directory, _, _ = round_setup
    output = freeze_a(round_setup)
    freeze_b(round_setup)
    output.write_bytes(b"tampered prediction")
    with pytest.raises(ValueError, match="prediction changed"):
        p.authorize_score(directory, "A")
    assert not (directory / "VAL_EXPOSURE.json").exists()
    monkeypatch.setattr(p, "source_snapshot", lambda: {"git_head": "fixture", "files": {"code.py": "changed"}})
    with pytest.raises(ValueError, match="source changed"):
        p.read_lock(directory, "A")


def test_append_once_rejects_masks_and_nonfinite_protocol(round_setup):
    directory, assets, image = round_setup
    with pytest.raises(ValueError, match="mask payload"):
        p.create_lock(directory, "A", {"fixture": True}, {"VAL_mask": assets["image"]})
    with pytest.raises(ValueError):
        p.create_lock(directory, "A", {"rho": float("nan")}, assets)
    freeze_a(round_setup)
    with pytest.raises(FileExistsError):
        p.create_lock(directory, "A", {"changed": True}, assets)
    image.write_bytes(b"different inputs")
    with pytest.raises(ValueError, match="asset hash mismatch"):
        p.read_lock(directory, "A")


def test_record_checksum_binds_config(round_setup):
    directory, _, _ = round_setup
    freeze_a(round_setup)
    path = directory / "protocol-A.lock.json"
    record = json.loads(path.read_text())
    record["protocol"]["lambdas"] = [1]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum mismatch"):
        p.read_lock(directory, "A")
