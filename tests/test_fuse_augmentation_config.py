import copy
import hashlib
import json
from pathlib import Path

import pytest

from self_audit.candidate_fuse.config import config_sha256, load_config, normalize_config


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/fuse-semi-20261113/recipe.A.json"
A_AUG = ROOT / "configs/candidate_fuse/a_aug.json"


def _canonical_sha(path):
    value = json.loads(path.read_text())
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def test_historical_recipe_hashes_and_a_aug_single_field_delta():
    old = load_config(REPORT)
    augmented = load_config(A_AUG)
    assert config_sha256(old) == _canonical_sha(REPORT)
    assert config_sha256(augmented) == _canonical_sha(A_AUG)
    assert {**augmented, "augmentation": None} != old
    without = copy.deepcopy(augmented)
    del without["augmentation"]
    assert without == old
    assert config_sha256(without) == config_sha256(old)


@pytest.mark.parametrize("path", [REPORT, A_AUG])
def test_augmentation_is_opt_in_and_normalization_is_stable(path):
    raw = json.loads(path.read_text())
    normalized = normalize_config(raw)
    assert normalize_config(normalized) == normalized
    if path == A_AUG:
        assert normalized["augmentation"] == {"policy": "inplane_strong_v1"}
    else:
        assert "augmentation" not in normalized


@pytest.mark.parametrize("mutation", [
    {"policy": "unknown"},
    {"policy": "inplane_strong_v1", "extra": True},
])
def test_unknown_augmentation_policy_and_keys_rejected(mutation):
    raw = json.loads(A_AUG.read_text())
    raw["augmentation"] = mutation
    with pytest.raises(ValueError, match="augmentation"):
        normalize_config(raw)


def test_augmentation_rejected_for_v1_and_v3():
    raw = json.loads(REPORT.read_text())
    raw["schema"] = "self_audit.candidate_fuse.v1"
    del raw["experiment"]
    raw["training"]["teacher_epochs"] = 50
    raw["training"]["teacher_batch_size"] = 1
    raw["augmentation"] = {"policy": "inplane_strong_v1"}
    with pytest.raises(ValueError, match="augmentation"):
        normalize_config(raw)

    raw = json.loads(A_AUG.read_text())
    raw["schema"] = "self_audit.candidate_fuse.v3"
    with pytest.raises(ValueError, match="augmentation"):
        normalize_config(raw)
