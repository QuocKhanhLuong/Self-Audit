import json

from scripts.smoke_fuse_aug import run


def test_synthetic_v2_a_aug_cpu_end_to_end(tmp_path):
    receipt = run(tmp_path / "a_aug", threads=1)
    assert receipt["status"] == "SOFTWARE_ONLY"
    assert receipt["stages"] == ["T", "J", "VAL3", "deploy_export"]
    assert receipt["augmentation"] == {"policy": "inplane_strong_v1"}
    assert receipt["augmentation_rng_checkpointed"] is True
    assert receipt["teacher_has_no_grads_after_J"] is True
    assert receipt["teacher_runtime_invariant"] is True
    assert receipt["export_clean"] is True
    assert set(receipt["exports"]) == {"teacher", "student", "fusion"}
    assert set(receipt["cold_load"]) == {"teacher", "student", "fusion"}
    assert all(value["bitwise_equal"] is True for value in receipt["cold_load"].values())
    saved = json.loads((tmp_path / "a_aug" / "SOFTWARE_ONLY_A_AUG.json").read_text())
    assert saved == receipt
