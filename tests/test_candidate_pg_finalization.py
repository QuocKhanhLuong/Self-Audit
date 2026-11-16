from __future__ import annotations

import copy
import json

import pytest
import torch

from self_audit.candidate_pg.codebook import StudentCodeBranch, TeacherCodebook
from self_audit.candidate_pg.config import default_config
from self_audit.candidate_pg.evaluation import evaluate_epoch
from self_audit.candidate_pg.finalization import finalize_campaign
from self_audit.candidate_pg.student_pg import PGStudent


def _frames() -> list[dict[str, object]]:
    shape = (2, 17, 19)
    geometry = {
        "shape": shape,
        "spacing_zyx": [1.0, 1.0, 1.0],
        "affine_zyx": torch.eye(4).tolist(),
        "context": "full_volume",
    }
    values = []
    for index in range(2):
        image = torch.full((1, 1, *shape), 0.5 + 0.01 * index)
        target = torch.zeros((1, *shape), dtype=torch.long)
        target[..., :8, :8] = 1
        target[..., :8, 8:16] = 2
        target[..., 8:16, :8] = 3
        values.append({
            "patient": f"p{index}", "frame": index, "image": image,
            "target": target, "geometry": geometry,
        })
    return values


def _logs(models: dict[str, PGStudent], frames: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for arm, model in models.items():
        rows = []
        for split in ("VAL18", "TEST50"):
            metrics = evaluate_epoch(
                model, data=[dict(frame, split=split) for frame in frames],
                epoch=1, split=split, path="student_only",
            )
            per = metrics["per_class"]
            rows.append({
                "stage": "arms", "arm": arm, "epoch": 1, "update": 80,
                "split": split, "path": "student_only",
                "fg_dice": metrics["fg_dice"],
                "rv_dice": per["RV"]["dice"], "myo_dice": per["MYO"]["dice"],
                "lv_dice": per["LV"]["dice"], "fg_hd95": metrics["fg_hd95"],
                "rv_hd95": per["RV"]["hd95_mm"], "myo_hd95": per["MYO"]["hd95_mm"],
                "lv_hd95": per["LV"]["hd95_mm"], "fg_assd": metrics["fg_assd"],
                "rv_assd": per["RV"]["assd_mm"], "myo_assd": per["MYO"]["assd_mm"],
                "lv_assd": per["LV"]["assd_mm"],
            })
        result[arm] = rows
    return result


def test_finalize_campaign_exports_and_evaluates_real_pgstudent(tmp_path) -> None:
    torch.set_num_threads(1)
    config = default_config(synthetic=True)
    frames = _frames()
    models = {arm: PGStudent(config) for arm in ("A1", "A2", "A3", "A4")}
    # Exercise the real train-only CBKD path used by A3/A4.  A small frozen
    # TeacherCodebook keeps this CPU fixture quick while retaining the exact
    # 128-channel stage3 -> 64-dimensional H_S_Q contract.
    codebook = TeacherCodebook({"teacher_channels": 256, "embedding_dim": 64,
                                "codebook_size": 8, "seed": 42}).freeze()
    branch_config = {"student_channels": 128, "embedding_dim": 64,
                     "codebook_size": 8, "rho_temperature": 1.0}
    for arm in ("A3", "A4"):
        models[arm].attach_code_branch(StudentCodeBranch(branch_config, codebook))
    result = finalize_campaign(
        config, tmp_path, models, {arm: 1 for arm in models}, _logs(models, frames),
        {"labeled": frames, "val": frames, "test": frames, "synthetic": True},
        teacher=None,
    )
    assert result["status"] == "SOFTWARE_ONLY"
    assert result["test_transaction"]["status"] == "COMMITTED"
    assert all((tmp_path / "exports" / f"{arm}.pt").is_file() for arm in models)
    for arm in models:
        artifact = torch.load(tmp_path / "exports" / f"{arm}.pt", weights_only=True)
        assert artifact["config"]["pg"]["enabled"] == (arm in {"A2", "A4"})
        assert artifact["config"]["cbkd"]["enabled"] == (arm in {"A3", "A4"})
    assert (tmp_path / "FINAL_METRICS.json").is_file()
    assert (tmp_path / "ANALYSIS.json").is_file()
    assert (tmp_path / "SHUFFLE.json").is_file()
    shuffle = json.loads((tmp_path / "SHUFFLE.json").read_text())
    for arm in ("A3", "A4"):
        diagnostic = shuffle["arms"][arm]
        assert diagnostic["status"] == "PASS"
        assert diagnostic["frames"] == len(frames)
        assert diagnostic["valid_cells_only"] is True
        assert diagnostic["training"] is False
        assert diagnostic["baseline_qs_dice"] is not None
        assert diagnostic["shuffled_qs_dice"] is not None
        assert diagnostic["drop_qs_dice"] is not None
        assert all(float(diagnostic[key]) == float(diagnostic[key])
                   for key in ("baseline_qs_dice", "shuffled_qs_dice", "drop_qs_dice"))


def test_finalize_campaign_rejects_selected_metric_mismatch(tmp_path) -> None:
    torch.set_num_threads(1)
    config = default_config(synthetic=True)
    frames = _frames()
    models = {arm: PGStudent(config) for arm in ("A1", "A2", "A3", "A4")}
    logs = _logs(models, frames)
    logs["A1"] = copy.deepcopy(logs["A1"])
    logs["A1"][0]["fg_dice"] = float(logs["A1"][0]["fg_dice"]) + 0.1
    with pytest.raises(ValueError, match="metrics mismatch"):
        finalize_campaign(
            config, tmp_path, models, {arm: 1 for arm in models}, logs,
            {"labeled": frames, "val": frames, "test": frames, "synthetic": True},
            teacher=None,
        )
