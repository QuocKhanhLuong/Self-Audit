"""Resolved W2S-Match configs (arms W, S0, S2, S2-arb), derived from the PG K8 base.

Arms differ from the base only in the reviewed fields below.  The student
arms share every other field; ``pg.target_mode`` and ``cbkd.*`` are the only
fields that may differ between S0, S2 and S2-arb.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .config import default_config, validate_config

ARMS = ("W", "S0", "S2", "S2arb")
W2S_PARAMS = {"tau": 0.95, "tau_v": 0.99, "alpha_max": 0.5, "alpha_start_fraction": 0.2, "alpha_end_fraction": 0.5, "min_visible_views": 2}
CBKD_SA = {
    "enabled": True, "variant": "cbkd_sa_gated", "lambda_cb": 0.5, "warmup_updates_labeled_only": 1200, "ramp_updates": 400,
    "atom": {"patch_from": "student_stage3_phase", "decoder_hidden": 128, "p_pm_gate_rules": [1, 2, 4], "tau_star": "median_d2_minus_d1_on_train_cells"},
}


def make_config(arm: str, *, tau_v: float = 0.99) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown W2S arm {arm!r}")
    cfg = default_config()
    cfg["training"]["selection"] = {"split": "VAL20", "path": "student_only", "metric": "fg_dice_patient_macro"}
    cfg["training"]["eval_splits"] = ["train_labeled", "VAL20", "VAL18", "TEST50"]
    cfg["teacher"]["select"] = "VAL20 FG of teacher region head"
    if arm == "W":
        cfg["teacher"]["semi"]["threshold"]["clip"] = [0.95, 0.95]
        return validate_config(cfg)
    w2s = dict(W2S_PARAMS, tau_v=float(tau_v))
    cfg["pg"]["w2s"] = w2s
    cfg["pg"]["enabled"] = True
    cfg["pg"]["target_mode"] = "self" if arm == "S0" else "w2s_arbiter"
    if arm == "S2":
        cfg["cbkd"].update(copy.deepcopy(CBKD_SA))
    else:
        cfg["cbkd"]["enabled"] = False
    return validate_config(cfg)


def write_all(directory: str | Path, *, tau_v: float = 0.99) -> list[Path]:
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    paths = []
    for arm in ARMS:
        path = root / f"w2s_{arm}.json"
        path.write_text(json.dumps(make_config(arm, tau_v=tau_v), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


__all__ = ["ARMS", "make_config", "write_all"]
