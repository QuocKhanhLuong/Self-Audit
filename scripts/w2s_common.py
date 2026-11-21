"""Shared helpers for the W2S-Match run scripts (teacher loading, subset metrics, JSON cleaning)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



def load_config(path: Path) -> dict[str, Any]:
    from self_audit.candidate_pg.config import validate_config

    return validate_config(json.loads(Path(path).read_text()))


def load_teacher(cfg: dict[str, Any], run: Path, device: Any):
    """Frozen teacher from ``<run>/S2_RECEIPT.json`` (same loading as the engine's stage-only path)."""
    import torch
    from collections.abc import Mapping

    from self_audit.candidate_pg import engine

    teacher, meta = engine._build_teacher(cfg, synthetic=False)
    if meta.get("kind") == "m3hl":  # hash-pinned in the config; no S2 receipt
        teacher.to(device)
        return teacher, Path(cfg["pg"]["m3hl"]["checkpoint"]["path"])
    receipt = engine._read_receipt(run / "S2_RECEIPT.json")
    artifact = engine._receipt_artifact(receipt, names=("checkpoint", "checkpoint_path"), label="S2")
    payload = torch.load(artifact, map_location=device, weights_only=False)
    state = payload.get("model", payload.get("teacher", payload)) if isinstance(payload, Mapping) else payload
    teacher.load_state_dict(state, strict=True)
    teacher.to(device).eval().requires_grad_(False)
    return teacher, Path(artifact)


def subset_mean(per_patient: dict[str, Any], ids: set[str]) -> dict[str, Any]:
    """Patient-macro means over a subset of per-patient results (FG and per-class Dice / HD95)."""
    rows = [v for k, v in per_patient.items() if k in ids]
    if not rows:
        return {"patients": 0}

    def mean(values):
        values = [float(v) for v in values if v is not None]
        return sum(values) / len(values) if values else None

    out = {"patients": len(rows), "fg_dice": mean(r.get("fg_dice") for r in rows), "fg_hd95": mean(r.get("fg_hd95") for r in rows)}
    for name in ("RV", "MYO", "LV"):
        classes = [(r.get("per_class") or {}).get(name) or {} for r in rows]
        out[f"{name.lower()}_dice"] = mean(c.get("dice") for c in classes)
        out[f"{name.lower()}_hd95"] = mean(c.get("hd95_mm") for c in classes)
    return out


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write_once(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(clean(payload), indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
