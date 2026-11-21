"""Candidate PG v2 configuration contract.

The campaign base is copied from the exact user JSON reference. Only the
explicitly authorized ancillary fields (manifest/runtime receipts, the Q3D
boundary/SDF compatibility head, S3 optimizer, and deterministic warmup
runtime choices) are additive; no framework defaults are inferred here.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "self_audit.candidate_pg.v2"
METHOD = "candidate_pg"
SEED = 42
ARM_KEYS = ("pg", "cbkd")

# The bytes of configs/candidate_fuse/semi_k8.json are distinct from the
# source K8 lock consumed by fuse_build_semi_manifest.py.
SPLIT_LOCK_SHA256 = "cdec2b85c549a1cf9a19eea8790b7d6dbe155fdbe7a4798b60315412cbead639"
K8_SOURCE_SHA256 = "60dd11747570f3ac183e8636050a3ac3d73651cb43b37deba9ff7422b442a433"
TRAIN_MANIFEST_SHA256 = "ef7562a251e5c30622ab00dfa46a5b62e0c4552cc0aac06b24caa3e63ff23363"
VAL_MANIFEST_SHA256 = "e3421779441afcab8f8feadbd293bedad6c00877730eb3c7151fa6a15667e88b"
TEST_MANIFEST_SHA256 = "84f24ca7a50e4516548cc16037b6159eaed2eb1ea07d94287e8305894a951297"
_HEX = set("0123456789abcdef")


def _base_manifest_refs() -> dict[str, dict[str, str]]:
    return {
        "train": {"path": "/root/mount_data/prepared/ACDC/fuse-k8-20261113/train.json", "sha256": TRAIN_MANIFEST_SHA256},
        "val": {"path": "/root/mount_data/prepared/ACDC/fuse-final-20261113/development.json", "sha256": VAL_MANIFEST_SHA256},
        "test": {"path": "/root/mount_data/prepared/ACDC/fuse-final-20261113/test.json", "sha256": TEST_MANIFEST_SHA256},
    }


def _reference() -> dict[str, Any]:
    """Exact original user JSON, before explicitly authorized additions."""
    return {
        "schema": SCHEMA,
        "seed": 42,
        "data": {
            "dataset": "acdc", "split_lock": "configs/candidate_fuse/semi_k8.json", "k_labeled": 8,
            "unlabeled": "all_train_cine_frames_minus_labeled_masks", "forbid_splits_in_training": ["VAL18", "TEST50"],
            "native_geometry": True, "per_step": {"labeled": 1, "unlabeled": 2}, "u_partner_policy": "different_patient", "u_sampling": "patient_balanced",
        },
        "augmentation": {
            "policy": "inplane_strong_v1", "weak": "spatial_only", "strong": "intensity_only_on_weak_canvas", "n_strong_views": 2, "clamp": [0.0, 1.0],
            "cutmix": {"area_fraction": [0.25, 0.5], "align_px": 16, "full_depth": True, "pad_weight": 0.0},
        },
        "student": {
            "arch": "unetrpp_lite_fuse", "channels": [16, 32, 64, 128, 256], "num_classes": 4, "init": "cinema_distill",
            "fusion": {"levels": 5, "gate_bias": -2.0, "inside_encoder": True}, "teacher_dropout_on_labeled": {"start": 0.0, "end": 0.5}, "fp_skip_dropout": 0.5,
        },
        "cinema_init": {
            "hf_repo": "mathpluscode/CineMA", "files": ["pretrained/cinema.safetensors", "pretrained/config.yaml"],
            "taps": {"cinema_conv_1_8": "student.stage3", "cinema_tokens_1_16": "student.stage4"},
            "projector": "conv1x1x1_discard_after_init", "loss": "cosine", "data": "TRAIN images only (L images + U)",
            "optimizer": {"name": "adamw", "lr": 0.0003, "weight_decay": 0.0001}, "updates": 10000,
        },
        "teacher": {
            "arch": "mednext_l", "kernel": 3, "channels": [32, 64, 128, 256, 512],
            "block_counts": [3, 4, 8, 8, 8, 8, 8, 4, 3], "exp_ratios": [3, 4, 8, 8, 8, 8, 8, 4, 3],
            "down_strides": [[1, 2, 2], [1, 2, 2], [2, 2, 2], [1, 2, 2]], "feature_phases": [[1, 1, 1], [1, 2, 2], [1, 4, 4], [2, 8, 8], [2, 16, 16]],
            "heads": {"region": 4, "sdf": {"channels": 3, "radius_mm": 5.0}}, "init": "random",
            "semi": {
                "method": "unimatch_v1", "target": "self_weak_stopgrad", "stream_weights": {"s1": 0.25, "s2": 0.25, "fp": 0.5},
                "fp": "dropout3d_0.5_on_encoder_skips", "threshold": {"mode": "class_adaptive", "ema": 0.999, "clip": [0.5, 0.95]},
                "sdf_consistency": {"weight": 0.5, "on": "confident_voxels", "loss": "l1_normalized"},
                "lambda_u": {"max": 1.0, "warmup_updates_labeled_only": 800, "ramp_fraction": 0.1},
            },
            "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001}, "updates": 8000, "select": "VAL18 FG of teacher region head",
        },
        "codebook": {
            "teacher_tap": "mednext_level3_phase_2_8_8_aligned_to_student_stage3_grid_1_8_8", "P_T": "conv1x1x1_256_to_64",
            "codebook": {"size": 128, "dim": 64, "dead_code_restart_every": 500, "usage_entropy_weight": 0.01},
            "H_T_Q": "conv1x1x1_64_64_relu_64_4", "train_labels": "L_K pooled GT + detached teacher pseudo-labels on U (confidence-weighted)",
            "commit_beta": 0.25, "rho_temperature": 1.0, "updates": 4000,
        },
        "pg": {
            "enabled": True, "target_source": "fusion_weak", "threshold": {"mode": "class_adaptive", "ema": 0.999, "clip": [0.5, 0.95]},
            "agreement": {"paths": ["teacher_only_region", "student_only_clean_weak"], "form": "sqrt(pT[y]*pS[y])"},
            "boundary": {"near_mm": 2.5, "radius_mm": 5.0, "mode": "soft_weight"}, "stream_weights": {"s1": 0.25, "s2": 0.25, "fp": 0.5},
            "lambda_u": {"max": 1.0, "warmup_updates_labeled_only": 800, "ramp_fraction": 0.1}, "loss": "weighted_ce_plus_weighted_dice",
        },
        "cbkd": {
            "enabled": True, "variant": "aux_code_branch", "codebook": "full_C_frozen", "student_tap": "stage3_post_epa_128ch",
            "P_S": "conv1x1x1_128_to_64", "H_S_Q": "conv1x1x1_64_64_relu_64_4", "rho_temperature": 1.0,
            "weights": {"assign": 1.0, "qtask": 1.0, "commit": 0.25, "qpred": 0.0}, "lambda_cb": 0.5,
            "inputs": "clean weak U (no CutMix) for assign/commit; labeled student-only forward for assign/commit/qtask", "warmup_updates_labeled_only": 800,
        },
        "training": {
            "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001, "schedule": "reuse candidate_fuse _schedule_lr"},
            "updates": 8000, "updates_per_epoch": 80, "eval_every_epochs": 1, "eval_splits": ["train_labeled", "VAL18", "TEST50"],
            "selection": {"split": "VAL18", "path": "student_only", "metric": "fg_dice_patient_macro"},
        },
        "export": {"strip_prefixes": ["fusions.", "mimic_heads.", "code_branch.", "P_S.", "H_S_Q."], "assert_no_teacher_keys": True},
    }


def default_config(*, synthetic: bool = False) -> dict[str, Any]:
    # Keep the production recipe in one materialized JSON source.  The
    # immutable ``_reference`` below remains the validation contract for the
    # original request values, so changing the production file cannot silently
    # broaden or retune those values.
    config = _load_json(_base_config_path())
    runtime = config.get("runtime")
    _require(isinstance(runtime, Mapping), "PG base config must declare runtime explicitly")
    runtime["synthetic"] = bool(synthetic)
    return config


def _base_config_path() -> Path:
    """Return the repository-owned materialized PG production config."""

    return Path(__file__).resolve().parents[2] / "configs" / "candidate_pg" / "pg_k8_base.json"


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# Reviewed deviations from the PG v2 reference used by the W2S-Match campaign.
# ``None`` means any value (type-checked by the consumer).
_W2S_ALTERNATIVES: dict[str, Any] = {
    "teacher.semi.threshold.clip": ([0.5, 0.95], [0.95, 0.95]),
    "teacher.select": ("VAL18 FG of teacher region head", "VAL20 FG of teacher region head", "VAL10 FG of teacher region head"),
    # SSL4MIS split campaign (70/10/20, 7 labeled patients, ED/ES-only U)
    "data.k_labeled": (8, 7),
    "data.unlabeled": ("all_train_cine_frames_minus_labeled_masks", "ed_es_frames_of_remaining_train_patients"),
    "data.forbid_splits_in_training": (["VAL18", "TEST50"], ["VAL10", "TEST20"]),
    "cbkd.variant": ("aux_code_branch", "cbkd_sa_gated", "cbkd_sa", "cbkd_sa_out"),
    "cbkd.warmup_updates_labeled_only": None,
    "training.selection": None,
    "training.eval_splits": None,
    "training.updates": None,  # student epoch budget (training.epochs x updates_per_epoch); checked below
}
W2S_TARGET_MODES = ("fusion", "self", "w2s_arbiter", "teacher")


def _assert_reference_values(actual: Mapping[str, Any], reference: Mapping[str, Any], path: str = "") -> None:
    """Require every original reference key/value while permitting extras."""

    for key, expected in reference.items():
        location = f"{path}.{key}" if path else str(key)
        _require(key in actual, f"PG exact reference key missing: {location}")
        observed = actual[key]
        if location in _W2S_ALTERNATIVES:
            # W2S-Match (2026-11-19): a closed set of reviewed alternatives; everything else stays pinned.
            allowed = _W2S_ALTERNATIVES[location]
            _require(allowed is None or observed in allowed, f"PG reference value {location} outside the reviewed W2S alternatives")
            continue
        if location in {"pg.enabled", "cbkd.enabled"}:
            # Resolved A1/A2/A3/A4 configs are intentionally the same locked
            # recipe with only these two Boolean switches changed.
            _require(type(observed) is bool, f"PG arm switch must be Boolean: {location}")
            continue
        if isinstance(expected, Mapping):
            _require(isinstance(observed, Mapping), f"PG exact reference section changed: {location}")
            _assert_reference_values(observed, expected, location)
        else:
            _require(observed == expected, f"PG exact reference value changed: {location}")


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(config, Mapping), "PG config must be a mapping")
    value = copy.deepcopy(dict(config))
    reference = _reference()
    _require(set(value) == set(default_config(synthetic=bool(value.get("runtime", {}).get("synthetic"))).keys()), "PG base config has unexpected or missing top-level keys")
    _require(value.get("schema") == SCHEMA and type(value.get("seed")) is int and value["seed"] == SEED, "unsupported PG schema or seed")
    _assert_reference_values(value, reference)
    data = value["data"]
    allowed_data = set(default_config()["data"])
    optional_data = {"materialization", "ssl4mis"}  # ssl4mis: {lock, receipt} refs of the SSL4MIS-split campaign
    _require(allowed_data <= set(data) <= allowed_data | optional_data, "PG data fields must be explicit and complete")
    _require("ssl4mis" not in data or (isinstance(data["ssl4mis"], Mapping) and set(data["ssl4mis"]) == {"lock", "receipt"}
             and all(isinstance(v, Mapping) and set(v) == {"path", "sha256"} for v in data["ssl4mis"].values())), "data.ssl4mis must be {lock, receipt} path/sha256 refs")
    _require(data["split_lock"] == "configs/candidate_fuse/semi_k8.json" and data["split_lock_sha256"] == SPLIT_LOCK_SHA256 and data["source_k8_sha256"] == K8_SOURCE_SHA256, "PG split/source lock SHA pins are invalid")
    _require(set(data["manifests"]) == {"train", "val", "test"}, "PG manifests must declare train, val and test")
    for role, ref in data["manifests"].items():
        _require(isinstance(ref, Mapping) and set(ref) == {"path", "sha256"}, f"PG {role} manifest reference is incomplete")
        _require((ref["path"] is None) == (ref["sha256"] is None), f"PG {role} manifest path/SHA must be paired")
        _require(ref["path"] is None or isinstance(ref["path"], str), f"PG {role} manifest path must be a string or null")
        _require(ref["sha256"] is None or _is_sha(ref["sha256"]), f"PG {role} manifest SHA256 is invalid")
    materialization = data.get("materialization")
    if materialization is not None:
        fields = {"receipt", "sha256"}
        runtime_for_materialization = value.get("runtime")
        _require(isinstance(runtime_for_materialization, Mapping) and runtime_for_materialization.get("synthetic") is False, "PG materialization lineage is real-data only")
        _require(isinstance(materialization, Mapping) and set(materialization) == fields, "PG materialization lineage fields are invalid")
        _require(isinstance(materialization["receipt"], str) and bool(materialization["receipt"]), "PG preparation receipt path is required")
        _require(_is_sha(materialization["sha256"]), "PG preparation receipt SHA256 is invalid")
    runtime = value["runtime"]
    _require(set(runtime) == set(default_config()["runtime"]), "PG runtime fields must be explicit")
    _require(runtime["cpu_amp_dtype"] == "float32" and runtime["gpu_amp_dtype"] == "bfloat16" and runtime["deterministic"] is True and runtime["tf32"] is False, "PG runtime precision/determinism is locked")
    _require(value["teacher"]["heads"]["sdf"] == {"channels": 3, "radius_mm": 5.0}, "PG public SDF head must remain three foreground channels")
    _require(value["teacher"]["heads"].get("compatibility") == default_config()["teacher"]["heads"]["compatibility"], "PG Q3D compatibility heads are locked")
    teacher = value["teacher"]
    _require(teacher.get("source_repo") == "external/MedNeXt", "PG teacher source_repo must remain external/MedNeXt")
    _require(teacher.get("source_commit") == "0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba", "PG teacher source_commit is not the verified MedNeXt revision")
    _require(teacher.get("weights") is None and teacher.get("weights_sha256") is None, "PG teacher must use the declared random initialization without ACDC-pretrained weights")
    for section in ("cinema_init", "teacher", "codebook", "training"):
        optimizer = value[section].get("optimizer")
        _require(isinstance(optimizer, Mapping) and optimizer.get("name") == "adamw", f"PG {section} optimizer must be explicit lowercase adamw")
    _require(value["training"]["optimizer"].get("schedule") == "reuse candidate_fuse _schedule_lr" and value["training"].get("warmup_fraction") == 0.05, "PG student schedule/warmup is locked")
    _require(isinstance(value["pg"].get("enabled"), bool) and isinstance(value["cbkd"].get("enabled"), bool), "PG arm switches must be boolean")
    training = value["training"]
    if "epochs" in training:
        _require(type(training["epochs"]) is int and training["epochs"] > 0 and training["updates"] == training["epochs"] * training["updates_per_epoch"],
                 "training.updates must equal training.epochs x training.updates_per_epoch")
    mode = value["pg"].get("target_mode", "fusion")
    _require(mode in W2S_TARGET_MODES, "pg.target_mode must be fusion, self, teacher or w2s_arbiter")
    if value["cbkd"].get("variant") in ("cbkd_sa_gated", "cbkd_sa_out") and value["cbkd"]["enabled"]:
        _require(mode == "w2s_arbiter", "cbkd_sa_gated/cbkd_sa_out require pg.target_mode='w2s_arbiter'")
    source = value["pg"].get("teacher_source", "mednext")
    _require(source in ("mednext", "m3hl"), "pg.teacher_source must be mednext or m3hl")
    if source == "m3hl":
        m3 = value["pg"].get("m3hl")
        _require(isinstance(m3, Mapping) and isinstance(m3.get("checkpoint"), Mapping) and _is_sha(m3["checkpoint"].get("sha256"))
                 and isinstance(m3["checkpoint"].get("path"), str) and str(m3.get("source_commit", "")).startswith("19151b3"),
                 "pg.m3hl needs checkpoint {path, sha256} and source_commit 19151b3")
        _require(mode in ("teacher", "w2s_arbiter"), "the M3HL teacher needs pg.target_mode teacher or w2s_arbiter")
    _require(value["cbkd"].get("variant") != "cbkd_sa_out" or not value["cbkd"]["enabled"] or source == "m3hl", "cbkd_sa_out needs pg.teacher_source='m3hl'")
    _require(not (value["cbkd"]["enabled"] and value["cbkd"].get("variant") == "cbkd_sa_gated" and source == "m3hl"), "cbkd_sa_gated needs the MedNeXt teacher; use cbkd_sa_out with M3HL")
    if mode != "fusion":
        w2s = value["pg"].get("w2s")
        _require(isinstance(w2s, Mapping), "pg.w2s must be declared for W2S target modes")
        _require(w2s.get("tau") == 0.95, "W2S student threshold is the fixed 0.95")
        _require(0.0 < float(w2s.get("tau_v", 0.99)) <= 1.0 and 0.0 <= float(w2s.get("alpha_max", 0.5)) <= 1.0, "invalid W2S tau_v/alpha_max")
    selection = value["training"]["selection"]
    _require(isinstance(selection, Mapping) and selection.get("split") in ("VAL18", "VAL20", "VAL10") and selection.get("path") == "student_only"
             and selection.get("metric") == "fg_dice_patient_macro", "PG selection must be student_only fg_dice_patient_macro on VAL18/VAL20")
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid PG JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("PG JSON must contain an object")
    return dict(value)


def load_config(path: str | Path) -> dict[str, Any]:
    value = _load_json(path)
    if set(value) == set(ARM_KEYS):
        for key in ARM_KEYS:
            if not isinstance(value[key], Mapping) or set(value[key]) != {"enabled"} or not isinstance(value[key]["enabled"], bool):
                raise ValueError("PG arm files must contain only pg.enabled and cbkd.enabled")
        return value
    return validate_config(value)


def apply_arm(base: Mapping[str, Any], arm: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    config = validate_config(base)
    switches = load_config(arm) if isinstance(arm, (str, Path)) else dict(arm)
    _require(set(switches) == set(ARM_KEYS), "PG arm must contain only pg and cbkd switch dictionaries")
    for key in ARM_KEYS:
        _require(isinstance(switches[key], Mapping) and set(switches[key]) == {"enabled"}, f"PG arm {key} must contain only enabled")
        config[key] = copy.deepcopy(config[key]); config[key]["enabled"] = bool(switches[key]["enabled"])
    return config


def config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(validate_config(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ARM_KEYS", "K8_SOURCE_SHA256", "METHOD", "SCHEMA", "SEED", "SPLIT_LOCK_SHA256", "TEST_MANIFEST_SHA256",
    "TRAIN_MANIFEST_SHA256", "VAL_MANIFEST_SHA256", "apply_arm", "config_sha256", "default_config", "load_config", "validate_config",
]
