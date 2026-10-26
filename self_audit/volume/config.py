"""Validated opt-in configuration for the native-volume trainer.

The native-volume path is intentionally a separate schema.  Nothing in this
module mutates or migrates the historical ``self_audit.semi_v2`` recipes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA = "self_audit.volume.v1"


def default_volume_config() -> dict[str, Any]:
    """Return a conservative CPU-smoke configuration.

    Eight outer iterations is a software smoke budget, not a quality run.  A
    future real pilot must opt into its larger budget explicitly at the CLI.
    """

    return {
        "schema": SCHEMA,
        "model": {
            "backbone": "tiny_test",
            "source_repo": "",
            "source_commit": "",
            "weights": "",
            "weights_sha256": "",
        },
        "architecture": {
            "name": "volume25d",
            "depth_mode": "off",
            "widths": [8, 16, 24, 32],
            "z_kernels": [[1, 3, 3], [1, 3, 3]],
            # Volume25D applies the native 2-D stride per slice; non-neutral
            # cross-Z strides belong to the explicit hybrid/CNN3D controls.
            "z_strides": [[1, 1, 1], [1, 1, 1]],
            "residual_init": "zero",
            "skip_fusion": "legacy",
            "skip_stopgrad": False,
        },
        "window": {"mode": "none"},
        "geometry": {
            "slab_depth": 3,
            "crop_size": 32,
            "spacing_xyz_mm": None,
            "fov_xyz_mm": None,
            "flip": False,
            "resampling": "native_z",
        },
        "loss": {"mode": "legacy_pooled"},
        "optimization": {
            "recipe": "grouped_adamw",
            "body_lr": 3e-4,
            "head_lr": 3e-3,
            "encoder_lr": 3e-4,
            "decoder_lr": 3e-4,
            "weight_decay": 1e-4,
            "clip_norm": 5.0,
            "scheduler": "constant",
            "lr_warmup_steps": 0,
            "min_lr_ratio": 0.1,
        },
        "training": {
            "steps": 8,
            "pseudo_delay": 0,
            "seed": 42,
            "microbatch": 1,
            "accumulation": 1,
            "ema_decay": 0.99,
            "pseudo_threshold": 0.7,
            "pseudo_weight": 1.0,
            "pseudo_ramp_steps": 0,
            "pseudo_view": "identity",
            "pseudo_student_mode": "train",
            "method": "ema_full",
            "precision": "fp32",
            "threads": 1,
            "save_every": 1,
        },
        "regions": {"mode": "none", "weight": 0.0, "refresh_every": 0,
                     "spatial_weight": 1.0, "target_num_regions": 32, "min_region_size": 8,
                     "iterations": 4, "boundary_threshold": 0.2, "boundary_downweight": 0.1,
                     "boundary_rejection": False, "loss_type": "mse", "detached": True},
        "temporal": {"mode": "none", "weight": 0.0, "max_disp_xyz": [4.0, 16.0, 16.0],
                      "consistency_threshold": 1.5, "loss_type": "mse", "slice_wise": True,
                      "window": "none", "occlusion": True},
        "prediction_model": "student",
    }


def _expect_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _strict_section(value: Any, name: str, defaults: Mapping[str, Any]) -> dict[str, Any]:
    obj = dict(_expect_mapping(value, name))
    unknown = set(obj) - set(defaults)
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    out = dict(defaults)
    out.update(obj)
    return out


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (value < 0 if allow_zero else value < 1):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'} integer")
    return value


def _finite_number(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    v = float(value)
    if minimum is not None and v < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and v > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return v


def validate_volume_config(config: Mapping[str, Any], *, allow_full_run: bool = False,
                           stop_steps: int | None = None) -> dict[str, Any]:
    """Validate and normalize an opt-in ``self_audit.volume.v1`` recipe."""

    defaults = default_volume_config()
    if not isinstance(config, Mapping):
        raise ValueError("volume config must be an object")
    metadata_fields = {"id", "parent", "changed_factors", "data", "correction", "historical_reference",
                       "historical_a0_source", "historical_a0_sha256"}
    unknown = set(config) - (set(defaults) | metadata_fields)
    if unknown:
        raise ValueError(f"unknown volume config fields: {sorted(unknown)}")
    c = copy.deepcopy(defaults)
    c.update(copy.deepcopy(dict(config)))
    if c.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")

    raw_model = dict(_expect_mapping(c["model"], "model"))
    # The v1 schema deliberately separates model provenance from architecture
    # controls.  Do not migrate legacy aliases here: doing so can silently
    # change an experiment while leaving its source JSON apparently intact.
    arch_aliases = {"architecture", "name", "depth_mode", "widths", "z_kernels", "z_strides",
                    "residual_init", "skip_fusion", "skip_stopgrad"}
    overlap = arch_aliases & set(raw_model)
    if overlap:
        raise ValueError(f"architecture fields must be declared under architecture, not model: {sorted(overlap)}")
    c["model"] = _strict_section(c["model"], "model", defaults["model"])
    model = c["model"]
    if model["backbone"] not in {"tiny_test", "dinov3_convnext_tiny"}:
        raise ValueError("model.backbone must be tiny_test or dinov3_convnext_tiny")
    if model["backbone"] == "tiny_test" and not model["source_repo"] and not model["weights"]:
        # Empty provenance is valid only for the declared synthetic smoke.
        pass
    for key in ("source_repo", "source_commit", "weights", "weights_sha256"):
        if not isinstance(model[key], str):
            raise ValueError(f"model.{key} must be a string")

    c["architecture"] = _strict_section(c["architecture"], "architecture", defaults["architecture"])
    arch = c["architecture"]
    if arch["name"] not in {"volume25d", "hybrid", "cnn3d"}:
        raise ValueError("architecture.name must be volume25d, hybrid or cnn3d")
    if arch["depth_mode"] == "anisotropic":
        arch["depth_mode"] = "off"
    if arch["depth_mode"] not in {"off", "mix", "planar_control"}:
        raise ValueError("architecture.depth_mode is invalid")
    if arch["skip_fusion"] not in {"legacy", "delayed", "matched_additive"}:
        raise ValueError("architecture.skip_fusion is invalid")
    if arch["residual_init"] in {"legacy", "zero_old_function", "zero_init"}:
        arch["residual_init"] = "zero"
    if arch["residual_init"] not in {"zero", "kaiming"}:
        raise ValueError("architecture.residual_init is invalid")
    if not isinstance(arch["skip_stopgrad"], bool):
        raise ValueError("architecture.skip_stopgrad must be bool")
    if not isinstance(arch["widths"], list) or not arch["widths"] or any(_positive_int(x, "architecture.widths") <= 0 for x in arch["widths"]):
        raise ValueError("architecture.widths must be a non-empty list of positive integers")
    for key in ("z_kernels", "z_strides"):
        if not isinstance(arch[key], list):
            raise ValueError(f"architecture.{key} must be a list")
        for item in arch[key]:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError(f"architecture.{key} entries must be XYZ triples")
            if any(_positive_int(v, f"architecture.{key}") <= 0 for v in item):
                raise ValueError(f"architecture.{key} entries must be positive")
    if arch["name"] == "hybrid" and any(tuple(item[1:]) != (1, 1) for item in arch["z_kernels"]):
        raise ValueError("hybrid z_kernels must be anisotropic (kz,1,1); XY mixing belongs to the 2-D path")

    c["window"] = _strict_section(c["window"], "window", defaults["window"])
    if c["window"].get("mode") not in {"none", "fixed", "adaptive"}:
        raise ValueError("window.mode is invalid")

    c["geometry"] = _strict_section(c["geometry"], "geometry", defaults["geometry"])
    g = c["geometry"]
    _positive_int(g["slab_depth"], "geometry.slab_depth")
    if isinstance(g["crop_size"], (list, tuple)):
        if len(g["crop_size"]) != 2:
            raise ValueError("geometry.crop_size must be an integer or length-2 list")
        g["crop_size"] = [_positive_int(v, "geometry.crop_size") for v in g["crop_size"]]
    else:
        _positive_int(g["crop_size"], "geometry.crop_size")
    if g["resampling"] not in {"native_z", "native", "configured", "configured_physical_fov"}:
        raise ValueError("geometry.resampling is invalid")
    if g["spacing_xyz_mm"] is not None and (not isinstance(g["spacing_xyz_mm"], list) or len(g["spacing_xyz_mm"]) != 3):
        raise ValueError("geometry.spacing_xyz_mm must be length-3 or null")
    if g["fov_xyz_mm"] is not None and (not isinstance(g["fov_xyz_mm"], list) or len(g["fov_xyz_mm"]) != 3):
        raise ValueError("geometry.fov_xyz_mm must be length-3 or null")
    for key in ("spacing_xyz_mm", "fov_xyz_mm"):
        if g[key] is not None:
            g[key] = [_finite_number(v, f"geometry.{key}", minimum=0.0) for v in g[key]]
            if any(v <= 0 for v in g[key]):
                raise ValueError(f"geometry.{key} values must be positive")
    if g["resampling"] in {"native_z", "native"} and g["spacing_xyz_mm"] is not None:
        raise ValueError("native_z/native geometry cannot declare spacing_xyz_mm; choose configured resampling explicitly")
    if g["resampling"] in {"configured", "configured_physical_fov"} and g["spacing_xyz_mm"] is None:
        raise ValueError("configured geometry requires finite positive spacing_xyz_mm")
    if g["resampling"] == "configured_physical_fov" and g["fov_xyz_mm"] is None:
        raise ValueError("configured_physical_fov geometry requires fov_xyz_mm")
    if not isinstance(g["flip"], bool):
        raise ValueError("geometry.flip must be bool")

    c["loss"] = _strict_section(c["loss"], "loss", defaults["loss"])
    if c["loss"]["mode"] not in {"legacy_pooled", "volume_mean"}:
        raise ValueError("loss.mode is invalid")

    c["optimization"] = _strict_section(c["optimization"], "optimization", defaults["optimization"])
    o = c["optimization"]
    if o["recipe"] == "scheduler_only":
        raise ValueError("optimization.recipe alias scheduler_only is unsupported; declare grouped_adamw explicitly")
    if o["recipe"] not in {"legacy", "grouped_adamw"}:
        raise ValueError("optimization.recipe is invalid")
    for key in ("body_lr", "head_lr", "encoder_lr", "decoder_lr"):
        if o[key] is None and key in {"encoder_lr", "decoder_lr"}:
            continue
        _finite_number(o[key], f"optimization.{key}", minimum=0.0)
        if float(o[key]) <= 0:
            raise ValueError(f"optimization.{key} must be positive")
    _finite_number(o["weight_decay"], "optimization.weight_decay", minimum=0.0)
    _finite_number(o["clip_norm"], "optimization.clip_norm", minimum=0.0)
    if float(o["clip_norm"]) <= 0:
        raise ValueError("optimization.clip_norm must be positive")
    if o["scheduler"] not in {"constant", "cosine"}:
        raise ValueError("optimization.scheduler is invalid")
    _positive_int(o["lr_warmup_steps"], "optimization.lr_warmup_steps", allow_zero=True)
    _finite_number(o["min_lr_ratio"], "optimization.min_lr_ratio", minimum=0.0, maximum=1.0)

    c["training"] = _strict_section(c["training"], "training", defaults["training"])
    t = c["training"]
    steps = _positive_int(t["steps"], "training.steps")
    if stop_steps is not None:
        if isinstance(stop_steps, bool) or not isinstance(stop_steps, int) or stop_steps < 1:
            raise ValueError("stop_steps must be a positive integer")
        effective_steps = stop_steps
    else:
        effective_steps = steps
    if effective_steps > 8 and not allow_full_run and not bool(c.get("historical_reference", False)):
        raise ValueError("training.steps > 8 requires explicit allow_full_run")
    _positive_int(t["pseudo_delay"], "training.pseudo_delay", allow_zero=True)
    _positive_int(t["seed"], "training.seed", allow_zero=True)
    _positive_int(t["microbatch"], "training.microbatch")
    _positive_int(t["accumulation"], "training.accumulation")
    if t["accumulation"] > 1 and c["loss"]["mode"] == "legacy_pooled":
        raise ValueError("legacy_pooled accumulation>1 is unsupported; use volume_mean or accumulation=1")
    _finite_number(t["ema_decay"], "training.ema_decay", minimum=0.0, maximum=1.0)
    if not 0 <= float(t["ema_decay"]) < 1:
        raise ValueError("training.ema_decay must be in [0,1)")
    _finite_number(t["pseudo_threshold"], "training.pseudo_threshold", minimum=0.0, maximum=1.0)
    _finite_number(t["pseudo_weight"], "training.pseudo_weight", minimum=0.0)
    _positive_int(t["pseudo_ramp_steps"], "training.pseudo_ramp_steps", allow_zero=True)
    if t["pseudo_view"] not in {"identity", "strong_intensity"}:
        raise ValueError("training.pseudo_view must be identity or strong_intensity")
    if t["pseudo_student_mode"] not in {"train", "eval"}:
        raise ValueError("training.pseudo_student_mode must be train or eval")
    if t["method"] == "supervised_only":
        t["method"] = "supervised"
    if t["method"] not in {"supervised", "supervised_extra", "ema_full", "ema_full_c01"}:
        raise ValueError("training.method is invalid")
    if t["precision"] in {"amp_bf16", "amp_fp16"}:
        raise ValueError("training.precision aliases amp_bf16/amp_fp16 are unsupported; declare amp explicitly")
    if t["precision"] not in {"fp32", "amp"}:
        raise ValueError("training.precision is invalid")
    _positive_int(t["threads"], "training.threads")
    _positive_int(t["save_every"], "training.save_every")
    if t["pseudo_delay"] > steps and t["method"] not in {"supervised", "supervised_extra"}:
        raise ValueError("training.pseudo_delay exceeds steps")

    raw_regions = dict(_expect_mapping(c["regions"], "regions"))
    detached = raw_regions.pop("detached", None)
    c["regions"] = _strict_section(raw_regions, "regions", defaults["regions"])
    if detached is not None:
        if not isinstance(detached, bool):
            raise ValueError("regions.detached must be bool")
        c["regions"]["detached"] = detached
    if c["regions"]["mode"] in {"image_only", "image"}:
        c["regions"]["mode"] = "image"
    elif c["regions"]["mode"] in {"ema_derived", "ema"}:
        c["regions"]["mode"] = "ema"
    if c["regions"]["mode"] not in {"none", "image", "ema"}:
        raise ValueError("regions.mode is invalid")
    if not bool(c["regions"].get("detached", True)):
        raise ValueError("regions.detached=false is unsupported; region targets must be detached")
    _finite_number(c["regions"]["weight"], "regions.weight", minimum=0.0)
    _positive_int(c["regions"]["refresh_every"], "regions.refresh_every", allow_zero=True)
    for key in ("target_num_regions", "min_region_size", "iterations"):
        _positive_int(c["regions"][key], f"regions.{key}")
    for key in ("spatial_weight", "boundary_threshold", "boundary_downweight"):
        _finite_number(c["regions"][key], f"regions.{key}", minimum=0.0)
    if c["regions"]["loss_type"] not in {"mse", "kl"} or not isinstance(c["regions"]["boundary_rejection"], bool):
        raise ValueError("regions loss/boundary settings are invalid")
    raw_temporal = dict(_expect_mapping(c["temporal"], "temporal"))
    c["temporal"] = _strict_section(raw_temporal, "temporal", defaults["temporal"])
    if c["temporal"]["mode"] in {"identity_diagnostic", "identity"}:
        c["temporal"]["mode"] = "identity"
    elif c["temporal"]["mode"] in {"motion_phase_correlation", "motion"}:
        c["temporal"]["mode"] = "motion"
    if c["temporal"]["mode"] == "disabled":
        c["temporal"]["mode"] = "none"
    if c["temporal"]["mode"] not in {"none", "identity", "motion"}:
        raise ValueError("temporal.mode is invalid")
    if c["temporal"]["window"] not in {"none", "hanning", "tukey"}:
        raise ValueError("temporal.window is unsupported; choose none, hanning or tukey")
    _finite_number(c["temporal"]["weight"], "temporal.weight", minimum=0.0)
    if not isinstance(c["temporal"]["max_disp_xyz"], (list, tuple)) or len(c["temporal"]["max_disp_xyz"]) != 3:
        raise ValueError("temporal.max_disp_xyz must be length-3")
    c["temporal"]["max_disp_xyz"] = [_finite_number(v, "temporal.max_disp_xyz", minimum=0.0) for v in c["temporal"]["max_disp_xyz"]]
    _finite_number(c["temporal"]["consistency_threshold"], "temporal.consistency_threshold", minimum=0.0)
    if c["temporal"]["loss_type"] not in {"mse", "kl"} or not isinstance(c["temporal"]["slice_wise"], bool) or not isinstance(c["temporal"]["occlusion"], bool):
        raise ValueError("temporal settings are invalid")
    if c["prediction_model"] not in {"student", "ema"}:
        raise ValueError("prediction_model must be student or ema")
    if c["training"]["method"] in {"supervised", "supervised_extra"} and c["prediction_model"] == "ema":
        raise ValueError("supervised-only methods have no EMA prediction model")
    if c["training"]["method"] in {"supervised", "supervised_extra"}:
        if c["regions"]["mode"] != "none" and float(c["regions"]["weight"]) > 0:
            raise ValueError("active region auxiliaries require an unlabeled EMA method")
        if c["temporal"]["mode"] != "none" and float(c["temporal"]["weight"]) > 0:
            raise ValueError("active temporal auxiliaries require an unlabeled EMA method")
    if c["training"]["precision"] == "amp" and not allow_full_run:
        # AMP is explicit but only allowed after the bounded software smoke.
        raise ValueError("AMP requires explicit full-run opt-in")

    if "correction" in config:
        correction = _strict_section(config["correction"], "correction", {
            "protocol": "none", "mode": "manual_only", "dataset_role": "TRAIN",
        })
        if correction["protocol"] not in {"none", "automatic_only", "interactive_extra_supervision"}:
            raise ValueError("correction.protocol is invalid")
        if correction["mode"] not in {"manual_only", "geodesic_3d", "geodesic_2d", "image_guided"}:
            raise ValueError("correction.mode is invalid")
        if correction["dataset_role"] not in {"TRAIN", "FIXTURE"}:
            raise ValueError("correction.dataset_role must be TRAIN or FIXTURE")
        c["correction"] = correction
    return c


def load_volume_config(path: str | Path, *, allow_full_run: bool = False) -> dict[str, Any]:
    p = Path(path)
    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read volume config: {p}") from exc
    return validate_volume_config(raw, allow_full_run=allow_full_run)


def config_sha256(config: Mapping[str, Any]) -> str:
    """Digest normalized JSON so manifests can pin the exact recipe."""
    normalized = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(normalized).hexdigest()


__all__ = ["SCHEMA", "config_sha256", "default_volume_config", "load_volume_config", "validate_volume_config"]
