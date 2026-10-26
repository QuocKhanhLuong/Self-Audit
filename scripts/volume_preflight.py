#!/usr/bin/env python3
"""Validate the native-volume registry and one experiment configuration.

This command is intentionally side-effect free.  It checks that an ablation
has a declared parent/factor contract and that its runtime budget is bounded;
it does not construct a dataset, read masks, download weights, or start
training.  The smoke and profile commands import the small public helpers in
this module so all three commands apply the same validation rules.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping

# Direct execution from an arbitrary working directory (for example
# ``python /repo/scripts/volume_preflight.py`` while cwd is /tmp) does not put
# the repository root on sys.path.  Bootstrap it before importing the strict
# integration validator; silently falling back to local checks would make an
# absolute-path preflight weaker than a cwd invocation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA = "self_audit.volume.v1"
REGISTRY_SCHEMA = "self_audit.volume.registry.v1"
IDS = (
    [f"V{i:02d}" for i in range(9)]
    + [f"O{i:02d}" for i in range(1, 4)]
    + ["S01", "S02"]
    + [f"R{i:02d}" for i in range(3)]
    + ["T00", "T01", "I00", "I01", "C00", "C01"]
)
_ID_RE = re.compile(r"^(?:V(?:0[0-8])|O0[1-3]|S0[12]|R0[0-2]|T0[01]|I0[01]|C0[01])$")
REQUIRED_SECTIONS = (
    "model",
    "geometry",
    "loss",
    "optimization",
    "training",
    "regions",
    "temporal",
)
FORBIDDEN_TOKENS = (
    "u_mask",
    "val_mask",
    "test_mask",
    "external_teacher",
    "cinema_teacher",
    "totalsegmentator",
    "sam_mask",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_registry_path() -> Path:
    return repository_root() / "configs" / "native_volume" / "REGISTRY.json"


CANONICAL_DINO = {
    "backbone": "dinov3_convnext_tiny",
    "source_repo": "/root/self-audit-pilot-artifacts/backbone/dinov3",
    "source_commit": "6876159a11b4df116f30f667f8c9888617df0751",
    "weights": "/root/self-audit-pilot-artifacts/backbone/pretrained/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    "weights_sha256": "21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54",
}
CANONICAL_DECODER_WIDTHS = [96, 128, 192, 256]
CANONICAL_CNN_WIDTHS = [16, 32, 64, 128]
SYNTHETIC_DECODER_WIDTHS = [8, 16, 24, 32]


def _walk_forbidden(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_path = f"{path}.{key}" if path else str(key)
            lowered = str(key).lower().replace("-", "_")
            if any(token in lowered for token in FORBIDDEN_TOKENS):
                found.append(key_path)
            found.extend(_walk_forbidden(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_walk_forbidden(item, f"{path}[{index}]"))
    return found


def load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON configuration: {target}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a JSON object: {target}")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    config = load_json(path)
    validate_config(config)
    return config


def _positive_number(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")


def validate_config(config: Mapping[str, Any], *, registry_ids: set[str] | None = None, config_id: str | None = None) -> None:
    """Validate one config without importing optional training dependencies."""
    if config.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")
    forbidden_paths = _walk_forbidden(config)
    if forbidden_paths:
        raise ValueError(f"{config_id or 'config'}: forbidden mask/teacher reference in {', '.join(forbidden_paths)}")
    # Always use the integration owner's strict validator.  Import bootstrap
    # above makes this deterministic for both cwd and absolute invocations;
    # an unavailable package is a hard preflight error, never a local-check
    # fallback that could falsely certify a recipe.
    from self_audit.volume.config import validate_volume_config
    validate_volume_config(config, allow_full_run=False)
    exp_id = config_id or "config"
    if config_id is not None and (not _ID_RE.fullmatch(config_id) or config_id not in IDS):
        raise ValueError(f"unknown native-volume experiment id: {config_id!r}")
    if registry_ids is not None and config_id is not None and config_id not in registry_ids:
        raise ValueError(f"{config_id} is not declared by the registry")
    for section in REQUIRED_SECTIONS:
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"{exp_id}: missing object section {section}")
    model = config["model"]
    if not isinstance(model, Mapping):
        raise ValueError(f"{exp_id}: model must be an object")
    historical = bool(config.get("historical_reference", False))
    if not historical:
        for key, expected in CANONICAL_DINO.items():
            if model.get(key) != expected:
                raise ValueError(f"{exp_id}: production model.{key} must match canonical DINO provenance")
        if config.get("window", {}).get("mode") != "fixed":
            raise ValueError(f"{exp_id}: production configs require window.mode=fixed")
    architecture = config.get("architecture", {})
    if not isinstance(architecture, Mapping) or architecture.get("name") not in {"volume25d", "hybrid", "cnn3d"}:
        raise ValueError(f"{exp_id}: unsupported architecture.name")
    if architecture.get("depth_mode", "off") not in {"off", "mix", "planar_control", "anisotropic"}:
        raise ValueError(f"{exp_id}: unsupported architecture.depth_mode")
    if not historical and architecture.get("name") in {"volume25d", "hybrid"} and architecture.get("widths") != CANONICAL_DECODER_WIDTHS:
        raise ValueError(f"{exp_id}: production architecture.widths must be {CANONICAL_DECODER_WIDTHS}")
    if not historical and architecture.get("name") == "cnn3d":
        if architecture.get("widths") != CANONICAL_CNN_WIDTHS:
            raise ValueError(f"{exp_id}: production CNN3D widths must be {CANONICAL_CNN_WIDTHS}")
        if len(architecture.get("widths", [])) != len(architecture.get("z_kernels", [])):
            raise ValueError(f"{exp_id}: CNN3D widths and z_kernels must declare matching stages")
    geometry = config["geometry"]
    slab_depth = geometry.get("slab_depth")
    if not isinstance(slab_depth, int) or slab_depth < 1 or slab_depth > 32:
        raise ValueError(f"{exp_id}: geometry.slab_depth must be in [1,32]")
    crop = geometry.get("crop_size")
    if isinstance(crop, int):
        if crop < 1 or crop > 256:
            raise ValueError(f"{exp_id}: geometry.crop_size must be in [1,256]")
    elif isinstance(crop, list) and len(crop) == 2 and all(isinstance(x, int) and 1 <= x <= 256 for x in crop):
        pass
    else:
        raise ValueError(f"{exp_id}: geometry.crop_size must be an int or bounded [height,width]")
    loss_mode = config["loss"].get("mode")
    if loss_mode not in {"legacy_pooled", "volume_mean"}:
        raise ValueError(f"{exp_id}: loss.mode must be legacy_pooled or volume_mean")
    optimization = config["optimization"]
    if optimization.get("recipe") not in {"legacy", "grouped_adamw"}:
        raise ValueError(f"{exp_id}: unsupported optimization.recipe")
    training = config["training"]
    steps = training.get("steps")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError(f"{exp_id}: training.steps must be a positive integer")
    if steps > 10 and not bool(config.get("historical_reference", False)):
        raise ValueError(f"{exp_id}: non-historical runtime steps must be <= 10")
    if training.get("precision") not in {"fp32", "amp"}:
        raise ValueError(f"{exp_id}: unsupported training.precision")
    if training.get("microbatch", 1) < 1 or training.get("accumulation", 1) < 1:
        raise ValueError(f"{exp_id}: microbatch and accumulation must be positive")
    # The historical A0 descriptor is immutable and intentionally records its
    # canonical batch-24 recipe.  It is provenance-only (never launched by
    # smoke/profile), so permit that one value while retaining the tighter
    # tooling bound for all executable arms.
    microbatch_cap = 24 if bool(config.get("historical_reference", False)) else 8
    if int(training.get("microbatch", 1)) > microbatch_cap or int(training.get("accumulation", 1)) > 8:
        raise ValueError(
            f"{exp_id}: bounded tooling supports microbatch <= {microbatch_cap} and accumulation <= 8"
        )
    if int(training.get("threads", 1)) > 32:
        raise ValueError(f"{exp_id}: bounded tooling supports threads <= 32")
    if config.get("prediction_model", "student") not in {"student", "ema"}:
        raise ValueError(f"{exp_id}: prediction_model must be student or ema")


def validate_runtime_bounds(config: Mapping[str, Any], *, max_seconds: float | None = None,
                            max_minutes: float | None = None) -> None:
    """Fail closed on every bounded synthetic override before allocation."""
    if max_seconds is not None and (not isinstance(max_seconds, (int, float)) or isinstance(max_seconds, bool)
                                    or not math.isfinite(float(max_seconds)) or not 0 < float(max_seconds) <= 600):
        raise ValueError("max_seconds must be finite and in (0,600]")
    if max_minutes is not None and (not isinstance(max_minutes, (int, float)) or isinstance(max_minutes, bool)
                                    or not math.isfinite(float(max_minutes)) or not 0 < float(max_minutes) <= 20):
        raise ValueError("max_minutes must be finite and in (0,20]")
    geometry = config.get("geometry", {})
    depth = geometry.get("slab_depth")
    crop = geometry.get("crop_size")
    if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= 16:
        raise ValueError("synthetic runtime geometry.slab_depth must be an integer in [1,16]")
    dims = [crop] if isinstance(crop, int) else list(crop) if isinstance(crop, (list, tuple)) else []
    if len(dims) not in {1, 2} or any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 256 for v in dims):
        raise ValueError("synthetic runtime geometry.crop_size must be bounded integer dimensions")
    training = config.get("training", {})
    for key, cap in (("steps", 8), ("microbatch", 8), ("accumulation", 8), ("threads", 32)):
        value = training.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= cap:
            raise ValueError(f"synthetic runtime training.{key} must be an integer in [1,{cap}]")
    if training["microbatch"] * training["accumulation"] > 64:
        raise ValueError("synthetic runtime logical batch must be <= 64")


def synthetic_runtime_config(
    config: Mapping[str, Any],
    *,
    steps: int | None = None,
    crop_size: int | list[int] | tuple[int, int] = 64,
    microbatch: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an explicit bounded CPU override and a receipt of every change.

    Scientific recipes remain hashable DINO/fixed-window configs.  This helper
    is the only place smoke/profile may substitute the tiny synthetic backbone,
    crop, step count, or logical microbatch; all overrides are returned for the
    receipt rather than silently mutating the source recipe.
    """
    runtime = copy.deepcopy(dict(config))
    overrides: dict[str, Any] = {}

    def set_value(path: str, parent: dict[str, Any], key: str, value: Any) -> None:
        old = parent.get(key)
        if old != value:
            overrides[path] = {"from": old, "to": value}
            parent[key] = value

    model = runtime.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("model must be an object for synthetic runtime overrides")
    set_value("model.backbone", model, "backbone", "tiny_test")
    architecture = runtime.setdefault("architecture", {})
    if not isinstance(architecture, dict):
        raise ValueError("architecture must be an object for synthetic runtime overrides")
    old_widths = architecture.get("widths")
    # Every executable synthetic architecture uses four explicit tiny stages;
    # CNN3D retains its separate topology while matching this four-stage
    # synthetic width tuple (production CNN rows declare [16,32,64,128]).
    synthetic_widths = SYNTHETIC_DECODER_WIDTHS
    overrides["architecture.widths"] = {"from": old_widths, "to": list(synthetic_widths),
                                         "explicit": old_widths == synthetic_widths}
    architecture["widths"] = list(synthetic_widths)
    geometry = runtime.setdefault("geometry", {})
    if not isinstance(geometry, dict):
        raise ValueError("geometry must be an object for synthetic runtime overrides")
    set_value("geometry.crop_size", geometry, "crop_size", crop_size)
    training = runtime.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("training must be an object for synthetic runtime overrides")
    if steps is not None:
        if steps < 1:
            raise ValueError("synthetic runtime steps must be positive")
        set_value("training.steps", training, "steps", int(steps))
    effective_microbatch = int(training.get("microbatch", 1)) if microbatch is None else int(microbatch)
    if effective_microbatch < 1:
        raise ValueError("synthetic runtime microbatch must be positive")
    old_microbatch = training.get("microbatch")
    # Record the effective logical-batch choice even when the bounded default
    # equals the recipe, so every synthetic receipt proves its batch override
    # rather than silently inheriting a production setting.
    if old_microbatch == effective_microbatch:
        overrides["training.microbatch"] = {"from": old_microbatch, "to": effective_microbatch, "explicit": True}
    else:
        set_value("training.microbatch", training, "microbatch", effective_microbatch)
    validate_runtime_bounds(runtime)
    return runtime, overrides


def load_registry(path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    registry_path = Path(path) if path is not None else default_registry_path()
    registry = load_json(registry_path)
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"registry schema must be {REGISTRY_SCHEMA}")
    entries = registry.get("experiments")
    if not isinstance(entries, list):
        raise ValueError("registry.experiments must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("each registry experiment must be an object")
        exp_id = entry.get("id")
        if not isinstance(exp_id, str) or exp_id in by_id:
            raise ValueError(f"duplicate/invalid registry id: {exp_id!r}")
        if exp_id not in IDS:
            raise ValueError(f"registry contains unexpected id: {exp_id}")
        if not isinstance(entry.get("config"), str) or not entry["config"].endswith(".json"):
            raise ValueError(f"{exp_id}: registry config path must be a JSON path")
        by_id[exp_id] = dict(entry)
    missing = sorted(set(IDS) - set(by_id))
    if missing:
        raise ValueError(f"registry missing experiment ids: {', '.join(missing)}")
    root = registry_path.parent
    for exp_id, entry in by_id.items():
        config_path = root / entry["config"]
        config = load_config(config_path)
        validate_config(config, registry_ids=set(by_id), config_id=exp_id)
        if bool(entry.get("historical_reference", False)):
            source_rel = entry.get("historical_a0_source")
            expected_sha = entry.get("historical_a0_sha256")
            if not isinstance(source_rel, str) or not isinstance(expected_sha, str):
                raise ValueError(f"{exp_id}: historical reference requires source path and SHA-256")
            source = repository_root() / source_rel
            if not source.is_file():
                raise ValueError(f"{exp_id}: historical A0 config is missing: {source_rel}")
            actual_sha = sha256_file(source)
            if actual_sha != expected_sha:
                raise ValueError(f"{exp_id}: historical A0 SHA-256 mismatch ({actual_sha} != {expected_sha})")
            canonical = load_json(source)
            recipe = entry.get("historical_recipe", {})
            checks = {
                "method": "ema_full",
                "steps": 10000,
                "warmup_steps": 1000,
                "batch_size": 24,
                "crop_size": 256,
            }
            observed = {
                "method": canonical.get("method"),
                "steps": canonical.get("steps"),
                "warmup_steps": canonical.get("warmup_steps"),
                "batch_size": canonical.get("batch_size"),
                "crop_size": canonical.get("crop_size"),
            }
            if observed != checks or any(recipe.get(key) != value for key, value in checks.items() if key in recipe):
                raise ValueError(f"{exp_id}: historical recipe fields do not match canonical A0 ({observed})")
            if config.get("historical_a0_source") != source_rel or config.get("historical_a0_sha256") != expected_sha:
                raise ValueError(f"{exp_id}: config historical reference does not match registry")
    # A parent may only point to an earlier, declared arm.  This catches typo
    # cycles before an actual training invocation can be attempted.
    for exp_id, entry in by_id.items():
        parent = entry.get("parent")
        seen: set[str] = set()
        while parent is not None:
            if parent in seen or parent == exp_id:
                raise ValueError(f"registry parent cycle involving {exp_id}")
            seen.add(parent)
            if parent not in by_id:
                raise ValueError(f"{exp_id}: parent {parent} is not declared")
            parent = by_id[parent].get("parent")
    return registry, by_id


def config_path_for(exp_id: str, registry_path: str | Path | None = None) -> Path:
    _, entries = load_registry(registry_path)
    if exp_id not in entries:
        raise ValueError(f"unknown experiment id: {exp_id}")
    registry_file = Path(registry_path) if registry_path is not None else default_registry_path()
    return registry_file.parent / entries[exp_id]["config"]


def bounded_steps(config: Mapping[str, Any], requested: int | None = None, *, historical_reference: bool = False) -> int:
    configured = int(config["training"].get("steps", 1))
    # Historical V00 is represented faithfully in its config, but tooling is
    # never allowed to launch its 10k-step recipe accidentally.
    cap = 8 if historical_reference else 10
    if requested is not None:
        if requested < 1:
            raise ValueError("requested steps must be positive")
        cap = min(cap, requested)
    return min(configured, cap)


def make_receipt(command: str, config_path: Path, config: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": "self_audit.volume.tooling_receipt.v1",
        "command": command,
        "status": "PASS",
        "config_id": config_path.stem,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "synthetic": True,
        "real_train_smoke": "NOT_RUN",
        "gpu": "NOT_RUN",
    }
    receipt.update(extra)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="native-volume JSON config")
    parser.add_argument("--registry", type=Path, default=None, help="registry JSON (default: configs/native_volume/REGISTRY.json)")
    parser.add_argument("--output", type=Path, default=None, help="optional receipt JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        registry, entries = load_registry(args.registry)
        config = load_config(args.config)
        config_id = args.config.stem
        validate_config(config, registry_ids=set(entries), config_id=config_id)
        if config_id not in entries:
            raise ValueError(f"config filename stem {config_id!r} is not in registry")
        entry = entries[config_id]
        receipt = make_receipt(
            "preflight",
            args.config,
            config,
            parent=entry.get("parent"),
            changed_factors=entry.get("changed_factors", []),
            registry_sha256=sha256_file(args.registry or default_registry_path()),
            bounded_steps=("NOT_APPLICABLE_HISTORICAL_REFERENCE" if bool(entry.get("historical_reference", False))
                           else bounded_steps(config)),
            config_status=entry.get("status", "IMPLEMENTED"),
            historical_reference=bool(entry.get("historical_reference", False)),
            synthetic=not bool(entry.get("historical_reference", False)),
        )
    except (OSError, ValueError, KeyError) as exc:
        receipt = {"schema": "self_audit.volume.tooling_receipt.v1", "command": "preflight", "status": "FAIL", "error": str(exc)}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
