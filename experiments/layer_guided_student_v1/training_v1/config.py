"""Versioned training settings, separate from the representation-only scaffold."""

import math
from pathlib import Path

from integration.artifacts import (
    ARCHITECTURE,
    CONTROL,
    PREPROCESSING,
    sha256,
)
from integration.artifacts import (
    source_hashes as implementation_hashes,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = {
    "schema": "layer_guided.training_recipe.v1",
    "architecture": ARCHITECTURE,
    "seed": 42,
    "detail_bound": 1.0,
    "preprocessing": PREPROCESSING,
    "semantic_policy": "frozen_historical_q_reliability_v1",
    "feature_objective": "full_native_feature_cosine_v1",
    "optimizer": "adamw",
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "feature_weight": 1.0,
    "semantic_weight": 1.0,
    "gradient_clip_norm": 5.0,
    "epochs": 50,
    "batch_size": 1,
    "max_steps": 32,
    "threads": 4,
    "guard": "off",
    "refiner": "off",
    "ontology": ["BG", "RV", "MYO", "LV"],
}


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or set(recipe) != set(DEFAULT_RECIPE):
        raise ValueError(
            "explicit layer-guided training recipe required; unknown/missing keys"
        )
    for key in (
        "schema",
        "preprocessing",
        "semantic_policy",
        "feature_objective",
        "optimizer",
        "guard",
        "refiner",
        "ontology",
    ):
        if recipe[key] != DEFAULT_RECIPE[key]:
            raise ValueError(f"unsupported training {key}")
    if recipe["architecture"] not in (ARCHITECTURE, CONTROL):
        raise ValueError(
            "new layer-guided architecture required; no historical student fallback"
        )
    if type(recipe["seed"]) is not int or not 0 <= recipe["seed"] < 2**32:
        raise ValueError("32-bit nonnegative seed required")
    if (
        type(recipe["detail_bound"]) not in (int, float)
        or recipe["detail_bound"] != 1.0
    ):
        raise ValueError("reference detail bound must remain 1.0")
    for key in ("epochs", "batch_size", "threads"):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ValueError(f"positive integer {key} required")
    if recipe["max_steps"] is not None and (
        type(recipe["max_steps"]) is not int or recipe["max_steps"] < 1
    ):
        raise ValueError(
            "max_steps must be a positive integer or null for the full budget"
        )
    for key in (
        "learning_rate",
        "weight_decay",
        "feature_weight",
        "semantic_weight",
        "gradient_clip_norm",
    ):
        value = recipe[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"finite numeric {key} required")
        if value < 0 or (key != "weight_decay" and value == 0):
            raise ValueError(f"invalid {key}")
    return {**recipe, "ontology": list(recipe["ontology"])}


def constructor_recipe(recipe):
    config = validate_recipe(recipe)
    return {
        "schema": "layer_guided.recipe.v1",
        "architecture": config["architecture"],
        "seed": config["seed"],
        "detail_bound": config["detail_bound"],
        "preprocessing": config["preprocessing"],
        "guard": "off",
        "refiner": "proposal_only",
        "semantic_admission": "off",
    }


def source_hashes():
    return {
        **implementation_hashes(),
        **{
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / "training_v1").glob("*.py"))
        },
    }
