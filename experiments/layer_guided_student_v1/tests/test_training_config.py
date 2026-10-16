import copy
import subprocess
import sys
from pathlib import Path

import pytest
from training_v1.config import DEFAULT_RECIPE, constructor_recipe, validate_recipe


@pytest.mark.parametrize(
    "change",
    [
        {"architecture": "GuidanceStudent"},
        {"semantic_policy": "conditional_z"},
        {"guard": "on"},
        {"optimizer": "guarded_adamw"},
        {"learning_rate": float("nan")},
        {"epochs": True},
        {"batch_size": 0},
        {"max_steps": 0},
        {"ontology": ["BG", "RV", "LV", "MYO"]},
    ],
)
def test_training_recipe_rejects_silent_policy_changes(change):
    with pytest.raises(ValueError):
        validate_recipe({**copy.deepcopy(DEFAULT_RECIPE), **change})


def test_constructor_is_exact_architecture_with_separate_training_policy():
    assert (
        constructor_recipe(DEFAULT_RECIPE)["architecture"]
        == "layer_guided_convnext_tiny_v1"
    )
    assert constructor_recipe(DEFAULT_RECIPE)["semantic_admission"] == "off"
    assert validate_recipe({**DEFAULT_RECIPE, "max_steps": None})["epochs"] == 50
    with pytest.raises(ValueError):
        validate_recipe({**DEFAULT_RECIPE, "resume": "old.pt"})


def test_training_cli_help_from_outside_checkout(tmp_path):
    script = Path(__file__).resolve().parents[3] / "scripts" / "train_astra_student.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--bank-manifest-sha256" in result.stdout
    assert "32-update pilot" in result.stdout
