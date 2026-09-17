"""Contract tests for the joint-from-epoch-1 profiles and their native runner.

These cover the schema and wiring half of the profile only:

  * both profiles parse under the strict Schema Version 1 loader;
  * each declares exactly ONE schedule interval covering [0, 130) with
    ``trainable: all`` and ``objective: retained_final_annotation``, so both
    modules and the accept/reject gate are live from the first optimizer step;
  * the interval learning rates are the canonical EARLY-training rates, not the
    canonical joint fine-tune rates;
  * best selection is gated from epoch 0, which is the start of the gated
    interval, under the unchanged selection rule;
  * the ACDC and M&Ms profiles keep separate dataset contracts and separate
    output identities, so their runs cannot be mixed;
  * the profiles carry the baseline ``window_mode: current`` while the native
    Candidate C runner selects ``candidate_c`` explicitly;
  * the native runner's CURRICULUM enum is closed, defaults to
    ``joint_from_start`` for both flows, and keeps the staged flow's original
    predicted-history exposure behaviour.

Nothing here trains anything, downloads an encoder or touches a GPU: the
runner is exercised only through its config-generation block.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml

from self_audit.training.unified_config import load_unified_config

ROOT = Path(__file__).resolve().parents[1]

ACDC_PROFILE = ROOT / "configs/self_audit_joint_from_start.yaml"
MNMS_PROFILE = ROOT / "configs/self_audit_joint_from_start_mnms.yaml"
ACDC_STAGED = ROOT / "configs/self_audit_full.yaml"
MNMS_STAGED = ROOT / "configs/self_audit_full_mnms.yaml"
RUNNER = ROOT / "scripts/run_acdc_mnms_candidate_c.sh"

PROFILES = (ACDC_PROFILE, MNMS_PROFILE)

#: The early-training rates the profiles adopt, taken from the canonical
#: staged bootstrap / auditor intervals rather than from the epoch-120
#: fine-tune interval.
EXPECTED_ENCODER_LR = 3e-5
EXPECTED_ANNOTATION_LR = 3e-4
EXPECTED_AUDITOR_LR = 3e-4

#: The canonical joint fine-tune rates, which must NOT be reused here: they
#: assume 120 epochs of prior convergence.
STAGED_JOINT_LRS = (1e-6, 1e-5, 1e-5)

TOTAL_EPOCHS = 130


@pytest.fixture(scope="module")
def loaded():
    return {path: load_unified_config(path) for path in PROFILES}


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.name)
def test_profile_parses_strictly(path, loaded):
    """The strict Schema Version 1 loader accepts the profile unchanged."""
    config = loaded[path]
    assert config.schema_version == 1
    assert config.experiment.protocol == "unified_schedule_v1"


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.name)
def test_single_all_trainable_interval_from_epoch_zero(path, loaded):
    """Exactly one interval, covering [0, 130), training everything."""
    schedule = loaded[path].training.schedule
    assert schedule.total_epochs == TOTAL_EPOCHS
    assert len(schedule.intervals) == 1, (
        "the joint-from-start profile must not carry a staged curriculum"
    )
    interval = schedule.intervals[0]
    assert interval.start_epoch == 0
    assert interval.end_epoch == TOTAL_EPOCHS
    assert interval.name == "joint_self_audit"
    assert interval.trainable == "all"
    assert interval.objective == "retained_final_annotation"
    assert interval.transition_population == "active_attempted"
    assert interval.rollout == "threshold_gate"
    assert interval.annotation_weight == 1.0
    assert interval.audit_weight == 1.0
    assert interval.reset_optimizer is True
    # Epoch 0 resolves to that interval, so the gate is consulted from the
    # first step rather than after a frozen bootstrap.
    assert schedule.get_interval(0) is interval
    assert schedule.get_interval(TOTAL_EPOCHS - 1) is interval


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.name)
def test_interval_uses_early_training_learning_rates(path, loaded):
    """Start rates are the early-training rates, not the fine-tune rates."""
    interval = loaded[path].training.schedule.intervals[0]
    assert interval.encoder_lr == pytest.approx(EXPECTED_ENCODER_LR)
    assert interval.annotation_lr == pytest.approx(EXPECTED_ANNOTATION_LR)
    assert interval.auditor_lr == pytest.approx(EXPECTED_AUDITOR_LR)
    assert (
        interval.encoder_lr,
        interval.annotation_lr,
        interval.auditor_lr,
    ) != STAGED_JOINT_LRS


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.name)
def test_selection_gated_from_epoch_zero_on_unchanged_rule(path, loaded):
    """Selection opens at the gated interval's start; the rule is unchanged."""
    config = loaded[path]
    checkpoint = config.checkpoint
    interval = config.training.schedule.intervals[0]
    assert checkpoint.best_selection_min_epoch == 0
    assert checkpoint.best_selection_min_epoch == interval.start_epoch
    assert checkpoint.best_selection_min_epoch < interval.end_epoch
    assert checkpoint.best_metric == "final_foreground_macro_dice"
    assert checkpoint.best_metric_mode == "max"
    assert checkpoint.save_best is True
    assert checkpoint.save_last is True


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.name)
def test_baseline_window_mode_and_no_auxiliary_exposure(path, loaded):
    """Fair baseline network, and no duplicated auxiliary rollout."""
    config = loaded[path]
    assert config.model.window_mode == "current"
    # False here means "do not run a second, auxiliary predicted-history
    # rollout".  The gated joint rollout still consumes accepted predicted
    # auditor feedback, so this does NOT disable feedback.
    assert config.training.rollout.predicted_history_exposure is False
    assert config.training.rollout.tau == 0.0
    assert config.training.rollout.max_turns == 3


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.name)
def test_lr_curve_warmup_ramp_retained(path, loaded):
    """The optimizer LR ramp is inherited from the canonical recipe."""
    lr_curve = loaded[path].training.lr_curve
    assert lr_curve.name == "warmup_cosine"
    assert lr_curve.warmup_epochs == 5
    assert lr_curve.min_ratio == 0.0


@pytest.mark.parametrize(
    "profile,staged",
    [(ACDC_PROFILE, ACDC_STAGED), (MNMS_PROFILE, MNMS_STAGED)],
    ids=["acdc", "mnms"],
)
def test_dataset_and_runtime_contract_copied_from_staged_baseline(profile, staged):
    """Everything outside the schedule / selection / identity is untouched."""
    new = yaml.safe_load(profile.read_text(encoding="utf-8"))
    old = yaml.safe_load(staged.read_text(encoding="utf-8"))

    assert new["dataset"] == old["dataset"]
    assert new["calibration"] == old["calibration"]
    assert new["diagnostics"] == old["diagnostics"]
    assert new["experiment"]["seed"] == old["experiment"]["seed"]
    assert new["experiment"]["device"] == old["experiment"]["device"]
    assert new["experiment"]["deterministic"] == old["experiment"]["deterministic"]

    for key in ("encoder_name", "pretrained_encoder", "fallback", "shared_channels",
                "num_classes", "window_k", "max_turns", "window_mode", "candidate_c"):
        assert new["model"][key] == old["model"][key], key
    assert new["model"]["pretrained_encoder"] is True

    for key in ("amp", "optimizer", "lr_curve", "annotation_loss", "audit_loss",
                "counterfactual"):
        assert new["training"][key] == old["training"][key], key
    assert new["training"]["amp"]["dtype"] == "bfloat16"
    assert new["training"]["rollout"] == old["training"]["rollout"]


def test_acdc_and_mnms_profiles_stay_separate():
    """The two profiles never share a dataset root, split or output path."""
    acdc = yaml.safe_load(ACDC_PROFILE.read_text(encoding="utf-8"))
    mnms = yaml.safe_load(MNMS_PROFILE.read_text(encoding="utf-8"))

    assert acdc["dataset"]["name"] == "acdc"
    assert mnms["dataset"]["name"] == "mnms"
    assert acdc["dataset"]["data_root"] != mnms["dataset"]["data_root"]
    assert acdc["dataset"]["data_root"] == "preprocessed_data/ACDC"
    assert mnms["dataset"]["data_root"] == "preprocessed_data/mnm"
    assert acdc["dataset"]["split_manifest"] == "splits/acdc_patient_split_seed42.json"
    assert mnms["dataset"]["split_manifest"] is None
    assert acdc["dataset"]["class_mapping"] == {0: 0, 1: 1, 2: 2, 3: 3}
    assert mnms["dataset"]["class_mapping"] == {0: 0, 1: 3, 2: 2, 3: 1}

    assert acdc["experiment"]["name"] != mnms["experiment"]["name"]
    assert acdc["checkpoint"]["output_dir"] != mnms["checkpoint"]["output_dir"]
    assert acdc["logging"]["report_dir"] != mnms["logging"]["report_dir"]
    assert acdc["logging"]["wandb"]["run_name"] != mnms["logging"]["wandb"]["run_name"]


def test_new_output_identities_do_not_collide_with_staged_runs():
    """Nothing here writes into a staged baseline's directory."""
    staged_dirs = set()
    for path in (ACDC_STAGED, MNMS_STAGED):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        staged_dirs.add(cfg["checkpoint"]["output_dir"])
        staged_dirs.add(cfg["logging"]["report_dir"])

    for path in PROFILES:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["checkpoint"]["output_dir"] not in staged_dirs
        assert cfg["logging"]["report_dir"] not in staged_dirs
        assert "joint_from_start" in cfg["checkpoint"]["output_dir"]
        assert "joint_from_start" in cfg["logging"]["report_dir"]


def test_staged_baseline_configs_are_preserved():
    """The staged baselines still carry their original three intervals."""
    for path in (ACDC_STAGED, MNMS_STAGED):
        config = load_unified_config(path)
        names = [i.name for i in config.training.schedule.intervals]
        assert names == ["annotation_bootstrap", "auditor_training", "joint_self_audit"]
        assert config.checkpoint.best_selection_min_epoch == 120
        assert config.training.rollout.predicted_history_exposure is False


# --------------------------------------------------------------------------
# Native runner wiring.  Config generation only: no training, no GPU.
# --------------------------------------------------------------------------


def _extract_generation_block() -> str:
    text = RUNNER.read_text(encoding="utf-8")
    match = re.search(r"^python - <<'PY'\n(.*?)^PY$", text, re.MULTILINE | re.DOTALL)
    assert match, "could not locate the config-generation heredoc in the runner"
    return match.group(1)


def _run_generation(tmp_path: Path, curriculum: str, sources: tuple[Path, Path]):
    """Run the runner's config-generation block only. No training, no GPU."""
    script = tmp_path / "generate.py"
    script.write_text(_extract_generation_block(), encoding="utf-8")
    acdc_out = tmp_path / "acdc.yaml"
    mnms_out = tmp_path / "mnms.yaml"
    env = {
        **os.environ,
        "BATCH_SIZE": "8",
        "CURRICULUM": curriculum,
        "ACDC_SOURCE_CONFIG": str(sources[0]),
        "MNMS_SOURCE_CONFIG": str(sources[1]),
        "ACDC_CONFIG": str(acdc_out),
        "MNMS_CONFIG": str(mnms_out),
    }
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    # The generation block must never reach a training command.
    assert "train_self_audit" not in proc.stdout
    return proc.stdout, acdc_out, mnms_out


def test_runner_defaults_to_joint_from_start_for_both_flows():
    """The default CURRICULUM selects the new profile for ACDC and M&Ms."""
    text = RUNNER.read_text(encoding="utf-8")
    assert 'CURRICULUM="${CURRICULUM:-joint_from_start}"' in text
    assert "configs/self_audit_joint_from_start.yaml" in text
    assert "configs/self_audit_joint_from_start_mnms.yaml" in text
    # Both native runs stay independent: no resume is ever passed as a command
    # argument (the prose above the enum explains why, so comments are skipped).
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "--resume" not in code


def test_runner_curriculum_enum_fails_closed():
    """An unrecognised CURRICULUM aborts before anything is generated."""
    env = {**os.environ, "CURRICULUM": "definitely_not_a_profile"}
    proc = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "definitely_not_a_profile" in proc.stderr
    assert "joint_from_start" in proc.stderr and "staged" in proc.stderr


def test_runner_joint_generation_is_candidate_c_without_extra_exposure(tmp_path):
    """Joint flow: explicit candidate_c, batch 8, exposure left OFF."""
    stdout, acdc_out, mnms_out = _run_generation(
        tmp_path, "joint_from_start", (ACDC_PROFILE, MNMS_PROFILE)
    )
    assert "strict schema validation OK" in stdout

    for generated in (acdc_out, mnms_out):
        cfg = yaml.safe_load(generated.read_text(encoding="utf-8"))
        assert cfg["model"]["window_mode"] == "candidate_c"
        assert cfg["training"]["rollout"]["predicted_history_exposure"] is False
        intervals = cfg["training"]["schedule"]["intervals"]
        assert len(intervals) == 1
        assert [i["batch_size"] for i in intervals] == [8]
        # The generated run config is still strictly valid.
        load_unified_config(generated)

    assert "epochs [0, 130)" in stdout
    assert "trainable=all" in stdout
    assert "objective=retained_final_annotation" in stdout
    assert "best_selection_min_epoch=0" in stdout
    # The banner must state the flag's real meaning, not imply feedback is off.
    assert "live from epoch 0" in stdout
    assert "no curriculum warmup interval" in stdout


def test_runner_staged_generation_retains_original_exposure(tmp_path):
    """Staged flow keeps its original predicted-history exposure behaviour."""
    stdout, acdc_out, mnms_out = _run_generation(
        tmp_path, "staged", (ACDC_STAGED, MNMS_STAGED)
    )
    for generated in (acdc_out, mnms_out):
        cfg = yaml.safe_load(generated.read_text(encoding="utf-8"))
        assert cfg["model"]["window_mode"] == "candidate_c"
        assert cfg["training"]["rollout"]["predicted_history_exposure"] is True
        assert cfg["training"]["rollout"]["predicted_history_weight"] == 0.1
        intervals = cfg["training"]["schedule"]["intervals"]
        assert len(intervals) == 3
        assert [i["batch_size"] for i in intervals] == [8, 8, 8]
        load_unified_config(generated)

    assert "epochs [0, 100)" in stdout
    assert "epochs [120, 130)" in stdout
    assert "best_selection_min_epoch=120" in stdout


def test_runner_generation_validation_is_fail_closed(tmp_path):
    """A profile the strict loader rejects aborts generation, not a warning."""
    broken = tmp_path / "broken.yaml"
    cfg = yaml.safe_load(ACDC_PROFILE.read_text(encoding="utf-8"))
    # Selection outside the gated interval is rejected by the loader.
    cfg["checkpoint"]["best_selection_min_epoch"] = 999
    broken.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    script = tmp_path / "generate.py"
    script.write_text(_extract_generation_block(), encoding="utf-8")
    env = {
        **os.environ,
        "BATCH_SIZE": "8",
        "CURRICULUM": "joint_from_start",
        "ACDC_SOURCE_CONFIG": str(broken),
        "MNMS_SOURCE_CONFIG": str(MNMS_PROFILE),
        "ACDC_CONFIG": str(tmp_path / "acdc.yaml"),
        "MNMS_CONFIG": str(tmp_path / "mnms.yaml"),
    }
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0
    assert "best_selection_min_epoch" in proc.stderr
    assert "validation skipped" not in proc.stdout


def test_runner_does_not_write_an_intermediate_banner_artifact():
    """The resolved schedule is printed inline, not staged on disk."""
    text = RUNNER.read_text(encoding="utf-8")
    assert "BANNER_FILE" not in text


def test_runner_keeps_native_batch_size_override():
    """BATCH_SIZE stays the existing native override, defaulting to 8."""
    text = RUNNER.read_text(encoding="utf-8")
    assert 'BATCH_SIZE="${BATCH_SIZE:-8}"' in text


def test_pipeline_runner_default_config_unchanged():
    """run_full_pipeline.sh still defaults to the canonical staged config."""
    text = (ROOT / "scripts/run_full_pipeline.sh").read_text(encoding="utf-8")
    assert 'CONFIG="configs/self_audit_full.yaml"' in text
    assert "configs/self_audit_joint_from_start.yaml" in text
