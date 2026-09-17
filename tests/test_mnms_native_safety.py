"""Safety properties of the native M&Ms Candidate C path.

Two defects are pinned here:

1. The native runner used to generate its run configs at a fixed path keyed only
   on the curriculum.  The M&Ms leg reads its config hours after generation (the
   ACDC leg runs first), so a second invocation started in between rewrote the
   config the first invocation's pending M&Ms leg was about to read, and that
   leg trained under the second invocation's ``BATCH_SIZE``.
2. ``discover_mnms_records`` silently skipped an image whose mask was missing,
   and let a dictionary comprehension drop files that normalize to one case key,
   so a partially copied supervised cohort trained smaller than deployed without
   any failure.

Everything here is bounded: no training runs, no GPU is used, and the runner's
training command is replaced by a shim that records its arguments and exits.
The runner tests execute inside a throwaway repository layout under ``tmp_path``
so that no file in the developer's workspace is created, rewritten or deleted.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys

import numpy as np
import pytest
import yaml

from self_audit.data.mnms import _volume_key, discover_mnms_records
from self_audit.training._utils import validate_dataset_splits


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_acdc_mnms_candidate_c.sh"
PROFILE_CONFIGS = (
    "self_audit_joint_from_start.yaml",
    "self_audit_joint_from_start_mnms.yaml",
    "self_audit_full.yaml",
    "self_audit_full_mnms.yaml",
)


# ---------------------------------------------------------------------------
# Strict pairing of supervised cohorts
# ---------------------------------------------------------------------------


def _write_volume(path: Path, shape: tuple[int, int, int] = (8, 8, 2)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npz":
        np.savez(path, np.zeros(shape, dtype=np.float32))
    else:
        np.save(path, np.zeros(shape, dtype=np.float32))


def _write_mask(path: Path, shape: tuple[int, int, int] = (8, 8, 2)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[0:2, 0:2, :] = 1
    mask[2:4, 2:4, :] = 2
    mask[4:6, 4:6, :] = 3
    if path.suffix == ".npz":
        np.savez(path, mask)
    else:
        np.save(path, mask)


def _populate(split_dir: Path, case_ids: list[str], *, mask_suffix: str = "") -> None:
    for case_id in case_ids:
        _write_volume(split_dir / "volumes" / f"{case_id}.npy")
        _write_mask(split_dir / "masks" / f"{case_id}{mask_suffix}.npy")


def test_strict_pairing_rejects_an_image_without_a_mask(tmp_path: Path) -> None:
    split = tmp_path / "training"
    _populate(split, ["A001_t00", "B002_t00"])
    (split / "masks" / "B002_t00.npy").unlink()

    with pytest.raises(FileNotFoundError) as excinfo:
        discover_mnms_records(tmp_path, split="training", strict_pairing=True)
    message = str(excinfo.value)
    assert "images with no mask" in message
    assert "B002_t00" in message

    # The tolerant default keeps its existing behaviour for callers that rely
    # on it (external trees may deliberately carry unlabelled volumes).
    records = discover_mnms_records(tmp_path, split="training")
    assert [record.case_id for record in records] == ["A001_t00"]


def test_strict_pairing_rejects_a_mask_without_an_image(tmp_path: Path) -> None:
    split = tmp_path / "training"
    _populate(split, ["A001_t00"])
    _write_mask(split / "masks" / "C003_t00.npy")

    with pytest.raises(FileNotFoundError) as excinfo:
        discover_mnms_records(tmp_path, split="training", strict_pairing=True)
    message = str(excinfo.value)
    assert "masks with no image" in message
    assert "C003_t00" in message

    records = discover_mnms_records(tmp_path, split="training")
    assert [record.case_id for record in records] == ["A001_t00"]


def _ambiguity_duplicate_image_key(split: Path) -> None:
    """``A001_t00.npy`` and ``A001_t00.npz`` normalize to the same case key."""
    _populate(split, ["A001_t00"])
    _write_volume(split / "volumes" / "A001_t00.npz")


def _ambiguity_duplicate_mask_key(split: Path) -> None:
    _populate(split, ["A001_t00"])
    _write_mask(split / "masks" / "A001_t00.npz")


def _ambiguity_two_candidate_masks(split: Path) -> None:
    """One image, two masks that both match it (``A001_t00`` and ``A001_t00_gt``)."""
    _populate(split, ["A001_t00"])
    _write_mask(split / "masks" / "A001_t00_gt.npy")


def _ambiguity_one_mask_two_images(split: Path) -> None:
    """``p`` and ``p_gt`` are both images and would share the single mask ``p_gt``."""
    _write_volume(split / "volumes" / "A001_t00.npy")
    _write_volume(split / "volumes" / "A001_t00_gt.npy")
    _write_mask(split / "masks" / "A001_t00_gt.npy")


@pytest.mark.parametrize(
    "build, fragment",
    [
        (_ambiguity_duplicate_image_key, "files sharing one normalized key"),
        (_ambiguity_duplicate_mask_key, "files sharing one normalized key"),
        (_ambiguity_two_candidate_masks, "images matching several masks"),
        (_ambiguity_one_mask_two_images, "masks claimed by several images"),
    ],
    ids=["duplicate_image_key", "duplicate_mask_key", "two_candidate_masks", "one_mask_two_images"],
)
def test_strict_pairing_rejects_ambiguous_layouts(tmp_path: Path, build, fragment: str) -> None:
    split = tmp_path / "training"
    build(split)

    with pytest.raises(FileNotFoundError) as excinfo:
        discover_mnms_records(tmp_path, split="training", strict_pairing=True)
    assert fragment in str(excinfo.value)

    # The tolerant default is unchanged: it still returns a cohort, which is
    # exactly why it must not be used for supervised training.
    assert discover_mnms_records(tmp_path, split="training")


def test_strict_pairing_rejects_a_case_discovered_twice(tmp_path: Path) -> None:
    """``train`` and ``training`` both exist and offer the same case."""
    _populate(tmp_path / "train", ["A001_t00"])
    _populate(tmp_path / "training", ["A001_t00"])

    with pytest.raises(FileNotFoundError) as excinfo:
        discover_mnms_records(tmp_path, split="train", strict_pairing=True)
    assert "case ids discovered more than once" in str(excinfo.value)

    # Tolerant discovery still keeps the first and returns one record.
    records = discover_mnms_records(tmp_path, split="train")
    assert [record.case_id for record in records] == ["A001_t00"]


def test_tolerant_selection_matches_the_previous_dictionary_behaviour(tmp_path: Path) -> None:
    """A duplicated key must resolve to the same file the old comprehension kept."""
    split = tmp_path / "training"
    _populate(split, ["A001_t00"])
    _write_volume(split / "volumes" / "A001_t00.npz")
    _write_mask(split / "masks" / "A001_t00.npz")

    expected_image = {
        _volume_key(p): p for p in (split / "volumes").iterdir() if p.is_file()
    }["A001_t00"]
    expected_mask = {
        _volume_key(p): p for p in (split / "masks").iterdir() if p.is_file()
    }["A001_t00"]

    records = discover_mnms_records(tmp_path, split="training")
    assert len(records) == 1
    assert records[0].image_path == expected_image
    assert records[0].mask_path == expected_mask


@pytest.mark.parametrize("suffix", ["", "_gt", "_label", "_seg"])
def test_strict_pairing_accepts_every_supported_mask_suffix(tmp_path: Path, suffix: str) -> None:
    split = tmp_path / "training"
    _populate(split, ["A001_t00", "B002_t00"], mask_suffix=suffix)

    records = discover_mnms_records(tmp_path, split="training", strict_pairing=True)
    assert [record.case_id for record in records] == ["A001_t00", "B002_t00"]
    for record in records:
        assert record.mask_path.name == f"{record.case_id}{suffix}.npy"


def test_native_split_validation_fails_on_an_unpaired_training_volume(tmp_path: Path) -> None:
    _populate(tmp_path / "training", ["A001_t00", "B002_t00"])
    _populate(tmp_path / "validation", ["C003_t00"])
    (tmp_path / "training" / "masks" / "B002_t00.npy").unlink()

    config = {
        "dataset": "mnms",
        "data_root": str(tmp_path),
        "train_split": "train",
        "val_split": "val",
        "test_split": None,
        "depth_axis": 2,
        "class_mapping": {0: 0, 1: 3, 2: 2, 3: 1},
    }
    with pytest.raises(FileNotFoundError) as excinfo:
        validate_dataset_splits(config)
    assert "not unambiguously paired" in str(excinfo.value)


def test_native_split_validation_passes_on_a_fully_paired_cohort(tmp_path: Path) -> None:
    _populate(tmp_path / "training", ["A001_t00", "B002_t00"])
    _populate(tmp_path / "validation", ["C003_t00"], mask_suffix="_gt")

    config = {
        "dataset": "mnms",
        "data_root": str(tmp_path),
        "train_split": "train",
        "val_split": "val",
        "test_split": None,
        "depth_axis": 2,
        "class_mapping": {0: 0, 1: 3, 2: 2, 3: 1},
    }
    validation = validate_dataset_splits(config)
    assert validation["validated"] is True
    assert validation["case_counts"] == {"train": 2, "val": 1}


def test_external_only_validation_stays_tolerant(tmp_path: Path) -> None:
    """An evaluation-only tree may carry an unlabelled volume; that is not an error."""
    _populate(tmp_path / "testing", ["A001_t00", "B002_t00"])
    (tmp_path / "testing" / "masks" / "B002_t00.npy").unlink()

    validation = validate_dataset_splits(
        {
            "dataset": "mnms",
            "data_root": str(tmp_path),
            "test_split": "testing",
            "depth_axis": 2,
            "class_mapping": {0: 0, 1: 3, 2: 2, 3: 1},
        }
    )
    assert validation["validated"] is True


def test_alternative_raw_label_encodings_are_still_configurable(tmp_path: Path) -> None:
    """Strict pairing must not have hard-locked the raw-to-ACDC mapping."""
    _populate(tmp_path / "training", ["A001_t00"])
    _populate(tmp_path / "validation", ["C003_t00"])

    validation = validate_dataset_splits(
        {
            "dataset": "mnms",
            "data_root": str(tmp_path),
            "train_split": "train",
            "val_split": "val",
            "test_split": None,
            "depth_axis": 2,
            # A deliberate non-canonical encoding, not the default remap.
            "class_mapping": {0: 0, 1: 2, 2: 1, 3: 3},
        }
    )
    assert validation["raw_to_acdc"] == {0: 0, 1: 2, 2: 1, 3: 3}


# ---------------------------------------------------------------------------
# Per-invocation run-config isolation in the native runner
# ---------------------------------------------------------------------------


CONFIG_DIR_RE = re.compile(r"^Run configs\s+: (\S+)", re.MULTILINE)
# Both invocations deliberately share one STAMP: only mktemp, not the
# timestamp, can keep their generated configs apart.
SHARED_STAMP = "sharedstamp"


@pytest.fixture
def isolated_repo(tmp_path: Path) -> Path:
    """A throwaway repository root holding only what the runner touches.

    The runner resolves its root from its own location and writes ``runs/``,
    ``logs/`` and ``run_configs/`` there, so running it from a copy keeps the
    developer's workspace untouched: nothing under the real repository is
    created, rewritten or removed by these tests.
    """

    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "configs").mkdir()
    shutil.copy2(RUNNER, repo / "scripts" / RUNNER.name)
    for name in PROFILE_CONFIGS:
        shutil.copy2(ROOT / "configs" / name, repo / "configs" / name)
    # The generation block imports the real strict loader from ``<root>/src``.
    (repo / "src").symlink_to(ROOT / "src")
    return repo


def _write_python_shim(
    bin_dir: Path,
    log_path: Path,
    runner: Path,
    nested: dict[str, str] | None,
) -> Path:
    """A fake ``python`` that runs the real one for config generation only.

    The runner's training command is never executed: its arguments are appended
    to ``log_path`` and the shim exits 0.  When ``nested`` is given, the ACDC
    training call first runs a whole second invocation of the runner to
    completion, which is exactly the interleaving that used to corrupt the first
    invocation's pending M&Ms leg.
    """

    bin_dir.mkdir(parents=True, exist_ok=True)
    nested_block = ""
    if nested is not None:
        assignments = "\n".join(f'  export {k}="{v}"' for k, v in nested.items())
        nested_block = f"""
if [[ "$is_training" == "1" && "$config_arg" == *acdc* && -z "${{SELF_AUDIT_TEST_NESTED:-}}" ]]; then
  (
    export SELF_AUDIT_TEST_NESTED=1
{assignments}
    bash "{runner}"
  )
fi
"""
    shim = bin_dir / "python"
    shim.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
is_training=0
config_arg=""
take_next=0
for arg in "$@"; do
  if [[ "$take_next" == "1" ]]; then config_arg="$arg"; take_next=0; fi
  if [[ "$arg" == "--config" ]]; then take_next=1; fi
  if [[ "$arg" == *train_self_audit.py ]]; then is_training=1; fi
done
if [[ "$is_training" != "1" ]]; then
  exec "{sys.executable}" "$@"
fi
{nested_block}
printf '%s\\n' "$*" >> "{log_path}"
exit 0
""",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


def _invoke_runner(
    repo: Path, env_extra: dict[str, str], bin_dir: Path
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CUDA_VISIBLE_DEVICES": "",
        **env_extra,
    }
    return subprocess.run(
        ["bash", str(repo / "scripts" / RUNNER.name)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _workspace_snapshot() -> dict[str, list[str]]:
    """What the developer's own run_configs/ and logs/ contain right now."""
    return {
        name: sorted(p.name for p in (ROOT / name).iterdir())
        for name in ("run_configs", "logs")
        if (ROOT / name).is_dir()
    }


def _trainer_calls(log: Path) -> list[str]:
    return [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_concurrent_invocations_cannot_rewrite_a_pending_mnms_leg(
    tmp_path: Path, isolated_repo: Path
) -> None:
    before = _workspace_snapshot()
    log = tmp_path / "trainer_calls.log"
    bin_dir = tmp_path / "bin"
    _write_python_shim(
        bin_dir,
        log,
        isolated_repo / "scripts" / RUNNER.name,
        {
            "BATCH_SIZE": "2",
            # Same stamp as the outer invocation on purpose.
            "STAMP": SHARED_STAMP,
            "ACDC_RUN": "nested_acdc",
            "MNMS_RUN": "nested_mnms",
        },
    )

    proc = _invoke_runner(
        isolated_repo,
        {
            "BATCH_SIZE": "8",
            "STAMP": SHARED_STAMP,
            "ACDC_RUN": "outer_acdc",
            "MNMS_RUN": "outer_mnms",
        },
        bin_dir,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    reported = CONFIG_DIR_RE.findall(proc.stdout)
    assert len(reported) == 2, proc.stdout
    outer_rel, nested_rel = reported
    outer_dir, nested_dir = isolated_repo / outer_rel, isolated_repo / nested_rel

    # Same stamp, same curriculum, different directory: mktemp is what separates
    # them, not the timestamp.
    assert SHARED_STAMP in outer_rel and SHARED_STAMP in nested_rel
    assert outer_rel != nested_rel
    assert outer_dir.is_dir() and nested_dir.is_dir()
    assert outer_dir.parent == isolated_repo / "run_configs"

    calls = _trainer_calls(log)
    assert len(calls) == 4, calls

    outer_mnms_call = next(call for call in calls if f"{outer_rel}/mnms" in call)
    outer_config = yaml.safe_load(
        (isolated_repo / re.search(r"--config (\S+)", outer_mnms_call).group(1)).read_text(
            encoding="utf-8"
        )
    )
    # The decisive assertion: the pending M&Ms leg still saw ITS OWN batch size,
    # not the one the interleaved invocation asked for.
    assert {i["batch_size"] for i in outer_config["training"]["schedule"]["intervals"]} == {8}

    nested_mnms_call = next(call for call in calls if f"{nested_rel}/mnms" in call)
    nested_config = yaml.safe_load(
        (isolated_repo / re.search(r"--config (\S+)", nested_mnms_call).group(1)).read_text(
            encoding="utf-8"
        )
    )
    assert {i["batch_size"] for i in nested_config["training"]["schedule"]["intervals"]} == {2}

    # Both invocations kept the curriculum and network they asked for.
    for generated in (outer_config, nested_config):
        assert generated["model"]["window_mode"] == "candidate_c"
        assert generated["dataset"]["name"] == "mnms"
        assert generated["dataset"]["class_mapping"] == {0: 0, 1: 3, 2: 2, 3: 1}
        intervals = generated["training"]["schedule"]["intervals"]
        assert len(intervals) == 1
        assert intervals[0]["trainable"] == "all"
        assert intervals[0]["rollout"] == "threshold_gate"

    # Generated configs are provenance: kept on disk and not writable in place.
    for directory in (outer_dir, nested_dir):
        assert sorted(p.name for p in directory.iterdir()) == [
            "acdc_candidate_c_joint_from_start.yaml",
            "mnms_candidate_c_joint_from_start.yaml",
        ]
        for generated_path in directory.iterdir():
            assert not stat.S_IMODE(generated_path.stat().st_mode) & 0o222

    assert _workspace_snapshot() == before


def test_each_leg_keeps_its_own_data_root_and_output_identity(
    tmp_path: Path, isolated_repo: Path
) -> None:
    before = _workspace_snapshot()
    log = tmp_path / "trainer_calls.log"
    bin_dir = tmp_path / "bin"
    _write_python_shim(bin_dir, log, isolated_repo / "scripts" / RUNNER.name, None)

    proc = _invoke_runner(
        isolated_repo,
        {
            "BATCH_SIZE": "8",
            "STAMP": SHARED_STAMP,
            "ACDC_RUN": "iso_acdc",
            "MNMS_RUN": "iso_mnms",
        },
        bin_dir,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    calls = _trainer_calls(log)
    assert len(calls) == 2, calls
    acdc_call, mnms_call = calls

    assert "--data_root preprocessed_data/ACDC/training" in acdc_call
    assert "--output_dir runs/iso_acdc/weights" in acdc_call
    assert "--report_dir runs/iso_acdc/reports" in acdc_call
    assert "--wandb_project self-audit-acdc-candidate-c" in acdc_call

    assert "--data_root preprocessed_data/mnm" in mnms_call
    assert "--data_root preprocessed_data/mnm/" not in mnms_call
    assert "--output_dir runs/iso_mnms/weights" in mnms_call
    assert "--report_dir runs/iso_mnms/reports" in mnms_call
    assert "--wandb_project self-audit-mnms-candidate-c" in mnms_call

    # Neither leg resumes, and neither is handed the other's config.
    assert "--resume" not in acdc_call and "--resume" not in mnms_call
    assert "mnms_candidate_c" not in acdc_call
    assert "acdc_candidate_c" not in mnms_call

    assert _workspace_snapshot() == before


def test_generated_run_config_directory_cannot_be_pointed_at_an_existing_path(
    tmp_path: Path, isolated_repo: Path
) -> None:
    """There is no env override that could make two invocations share a directory."""
    before = _workspace_snapshot()
    log = tmp_path / "trainer_calls.log"
    bin_dir = tmp_path / "bin"
    _write_python_shim(bin_dir, log, isolated_repo / "scripts" / RUNNER.name, None)

    hijack = isolated_repo / "run_configs" / "hijacked"
    hijack.mkdir(parents=True)

    proc = _invoke_runner(
        isolated_repo,
        {
            "BATCH_SIZE": "8",
            "STAMP": SHARED_STAMP,
            "ACDC_RUN": "hijack_acdc",
            "MNMS_RUN": "hijack_mnms",
            "RUN_CONFIG_DIR": str(hijack),
        },
        bin_dir,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    reported = CONFIG_DIR_RE.findall(proc.stdout)
    assert len(reported) == 1
    assert reported[0] != str(hijack)
    assert list(hijack.iterdir()) == []
    # The value is always allocated, never taken from the environment.
    runner_text = RUNNER.read_text(encoding="utf-8")
    assert 'RUN_CONFIG_DIR="$(mktemp -d "run_configs/' in runner_text
    assert "${RUN_CONFIG_DIR:-" not in runner_text

    assert _workspace_snapshot() == before
