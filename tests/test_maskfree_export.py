"""Focused checks for the mask-free export, freeze and launch layer (W7).

Three checks, matching the W7 acceptance criteria:

1. asymmetric native volume roundtrip - stored-grid units assemble into a native
   volume through the explicit reverse shape/axis transform, unexported slices
   stay unresolved rather than becoming background, and a source without
   verifiable inverse geometry degrades to an honestly labelled stored-grid
   export instead of a fake native one;
2. the freeze manifest is immutable, tamper-evident, refuses divergent
   identities, and reports its own comparison completeness;
3. the launchers are syntactically valid bash with a real ``--help`` and a
   no-execution default, the checked-in configs contain only W5 schema fields,
   and the experiment manifest enumerates exactly ``REQUIRED_COMPARISONS``.

All fixtures are CPU synthetic arrays. They are software evidence about this
module's arithmetic and file handling only. No GPU was used, no real cardiac
data was read, and nothing here tests whether any produced label is anatomically
correct.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
# Canonical import path: `self_audit_maskfree` with PYTHONPATH=.:src. Importing
# it as `src.self_audit_maskfree` as well would duplicate the contract
# dataclasses under two module identities.
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import self_audit_maskfree.export as export_module  # noqa: E402
from self_audit_maskfree.export import (  # noqa: E402
    REQUIRED_COMPARISONS,
    ExportError,
    FreezeError,
    assemble_volume,
    export_prediction,
    freeze_predictions,
    normalize_record,
    resolve_native_target,
    validate_freeze,
    verified_freeze_session,
)

STORED_HW = (16, 16)
NATIVE_SLICE_HW = (37, 53)   # deliberately asymmetric and not the stored size
NUM_SLICES = 5


def _affine() -> list[list[float]]:
    return [[1.25, 0.0, 0.0, -80.0],
            [0.0, 1.4, 0.0, -90.0],
            [0.0, 0.0, 9.0, -30.0],
            [0.0, 0.0, 0.0, 1.0]]


def _record(slice_index: int, *, source_format: str = "nifti",
            geometry_available: bool = True,
            stored_to_native_axes: tuple[int, int, int] = (1, 2, 0)) -> dict:
    """One W3-shaped record.

    Stored order is ``(Z, H, W) = (5, 37, 53)``. With ``stored_to_native_axes =
    (1, 2, 0)`` the native array order is ``(37, 53, 5)`` and the depth axis is
    native axis 2 - an asymmetric, non-identity transform, which is the case a
    silent resize would get wrong without anyone noticing.
    """
    stored_shape = (NUM_SLICES, NATIVE_SLICE_HW[0], NATIVE_SLICE_HW[1])
    native_shape = tuple(stored_shape[a] for a in stored_to_native_axes)
    depth_axis = list(stored_to_native_axes).index(0)
    return {
        "study_id": "patient001_frame01",
        "patient_id": "patient001",
        "unit_id": f"patient001_frame01_z{slice_index:02d}",
        "dataset": "acdc",
        "split": "train",
        "protocol": "spatial_predictive",
        "slice_index": slice_index,
        "num_slices": NUM_SLICES,
        "source_format": source_format,
        "source_path": "/workspace/images/patient001_frame01.nii.gz",
        "native_shape": list(native_shape),
        "native_slice_shape": list(NATIVE_SLICE_HW),
        "depth_axis": depth_axis,
        "stored_to_native_axes": list(stored_to_native_axes),
        "native_geometry": {
            "available": geometry_available,
            "affine": _affine() if geometry_available else None,
            "spacing": [1.25, 1.4, 9.0],
            "spacing_mm": [1.25, 1.4, 9.0],
            "spacing_valid": True,
            "spatial_unit": "mm",
            "xyzt_units": {"space": "mm", "time": "unknown"},
            "reason": None if geometry_available else "source array has no affine sidecar",
        },
        "partition_id": "partition_abc123",
        "manifest_id": "manifest_def456",
    }


def _unit_arrays(slice_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(100 + slice_index)
    labels = torch.zeros(STORED_HW, dtype=torch.long)
    labels[4:12, 4:12] = 2          # MYO ring
    labels[6:10, 6:10] = 3          # LV cavity inside it
    labels[2:5, 10:14] = 1          # RV
    probabilities = torch.nn.functional.one_hot(labels, 4).permute(2, 0, 1).float()
    probabilities = probabilities * 0.9 + 0.025
    validity = torch.rand(STORED_HW, generator=generator).clamp(0.2, 1.0)
    return labels, probabilities, validity


def _export_units(root: Path, *, prediction_name: str, indices=range(NUM_SLICES),
                  **record_kwargs) -> list[dict]:
    shards = []
    for index in indices:
        labels, probabilities, validity = _unit_arrays(index)
        shards.append(export_prediction(
            root,
            record=_record(index, **record_kwargs),
            prediction_name=prediction_name,
            labels=labels,
            probabilities=probabilities,
            validity=validity,
            alternatives=[{"role_permutation": [0, 2, 1, 3], "semantic_unresolved": True}],
            checkpoint_id="producer_final@epoch149",
            version="pseudolabels.v3",
        ))
    return shards


# --------------------------------------------------------------------------- #
# check 1: the reverse geometry transform, and its refusal to fake one
# --------------------------------------------------------------------------- #
def test_asymmetric_native_roundtrip_and_stored_grid_fallback(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _export_units(root, prediction_name="student_audited")
    volume = assemble_volume(root, prediction_name="student_audited",
                             study_id="patient001_frame01", write_nifti=False)

    # The volume lands on the native grid, in native axis order, not stored order.
    assert volume["grid"] == "native"
    assert volume["native_export_available"] is True
    assert volume["volume_shape"] == [NATIVE_SLICE_HW[0], NATIVE_SLICE_HW[1], NUM_SLICES]
    assert volume["probabilities_shape"] == [4, NATIVE_SLICE_HW[0], NATIVE_SLICE_HW[1],
                                             NUM_SLICES]
    assert volume["label_interpolation"] == "nearest"
    assert volume["probability_interpolation"] == "bilinear+renormalized"
    assert volume["complete_volume"] is True
    assert volume["missing_slice_indices"] == []

    labels = np.load(root / "predictions/student_audited/volumes/patient001_frame01/labels.npy")
    probabilities = np.load(
        root / "predictions/student_audited/volumes/patient001_frame01/probabilities.npy")
    validity = np.load(
        root / "predictions/student_audited/volumes/patient001_frame01/validity.npy")

    assert labels.shape == (NATIVE_SLICE_HW[0], NATIVE_SLICE_HW[1], NUM_SLICES)
    # Nearest label interpolation may not invent a class outside the fixed order.
    assert set(np.unique(labels)).issubset({0, 1, 2, 3})
    # All four named roles survive the asymmetric resize; a silent argmax over a
    # blurred probability stack would be free to lose the thin MYO ring.
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    assert np.allclose(probabilities.sum(axis=0), 1.0, atol=1e-5)
    assert validity.min() >= 0.0 and validity.max() <= 1.0

    # With verified inverse geometry the volume is written as a native-grid
    # NIfTI carrying the record's affine, not an unlabelled array.
    nib = pytest.importorskip("nibabel")
    nifti_root = tmp_path / "nifti"
    _export_units(nifti_root, prediction_name="student_audited")
    nifti_volume = assemble_volume(nifti_root, prediction_name="student_audited",
                                   study_id="patient001_frame01", write_nifti=True)
    assert nifti_volume["grid"] == "native"
    image = nib.load(str(nifti_root / "predictions/student_audited/volumes/"
                         "patient001_frame01/labels.nii.gz"))
    assert image.shape == (NATIVE_SLICE_HW[0], NATIVE_SLICE_HW[1], NUM_SLICES)
    assert np.allclose(image.affine, np.asarray(_affine()))
    assert image.header.get_xyzt_units()[0] == "mm"
    volume_sidecar = json.loads((nifti_root / "predictions/student_audited/volumes/"
                                 "patient001_frame01/volume.json").read_text())
    assert volume_sidecar["spacing_mm"] == [1.25, 1.4, 9.0]
    assert volume_sidecar["spacing_valid"] is True

    # A slice no unit produced stays unresolved: validity 0, and it is named.
    gap_root = tmp_path / "gap"
    _export_units(gap_root, prediction_name="student_audited", indices=[0, 1, 3, 4])
    gap = assemble_volume(gap_root, prediction_name="student_audited",
                          study_id="patient001_frame01", write_nifti=False)
    assert gap["missing_slice_indices"] == [2]
    assert gap["complete_volume"] is False
    gap_validity = np.load(
        gap_root / "predictions/student_audited/volumes/patient001_frame01/validity.npy")
    assert float(gap_validity[:, :, 2].max()) == 0.0
    assert float(gap_validity[:, :, 0].max()) > 0.0

    # An .npy source has no native affine. It must export stored-grid, and must
    # say why, rather than being relabelled native because the numbers fit.
    npy_root = tmp_path / "npy"
    _export_units(npy_root, prediction_name="E0_intensity_grouping", source_format="npy")
    npy_volume = assemble_volume(npy_root, prediction_name="E0_intensity_grouping",
                                 study_id="patient001_frame01", write_nifti=False)
    assert npy_volume["grid"] == "stored"
    assert npy_volume["native_export_available"] is False
    assert "no native affine" in npy_volume["native_export_reason"]
    assert npy_volume["volume_shape"] == [NUM_SLICES, *STORED_HW]
    assert npy_volume["affine"] is None
    assert not (npy_root / "predictions/E0_intensity_grouping/volumes/"
                "patient001_frame01/labels.nii.gz").exists()

    # An inconsistent claimed inverse transform is caught arithmetically.
    bad = _record(0, stored_to_native_axes=(0, 1, 2))
    bad["native_shape"] = [NUM_SLICES, NATIVE_SLICE_HW[0], NATIVE_SLICE_HW[1]]
    bad["depth_axis"] = 1  # contradicts the permutation: depth is axis 0 here
    decision = resolve_native_target(bad)
    assert decision["native_export_available"] is False
    assert "depth_axis" in decision["reason"]

    # Re-exporting the same unit with different content is refused outright.
    labels, probabilities, validity = _unit_arrays(0)
    with pytest.raises(ExportError, match="different content"):
        export_prediction(root, record=_record(0), prediction_name="student_audited",
                          labels=labels, probabilities=probabilities,
                          validity=validity * 0.5, alternatives=None,
                          checkpoint_id="producer_final@epoch149", version="pseudolabels.v3")

    # The unit boundary is strict: nearest interpolation is reserved for the
    # explicitly recorded native assembly step. Fractional labels and
    # unnormalised probabilities cannot be silently coerced here.
    with pytest.raises(ExportError, match="integer dtype"):
        export_prediction(root / "fractional", record=_record(0),
                          prediction_name="student_audited", labels=labels.float(),
                          probabilities=probabilities, validity=validity,
                          alternatives=None, checkpoint_id="producer_final@epoch149",
                          version="pseudolabels.v3")
    with pytest.raises(ExportError, match="sum to 1"):
        export_prediction(root / "unnormalized", record=_record(0),
                          prediction_name="student_audited", labels=labels,
                          probabilities=probabilities * 0.5, validity=validity,
                          alternatives=None, checkpoint_id="producer_final@epoch149",
                          version="pseudolabels.v3")


# --------------------------------------------------------------------------- #
# check 2: the freeze is immutable, tamper-evident and honest about its scope
# --------------------------------------------------------------------------- #
def test_freeze_manifest_is_immutable_and_tamper_evident(tmp_path: Path) -> None:
    root = tmp_path / "run"
    checkpoint = tmp_path / "producer_final.pt"
    checkpoint.write_bytes(b"not a real checkpoint, only bytes to hash")

    # The trainer hands freeze_predictions the per-unit shards it streamed, so
    # this is the real call shape: assembly happens inside the freeze.
    entries: list[dict] = []
    for name in ("student_audited", "student_no_audit"):
        entries.extend(_export_units(root, prediction_name=name))

    manifest = freeze_predictions(root, entries, {"producer_final": checkpoint},
                                  dataset="acdc", protocol="spatial_predictive", epoch=149)
    validate_freeze(root / "freeze_manifest.json")

    kinds = {p["kind"] for p in manifest["predictions"]}
    assert kinds == {"unit", "volume"}
    volumes = [p for p in manifest["predictions"] if p["kind"] == "volume"]
    assert len(volumes) == 2
    assert all(v["grid"] == "native" for v in volumes)
    assert all(v["complete_volume"] for v in volumes)
    # Every frozen unit shard and every assembled volume file is hashed.
    assert all(f["sha256"] for p in manifest["predictions"] for f in p["files"])

    # A partial comparison set is recorded as incomplete, not accepted as full.
    assert manifest["completeness"]["complete"] is False
    assert "E5_evidence_plus_challenge" in manifest["completeness"]["missing"]
    assert manifest["compared_methods"] == ["student_audited", "student_no_audit"]
    assert manifest["lineage"]["version"] == "pseudolabels.v3"

    # Re-freezing identical content is idempotent and returns the same identity.
    again = freeze_predictions(root, entries, {"producer_final": checkpoint},
                               dataset="acdc", protocol="spatial_predictive", epoch=149)
    assert again["freeze_id"] == manifest["freeze_id"]

    # A changed frozen prediction file is caught by hash.
    victim = root / entries[0]["files"][0]
    original = victim.read_bytes()
    victim.write_bytes(original + b"\x00")
    with pytest.raises(FreezeError, match="content changed|size changed"):
        validate_freeze(root / "freeze_manifest.json")
    victim.write_bytes(original)
    validate_freeze(root / "freeze_manifest.json")

    # A changed checkpoint is caught too: a frozen comparison names exact weights.
    checkpoint.write_bytes(b"different weights entirely")
    with pytest.raises(FreezeError, match="checkpoint changed"):
        validate_freeze(root / "freeze_manifest.json")
    checkpoint.write_bytes(b"not a real checkpoint, only bytes to hash")

    # An edited manifest body no longer matches its own manifest_id.
    payload = json.loads((root / "freeze_manifest.json").read_text())
    payload["epoch"] = 120
    (root / "tampered.json").write_text(json.dumps(payload))
    payload["root"] = str(root)
    with pytest.raises(FreezeError, match="body was modified"):
        validate_freeze(payload)

    # A second, different freeze must not overwrite the first.
    _export_units(root, prediction_name="E4_selection_evidence")
    entries.append(assemble_volume(root, prediction_name="E4_selection_evidence",
                                   study_id="patient001_frame01", write_nifti=False))
    with pytest.raises(FreezeError, match="refusing to overwrite"):
        freeze_predictions(root, entries, {"producer_final": checkpoint},
                           dataset="acdc", protocol="spatial_predictive", epoch=149)

    # Mixed label versions or protocols are not one comparison.
    divergent = dict(entries[0])
    divergent["version"] = "pseudolabels.v4"
    with pytest.raises(FreezeError, match="disagree on version"):
        freeze_predictions(tmp_path / "other", [entries[0], divergent],
                           {"producer_final": checkpoint},
                           dataset="acdc", protocol="spatial_predictive", epoch=149)

    # An empty comparison set is refused rather than frozen as a valid nothing.
    with pytest.raises(FreezeError, match="empty comparison set"):
        freeze_predictions(tmp_path / "empty", [], None,
                           dataset="acdc", protocol="spatial_predictive")


def test_freeze_assembles_by_volume_id_and_strict_artifact_completeness(tmp_path: Path) -> None:
    root = tmp_path / "run"
    entries: list[dict] = []
    expected_units: list[str] = []
    for volume_id in ("acquisition_a", "acquisition_b"):
        for index in range(NUM_SLICES):
            labels, probabilities, validity = _unit_arrays(index)
            record = _record(index)
            record["volume_id"] = volume_id
            record["acquisition_id"] = volume_id
            record["unit_id"] = f"{volume_id}_z{index:02d}"
            expected_units.append(record["unit_id"])
            entries.append(export_prediction(
                root, record=record, prediction_name="student_audited",
                labels=labels, probabilities=probabilities, validity=validity,
                alternatives=[], checkpoint_id="student_audited@epoch149",
                version="pseudolabels.v3"))
    checkpoint = tmp_path / "student_audited.pt"
    checkpoint.write_bytes(b"weights")
    manifest = freeze_predictions(
        root, entries, {"student_audited": checkpoint}, dataset="acdc",
        protocol="spatial_predictive", epoch=149,
        required_methods=("student_audited",), required_unit_ids=expected_units)
    validate_freeze(manifest, require_complete=True)
    volumes = [item for item in manifest["predictions"] if item["kind"] == "volume"]
    assert {item["volume_id"] for item in volumes} == {"acquisition_a", "acquisition_b"}
    assert manifest["completeness"]["units"]["complete"] is True


def test_verified_freeze_session_hashes_once_per_phase_and_rejects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    checkpoint = tmp_path / "producer.pt"
    checkpoint.write_bytes(b"checkpoint")
    entries = _export_units(root, prediction_name="student_audited")
    manifest = freeze_predictions(
        root, entries, {"producer": checkpoint}, dataset="acdc",
        protocol="spatial_predictive", epoch=149,
        required_methods=("student_audited",),
    )

    calls: list[Path] = []
    real_sha256 = export_module._sha256

    def counting_sha256(path: Path) -> str:
        calls.append(path)
        return real_sha256(path)

    monkeypatch.setattr(export_module, "_sha256", counting_sha256)
    with verified_freeze_session(manifest) as exact_manifest:
        entry_calls = len(calls)
        validate_freeze(exact_manifest)
        # The authoritative identity/completeness checks still run, but the
        # exact object admitted by the session does not rehash its corpus.
        assert len(calls) == entry_calls
    # Exit is a second complete physical pass that catches in-session changes.
    assert len(calls) > entry_calls

    copied = dict(manifest)
    before_copy_validation = len(calls)
    validate_freeze(copied)
    assert len(calls) > before_copy_validation

    with pytest.raises(FreezeError, match="body was modified|active verified"):
        with verified_freeze_session(manifest) as exact_manifest:
            exact_manifest["epoch"] = 148


# --------------------------------------------------------------------------- #
# check 3: launchers and configs
# --------------------------------------------------------------------------- #
MASKFREE_CONFIG_FIELDS = {
    "dataset", "data_root", "output_dir", "total_epochs", "seed", "batch_size",
    "accumulation_steps", "image_size", "lr", "weight_decay", "warmup_epochs",
    "width", "feature_dim", "protocol", "depth_axis", "device", "audit_device", "num_workers",
    "amp", "wandb_mode", "wandb_project", "run_id", "max_steps", "max_epochs",
    "resume", "allow_cpu", "epoch_validation", "epoch_reference_config",
}

FORBIDDEN_CONFIG_SUBSTRINGS = (
    "pretrained", "mask_root", "label_root", "gt_", "ground_truth", "annotation",
    "checkpoint_path", "teacher", "sam_", "medsam",
)


@pytest.mark.parametrize("script", ["scripts/run_maskfree_full.sh",
                                    "scripts/run_maskfree_acdc_mnms.sh"])
def test_launcher_syntax_help_and_no_execution_default(script: str) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable on this host")
    path = REPO_ROOT / script

    syntax = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr

    helped = subprocess.run([bash, str(path), "--help"], capture_output=True, text=True,
                            cwd=REPO_ROOT)
    assert helped.returncode == 0, helped.stderr
    assert "USAGE" in helped.stdout
    assert "RUN_FULL" in helped.stdout

    unknown = subprocess.run([bash, str(path), "--not-a-flag"], capture_output=True,
                             text=True, cwd=REPO_ROOT)
    assert unknown.returncode != 0
    assert "unknown argument" in unknown.stderr


def test_launcher_writes_a_read_only_resolved_config_through_w5(tmp_path: Path) -> None:
    """The launcher's stage 1b, actually executed - no GPU, no real data.

    RUN_FULL=1 with ALLOW_CPU=1 and an empty data root: the resolved config must
    be written read-only through W5's loader and must round-trip through that
    same loader, and the run must then STOP at the inventory stage rather than
    proceeding on an unusable dataset.
    """
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable on this host")
    pytest.importorskip("self_audit_maskfree.config",
                        reason="W5 config module has not landed in this checkout")
    from self_audit_maskfree.config import load_config

    workspace = tmp_path / "workspace"
    (workspace / "empty_data_root").mkdir(parents=True)
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(tmp_path),
        "WORKSPACE": str(workspace),
        "RUN_FULL": "1",
        "ALLOW_CPU": "1",
        "RUN_ID": "pytest_stage1b",
        "DATA_ROOT": str(workspace / "empty_data_root"),
        "PYTHON": sys.executable,
    }
    result = subprocess.run([bash, "scripts/run_maskfree_full.sh", "acdc"],
                            capture_output=True, text=True, cwd=REPO_ROOT, env=env)

    resolved = workspace / "run_configs/maskfree150/pytest_stage1b/acdc.yaml"
    assert resolved.is_file(), result.stdout + result.stderr
    assert resolved.stat().st_mode & 0o222 == 0, "the resolved config must be read-only"

    config = load_config(resolved)
    assert config.run_id == "pytest_stage1b"
    assert config.output_dir == str(workspace)   # workspace root, not the run dir
    assert config.total_epochs == 150
    assert config.batch_size == 8 and config.accumulation_steps == 1
    assert config.device == "cpu" and config.allow_cpu is True

    # An empty data root is not a run: the sequence stops at inventory and never
    # reaches preflight or training.
    assert result.returncode != 0
    assert "stage 4" not in result.stdout.split("stage 1b")[-1]
    assert not (workspace / "runs/maskfree150/acdc/pytest_stage1b/checkpoints").exists()


def test_configs_match_w5_schema_and_experiment_manifest(tmp_path: Path) -> None:
    for dataset in ("acdc", "mnms"):
        path = REPO_ROOT / f"configs/maskfree_{dataset}_150.yaml"
        payload = yaml.safe_load(path.read_text())
        unknown = set(payload) - MASKFREE_CONFIG_FIELDS
        assert not unknown, f"{path.name} carries keys W5's strict loader rejects: {unknown}"
        assert payload["dataset"] == dataset
        assert payload["total_epochs"] == 150
        assert payload["batch_size"] == 8
        assert payload["accumulation_steps"] == 1
        assert payload["image_size"] == 224
        assert payload["seed"] == 42
        assert payload["allow_cpu"] is False
        text = path.read_text().lower()
        for forbidden in FORBIDDEN_CONFIG_SUBSTRINGS:
            assert forbidden not in text.replace("no pretrained", "").replace(
                "no reference/annotation", "").replace("gt-driven", ""), \
                f"{path.name} mentions a supervision key: {forbidden}"

    # The launcher writes the resolved config through W5's loader; if that module
    # has landed, prove the checked-in template actually parses under it.
    try:
        from self_audit_maskfree.config import load_config
    except ImportError:
        pytest.xfail("self_audit_maskfree/config.py (W5) has not landed in this checkout; "
                     "the launcher's resolved-config stage cannot be executed yet")
    else:
        config = load_config(REPO_ROOT / "configs/maskfree_acdc_150.yaml")
        assert config.dataset == "acdc"
        assert config.total_epochs == 150


def test_experiment_manifest_enumerates_required_comparisons() -> None:
    manifest = yaml.safe_load((REPO_ROOT / "configs/maskfree_experiments.yaml").read_text())
    names = [entry["name"] for entry in manifest["comparisons"]]
    assert tuple(names) == REQUIRED_COMPARISONS, (
        "configs/maskfree_experiments.yaml and export.REQUIRED_COMPARISONS have drifted; "
        "a freeze would then report completeness against a different method set than the "
        "one the experiment manifest declares")
    assert manifest["coverage_levels"] == [25, 50, 75, 100]
    assert manifest["semantic_order"] == {0: "BG", 1: "RV", 2: "MYO", 3: "LV"}
