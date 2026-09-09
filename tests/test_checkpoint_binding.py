"""Regression tests for Wave 3.1: producer-side checkpoint state binding.

The failure these guard against is subtle: the runner used to collect its
validation transition cache from the live last-epoch model and then hash
``phase_c_best.pt`` for the artifact, so the artifact could name weights that
were never measured.  Hashing a file beside a live model is not binding.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from scripts.train_self_audit_legacy import (
    bind_post_training_checkpoint,
    run_post_training_calibration,
)
from src.self_audit.data.common import VolumeRecord, VolumeSliceDataset
from src.self_audit.models.self_audit_net import SelfAuditNet
from src.self_audit.provenance import (
    MEMBERSHIP_COVERAGE,
    UNKNOWN,
    _unwrap_dataset,
    build_lineage,
    checkpoint_producer,
    cohort_identity,
    git_source_provenance,
    preprocessing_descriptor,
    resolve_model_identity,
    state_digest,
    verify_model_config,
)
from src.self_audit.training._utils import (
    bind_evaluation_checkpoint,
    load_checkpoint,
    save_checkpoint,
    verify_bound_state,
)

TINY_MODEL_CONFIG: dict[str, Any] = {
    "encoder_name": "convnext_tiny",
    "shared_channels": 16,
    "num_classes": 4,
    "window_k": 4,
    "max_turns": 2,
}


def _tiny_net(seed: int) -> SelfAuditNet:
    torch.manual_seed(seed)
    return SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=2,
    ).eval()


class _BufferModule(nn.Module):
    """Exercises the dtype corners a NumPy-based digest would choke on."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 3))
        self.register_buffer("steps", torch.tensor(7))
        self.register_buffer("bf16", torch.zeros(4, dtype=torch.bfloat16))
        self.register_buffer("empty", torch.zeros(0))


class _SyntheticDataset(Dataset):
    def __init__(self, count: int = 4, size: int = 32) -> None:
        torch.manual_seed(11)
        self.images = torch.randn(count, 3, size, size)
        self.masks = torch.randint(0, 4, (count, size, size))
        self.case_ids = [f"case_{index // 2:03d}" for index in range(count)]

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self.images[index],
            "mask": self.masks[index],
            "case_id": self.case_ids[index],
        }


# ---------------------------------------------------------------------------
# state_digest
# ---------------------------------------------------------------------------


def test_state_digest_covers_buffers_and_exotic_dtypes() -> None:
    module = _BufferModule()
    baseline = state_digest(module)
    assert baseline == state_digest(module.state_dict())

    module.steps += 1
    assert state_digest(module) != baseline, "scalar integer buffer must be covered"

    module.steps -= 1
    assert state_digest(module) == baseline

    module.bf16[0] = torch.tensor(1.0, dtype=torch.bfloat16)
    assert state_digest(module) != baseline, "bfloat16 buffer must be covered"


def test_state_digest_separates_shape_from_content() -> None:
    flat = {"w": torch.zeros(6)}
    square = {"w": torch.zeros(2, 3)}
    assert state_digest(flat) != state_digest(square)

    float32 = {"w": torch.zeros(4, dtype=torch.float32)}
    float64 = {"w": torch.zeros(4, dtype=torch.float64)}
    assert state_digest(float32) != state_digest(float64)

    renamed = {"other": torch.zeros(6)}
    assert state_digest(flat) != state_digest(renamed)


# ---------------------------------------------------------------------------
# Producer provenance
# ---------------------------------------------------------------------------


def test_saved_checkpoint_records_producer_and_survives_weights_only(tmp_path: Path) -> None:
    net = _tiny_net(3)
    path = save_checkpoint(tmp_path / "phase_c_best.pt", net, epoch=2, config={"model": TINY_MODEL_CONFIG})

    payload = torch.load(path, map_location="cpu", weights_only=True)
    provenance = payload["provenance"]
    assert provenance["provenance_schema_version"] == 1
    assert provenance["state_digest"] == state_digest(net)
    assert provenance["producer_git_sha"] == git_source_provenance()["git_sha"]
    assert provenance["model_identity"]["window_k"] == 4
    assert provenance["model_identity"]["entropy_version"] == net.annotation_expert.entropy_version


def test_legacy_checkpoint_producer_stays_unknown(tmp_path: Path) -> None:
    net = _tiny_net(4)
    legacy = tmp_path / "legacy.pt"
    torch.save({"model": {k: v.clone() for k, v in net.state_dict().items()}, "epoch": 1}, legacy)

    producer = checkpoint_producer(torch.load(legacy, map_location="cpu", weights_only=True))
    assert producer["producer_recorded"] is False
    assert producer["producer_git_sha"] == UNKNOWN
    assert producer["producer_git_sha"] != git_source_provenance()["git_sha"]
    assert producer["producer_state_digest"] is None


def test_git_provenance_separates_source_from_report_only_dirt() -> None:
    source = git_source_provenance()
    assert set(source) == {
        "git_sha",
        "dirty",
        "source_dirty",
        "source_paths",
        "source_content_signature",
        "source_content_signature_known",
        "source_signature_version",
        "source_signature_scope",
    }
    # A report-only change must not be able to claim the model code changed:
    # the dirty scope is restricted to code/config paths, and reports/ is not
    # among them.
    assert "reports" not in source["source_paths"]
    assert "src" in source["source_paths"]
    # The content signature is the field that answers "did the code change?";
    # the commit SHA moves on report-only commits and cannot.
    assert source["source_content_signature_known"] is True
    assert len(source["source_content_signature"]) == 64
    assert "reports" in source["source_signature_scope"]["excluded_dirs"]
    assert "tests" in source["source_signature_scope"]["excluded_dirs"]


def test_resolved_identity_reads_live_model_not_filename() -> None:
    net = _tiny_net(5)
    identity = resolve_model_identity(net)
    assert identity["window_k"] == 4
    assert identity["max_turns"] == 2
    assert identity["shared_channels"] == 16
    assert identity["encoder_backend"] in {"timm", "fallback_synthetic"}
    assert identity["entropy_version"] == net.annotation_expert.entropy_version


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


def _write_best_and_last(tmp_path: Path) -> tuple[SelfAuditNet, SelfAuditNet]:
    best = _tiny_net(101)
    last = _tiny_net(202)
    assert state_digest(best) != state_digest(last), "fixture must have best != last"
    save_checkpoint(tmp_path / "phase_c_best.pt", best, epoch=3, config={"model": TINY_MODEL_CONFIG})
    save_checkpoint(tmp_path / "phase_c_last.pt", last, epoch=5, config={"model": TINY_MODEL_CONFIG})
    return best, last


def test_binding_loads_best_not_live_last(tmp_path: Path) -> None:
    best, last = _write_best_and_last(tmp_path)
    live = _tiny_net(202)
    assert state_digest(live) == state_digest(last)

    binding = bind_post_training_checkpoint(
        live,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        config={"model": TINY_MODEL_CONFIG},
    )
    assert binding.role == "best"
    assert binding.fallback_used is False
    assert binding.path == tmp_path / "phase_c_best.pt"
    assert binding.state_digest == state_digest(best)
    assert binding.state_digest == state_digest(live)
    assert binding.state_digest != state_digest(last)
    assert binding.file_state_digest == binding.state_digest
    assert binding.epoch == 3
    assert binding.restored == ("model",)


def test_last_fallback_is_declared_not_silent(tmp_path: Path) -> None:
    last = _tiny_net(303)
    save_checkpoint(tmp_path / "phase_c_last.pt", last, epoch=9, config={"model": TINY_MODEL_CONFIG})

    binding = bind_post_training_checkpoint(
        _tiny_net(404),
        output_dir=tmp_path,
        device=torch.device("cpu"),
        config={"model": TINY_MODEL_CONFIG},
    )
    assert binding.role == "last"
    assert binding.fallback_used is True
    assert binding.state_digest == state_digest(last)
    assert any("best:" in entry for entry in binding.considered)


def test_missing_both_checkpoints_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No evaluation checkpoint available"):
        bind_post_training_checkpoint(
            _tiny_net(1),
            output_dir=tmp_path,
            device=torch.device("cpu"),
            config={"model": TINY_MODEL_CONFIG},
        )


def test_config_mismatch_fails_binding(tmp_path: Path) -> None:
    _write_best_and_last(tmp_path)
    mismatched = dict(TINY_MODEL_CONFIG)
    mismatched["window_k"] = 8
    with pytest.raises(ValueError, match="does not match its configuration"):
        bind_post_training_checkpoint(
            _tiny_net(7),
            output_dir=tmp_path,
            device=torch.device("cpu"),
            config={"model": mismatched},
        )


def test_incompatible_state_dict_fails_binding(tmp_path: Path) -> None:
    wide = SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=32,
        window_k=4,
        max_turns=2,
    ).eval()
    save_checkpoint(tmp_path / "phase_c_best.pt", wide, epoch=1)

    with pytest.raises((ValueError, RuntimeError)):
        bind_post_training_checkpoint(
            _tiny_net(8),
            output_dir=tmp_path,
            device=torch.device("cpu"),
            config=None,
        )


def test_tampered_checkpoint_state_is_rejected(tmp_path: Path) -> None:
    net = _tiny_net(9)
    path = save_checkpoint(tmp_path / "phase_c_best.pt", net, epoch=1)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    first_key = sorted(payload["model"])[0]
    payload["model"][first_key] = payload["model"][first_key] + 1.0
    torch.save(payload, path)

    with pytest.raises(ValueError, match="refusing to bind a tampered checkpoint"):
        bind_post_training_checkpoint(
            _tiny_net(9),
            output_dir=tmp_path,
            device=torch.device("cpu"),
            config=None,
        )


def test_binding_does_not_restore_rng_or_optimizer(tmp_path: Path) -> None:
    torch.manual_seed(1234)
    net = _tiny_net(11)
    save_checkpoint(tmp_path / "phase_c_best.pt", net, epoch=1, config={"model": TINY_MODEL_CONFIG})

    live = _tiny_net(12)
    torch.manual_seed(999)
    before = torch.get_rng_state().clone()
    binding = bind_post_training_checkpoint(
        live,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        config={"model": TINY_MODEL_CONFIG},
    )
    after = torch.get_rng_state()
    assert torch.equal(before, after), "evaluation binding must not restore checkpoint RNG state"
    assert binding.restored == ("model",)
    assert "optimizer" not in inspect.signature(bind_evaluation_checkpoint).parameters
    assert "scheduler" not in inspect.signature(bind_evaluation_checkpoint).parameters

    # The legacy resume path keeps its RNG restore; only the evaluation bind
    # opts out, so this is a deliberate difference and not a lost feature.
    torch.manual_seed(999)
    load_checkpoint(tmp_path / "phase_c_best.pt", restore_rng=True)
    assert not torch.equal(before, torch.get_rng_state())


def test_verify_bound_state_detects_live_mutation(tmp_path: Path) -> None:
    _write_best_and_last(tmp_path)
    live = _tiny_net(13)
    binding = bind_post_training_checkpoint(
        live,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        config={"model": TINY_MODEL_CONFIG},
    )
    assert verify_bound_state(live, binding, boundary="probe") == binding.state_digest

    with torch.no_grad():
        next(iter(live.parameters())).add_(1.0)
    with pytest.raises(ValueError, match="Model state changed after checkpoint binding"):
        verify_bound_state(live, binding, boundary="probe")


# ---------------------------------------------------------------------------
# Lineage honesty
# ---------------------------------------------------------------------------


def test_cohort_identity_labels_truncated_measurement() -> None:
    loader = DataLoader(_SyntheticDataset(count=4), batch_size=1, shuffle=False)
    full = cohort_identity(loader, split_name="val", max_batches=None, batch_size=1)
    assert full["covers_full_split"] is True
    assert full["coverage"] == MEMBERSHIP_COVERAGE

    truncated = cohort_identity(loader, split_name="val", max_batches=2, batch_size=1)
    assert truncated["covers_full_split"] is False
    assert "not a full-cohort measurement" in truncated["subset_note"]


def test_lineage_keeps_producer_and_evaluation_revisions_apart(tmp_path: Path) -> None:
    _write_best_and_last(tmp_path)
    live = _tiny_net(14)
    binding = bind_post_training_checkpoint(
        live,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        config={"model": TINY_MODEL_CONFIG},
    )
    loader = DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False)
    lineage = build_lineage(
        binding=binding.as_dict(),
        loader=loader,
        split_name="val",
        metric_contract="foreground_dice_exclude_v1",
        metric_space="slice_proxy",
        neutral_margin=0.005,
        t_max=2,
        batch_size=2,
    )
    assert lineage["checkpoint"]["state_digest"] == binding.state_digest
    assert lineage["checkpoint"]["producer"]["producer_git_sha"] is not None
    assert "git_sha" in lineage["evaluation_code"]
    assert lineage["semantics"]["entropy_version"] == live.annotation_expert.entropy_version
    assert lineage["semantics"]["t_max"] == 2
    assert lineage["semantics"]["num_classes"] == 4
    assert lineage["split"]["coverage"] == MEMBERSHIP_COVERAGE
    assert lineage["preprocessing"]["signature"]


# ---------------------------------------------------------------------------
# The actual runner branch
# ---------------------------------------------------------------------------


def test_runner_post_training_branch_binds_best_at_every_consumer(tmp_path: Path, monkeypatch) -> None:
    """The collector, the calibration sweep and both diagnostics see best."""

    import scripts.train_self_audit_legacy as runner

    output_dir = tmp_path / "weights"
    report_dir = tmp_path / "reports"
    output_dir.mkdir()
    report_dir.mkdir()
    best, last = _write_best_and_last(output_dir)
    best_digest = state_digest(best)
    last_digest = state_digest(last)

    live = _tiny_net(202)
    assert state_digest(live) == last_digest, "runner starts from the live last-epoch weights"

    observed: dict[str, list[str]] = {"collector": [], "sweep": [], "diagnostic": []}

    real_collect = runner.collect_validation_transition_cache
    real_sweep = runner.sweep_thresholds
    real_diagnostic = runner._diagnostic_at_tau

    def spy_collect(model, *args, **kwargs):
        observed["collector"].append(state_digest(model))
        return real_collect(model, *args, **kwargs)

    def spy_sweep(cache, thresholds, **kwargs):
        observed["sweep"].append(state_digest(live))
        return real_sweep(cache, thresholds, **kwargs)

    def spy_diagnostic(model, *args, **kwargs):
        observed["diagnostic"].append(state_digest(model))
        return real_diagnostic(model, *args, **kwargs)

    monkeypatch.setattr(runner, "collect_validation_transition_cache", spy_collect)
    monkeypatch.setattr(runner, "sweep_thresholds", spy_sweep)
    monkeypatch.setattr(runner, "_diagnostic_at_tau", spy_diagnostic)

    loader = DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False)
    args = argparse.Namespace(
        no_tqdm=True,
        max_val_batches=1,
        threshold_min=-0.02,
        threshold_max=0.02,
        threshold_steps=5,
        use_calibrated_tau=False,
        skip_calibration=False,
    )
    report: dict[str, Any] = {"tau": {}}
    calibration_path = report_dir / "calibration.json"

    headline_tau, headline_source = run_post_training_calibration(
        live,
        loader,
        torch.device("cpu"),
        args=args,
        config_c={"model": TINY_MODEL_CONFIG, "val_split": "val"},
        output_dir=output_dir,
        report_dir=report_dir,
        calibration_path=calibration_path,
        report=report,
        tau_accept=0.0,
        tau_source="default:0.0",
        t_max=2,
        neutral_margin_c=0.005,
        headline_tau=0.0,
        headline_tau_source="default:0.0",
    )

    assert observed["collector"] == [best_digest]
    assert observed["sweep"] == [best_digest]
    assert observed["diagnostic"] == [best_digest, best_digest]
    assert last_digest not in observed["collector"] + observed["sweep"] + observed["diagnostic"]

    binding_record = report["checkpoint_binding"]
    assert binding_record["checkpoint_role"] == "best"
    assert binding_record["state_digest"] == best_digest
    assert binding_record["fallback_used"] is False
    assert Path(binding_record["checkpoint_path"]) == output_dir / "phase_c_best.pt"

    # The artifact names the checkpoint that was measured, and carries the
    # lineage of the state it was measured under.
    artifact = report["calibration"]["artifact"]
    assert artifact["checkpoint_path"] == str(output_dir / "phase_c_best.pt")
    assert artifact["extra"]["lineage"]["checkpoint"]["state_digest"] == best_digest
    # The lineage is a first-class artifact field, not only an extra, and the
    # written artifact is verified against runtime-rebuilt lineage before the
    # calibrated diagnostic runs.
    assert artifact["lineage"]["checkpoint"]["state_digest"] == best_digest
    verification = report["calibration"]["lineage_verification"]
    assert verification["verified"] is True
    assert verification["boundary"] == "post_save_roundtrip"
    assert verification["cohort"]["role"] == "calibration"

    cache = torch.load(report_dir / "validation_transitions.pt", weights_only=False)
    assert cache["lineage"]["checkpoint"]["state_digest"] == best_digest

    # ``--max_val_batches`` truncated the diagnostics: the record says so.
    cohort = report["final_diagnostic"]["cohort"]
    assert cohort["max_batches"] == 1
    assert cohort["covers_full_split"] is False
    assert headline_source == "default:0.0"
    assert headline_tau == 0.0


def test_runner_post_training_branch_fails_without_checkpoint(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    loader = DataLoader(_SyntheticDataset(count=2), batch_size=2, shuffle=False)
    args = argparse.Namespace(
        no_tqdm=True,
        max_val_batches=1,
        threshold_min=-0.02,
        threshold_max=0.02,
        threshold_steps=3,
        use_calibrated_tau=False,
        skip_calibration=False,
    )
    with pytest.raises(FileNotFoundError):
        run_post_training_calibration(
            _tiny_net(21),
            loader,
            torch.device("cpu"),
            args=args,
            config_c={"model": TINY_MODEL_CONFIG, "val_split": "val"},
            output_dir=tmp_path / "empty",
            report_dir=report_dir,
            calibration_path=report_dir / "calibration.json",
            report={"tau": {}},
            tau_accept=0.0,
            tau_source="default:0.0",
            t_max=2,
            neutral_margin_c=0.005,
            headline_tau=0.0,
            headline_tau_source="default:0.0",
        )


# ---------------------------------------------------------------------------
# Recipe identity vs cohort identity, and honest sampling
# ---------------------------------------------------------------------------


def _volume_record(tmp_path: Path, case_id: str, shape: tuple[int, int, int] = (3, 40, 48)) -> VolumeRecord:
    volumes = tmp_path / "volumes"
    masks = tmp_path / "masks"
    volumes.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    image = np.random.default_rng(abs(hash(case_id)) % 2**31).standard_normal(shape).astype(np.float32)
    mask = np.zeros(shape, dtype=np.int64)
    mask[:, 5:15, 6:18] = 1
    image_path = volumes / f"{case_id}.npy"
    mask_path = masks / f"{case_id}.npy"
    np.save(image_path, image)
    np.save(mask_path, mask)
    return VolumeRecord(
        case_id=case_id,
        patient_id=case_id.split("_")[0],
        image_path=image_path,
        mask_path=mask_path,
        spacing=(10.0, 1.5, 2.0),
        source_format="npy",
    )


def test_preprocessing_recipe_signature_ignores_cohort_size(tmp_path: Path) -> None:
    """Two disjoint cohorts processed the same way share one recipe signature."""

    left = VolumeSliceDataset([_volume_record(tmp_path, "patient001_ED")], image_size=(32, 32))
    right = VolumeSliceDataset(
        [_volume_record(tmp_path, "patient002_ED"), _volume_record(tmp_path, "patient003_ED")],
        image_size=(32, 32),
    )
    left_recipe = preprocessing_descriptor(left)
    right_recipe = preprocessing_descriptor(right)
    assert left_recipe["signature"] == right_recipe["signature"]
    assert "num_records" not in left_recipe and "num_slices" not in left_recipe

    rescaled = VolumeSliceDataset([_volume_record(tmp_path, "patient004_ED")], image_size=(16, 16))
    assert preprocessing_descriptor(rescaled)["signature"] != left_recipe["signature"]

    # Cohort size lives in the cohort record, where it belongs.
    cohort = cohort_identity(left, split_name="val")
    assert cohort["available_slices"] == len(left)
    assert cohort["num_records"] == 1


def test_preprocessing_records_declared_geometry_semantics(tmp_path: Path) -> None:
    dataset = VolumeSliceDataset([_volume_record(tmp_path, "patient005_ED")], image_size=(32, 32))
    geometry = preprocessing_descriptor(dataset)["geometry"]
    assert geometry["status"] == "declared_by_pipeline_recipe"
    assert geometry["source_axis_order"] == "ZHW"
    assert geometry["network_axis_order"] == "ZHW"
    assert geometry["resize_semantics"] == "in_plane_bilinear_image_nearest_mask_z_preserved"
    assert geometry["spacing_semantics"] == "effective_spacing_rescaled_by_inplane_resize_ratio"
    assert "effective_spacing" in geometry["declared_keys"]

    # Per-record spacing availability is cohort content, reported over every
    # record, never inferred from one sample.
    cohort = cohort_identity(dataset, split_name="val")
    assert cohort["spacing_known_records"] == 1
    assert cohort["spacing_known_all_records"] is True


def test_geometry_descriptor_never_indexes_the_dataset(tmp_path: Path) -> None:
    """Building the recipe must not load GT, run transforms or consume RNG."""

    dataset = VolumeSliceDataset([_volume_record(tmp_path, "patient006_ED")], image_size=(32, 32))
    calls: list[int] = []
    original = type(dataset).__getitem__

    def _tracking_getitem(self, index):  # pragma: no cover - fails the assert below if hit
        calls.append(int(index))
        return original(self, index)

    torch.manual_seed(4242)
    before = torch.get_rng_state().clone()
    try:
        type(dataset).__getitem__ = _tracking_getitem
        preprocessing_descriptor(dataset)
    finally:
        type(dataset).__getitem__ = original
    assert calls == []
    assert torch.equal(before, torch.get_rng_state())


def test_unknown_dataset_claims_no_normalization() -> None:
    recipe = preprocessing_descriptor(_SyntheticDataset(count=2))
    assert recipe["normalization"] == UNKNOWN
    assert recipe["input_construction"] == UNKNOWN
    assert recipe["geometry"]["status"] == "unknown_dataset_no_geometry_contract"


def test_subset_cohort_cannot_claim_full_split() -> None:
    base = _SyntheticDataset(count=4)
    subset = Subset(base, [0, 2])
    loader = DataLoader(subset, batch_size=1, shuffle=False)
    cohort = cohort_identity(loader, split_name="val", batch_size=1)
    assert cohort["covers_full_split"] is False
    assert cohort["iterated_slices"] == 2
    assert cohort["available_slices"] == 4
    assert cohort["sampling"]["subset_size"] == 2
    assert cohort["sampling"]["subset_indices_signature"]
    assert "subset of the split" in cohort["subset_note"]


def test_unwrap_preserves_direct_and_nested_subset_indices() -> None:
    """A Subset is not transparent, and nesting composes outer-to-inner."""

    base = torch.utils.data.TensorDataset(torch.arange(10))
    direct = Subset(base, [2, 4, 6])
    unwrapped, indices = _unwrap_dataset(direct)
    assert unwrapped is base
    assert indices == [2, 4, 6]

    nested = DataLoader(Subset(direct, [1]), batch_size=1)
    unwrapped, indices = _unwrap_dataset(nested)
    assert unwrapped is base
    assert indices == [4]

    plain, none_indices = _unwrap_dataset(DataLoader(base, batch_size=1))
    assert plain is base
    assert none_indices is None


def test_nested_subset_cohort_reports_the_inner_selection() -> None:
    base = _SyntheticDataset(count=4)
    loader = DataLoader(Subset(Subset(base, [0, 2, 3]), [2]), batch_size=1, shuffle=False)
    cohort = cohort_identity(loader, split_name="val", batch_size=1)
    assert cohort["iterated_slices"] == 1
    assert cohort["available_slices"] == 4
    assert cohort["covers_full_split"] is False
    assert cohort["sampling"]["subset_size"] == 1


def test_drop_last_remainder_cannot_claim_full_split() -> None:
    loader = DataLoader(_SyntheticDataset(count=3), batch_size=2, shuffle=False, drop_last=True)
    cohort = cohort_identity(loader, split_name="val", batch_size=2)
    assert cohort["covers_full_split"] is False
    assert "drop_last" in cohort["subset_note"]

    exact = DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False, drop_last=True)
    assert cohort_identity(exact, split_name="val", batch_size=2)["covers_full_split"] is True


def test_shuffled_full_pass_is_recorded_as_shuffled_but_complete() -> None:
    loader = DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=True)
    cohort = cohort_identity(loader, split_name="val", batch_size=2)
    assert cohort["sampling"]["shuffled"] is True
    assert cohort["covers_full_split"] is True


def test_unsupported_sampler_fails_loudly() -> None:
    dataset = _SyntheticDataset(count=4)

    class _CustomSampler(torch.utils.data.Sampler):
        def __iter__(self):
            return iter([0, 1])

        def __len__(self) -> int:
            return 2

    loader = DataLoader(dataset, batch_size=1, sampler=_CustomSampler())
    with pytest.raises(ValueError, match="Unsupported sampler"):
        cohort_identity(loader, split_name="val", batch_size=1)


def test_lineage_resolves_full_versioned_metric_contract(tmp_path: Path) -> None:
    _write_best_and_last(tmp_path)
    live = _tiny_net(31)
    binding = bind_post_training_checkpoint(
        live,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        config={"model": TINY_MODEL_CONFIG},
    )
    lineage = build_lineage(
        binding=binding.as_dict(),
        metric_contract="foreground_dice_exclude_v1",
        t_max=2,
    )
    definition = lineage["semantics"]["metric_contract_definition"]
    assert definition["name"] == "foreground_dice_exclude_v1"
    assert definition["version"] == 1
    assert definition["empty_policy"] == "exclude"
    assert definition["classes"] == [1, 2, 3]
    assert definition["neutral_margin"] > 0.0
    assert lineage["semantics"]["metric_contract_version"] == 1


def test_unresolvable_live_attribute_is_not_silently_skipped() -> None:
    class _Opaque(nn.Module):
        """A module the identity resolver cannot read window_k off of."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

    with pytest.raises(ValueError, match="Cannot resolve live model attribute"):
        verify_model_config(_Opaque(), {"model": TINY_MODEL_CONFIG})


# ---------------------------------------------------------------------------
# Metric semantics come from the measurement, not from the caller
# ---------------------------------------------------------------------------


def _real_cache(model: SelfAuditNet, loader: DataLoader) -> dict[str, Any]:
    from src.self_audit.training.finetune_joint import collect_validation_transition_cache

    return collect_validation_transition_cache(
        model,
        loader,
        torch.device("cpu"),
        t_max=2,
        disable_tqdm=True,
        metric_contract="foreground_dice_exclude_v1",
    )


def test_lineage_semantics_are_complete_from_a_real_cache(tmp_path: Path) -> None:
    """metric_space and neutral_margin come out of the collected cache."""

    _write_best_and_last(tmp_path)
    live = _tiny_net(41)
    binding = bind_post_training_checkpoint(
        live,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        config={"model": TINY_MODEL_CONFIG},
    )
    loader = DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False)
    cache = _real_cache(live, loader)

    lineage = build_lineage(
        binding=binding.as_dict(),
        loader=loader,
        split_name="val",
        cache=cache,
        t_max=2,
        batch_size=2,
        observed_samples=int(cache["initial_dice"].shape[0]),
    )
    semantics = lineage["semantics"]
    assert semantics["metric_space"] == cache["metric_space"]
    assert semantics["neutral_margin"] == pytest.approx(float(cache["neutral_margin"]))
    assert semantics["metric_space"] is not None and semantics["neutral_margin"] is not None
    assert semantics["metric_contract"] == cache["metric_contract"]
    assert semantics["metric_contract_version"] == cache["metric_contract_version"]
    assert semantics["empty_class_policy"] == cache["empty_class_policy"]
    assert semantics["cache_schema_version"] == cache["cache_schema_version"]
    assert semantics["semantics_source"] == "cache_metadata"
    assert semantics["metric_contract_definition"]["metric_space"] == semantics["metric_space"]


def test_lineage_rejects_arguments_contradicting_the_cache(tmp_path: Path) -> None:
    from src.self_audit.evaluation.contracts import ContractMismatchError

    _write_best_and_last(tmp_path)
    live = _tiny_net(42)
    binding = bind_post_training_checkpoint(
        live,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        config={"model": TINY_MODEL_CONFIG},
    )
    loader = DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False)
    cache = _real_cache(live, loader)

    for label, kwargs in (
        ("neutral_margin", {"neutral_margin": 0.0}),
        ("metric_space", {"metric_space": "volume_native"}),
        ("metric_contract", {"metric_contract": "multiclass_dice_legacy_one_v1"}),
    ):
        with pytest.raises(ContractMismatchError, match=label):
            build_lineage(binding=binding.as_dict(), cache=cache, t_max=2, **kwargs)

    # A cache whose own embedded semantics contradict the contract it names is
    # rejected too: the contradiction is in the measurement, not the caller.
    for field, value in (
        ("metric_space", "volume_native"),
        ("neutral_margin", 0.25),
        ("metric_contract_version", 99),
        ("empty_class_policy", "legacy_one"),
    ):
        corrupted = dict(cache)
        corrupted[field] = value
        with pytest.raises(ContractMismatchError, match=f"cache {field}"):
            build_lineage(binding=binding.as_dict(), cache=corrupted, t_max=2)


def test_agreeing_arguments_are_accepted_as_assertions(tmp_path: Path) -> None:
    _write_best_and_last(tmp_path)
    live = _tiny_net(43)
    binding = bind_post_training_checkpoint(
        live,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        config={"model": TINY_MODEL_CONFIG},
    )
    loader = DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False)
    cache = _real_cache(live, loader)
    lineage = build_lineage(
        binding=binding.as_dict(),
        cache=cache,
        metric_contract=cache["metric_contract"],
        metric_space=cache["metric_space"],
        neutral_margin=float(cache["neutral_margin"]),
        t_max=2,
    )
    assert lineage["semantics"]["metric_space"] == cache["metric_space"]


# ---------------------------------------------------------------------------
# Cohort coverage cannot outrun what was observed
# ---------------------------------------------------------------------------


def test_observed_shortfall_cannot_claim_full_split() -> None:
    loader = DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False)
    full = cohort_identity(loader, split_name="val", batch_size=2, observed_samples=4)
    assert full["covers_full_split"] is True

    short = cohort_identity(loader, split_name="val", batch_size=2, observed_samples=3)
    assert short["covers_full_split"] is False
    assert "3 of 4 available samples" in short["subset_note"]


def test_shuffled_truncated_prefix_is_refused() -> None:
    loader = DataLoader(_SyntheticDataset(count=4), batch_size=1, shuffle=True)
    with pytest.raises(ValueError, match="truncated prefix of a shuffled loader"):
        cohort_identity(loader, split_name="val", batch_size=1, max_batches=2)

    # Unshuffled truncation stays describable: the prefix is deterministic.
    ordered = DataLoader(_SyntheticDataset(count=4), batch_size=1, shuffle=False)
    truncated = cohort_identity(ordered, split_name="val", batch_size=1, max_batches=2)
    assert truncated["covers_full_split"] is False
    assert truncated["sampling"]["shuffled"] is False


# ---------------------------------------------------------------------------
# The actual cache CLI
# ---------------------------------------------------------------------------


def _stub_cache_cli(monkeypatch, model: SelfAuditNet, loader: DataLoader):
    """Point the real CLI at a tiny in-memory model/loader, nothing else stubbed."""

    import scripts.cache_validation_transitions as cli
    from src.self_audit.training.unified_config import ResolvedExecutionConfig

    config = {
        "model": dict(TINY_MODEL_CONFIG),
        "val_split": "val",
        "audit": {"t_max": 2},
        "seed": 7,
    }

    class _StubResolvedConfig(ResolvedExecutionConfig):
        def build_model(self, device: torch.device | None = None) -> torch.nn.Module:
            return model

        def build_dataset(self, split: str | None = None, train: bool = False) -> Dataset:
            return loader.dataset

        def build_dataloader(self, dataset, train: bool = False, batch_size: int | None = None) -> DataLoader:
            return loader

        def validate_splits(self) -> dict[str, Any]:
            return {"validated": True}

    stub_resolved = _StubResolvedConfig(unified_config=None, flat_config=dict(config), is_unified=False)
    monkeypatch.setattr(cli, "resolve_downstream_config", lambda *args, **kwargs: stub_resolved)
    return cli


def test_cache_cli_writes_complete_consistent_lineage(tmp_path: Path, monkeypatch) -> None:
    """The CLI's saved cache carries a lineage with no missing semantics."""

    checkpoint = tmp_path / "phase_c_best.pt"
    net = _tiny_net(51)
    save_checkpoint(checkpoint, net, epoch=4, config={"model": TINY_MODEL_CONFIG})
    loader = DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False)
    output = tmp_path / "cache" / "validation_transitions.pt"

    cli = _stub_cache_cli(monkeypatch, _tiny_net(52), loader)
    monkeypatch.setattr(
        "sys.argv",
        [
            "cache_validation_transitions.py",
            "--config", "unused.yaml",
            "--checkpoint", str(checkpoint),
            "--output", str(output),
            "--no_tqdm",
        ],
    )
    cli.main()

    saved = torch.load(output, weights_only=False)
    lineage = saved["lineage"]
    semantics = lineage["semantics"]
    # Completeness: nothing the measurement knows is left as None.
    for key in (
        "metric_contract",
        "metric_contract_version",
        "metric_contract_definition",
        "metric_space",
        "neutral_margin",
        "empty_class_policy",
        "cache_schema_version",
        "t_max",
    ):
        assert semantics[key] is not None, f"lineage semantics.{key} must not be None"
    # Consistency: the lineage says what the cache itself recorded.
    assert semantics["metric_space"] == saved["metric_space"]
    assert semantics["neutral_margin"] == pytest.approx(float(saved["neutral_margin"]))
    assert semantics["metric_contract"] == saved["metric_contract"]
    assert semantics["empty_class_policy"] == saved["empty_class_policy"]
    assert semantics["t_max"] == 2
    # And the state it was measured under is the checkpoint that was bound.
    assert lineage["checkpoint"]["state_digest"] == state_digest(net)
    assert lineage["checkpoint"]["checkpoint_path"] == str(checkpoint)
    assert lineage["cohort"]["observed_samples"] == int(saved["initial_dice"].shape[0])
    assert lineage["preprocessing"]["signature"]


def test_cache_cli_verifies_bound_state_before_writing(tmp_path: Path, monkeypatch) -> None:
    """A model mutated after collection must stop the write, not be saved."""

    checkpoint = tmp_path / "phase_c_best.pt"
    save_checkpoint(checkpoint, _tiny_net(53), epoch=1, config={"model": TINY_MODEL_CONFIG})
    loader = DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False)
    output = tmp_path / "cache" / "validation_transitions.pt"
    model = _tiny_net(54)

    cli = _stub_cache_cli(monkeypatch, model, loader)
    real_collect = cli.collect_validation_transition_cache

    def mutating_collect(bound_model, *args, **kwargs):
        cache = real_collect(bound_model, *args, **kwargs)
        with torch.no_grad():
            next(iter(bound_model.parameters())).add_(1.0)
        return cache

    monkeypatch.setattr(cli, "collect_validation_transition_cache", mutating_collect)
    monkeypatch.setattr(
        "sys.argv",
        [
            "cache_validation_transitions.py",
            "--config", "unused.yaml",
            "--checkpoint", str(checkpoint),
            "--output", str(output),
            "--no_tqdm",
        ],
    )
    with pytest.raises(ValueError, match="Model state changed after checkpoint binding"):
        cli.main()
    assert not output.exists(), "no cache may be written once the binding is broken"


# ---------------------------------------------------------------------------
# Describing the live state without overwriting it
# ---------------------------------------------------------------------------


def test_bind_existing_state_describes_without_loading(tmp_path: Path) -> None:
    from src.self_audit.training._utils import bind_existing_evaluation_state

    net = _tiny_net(61)
    before = state_digest(net)
    save_checkpoint(tmp_path / "phase_c_best.pt", net, epoch=6, config={"model": TINY_MODEL_CONFIG})

    binding = bind_existing_evaluation_state(
        net,
        [("best", tmp_path / "phase_c_best.pt")],
        map_location=torch.device("cpu"),
        config={"model": TINY_MODEL_CONFIG},
    )
    assert binding.restored == (), "nothing may be restored by a describe-only bind"
    assert binding.state_digest == before
    assert state_digest(net) == before, "the live weights must be untouched"
    assert binding.checkpoint_sha256


def test_bind_existing_state_refuses_to_overwrite_a_mismatch(tmp_path: Path) -> None:
    from src.self_audit.training._utils import bind_existing_evaluation_state

    save_checkpoint(tmp_path / "phase_c_best.pt", _tiny_net(62), epoch=1, config={"model": TINY_MODEL_CONFIG})
    live = _tiny_net(63)
    before = state_digest(live)

    with pytest.raises(ValueError, match="will not overwrite the live weights"):
        bind_existing_evaluation_state(
            live,
            [("best", tmp_path / "phase_c_best.pt")],
            map_location=torch.device("cpu"),
            config={"model": TINY_MODEL_CONFIG},
        )
    assert state_digest(live) == before, "a failed describe-only bind must not change the model"


# ---------------------------------------------------------------------------
# Consuming a calibration artifact: verified or refused
# ---------------------------------------------------------------------------


def _write_verifiable_calibration(
    tmp_path: Path, model: SelfAuditNet, loader: DataLoader, *, with_lineage: bool = True
) -> Path:
    """Write an artifact whose lineage matches ``model``/``loader`` exactly."""

    from src.self_audit.evaluation.threshold import save_calibration, select_threshold, sweep_thresholds
    from src.self_audit.training._utils import bind_existing_evaluation_state

    save_checkpoint(tmp_path / "phase_c_best.pt", model, epoch=3, config={"model": TINY_MODEL_CONFIG})
    binding = bind_existing_evaluation_state(
        model,
        [("best", tmp_path / "phase_c_best.pt")],
        map_location=torch.device("cpu"),
        config={"model": TINY_MODEL_CONFIG},
    )
    cache = _real_cache(model, loader)
    lineage = build_lineage(
        binding=binding.as_dict(),
        loader=loader,
        split_name="val",
        cache=cache,
        t_max=2,
        batch_size=getattr(loader, "batch_size", None),
        observed_samples=int(cache["initial_dice"].shape[0]),
    )
    thresholds = np.linspace(-0.02, 0.02, 5)
    selected = select_threshold(sweep_thresholds(cache, thresholds, neutral_margin=float(cache["neutral_margin"])))
    path = tmp_path / "calibration.json"
    save_calibration(
        path,
        tau_accept=float(selected["tau_accept"]),
        neutral_margin=float(cache["neutral_margin"]),
        source_split="val",
        checkpoint_path=binding.path,
        t_max=2,
        threshold_grid=thresholds,
        selected_row=selected,
        metric_space=str(cache["metric_space"]),
        metric_contract=str(cache["metric_contract"]),
        lineage=lineage if with_lineage else None,
    )
    return path


def _tau_args(**overrides: Any) -> argparse.Namespace:
    base = {"tau_accept": None, "use_calibrated_tau": True}
    base.update(overrides)
    return argparse.Namespace(**base)


def _resolve_with(model: SelfAuditNet, loader: DataLoader, tmp_path: Path, artifact: Path, **overrides: Any):
    import scripts.train_self_audit_legacy as runner

    cache = _real_cache(model, loader)
    return runner._resolve_tau_accept(
        _tau_args(**overrides),
        {},
        artifact,
        model=model,
        val_loader=loader,
        config_c={"model": TINY_MODEL_CONFIG, "val_split": "val"},
        output_dir=tmp_path,
        device=torch.device("cpu"),
        metric_contract=str(cache["metric_contract"]),
        neutral_margin=float(cache["neutral_margin"]),
        t_max=2,
    )


def test_calibrated_tau_is_verified_against_starting_state(tmp_path: Path) -> None:
    model = _tiny_net(71)
    loader = DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False)
    artifact = _write_verifiable_calibration(tmp_path, model, loader)

    tau, source, verification = _resolve_with(model, loader, tmp_path, artifact)
    assert source == f"calibration_artifact:{artifact}"
    assert verification is not None
    assert verification["verified"] is True
    assert verification["boundary"] == "phase_c_starting_state"
    assert verification["cohort"]["role"] == "calibration"
    assert verification["starting_state_binding"]["state_digest"] == state_digest(model)
    assert verification["starting_state_binding"]["restored"] == []
    assert isinstance(tau, float)


def test_explicit_cli_tau_cannot_skip_artifact_verification(tmp_path: Path) -> None:
    from src.self_audit.evaluation.calibration_lineage import CalibrationLineageError

    model = _tiny_net(72)
    loader = DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False)
    artifact = _write_verifiable_calibration(tmp_path, model, loader)

    tau, source, verification = _resolve_with(model, loader, tmp_path, artifact, tau_accept=0.01)
    assert source == "cli:--tau_accept"
    assert tau == pytest.approx(0.01)
    assert verification is not None and verification["verified"] is True

    # ... and the override does not rescue an artifact that fails verification.
    with torch.no_grad():
        next(iter(model.parameters())).add_(1.0)
    with pytest.raises((ValueError, CalibrationLineageError)):
        _resolve_with(model, loader, tmp_path, artifact, tau_accept=0.01)


def test_legacy_calibration_without_lineage_is_refused(tmp_path: Path) -> None:
    model = _tiny_net(73)
    loader = DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False)
    artifact = _write_verifiable_calibration(tmp_path, model, loader, with_lineage=False)

    # Matched by message, not class: the runner and this test reach the same
    # module through different import paths (``self_audit`` vs ``src.self_audit``).
    with pytest.raises(ValueError, match="carries no lineage block"):
        _resolve_with(model, loader, tmp_path, artifact)


def test_uncalibrated_run_does_not_consult_or_verify_anything(tmp_path: Path) -> None:
    model = _tiny_net(74)
    loader = DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False)
    artifact = _write_verifiable_calibration(tmp_path, model, loader)

    tau, source, verification = _resolve_with(
        model, loader, tmp_path, artifact, use_calibrated_tau=False
    )
    assert source == "default:0.0"
    assert tau == 0.0
    assert verification is None


# ---------------------------------------------------------------------------
# Source content identity
# ---------------------------------------------------------------------------


def _fake_repo(root: Path) -> Path:
    """A miniature repository with one file in each scoped and excluded area."""

    (root / "src" / "self_audit" / "data").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "configs").mkdir()
    (root / "reports" / "astra").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "checkpoints").mkdir()
    (root / "src" / "self_audit" / "__pycache__").mkdir()
    (root / "data").mkdir()

    (root / "src" / "self_audit" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "src" / "self_audit" / "data" / "common.py").write_text("COMMON = 1\n", encoding="utf-8")
    (root / "scripts" / "run.py").write_text("print(VALUE)\n", encoding="utf-8")
    (root / "scripts" / "run_full_pipeline.sh").write_text("#!/bin/sh\necho 1\n", encoding="utf-8")
    (root / "scripts" / "run_full_pipeline.ps1").write_text("Write-Output 1\n", encoding="utf-8")
    (root / "configs" / "joint.yaml").write_text("tau_accept: 0.0\n", encoding="utf-8")
    (root / "reports" / "astra" / "review.md").write_text("first draft\n", encoding="utf-8")
    (root / "tests" / "test_model.py").write_text("assert True\n", encoding="utf-8")
    (root / "checkpoints" / "phase_c_best.pt").write_bytes(b"\x00\x01\x02")
    (root / "src" / "self_audit" / "__pycache__" / "model.pyc").write_bytes(b"\x00compiled")
    (root / "data" / "raw_volume.npy").write_bytes(b"\x00rawdata")
    return root


def test_source_signature_is_deterministic_and_scoped(tmp_path: Path) -> None:
    from src.self_audit.provenance import source_content_signature

    repo = _fake_repo(tmp_path / "repo")
    first = source_content_signature(repo)
    second = source_content_signature(repo)

    assert first["source_content_signature"] == second["source_content_signature"]
    assert first["source_content_signature_known"] is True
    assert first["source_signature_version"] == 2
    # Exactly the six scoped files: the report, the test, the checkpoint,
    # the .pyc, and raw data are all outside the scope.
    assert first["source_signature_scope"]["file_count"] == 6


def test_report_only_change_leaves_the_source_signature_untouched(tmp_path: Path) -> None:
    from src.self_audit.provenance import source_content_signature

    repo = _fake_repo(tmp_path / "repo")
    before = source_content_signature(repo)["source_content_signature"]

    (repo / "reports" / "astra" / "review.md").write_text("second draft, much longer\n", encoding="utf-8")
    (repo / "reports" / "astra" / "new_report.md").write_text("a whole new report\n", encoding="utf-8")
    (repo / "tests" / "test_model.py").write_text("assert True  # tightened\n", encoding="utf-8")
    (repo / "checkpoints" / "phase_c_best.pt").write_bytes(b"\xff\xfe\xfd")
    (repo / "data" / "raw_volume.npy").write_bytes(b"\xff\xfe\xfd\x00")

    after = source_content_signature(repo)["source_content_signature"]
    assert after == before, "report-, test-, checkpoint- and raw-data-only changes are not source changes"


def test_source_edit_changes_the_signature(tmp_path: Path) -> None:
    from src.self_audit.provenance import source_content_signature

    repo = _fake_repo(tmp_path / "repo")
    baseline = source_content_signature(repo)["source_content_signature"]

    # A one-character change in scoped code.
    (repo / "src" / "self_audit" / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert source_content_signature(repo)["source_content_signature"] != baseline
    (repo / "src" / "self_audit" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert source_content_signature(repo)["source_content_signature"] == baseline

    # A configuration change counts too: it changes what the run computes.
    (repo / "configs" / "joint.yaml").write_text("tau_accept: 0.5\n", encoding="utf-8")
    assert source_content_signature(repo)["source_content_signature"] != baseline
    (repo / "configs" / "joint.yaml").write_text("tau_accept: 0.0\n", encoding="utf-8")

    # So does adding a scoped file, and so does moving one.
    (repo / "scripts" / "extra.py").write_text("pass\n", encoding="utf-8")
    with_added = source_content_signature(repo)["source_content_signature"]
    assert with_added != baseline
    (repo / "scripts" / "extra.py").rename(repo / "scripts" / "renamed.py")
    assert source_content_signature(repo)["source_content_signature"] != with_added


def test_unavailable_source_is_honest_not_verified(tmp_path: Path) -> None:
    from src.self_audit.provenance import git_source_provenance, source_content_signature

    empty = tmp_path / "not_a_repo"
    empty.mkdir()
    record = source_content_signature(empty)
    assert record["source_content_signature"] == UNKNOWN
    assert record["source_content_signature_known"] is False
    assert record["source_signature_scope"]["file_count"] == 0

    # An unreadable scoped file must not silently narrow the signature either.
    repo = _fake_repo(tmp_path / "repo")
    unreadable = repo / "src" / "self_audit" / "model.py"
    unreadable.chmod(0o000)
    try:
        degraded = source_content_signature(repo)
    finally:
        unreadable.chmod(0o644)
    assert degraded["source_content_signature_known"] is False
    assert degraded["source_content_signature"] == UNKNOWN

    # The same honesty at the git-provenance boundary.
    provenance = git_source_provenance(empty)
    assert provenance["source_content_signature_known"] is False
    assert provenance["source_content_signature"] == UNKNOWN


def test_producing_source_signature_is_persisted_and_survives_weights_only(tmp_path: Path) -> None:
    from src.self_audit.provenance import git_source_provenance

    net = _tiny_net(81)
    path = save_checkpoint(tmp_path / "phase_c_best.pt", net, epoch=2, config={"model": TINY_MODEL_CONFIG})
    payload = torch.load(path, map_location="cpu", weights_only=True)
    provenance = payload["provenance"]

    runtime = git_source_provenance()
    assert provenance["producer_source_content_signature"] == runtime["source_content_signature"]
    assert provenance["producer_source_content_signature_known"] is True
    assert provenance["producer_source_signature_version"] == 2
    assert provenance["producer_source_signature_scope"]["roots"] == runtime["source_signature_scope"]["roots"]

    producer = checkpoint_producer(payload)
    assert producer["producer_source_content_signature"] == runtime["source_content_signature"]
    assert producer["producer_source_content_signature_known"] is True


def test_legacy_checkpoint_producing_source_stays_unknown(tmp_path: Path) -> None:
    from src.self_audit.provenance import git_source_provenance

    net = _tiny_net(82)
    legacy = tmp_path / "legacy.pt"
    torch.save({"model": {k: v.clone() for k, v in net.state_dict().items()}, "epoch": 1}, legacy)

    producer = checkpoint_producer(torch.load(legacy, map_location="cpu", weights_only=True))
    assert producer["producer_source_content_signature"] is None
    assert producer["producer_source_content_signature_known"] is False
    assert producer["producer_source_content_signature"] != git_source_provenance()["source_content_signature"]


def test_lineage_carries_producing_and_evaluating_source_separately(tmp_path: Path) -> None:
    _write_best_and_last(tmp_path)
    live = _tiny_net(83)
    binding = bind_post_training_checkpoint(
        live,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        config={"model": TINY_MODEL_CONFIG},
    )
    lineage = build_lineage(binding=binding.as_dict(), metric_contract="foreground_dice_exclude_v1", t_max=2)

    evaluating = lineage["evaluation_code"]["source_content_signature"]
    producing = lineage["checkpoint"]["producer"]["producer_source_content_signature"]
    assert lineage["evaluation_code"]["source_content_signature_known"] is True
    assert evaluating and producing
    # Two separate slots: this run happens to have produced the checkpoint, so
    # they agree, but a consumer reads them independently and never infers one
    # from the other.
    assert "producer_source_content_signature" not in lineage["evaluation_code"]
    assert "source_content_signature" not in lineage["checkpoint"]["producer"]


# ---------------------------------------------------------------------------
# Wave 3.1 regression tests: source identity scope, error handling, invariants
# ---------------------------------------------------------------------------


def test_temp_tree_regression_data_common_edit(tmp_path: Path) -> None:
    from src.self_audit.provenance import source_content_signature

    repo = _fake_repo(tmp_path / "repo")
    baseline = source_content_signature(repo)["source_content_signature"]

    # 1. Modifying src/self_audit/data/common.py changes the signature.
    (repo / "src" / "self_audit" / "data" / "common.py").write_text("COMMON = 2\n", encoding="utf-8")
    assert source_content_signature(repo)["source_content_signature"] != baseline

    # 2. Reverting restores the exact baseline signature.
    (repo / "src" / "self_audit" / "data" / "common.py").write_text("COMMON = 1\n", encoding="utf-8")
    assert source_content_signature(repo)["source_content_signature"] == baseline

    # 3. Modifying raw data at repository root does NOT change signature (no raw-data walks).
    (repo / "data" / "raw_volume.npy").write_bytes(b"\x99mutated_raw")
    (repo / "data" / "extra_volume.npy").write_bytes(b"\x88extra_raw")
    (repo / "data" / "code_leak.py").write_text("LEAK = 1\n", encoding="utf-8")
    assert source_content_signature(repo)["source_content_signature"] == baseline


def test_temp_tree_regression_wrapper_edits(tmp_path: Path) -> None:
    from src.self_audit.provenance import source_content_signature

    repo = _fake_repo(tmp_path / "repo")
    baseline = source_content_signature(repo)["source_content_signature"]

    # .sh wrapper edit changes signature; reverting restores it.
    (repo / "scripts" / "run_full_pipeline.sh").write_text("#!/bin/sh\necho 2\n", encoding="utf-8")
    assert source_content_signature(repo)["source_content_signature"] != baseline
    (repo / "scripts" / "run_full_pipeline.sh").write_text("#!/bin/sh\necho 1\n", encoding="utf-8")
    assert source_content_signature(repo)["source_content_signature"] == baseline

    # .ps1 wrapper edit changes signature; reverting restores it.
    (repo / "scripts" / "run_full_pipeline.ps1").write_text("Write-Output 2\n", encoding="utf-8")
    assert source_content_signature(repo)["source_content_signature"] != baseline
    (repo / "scripts" / "run_full_pipeline.ps1").write_text("Write-Output 1\n", encoding="utf-8")
    assert source_content_signature(repo)["source_content_signature"] == baseline


def test_temp_tree_regression_report_and_test_only_invariants(tmp_path: Path) -> None:
    from src.self_audit.provenance import source_content_signature

    repo = _fake_repo(tmp_path / "repo")
    baseline = source_content_signature(repo)["source_content_signature"]

    # Edits and additions in reports/ and tests/ leave signature strictly invariant.
    (repo / "reports" / "astra" / "review.md").write_text("updated review draft\n", encoding="utf-8")
    (repo / "reports" / "new_report.md").write_text("brand new report\n", encoding="utf-8")
    (repo / "tests" / "test_model.py").write_text("assert 1 + 1 == 2\n", encoding="utf-8")
    (repo / "tests" / "test_extra.py").write_text("def test_extra(): pass\n", encoding="utf-8")

    assert source_content_signature(repo)["source_content_signature"] == baseline


def test_temp_tree_regression_monkeypatched_unreadable_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.self_audit.provenance import _scoped_source_files, source_content_signature

    repo = _fake_repo(tmp_path / "repo")
    target_dir = repo / "src" / "self_audit" / "data"

    original_iterdir = Path.iterdir

    def mock_iterdir(self: Path) -> Any:
        if self.resolve() == target_dir.resolve():
            raise PermissionError("Simulated unreadable directory: permission denied")
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", mock_iterdir)

    # 1. _scoped_source_files must NOT swallow OSError/PermissionError and continue;
    # it must raise OSError.
    with pytest.raises(OSError):
        _scoped_source_files(repo)

    # 2. source_content_signature must fail closed: return unknown False, never partial known.
    record = source_content_signature(repo)
    assert record["source_content_signature"] == UNKNOWN
    assert record["source_content_signature_known"] is False


def test_temp_tree_regression_missing_scope_roots_fail_closed(tmp_path: Path) -> None:
    import shutil
    from src.self_audit.provenance import (
        SOURCE_SIGNATURE_ROOTS,
        _scoped_source_files,
        git_source_provenance,
        source_content_signature,
    )

    for relative_root in SOURCE_SIGNATURE_ROOTS:
        repo = _fake_repo(tmp_path / f"repo_missing_{relative_root.replace('/', '_')}")
        target_root = repo / relative_root
        shutil.rmtree(target_root)

        # 1. Missing required root explicitly raises FileNotFoundError from _scoped_source_files.
        with pytest.raises(FileNotFoundError, match=f"Missing required scope root: {relative_root}"):
            _scoped_source_files(repo)

        # 2. source_content_signature fails closed: unknown and False.
        sig = source_content_signature(repo)
        assert sig["source_content_signature"] == UNKNOWN
        assert sig["source_content_signature_known"] is False

        # 3. git_source_provenance also fails closed.
        prov = git_source_provenance(repo)
        assert prov["source_content_signature"] == UNKNOWN
        assert prov["source_content_signature_known"] is False
