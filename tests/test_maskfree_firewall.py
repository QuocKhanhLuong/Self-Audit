"""Independent integration firewall tests for the mask-free pipeline (W8 red team).

Adversarial tests written against the frozen `maskfree150` contract by a worker
who owns no production code. They are deliberately *integration* tests: each one
drives the real public APIs end to end -- discovery, unit construction, bank
generation, audit, verification gating, trainer -- rather than asserting things
about a single module in isolation. A firewall that holds in one module and
leaks at a seam is not a firewall.

Every test here passes against the current implementation. There are no `xfail`
markers: a firewall regression must turn this suite red, not quietly flip an
expected failure into an expected pass.

Fixtures are synthetic CPU `.npy` volumes in an M&Ms-shaped tree. They are
software evidence about the implemented data flow only, and say nothing about
real ACDC or M&Ms content, which this checkout does not contain.

Run with the canonical import path::

    PYTHONPATH=.:src python3 -m pytest tests/test_maskfree_firewall.py -v
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from self_audit_maskfree.auditor import audit_bank  # noqa: E402
from self_audit_maskfree.contracts import ScoringView  # noqa: E402
from self_audit_maskfree.data import (  # noqa: E402
    FreezeReceiptError,
    ImageOnlyDataset,
    VerificationAccessError,
    build_partition,
    build_training_unit,
    discover_dataset,
    load_verification_unit,
)
from self_audit_maskfree.evaluation.verification import verify_frozen_bank  # noqa: E402
from self_audit_maskfree.hypotheses import generate_bank  # noqa: E402
from self_audit_maskfree.observation import ObservationContractError, ObservationModel  # noqa: E402

NATIVE_HW = (72, 68)
DEPTH = 5
IMAGE_SIZE = 64


def _tree(tmp_path: Path, n_cases: int = 4) -> Path:
    """M&Ms-shaped preprocessed tree: one ED and one ES volume per case, no masks."""
    volumes = tmp_path / "preprocessed_data" / "mnm" / "train" / "volumes"
    volumes.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for index in range(1, n_cases + 1):
        for phase in ("ED", "ES"):
            np.save(
                volumes / f"case{index:03d}_{phase}.npy",
                rng.normal(120.0, 25.0, (*NATIVE_HW, DEPTH)).astype(np.float32),
            )
    return volumes


def _all_units(manifest: dict) -> dict[str, object]:
    return {
        record["unit_id"]: build_training_unit(
            manifest, record["unit_id"], image_size=IMAGE_SIZE, seed=42
        )
        for record in manifest["records"]
    }


def _identical(first, second) -> bool:
    return (
        torch.equal(first.fitting.image, second.fitting.image)
        and torch.equal(first.fitting.context, second.fitting.context)
        and torch.equal(first.fitting.support, second.fitting.support)
        and torch.equal(first.selection.image, second.selection.image)
        and torch.equal(first.selection.support, second.selection.support)
        and first.record["normalization"] == second.record["normalization"]
    )


def _mutate_role(volumes: Path, glob: str, role_mask: torch.Tensor, *, scale: float, shift: float) -> None:
    """Rewrite one observation role, in every slice of every file matching ``glob``."""
    mask = role_mask.numpy()
    for path in sorted(volumes.glob(glob)):
        array = np.load(path)
        array[mask, :] = array[mask, :] * scale + shift
        np.save(path, array)


def _freeze_receipt(tmp_path: Path, manifest: dict, partition_ids: list[str]) -> tuple[dict, Path]:
    """Create a complete W7 freeze; arbitrary hash-only receipts are invalid."""
    from self_audit_maskfree.export import REQUIRED_COMPARISONS, freeze_predictions

    tag = "valid" if partition_ids == [manifest["records"][0]["partition_id"]] else "wrong"
    output = tmp_path / f"freeze_{tag}"
    output.mkdir(parents=True, exist_ok=True)
    unit = manifest["records"][0]
    entries: list[dict] = []
    checkpoints: dict[str, Path] = {}
    frozen: Path | None = None
    for method in REQUIRED_COMPARISONS:
        method_dir = output / "predictions" / method
        method_dir.mkdir(parents=True, exist_ok=True)
        labels = method_dir / "labels.bin"
        validity = method_dir / "validity.bin"
        labels.write_bytes(f"labels:{method}".encode())
        validity.write_bytes(f"validity:{method}".encode())
        if frozen is None:
            frozen = labels
        entries.append(
            {
                "kind": "volume",
                "prediction_name": method,
                "unit_id": unit["unit_id"],
                "volume_id": unit["volume_id"],
                "study_id": unit["study_id"],
                "patient_id": unit["patient_id"],
                "split": unit["split"],
                "dataset": manifest["dataset"],
                "protocol": manifest["resolved_protocol"],
                "manifest_id": manifest["manifest_id"],
                "partition_id": partition_ids[0],
                "checkpoint_id": method,
                "version": "maskfree150.test.v1",
                "grid": "stored",
                "native_export_available": False,
                "native_export_reason": "synthetic test artifact",
                "volume_shape": [DEPTH, *NATIVE_HW],
                "complete_volume": True,
                "files": [
                    str(labels.relative_to(output)),
                    str(validity.relative_to(output)),
                ],
            }
        )
        checkpoint = tmp_path / f"{method}.pt"
        checkpoint.write_bytes(method.encode())
        checkpoints[method] = checkpoint
    nuisance = tmp_path / "nuisance_provenance.json"
    nuisance.write_text(json.dumps({"source": "synthetic-test"}), encoding="utf-8")
    receipt = freeze_predictions(
        output,
        entries,
        checkpoints,
        dataset=manifest["dataset"],
        protocol=manifest["resolved_protocol"],
        required_methods=REQUIRED_COMPARISONS,
        required_unit_ids=[unit["unit_id"]],
        required_checkpoints=list(checkpoints),
        required_nuisance_files=[nuisance],
    )
    assert frozen is not None
    return receipt, frozen


def test_study_wide_verify_mutation_changes_no_training_unit(tmp_path):
    """The whole point of a study-scoped XY role map, tested at study scope.

    A per-unit role map would be a silent global leak: a voxel sealed for the ED
    unit would be an ordinary fitting pixel for the ES unit of the same patient,
    and the sealed intensity would resurface as fitting context one unit over.
    So the check is not "unit U ignores unit U's verify pixels" -- it is
    "rewriting every verify pixel of a study, in every slice of every one of its
    acquisitions, moves nothing in *any* unit of *any* split".

    The selection mutation is the positive control. If it changed nothing either,
    the comparison would be vacuous: the loader would simply be ignoring the
    files.
    """
    volumes = _tree(tmp_path)
    manifest = discover_dataset(tmp_path, "mnms", seed=42)
    before = _all_units(manifest)
    assert len(before) == 8 * DEPTH

    studies = {record["study_id"] for record in manifest["records"]}
    study = sorted(studies)[0]
    record = next(r for r in manifest["records"] if r["study_id"] == study)
    height, width = record["native_hw"]
    partition = build_partition(height, width, study_id=study, seed=42)

    # Every unit of this study must agree on one partition, or the map is not
    # study-scoped at all and the mutation below would prove nothing.
    study_units = [r for r in manifest["records"] if r["study_id"] == study]
    assert len(study_units) > 1, "need a multi-acquisition study for a cross-unit check"
    assert {r["partition_id"] for r in study_units} == {partition.partition_id}
    assert partition.counts()["verify"] > 0 and partition.counts()["guard_dropped"] > 0

    case = study.split(":")[-1]
    _mutate_role(volumes, f"{case}_*.npy", partition.verify, scale=-7.0, shift=999.0)

    after = _all_units(discover_dataset(tmp_path, "mnms", seed=42))
    changed = [unit_id for unit_id in before if not _identical(before[unit_id], after[unit_id])]
    assert changed == [], (
        f"sealed verification intensities reached {len(changed)} training unit(s): {changed}"
    )

    # Positive control: selection observations are adaptive evidence. They must
    # move the selection view and still leave every fitting view untouched.
    _mutate_role(volumes, f"{case}_*.npy", partition.select, scale=3.0, shift=-50.0)
    controlled = _all_units(discover_dataset(tmp_path, "mnms", seed=42))
    moved = [
        unit_id
        for unit_id in before
        if not torch.equal(before[unit_id].selection.image, controlled[unit_id].selection.image)
    ]
    assert moved, "selection mutation moved nothing; the comparison would be vacuous"
    for unit_id in before:
        assert torch.equal(before[unit_id].fitting.image, controlled[unit_id].fitting.image)
        assert torch.equal(before[unit_id].fitting.context, controlled[unit_id].fitting.context)
        assert (
            before[unit_id].record["normalization"] == controlled[unit_id].record["normalization"]
        ), "selection intensities leaked into the fit-side normalization statistics"


def test_sealed_verification_is_reachable_only_behind_a_valid_freeze(tmp_path):
    """`O_verify` is gated where the contract says the gate lives, not in the scorer.

    `ObservationModel.score` is generic mathematics and will score whatever view
    it is handed; that is not the leak. The gates that matter are the two APIs
    that can actually *produce* or *consume* a verification view, plus the
    auditor that must refuse one. All four are exercised here against the real
    implementations.
    """
    _tree(tmp_path)
    manifest = discover_dataset(tmp_path, "mnms", seed=42)
    unit_id = manifest["records"][0]["unit_id"]
    unit = build_training_unit(manifest, unit_id, image_size=IMAGE_SIZE, seed=42)
    partition_id = unit.fitting.partition_id

    # 1. The training dataset has no verification surface at all.
    dataset = ImageOnlyDataset(manifest, split="train", image_size=IMAGE_SIZE, seed=42)
    with pytest.raises(VerificationAccessError):
        dataset.load_verification_unit(manifest, unit_id, {})

    # 2. No receipt, and a receipt for a different manifest, are both refused.
    with pytest.raises(FreezeReceiptError):
        load_verification_unit(manifest, unit_id, {}, image_size=IMAGE_SIZE, seed=42)
    with pytest.raises(FreezeReceiptError):
        load_verification_unit(
            manifest,
            unit_id,
            {
                "dataset": manifest["dataset"],
                "manifest_id": "a-different-manifest",
                "partition_ids": [partition_id],
                "files": {"/etc/hosts": "0" * 64},
            },
            image_size=IMAGE_SIZE,
            seed=42,
        )

    # 3. A receipt that does not cover this partition is refused.
    receipt, frozen = _freeze_receipt(tmp_path, manifest, ["a" * 32])
    with pytest.raises(FreezeReceiptError):
        load_verification_unit(manifest, unit_id, receipt, image_size=IMAGE_SIZE, seed=42)

    # 4. A valid receipt releases exactly one verify-role view.
    receipt, frozen = _freeze_receipt(tmp_path, manifest, [partition_id])
    verify_view = load_verification_unit(manifest, unit_id, receipt, image_size=IMAGE_SIZE, seed=42)
    assert verify_view.role == "verify"
    assert int(verify_view.support.sum()) > 0
    assert verify_view.partition_id == partition_id
    # Sealed observations are disjoint from everything the unit trained on.
    assert not bool((verify_view.support & unit.fitting.support).any())
    assert not bool((verify_view.support & unit.selection.support).any())

    # 5. Mutating a frozen file after the freeze re-seals the data.
    frozen_original = frozen.read_bytes()
    frozen.write_text(json.dumps({"predictions": ["E5", "tampered"]}), encoding="utf-8")
    with pytest.raises(FreezeReceiptError):
        load_verification_unit(manifest, unit_id, receipt, image_size=IMAGE_SIZE, seed=42)
    frozen.write_bytes(frozen_original)

    # 6. The auditor refuses a verification view outright, and the post-freeze
    #    verifier refuses a selection view. Neither role can stand in for the other.
    model = ObservationModel()
    bank = generate_bank(unit.fitting, seed=42)
    with pytest.raises(Exception) as audit_refusal:
        audit_bank(bank, unit.fitting, verify_view, model)
    assert "verif" in str(audit_refusal.value).lower()

    fitted = [model.fit(hypothesis, unit.fitting) for hypothesis in bank]
    with pytest.raises(ValueError) as verify_refusal:
        verify_frozen_bank(fitted, unit.selection, model, receipt)
    assert "verif" in str(verify_refusal.value).lower()


def test_candidate_generation_and_scoring_are_structurally_fit_only(tmp_path):
    """Bank construction cannot see withheld intensities, and a fit cannot score itself.

    Three separate claims, each checked against the real objects:

    * `generate_bank` takes no scoring parameter -- the firewall is in the
      signature, not in a runtime check a caller could skip;
    * rewriting every selection and verification intensity leaves the produced
      bank byte-identical, so no indirect path (normalization, context, features)
      carries withheld values into a candidate;
    * a fit now records its own support, so scoring it against the pixels it was
      fitted on is refused instead of silently returning a better number.
    """
    volumes = _tree(tmp_path)
    manifest = discover_dataset(tmp_path, "mnms", seed=42)
    unit_id = manifest["records"][0]["unit_id"]
    unit = build_training_unit(manifest, unit_id, image_size=IMAGE_SIZE, seed=42)

    parameters = inspect.signature(generate_bank).parameters
    assert "selection_view" not in parameters and "scoring_view" not in parameters
    assert [name for name in parameters] == ["fitting_view", "features", "seed"]

    bank_before = generate_bank(unit.fitting, seed=42)
    assert len(bank_before) == 4

    study = manifest["records"][0]["study_id"]
    height, width = manifest["records"][0]["native_hw"]
    partition = build_partition(height, width, study_id=study, seed=42)
    case = study.split(":")[-1]
    _mutate_role(volumes, f"{case}_*.npy", partition.select, scale=-4.0, shift=310.0)
    _mutate_role(volumes, f"{case}_*.npy", partition.verify, scale=6.0, shift=-77.0)

    rebuilt = build_training_unit(
        discover_dataset(tmp_path, "mnms", seed=42), unit_id, image_size=IMAGE_SIZE, seed=42
    )
    bank_after = generate_bank(rebuilt.fitting, seed=42)
    for original, candidate in zip(bank_before, bank_after):
        assert original.candidate_id == candidate.candidate_id
        assert torch.equal(original.labels, candidate.labels), (
            f"candidate {original.candidate_id} moved when withheld intensities changed"
        )
        assert torch.equal(original.validity, candidate.validity)

    # A fit may not be scored on its own fitting support.
    model = ObservationModel()
    fitted = model.fit(bank_before[0], unit.fitting)
    contaminated = ScoringView(
        image=unit.fitting.image.clone(),
        support=unit.fitting.support.clone(),
        study_id=unit.fitting.study_id,
        unit_id=unit.fitting.unit_id,
        role="select",
        partition_id=unit.fitting.partition_id,
    )
    with pytest.raises(ObservationContractError):
        model.score(fitted, contaminated)
    # The genuinely withheld selection observations still score normally.
    honest = model.score(fitted, unit.selection)
    assert honest.available and honest.total is not None


def test_bounded_run_is_partial_and_never_opens_verification(tmp_path):
    """A short run must report itself partial and must not reach `O_verify`.

    This is the "misleading 150 completion" check driven through the real
    trainer: an interrupted or bounded run has to fail closed at the status
    field, and finalization -- the only stage permitted to unseal verification
    data -- must not be attempted at all.
    """
    from self_audit_maskfree.config import MaskfreeConfig
    from self_audit_maskfree.trainer import MaskfreeTrainer

    _tree(tmp_path, n_cases=6)
    config = MaskfreeConfig(
        dataset="mnms",
        data_root=str(tmp_path),
        output_dir=str(tmp_path / "runs"),
        total_epochs=150,
        max_epochs=1,
        warmup_epochs=1,
        image_size=32,
        batch_size=2,
        device="cpu",
        allow_cpu=True,
        amp=False,
        wandb_mode="disabled",
    )
    report = MaskfreeTrainer(config).run()

    assert report["status"] != "completed"
    finalization = report["finalization"]
    assert finalization["attempted"] is False
    assert finalization["available"] is False
    assert finalization["reason"]
    # Nothing in a bounded run may carry a verification score.
    assert "verification" not in finalization
    blob = json.dumps(report, default=str)
    assert '"role": "verify"' not in blob
