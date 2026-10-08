"""Independent slow accounting oracle for the allocation-bounded regional path.

Synthetic CPU checks.  These do not qualify CUDA equivalence or label quality.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from self_audit_maskfree import auditor, hypotheses
from self_audit_maskfree.contracts import FittingView, ScoringView, hypothesis_from_labels
from self_audit_maskfree.observation import ObservationModel
from self_audit_maskfree.trainer import MaskfreeTrainer


def _pattern(kind: str, size: int = 32) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    if kind == "empty":
        return torch.zeros(size, size, dtype=torch.long)
    if kind == "fragmented":
        return (yy % 2) * 2 + xx % 2
    if kind == "stripes":
        return (xx // 4) % 4
    if kind == "random":
        return torch.randint(4, (size, size), generator=torch.Generator().manual_seed(29))
    labels = torch.zeros(size, size, dtype=torch.long)
    labels[2:-2, 2:-2] = 1
    labels[6:-6, 6:-6] = 2
    labels[10:-10, 10:-10] = 3
    return labels


def _slow_index(labels):
    """Original mask-per-component enumeration, independent of indexed code."""
    originals, converged = auditor._selected_regions(labels)
    ids = torch.empty_like(labels, dtype=torch.long)
    regions = []
    for index, region in enumerate(originals):
        ids[region["mask"]] = index
        regions.append({"role": region["role"], "component": region["component"],
                        "index": index, "pixels": int(region["mask"].sum())})
    return regions, ids, converged


def _slow_counts(ids, support, count):
    return [int(((ids == index) & support).sum()) for index in range(count)]


@pytest.mark.parametrize("kind", ["empty", "nested", "fragmented", "stripes", "random"])
@pytest.mark.parametrize("transpose", [False, True])
def test_index_matches_original_masks_order_and_every_region_count(kind, transpose):
    labels = _pattern(kind)
    if transpose:
        labels = labels.T
    before = labels.clone()
    old, old_converged = auditor._selected_regions(labels)
    new, ids, new_converged = auditor._indexed_selected_regions(labels)
    assert old_converged == new_converged
    assert len(old) == len(new)
    assert ids.dtype == torch.long
    support = torch.zeros_like(labels, dtype=torch.bool)
    support[::3, 1::2] = True
    counts = auditor._region_counts(ids, support, len(new))
    difference = labels != labels.roll(1, dims=0)
    disagreements = auditor._region_counts(ids, difference & support, len(new))
    for left, right in zip(old, new):
        mask = ids == right["index"]
        assert torch.equal(mask, left["mask"])
        assert (left["role"], left["component"]) == (right["role"], right["component"])
        assert right["pixels"] == int(mask.sum())
        assert counts[right["index"]] == int((mask & support).sum())
        assert disagreements[right["index"]] == int((mask & support & difference).sum())
        assert not any(isinstance(value, torch.Tensor) for value in right.values())
    assert torch.equal(before, labels)


def test_nonconvergence_preserves_original_whole_role_fallback(monkeypatch):
    labels = _pattern("random")
    monkeypatch.setattr(auditor, "connected_components",
                        lambda mask: (torch.zeros_like(mask, dtype=torch.long), False))
    old, _ = auditor._selected_regions(labels)
    new, ids, converged = auditor._indexed_selected_regions(labels)
    assert converged is False
    for left, right in zip(old, new):
        assert right["component"] == 0
        assert left["role"] == right["role"]
        assert torch.equal(left["mask"], ids == right["index"])


def _views(labels, support_mode="normal"):
    size = labels.shape[0]
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    select = ((yy // 4 + xx // 4) % 4) == 0
    if support_mode == "none":
        select = torch.zeros_like(select)
    image = labels.float() * 0.6 - 1.0 + xx.float() / (size * 10)
    fitting_image = torch.where(~select, image, 0).unsqueeze(0)
    fitting = FittingView(image=fitting_image, support=~select,
                          context=fitting_image.repeat(3, 1, 1),
                          study_id="index-test", unit_id="u", partition_id="p")
    scoring = ScoringView(image=torch.where(select, image, 0).unsqueeze(0),
                          support=select, study_id="index-test", unit_id="u",
                          partition_id="p", role="select")
    return fitting, scoring


@pytest.mark.parametrize("kind", ["nested", "fragmented", "stripes"])
@pytest.mark.parametrize("support_mode", ["normal", "none"])
def test_complete_audit_matches_slow_accounting_and_preserves_budget(monkeypatch, kind, support_mode):
    labels = _pattern(kind)
    fitting, scoring = _views(labels, support_mode)
    bank = [hypothesis_from_labels(array, name, name) for array, name in [
        (labels, "incumbent"), (labels.roll(1, 1), "shift"),
        ((labels + 1) % 4, "permutation"), (torch.zeros_like(labels), "all-bg")]]
    before = auditor.audit_banks([bank], [fitting], [scoring], ObservationModel())[0]
    with monkeypatch.context() as patch:
        patch.setattr(auditor, "_indexed_selected_regions", _slow_index)
        patch.setattr(auditor, "_region_counts", _slow_counts)
        after = auditor.audit_banks([bank], [fitting], [scoring], ObservationModel())[0]
    assert before.trace == after.trace
    assert before.selected.candidate_id == after.selected.candidate_id
    assert torch.equal(before.regional_margin, after.regional_margin)
    assert torch.equal(before.validity, after.validity)
    assert torch.equal(before.selected.labels, after.selected.labels)
    assert before.trace["regional_score_calls"] <= auditor.MAX_REGION_SCORE_CALLS


@pytest.mark.parametrize("channels", [4, 8, 16])
def test_canonical_feature_transfer_keeps_only_consumed_detached_channels(channels):
    trainer = object.__new__(MaskfreeTrainer)
    trainer.components = SimpleNamespace(generate_bank=hypotheses.generate_bank)
    features = torch.randn(2, channels, 8, 8, requires_grad=True)
    result = trainer._candidate_features_to_cpu(features)
    assert result.shape == (2, min(channels, 8), 8, 8)
    assert not result.requires_grad
    assert torch.equal(result, features.detach()[:, :8])


def test_custom_feature_generator_keeps_all_channels():
    trainer = object.__new__(MaskfreeTrainer)
    trainer.components = SimpleNamespace(generate_bank=lambda *args, **kwargs: None)
    features = torch.randn(2, 16, 8, 8, requires_grad=True)
    result = trainer._candidate_features_to_cpu(features)
    assert result.shape == features.shape
    assert not result.requires_grad
    assert torch.equal(result, features.detach())


def test_feature_slice_preserves_all_candidate_tensors_and_content_identity():
    fitting, _ = _views(_pattern("nested"))
    features = torch.randn(16, 32, 32, generator=torch.Generator().manual_seed(71))
    full = hypotheses.generate_bank(fitting, features, seed=23)
    sliced = hypotheses.generate_bank(fitting, features[:8], seed=23)
    for left, right in zip(full, sliced):
        assert left.candidate_id == right.candidate_id
        assert left.source == right.source
        assert left.alternatives == right.alternatives
        assert left.semantic_unresolved == right.semantic_unresolved
        assert torch.equal(left.labels, right.labels)
        assert torch.equal(left.probabilities, right.probabilities)
        assert torch.equal(left.validity, right.validity)
        for key in ("bank_id", "feature_identity", "ontology_trace", "prior_penalty"):
            assert left.metadata.get(key) == right.metadata.get(key)
    assert hypotheses.bank_identity(fitting, full, features=features, seed=23) == \
           hypotheses.bank_identity(fitting, sliced, features=features[:8], seed=23)


@pytest.mark.parametrize("total_epochs", [50, 150])
def test_b64_profiler_and_rental_gate_bind_the_explicit_schedule(tmp_path, total_epochs):
    import json
    from scripts import profile_maskfree, benchmark_maskfree_rental
    from self_audit_maskfree.config import MaskfreeConfig
    config = MaskfreeConfig(dataset="acdc", data_root=str(tmp_path), output_dir=str(tmp_path / "out"),
                            total_epochs=total_epochs, batch_size=64, image_size=224, amp=False,
                            device="cpu", allow_cpu=True, candidate_workers=8, candidate_chunk_size=8)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    args = profile_maskfree.build_parser().parse_args([
        "--config", str(path), "--batch-size", "64", "--image-size", "224",
        "--total-epochs", str(total_epochs)])
    resolved, info = profile_maskfree._prepare_config(args, tmp_path / "profile")
    assert resolved.total_epochs == total_epochs and resolved.batch_size == 64
    assert info["scientific_requirements"]["total_epochs"] == total_epochs
    assert config.scientific_identity()["total_epochs"] == total_epochs
    args.total_epochs = 50 if total_epochs == 150 else 150
    with pytest.raises(ValueError, match="mismatches"):
        profile_maskfree._prepare_config(args, tmp_path / "bad")
    mnms = tmp_path / "MnM/extracted"
    args = benchmark_maskfree_rental.parser().parse_args([
        "--output", str(tmp_path / "gate"), "--batch-size", "64",
        "--total-epochs", str(total_epochs), "--mnms-data-root", str(mnms),
        "--candidate-workers", "8", "--print-plan"])
    configs, commands = benchmark_maskfree_rental.configs_and_commands(args)
    assert configs["mnms"].data_root == str(mnms.resolve())
    assert all(c.batch_size == 64 and c.total_epochs == total_epochs for c in configs.values())
    command = commands[-1]
    assert command[command.index("--total-epochs") + 1] == str(total_epochs)
    assert command[command.index("--batch-size") + 1] == "64"


def test_default_likelihood_cannot_identify_a_global_foreground_name_permutation():
    """Document the identifiability limit, not a claim of actual clinical error."""
    labels = _pattern("stripes", 32)
    fitting, scoring = _views(labels)
    permutation = torch.tensor([0, 3, 2, 1], dtype=torch.long)
    first = hypothesis_from_labels(labels, "original", "diagnostic")
    renamed = hypothesis_from_labels(permutation[labels], "renamed", "diagnostic")
    model = ObservationModel()
    score1 = model.score(model.fit(first, fitting), scoring)
    score2 = model.score(model.fit(renamed, fitting), scoring)
    assert score1.available and score2.available
    assert not torch.equal(first.labels, renamed.labels)
    assert score1.count == score2.count
    assert score1.total == pytest.approx(score2.total, abs=2e-12, rel=2e-12)
