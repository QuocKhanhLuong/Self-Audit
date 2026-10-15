import json
from dataclasses import replace

import pytest
import torch
from test_parity import TinyEncoder
from torch import nn

from integration import artifacts as art
from integration.geometry import (
    GridGeometry,
    TeacherRegistration,
    coarse_evidence,
    coarse_semantic_loss,
    masked_feature_objective,
    normalize_cine,
    slice_contexts,
    transport_density,
)
from integration.steps import guarded_decoder_step, image_proposal, witness_plan
from layer_upgrade import FrozenImageEnergy, LayerGuidedStudent


def geometry(hw=(47, 53), teacher=(7, 9)):
    affine = tuple(tuple(float(i == j) for j in range(4)) for i in range(4))
    return GridGeometry(
        hw,
        teacher,
        "1" * 64,
        affine,
        "unknown",
        TeacherRegistration(
            hw, teacher, "1" * 64, affine, "6" * 64, (("synthetic_fixture", "7" * 64),)
        ),
    )


def source(g=None):
    return {
        "kind": "conditional_z_mass",
        "manifest_sha256": (g or geometry()).registration.source_manifest_sha256,
        "orientation_verified": True,
        "ontology": ["BG", "RV", "MYO", "LV"],
        "registration_sha256": (g or geometry()).registration.sha256,
    }


@pytest.mark.parametrize("hw", [(32, 32), (47, 53), (64, 96), (65, 97)])
def test_mass_support_transport_conserves_native_area_and_ignores_padding(hw):
    g = geometry(hw)
    mass = torch.rand(2, 4, *g.teacher_hw, dtype=torch.double)
    support = mass.sum(1)
    mapped = transport_density(mass, g)
    expected = mass.sum((-2, -1)) * (hw[0] / 7) * (hw[1] / 9)
    torch.testing.assert_close(
        mapped.sum((-2, -1)) * 16, expected, rtol=1e-12, atol=1e-12
    )
    target, weights, receipt = coarse_evidence(mass, support, g, source=source(g))
    torch.testing.assert_close(mapped.sum(1), weights)
    assert receipt["geometry"]["spatial_unit"] == "unknown"
    assert (weights[..., (hw[0] + 3) // 4 :, :] == 0).all()
    assert (weights[..., :, (hw[1] + 3) // 4 :] == 0).all()
    empty_target, empty_weight, _ = coarse_evidence(
        torch.zeros_like(mass), torch.zeros_like(support), g, source=source(g)
    )
    assert empty_weight.sum() == 0 and (empty_target == 0.25).all()
    logits = torch.randn_like(target, requires_grad=True)
    coarse_semantic_loss(
        {"coarse_logits": logits}, empty_target, empty_weight
    ).backward()
    assert logits.grad.abs().sum() == 0


def test_partial_border_support_and_invalid_semantic_sources():
    g = geometry((47, 53))
    mass = torch.full((1, 4, 7, 9), 0.25, dtype=torch.double)
    support = mass.sum(1)
    _, weights, _ = coarse_evidence(mass, support, g, source=source())
    assert weights[0, 11, 13] == pytest.approx(3 / 16)
    for bad in (
        {**source(), "orientation_verified": False},
        {**source(), "kind": "invert_Q"},
    ):
        with pytest.raises(ValueError):
            coarse_evidence(mass, support, g, source=bad)
    with pytest.raises(ValueError):
        transport_density(torch.zeros_like(mass, dtype=torch.long), g)
    with pytest.raises(ValueError):
        coarse_evidence(mass, support + 1, g, source=source())


def test_teacher_source_registration_rejects_mirrors_other_images_and_unknown_axes():
    g = geometry()
    mirrored = tuple(
        tuple(-v if i == 0 else v for v in row) for i, row in enumerate(g.affine_xyzt)
    )
    with pytest.raises(ValueError, match="registration mismatch"):
        replace(g, affine_xyzt=mirrored)
    with pytest.raises(ValueError, match="registration mismatch"):
        replace(g, registration=replace(g.registration, image_sha256="0" * 64))
    with pytest.raises(ValueError, match="source axes"):
        replace(g.registration, source_axis_order="TZXYC")
    mass = torch.full((1, 4, *g.teacher_hw), 0.25)
    with pytest.raises(ValueError, match="source registration"):
        coarse_evidence(
            mass, mass.sum(1), g, source={**source(g), "registration_sha256": "0" * 64}
        )
    with pytest.raises(ValueError, match="source manifest"):
        coarse_evidence(
            mass, mass.sum(1), g, source={**source(g), "manifest_sha256": "0" * 64}
        )


def test_corrupt_fixed_buffers_rejected_before_saving(tmp_path, approved_fixture):
    recipe, approval, _ = approved_fixture
    model, _ = art.construct_student(recipe, approval)
    model.input_adapter.std.zero_()
    with pytest.raises(ValueError, match="fixed preprocessing"):
        art.save_checkpoint(model, tmp_path / "invalid", recipe, lineage())
    assert not (tmp_path / "invalid").exists()


def test_one_cine_normalization_preserves_neighbor_intensity_relationships():
    raw = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float() - 50
    normalized, receipt = normalize_cine(raw, "1" * 64)
    assert normalized.min() == 0 and normalized.max() == 1
    assert receipt["minimum"] == -50 and receipt["maximum"] == 69
    batch = slice_contexts(normalized, [(0, 1), (1, 0)])
    assert torch.equal(batch[0, 0], normalized[0, 0])
    assert torch.equal(batch[1, 0], batch[1, 1])
    flat, _ = normalize_cine(torch.ones_like(raw), "1" * 64)
    assert flat.sum() == 0
    with pytest.raises(ValueError):
        slice_contexts(normalized, [(0, 3)])


def test_masked_features_see_aligned_mask_and_never_teacher_in_forward():
    model = LayerGuidedStudent(TinyEncoder(), decoder_channels=(16, 24, 32, 40)).eval()
    g = geometry()
    images = torch.rand(1, 3, *g.native_hw)
    teacher = torch.rand(1, 64, *g.teacher_hw, requires_grad=True)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    mask[0, 1, 1] = True
    seen = []
    hook = model.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].detach().clone())
    )
    loss, receipt = masked_feature_objective(model, images, teacher, g, mask)
    hook.remove()
    assert (seen[0][..., 16:32, 16:32] == 0).all()
    assert torch.equal(seen[0][..., :16, :16], images[..., :16, :16])
    loss.backward()
    assert teacher.grad is None and model.feature_project.weight.grad.norm() > 0
    assert receipt["masked_pixels"] == 256 and receipt["teacher_in_forward"] is False
    mask[0, -1, -1] = True
    with pytest.raises(ValueError, match="fully native"):
        masked_feature_objective(model, images, teacher, g, mask)


class OfficialFixture(nn.Module):
    """Manufactured stage API, explicitly not official DINO weights."""

    embed_dims = (96, 192, 384, 768)

    def __init__(self):
        super().__init__()
        channels = (3, *self.embed_dims)
        self.downsample_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    4 if i == 0 else 2,
                    stride=4 if i == 0 else 2,
                )
                for i in range(4)
            ]
        )
        self.stages = nn.ModuleList([nn.Identity() for _ in range(4)])


@pytest.fixture
def approved_fixture(tmp_path, monkeypatch):
    repo = tmp_path / "fake_official_source"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# manufactured fixture, not upstream code\n")
    weights = tmp_path / "manufactured_weights.pth"
    weights.write_bytes(b"explicit mock fixture, not a pretrained model")
    approval = {
        "schema": "layer_guided.backbone_approval.v1",
        "approved": True,
        "model": "dinov3_convnext_tiny",
        "lineage": "official_self_supervised_dinov3",
        "repo": str(repo),
        "weights": str(weights),
        "weights_sha256": art.sha256(weights),
        "source_commit": "a" * 40,
        "review_record": "TEST_FIXTURE_ONLY",
    }

    def git(command, **kwargs):
        if "rev-parse" in command:
            return "a" * 40 + "\n"
        if "status" in command:
            return ""
        return "https://github.com/facebookresearch/dinov3.git\n"

    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs == {
            "source": "local",
            "pretrained": True,
            "weights": str(weights),
        }
        return OfficialFixture()

    monkeypatch.setattr(art.subprocess, "check_output", git)
    monkeypatch.setattr(torch.hub, "load", load)
    recipe = json.loads((art.ROOT / "recipe.json").read_text())
    return recipe, approval, calls


def lineage():
    return {
        "bank_manifest_sha256": "3" * 64,
        "teacher": {
            "frozen": True,
            "checkpoint_sha256": "4" * 64,
            "source": "SYNTHETIC_TEST",
        },
        "train_patients": ["TRAIN_A"],
        "locked_splits": {"train": ["TRAIN_A"], "val": ["VAL_B"], "test": []},
        "optimizer_steps": 0,
    }


def test_checkpoint_roundtrip_and_legacy_rejection(tmp_path, approved_fixture):
    recipe, approval, calls = approved_fixture
    model, _ = art.construct_student(recipe, approval)
    images = torch.rand(1, 3, 47, 53)
    with torch.no_grad():
        expected = model(images)["logits"]
    path = tmp_path / "artifact"
    digest = art.save_checkpoint(model, path, recipe, lineage())
    restored, metadata = art.load_checkpoint(path, digest, approval)
    with torch.no_grad():
        torch.testing.assert_close(restored(images)["logits"], expected, atol=0, rtol=0)
    assert metadata["status"] == "INITIALIZED_ONLY" and len(calls) == 2
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    torch.save({"model": {}, "width": 32}, legacy / "last.pt")
    with pytest.raises(ValueError, match="historical tiny"):
        art.load_checkpoint(legacy, "0" * 64, approval)
    (path / "model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        art.load_checkpoint(path, digest, approval)


@pytest.mark.parametrize(
    "change", ["missing", "hash", "approval", "architecture", "guard", "source"]
)
def test_pretrained_failures_happen_before_hub_execution(
    approved_fixture, monkeypatch, change
):
    recipe, approval, calls = approved_fixture
    if change == "missing":
        approval["weights"] += ".missing"
    elif change == "hash":
        approval["weights_sha256"] = "0" * 64
    elif change == "approval":
        approval["approved"] = False
    elif change == "architecture":
        recipe["architecture"] = "GuidanceStudent"
    elif change == "guard":
        recipe["guard"] = "AdamW"
    elif change == "source":
        approval["source_commit"] = "b" * 40
    with pytest.raises((ValueError, FileNotFoundError)):
        art.construct_student(recipe, approval)
    assert not calls


def test_conventional_control_has_same_parameters_and_exposes_detail_gradient(
    approved_fixture,
):
    recipe, approval, _ = approved_fixture
    split, _ = art.construct_student(recipe, approval)
    control, _ = art.construct_student(
        {**recipe, "architecture": art.CONTROL}, approval
    )
    assert sum(p.numel() for p in split.parameters()) == sum(
        p.numel() for p in control.parameters()
    )
    control.load_state_dict(split.state_dict())
    images = torch.rand(1, 3, 32, 32)
    for model in (split, control):
        model(images)["coarse_logits"].square().mean().backward()
    assert split.detail_head.weight.grad is None
    assert control.detail_head.weight.grad.norm() > 0


def test_guard_scope_and_detached_proposals():
    model = LayerGuidedStudent(TinyEncoder(), decoder_channels=(16, 24, 32, 40)).eval()
    plan = witness_plan(
        ["TRAIN_A"],
        [
            {
                "patient_id": "TRAIN_A",
                "split": "train",
                "class_name": "RV",
                "source_manifest_sha256": "5" * 64,
                "sample_id": "t0_z0",
            }
        ],
    )
    selected = next(model.refine[0].parameters())
    old_encoder = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    objective = lambda: selected.square().mean()
    receipt = guarded_decoder_step(
        model, objective, lambda: [objective()], plan=plan, step_size=0.01
    )
    assert receipt["accepted"] and receipt["actual_displacement_norm"] > 0
    assert receipt["optimizer"] == "additive_sgd_no_state_no_decay"
    assert all(
        torch.equal(v, old_encoder[k]) for k, v in model.encoder.state_dict().items()
    )
    assert all(p.grad is None for p in model.parameters())
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        guarded_decoder_step(
            model, objective, lambda: [objective()], plan=plan, step_size=0.01
        )
    with pytest.raises(ValueError, match="all selected"):
        witness_plan(["TRAIN_A", "TRAIN_B"], plan["witnesses"])
    energy = FrozenImageEnergy(
        torch.tensor([[0.0, -5.0, 0.0, 0.0]]),
        torch.empty(2, 0, dtype=torch.long),
        torch.empty(0),
        torch.full((1, 4), 0.25),
        torch.zeros(1),
        lam=0,
        beta=0,
    )
    proposal, report = image_proposal(
        torch.tensor([[3.0, 0.0, 0.0, 0.0]], requires_grad=True), energy
    )
    assert not proposal.requires_grad and report["semantic_admission"] is False
    assert proposal.argmax(1).item() == 1
