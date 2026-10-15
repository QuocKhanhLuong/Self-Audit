"""Explicit local DINO constructor and a new, incompatible checkpoint schema."""

import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path

import torch
from torch.nn import functional as F

from layer_upgrade import LayerGuidedStudent, OfficialConvNeXtStages, block_mean

from .geometry import require_digest

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = "layer_guided_convnext_tiny_v1"
CONTROL = "layer_guided_convnext_tiny_conventional_v1"
PREPROCESSING = "layer_guided.per_cine_minmax.v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def source_hashes():
    return {
        str(p.relative_to(ROOT)): sha256(p)
        for folder in ("layer_upgrade", "integration")
        for p in sorted((ROOT / folder).glob("*.py"))
    }


def validate_recipe(recipe):
    expected = {
        "schema",
        "architecture",
        "seed",
        "detail_bound",
        "preprocessing",
        "guard",
        "refiner",
        "semantic_admission",
    }
    if set(recipe) != expected or recipe["schema"] != "layer_guided.recipe.v1":
        raise ValueError("explicit layer-guided recipe schema required")
    if recipe["architecture"] not in (ARCHITECTURE, CONTROL):
        raise ValueError(
            "incompatible architecture; historical GuidanceStudent is not loadable"
        )
    if type(recipe["seed"]) is not int or not 0 <= recipe["seed"] < 2**32:
        raise ValueError("nonnegative 32-bit initialization seed required")
    if recipe["detail_bound"] != 1.0 or recipe["preprocessing"] != PREPROCESSING:
        raise ValueError(
            "v1 preserves the reference bound and declared per-cine transform"
        )
    if (
        recipe["guard"] != "off"
        or recipe["refiner"] != "proposal_only"
        or recipe["semantic_admission"] != "off"
    ):
        raise ValueError(
            "guard experiment and semantic admission require a separately specified recipe"
        )
    return dict(recipe)


def verify_backbone(approval):
    """Verify an externally reviewed local receipt before executing hub code."""
    if (
        approval.get("schema") != "layer_guided.backbone_approval.v1"
        or approval.get("approved") is not True
    ):
        raise ValueError(
            "BLOCKED_PRETRAINED: reviewed local backbone approval required"
        )
    if (
        approval.get("model") != "dinov3_convnext_tiny"
        or approval.get("lineage") != "official_self_supervised_dinov3"
    ):
        raise ValueError("BLOCKED_PRETRAINED: unsupported backbone lineage")
    if (
        not isinstance(approval.get("review_record"), str)
        or not approval["review_record"].strip()
    ):
        raise ValueError("BLOCKED_PRETRAINED: external review record required")
    expected = require_digest(approval.get("weights_sha256"))
    commit = approval.get("source_commit", "")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("BLOCKED_PRETRAINED: reviewed source commit required")
    repo, weights = (
        Path(approval["repo"]).resolve(),
        Path(approval["weights"]).resolve(),
    )
    if not (repo / "hubconf.py").is_file() or not weights.is_file():
        raise FileNotFoundError(
            "BLOCKED_PRETRAINED: local source and approved weights are required"
        )
    if sha256(weights) != expected:
        raise ValueError("BLOCKED_PRETRAINED: checkpoint digest mismatch")

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True
        ).strip()

    if git("rev-parse", "HEAD") != commit or git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise ValueError(
            "BLOCKED_PRETRAINED: source must match a clean reviewed checkout"
        )
    if git("remote", "get-url", "origin").removesuffix(".git") not in {
        "https://github.com/facebookresearch/dinov3",
        "git@github.com:facebookresearch/dinov3",
    }:
        raise ValueError("BLOCKED_PRETRAINED: official source origin required")
    return {
        **approval,
        "repo": str(repo),
        "weights": str(weights),
        "verified_weights_sha256": expected,
        "source_commit": commit,
        "upstream_lineage": "declared_self_supervised_not_independently_audited",
    }


class ConventionalHeadStudent(LayerGuidedStudent):
    """Matched-parameter control: bilinear coarse logits plus unprojected detail.

    Coarse supervision applies D to this native head, so it can affect detail.
    The reference split composition is computed and discarded: disclose its
    overhead when comparing timings, rather than claiming equal FLOPs.
    """

    def forward(self, images):
        out = super().forward(images)
        native = (
            F.interpolate(
                out["coarse_logits"],
                out["logits_padded"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            + out["bounded_detail"]
        )
        out["logits_padded"] = native
        out["logits"] = native[..., : images.shape[-2], : images.shape[-1]]
        out["coarse_logits"] = block_mean(native)
        return out


def construct_student(recipe, approval, device="cpu"):
    recipe = validate_recipe(recipe)
    provenance = verify_backbone(approval)
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda") or (
        selected.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("requested CPU/CUDA runtime unavailable; no device fallback")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(recipe["seed"])
        backbone = torch.hub.load(
            provenance["repo"],
            "dinov3_convnext_tiny",
            source="local",
            pretrained=True,
            weights=provenance["weights"],
        )
        encoder = OfficialConvNeXtStages(backbone)
        if encoder.channels != (96, 192, 384, 768):
            raise ValueError("unexpected official ConvNeXt-Tiny stage channels")
        cls = (
            LayerGuidedStudent
            if recipe["architecture"] == ARCHITECTURE
            else ConventionalHeadStudent
        )
        model = cls(encoder, detail_bound=recipe["detail_bound"]).to(selected).eval()
    # Verify raw native stages at the actual selected device before returning.
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 32, 32, device=selected))
    if not all(torch.isfinite(x).all() for x in out.values()):
        raise ValueError("nonfinite official-backbone native-stage preflight")
    if sha256(provenance["weights"]) != provenance["verified_weights_sha256"]:
        raise ValueError("backbone weights changed during load")
    model.integration_identity = {
        "recipe": recipe,
        "backbone": provenance,
        "sources": source_hashes(),
    }
    return model, provenance


def validate_lineage(lineage):
    require_digest(lineage.get("bank_manifest_sha256"))
    teacher = lineage.get("teacher", {})
    require_digest(teacher.get("checkpoint_sha256"))
    if teacher.get("frozen") is not True or not teacher.get("source"):
        raise ValueError("frozen teacher source provenance required")
    splits = lineage.get("locked_splits", {})
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("three explicit disjoint patient lists required")
    seen = set()
    for ids in splits.values():
        if not isinstance(ids, list) or any(
            not isinstance(pid, str) or not pid for pid in ids
        ):
            raise ValueError("patient lists required")
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("duplicate or overlapping patient identities")
        seen.update(ids)
    patients = lineage.get("train_patients")
    if (
        not isinstance(patients, list)
        or not patients
        or len(patients) != len(set(patients))
        or not set(patients) <= set(splits["train"])
    ):
        raise ValueError("checkpoint lineage must contain TRAIN patients only")
    if (
        type(lineage.get("optimizer_steps")) is not int
        or lineage["optimizer_steps"] < 0
    ):
        raise ValueError("actual optimizer step count required")


def save_checkpoint(model, out, recipe, lineage):
    """Freeze an explicitly initialized/trained experimental model, never old v1."""
    recipe = validate_recipe(recipe)
    validate_lineage(lineage)
    identity = getattr(model, "integration_identity", None)
    if (
        not identity
        or identity["recipe"] != recipe
        or identity["sources"] != source_hashes()
    ):
        raise ValueError(
            "model must come from the approved constructor at this source revision"
        )
    metadata = {
        "schema": "layer_guided.checkpoint.v1",
        **identity,
        "lineage": lineage,
        "preprocessing": PREPROCESSING,
        "geometry": "native_pad32_coarse4_feature16_pixel_edges",
        "semantic_admission": "off",
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_runtime": torch.version.cuda,
        },
        "status": "INITIALIZED_ONLY"
        if lineage["optimizer_steps"] == 0
        else "REPRESENTATION_ONLY",
    }
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    state = model.state_dict()
    if any(not torch.isfinite(t).all() for t in state.values()):
        raise ValueError("cannot freeze nonfinite model state")
    validate_fixed_buffers(state, recipe)
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=False)
    torch.save({"metadata": metadata, "state_dict": state}, folder / "model.pt")
    manifest = {
        "metadata": metadata,
        "model_sha256": sha256(folder / "model.pt"),
        "recipe_sha256": canonical_hash(recipe),
    }
    (folder / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return sha256(folder / "MANIFEST.json")


def validate_fixed_buffers(state, recipe):
    expected = {
        "detail_bound": torch.tensor(recipe["detail_bound"]),
        "input_adapter.mean": torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        "input_adapter.std": torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
    }
    for name, value in expected.items():
        if name not in state or not torch.equal(state[name].detach().cpu(), value):
            raise ValueError(f"fixed preprocessing/detail buffer changed: {name}")


def load_checkpoint(folder, expected_manifest_sha256, approval, device="cpu"):
    folder = Path(folder)
    require_digest(expected_manifest_sha256)
    if not (folder / "MANIFEST.json").is_file():
        raise ValueError(
            "incompatible artifact: historical tiny checkpoints are not layer-guided checkpoints"
        )
    if {p.name for p in folder.iterdir()} != {"model.pt", "MANIFEST.json"} or any(
        p.is_symlink() for p in folder.iterdir()
    ):
        raise ValueError("unexpected checkpoint inventory")
    if sha256(folder / "MANIFEST.json") != expected_manifest_sha256:
        raise ValueError("checkpoint manifest digest mismatch")
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    metadata = manifest["metadata"]
    if metadata.get("schema") != "layer_guided.checkpoint.v1":
        raise ValueError("incompatible checkpoint schema")
    validate_recipe(metadata["recipe"])
    validate_lineage(metadata["lineage"])
    expected_status = (
        "INITIALIZED_ONLY"
        if metadata["lineage"]["optimizer_steps"] == 0
        else "REPRESENTATION_ONLY"
    )
    if (
        metadata["status"] != expected_status
        or metadata["preprocessing"] != PREPROCESSING
        or metadata["geometry"] != "native_pad32_coarse4_feature16_pixel_edges"
    ):
        raise ValueError("checkpoint status/preprocessing/geometry mismatch")
    if (
        metadata["sources"] != source_hashes()
        or metadata["semantic_admission"] != "off"
    ):
        raise ValueError("checkpoint source/admission mismatch")
    if (
        canonical_hash(metadata["recipe"]) != manifest["recipe_sha256"]
        or sha256(folder / "model.pt") != manifest["model_sha256"]
    ):
        raise ValueError("checkpoint payload/recipe digest mismatch")
    payload = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    if payload.get("metadata") != metadata:
        raise ValueError("checkpoint payload lineage mismatch")
    validate_fixed_buffers(payload["state_dict"], metadata["recipe"])
    model, provenance = construct_student(metadata["recipe"], approval, device)
    if provenance != metadata["backbone"]:
        raise ValueError("checkpoint backbone provenance mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError("nonfinite checkpoint state")
    return model.eval(), metadata
