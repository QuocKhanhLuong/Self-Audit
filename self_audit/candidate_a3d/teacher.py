"""A-3D dual-branch teacher with an explicit native 3-D contract.

The production backbone adapter is deliberately hash-pinned and local-only.
``tiny_test3d`` is the only manufactured implementation used by the smoke
tests.  A MedNeXt-L k3 model is created through the official local
``nnunet_mednext.create_mednext_v1`` factory when a verified checkout is
provided.  The adapter uses the verified upstream v1 module sequence directly;
it refuses to treat logits as features, so the boundary branch gets a real
high-resolution path while the original MedNeXt output remains the region
branch.

All tensors use ``[B,C,D,H,W]``.  The public teacher output is
``region_logits``, ``boundary_logits`` and normalized ``sdf`` with shape
``[B,4,D,H,W]``.  Padding is internal and outputs are cropped back to the
original volume; the caller owns the validated native geometry.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import itertools
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils import checkpoint as torch_checkpoint

from ..losses import supervised_loss

CLASS_NAMES = ("BG", "RV", "MYO", "LV")
UPSTREAM_URL = "https://github.com/MIC-DKFZ/MedNeXt"
UPSTREAM_MAIN_SHA = "0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba"
UPSTREAM_VERIFIED_FILES = (
    "nnunet_mednext/network_architecture/mednextv1/MedNextV1.py",
    "nnunet_mednext/network_architecture/mednextv1/blocks.py",
    "nnunet_mednext/network_architecture/mednextv1/create_mednext_v1.py",
)


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        # Keep at least two values per normalization group so a native volume
        # that pads down to 1x1x1 remains trainable on CPU smoke fixtures.
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


def _finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _class_mapping(value: Any, *, field: str = "class_mapping") -> tuple[str, ...]:
    if isinstance(value, Mapping):
        try:
            value = [value[str(i)] if str(i) in value else value[i] for i in range(4)]
        except (KeyError, TypeError):
            raise ValueError(f"{field} must contain class IDs 0..3") from None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field} must be an ordered four-class mapping")
    names = tuple(str(item).strip().upper() for item in value)
    if names != CLASS_NAMES:
        raise ValueError(f"{field} must use fixed order {list(CLASS_NAMES)}")
    return names


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"backbone source is not a readable git checkout: {path}") from exc


def _git_dirty(path: Path) -> bool:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
            text=True, stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot inspect backbone source: {path}") from exc
    return bool(output.strip())


def _under(path: str | Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _geometry_spacing(geometry: Any, shape: Sequence[int], *, device: torch.device,
                      dtype: torch.dtype) -> Tensor:
    """Validate public native geometry and return ``[B,3]`` spacing in mm."""

    if geometry is None:
        raise ValueError("A-3D teacher requires explicit runtime geometry")
    if not isinstance(geometry, Mapping) or set(geometry) != {
        "shape", "spacing_zyx", "affine_zyx", "context",
    }:
        raise ValueError(
            "runtime geometry must be the explicit mapping "
            "{shape, spacing_zyx, affine_zyx, context}"
        )
    declared_shape = tuple(geometry["shape"])
    spacing = geometry["spacing_zyx"]
    affine = torch.as_tensor(geometry["affine_zyx"], dtype=torch.float64)
    if affine.shape != (4, 4) or not torch.isfinite(affine).all():
        raise ValueError("runtime geometry affine_zyx must be a finite 4x4 matrix")
    if geometry["context"] != "full_volume":
        raise ValueError("A-3D teacher requires context=full_volume")
    if len(declared_shape) != 3 or any(
        isinstance(value, bool) or int(value) != value for value in declared_shape
    ):
        raise ValueError("runtime geometry shape must contain three integer dimensions")
    if tuple(int(value) for value in declared_shape) != tuple(int(value) for value in shape):
        raise ValueError("runtime geometry shape must equal the native input volume")
    spacing_tensor = torch.as_tensor(spacing, device=device, dtype=dtype)
    if spacing_tensor.ndim == 1:
        if spacing_tensor.numel() != 3:
            raise ValueError("spacing_zyx must contain dz,dy,dx in millimetres")
        spacing_tensor = spacing_tensor[None]
    if spacing_tensor.ndim != 2 or spacing_tensor.shape[1] != 3:
        raise ValueError("spacing_zyx must have shape [3] or [B,3]")
    if not torch.isfinite(spacing_tensor).all() or (spacing_tensor <= 0).any():
        raise ValueError("spacing_zyx must be finite and positive in millimetres")
    return spacing_tensor


def _segmentation_checkpoint_payload(payload: Any) -> tuple[Mapping[str, Tensor], dict[str, Any]]:
    """Extract a state dictionary and mandatory TRAIN provenance.

    A raw tensor dictionary is intentionally rejected.  A segmentation
    checkpoint without the labeled/U cohort and hash cannot be shown to fit
    the current dense TRAIN budget, even if its output head has four classes.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("MedNeXt checkpoint must be a mapping with state_dict and metadata")  # noqa: TRY004 - fail-closed checkpoint API
    state = payload.get("state_dict", payload.get("model_state_dict"))
    metadata = payload.get("metadata", payload.get("provenance"))
    if not isinstance(state, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError(  # noqa: TRY004 - fail-closed checkpoint API
            "MedNeXt checkpoint must contain state_dict and metadata with TRAIN provenance; "
            "raw/unknown-cohort weights are rejected"
        )
    required = {"labeled_ids", "unlabeled_ids", "label_budget", "data_sha256", "class_mapping"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"MedNeXt checkpoint metadata missing {missing}")
    if not isinstance(metadata["labeled_ids"], (list, tuple)) or not isinstance(
        metadata["unlabeled_ids"], (list, tuple)
    ):
        raise ValueError("MedNeXt checkpoint TRAIN IDs must be lists")  # noqa: TRY004 - fail-closed checkpoint API
    labeled = [str(value) for value in metadata["labeled_ids"]]
    unlabeled = [str(value) for value in metadata["unlabeled_ids"]]
    if len(set(labeled)) != len(labeled) or len(set(unlabeled)) != len(unlabeled):
        raise ValueError("MedNeXt checkpoint TRAIN IDs must be unique")
    if set(labeled) & set(unlabeled):
        raise ValueError("MedNeXt checkpoint labeled/U IDs overlap")
    if type(metadata["label_budget"]) is not int or metadata["label_budget"] != len(labeled):
        raise ValueError("MedNeXt checkpoint label_budget does not match labeled_ids")
    if (
        not isinstance(metadata["data_sha256"], str)
        or len(metadata["data_sha256"]) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in metadata["data_sha256"])
    ):
        raise ValueError("MedNeXt checkpoint data_sha256 must be a 64-character hash")
    mapping = _class_mapping(metadata["class_mapping"], field="MedNeXt checkpoint class_mapping")
    normalized = dict(metadata)
    normalized.update(labeled_ids=labeled, unlabeled_ids=unlabeled,
                      label_budget=int(metadata["label_budget"]),
                      data_sha256=metadata["data_sha256"], class_mapping=list(mapping))
    return state, normalized


class _Residual3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x + self.body(x))


class TinyEncoder3D(nn.Module):
    """Small deterministic 3-D encoder for synthetic contracts only."""

    def __init__(self, in_channels: int = 1, widths: Sequence[int] = (8, 16, 24, 32),
                 num_classes: int = 4) -> None:
        super().__init__()
        widths = tuple(int(width) for width in widths)
        if len(widths) != 4 or any(width <= 0 for width in widths):
            raise ValueError("tiny 3-D encoder needs four positive widths")
        self.channels = widths
        self.num_classes = int(num_classes)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, widths[0], 3, padding=1, bias=False),
            nn.GroupNorm(_groups(widths[0]), widths[0]), nn.GELU(),
            _Residual3D(widths[0]),
        )
        self.down = nn.ModuleList()
        for before, after in itertools.pairwise(widths):
            self.down.append(nn.Sequential(
                nn.Conv3d(before, after, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(_groups(after), after), nn.GELU(), _Residual3D(after),
            ))
        self.region_head = nn.Conv3d(widths[0], self.num_classes, 1)

    def forward_features(self, x: Tensor) -> tuple[Tensor, ...]:
        feature = self.stem(x)
        features = [feature]
        for layer in self.down:
            feature = layer(feature)
            features.append(feature)
        return tuple(features)

    def forward_region(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        # This is the complete tiny reference segmentation path.  It remains
        # independent of the boundary branch and is never replaced by a head
        # on the final feature alone.
        feature = features[-1]
        for skip in reversed(features[:-1]):
            feature = F.interpolate(feature, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            feature = feature[:, : skip.shape[1]] + skip
        return self.region_head(feature)


class MedNeXtL3DAdapter(nn.Module):
    """Hash-pinned adapter around a local official MedNeXt-L k3 checkout.

    The upstream full model remains the region path.  Boundary processing is
    possible only through the validated upstream v1 module sequence; silently
    using segmentation logits as boundary features is rejected because that
    would not be a separate high-resolution branch.
    """

    def __init__(self, model: nn.Module, *, source_repo: Path | str, source_commit: str,
                 weights_sha256: str | None, features_api: str,
                 feature_channels: Sequence[int] | None = None) -> None:
        super().__init__()
        self.model = model
        self.source_repo = str(source_repo)
        self.source_commit = source_commit
        self.weights_sha256 = weights_sha256
        self.features_api = features_api
        channels = getattr(model, "feature_channels", None) or feature_channels
        if channels is None:
            channels = getattr(model, "channels", None)
        if channels is None:
            raise ValueError("MedNeXt adapter must declare feature_channels/channels")
        self.channels = tuple(int(value) for value in channels)
        if len(self.channels) < 2 or any(value <= 0 for value in self.channels):
            raise ValueError("MedNeXt feature channels must be positive")

    def _native_forward_with_features(self, x: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        """Run the official MedNeXt-v1 module sequence once.

        The upstream v1 model does not expose an encoder method.  Its public
        implementation is a fixed stem, four encoder/down blocks, bottleneck,
        four up/decoder blocks and ``out_0``.  Reusing those exact modules lets
        the boundary branch see true high-resolution features while retaining
        the original full segmentation route for the region output.
        """

        model = self.model
        if not getattr(model, "_candidate_a_native_v1", False):
            raise RuntimeError("native MedNeXt forward sequence was not validated")

        # The official L factory enables ``outside_block`` activation
        # checkpointing.  Calling the blocks directly would be numerically
        # equivalent but would silently discard the configured memory contract
        # during training.  Keep the upstream dispatch (including its dummy
        # tensor argument and checkpoint implementation) for every block.
        def apply_block(block: nn.Module, value: Tensor, *, sequential: bool = False) -> Tensor:
            if not getattr(model, "outside_block_checkpointing", False):
                return block(value)
            dummy = getattr(model, "dummy_tensor", None)
            if dummy is None:
                raise RuntimeError("verified MedNeXt checkpoint path requires dummy_tensor")
            if sequential:
                iterative = getattr(model, "iterative_checkpoint", None)
                if not callable(iterative):
                    raise RuntimeError("verified MedNeXt checkpoint path requires iterative_checkpoint")
                return iterative(block, value)
            return torch_checkpoint.checkpoint(block, value, dummy)

        x = model.stem(x)
        x_res_0 = apply_block(model.enc_block_0, x, sequential=True)
        x = apply_block(model.down_0, x_res_0)
        x_res_1 = apply_block(model.enc_block_1, x, sequential=True)
        x = apply_block(model.down_1, x_res_1)
        x_res_2 = apply_block(model.enc_block_2, x, sequential=True)
        x = apply_block(model.down_2, x_res_2)
        x_res_3 = apply_block(model.enc_block_3, x, sequential=True)
        x = apply_block(model.down_3, x_res_3)
        x = apply_block(model.bottleneck, x, sequential=True)
        features = (x_res_0, x_res_1, x_res_2, x_res_3, x)
        x = apply_block(model.up_3, x)
        x = apply_block(model.dec_block_3, x_res_3 + x, sequential=True)
        x = apply_block(model.up_2, x)
        x = apply_block(model.dec_block_2, x_res_2 + x, sequential=True)
        x = apply_block(model.up_1, x)
        x = apply_block(model.dec_block_1, x_res_1 + x, sequential=True)
        x = apply_block(model.up_0, x)
        x = apply_block(model.dec_block_0, x_res_0 + x, sequential=True)
        return features, apply_block(model.out_0, x)

    def forward_features(self, x: Tensor) -> tuple[Tensor, ...]:
        features, _ = self._native_forward_with_features(x)
        return features

    def forward_region(self, x: Tensor) -> Tensor:
        # Keep the upstream public forward for the region-only reference.
        # ``forward_with_features`` below is used by the dual branch to avoid
        # a second trunk pass during actual training.
        output = self.model(x)
        if isinstance(output, (list, tuple)):
            if len(output) != 1:
                raise ValueError("MedNeXt deep supervision must be disabled for the teacher adapter")
            output = output[0]
        if not isinstance(output, Tensor) or output.ndim != 5:
            raise ValueError("MedNeXt full segmentation path must return [B,C,D,H,W]")
        return output

    def forward_with_features(self, x: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        return self._native_forward_with_features(x)


def _load_mednext(config: Mapping[str, Any]) -> tuple[MedNeXtL3DAdapter, dict[str, Any]]:
    required = ("source_repo", "source_commit", "weights", "weights_sha256")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"MedNeXt-L k3 config missing {missing}")
    source_repo = Path(str(config["source_repo"])).expanduser().resolve()
    if not source_repo.is_dir():
        raise FileNotFoundError(f"MedNeXt source checkout not found: {source_repo}")
    head = _git_head(source_repo)
    if str(config["source_commit"]) != UPSTREAM_MAIN_SHA:
        raise ValueError(
            f"MedNeXt adapter is verified only at upstream commit {UPSTREAM_MAIN_SHA}; "
            f"received {config['source_commit']}"
        )
    if head != str(config["source_commit"]):
        raise ValueError(f"MedNeXt source commit mismatch: expected {config['source_commit']}, got {head}")
    if _git_dirty(source_repo):
        raise ValueError("MedNeXt source checkout has tracked or untracked modifications")

    module_name = str(config.get("factory_module", "nnunet_mednext"))
    factory_name = str(config.get("factory_name", "create_mednext_v1"))
    cached = sys.modules.get(module_name)
    if cached is not None and not _under(getattr(cached, "__file__", ""), source_repo):
        raise RuntimeError(f"cached {module_name} module is outside the declared MedNeXt checkout")
    sys.path.insert(0, str(source_repo))
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "local MedNeXt checkout must expose nnunet_mednext.create_mednext_v1; "
            "no remote import or architecture fallback is allowed"
        ) from exc
    finally:
        try:
            sys.path.remove(str(source_repo))
        except ValueError:
            pass
    if not _under(getattr(module, "__file__", ""), source_repo):
        raise RuntimeError(f"imported {module_name} is outside the declared MedNeXt checkout")
    foreign_submodules = []
    prefix = module_name + "."
    for name, loaded in tuple(sys.modules.items()):
        if name.startswith(prefix) and getattr(loaded, "__file__", None) \
                and not _under(loaded.__file__, source_repo):
            foreign_submodules.append((name, loaded.__file__))
    if foreign_submodules:
        raise RuntimeError(f"MedNeXt imported foreign submodules: {foreign_submodules}")
    input_channels = int(config.get("input_channels", 1))
    try:
        model = factory(
            num_input_channels=input_channels, num_classes=4, model_id="L", kernel_size=3,
            deep_supervision=False,
        )
    except TypeError as exc:
        raise TypeError(
            "MedNeXt factory must implement create_mednext_v1(num_input_channels, "
            "num_classes, model_id='L', kernel_size=3, deep_supervision=False)"
        ) from exc
    if not isinstance(model, nn.Module):
        raise TypeError("MedNeXt factory did not return a torch module")
    weights = config.get("weights")
    expected_hash = config.get("weights_sha256")
    checkpoint_metadata = None
    if weights is None:
        if expected_hash not in (None, ""):
            raise ValueError("weights_sha256 must be empty when weights is null")
        weights_hash = None
        quality = "UNTRAINED_RANDOM_INIT"
    else:
        weight_path = Path(str(weights)).expanduser().resolve()
        if not weight_path.is_file():
            raise FileNotFoundError(f"MedNeXt checkpoint not found: {weight_path}")
        weights_hash = _sha256(weight_path)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or weights_hash != expected_hash:
            raise ValueError("MedNeXt checkpoint SHA256 mismatch")
        payload = torch.load(weight_path, map_location="cpu", weights_only=True)
        state, checkpoint_metadata = _segmentation_checkpoint_payload(payload)
        head_weight = state.get("out_0.conv_out.weight")
        if not isinstance(head_weight, Tensor) or head_weight.ndim != 5 or head_weight.shape[0] != 4:
            raise ValueError("MedNeXt checkpoint must expose a four-class out_0.conv_out.weight")
        model.load_state_dict(state, strict=True)
        quality = "LOCAL_HASH_PINNED_WEIGHTS"
    native_names = (
        "stem", "enc_block_0", "down_0", "enc_block_1", "down_1", "enc_block_2",
        "down_2", "enc_block_3", "down_3", "bottleneck", "up_3", "dec_block_3",
        "up_2", "dec_block_2", "up_1", "dec_block_1", "up_0", "dec_block_0", "out_0",
    )
    if all(hasattr(model, name) for name in native_names):
        if bool(getattr(model, "do_ds", False)):
            raise ValueError("MedNeXt deep supervision must be disabled for A-3D teacher")
        model._candidate_a_native_v1 = True
        channels = (
            int(model.stem.out_channels),
            int(model.down_0.conv3.out_channels),
            int(model.down_1.conv3.out_channels),
            int(model.down_2.conv3.out_channels),
            int(model.down_3.conv3.out_channels),
        )
        features_api = "native_mednext_v1_sequence"
    else:
        raise ValueError(
            "MedNeXt local model does not match the verified upstream v1 module "
            "sequence; refusing to use logits as a boundary representation"
        )
    adapter = MedNeXtL3DAdapter(
        model, source_repo=source_repo, source_commit=head,
        weights_sha256=weights_hash, features_api=features_api,
        feature_channels=channels,
    )
    return adapter, {
        "backbone": "MedNeXt-L-k3",
        "source_repo": str(source_repo), "source_commit": head,
        "upstream_url": UPSTREAM_URL, "reference_main_sha": UPSTREAM_MAIN_SHA,
        "weights_sha256": weights_hash, "weights_status": quality,
        "features_api": features_api, "factory": f"{module_name}.{factory_name}",
        "verified_source_files": [
            f"{UPSTREAM_URL}/blob/{UPSTREAM_MAIN_SHA}/{relative}"
            for relative in UPSTREAM_VERIFIED_FILES
        ],
        "pretraining_data": checkpoint_metadata,
    }


class _RegionPath3D(nn.Module):
    """A full-resolution region decoder for the synthetic trunk."""

    def __init__(self, channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        channels = tuple(channels)
        width = channels[0]
        self.lateral = nn.ModuleList(nn.Conv3d(channels[index], width, 1) for index in range(len(channels)))
        self.refine = nn.ModuleList(_Residual3D(width) for _ in channels)
        self.head = nn.Conv3d(width, out_channels, 1)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        feature = self.refine[-1](self.lateral[-1](features[-1]))
        for index in range(len(features) - 2, -1, -1):
            skip = self.lateral[index](features[index])
            feature = F.interpolate(feature, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            feature = self.refine[index](feature + skip)
        return self.head(feature)


class _BoundaryPath3D(nn.Module):
    """Separate high-resolution/detail and broad-context boundary processing."""

    def __init__(self, in_channels: int, deep_channels: int, width: int, out_channels: int) -> None:
        super().__init__()
        self.detail = nn.Sequential(
            nn.Conv3d(in_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width), nn.GELU(), _Residual3D(width),
        )
        self.context = nn.Sequential(
            nn.Conv3d(deep_channels, width, 1, bias=False),
            nn.GroupNorm(_groups(width), width), nn.GELU(),
        )
        self.spacing_context = nn.Sequential(
            nn.Linear(3, width), nn.LayerNorm(width), nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width), nn.GELU(), _Residual3D(width),
        )
        self.seg_head = nn.Conv3d(width, out_channels, 1)
        self.sdf_head = nn.Conv3d(width, out_channels, 1)

    def forward(self, high: Tensor, deep: Tensor, spacing_zyx: Tensor) -> tuple[Tensor, Tensor]:
        detail = self.detail(high)
        context = self.context(deep)
        if spacing_zyx.ndim != 2 or spacing_zyx.shape[0] != detail.shape[0] or spacing_zyx.shape[1] != 3:
            raise ValueError("boundary spacing conditioning must have shape [B,3]")
        _finite("boundary spacing", spacing_zyx)
        if (spacing_zyx <= 0).any():
            raise ValueError("boundary spacing must be positive")
        # Physical spacing is part of the boundary/SDF input.  Log-mm keeps
        # anisotropic values numerically bounded while preserving scale.
        spacing_dtype = next(self.spacing_context.parameters()).dtype
        spacing_token = self.spacing_context(torch.log(spacing_zyx).to(spacing_dtype)).to(context.dtype)
        context = context + spacing_token[..., None, None, None]
        context = F.interpolate(context, size=detail.shape[-3:], mode="trilinear", align_corners=False)
        fused = self.fuse(detail + context)
        return self.seg_head(fused), torch.tanh(self.sdf_head(fused))


class DualBranchTeacher3D(nn.Module):
    """Shared 3-D trunk with an original region path and a separate boundary path."""

    def __init__(self, encoder: nn.Module, *, input_channels: int = 1,
                 num_classes: int = 4, class_mapping: Sequence[str] = CLASS_NAMES,
                 boundary_width: int | None = None, backbone_meta: Mapping[str, Any] | None = None,
                 checkpoint_boundary: bool = False) -> None:
        super().__init__()
        if num_classes != 4:
            raise ValueError("A-3D requires exactly four classes")
        self.encoder = encoder
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.class_mapping = _class_mapping(class_mapping)
        self.backbone_meta = copy.deepcopy(dict(backbone_meta or {}))
        if type(checkpoint_boundary) is not bool:
            raise ValueError("checkpoint_boundary must be a boolean")
        self.checkpoint_boundary = checkpoint_boundary
        self._frozen_modules: tuple[str, ...] = ()
        self._stage = "teacher_pretrain"
        # A real MedNeXt adapter owns its complete original segmentation
        # decoder.  The tiny fixture has an explicit local region path so the
        # synthetic contract does not pretend to reproduce an upstream model.
        self._external_region_path = callable(getattr(encoder, "forward_region", None)) \
            and not isinstance(encoder, TinyEncoder3D)
        channels = tuple(int(value) for value in getattr(encoder, "channels", ()))
        if len(channels) < 2:
            raise ValueError("A-3D encoder must expose at least two feature stages")
        self.encoder_channels = channels
        width = int(boundary_width or channels[0])
        if width <= 0:
            raise ValueError("boundary_width must be positive")
        # The verified MedNeXt adapter already owns the complete original
        # segmentation decoder.  Do not instantiate an unused synthetic
        # decoder beside it; retain a parameter-free stage handle so the
        # warm-up/freeze API is identical for tiny and production paths.
        self.region_path = (
            nn.Identity() if self._external_region_path else _RegionPath3D(channels, num_classes)
        )
        self.padding_multiple = 2 ** (len(channels) - 1)
        self.padding_policy = (
            "external_minimum_bottleneck_2x2x2"
            if self._external_region_path else "stride_multiple"
        )
        self.boundary_path = _BoundaryPath3D(channels[0], channels[-1], width, num_classes)

    def _check_input(self, x: Tensor) -> tuple[int, int, int]:
        if x.ndim != 5 or x.shape[1] != self.input_channels:
            raise ValueError(f"teacher expects [B,{self.input_channels},D,H,W] input")
        if any(int(dim) <= 0 for dim in x.shape[2:]):
            raise ValueError("teacher volume dimensions must be positive")
        _finite("teacher input", x)
        return tuple(int(dim) for dim in x.shape[2:])

    def _pad(self, x: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        shape = tuple(int(dim) for dim in x.shape[-3:])
        multiple = self.padding_multiple
        minimum = multiple * 2 if self._external_region_path else multiple
        target = tuple(
            max(minimum, ((dim + multiple - 1) // multiple) * multiple)
            for dim in shape
        )
        padding = tuple(target_dim - dim for target_dim, dim in zip(target, shape))
        # F.pad takes W,H,D pairs.
        padded = F.pad(x, (0, padding[2], 0, padding[1], 0, padding[0]), mode="replicate")
        return padded, shape

    def forward(self, x: Tensor, *, geometry: Any) -> dict[str, Tensor]:
        original_shape = self._check_input(x)
        spacing_zyx = _geometry_spacing(
            geometry, original_shape, device=x.device, dtype=x.dtype,
        )
        if spacing_zyx.shape[0] == 1 and x.shape[0] != 1:
            spacing_zyx = spacing_zyx.expand(x.shape[0], -1)
        if spacing_zyx.shape[0] != x.shape[0]:
            raise ValueError("runtime geometry spacing batch must equal input batch")
        padded, _ = self._pad(x)
        fused_forward = getattr(self.encoder, "forward_with_features", None)
        if callable(fused_forward):
            features, native_region = fused_forward(padded)
        else:
            features = self.encoder.forward_features(padded)
            native_region = None
        if not isinstance(features, (tuple, list)) or len(features) != len(self.encoder_channels):
            raise ValueError("A-3D encoder must return one feature tensor per declared stage")
        for index, (feature, channels) in enumerate(zip(features, self.encoder_channels)):
            if feature.ndim != 5 or feature.shape[1] != channels:
                raise ValueError(f"A-3D feature stage {index} shape mismatch")
            _finite(f"A-3D feature stage {index}", feature)
        if self._external_region_path:
            # Keep the complete upstream segmentation route for P_R.  The
            # boundary path consumes explicit feature stages and never replaces
            # or reuses the upstream logits as its representation.
            region = native_region if native_region is not None else self.encoder.forward_region(padded)
            if region.ndim != 5 or region.shape[0] != x.shape[0] or region.shape[1] != self.num_classes \
                    or tuple(region.shape[-3:]) != tuple(padded.shape[-3:]):
                raise ValueError("MedNeXt original region path must return [B,4,D,H,W] at input size")
            _finite("MedNeXt region logits", region)
        else:
            region = self.region_path(features)
        if self.checkpoint_boundary and torch.is_grad_enabled():
            # The non-reentrant implementation supports warm-up, where the
            # frozen encoder produces tensors without ``requires_grad`` while
            # the boundary parameters still need gradients.  Inference and
            # teacher snapshots run without grad and therefore keep the normal
            # single-pass path.
            boundary, sdf = torch_checkpoint.checkpoint(
                self.boundary_path,
                features[0], features[-1], spacing_zyx,
                use_reentrant=False,
            )
        else:
            boundary, sdf = self.boundary_path(features[0], features[-1], spacing_zyx)
        region = region[..., : original_shape[0], : original_shape[1], : original_shape[2]]
        boundary = boundary[..., : original_shape[0], : original_shape[1], : original_shape[2]]
        sdf = sdf[..., : original_shape[0], : original_shape[1], : original_shape[2]]
        return {"region_logits": region, "boundary_logits": boundary, "sdf": sdf}

    def train(self, mode: bool = True) -> DualBranchTeacher3D:
        super().train(mode)
        if mode:
            for name in self._frozen_modules:
                getattr(self, name).eval()
                getattr(self, name).requires_grad_(False)
        return self


def build_teacher3d(model_config: Mapping[str, Any] | str | Path,
                    *, synthetic: bool = False,
                    checkpoint_state: Mapping[str, Tensor] | None = None,
                    checkpoint_metadata: Mapping[str, Any] | None = None) -> tuple[DualBranchTeacher3D, dict[str, Any]]:
    """Build A-3D teacher with explicit local backbone/checkpoint provenance."""

    if isinstance(model_config, (str, Path)):
        config = json.loads(Path(model_config).read_text())
    else:
        config = copy.deepcopy(dict(model_config))
    if isinstance(config.get("model"), Mapping) and "backbone" not in config:
        nested = copy.deepcopy(dict(config["model"]))
        nested.update({key: value for key, value in config.items() if key != "model"})
        config = nested
    if _class_mapping(config.get("class_mapping", CLASS_NAMES)) != CLASS_NAMES:
        raise ValueError("A-3D class mapping must be BG,RV,MYO,LV")
    backbone = str(config.get("backbone", ""))
    if backbone == "tiny_test3d":
        if not synthetic:
            raise ValueError("tiny_test3d is permitted only for manufactured synthetic checks")
        input_channels = int(config.get("input_channels", 1))
        encoder = TinyEncoder3D(in_channels=input_channels,
                                widths=tuple(config.get("feature_channels", (8, 16, 24, 32))))
        backbone_meta = {"backbone": "tiny_test3d", "pretrained": False,
                         "weights_status": "SYNTHETIC_RANDOM_INIT"}
    elif backbone in {"mednext_l_k3", "mednext_l_3d_k3"}:
        input_channels = int(config.get("input_channels", 1))
        encoder, backbone_meta = _load_mednext(config)
    elif backbone == "refineformer3d_standard":
        raise ValueError("RefineFormer3D Standard adapter is not verified in this checkout")
    else:
        raise ValueError("unsupported A-3D backbone; use mednext_l_k3 or synthetic tiny_test3d")
    teacher = DualBranchTeacher3D(
        encoder, input_channels=input_channels,
        class_mapping=CLASS_NAMES, boundary_width=config.get("boundary_width"),
        backbone_meta=backbone_meta,
        checkpoint_boundary=config.get("checkpoint_boundary", False),
    )
    if checkpoint_state is not None:
        if checkpoint_metadata is None:
            raise ValueError("checkpoint_metadata is required with checkpoint_state")
        if checkpoint_metadata.get("model_class") != "DualBranchTeacher3D":
            raise ValueError("A-3D checkpoint model_class mismatch")
        if tuple(checkpoint_metadata.get("class_mapping", ())) != CLASS_NAMES:
            raise ValueError("A-3D checkpoint class mapping mismatch")
        teacher.load_state_dict(checkpoint_state, strict=True)
        quality = "TRAINED_CHECKPOINT"
    else:
        quality = backbone_meta.get("weights_status", "UNTRAINED_RANDOM_INIT")
    provenance = {
        "model_class": "DualBranchTeacher3D", "architecture": "shared_3d_trunk_original_region_separate_boundary_sdf",
        "input_channels": input_channels, "num_classes": 4,
        "class_mapping": list(CLASS_NAMES), "backbone": backbone_meta,
        "pretraining_data": backbone_meta.get("pretraining_data"),
        "quality": quality, "synthetic": bool(synthetic),
        "padding_policy": teacher.padding_policy,
        "padding_multiple": teacher.padding_multiple,
        "checkpoint_boundary": teacher.checkpoint_boundary,
        "accuracy_claim": "NOT_MEASURED",
        "class_mapping_trust": "metadata_self_declared_semantic_permutation_not_empirically_verified",
    }
    teacher.provenance = copy.deepcopy(provenance)
    return teacher, provenance


def teacher_loss(outputs: Mapping[str, Tensor], y: Tensor, sdf_targets: Tensor,
                 lambda_distance: float) -> Tensor:
    """CE+Dice on both branches plus normalized signed-distance MAE for 3-D."""

    if set(outputs) != {"region_logits", "boundary_logits", "sdf"}:
        raise ValueError("A-3D teacher outputs must contain region_logits, boundary_logits and sdf")
    region, boundary, sdf = (outputs["region_logits"], outputs["boundary_logits"], outputs["sdf"])
    if region.ndim != 5 or region.shape[1] != 4 or boundary.shape != region.shape or sdf.shape != region.shape:
        raise ValueError("A-3D outputs must have matching [B,4,D,H,W] shapes")
    labels = y[:, 0] if y.ndim == 5 and y.shape[1] == 1 else y
    if labels.ndim != 4 or tuple(labels.shape) != tuple(region.shape[:1] + region.shape[2:]):
        raise ValueError("A-3D labels must have shape [B,D,H,W] or [B,1,D,H,W]")
    labels = labels.long()
    valid = labels != 255
    if not bool(valid.any()) or bool(((labels != 255) & ((labels < 0) | (labels > 3))).any()):
        raise ValueError("A-3D labels have no valid support or contain unknown IDs")
    if not isinstance(sdf_targets, Tensor) or sdf_targets.shape != sdf.shape:
        raise ValueError("A-3D sdf_targets must match sdf shape")
    _finite("region_logits", region); _finite("boundary_logits", boundary)
    _finite("sdf", sdf); _finite("sdf_targets", sdf_targets)
    if bool((sdf < -1.00001).any()) or bool((sdf > 1.00001).any()):
        raise ValueError("A-3D sdf must be in [-1,1]")
    if bool((sdf_targets < -1.00001).any()) or bool((sdf_targets > 1.00001).any()):
        raise ValueError("A-3D sdf_targets must be in [-1,1]")
    if isinstance(lambda_distance, bool) or not isinstance(lambda_distance, (int, float)) or lambda_distance < 0:
        raise ValueError("lambda_distance must be a non-negative finite scalar")
    if not torch.isfinite(torch.tensor(float(lambda_distance))):
        raise ValueError("lambda_distance must be finite")

    distance = (sdf - sdf_targets).abs().masked_select(valid.unsqueeze(1).expand_as(sdf)).mean()
    total = supervised_loss(region, labels) + supervised_loss(boundary, labels) + float(lambda_distance) * distance
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite A-3D teacher loss")
    return total


def set_teacher_stage(teacher: DualBranchTeacher3D, stage: str, freeze: bool = False) -> DualBranchTeacher3D:
    """Apply explicit pretrain, boundary warm-up, joint and frozen stages."""

    if not isinstance(teacher, DualBranchTeacher3D):
        raise TypeError("teacher must be DualBranchTeacher3D")
    aliases = {"teacher_pretrain": "pretrain", "teacher_warmup": "warmup"}
    stage = aliases.get(stage, stage)
    if stage not in {"pretrain", "warmup", "joint"}:
        raise ValueError("stage must be pretrain, warmup or joint")
    teacher.requires_grad_(not freeze)
    teacher._frozen_modules = ()
    teacher._stage = stage
    if freeze:
        teacher.eval()
        teacher._frozen_modules = ("encoder", "region_path", "boundary_path")
        return teacher
    teacher.train()
    if stage == "warmup":
        teacher.encoder.requires_grad_(False)
        teacher.region_path.requires_grad_(False)
        teacher.encoder.eval(); teacher.region_path.eval()
        teacher._frozen_modules = ("encoder", "region_path")
    return teacher


__all__ = [
    "CLASS_NAMES",
    "UPSTREAM_MAIN_SHA",
    "UPSTREAM_URL",
    "UPSTREAM_VERIFIED_FILES",
    "DualBranchTeacher3D",
    "MedNeXtL3DAdapter",
    "TinyEncoder3D",
    "build_teacher3d",
    "set_teacher_stage",
    "teacher_loss",
]
