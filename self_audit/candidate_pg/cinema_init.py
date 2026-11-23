"""CineMA SAX initialisation for Candidate PG.

The real path in this module is intentionally narrow: it resolves exactly the
public CineMA backbone and its matching ``pretrained/config.yaml`` from
``mathpluscode/CineMA``.  No fine-tuned task checkpoint, guessed architecture,
or random-weight fallback is accepted.  Tests may inject a tiny encoder with
``encoder_factory``; injected runs are software-only and are marked as such in
the receipts.

The source model uses the official ConvMAE preprocessing contract (SAX spacing
``[1, 1, 10]``, crop/pad ``192x192x16``).  Student tensors remain native
``[B, 1, D, H, W]``.  Distillation samples the CineMA feature lattice through
the affine that describes the crop/resampling operation and masks padded or
cropped-out support before taking a cosine loss.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:  # optional at import time; real execution reports a useful error
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - dependency checked at real load time
    hf_hub_download = None  # type: ignore[assignment]

try:
    import yaml
except Exception:  # pragma: no cover - dependency checked at real load time
    yaml = None  # type: ignore[assignment]

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover - dependency checked at real load time
    safe_open = None  # type: ignore[assignment]


HF_REPO = "mathpluscode/CineMA"
HF_CONFIG_FILE = "pretrained/config.yaml"
HF_WEIGHTS_FILE = "pretrained/cinema.safetensors"
# Preserve the exact order in the PG request while resolving by named roles.
HF_FILES = (HF_WEIGHTS_FILE, HF_CONFIG_FILE)
OFFICIAL_SAX_SPACING = (1.0, 1.0, 10.0)
OFFICIAL_SAX_SIZE = (192, 192, 16)  # H, W, D in the official SAX model
UPDATES_PER_EPOCH = 80
_SHA256 = set("0123456789abcdef")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256


def _cfg_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("cinema_init", config)
    if not isinstance(value, Mapping):
        raise ValueError("cinema_init configuration must be a mapping")
    return value


def _required_source_settings(cinema_config: Mapping[str, Any]) -> tuple[str, str | None]:
    repo = cinema_config.get("hf_repo", HF_REPO)
    if repo != HF_REPO:
        raise ValueError(f"CineMA source is locked to {HF_REPO!r}; got {repo!r}")
    files = cinema_config.get("files", HF_FILES)
    if isinstance(files, str) or not isinstance(files, Sequence) or tuple(sorted(files)) != tuple(sorted(HF_FILES)):
        raise ValueError("CineMA loader may fetch only pretrained/config.yaml and pretrained/cinema.safetensors")
    revision = cinema_config.get("hf_revision")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("hf_revision must be a non-empty string when provided")
    return repo, revision


def _validate_official_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the architecture fields supplied by CineMA's YAML.

    The numeric values below are *source assertions*, not architecture
    defaults.  The feature strides and channel widths used by the loader are
    derived from the validated YAML object.
    """

    try:
        data = raw["data"]
        sax = data["sax"]
        model = raw["model"]
    except (KeyError, TypeError) as exc:
        raise ValueError("CineMA config is missing data.sax/model sections") from exc
    if tuple(float(v) for v in sax.get("spacing", ())) != OFFICIAL_SAX_SPACING:
        raise ValueError("CineMA SAX spacing differs from official [1,1,10] config")
    if tuple(int(v) for v in sax.get("patch_size", ())) != OFFICIAL_SAX_SIZE:
        raise ValueError("CineMA SAX patch_size differs from official [192,192,16] config")
    if int(sax.get("in_chans", -1)) != 1:
        raise ValueError("CineMA SAX in_chans must be one")
    views = model.get("views")
    if isinstance(views, str) or not isinstance(views, Sequence) or "sax" not in views:
        raise ValueError("CineMA config must declare SAX among model.views")
    if tuple(int(v) for v in model.get("patch_size", ())) != (4, 4, 1):
        raise ValueError("CineMA model.patch_size is not the official SAX ConvMAE value")
    if tuple(int(v) for v in model.get("scale_factor", ())) != (2, 2, 1):
        raise ValueError("CineMA model.scale_factor is not the official SAX ConvMAE value")
    channels = tuple(int(v) for v in model.get("enc_conv_chans", ()))
    if not channels or any(v <= 0 for v in channels):
        raise ValueError("CineMA model.enc_conv_chans must be positive")
    n_blocks = int(model.get("enc_conv_n_blocks", 0))
    if n_blocks <= 0:
        raise ValueError("CineMA model.enc_conv_n_blocks must be positive")
    size = model.get("size")
    if size not in {"tiny", "base", "large", "huge"}:
        raise ValueError("CineMA model.size is not one of the official sizes")

    # Official ConvMAE: first patch, one scale factor per conv block, then one
    # scale factor for the ViT patch embedding.  This gives 1/8 skips and
    # 1/16 tokens in XY; z remains patch-size 1 throughout.
    patch = tuple(int(v) for v in model["patch_size"])
    scale = tuple(int(v) for v in model["scale_factor"])
    conv_scales = [patch, *([scale] * len(channels))]
    skip_strides: list[tuple[int, int, int]] = []
    running = [1, 1, 1]
    for index, factor in enumerate(conv_scales[:-1]):
        running = [a * b for a, b in zip(running, factor, strict=True)]
        if index < len(channels):
            skip_strides.append(tuple(running))
    token_stride = tuple(a * b for a, b in zip(running, conv_scales[-1], strict=True))
    architecture = {
        "size": size,
        "views": [str(v) for v in views],
        "sax_spacing": list(OFFICIAL_SAX_SPACING),
        "sax_patch_size_hwd": list(OFFICIAL_SAX_SIZE),
        "in_chans": 1,
        "patch_size_hwd": list(patch),
        "scale_factor_hwd": list(scale),
        "enc_conv_chans": list(channels),
        "enc_conv_n_blocks": n_blocks,
        "skip_strides_zyx": [list(reversed(x)) for x in skip_strides],
        "token_stride_zyx": list(reversed(token_stride)),
        "cinema_conv_1_8_present": any(x[:2] == (8, 8) for x in skip_strides),
    }
    return architecture


def _read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse CineMA pretrained/config.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("CineMA pretrained/config.yaml must contain a mapping")
    return copy.deepcopy(dict(raw))


@dataclass(frozen=True)
class CinemaSource:
    repo: str
    revision: str | None
    config_path: str
    weights_path: str
    config_sha256: str
    weights_sha256: str
    config: dict[str, Any]
    architecture: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "hf_repo": self.repo,
            "hf_revision": self.revision,
            "files": list(HF_FILES),
            "config_path": self.config_path,
            "weights_path": self.weights_path,
            "config_sha256": self.config_sha256,
            "weights_sha256": self.weights_sha256,
            "config": _jsonable(self.config),
            "architecture": _jsonable(self.architecture),
        }


def resolve_cinema_source(cinema_config: Mapping[str, Any], *, allow_download: bool = True) -> CinemaSource:
    """Resolve and validate the two hashable official CineMA source files."""

    repo, revision = _required_source_settings(cinema_config)
    explicit = cinema_config.get("source_paths")
    if explicit is not None:
        if not bool(cinema_config.get("synthetic", False)):
            raise ValueError("real CineMA initialisation must resolve the two files from Hugging Face")
        if not isinstance(explicit, Mapping) or set(explicit) != {"config", "weights"}:
            raise ValueError("source_paths must contain only config and weights")
        config_path, weights_path = Path(str(explicit["config"])), Path(str(explicit["weights"]))
    else:
        if not allow_download:
            raise RuntimeError("CineMA source paths are not available and downloads are disabled")
        if hf_hub_download is None:
            raise RuntimeError("huggingface_hub is required for the real CineMA source")
        kwargs = {} if revision is None else {"revision": revision}
        # Exactly two Hub calls: no repository snapshot or fine-tuned artifact.
        config_path = Path(hf_hub_download(repo_id=repo, filename=HF_CONFIG_FILE, **kwargs))
        weights_path = Path(hf_hub_download(repo_id=repo, filename=HF_WEIGHTS_FILE, **kwargs))
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("CineMA source config/weights files must both exist")
    raw = _read_yaml(config_path)
    architecture = _validate_official_config(raw)
    return CinemaSource(
        repo=repo,
        revision=revision,
        config_path=str(config_path.resolve()),
        weights_path=str(weights_path.resolve()),
        config_sha256=_sha256(config_path),
        weights_sha256=_sha256(weights_path),
        config=raw,
        architecture=architecture,
    )


def _make_omegaconf(raw: Mapping[str, Any]) -> Any:
    try:
        from omegaconf import OmegaConf  # type: ignore
    except Exception as exc:  # pragma: no cover - real environment dependency
        raise RuntimeError(
            "The real CineMA loader requires the official CineMA dependency omegaconf; "
            "install mathpluscode/CineMA before a real run"
        ) from exc
    return OmegaConf.create(raw)


class _OfficialSaxEncoder(nn.Module):
    """Expose official ConvMAE skip/token taps without importing a fine-tuned head."""

    def __init__(self, model: nn.Module, architecture: Mapping[str, Any]) -> None:
        super().__init__()
        self.model = model
        strides = [tuple(int(v) for v in item) for item in architecture.get("skip_strides_zyx", ())]
        self._conv_1_8_index = next((i for i, stride in enumerate(strides) if stride == (1, 8, 8)), None)

    def forward(self, image_hwd: Tensor) -> dict[str, Tensor | None]:
        # Official model expects [B,C,H,W,D], whereas PG runtime is [B,C,D,H,W].
        if image_hwd.ndim != 5:
            raise ValueError("CineMA SAX input must be [B,1,H,W,D]")
        down = self.model.enc_down_dict["sax"]
        skips, patch_tokens = down(image_hwd, mask=None)
        encoded = self.model.encoder(patch_tokens)
        n = patch_tokens.shape[1]
        tokens = encoded[:, 1 : 1 + n]
        tokens = self.model.enc_fusion_dict["sax"](skips, tokens, mask=None)
        conv = skips[self._conv_1_8_index] if self._conv_1_8_index is not None and len(skips) > self._conv_1_8_index else None
        # Convert official H,W,D layouts back to PG D,H,W layouts.
        if conv is not None:
            conv = conv.permute(0, 1, 4, 2, 3)
        # ``tokens`` are [B,N,C] and are reshaped by the caller using the
        # architecture-derived 1/16 grid.  Keep the official channel order.
        return {"conv_1_8": conv, "tokens_1_16": tokens}


def _load_official_encoder(source: CinemaSource, device: torch.device) -> nn.Module:
    try:
        from cinema.mae.mae import get_model as cinema_get_model  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on external package
        raise RuntimeError(
            "Official CineMA Python source is required for real initialisation; "
            "install mathpluscode/CineMA (no stand-in architecture is used)"
        ) from exc
    if safe_open is None:
        raise RuntimeError("safetensors is required for the official CineMA checkpoint")
    model = cinema_get_model(_make_omegaconf(source.config))
    state: dict[str, Tensor] = {}
    with safe_open(source.weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            state[key] = handle.get_tensor(key)
    # An ablated student (e.g. SliceSSMStudent without SliceAlign or z-SSM blocks) declares the prefixes of the modules it
    # removed; artifact keys under exactly those prefixes are dropped (recorded), every other unexpected key is still an error.
    ignorable = tuple(getattr(model, "s1_ignorable_prefixes", ()) or ())
    model_keys = set(model.state_dict())
    dropped = sorted(k for k in state if k not in model_keys and ignorable and k.startswith(ignorable))
    if dropped:
        state = {k: v for k, v in state.items() if k not in dropped}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"strict CineMA pretrained load failed: missing={missing}, unexpected={unexpected}")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return _OfficialSaxEncoder(model, source.architecture)


def _call_factory(factory: Callable[..., Any], role: str, architecture: Mapping[str, Any]) -> Any:
    """Call an explicitly role-aware test factory.

    Multiple signature retries are deliberately forbidden: a typo in an
    injected factory must fail closed instead of silently constructing a
    different architecture.
    """

    return factory(role=role, architecture=architecture)


def _tensor_image(value: Any) -> Tensor:
    image = torch.as_tensor(value, dtype=torch.float32)
    if image.ndim == 5 and image.shape[0] == 1 and image.shape[1] == 1:
        image = image[0, 0]
    elif image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or not torch.isfinite(image).all():
        raise ValueError("TRAIN CineMA image must be finite [D,H,W] or [1,1,D,H,W]")
    return image


def _item_parts(item: Any) -> tuple[Tensor, Mapping[str, Any], str]:
    if isinstance(item, Mapping):
        image = item.get("image")
        geometry = item.get("geometry", {})
        role = str(item.get("split", item.get("role", "train"))).lower()
        if image is None:
            raise ValueError("TRAIN item mapping requires image")
    else:
        image = getattr(item, "image", item)
        geometry = getattr(item, "geometry", {})
        role = str(getattr(item, "split", getattr(item, "role", "train"))).lower()
    if role not in {"train", "training", ""}:
        raise ValueError("CineMA S1 accepts TRAIN images only; VAL/TEST item detected")
    if not isinstance(geometry, Mapping):
        raise ValueError("TRAIN item geometry must be a mapping")
    image_tensor = _tensor_image(image)
    declared_shape = geometry.get("shape")
    if declared_shape is not None and tuple(int(v) for v in declared_shape) != tuple(int(v) for v in image_tensor.shape):
        raise ValueError("TRAIN item geometry.shape must match the native image shape")
    return image_tensor, geometry, role


def _affine(value: Any, name: str, *, identity: bool = False) -> np.ndarray:
    if value is None and identity:
        return np.eye(4, dtype=np.float64)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} must have homogeneous final row")
    if abs(float(np.linalg.det(matrix[:3, :3]))) <= 1e-12:
        raise ValueError(f"{name} spatial transform is singular")
    return matrix


@dataclass
class PreparedSax:
    """One CineMA view and its explicit native-to-processed geometry."""

    model_input_hwd: Tensor  # [1, 1, H, W, D]
    valid_hwd: Tensor  # [1, 1, H, W, D], 1 only for unpadded crop support
    native_shape_zyx: tuple[int, int, int]
    processed_shape_hwd: tuple[int, int, int]
    processed_affine_zyx: np.ndarray
    native_to_processed: np.ndarray
    support_processed_zyx: tuple[int, int, int]
    support_processed_bounds_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    details: dict[str, Any]


def _resample_to_spacing(image: Tensor, spacing_zyx: tuple[float, float, float]) -> tuple[Tensor, tuple[float, float, float]]:
    target_zyx = tuple(reversed(OFFICIAL_SAX_SPACING))
    if any(v <= 0 or not math.isfinite(v) for v in spacing_zyx):
        raise ValueError("geometry spacing_zyx must contain positive finite values")
    native = tuple(int(v) for v in image.shape)
    out_shape = tuple(max(1, int(round(n * s / t))) for n, s, t in zip(native, spacing_zyx, target_zyx, strict=True))
    if out_shape == native:
        return image, (1.0, 1.0, 1.0)
    result = F.interpolate(image[None, None], size=out_shape, mode="trilinear", align_corners=False)[0, 0]
    scale = tuple(o / n for o, n in zip(out_shape, native, strict=True))
    return result, scale


def _normalise_intensity(image: Tensor) -> Tensor:
    # Official inference applies ScaleIntensityd to the already processed
    # uint8 volume.  This is its continuous equivalent for float TRAIN frames.
    lo, hi = image.amin(), image.amax()
    if float(hi - lo) <= 1e-12:
        return torch.zeros_like(image)
    return ((image - lo) / (hi - lo)).clamp(0.0, 1.0)


def prepare_cinema_sax(image: Any, geometry: Mapping[str, Any] | None = None, *, synthetic: bool = False) -> PreparedSax:
    """Apply official SAX spacing/intensity/crop-pad operations.

    ``geometry['cinema_crop']`` may provide ``start_zyx`` (inclusive) and an
    optional ``size_zyx`` after spacing conversion.  A real native run must
    declare this crop when its native field is not already the 192x192 SAX
    canvas; synthetic tests may use an explicit centre crop for logic checks.
    """

    geometry = geometry or {}
    tensor = _tensor_image(image)
    native_shape = tuple(int(v) for v in tensor.shape)
    spacing = tuple(float(v) for v in geometry.get("spacing_zyx", tuple(reversed(OFFICIAL_SAX_SPACING))))
    resampled, scale = _resample_to_spacing(tensor, spacing)
    rz, ry, rx = (int(v) for v in resampled.shape)
    crop = geometry.get("cinema_crop")
    if crop is not None and not isinstance(crop, Mapping):
        raise ValueError("geometry.cinema_crop must be a mapping")
    if crop is None:
        if (ry, rx) != OFFICIAL_SAX_SIZE[:2] and not synthetic:
            raise ValueError("real CineMA S1 requires the explicit affine cinema_crop for native XY geometry")
        # A preprocessed CineMA frame is already on the target canvas.  For
        # synthetic tests only, use an explicit deterministic centre crop.
        sy = max(0, (ry - OFFICIAL_SAX_SIZE[0]) // 2)
        sx = max(0, (rx - OFFICIAL_SAX_SIZE[1]) // 2)
        sz = 0
    else:
        start = crop.get("start_zyx", crop.get("origin_zyx", (0, 0, 0)))
        if not isinstance(start, Sequence) or len(start) != 3:
            raise ValueError("cinema_crop.start_zyx must be a z,y,x triple")
        sz, sy, sx = (int(v) for v in start)
    target_h, target_w, target_d = OFFICIAL_SAX_SIZE
    if crop is not None and crop.get("size_zyx") is not None:
        size = tuple(int(v) for v in crop["size_zyx"])
        if len(size) != 3 or size[1:] != (target_h, target_w) or size[0] <= 0:
            raise ValueError("cinema_crop.size_zyx must declare a positive z extent and 192x192 XY")
    # Crop and end-pad exactly as official SpatialPadd(method='end').
    out = torch.zeros((target_d, target_h, target_w), dtype=torch.float32)
    support = [0, 0, 0]
    for axis, (start_i, length_i, target_i) in enumerate(((sz, rz, target_d), (sy, ry, target_h), (sx, rx, target_w))):
        src_start = max(0, start_i)
        src_end = min(length_i, start_i + target_i)
        support[axis] = max(0, src_end - src_start)
    src_z0, src_y0, src_x0 = max(0, sz), max(0, sy), max(0, sx)
    src_z1, src_y1, src_x1 = min(rz, sz + target_d), min(ry, sy + target_h), min(rx, sx + target_w)
    dst_z0, dst_y0, dst_x0 = max(0, -sz), max(0, -sy), max(0, -sx)
    if src_z1 > src_z0 and src_y1 > src_y0 and src_x1 > src_x0:
        out[
            dst_z0 : dst_z0 + src_z1 - src_z0,
            dst_y0 : dst_y0 + src_y1 - src_y0,
            dst_x0 : dst_x0 + src_x1 - src_x0,
        ] = resampled[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
    out = _normalise_intensity(out)
    valid = torch.zeros_like(out)
    if src_z1 > src_z0 and src_y1 > src_y0 and src_x1 > src_x0:
        valid[
            dst_z0 : dst_z0 + src_z1 - src_z0,
            dst_y0 : dst_y0 + src_y1 - src_y0,
            dst_x0 : dst_x0 + src_x1 - src_x0,
        ] = 1.0
    # Processed affine is derived from native affine and the declared crop,
    # preserving oblique/rotated physical geometry (not an identity guess).
    native_affine = _affine(geometry.get("affine_zyx"), "geometry.affine_zyx", identity=True)
    # ``F.interpolate(..., align_corners=False)`` maps output center ``p`` to
    # input coordinate ``(p+0.5)/scale - 0.5``.  Crop start is applied after
    # this resampling, so the processed->native translation is positive.
    scale_matrix = np.diag([1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2], 1.0])
    scale_matrix[:3, 3] = (
        (sz + 0.5) / scale[0] - 0.5,
        (sy + 0.5) / scale[1] - 0.5,
        (sx + 0.5) / scale[2] - 0.5,
    )
    explicit_processed_affine = None
    if crop is not None:
        explicit_processed_affine = crop.get("processed_affine_zyx", crop.get("affine_zyx"))
    processed_affine = (
        _affine(explicit_processed_affine, "cinema_crop.affine_zyx")
        if explicit_processed_affine is not None
        else native_affine @ scale_matrix
    )
    native_to_processed = np.linalg.inv(processed_affine) @ native_affine
    support_bounds = (
        (dst_z0, dst_z0 + max(0, src_z1 - src_z0)),
        (dst_y0, dst_y0 + max(0, src_y1 - src_y0)),
        (dst_x0, dst_x0 + max(0, src_x1 - src_x0)),
    )
    return PreparedSax(
        model_input_hwd=out.permute(1, 2, 0)[None, None],
        valid_hwd=valid.permute(1, 2, 0)[None, None],
        native_shape_zyx=native_shape,
        processed_shape_hwd=(target_h, target_w, target_d),
        processed_affine_zyx=processed_affine,
        native_to_processed=native_to_processed,
        support_processed_zyx=(max(0, src_z1 - src_z0), max(0, src_y1 - src_y0), max(0, src_x1 - src_x0)),
        support_processed_bounds_zyx=support_bounds,
        details={
            "spacing_zyx": list(spacing),
            "target_spacing_zyx": list(reversed(OFFICIAL_SAX_SPACING)),
            "resample_scale_zyx": list(scale),
            "crop_start_zyx": [sz, sy, sx],
            "padded_shape_zyx": [target_d, target_h, target_w],
            "valid_voxels": int(valid.sum().item()),
            "support_processed_bounds_zyx": [list(pair) for pair in support_bounds],
            "synthetic_preprocessing": bool(synthetic),
        },
    )


def _feature_to_grid(value: Tensor, *, feature_name: str, architecture: Mapping[str, Any]) -> Tensor | None:
    if value is None:
        return None
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value)
    if feature_name == "tokens_1_16" and value.ndim == 3:
        stride = tuple(int(v) for v in architecture["token_stride_zyx"])
        grid_zyx = tuple(max(1, int(round(n / s))) for n, s in zip((16, 192, 192), stride, strict=True))
        if value.shape[1] != math.prod(grid_zyx):
            raise ValueError(f"CineMA token count {value.shape[1]} does not match YAML-derived grid {grid_zyx}")
        # Official patchify_3d flattens H,W,D (not D,H,W). Restore that order
        # before exposing the PG-native D,H,W feature tensor.
        grid_hwd = (grid_zyx[1], grid_zyx[2], grid_zyx[0])
        value = value.reshape(value.shape[0], *grid_hwd, value.shape[2]).permute(0, 4, 3, 1, 2)
    if value.ndim != 5:
        raise ValueError(f"{feature_name} must be [B,C,D,H,W] or token [B,N,C]")
    return value


def backproject_feature(
    feature: Tensor,
    prepared: PreparedSax,
    target_shape_zyx: Sequence[int],
    *,
    source_stride_zyx: Sequence[int],
    target_stride_zyx: Sequence[int] | None = None,
    target_offset_zyx: Sequence[float] | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample a CineMA feature on the native student lattice.

    Returns ``(sampled, valid)`` where valid has shape ``[B,1,D,H,W]`` and is
    zero for every center outside the real cropped support (including padded
    z slices).  ``grid_sample`` is used only after affine coordinates are
    computed, so oblique crop metadata is respected.
    """

    if feature.ndim != 5:
        raise ValueError("feature must be [B,C,D,H,W]")
    target = tuple(int(v) for v in target_shape_zyx)
    if len(target) != 3 or any(v <= 0 for v in target):
        raise ValueError("target_shape_zyx must be positive D,H,W")
    source_stride = tuple(float(v) for v in source_stride_zyx)
    if len(source_stride) != 3 or any(v <= 0 for v in source_stride):
        raise ValueError("source_stride_zyx must be a positive triple")
    native_stride = tuple(
        float(v) for v in (target_stride_zyx if target_stride_zyx is not None else
                           tuple(float(n) / t for n, t in zip(prepared.native_shape_zyx, target, strict=True)))
    )
    if len(native_stride) != 3 or any(v <= 0 for v in native_stride):
        raise ValueError("target_stride_zyx must be a positive triple")
    offsets = tuple(float(v) for v in (target_offset_zyx or (0.0, 0.0, 0.0)))
    if len(offsets) != 3 or any(not math.isfinite(v) for v in offsets):
        raise ValueError("target_offset_zyx must be a finite triple")
    indices = [torch.arange(t, dtype=torch.float32, device=feature.device) * s + o for t, s, o in zip(target, native_stride, offsets, strict=True)]
    zz, yy, xx = torch.meshgrid(*indices, indexing="ij")
    native = torch.stack([zz, yy, xx, torch.ones_like(zz)], dim=-1)
    affine = torch.as_tensor(prepared.native_to_processed, dtype=torch.float32, device=feature.device)
    processed = torch.einsum("...j,ij->...i", native, affine.T)[..., :3]
    source = torch.stack(
        [
            # ConvMAE's no-padding kernel=stride output is centred at
            # ``j * stride + (stride - 1) / 2`` in processed-voxel
            # coordinates.  Inverting that centre relation gives the
            # feature-lattice index ``p / stride - 0.5 + 0.5 / stride``.
            # The ``+ 0.5 / stride`` term is easy to lose and would shift
            # every sampled tap by half a voxel for stride one (and by
            # 0.4375 for the 1/8 tap).
            (processed[..., 0] / source_stride[0] - 0.5 + 0.5 / source_stride[0]),
            (processed[..., 1] / source_stride[1] - 0.5 + 0.5 / source_stride[1]),
            (processed[..., 2] / source_stride[2] - 0.5 + 0.5 / source_stride[2]),
        ],
        dim=-1,
    )
    # grid_sample's final coordinate order is x,y,z and coordinates are
    # normalised against W,H,D respectively.
    d, h, w = feature.shape[-3:]
    grid = torch.stack(
        [
            2 * (source[..., 2] + 0.5) / max(w, 1) - 1,
            2 * (source[..., 1] + 0.5) / max(h, 1) - 1,
            2 * (source[..., 0] + 0.5) / max(d, 1) - 1,
        ],
        dim=-1,
    )[None].expand(feature.shape[0], -1, -1, -1, -1)
    sampled = F.grid_sample(feature.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    # Processed coordinates are measured in input voxel centres.  Support is
    # bounded by the actual resampled crop, not the end-padded model canvas.
    bounds = prepared.support_processed_bounds_zyx
    valid = (
        (processed[..., 0] >= bounds[0][0] - 0.5) & (processed[..., 0] < bounds[0][1] - 0.5)
        & (processed[..., 1] >= bounds[1][0] - 0.5) & (processed[..., 1] < bounds[1][1] - 0.5)
        & (processed[..., 2] >= bounds[2][0] - 0.5) & (processed[..., 2] < bounds[2][1] - 0.5)
    ).to(sampled.dtype)[None, None]
    return sampled, valid


def _build_student(config: Mapping[str, Any], *, synthetic: bool, student_encoder: nn.Module | None, factory: Callable[..., Any] | None, architecture: Mapping[str, Any]) -> nn.Module:
    if student_encoder is not None:
        return student_encoder
    if synthetic:
        raise ValueError("synthetic S1 requires an explicitly injected student_encoder")
    if factory is not None:
        candidate = _call_factory(factory, "student", architecture)
        if isinstance(candidate, nn.Module):
            return candidate
        raise TypeError("encoder_factory must produce a torch.nn.Module for the student role")
    # Real execution has one canonical PG student constructor.  Do not probe
    # alternate module paths or catch import/signature failures here: a source
    # or architecture mismatch must stop S1 instead of silently changing the
    # model being distilled.
    from .student_pg import build_student

    return build_student(config)


def _extract_features(output: Any, *, source: bool = False) -> tuple[Tensor | None, Tensor | None] | tuple[Tensor, ...]:
    if isinstance(output, Mapping):
        if source:
            return output.get("conv_1_8"), output.get("tokens_1_16")
        value = output.get("features")
        if value is None and "stage3" in output and "stage4" in output:
            return (output["stage0"], output.get("stage1", output["stage0"]), output.get("stage2", output["stage3"]), output["stage3"], output["stage4"])
        if isinstance(value, (list, tuple)):
            return tuple(value)
    if isinstance(output, (list, tuple)):
        return tuple(output)
    raise TypeError("encoder output must be a mapping with features or a feature tuple")


def _student_forward(student: nn.Module, image: Tensor, geometry: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    # PGStudent's public forward contract includes native geometry.  A
    # TypeError from the model itself must not trigger a second, potentially
    # different call that could hide an actual shape or dtype failure.
    output: Any = student(image, geometry)
    features = _extract_features(output)
    if len(features) < 5:
        if len(features) == 2:
            return features[0], features[1]
        raise ValueError("student encoder must expose stage3/stage4 (five features or a two-feature tuple)")
    return features[3], features[4]


def _student_phase(
    student: nn.Module,
    stage_index: int,
    fallback: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    phases = getattr(student, "phases", None)
    if phases is not None and len(phases) > stage_index:
        phase = phases[stage_index]
        if isinstance(phase, (tuple, list)):
            step, offset = phase[0], phase[1]
        else:
            step, offset = getattr(phase, "step", None), getattr(phase, "offset", None)
        if step is not None and len(step) == 3:
            return tuple(int(v) for v in step), tuple(float(v) for v in (offset or (0.0, 0.0, 0.0)))
    return fallback, (0.0, 0.0, 0.0)


def _freeze_student_decoder(student: nn.Module) -> list[nn.Parameter]:
    trainable: list[nn.Parameter] = []
    for name, parameter in student.named_parameters():
        lowered = name.lower()
        keep = not any(token in lowered for token in ("decoder", "logits_head", "head", "fusion", "mimic", "projector"))
        parameter.requires_grad_(keep)
        if keep:
            trainable.append(parameter)
    if not trainable:
        trainable = [p for p in student.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("student encoder exposes no trainable encoder parameters")
    return trainable


def _make_projector(student_channels: int, cinema_channels: int, device: torch.device) -> nn.Module:
    layer = nn.Conv3d(int(student_channels), int(cinema_channels), kernel_size=1).to(device)
    return layer


def cosine_loss_valid(projected: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    if projected.shape != target.shape:
        raise ValueError(f"projected/target feature shapes differ: {tuple(projected.shape)} vs {tuple(target.shape)}")
    if valid.ndim != 5 or valid.shape[0] != projected.shape[0] or valid.shape[-3:] != projected.shape[-3:]:
        raise ValueError("valid feature support has incompatible shape")
    cosine = 1.0 - F.cosine_similarity(projected.float(), target.float(), dim=1, eps=1e-6)
    weight = valid[:, 0].float()
    return (cosine * weight).sum() / weight.sum().clamp_min(1.0)


def _extract_cinema(source_model: nn.Module, prepared: PreparedSax, architecture: Mapping[str, Any]) -> tuple[Tensor | None, Tensor | None]:
    try:
        source_device = next(source_model.parameters()).device
    except StopIteration:
        source_device = prepared.model_input_hwd.device
    with torch.no_grad():
        output = source_model(prepared.model_input_hwd.to(source_device))
    conv, tokens = _extract_features(output, source=True)
    if conv is not None:
        conv = torch.as_tensor(conv)
        # Official source adapter already returns D,H,W; injected encoders may
        # return H,W,D and can opt into ``layout='hwd'`` in their mapping.
    if tokens is not None:
        tokens = torch.as_tensor(tokens)
    return conv, tokens


def _prepare_record(item: Any, *, synthetic: bool) -> tuple[Tensor, Mapping[str, Any], PreparedSax]:
    image, geometry, _ = _item_parts(item)
    prepared = prepare_cinema_sax(image, geometry, synthetic=synthetic)
    return image, geometry, prepared


def _encoder_only_state(student: nn.Module) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, parameter in student.state_dict().items():
        lowered = name.lower()
        if any(token in lowered for token in ("decoder", "logits_head", "head", "fusion", "mimic", "projector")):
            continue
        result[name] = parameter.detach().cpu()
    return result


def _save_checkpoint(path: Path, student: nn.Module, projectors: Mapping[str, nn.Module], optimizer: torch.optim.Optimizer, *, epoch: int, update: int, source: CinemaSource | None, synthetic: bool) -> None:
    payload = {
        "schema": "self_audit.candidate_pg.cinema_init.v1",
        "stage": "S1",
        "epoch": int(epoch),
        "update": int(update),
        "synthetic": bool(synthetic),
        "model": _encoder_only_state(student),
        "optimizer": optimizer.state_dict(),
        "projectors_discarded": True,
        "source": source.as_dict() if source is not None else None,
        "rng": {
            "torch_cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(payload, path)


def run_cinema_init(
    config: Mapping[str, Any] | str | Path,
    out: str | Path,
    train_data: Iterable[Any],
    device: torch.device | str = "cpu",
    *,
    encoder_factory: Callable[..., Any] | None = None,
    cinema_encoder: nn.Module | None = None,
    student_encoder: nn.Module | None = None,
    max_updates: int | None = None,
) -> dict[str, Any]:
    """Run S1 CineMA distillation and return its receipt paths/metrics.

    ``train_data`` is indexed lazily and split deterministically into a
    TRAIN-only optimisation subset and holdout.  The holdout is never read
    from VAL/TEST and labels, when present on a ``Frame``, are ignored.  An
    injected ``encoder_factory``/module is only for synthetic software tests;
    real runs resolve the official Hub files and instantiate the official
    CineMA source model plus the canonical PG student.  An injected factory
    must accept keyword arguments ``role`` (``"cinema"``) and
    ``architecture``; synthetic test students are passed explicitly as
    ``student_encoder``.
    """

    if isinstance(config, (str, Path)):
        config_mapping = json.loads(Path(config).read_text(encoding="utf-8"))
    elif isinstance(config, Mapping):
        config_mapping = config
    else:
        raise TypeError("config must be a mapping or JSON path")
    cinema_config = _cfg_section(config_mapping)
    from .runtime import configure_runtime
    runtime_flags = configure_runtime(device, str(config_mapping.get("runtime", {}).get("cublas_workspace_config", ":4096:8")))
    synthetic = bool(cinema_config.get("synthetic", config_mapping.get("runtime", {}).get("synthetic", False)))
    seed = int(config_mapping.get("seed", 42))
    if seed != 42 and not synthetic:
        raise ValueError("Candidate PG S1 seed is locked to 42")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    updates = int(cinema_config.get("updates", 10000))
    if max_updates is not None:
        updates = min(updates, int(max_updates))
    if updates <= 0:
        raise ValueError("CineMA S1 updates must be positive")
    if int(cinema_config.get("updates_per_epoch", UPDATES_PER_EPOCH)) != UPDATES_PER_EPOCH:
        raise ValueError("CineMA S1 uses a locked 80-update epoch")
    holdout_fraction = float(cinema_config.get("holdout_fraction", 0.1)) if synthetic else float(cinema_config["holdout_fraction"])
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must lie strictly between zero and one")
    records = train_data if isinstance(train_data, Sequence) else tuple(train_data)
    if len(records) < 2:
        raise ValueError("CineMA S1 needs at least two TRAIN images for a holdout")
    # Never use a random split: stable record order is part of the receipt.
    holdout_count = max(1, min(len(records) - 1, int(math.ceil(len(records) * holdout_fraction))))
    train_records = range(len(records) - holdout_count)
    holdout_records = range(len(records) - holdout_count, len(records))
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    identities = getattr(records, "identities", None)
    if identities is None:
        identities = []
        for index in range(len(records)):
            item = records[index]
            def get(name, default=None):
                return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)
            identity = {"patient_id": get("patient_id"), "frame_index": get("frame_index"), "split": get("split", "train")}
            if not synthetic and (identity["patient_id"] is None or identity["frame_index"] is None):
                raise ValueError("real S1 data list requires patient_id and frame_index")
            if synthetic and identity["patient_id"] is None:
                identity["synthetic_index"] = index
            identities.append(identity)
    data_list_bytes = json.dumps(_jsonable(identities), sort_keys=True, separators=(",", ":")).encode() + b"\n"
    data_list_hash = hashlib.sha256(data_list_bytes).hexdigest()
    (out_path / "data_list.json").write_bytes(data_list_bytes)
    epoch_path = out_path / "epochs.csv"
    with epoch_path.open("w", newline="") as stream:
        csv.writer(stream).writerow(("epoch", "update", "cosine_train", "cosine_holdout"))
    print(f"[S1] start | updates {updates}", flush=True)
    dev = torch.device(device)
    torch.set_num_threads(1)

    # Factories/modules are deliberately a synthetic-test seam only.  In a
    # real run the CineMA source and architecture must come from the locked
    # Hub files, and the canonical PG student constructor above must be used;
    # accepting an injected substitute here would bypass those provenance
    # checks while still producing a seemingly completed receipt.
    if not synthetic and any(value is not None for value in (encoder_factory, cinema_encoder, student_encoder)):
        raise ValueError("encoder_factory/cinema_encoder/student_encoder injection is permitted only for synthetic checks")

    source: CinemaSource | None = None
    if cinema_encoder is None:
        factory_value = encoder_factory
        if factory_value is not None:
            produced = _call_factory(factory_value, "cinema", {"synthetic": synthetic})
            if not isinstance(produced, nn.Module):
                raise TypeError("encoder_factory must produce a torch.nn.Module for the cinema role")
            cinema_encoder = produced
        else:
            source = resolve_cinema_source(cinema_config)
            cinema_encoder = _load_official_encoder(source, dev)
    cinema_encoder = cinema_encoder.to(dev).eval()
    for parameter in cinema_encoder.parameters():
        parameter.requires_grad_(False)
    architecture = source.architecture if source is not None else {"synthetic": True, "token_stride_zyx": [1, 16, 16]}
    student = _build_student(config_mapping, synthetic=synthetic, student_encoder=student_encoder, factory=encoder_factory, architecture=architecture).to(dev)
    student.train()
    trainable = _freeze_student_decoder(student)
    projectors = nn.ModuleDict()
    optimizer_cfg = cinema_config.get("optimizer")
    if optimizer_cfg is None and not synthetic:
        raise ValueError("real CineMA S1 requires cinema_init.optimizer with explicit name/lr/weight_decay")
    if optimizer_cfg is not None:
        if not isinstance(optimizer_cfg, Mapping) or set(optimizer_cfg) != {"name", "lr", "weight_decay"}:
            raise ValueError("cinema_init.optimizer must contain exactly name, lr, weight_decay")
        if str(optimizer_cfg["name"]).lower() != "adamw":
            raise ValueError("CineMA S1 optimizer is locked to AdamW")
        learning_rate = float(optimizer_cfg["lr"])
        weight_decay = float(optimizer_cfg["weight_decay"])
    else:
        learning_rate, weight_decay = 3e-4, 1e-4
    if not math.isfinite(learning_rate) or learning_rate <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("CineMA S1 optimizer values must be finite and positive/non-negative")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

    metrics_rows: list[dict[str, Any]] = []
    update = 0
    final_train_cos = float("nan")
    final_holdout_cos = float("nan")
    tap_presence: dict[str, bool | None] = {"conv_1_8": None, "tokens_1_16": None}

    def eval_split(records_to_eval: Sequence[int]) -> float:
        if not records_to_eval:
            return float("nan")
        student.eval()
        terms: list[float] = []
        with torch.no_grad():
            for index in records_to_eval:
                item = records[index]
                native_image, geometry, prepared = _prepare_record(item, synthetic=synthetic)
                conv, tokens = _extract_cinema(cinema_encoder, prepared, architecture)
                tap_presence["conv_1_8"] = conv is not None
                tap_presence["tokens_1_16"] = tokens is not None
                # CineMA sees its own cropped/padded canvas; the student keeps
                # the original native frame and geometry contract.
                student_image = native_image[None, None].to(dev)
                s3, s4 = _student_forward(student, student_image, geometry)
                local: list[Tensor] = []
                for name, source_feature, target_feature, stride, target_phase in (
                    ("conv_1_8", conv, s3, (1, 8, 8), _student_phase(student, 3, (1, 8, 8))),
                    ("tokens_1_16", tokens, s4, (1, 16, 16), _student_phase(student, 4, (1, 16, 16))),
                ):
                    if source_feature is None:
                        continue
                    source_grid = _feature_to_grid(source_feature, feature_name=name, architecture=architecture)
                    assert source_grid is not None
                    sampled, valid = backproject_feature(source_grid.to(dev), prepared, target_feature.shape[-3:], source_stride_zyx=stride, target_stride_zyx=target_phase[0], target_offset_zyx=target_phase[1])
                    projector = projectors[name] if name in projectors else None
                    if projector is None:
                        continue
                    local.append(cosine_loss_valid(projector(target_feature), sampled, valid))
                if local:
                    terms.append(float(torch.stack(local).mean().cpu()))
        student.train()
        return 1.0 - float(np.mean(terms)) if terms else float("nan")

    while update < updates:
        step_started = time.perf_counter()
        # One update is one frame; epoch boundaries are update boundaries, not
        # dataset-pass boundaries (TRAIN may contain many more than 80 frames).
        item = records[train_records[update % len(train_records)]]
        native_image, geometry, prepared = _prepare_record(item, synthetic=synthetic)
        update += 1
        student.train()
        conv, tokens = _extract_cinema(cinema_encoder, prepared, architecture)
        tap_presence["conv_1_8"] = conv is not None
        tap_presence["tokens_1_16"] = tokens is not None
        student_image = native_image[None, None].to(dev)
        s3, s4 = _student_forward(student, student_image, geometry)
        local_losses: list[Tensor] = []
        for name, source_feature, target_feature, stride, target_phase in (
            ("conv_1_8", conv, s3, (1, 8, 8), _student_phase(student, 3, (1, 8, 8))),
            ("tokens_1_16", tokens, s4, (1, 16, 16), _student_phase(student, 4, (1, 16, 16))),
        ):
            if source_feature is None:
                continue
            source_grid = _feature_to_grid(source_feature, feature_name=name, architecture=architecture)
            assert source_grid is not None
            source_grid = source_grid.to(dev)
            sampled, valid = backproject_feature(source_grid, prepared, target_feature.shape[-3:], source_stride_zyx=stride, target_stride_zyx=target_phase[0], target_offset_zyx=target_phase[1])
            if name not in projectors:
                projectors[name] = _make_projector(int(target_feature.shape[1]), int(sampled.shape[1]), dev)
                optimizer.add_param_group({"params": projectors[name].parameters()})
            local_losses.append(cosine_loss_valid(projectors[name](target_feature), sampled.detach(), valid))
        if not local_losses:
            raise ValueError("CineMA source exposed neither the required 1/8 conv tap nor 1/16 tokens")
        loss = torch.stack(local_losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite CineMA S1 cosine loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update % 50 == 0 or update == updates:
            diagnostic = {"update": update, "loss_cosine": float(loss.detach()),
                          "seconds": time.perf_counter() - step_started,
                          "memory_peak_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None}
            with (out_path / "diag.jsonl").open("a") as stream:
                stream.write(json.dumps(diagnostic, sort_keys=True) + "\n")
        if update % UPDATES_PER_EPOCH == 0 or update == updates:
            epoch = int(math.ceil(update / UPDATES_PER_EPOCH))
            final_train_cos = eval_split(train_records)
            final_holdout_cos = eval_split(holdout_records)
            row = {
                "stage": "S1",
                "epoch": epoch,
                "update": update,
                "split": "train",
                "cosine": final_train_cos,
                "holdout_cosine": final_holdout_cos,
                "updates_per_epoch": UPDATES_PER_EPOCH,
            }
            metrics_rows.append(row)
            epoch_row = {"epoch": epoch, "update": update, "cosine_train": final_train_cos, "cosine_holdout": final_holdout_cos}
            with epoch_path.open("a", newline="") as stream:
                csv.writer(stream).writerow(epoch_row.values())
            with (out_path / "epochs.jsonl").open("a") as stream:
                stream.write(json.dumps(epoch_row, sort_keys=True) + "\n")
            with (out_path / "run.log").open("a") as stream:
                stream.write(json.dumps({"level": "DEBUG", **epoch_row}, sort_keys=True) + "\n")
            print(f"[S1] ep {epoch}/{math.ceil(updates / UPDATES_PER_EPOCH)} | cosine train {final_train_cos:.4f} | holdout {final_holdout_cos:.4f}", flush=True)
            _save_checkpoint(out_path / f"checkpoint_epoch{epoch:04d}.pt", student, projectors, optimizer, epoch=epoch, update=update, source=source, synthetic=synthetic)

    final_path = out_path / "student_init.pt"
    _save_checkpoint(final_path, student, projectors, optimizer, epoch=int(math.ceil(update / UPDATES_PER_EPOCH)), update=update, source=source, synthetic=synthetic)
    checkpoint_sha256 = _sha256(final_path)
    metrics_path = out_path / "metrics.jsonl"
    metrics_path.write_text("".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in metrics_rows), encoding="utf-8")
    receipt = {
        "schema": "self_audit.candidate_pg.cinema_init.v1",
        "stage": "S1",
        "status": "SOFTWARE_ONLY_SYNTHETIC" if synthetic else "COMPLETED",
        "seed": seed,
        "data_list_sha256": data_list_hash,
        "data_list_path": str((out_path / "data_list.json").resolve()),
        "config_sha256": hashlib.sha256(json.dumps(_jsonable(config_mapping), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "cinema_parameter_count": sum(p.numel() for p in cinema_encoder.parameters()),
        "student_encoder_trainable_parameters": sum(p.numel() for p in trainable),
        "runtime": {**runtime_flags, "deterministic": torch.are_deterministic_algorithms_enabled(),
                    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                    "tf32_cudnn": torch.backends.cudnn.allow_tf32},
        "updates": update,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "train_count": len(train_records),
        "holdout_count": len(holdout_records),
        "holdout_policy": "last_train_records_fixed_order",
        "cosine_train": final_train_cos,
        "cosine_holdout": final_holdout_cos,
        "student_checkpoint": str(final_path.resolve()),
        "projectors_discarded": True,
        "checkpoint_path": str(final_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "source": source.as_dict() if source is not None else {"injected_encoder": True, "hf_repo": HF_REPO, "files": list(HF_FILES)},
        "architecture": _jsonable(architecture),
        "feature_taps": tap_presence,
        "preprocessing": {
            "sax_spacing": list(OFFICIAL_SAX_SPACING),
            "sax_patch_size_hwd": list(OFFICIAL_SAX_SIZE),
            "intensity": "ScaleIntensityd-equivalent minmax",
            "valid_support": "crop affine and end-padding support only",
            "non_16_slice_policy": "resample to 10mm z spacing; center crop if depth>16, end pad if depth<16; padded cells invalid",
            "student_input": "native [B,1,D,H,W]",
        },
        "metrics_path": str(metrics_path.resolve()),
    }
    receipt_path = out_path / "S1_RECEIPT.json"
    receipt_path.write_text(json.dumps(_jsonable(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[S1] end | {receipt['status']}", flush=True)
    return {
        "status": receipt["status"],
        "checkpoint_path": str(final_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "student_init": str(final_path.resolve()),
        "receipt_path": str(receipt_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "source": receipt["source"],
        "cosine_train": final_train_cos,
        "cosine_holdout": final_holdout_cos,
        "updates": update,
    }


def load_encoder_artifact(path: str | Path, model: nn.Module, *, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Strictly import the encoder-only S1 artifact into a student wrapper.

    Decoder/head/fusion parameters are intentionally absent from the artifact;
    every other missing or unexpected key is an error.  The engine should call
    this boundary rather than reaching into ``student_init.pt`` directly.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != "self_audit.candidate_pg.cinema_init.v1":
        raise ValueError("invalid CineMA S1 encoder artifact schema")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("CineMA S1 artifact has no encoder model state")
    # An ablated student (e.g. SliceSSMStudent without SliceAlign or z-SSM blocks) declares the prefixes of the modules it
    # removed; artifact keys under exactly those prefixes are dropped (recorded), every other unexpected key is still an error.
    ignorable = tuple(getattr(model, "s1_ignorable_prefixes", ()) or ())
    model_keys = set(model.state_dict())
    dropped = sorted(k for k in state if k not in model_keys and ignorable and k.startswith(ignorable))
    if dropped:
        state = {k: v for k, v in state.items() if k not in dropped}
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = tuple(str(name).lower() for name in ("decoder", "logits_head", "head", "fusion", "mimic"))
    disallowed_missing = [name for name in missing if not any(token in name.lower() for token in allowed_missing)]
    if unexpected or disallowed_missing:
        raise ValueError(f"S1 encoder artifact incompatibility: missing={disallowed_missing}, unexpected={unexpected}")
    model.to(device)
    return {"path": str(checkpoint_path.resolve()), "epoch": payload.get("epoch"), "update": payload.get("update"), "source": payload.get("source"),
            "dropped_ablated_keys": len(dropped)}


__all__ = [
    "HF_FILES",
    "HF_REPO",
    "CinemaSource",
    "PreparedSax",
    "prepare_cinema_sax",
    "backproject_feature",
    "cosine_loss_valid",
    "load_encoder_artifact",
    "resolve_cinema_source",
    "run_cinema_init",
]
