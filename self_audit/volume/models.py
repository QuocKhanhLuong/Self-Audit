"""Opt-in native-volume model ladder.

The historical :class:`self_audit.model.NativeStudent` is intentionally not
modified.  ``Volume25D`` wraps that model per slice, ``HybridVolume`` keeps a
2-D encoder and adds an explicit learned cross-Z residual, and ``CNN3D`` is a
small scratch-initialized anisotropic 3-D network.  These models are software
and synthetic-contract implementations; no pretrained or external teacher is
silently downloaded or attached.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import subprocess
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..model import NativeStudent, OfficialStages, TinyEncoder, build_model
from ..window import validate_window_config
from .fusion import CrossZResidual, SkipFusion


# NativeStudent separates encoder channels from decoder/lateral widths. The
# volume schema's ``architecture.widths`` names the latter.
_TINY_ENCODER_WIDTHS = (8, 16, 32, 64)
_TINY_DECODER_WIDTHS = (8, 16, 24, 32)
_DINO_ENCODER_WIDTHS = (96, 192, 384, 768)
_DINO_DECODER_WIDTHS = (96, 128, 192, 256)


def _as_dict(config: Mapping[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    value = config.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _arch_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    architecture = _as_dict(config, "architecture")
    if architecture:
        merged = dict(config)
        merged.update(architecture)
        return merged
    return dict(config)


def _positive_int(value: Any, name: str, default: int | None = None) -> int:
    if value is None and default is not None:
        result = int(default)
    elif isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    else:
        result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _groups(module: nn.Module, *, encoder: Iterable[nn.Parameter], head: Iterable[nn.Parameter]) -> dict[str, list[nn.Parameter]]:
    """Return disjoint encoder/body/head parameter groups.

    Modules may expose aliases (the 2.5-D wrapper does), so de-duplicate by
    object identity and put every trainable parameter in exactly one group.
    """

    all_params = [p for p in module.parameters() if p.requires_grad]
    enc = list(encoder)
    h = list(head)
    enc_ids = {id(p) for p in enc}
    head_ids = {id(p) for p in h}
    if enc_ids & head_ids:
        raise RuntimeError("encoder/head parameter groups overlap")
    body = [p for p in all_params if id(p) not in enc_ids and id(p) not in head_ids]
    seen = {id(p) for p in enc + body + h}
    if len(seen) != len(all_params):
        raise RuntimeError("parameter group construction dropped or duplicated parameters")
    return {"encoder": enc, "body": body, "head": h}


def _analytical_conv_cost(module: nn.Module) -> dict[str, int]:
    """Return parameter count and local MAC proxy for convolutional modules.

    This is a stage-local proxy only: it omits interpolation, normalization,
    activations, memory traffic, and spatial output sizes. Grouped/depthwise
    convolutions account for their ``groups`` divisor explicitly.
    """

    parameters = 0
    macs = 0
    for child in module.modules():
        if isinstance(child, (nn.Conv2d, nn.Conv3d)):
            kernel = int(torch.tensor(child.kernel_size).prod())
            parameters += child.weight.numel() + (child.bias.numel() if child.bias is not None else 0)
            macs += child.out_channels * (child.in_channels // child.groups) * kernel
    return {"parameters": int(parameters), "macs_per_output_voxel": int(macs)}


def _executed_mac_proxy(module: nn.Module, input_shape: tuple[int, ...]) -> dict[str, Any]:
    """Count executed Conv/Linear multiply-accumulates at one explicit shape.

    Hooks observe actual output tensors, so per-stage spatial sizes, batch/Z
    flattening, and grouped-convolution divisors are included. Nonlinearities,
    normalization, interpolation, and memory costs are intentionally declared
    omitted rather than converted into fabricated FLOPs.
    """

    if len(input_shape) != 5 or tuple(int(x) for x in input_shape[:2]) != (1, 3):
        raise ValueError("MAC proxy input_shape must be (B,3,Z,H,W) with B=1")
    shape = tuple(int(x) for x in input_shape)
    if any(x < 1 for x in shape):
        raise ValueError("MAC proxy input dimensions must be positive")
    try:
        parameter = next(module.parameters())
        device, dtype = parameter.device, parameter.dtype
    except StopIteration:
        device, dtype = torch.device("cpu"), torch.float32
    training_state = {id(child): child.training for child in module.modules()}
    records: list[dict[str, Any]] = []

    def hook(layer: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        if not isinstance(output, Tensor):
            return
        if isinstance(layer, (nn.Conv2d, nn.Conv3d)):
            kernel = int(torch.tensor(layer.kernel_size).prod())
            per_output = layer.in_channels // layer.groups * kernel
            kind = "conv2d" if isinstance(layer, nn.Conv2d) else "conv3d"
        elif isinstance(layer, nn.Linear):
            per_output = layer.in_features
            kind = "linear"
        else:  # pragma: no cover - hooks are registered only for these types
            return
        records.append({"kind": kind, "output_elements": int(output.numel()),
                        "macs": int(output.numel() * per_output)})

    handles = [child.register_forward_hook(hook) for child in module.modules()
               if isinstance(child, (nn.Conv2d, nn.Conv3d, nn.Linear))]
    try:
        module.eval()
        with torch.inference_mode():
            module(torch.zeros(shape, device=device, dtype=dtype))
    finally:
        for handle in handles:
            handle.remove()
        for child in module.modules():
            child.training = training_state[id(child)]
    return {
        "input_shape": list(shape),
        "macs": int(sum(record["macs"] for record in records)),
        "layers": records,
        "omitted_ops": ["bias_add", "normalization", "activation", "interpolation", "memory_traffic"],
    }


_ARCH_FIELDS = {
    "name", "depth_mode", "width", "widths", "z_kernels", "z_strides",
    "residual_init", "skip_fusion", "skip_stopgrad", "skip_stop_gradient",
    "backbone", "window",
}


def _validate_architecture(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize the explicit architecture section and reject ignored fields."""

    cfg = _arch_config(config)
    architecture = _as_dict(config, "architecture")
    unknown = set(architecture) - _ARCH_FIELDS
    if unknown:
        raise ValueError(f"unknown architecture fields: {sorted(unknown)}")
    return cfg


def _native_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Map the volume schema's model section to the legacy build_model API."""

    model = _as_dict(config, "model")
    arch = _as_dict(config, "architecture")
    backbone = model.get("backbone", config.get("backbone", "tiny_test") if isinstance(config, Mapping) else "tiny_test")
    native = {
        "backbone": backbone,
        "source_repo": model.get("source_repo", config.get("source_repo", "") if isinstance(config, Mapping) else ""),
        "source_commit": model.get("source_commit", config.get("source_commit", "") if isinstance(config, Mapping) else ""),
        "weights": model.get("weights", config.get("weights", "") if isinstance(config, Mapping) else ""),
        "weights_sha256": model.get("weights_sha256", config.get("weights_sha256", "") if isinstance(config, Mapping) else ""),
        "window": (config.get("window") if isinstance(config, Mapping) else None) or arch.get("window"),
    }
    if native["window"] is None:
        native["window"] = {"mode": "none"}
    return native


def _configured_widths(config: Mapping[str, Any] | None) -> tuple[int, ...] | None:
    """Return explicitly configured four-stage NativeStudent decoder widths."""

    arch = _as_dict(config, "architecture")
    # Scalar ``architecture.width`` is the residual bottleneck, not a native
    # stage-width list.  Only the explicit list controls NativeStudent stages.
    raw = arch.get("widths")
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raise ValueError("architecture.widths must be a four-element list; scalar width is a residual bottleneck")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("architecture.widths must contain exactly four stage widths")
    widths = tuple(_positive_int(v, "architecture.widths", 1) for v in raw)
    return widths


def _validate_window(window: Any) -> dict[str, Any]:
    if window is None:
        window = {"mode": "none"}
    if not isinstance(window, Mapping):
        raise ValueError("window must be a mapping")
    normalized = validate_window_config(dict(window))
    if normalized["mode"] not in {"none", "fixed"}:
        raise ValueError("native-volume model requires a fixed or disabled window; adaptive is a separate factor")
    return normalized


def _load_native_student(
    config: Mapping[str, Any] | None,
    *,
    synthetic: bool,
    window: Any = None,
    widths: tuple[int, ...] | None = None,
    architecture_only: bool = False,
) -> tuple[NativeStudent, dict[str, Any]]:
    """Load the exact legacy NativeStudent through a patchable seam.

    Non-synthetic construction delegates all source/weight/hash validation to
    ``self_audit.model.build_model``.  It therefore fails closed when assets or
    hashes are missing, rather than silently falling back to a tiny network.
    """

    native_cfg = _native_config(config)
    native_cfg["window"] = _validate_window(window if window is not None else native_cfg.get("window"))
    if architecture_only:
        backbone = native_cfg.get("backbone")
        if backbone == "tiny_test":
            if not synthetic:
                raise ValueError("tiny_test architecture-only construction is synthetic-only")
            if widths is not None and tuple(widths) != _TINY_DECODER_WIDTHS:
                raise ValueError(f"tiny_test decoder widths must be {_TINY_DECODER_WIDTHS}")
            if widths is not None:
                model = NativeStudent(TinyEncoder(), widths=widths, window=native_cfg["window"])
            else:
                model = NativeStudent(TinyEncoder(), widths=_TINY_DECODER_WIDTHS, window=native_cfg["window"])
            return model, {"pretrained": False, "synthetic": True, "architecture_only": True}
        if backbone != "dinov3_convnext_tiny":
            raise ValueError("unsupported architecture-only backbone; no fallback architecture")
        source_repo = native_cfg.get("source_repo")
        source_commit = native_cfg.get("source_commit")
        if not isinstance(source_repo, str) or not source_repo or not isinstance(source_commit, str) or not source_commit:
            raise ValueError("architecture-only DINO construction requires pinned source_repo/source_commit")
        source = Path(source_repo).resolve()
        try:
            head = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(source), "status", "--porcelain"],
                check=True, capture_output=True, text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError("pinned DINO source repository is unavailable") from exc
        if head != source_commit or dirty:
            raise ValueError("pinned DINO source must match the clean recorded commit")
        if widths is not None and tuple(widths) != _DINO_DECODER_WIDTHS:
            raise ValueError(f"architecture-only DINO decoder widths must be {_DINO_DECODER_WIDTHS}")
        # This is deliberately a local-only, non-pretrained construction.  No
        # weight path is opened or hashed, and torch.hub never receives a URL.
        try:
            original = torch.hub.load(
                str(source), "dinov3_convnext_tiny", source="local", pretrained=False, weights=None,
            )
        except TypeError:
            original = torch.hub.load(
                str(source), "dinov3_convnext_tiny", source="local", pretrained=False,
            )
        model = NativeStudent(OfficialStages(original), window=native_cfg["window"])
        if widths is not None:
            actual_widths = tuple(int(layer.out_channels) for layer in model.lateral)
            if actual_widths != tuple(widths):
                raise ValueError(
                    f"NativeStudent decoder widths {actual_widths} do not match declared {tuple(widths)}"
                )
        return model, {
            "pretrained": False,
            "architecture_only": True,
            "source_commit": head,
            "source_repo": str(source),
        }
    if not synthetic and native_cfg.get("backbone") == "dinov3_convnext_tiny":
        required = ("source_repo", "source_commit", "weights", "weights_sha256")
        if any(not isinstance(native_cfg.get(key), str) or not native_cfg.get(key) for key in required):
            raise ValueError("non-synthetic DINO route requires source_repo, source_commit, weights and weights_sha256")
        if len(native_cfg["weights_sha256"]) != 64:
            raise ValueError("weights_sha256 must be a 64-character hash")
    # The legacy builder intentionally owns its historical defaults.  For the
    # new synthetic volume route we construct TinyEncoder directly when a
    # four-stage width tuple is requested, making every width observable rather
    # than silently retaining the fixed [8,16,24,32] values.
    if widths is not None and native_cfg.get("backbone") == "tiny_test":
        if not synthetic:
            raise ValueError("custom architecture.widths are synthetic-only for tiny_test")
        if tuple(widths) != _TINY_DECODER_WIDTHS:
            raise ValueError(f"tiny_test decoder widths must be {_TINY_DECODER_WIDTHS}")
        model = NativeStudent(TinyEncoder(), widths=widths, window=native_cfg["window"])
        provenance = {"pretrained": False, "synthetic": True, "custom_widths": list(widths)}
    else:
        if widths is not None and tuple(widths) != _DINO_DECODER_WIDTHS and native_cfg.get("backbone") != "tiny_test":
            raise ValueError(f"DINO decoder widths must be {_DINO_DECODER_WIDTHS}")
        model, provenance = build_model(native_cfg, synthetic=synthetic, window=native_cfg["window"])
    if not isinstance(model, NativeStudent):
        raise TypeError("legacy build_model did not return NativeStudent")
    if widths is not None:
        actual_widths = tuple(int(layer.out_channels) for layer in model.lateral)
        if actual_widths != tuple(widths):
            raise ValueError(
                f"NativeStudent decoder widths {actual_widths} do not match declared {tuple(widths)}"
            )
    return model, dict(provenance)


def _flatten_volume(features: Tensor, batch: int, depth: int) -> Tensor:
    return features.permute(0, 2, 1, 3, 4).reshape(batch * depth, features.shape[1], features.shape[3], features.shape[4])


def _to_volume(features: Tensor, batch: int, depth: int) -> Tensor:
    return features.reshape(batch, depth, features.shape[1], features.shape[2], features.shape[3]).permute(0, 2, 1, 3, 4).contiguous()


def _native_inputs(model: NativeStudent, x: Tensor, valid_mask: Tensor | None) -> tuple[Tensor, Tensor, Tensor | None, int, int]:
    """Reproduce NativeStudent.features preprocessing exactly without edits."""

    if x.ndim != 4 or x.shape[1] != 3 or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("expected finite normalized Bx3xHxW input")
    if valid_mask is not None and (valid_mask.shape != (x.shape[0], x.shape[2], x.shape[3]) or valid_mask.dtype is not torch.bool):
        raise ValueError("geometric valid mask must be a bool BxHxW tensor")
    h, w = x.shape[-2:]
    pad = (0, (-w) % 32, 0, (-h) % 32)
    norm = F.pad(model.input_adapter(x), pad, mode="replicate")
    raw = F.pad(x, pad, mode="replicate")
    padded_valid = None
    if model.window is not None:
        if valid_mask is None:
            valid_mask = torch.ones((x.shape[0], h, w), dtype=torch.bool, device=x.device)
        padded_valid = F.pad(valid_mask[:, None], pad, mode="constant", value=False)[:, 0]
    return norm, raw, padded_valid, h, w


class Volume25D(nn.Module):
    """G1 control: unchanged NativeStudent applied independently at each Z."""

    architecture_name = "volume25d"

    def __init__(self, *, config: Mapping[str, Any] | None = None, synthetic: bool, window: Any = None,
                 architecture_only: bool = False,
                 native_model: NativeStudent | None = None, native_provenance: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if native_model is None:
            self.slice_model, loaded_provenance = _load_native_student(
                config,
                synthetic=synthetic,
                window=window,
                widths=_configured_widths(config),
                architecture_only=architecture_only,
            )
        else:
            self.slice_model = native_model
            loaded_provenance = dict(native_provenance or {})
        self.synthetic = bool(synthetic)
        self.native_provenance = loaded_provenance

    @property
    def encoder(self) -> nn.Module:
        return self.slice_model.encoder

    @property
    def head(self) -> nn.Module:
        return self.slice_model.head

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return _groups(self, encoder=self.slice_model.encoder.parameters(), head=self.slice_model.head.parameters())

    def capacity_report(self, *, input_shape: tuple[int, ...] | None = None) -> dict[str, Any]:
        native = _analytical_conv_cost(self.slice_model)
        report: dict[str, Any] = {
            "parameters": int(sum(p.numel() for p in self.parameters())),
            "macs_per_output_voxel": int(native["macs_per_output_voxel"]),
            "native": native,
            "new_residual": {"parameters": 0, "macs_per_output_voxel": 0},
        }
        if input_shape is not None:
            report["executed_mac_proxy"] = _executed_mac_proxy(self, input_shape)
        return report

    def forward(self, image: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        _check_volume_input(image, valid_mask)
        b, _, z, h, w = image.shape
        flat = image.permute(0, 2, 1, 3, 4).reshape(b * z, 3, h, w)
        flat_valid = None
        if valid_mask is not None:
            flat_valid = valid_mask.reshape(b * z, h, w)
        logits = self.slice_model(flat, valid_mask=flat_valid)
        return logits.reshape(b, z, 4, h, w).permute(0, 2, 1, 3, 4).contiguous()


class HybridVolume(nn.Module):
    """G2/G6 hybrid using every NativeStudent multiscale stage.

    The unchanged NativeStudent modules perform input adaptation, four
    multiscale stages, lateral/up/refine decoding, fixed SemanticWindow and
    native detail fusion.  New residuals are inserted after each refined
    scale.  With zero-initialized residuals this is the old per-slice function;
    activated residuals provide learned cross-Z influence at all scales.
    """

    architecture_name = "hybrid"

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        synthetic: bool,
        width: int = 16,
        widths: tuple[int, ...] | None = None,
        depth_mode: str = "mix",
        skip_fusion: str = "legacy",
        skip_stopgrad: bool = False,
        residual_init: str = "default",
        depth_kernels: tuple[int, ...] | None = None,
        architecture_only: bool = False,
    ) -> None:
        super().__init__()
        if depth_mode not in {"off", "mix", "planar_control"}:
            raise ValueError(f"unsupported depth_mode: {depth_mode}")
        raw_window = (_as_dict(config or {}, "window") or (config or {}).get("window")) if isinstance(config, Mapping) else None
        # Hybrid experiments default to the registered fixed-window
        # comparator.  An explicitly configured ``none`` remains a valid,
        # separately named future control; it is never silently rewritten.
        hybrid_window = {"mode": "fixed"} if raw_window is None else raw_window
        normalized_window = _validate_window(hybrid_window)
        self.slice_model, self.native_provenance = _load_native_student(
            config, synthetic=synthetic, window=normalized_window,
            widths=widths if widths is not None else _configured_widths(config),
            architecture_only=architecture_only,
        )
        self.synthetic = bool(synthetic)
        self.width = _positive_int(width, "width", 16)
        native_widths = tuple(int(module.out_channels) for module in self.slice_model.lateral)
        zero_init = residual_init in {"zero", "old_function", "zero_init"}
        if depth_kernels is None:
            default_kernel = 1 if depth_mode in {"off", "planar_control"} else 3
            depth_kernels = (default_kernel,) * len(native_widths)
        if len(depth_kernels) not in {1, len(native_widths)}:
            raise ValueError("hybrid z_kernels must have one value or one per native stage")
        if len(depth_kernels) == 1:
            depth_kernels = tuple(depth_kernels) * len(native_widths)
        if depth_mode in {"off", "planar_control"} and any(int(k) != 1 for k in depth_kernels):
            raise ValueError("off/planar_control cannot mix Z; all hybrid z_kernels must have depth 1")
        with torch.random.fork_rng(devices=[]):
            # Keep the finest residual under a stable name used by earlier
            # diagnostics and expose the other three scales explicitly.
            self.body = nn.ModuleDict({
                "depth_mixer": CrossZResidual(native_widths[0], mode=depth_mode, zero_init=zero_init,
                                              hidden_channels=self.width, depth_kernel=int(depth_kernels[0])),
                "depth_residuals": nn.ModuleList(
                    CrossZResidual(channels, mode=depth_mode, zero_init=zero_init,
                                   hidden_channels=self.width, depth_kernel=int(kernel))
                    for channels, kernel in zip(native_widths[1:], depth_kernels[1:])
                ),
            })
            self.skip = SkipFusion(native_widths[0], 32, native_widths[0], mode=skip_fusion,
                                    stop_gradient=skip_stopgrad,
                                    residual_init="zero" if zero_init else "default")
        self.depth_mode = depth_mode
        self.skip_fusion = skip_fusion
        self.skip_stopgrad = bool(skip_stopgrad)
        self.residual_init = residual_init

    @property
    def encoder(self) -> nn.Module:
        return self.slice_model.encoder

    @property
    def head(self) -> nn.Module:
        return self.slice_model.head

    @property
    def detail(self) -> nn.Module:
        return self.slice_model.detail

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return _groups(self, encoder=self.encoder.parameters(), head=self.head.parameters())

    def set_frozen_intervention(self, enabled: bool = True, *, branch: str = "skip") -> None:
        """Delegate the G6 frozen-checkpoint diagnostic to the residual fusion."""

        self.skip.set_frozen_intervention(enabled, branch=branch)

    def clear_frozen_intervention(self) -> None:
        self.skip.clear_frozen_intervention()

    def branch_intervention(self, branch: str = "skip"):
        """Temporarily zero an added residual input branch during inference."""

        return self.skip.branch_intervention(branch)

    def capacity_report(self, *, input_shape: tuple[int, ...] | None = None) -> dict[str, Any]:
        residual = _analytical_conv_cost(self.body)
        skip = self.skip.capacity_report()
        native = _analytical_conv_cost(self.slice_model)
        report: dict[str, Any] = {
            "parameters": int(sum(p.numel() for p in self.parameters())),
            "macs_per_output_voxel": int(
                native["macs_per_output_voxel"] + residual["macs_per_output_voxel"] + skip["macs_per_voxel"]
            ),
            "native": native,
            "new_residual": residual,
            "skip": skip,
            "total_new_parameters": int(residual["parameters"] + skip["parameters"]),
            "total_new_macs_per_output_voxel": int(residual["macs_per_output_voxel"] + skip["macs_per_voxel"]),
        }
        if input_shape is not None:
            report["executed_mac_proxy"] = _executed_mac_proxy(self, input_shape)
        return report

    def forward(self, image: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        _check_volume_input(image, valid_mask)
        b, _, z, h, w = image.shape
        planar = image.permute(0, 2, 1, 3, 4).reshape(b * z, 3, h, w)
        flat_valid = valid_mask.reshape(b * z, h, w) if valid_mask is not None else None
        norm, raw, padded_valid, native_h, native_w = _native_inputs(self.slice_model, planar, flat_valid)
        stages = self.slice_model.encoder(norm)
        if len(stages) != 4:
            raise ValueError("four native backbone stages required")
        for i, (feature, channels) in enumerate(zip(stages, self.slice_model.encoder.channels)):
            expected = (norm.shape[-2] // (4 * 2**i), norm.shape[-1] // (4 * 2**i))
            if feature.shape[1] != channels or feature.shape[-2:] != expected:
                raise ValueError("incorrect native backbone stride; no implicit feature resizing")

        d = self.slice_model.refine[3](self.slice_model.lateral[3](stages[3]))
        d = self._residual(3, _to_volume(d, b, z))
        for i in (2, 1, 0):
            skip = self.slice_model.lateral[i](stages[i])
            d_flat = d.permute(0, 2, 1, 3, 4).reshape(b * z, d.shape[1], d.shape[-2], d.shape[-1])
            up = self.slice_model.up[i](F.interpolate(d_flat, skip.shape[-2:], mode="bilinear", align_corners=False))
            merged = skip + up
            if i == 0 and self.slice_model.window is not None:
                window_valid = None
                if padded_valid is not None:
                    window_valid = F.interpolate(padded_valid[:, None].float(), size=merged.shape[-2:], mode="area").ge(1.0 - 1e-6)[:, 0]
                merged = merged + self.slice_model.window(merged, d_flat, window_valid)
            d = self._residual(i, _to_volume(self.slice_model.refine[i](merged), b, z))

        d_flat = d.permute(0, 2, 1, 3, 4).reshape(b * z, d.shape[1], d.shape[-2], d.shape[-1])
        detail_flat = self.slice_model.detail(raw)
        detail_volume = _to_volume(detail_flat, b, z)
        detail_volume = F.interpolate(
            detail_volume.permute(0, 2, 1, 3, 4).reshape(b * z, detail_volume.shape[1], detail_volume.shape[-2], detail_volume.shape[-1]),
            size=d.shape[-2:], mode="bilinear", align_corners=False,
        )
        detail_volume = _to_volume(detail_volume, b, z)
        # The native detail branch remains present.  The explicit G6 skip
        # intervention is additive around that branch, so zero-initialized
        # controls retain exact NativeStudent behavior.
        d_for_fuse = d + self.skip(d, detail_volume)
        d_for_fuse_flat = d_for_fuse.permute(0, 2, 1, 3, 4).reshape(b * z, d_for_fuse.shape[1], d_for_fuse.shape[-2], d_for_fuse.shape[-1])
        fused = self.slice_model.fuse(torch.cat((F.interpolate(d_for_fuse_flat, raw.shape[-2:], mode="bilinear", align_corners=False), detail_flat), 1))
        fused = fused[..., :native_h, :native_w]
        logits = self.slice_model.head(fused)
        return logits.reshape(b, z, 4, native_h, native_w).permute(0, 2, 1, 3, 4).contiguous()

    def _residual(self, scale: int, feature: Tensor) -> Tensor:
        if scale == 0:
            return self.body["depth_mixer"](feature)
        return self.body["depth_residuals"][scale - 1](feature)


def _kernel3(value: Any, *, default: tuple[int, int, int] = (1, 3, 3)) -> tuple[int, int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        value = (value, value, value)
    values = tuple(int(v) for v in value)
    if len(values) != 3 or any(v <= 0 or v % 2 == 0 for v in values):
        raise ValueError("3-D kernels must be three positive odd integers")
    return values


def _stride3(value: Any, *, default: tuple[int, int, int] = (1, 1, 1)) -> tuple[int, int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        value = (value, value, value)
    values = tuple(int(v) for v in value)
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError("3-D strides must be three positive integers")
    return values


class _Residual3D(nn.Module):
    def __init__(self, channels: int, kernel: tuple[int, int, int]) -> None:
        super().__init__()
        groups = _norm_groups(channels)
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=kernel, padding=tuple(k // 2 for k in kernel)),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=kernel, padding=tuple(k // 2 for k in kernel)),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x + self.net(x))


class CNN3D(nn.Module):
    """G3/G8 scratch-initialized anisotropic residual 3-D U-Net."""

    architecture_name = "cnn3d"

    def __init__(
        self,
        *,
        width: int = 16,
        widths: Iterable[Any] | None = None,
        z_kernels: Iterable[Any] | None = None,
        z_strides: Iterable[Any] | None = None,
    ) -> None:
        super().__init__()
        width = _positive_int(width, "width", 16)
        kernels = list([(1, 3, 3), (3, 3, 3)] if z_kernels is None else z_kernels)
        strides = list([(1, 1, 1), (1, 1, 1)] if z_strides is None else z_strides)
        if not kernels or not strides:
            raise ValueError("z_kernels and z_strides must be non-empty")
        if len(kernels) != len(strides):
            raise ValueError("z_kernels and z_strides must have equal lengths")
        if widths is None:
            stage_widths = [width * min(2**i, 4) for i in range(len(kernels))]
        else:
            stage_widths = [_positive_int(v, "widths", width) for v in widths]
            if len(stage_widths) != len(kernels):
                raise ValueError("cnn3d architecture.widths must match z_kernels length")
        self.encoder = nn.ModuleList()
        self.down_strides: list[tuple[int, int, int]] = []
        # Full 3-D CNN consumes only the center image channel.  The volume
        # loader's three channels are the z-neighbour triplet used by the
        # 2.5-D/hybrid controls; using side channels here would be an
        # unreported input-policy factor.
        in_channels = 1
        for i, (kernel_spec, stride_spec) in enumerate(zip(kernels, strides)):
            kernel = _kernel3(kernel_spec)
            stride = _stride3(stride_spec)
            out_channels = stage_widths[i]
            self.encoder.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=kernel, stride=stride,
                              padding=tuple(k // 2 for k in kernel)),
                    nn.GroupNorm(_norm_groups(out_channels), out_channels),
                    nn.GELU(),
                    _Residual3D(out_channels, kernel),
                )
            )
            self.down_strides.append(stride)
            in_channels = out_channels
        self.body = nn.Sequential(
            _Residual3D(in_channels, _kernel3(kernels[-1] if kernels else None)),
            _Residual3D(in_channels, _kernel3(kernels[-1] if kernels else None)),
        )
        self.decoder = nn.ModuleList()
        decoder_in = in_channels
        for i in reversed(range(max(0, len(kernels) - 1))):
            skip_channels = stage_widths[i]
            self.decoder.append(_Decoder3D(decoder_in, skip_channels, skip_channels, _kernel3(kernels[i])))
            decoder_in = skip_channels
        self.head = nn.Conv3d(decoder_in, 4, kernel_size=1)
        self.scratch_initialization = True
        self.z_kernels = tuple(_kernel3(k) for k in kernels)
        self.z_strides = tuple(self.down_strides)
        self.stage_widths = tuple(stage_widths)

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return _groups(self, encoder=self.encoder.parameters(), head=self.head.parameters())

    def capacity_report(self, *, input_shape: tuple[int, ...] | None = None) -> dict[str, Any]:
        report: dict[str, Any] = _analytical_conv_cost(self)
        if input_shape is not None:
            report["executed_mac_proxy"] = _executed_mac_proxy(self, input_shape)
        return report

    def forward(self, image: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        _check_volume_input(image, valid_mask)
        target_size = image.shape[2:]
        x = image[:, 1:2]
        skips: list[Tensor] = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        x = self.body(x)
        for decoder, skip in zip(self.decoder, reversed(skips[:-1])):
            x = decoder(x, skip)
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        return self.head(x)


class _Decoder3D(nn.Module):
    """Learned upsample/fusion block with an explicit encoder skip."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 kernel: tuple[int, int, int]) -> None:
        super().__init__()
        self.up = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=kernel,
                      padding=tuple(k // 2 for k in kernel)),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.GELU(),
            _Residual3D(out_channels, kernel),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.up(x)
        return self.fuse(torch.cat((x, skip), dim=1))


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _check_volume_input(image: Tensor, valid_mask: Tensor | None) -> None:
    if not isinstance(image, Tensor) or image.ndim != 5 or image.shape[1] != 3:
        raise ValueError("expected Bx3xZxHxW volume input")
    if not torch.isfinite(image).all():
        raise ValueError("volume input must be finite")
    if image.shape[0] <= 0 or image.shape[2] <= 0 or image.shape[3] <= 0 or image.shape[4] <= 0:
        raise ValueError("volume dimensions must be positive")
    if valid_mask is not None:
        if valid_mask.shape != (image.shape[0], image.shape[2], image.shape[3], image.shape[4]):
            raise ValueError("valid_mask must be BxZxHxW")
        if valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be bool")


def build_volume_model(
    config: Mapping[str, Any] | None,
    *,
    synthetic: bool = False,
    architecture_only: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build an explicit native-volume architecture and provenance record.

    ``architecture_only=True`` is an export/deployment seam: DINO builds are
    instantiated from the pinned clean local source with
    ``torch.hub.load(..., source='local', pretrained=False)`` and never inspect
    the configured pretrained-weight path. The normal training route retains
    strict source/weight/hash checks; callers must load a complete exported
    state dict strictly after an architecture-only construction.
    """

    cfg = _validate_architecture(config)
    model_cfg = _as_dict(config, "model")
    name = str(cfg.get("name", cfg.get("architecture_name", "volume25d"))).lower()
    default_depth_mode = "mix" if name == "hybrid" else "off"
    depth_mode = str(cfg.get("depth_mode", default_depth_mode))
    raw_widths = cfg.get("widths")
    if raw_widths is None:
        width = _positive_int(cfg.get("width", 16), "width", 16)
        widths = None
    elif isinstance(raw_widths, (list, tuple)):
        widths = tuple(_positive_int(value, "widths", 16) for value in raw_widths)
        width = _positive_int(cfg.get("width", widths[0] if widths else 16), "width", 16)
    else:
        raise ValueError("architecture.widths must be a list when provided")
    skip_fusion = str(cfg.get("skip_fusion", "legacy"))
    skip_stopgrad = bool(cfg.get("skip_stopgrad", cfg.get("skip_stop_gradient", False)))
    residual_init = str(cfg.get("residual_init", "default"))
    if name in {"volume25d", "2.5d", "volume_25d"}:
        # Volume25D can use explicit four-stage widths on the synthetic route;
        # depth/skip controls are required to stay neutral because this arm is
        # the unchanged per-slice control.
        if depth_mode != "off" or skip_fusion != "legacy" or skip_stopgrad:
            raise ValueError("volume25d requires depth_mode=off, skip_fusion=legacy and skip_stopgrad=false")
        if residual_init not in {"default", "zero", "zero_init", "old_function"}:
            raise ValueError("volume25d residual_init is not applicable; use the neutral default/zero value")
        if cfg.get("width") is not None:
            raise ValueError("volume25d does not support scalar architecture.width; use four native stage widths")
        if cfg.get("z_kernels") is not None and any(_kernel3(spec) != (1, 3, 3) for spec in cfg["z_kernels"]):
            raise ValueError("volume25d does not support z_kernels; use neutral (1,3,3) values")
        if cfg.get("z_strides") is not None and any(_stride3(spec) != (1, 1, 1) for spec in cfg["z_strides"]):
            raise ValueError("volume25d does not support z_strides; use neutral (1,1,1) values")
        if widths is not None and len(widths) != 4:
            raise ValueError("volume25d architecture.widths must contain four stages")
        model = Volume25D(
            config=config,
            synthetic=synthetic,
            window=cfg.get("window"),
            architecture_only=architecture_only,
        )
        name = "volume25d"
    elif name in {"hybrid", "hybrid2d3d", "hybrid_2d3d"}:
        if widths is not None and len(widths) != 4:
            raise ValueError("hybrid architecture.widths must contain four native stages")
        z_specs = cfg.get("z_kernels")
        if z_specs is None:
            depth_kernels = None
        else:
            normalized_kernels = tuple(_kernel3(spec) for spec in z_specs)
            if any(kernel[1:] != (1, 1) for kernel in normalized_kernels):
                raise ValueError(
                    "hybrid z_kernels only implement depthwise (kz,1,1); XY entries must be exactly (1,1)"
                )
            depth_kernels = tuple(kernel[0] for kernel in normalized_kernels)
        z_strides = cfg.get("z_strides")
        if z_strides is not None:
            normalized_strides = tuple(_stride3(spec) for spec in z_strides)
            if any(stride != (1, 1, 1) for stride in normalized_strides):
                raise ValueError("hybrid z_strides are not supported; use (1,1,1) explicitly")
        model = HybridVolume(
            config=config,
            synthetic=synthetic,
            width=width,
            widths=widths,
            depth_mode=depth_mode,
            skip_fusion=skip_fusion,
            skip_stopgrad=skip_stopgrad,
            residual_init=str(cfg.get("residual_init", "default")),
            depth_kernels=depth_kernels,
            architecture_only=architecture_only,
        )
        name = "hybrid"
    elif name in {"cnn3d", "full3d", "full_3d"}:
        if depth_mode != "off" or skip_fusion != "legacy" or skip_stopgrad:
            raise ValueError("cnn3d does not support hybrid depth/skip controls; use neutral values")
        if residual_init not in {"default", "zero", "zero_init", "old_function"}:
            raise ValueError("cnn3d residual_init is not applicable; use the neutral default/zero value")
        model = CNN3D(
            width=width,
            widths=widths,
            z_kernels=cfg.get("z_kernels"),
            z_strides=cfg.get("z_strides"),
        )
        name = "cnn3d"
    else:
        raise ValueError(f"unsupported volume architecture: {name}")
    groups = model.parameter_groups()
    native = getattr(model, "slice_model", None)
    if native is not None:
        encoder_widths = tuple(int(value) for value in native.encoder.channels)
        decoder_widths = tuple(int(layer.out_channels) for layer in native.lateral)
    elif isinstance(model, CNN3D):
        encoder_widths = tuple(int(value) for value in model.stage_widths)
        decoder_widths = tuple(int(value) for value in model.stage_widths)
    else:
        encoder_widths = ()
        decoder_widths = ()
    native_provenance = getattr(model, "native_provenance", {})
    if name == "cnn3d":
        initialization = "scratch"
    elif architecture_only:
        initialization = "architecture_only_uninitialized"
    elif native_provenance.get("pretrained"):
        initialization = "native_dino_pretrained"
    elif native_provenance.get("synthetic"):
        initialization = "native_tiny_synthetic"
    else:
        initialization = "native"
    provenance = {
        "architecture": name,
        "depth_mode": getattr(model, "depth_mode", "off" if name == "volume25d" else None),
        "skip_fusion": getattr(model, "skip_fusion", skip_fusion),
        "skip_stopgrad": bool(getattr(model, "skip_stopgrad", skip_stopgrad)),
        "synthetic": bool(synthetic),
        "external_teacher": False,
        "architecture_only": bool(architecture_only),
        "initialization": initialization,
        "encoder_widths": list(encoder_widths),
        "decoder_widths": list(decoder_widths),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "parameter_group_counts": {key: int(sum(p.numel() for p in value)) for key, value in groups.items()},
        # Synthetic software receipts include an executed Conv/Linear MAC proxy
        # at one declared shape. Real DINO construction avoids an unsolicited
        # forward; callers can request ``capacity_report(input_shape=...)``.
        "capacity": (
            model.capacity_report(input_shape=(1, 3, 3, 32, 32))
            if synthetic and hasattr(model, "capacity_report")
            else (model.capacity_report() if hasattr(model, "capacity_report") else {})
        ),
        "backbone": model_cfg.get("backbone", cfg.get("backbone", "tiny_test")),
        "native_provenance": native_provenance,
    }
    if len({id(p) for values in groups.values() for p in values}) != sum(len(values) for values in groups.values()):
        raise RuntimeError("volume model parameter groups overlap")
    return model, provenance


# Stable descriptive names for registry rows; these are aliases, not separate
# architectures or checkpoints.
FullCNN = CNN3D
Hybrid2D3D = HybridVolume
Volume25DModel = Volume25D

__all__ = [
    "CNN3D",
    "FullCNN",
    "Hybrid2D3D",
    "HybridVolume",
    "Volume25D",
    "Volume25DModel",
    "build_volume_model",
]
