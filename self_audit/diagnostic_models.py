"""Immutable feature-tap and teacher-suffix models for gradient diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from self_audit.bridge_alignment import BridgeAlignmentArtifact
from self_audit.compact_runtime import compact_bilinear


def _require_eval(module: nn.Module, name: str) -> None:
    if any(child.training for child in module.modules()):
        raise ValueError(f"{name} must already be in eval mode; diagnostics do not mutate model mode")


def _require_deterministic_cuda(tensor: Tensor, name: str) -> None:
    if tensor.device.type == "cuda" and not torch.are_deterministic_algorithms_enabled():
        raise ValueError(f"{name} CUDA VJPs require compact_runtime.deterministic_runtime(True)")


def _check_normalized_input(x: Tensor, channels: int, name: str) -> None:
    if not isinstance(x, Tensor) or x.ndim != 4 or x.shape[1] != channels:
        raise ValueError(f"{name} must be [B,{channels},H,W]")
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be finite floating point input")
    if x.numel() and (x.min() < -1e-5 or x.max() > 1.0 + 1e-5):
        raise ValueError(f"{name} must use the existing [0,1] normalized input")
    if min(x.shape[-2:]) <= 0:
        raise ValueError(f"{name} spatial dimensions must be positive")


def _module_sequence(owner: Any, name: str, count: int, label: str) -> tuple[nn.Module, ...]:
    """Validate an indexable module sequence without relying on import identity."""
    value = getattr(owner, name, None)
    if value is None or not hasattr(value, "__len__") or not hasattr(value, "__getitem__"):
        raise TypeError(f"{label} must expose module sequence {name}[0:{count}]")
    try:
        modules = tuple(value[index] for index in range(len(value)))
    except (TypeError, KeyError, IndexError) as exc:
        raise TypeError(f"{label}.{name} is not an indexable module sequence") from exc
    if len(modules) != count or any(not isinstance(module, nn.Module) for module in modules):
        raise TypeError(f"{label}.{name} must contain exactly {count} torch modules")
    return modules


def _compact_student_protocol(model: Any) -> tuple[nn.Module, nn.Module, nn.Module, int, tuple[int, ...]]:
    """Validate the strict-loaded compact tap contract structurally."""
    if not isinstance(model, nn.Module):
        raise TypeError("CompactStudentTap requires a strict-loaded torch module")
    encoder = getattr(model, "encoder", None)
    decoder = getattr(model, "decoder", None)
    head = getattr(model, "head", None)
    if (not isinstance(encoder, nn.Module) or not callable(encoder)
            or not isinstance(decoder, nn.Module) or not callable(decoder)
            or not isinstance(head, nn.Module) or not callable(head)
            or not callable(getattr(model, "_pad_geometry", None))):
        raise TypeError("compact model must expose encoder, decoder, head, and _pad_geometry")
    channels = getattr(model, "in_channels", getattr(encoder, "in_channels", None))
    widths = getattr(model, "widths", getattr(encoder, "widths", None))
    if type(channels) is not int or channels <= 0:
        raise TypeError("compact model must declare a positive integer in_channels")
    if (not isinstance(widths, Sequence) or isinstance(widths, (str, bytes)) or not widths
            or any(type(value) is not int or value <= 0 for value in widths)):
        raise TypeError("compact model must declare positive integer stage widths")
    return encoder, decoder, head, channels, tuple(widths)


def _native_teacher_protocol(
    teacher: Any, *, require_suffix: bool,
) -> tuple[tuple[nn.Module, ...], tuple[nn.Module, ...], tuple[nn.Module, ...]]:
    """Validate current or historical NativeStudent modules by their graph."""
    if not isinstance(teacher, nn.Module):
        raise TypeError("native teacher must be a strict-loaded torch module")
    input_adapter = getattr(teacher, "input_adapter", None)
    encoder = getattr(teacher, "encoder", None)
    if not isinstance(input_adapter, nn.Module) or not isinstance(encoder, nn.Module) or not callable(encoder):
        raise TypeError("native teacher must expose input_adapter and encoder modules")
    lateral = _module_sequence(teacher, "lateral", 4, "native teacher")
    refine = _module_sequence(teacher, "refine", 4, "native teacher")
    up = _module_sequence(teacher, "up", 3, "native teacher")
    channels = getattr(encoder, "channels", None)
    if (not isinstance(channels, Sequence) or isinstance(channels, (str, bytes))
            or len(channels) != 4 or any(type(value) is not int or value <= 0 for value in channels)):
        raise TypeError("native teacher encoder must declare four positive stage channel widths")
    if require_suffix:
        detail = getattr(teacher, "detail", None)
        fuse = getattr(teacher, "fuse", None)
        head = getattr(teacher, "head", None)
        if not isinstance(detail, nn.Module) or not isinstance(fuse, nn.Module) or not isinstance(head, nn.Module):
            raise TypeError("native teacher suffix must expose detail, fuse, and head modules")
        detail_layers = _module_sequence(teacher, "detail", len(detail), "native teacher")
        if (not detail_layers or not hasattr(detail_layers[0], "out_channels")
                or not hasattr(lateral[0], "out_channels")):
            raise TypeError("native teacher suffix must declare detail/lateral channel counts")
        if not hasattr(fuse, "__len__") or not len(fuse) or not hasattr(fuse[0], "in_channels"):
            raise TypeError("native teacher fuse must declare its input channel count")
    if getattr(teacher, "window", None) is not None or getattr(teacher, "teacher_memory", None) is not None:
        raise ValueError("teacher suffix alignment requires the declared plain NativeStudent graph")
    return lateral, refine, up


@dataclass(frozen=True)
class StudentFeatureContext:
    """One compact forward's bottleneck, detached skips, and native geometry."""

    h: Tensor
    skips: tuple[Tensor, ...]
    input_hw: tuple[int, int]
    padded_hw: tuple[int, int]
    valid_mask: Tensor | None
    model_token: int


class CompactStudentTap:
    """Expose the true ``CompactStudent.encoder[-1]`` bottleneck.

    Every skip is copied and detached at extraction time. Decoding replaces
    only the bottleneck tensor and uses the model's native decoder and head.
    """

    def __init__(self, model: nn.Module) -> None:
        encoder, decoder, head, channels, widths = _compact_student_protocol(model)
        _require_eval(model, "compact student")
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        self.in_channels = channels
        self.widths = widths

    def extract(self, x: Tensor, valid_mask: Tensor | None = None) -> StudentFeatureContext:
        _require_eval(self.model, "compact student")
        _check_normalized_input(x, self.in_channels, "compact input")
        if valid_mask is not None:
            if (not isinstance(valid_mask, Tensor) or valid_mask.dtype is not torch.bool
                    or valid_mask.shape != (x.shape[0], x.shape[2], x.shape[3])):
                raise ValueError("valid_mask must be boolean [B,H,W] and match the native input")
            valid_copy = valid_mask.detach().to(device=x.device).clone()
        else:
            valid_copy = None
        x_pad, orig_h, orig_w = self.model._pad_geometry(x)
        if (not isinstance(x_pad, Tensor) or x_pad.ndim != 4 or x_pad.shape[:2] != x.shape[:2]
                or (orig_h, orig_w) != tuple(x.shape[-2:])):
            raise ValueError("compact _pad_geometry must return a padded tensor and native input dimensions")
        with torch.no_grad():
            stages = self.encoder(x_pad)
        if not isinstance(stages, (tuple, list)) or len(stages) != len(self.widths) or not stages:
            raise ValueError("compact encoder did not return the declared stage list")
        for index, (stage, channels) in enumerate(zip(stages, self.widths, strict=True)):
            expected = (x_pad.shape[-2] // (2**index), x_pad.shape[-1] // (2**index))
            if stage.shape[1] != channels or stage.shape[-2:] != expected:
                raise ValueError("compact encoder stage shape differs from native geometry")
            if not torch.isfinite(stage).all():
                raise FloatingPointError("compact encoder produced non-finite features")
        return StudentFeatureContext(
            h=stages[-1].detach().clone(),
            skips=tuple(stage.detach().clone() for stage in stages[:-1]),
            input_hw=(orig_h, orig_w),
            padded_hw=tuple(x_pad.shape[-2:]),
            valid_mask=valid_copy,
            model_token=id(self.model),
        )

    def decode(self, context: StudentFeatureContext, h: Tensor | None = None) -> Tensor:
        _require_eval(self.model, "compact student")
        if not isinstance(context, StudentFeatureContext) or context.model_token != id(self.model):
            raise ValueError("feature context was not created by this CompactStudentTap")
        feature = context.h if h is None else h
        if feature.shape != context.h.shape or feature.device != context.h.device:
            raise ValueError("replacement bottleneck must match the extracted feature shape and device")
        if feature.dtype != context.h.dtype or not torch.isfinite(feature).all():
            raise ValueError("replacement bottleneck must be finite and use the extracted dtype")
        stages = [*context.skips, feature]
        decoded = self.decoder(stages)
        orig_h, orig_w = context.input_hw
        decoded = decoded[..., :orig_h, :orig_w]
        logits = self.head(decoded)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("native compact decoder produced non-finite logits")
        return logits


def feature_support_from_valid(
    valid_mask: Tensor | None,
    padded_hw: Sequence[int],
    feature_hw: Sequence[int] | Tensor,
) -> Tensor:
    """FALSE-pad native validity to the encoder canvas before bottleneck resize.

    ``padded_hw`` is ``StudentFeatureContext.padded_hw`` and ``feature_hw`` is
    either a BCHW feature tensor or ``(batch,height,width)``. For a 33x35
    native mask on a 40x40 canvas, this returns 16 full stride-8 cells.
    """
    if isinstance(feature_hw, Tensor):
        if feature_hw.ndim != 4:
            raise ValueError("feature tensor must be BCHW")
        batch, height, width = feature_hw.shape[0], feature_hw.shape[-2], feature_hw.shape[-1]
        device = feature_hw.device
    else:
        if len(feature_hw) != 3:
            raise ValueError("feature_hw must be (batch,height,width) or a BCHW tensor")
        batch, height, width = (int(value) for value in feature_hw)
        device = valid_mask.device if valid_mask is not None else torch.device("cpu")
    if len(padded_hw) != 2:
        raise ValueError("padded_hw must be (height,width)")
    padded_h, padded_w = (int(value) for value in padded_hw)
    if min(batch, height, width, padded_h, padded_w) <= 0 or padded_h % height or padded_w % width:
        raise ValueError("feature geometry must divide the positive padded canvas exactly")
    if valid_mask is None:
        return torch.ones((batch, height, width), dtype=torch.bool, device=device)
    if (valid_mask.dtype is not torch.bool or valid_mask.ndim != 3
            or valid_mask.shape[0] != batch
            or valid_mask.shape[-2] > padded_h or valid_mask.shape[-1] > padded_w):
        raise ValueError("valid_mask must be boolean [B,H,W] within the padded feature canvas")
    native_h, native_w = valid_mask.shape[-2:]
    padded = F.pad(valid_mask[:, None].to(device=device, dtype=torch.float32),
                   (0, padded_w - native_w, 0, padded_h - native_h), value=0.0)
    coverage = F.interpolate(padded, size=(height, width), mode="area")[:, 0]
    return coverage.ge(1.0 - 1e-6)


@torch.no_grad()
def extract_teacher_pre_refine0(teacher: nn.Module, x: Tensor) -> Tensor:
    """Extract the full teacher's image-derived input to ``refine[0]`` for TRAIN fit.

    This is a projector-fit target, not the diagnostic bridge forward. The
    bridge itself starts from compact ``h`` and runs only ``refine[0]`` onward.
    """
    lateral, refine, up = _native_teacher_protocol(teacher, require_suffix=False)
    _require_eval(teacher, "native teacher")
    _check_normalized_input(x, 3, "teacher input")
    h, w = x.shape[-2:]
    pad = (0, (-w) % 32, 0, (-h) % 32)
    x_pad = F.pad(teacher.input_adapter(x), pad, mode="replicate")
    stages = teacher.encoder(x_pad)
    if len(stages) != 4:
        raise ValueError("teacher alignment requires four native encoder stages")
    if len(teacher.encoder.channels) != 4:
        raise ValueError("teacher alignment requires four declared encoder channel widths")
    for index, (stage, channels) in enumerate(zip(stages, teacher.encoder.channels, strict=True)):
        expected_hw = (x_pad.shape[-2] // (4 * 2**index), x_pad.shape[-1] // (4 * 2**index))
        if stage.shape[1] != channels or stage.shape[-2:] != expected_hw:
            raise ValueError("teacher encoder stage differs from the declared stride-4 decoder geometry")
    d = refine[3](lateral[3](stages[3]))
    for index in (2, 1):
        skip = lateral[index](stages[index])
        d = refine[index](skip + up[index](
            compact_bilinear(d, skip.shape[-2:])))
    skip0 = lateral[0](stages[0])
    pre_refine0 = skip0 + up[0](compact_bilinear(d, skip0.shape[-2:]))
    if not torch.isfinite(pre_refine0).all():
        raise FloatingPointError("teacher pre-refine[0] feature is non-finite")
    return pre_refine0.detach()


class TeacherSuffixBridge(nn.Module):
    """Frozen NativeStudent suffix bridge with no raw-image/detail bypass.

    Graph: compact bottleneck -> verified P + declared bilinear alignment ->
    teacher ``refine[0]`` -> bilinear native upsample -> ``fuse`` with an
    all-zero detail tensor -> teacher ``head``. This is explicitly a teacher
    decoder suffix, not the full teacher decoder.
    """

    def __init__(
        self,
        teacher: nn.Module,
        alignment: BridgeAlignmentArtifact,
        *,
        expected_identities: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        lateral, _, _ = _native_teacher_protocol(teacher, require_suffix=True)
        if not isinstance(alignment, BridgeAlignmentArtifact):
            raise TypeError("a verified TRAIN-fitted BridgeAlignmentArtifact is required; random P is rejected")
        alignment.verify(expected_identities=expected_identities)
        _require_eval(teacher, "native teacher")
        weight = alignment.projector_weight
        bias = alignment.projector_bias
        teacher_stride4_channels = lateral[0].out_channels
        if weight.shape[0] != teacher_stride4_channels:
            raise ValueError("fitted P output channels do not match teacher refine[0]")
        self.teacher = teacher
        # Module construction is isolated so adapter creation cannot consume caller RNG.
        with torch.random.fork_rng(devices=[]):
            projector = nn.Conv2d(int(weight.shape[1]), int(weight.shape[0]), kernel_size=1, bias=True)
        with torch.no_grad():
            projector.weight.copy_(weight)
            projector.bias.copy_(bias)
        reference_parameter = next(teacher.parameters())
        if reference_parameter.dtype != weight.dtype:
            raise ValueError("teacher dtype differs from the hash-pinned float32 bridge projector")
        projector.to(device=reference_parameter.device)
        projector.weight.requires_grad_(False)
        projector.bias.requires_grad_(False)
        self.projector = projector
        self.alignment_sha256 = alignment.artifact_sha256
        detail_channels = teacher.detail[0].out_channels
        if teacher.fuse[0].in_channels != teacher_stride4_channels + detail_channels:
            raise ValueError("teacher fuse input does not match the declared zero-detail suffix")
        self.detail_channels = detail_channels
        self.eval()

    def forward(self, h: Tensor, input_hw: Sequence[int]) -> Tensor:
        _require_eval(self.teacher, "native teacher")
        if not isinstance(h, Tensor) or h.ndim != 4 or h.shape[1] != self.projector.in_channels:
            raise ValueError("bridge h must be BCHW with the fitted compact bottleneck channel count")
        if len(input_hw) != 2 or any(type(v) is not int or v <= 0 for v in input_hw):
            raise ValueError("input_hw must be positive (height,width)")
        if not torch.is_floating_point(h) or not torch.isfinite(h).all():
            raise FloatingPointError("bridge h must be finite floating point features")
        _require_deterministic_cuda(h, "teacher suffix bridge")
        if h.device != self.projector.weight.device or h.dtype != self.projector.weight.dtype:
            raise ValueError("bridge h, fitted P, and teacher must share the locked diagnostic device and dtype")
        height, width = int(input_hw[0]), int(input_hw[1])
        padded_h = height + (-height) % 32
        padded_w = width + (-width) % 32
        stride4_hw = (padded_h // 4, padded_w // 4)
        aligned = compact_bilinear(h, stride4_hw)
        projected = self.projector(aligned)
        decoded = self.teacher.refine[0](projected)
        upsampled = compact_bilinear(decoded, (padded_h, padded_w))
        # Primary bridge deliberately supplies no raw input or detail features.
        zero_detail = h.new_zeros((h.shape[0], self.detail_channels, padded_h, padded_w))
        fused = self.teacher.fuse(torch.cat((upsampled, zero_detail), dim=1))
        fused = fused[..., :height, :width]
        logits = self.teacher.head(fused)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("teacher suffix bridge produced non-finite logits")
        return logits


@dataclass(frozen=True)
class BridgeDependenceResult:
    keep_logits: Tensor
    zero_logits: Tensor
    shuffled_logits: Tensor
    keep_vs_zero_mean_abs: Tensor
    keep_vs_shuffle_mean_abs: Tensor
    h_vjp_norm: Tensor
    donor_indices: tuple[int, ...]


def _derangement(count: int, seed: int) -> Tensor:
    if count < 2:
        raise ValueError("cross-patient h shuffle needs at least two patients")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)
    identity = torch.arange(count)
    for _ in range(256):
        order = torch.randperm(count, generator=generator)
        if not torch.equal(order, identity):
            fixed = order.eq(identity)
            if not fixed.any():
                return order
    # A one-position rotation is a deterministic derangement for count >= 2.
    return identity.roll(1)


def check_bridge_h_dependence(
    bridge: TeacherSuffixBridge,
    h: Tensor,
    input_hw: Sequence[int],
    patient_ids: Sequence[str],
    seed: int,
) -> BridgeDependenceResult:
    """Measure keep/zero/cross-patient-shuffle effects and an h-to-logit VJP."""
    if not isinstance(bridge, TeacherSuffixBridge):
        raise TypeError("h-dependence check requires TeacherSuffixBridge")
    if h.ndim != 4 or h.shape[0] < 2 or len(patient_ids) != h.shape[0]:
        raise ValueError("h-dependence requires a batch of at least two patient feature maps")
    if any(not isinstance(value, str) or not value.strip() for value in patient_ids):
        raise ValueError("patient IDs must be nonempty strings")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("cross-patient h check requires unique patient IDs in the batch")
    _require_eval(bridge, "teacher suffix bridge")
    _require_deterministic_cuda(h, "bridge h-dependence")
    donors = _derangement(h.shape[0], seed)
    device_indices = donors.to(h.device)
    shuffled_h = h.detach().index_select(0, device_indices)
    probe_h = h.detach().clone().requires_grad_(True)
    devices = [h.device] if h.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        keep = bridge(probe_h, input_hw)
        with torch.no_grad():
            zero = bridge(torch.zeros_like(h), input_hw)
            shuffled = bridge(shuffled_h, input_hw)
        generator = torch.Generator(device="cpu")
        generator.manual_seed((int(seed) ^ 0x4F1BBCDC) & 0x7FFF_FFFF_FFFF_FFFF)
        cotangent = torch.randn(tuple(keep.shape), generator=generator, dtype=torch.float32).to(
            device=keep.device, dtype=keep.dtype)
        vjp = torch.autograd.grad(keep, probe_h, grad_outputs=cotangent, retain_graph=False, create_graph=False)[0]
    reduce_dims = tuple(range(1, keep.ndim))
    keep_zero = (keep.detach() - zero).abs().mean(dim=reduce_dims)
    keep_shuffle = (keep.detach() - shuffled).abs().mean(dim=reduce_dims)
    vjp_norm = torch.linalg.vector_norm(vjp.double().flatten(1), dim=1)
    if not all(torch.isfinite(value).all() for value in (keep, zero, shuffled, keep_zero, keep_shuffle, vjp_norm)):
        raise FloatingPointError("non-finite bridge h-dependence result")
    return BridgeDependenceResult(
        keep_logits=keep.detach(),
        zero_logits=zero.detach(),
        shuffled_logits=shuffled.detach(),
        keep_vs_zero_mean_abs=keep_zero.detach(),
        keep_vs_shuffle_mean_abs=keep_shuffle.detach(),
        h_vjp_norm=vjp_norm.detach(),
        donor_indices=tuple(int(value) for value in donors.tolist()),
    )


__all__ = [
    "BridgeDependenceResult",
    "CompactStudentTap",
    "StudentFeatureContext",
    "TeacherSuffixBridge",
    "check_bridge_h_dependence",
    "extract_teacher_pre_refine0",
    "feature_support_from_valid",
]
