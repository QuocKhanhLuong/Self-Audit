"""Exact-current-KD VJPs and D0-D5 feature-only gradient corrections."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class InvalidGradientDiagnostic(FloatingPointError):
    """A non-finite diagnostic is invalid and must not be scored as a fallback."""

    status = "INVALID_NONFINITE"


def _finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise InvalidGradientDiagnostic(f"non-finite {name}; gradient diagnostic is invalid")


class KDLossContext:
    """Detached common target, confidence mask, pixel weights, and denominator.

    The loss matches the current compact soft-KD objective: sharpen q by T,
    evaluate soft cross-entropy at logits/T, multiply by T^2 and the declared
    coefficient, then divide by the one frozen valid/confident denominator.
    All D1/D2/D4/D5 cotangents come from autograd of this same objective.
    """

    def __init__(
        self,
        target_probs: Tensor,
        valid_mask: Tensor,
        *,
        temperature: float = 1.0,
        coefficient: float = 1.0,
        confidence: float | None = 0.0,
        class_weights: Sequence[float] | None = None,
        denominator: str = "valid_pixel_count",
        pixel_weights: Tensor | None = None,
    ) -> None:
        if not isinstance(target_probs, Tensor) or target_probs.ndim != 4 or target_probs.shape[1] < 2:
            raise ValueError("target_probs must be [B,C,H,W]")
        if not isinstance(valid_mask, Tensor) or valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be boolean [B,H,W]")
        if valid_mask.shape != (target_probs.shape[0], target_probs.shape[2], target_probs.shape[3]):
            raise ValueError("valid_mask must match the target batch and native geometry")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or not math.isfinite(float(coefficient)) or coefficient < 0:
            raise ValueError("coefficient must be finite and non-negative")
        if confidence is not None and (
                isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence)) or not 0 <= confidence <= 1):
            raise ValueError("confidence must be null or lie in [0,1]")
        if denominator not in {"valid_pixel_count", "valid_weight_sum"}:
            raise ValueError("denominator must be valid_pixel_count or valid_weight_sum")
        target = target_probs.detach().clone()
        valid = valid_mask.detach().to(device=target.device).clone()
        _finite("teacher target", target)
        if (target < 0).any():
            raise ValueError("teacher probabilities must be non-negative")
        if valid.any():
            sums = target.sum(dim=1)[valid]
            if not torch.allclose(sums, torch.ones_like(sums), atol=1e-3, rtol=0):
                raise ValueError("teacher targets must sum to one on valid pixels")
        if pixel_weights is None:
            weights = torch.ones(valid.shape, dtype=torch.float32, device=target.device)
        else:
            if not isinstance(pixel_weights, Tensor) or pixel_weights.shape != valid.shape:
                raise ValueError("pixel_weights must match valid_mask [B,H,W]")
            if not torch.is_floating_point(pixel_weights):
                raise ValueError("pixel_weights must be floating point")
            weights = pixel_weights.detach().to(device=target.device, dtype=torch.float32).clone()
            _finite("KD pixel weights", weights)
            if (weights < 0).any():
                raise ValueError("KD pixel weights must be non-negative")
        teacher_confidence, teacher_class = target.max(dim=1)
        raw_confident = torch.ones_like(valid) if confidence is None else teacher_confidence.ge(float(confidence))
        active = valid & raw_confident
        if class_weights is not None:
            if (not isinstance(class_weights, Sequence) or len(class_weights) != target.shape[1]
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(float(value)) or value < 0 for value in class_weights)
                    or not any(float(value) > 0 for value in class_weights)):
                raise ValueError("class_weights must contain finite nonnegative mass for every target class")
            class_weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=target.device)
            weights = weights * class_weight_tensor[teacher_class]
        effective_weights = weights * active.to(weights.dtype)
        fixed_denominator = (float(active.sum().item()) if denominator == "valid_pixel_count"
                             else float(effective_weights.double().sum().item()))
        if not math.isfinite(fixed_denominator):
            raise InvalidGradientDiagnostic("non-finite fixed KD denominator")
        if abs(float(temperature) - 1.0) > 1e-7:
            soft_targets = F.softmax(torch.log(target.clamp(min=1e-12)) / float(temperature), dim=1)
        else:
            soft_targets = target
        self.target_probs = target
        self.soft_targets = soft_targets.detach().clone()
        self.valid_mask = valid
        self.confidence_mask = active.detach().clone()
        self.pixel_weights = effective_weights.detach().clone()
        self.denominator = fixed_denominator
        self.denominator_policy = denominator
        self.temperature = float(temperature)
        self.coefficient = float(coefficient)
        self.confidence = None if confidence is None else float(confidence)
        self.class_weights = None if class_weights is None else tuple(float(value) for value in class_weights)

    def __call__(self, logits: Tensor) -> Tensor:
        if not isinstance(logits, Tensor) or logits.shape != self.target_probs.shape:
            raise ValueError("KD logits must match the fixed target [B,C,H,W]")
        _finite("KD logits", logits)
        if self.denominator == 0.0 or self.coefficient == 0.0:
            return logits.sum() * 0.0
        target = self.soft_targets.to(device=logits.device, dtype=logits.dtype)
        weights = self.pixel_weights.to(device=logits.device, dtype=logits.dtype)
        log_probs = F.log_softmax(logits / self.temperature, dim=1)
        per_pixel = -(target * log_probs).sum(dim=1) * (self.temperature**2 * self.coefficient)
        loss = (per_pixel * weights).sum() / self.denominator
        _finite("KD loss", loss)
        return loss


@dataclass(frozen=True)
class GradientArmResult:
    logits: Mapping[str, Tensor]
    deltas: Mapping[str, Tensor]
    gradients: Mapping[str, Tensor]
    gradient_norms: Mapping[str, Tensor]
    displacement_norms: Mapping[str, Tensor]
    statuses: Mapping[str, tuple[str, ...]]
    kd_losses: Mapping[str, Tensor]
    cotangents: Mapping[str, Tensor]
    denominator: float
    accepted_pixels: int
    gradient_sources: Mapping[str, str]
    per_arm_kd_losses: Mapping[str, Mapping[str, Tensor]]
    forward_counts: Mapping[str, int]
    backward_counts: Mapping[str, int]
    shuffle_modes: Mapping[str, tuple[str, ...]]

    @property
    def forward_count(self) -> int:
        """Number of native model forward calls (student plus teacher bridge)."""
        return int(self.forward_counts["student_model"] + self.forward_counts["bridge_model"])

    @property
    def backward_count(self) -> int:
        """Number of explicit reverse-mode autograd.grad launches."""
        return int(self.backward_counts["autograd_grad_calls"])


def _feature_mask(feature_mask: Tensor | None, h: Tensor) -> Tensor:
    if feature_mask is None:
        return torch.ones((h.shape[0], *h.shape[-2:]), dtype=torch.bool, device=h.device)
    if feature_mask.ndim == 4 and feature_mask.shape[1] == 1:
        feature_mask = feature_mask[:, 0]
    if feature_mask.dtype is not torch.bool or feature_mask.shape != (h.shape[0], *h.shape[-2:]):
        raise ValueError("feature_mask must be boolean [B,Hf,Wf] matching h")
    return feature_mask.detach().to(device=h.device).clone()


def _seed_for_sample(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def _shuffle_d2(direction: Tensor, support: Tensor, seed: int, sample_ids: Sequence[str]) -> tuple[Tensor, list[str]]:
    shuffled = torch.zeros_like(direction)
    states: list[str] = []
    for batch_index in range(direction.shape[0]):
        spatial_indices = torch.nonzero(support[batch_index].reshape(-1), as_tuple=False).flatten()
        scalar_count = int(spatial_indices.numel()) * int(direction.shape[1])
        if scalar_count == 0:
            states.append("no_valid_coordinates")
            continue
        sample_id = sample_ids[batch_index]
        source = direction[batch_index].reshape(direction.shape[1], -1).index_select(1, spatial_indices)
        if scalar_count == 1:
            shuffled[batch_index].reshape(direction.shape[1], -1)[:, spatial_indices] = source
            states.append("identity_noop_single_scalar")
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_seed_for_sample(seed, sample_id))
        identity = torch.arange(scalar_count)
        for _ in range(256):
            order = torch.randperm(scalar_count, generator=generator)
            if not torch.equal(order, identity):
                break
        else:
            order = identity.roll(1)
        indices = spatial_indices.to(direction.device)
        permutation = order.to(direction.device)
        shuffled[batch_index].reshape(direction.shape[1], -1)[:, indices] = source.reshape(-1).index_select(
            0, permutation
        ).reshape(direction.shape[1], -1)
        source = direction[batch_index] * support[batch_index][None].to(direction.dtype)
        states.append("shuffle_identical_values" if torch.equal(shuffled[batch_index], source) else "shuffled")
    _finite("D3 shuffled D2 gradient", shuffled)
    return shuffled, states


def _norm_on_support(value: Tensor, support: Tensor) -> Tensor:
    masked = value * support[:, None].to(dtype=value.dtype)
    return torch.linalg.vector_norm(masked.double().flatten(1), dim=1)


def _make_correction(
    name: str,
    gradient: Tensor,
    h: Tensor,
    support: Tensor,
    rho: float,
    min_gradient_norm: float,
    min_feature_norm: float,
    *,
    source_status: tuple[str, ...] | None = None,
) -> tuple[Tensor, Tensor, Tensor, tuple[str, ...]]:
    masked_gradient = gradient * support[:, None].to(dtype=gradient.dtype)
    _finite(f"{name} masked gradient", masked_gradient)
    gradient_norm = _norm_on_support(masked_gradient, support)
    feature_norm = _norm_on_support(h, support)
    delta = torch.zeros_like(h)
    actual_norm = torch.zeros((h.shape[0],), device=h.device, dtype=torch.float64)
    statuses: list[str] = []
    for index in range(h.shape[0]):
        if source_status is not None and source_status[index] != "active":
            statuses.append(f"source_{source_status[index]}")
            continue
        sites = int(support[index].sum().item())
        if sites == 0:
            statuses.append("empty_support")
            continue
        gnorm = float(gradient_norm[index].item())
        hnorm = float(feature_norm[index].item())
        if not math.isfinite(gnorm) or not math.isfinite(hnorm):
            raise InvalidGradientDiagnostic(f"non-finite per-sample norm for {name}")
        if hnorm <= min_feature_norm:
            statuses.append("near_zero_feature")
            continue
        if gnorm <= min_gradient_norm:
            statuses.append("near_zero_gradient")
            continue
        radius = float(rho) * hnorm
        if radius == 0.0:
            statuses.append("zero_rho")
            continue
        delta[index] = (masked_gradient[index].double() * (-radius / gnorm)).to(dtype=h.dtype)
        actual_norm[index] = torch.linalg.vector_norm(delta[index].double().flatten())
        tolerance = max(1e-12, 10.0 * torch.finfo(h.dtype).eps * radius)
        if abs(float(actual_norm[index].item()) - radius) > tolerance:
            raise InvalidGradientDiagnostic(f"{name} correction cannot represent the locked per-sample norm")
        statuses.append("active")
    _finite(f"{name} feature correction", delta)
    _finite(f"{name} displacement norm", actual_norm)
    return delta.detach(), gradient_norm.detach(), actual_norm.detach(), tuple(statuses)


def compute_gradient_arms(
    h: Tensor,
    student_logits_fn: Callable[[Tensor], Tensor],
    bridge_logits_fn: Callable[[Tensor], Tensor],
    kd: KDLossContext,
    *,
    feature_mask: Tensor | None = None,
    rho: float,
    seed: int,
    sample_ids: Sequence[str] | None = None,
    min_gradient_norm: float = 1e-12,
    min_feature_norm: float = 1e-12,
    frozen_modules: Sequence[nn.Module] = (),
) -> GradientArmResult:
    """Compute exact D0-D5 VJPs and native-student predictions, one batch at a time.

    D1=J_S^T u_S, D2=J_B^T u_B, D3=valid-coordinate shuffled D2, D4=J_B^T u_S,
    and D5=J_S^T u_B. Every ``u`` is obtained by autograd of the same frozen
    masked/weighted/temperature-scaled KD objective and common denominator.
    All corrections alter only h and are decoded by ``student_logits_fn``.
    """
    if not isinstance(h, Tensor) or h.ndim != 4 or not torch.is_floating_point(h):
        raise ValueError("h must be a floating BCHW tensor")
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(float(rho)) or rho < 0:
        raise ValueError("rho must be finite and non-negative")
    if (isinstance(min_gradient_norm, bool) or not isinstance(min_gradient_norm, (int, float))
            or not math.isfinite(float(min_gradient_norm)) or min_gradient_norm < 0):
        raise ValueError("min_gradient_norm must be finite and non-negative")
    if (isinstance(min_feature_norm, bool) or not isinstance(min_feature_norm, (int, float))
            or not math.isfinite(float(min_feature_norm)) or min_feature_norm < 0):
        raise ValueError("min_feature_norm must be finite and non-negative")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if sample_ids is None or len(sample_ids) != h.shape[0] or any(
            not isinstance(value, str) or not value for value in sample_ids):
        raise ValueError("sample_ids must contain one nonempty stable ID per sample")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be unique within the diagnostic batch")
    if not frozen_modules:
        raise ValueError("frozen_modules must list the native student and teacher bridge")
    for module in frozen_modules:
        if any(child.training for child in module.modules()):
            raise ValueError("all diagnostic modules must be eval-mode; the runner never mutates model state")
    support = _feature_mask(feature_mask, h)
    _finite("bottleneck feature", h)
    devices = [h.device.index if h.device.index is not None else torch.cuda.current_device()] if h.device.type == "cuda" else []

    forward_counts = {"student_model": 0, "bridge_model": 0, "kd_objective": 0}
    backward_counts = {
        "autograd_grad_calls": 0,
        "cotangent_vjps": 0,
        "direct_kd_gradient_checks": 0,
        "native_vjps": 0,
        "cross_vjps": 0,
    }

    def model_forward(kind: str, function: Callable[[Tensor], Tensor], value: Tensor) -> Tensor:
        forward_counts[kind] += 1
        return function(value)

    def kd_forward(logits: Tensor) -> Tensor:
        forward_counts["kd_objective"] += 1
        return kd(logits)

    def gradient(category: str, *args: Any, **kwargs: Any) -> Tensor:
        backward_counts["autograd_grad_calls"] += 1
        backward_counts[category] += 1
        return torch.autograd.grad(*args, **kwargs)

    # fork_rng preserves caller RNG even if an accidental stochastic op appears in a model.
    with torch.random.fork_rng(devices=devices), torch.enable_grad():
        h_leaf = h.detach().clone().requires_grad_(True)
        z_student = model_forward("student_model", student_logits_fn, h_leaf)
        z_bridge = model_forward("bridge_model", bridge_logits_fn, h_leaf)
        if not isinstance(z_student, Tensor) or not isinstance(z_bridge, Tensor):
            raise ValueError("student and bridge functions must return logits tensors")
        if z_student.shape != kd.target_probs.shape or z_bridge.shape != kd.target_probs.shape:
            raise ValueError("native student and bridge logits must share target class order and native geometry")
        if z_bridge.shape != z_student.shape:
            raise ValueError("student and bridge logits must use the same output lattice")
        _finite("student logits", z_student)
        _finite("bridge logits", z_bridge)
        loss_student = kd_forward(z_student)
        loss_bridge = kd_forward(z_bridge)
        cotangent_student = gradient("cotangent_vjps",
            loss_student, z_student, retain_graph=True, create_graph=False, allow_unused=False
        )[0].detach()
        cotangent_bridge = gradient("cotangent_vjps",
            loss_bridge, z_bridge, retain_graph=True, create_graph=False, allow_unused=False
        )[0].detach()
        direct_student = gradient("direct_kd_gradient_checks",
            loss_student, h_leaf, retain_graph=True, create_graph=False, allow_unused=False
        )[0].detach()
        direct_bridge = gradient("direct_kd_gradient_checks",
            loss_bridge, h_leaf, retain_graph=True, create_graph=False, allow_unused=False
        )[0].detach()
        # D1/D2 are explicitly the output-Jacobian VJPs J^T u, where u is
        # obtained from the exact common current-KD loss above.
        g_student = gradient("native_vjps",
            z_student, h_leaf, grad_outputs=cotangent_student, retain_graph=True,
            create_graph=False, allow_unused=False,
        )[0].detach()
        g_bridge = gradient("native_vjps",
            z_bridge, h_leaf, grad_outputs=cotangent_bridge, retain_graph=True,
            create_graph=False, allow_unused=False,
        )[0].detach()
        tolerance = max(1e-8, 10.0 * torch.finfo(h_leaf.dtype).eps)
        for name, vjp, direct in (("D1", g_student, direct_student), ("D2", g_bridge, direct_bridge)):
            if not torch.allclose(vjp, direct, atol=tolerance, rtol=1e-4):
                difference = float((vjp - direct).abs().max().item())
                raise InvalidGradientDiagnostic(
                    f"{name} output VJP disagrees with direct gradient of the shared KD objective; max_abs={difference}"
                )
        g_d4 = gradient("cross_vjps",
            z_bridge, h_leaf, grad_outputs=cotangent_student, retain_graph=True,
            create_graph=False, allow_unused=False,
        )[0].detach()
        g_d5 = gradient("cross_vjps",
            z_student, h_leaf, grad_outputs=cotangent_bridge, retain_graph=False,
            create_graph=False, allow_unused=False,
        )[0].detach()
        for name, value in (("uS", cotangent_student), ("uB", cotangent_bridge),
                            ("D1 VJP", g_student), ("D2 VJP", g_bridge),
                            ("D4 VJP", g_d4), ("D5 VJP", g_d5)):
            _finite(name, value)

        raw_gradients = {
            "D1": g_student,
            "D2": g_bridge,
            "D4": g_d4,
            "D5": g_d5,
        }
        d3_gradient, d3_shuffle_status = _shuffle_d2(g_bridge, support, seed, sample_ids)
        raw_gradients["D3"] = d3_gradient
        h_base = h_leaf.detach()
        deltas: dict[str, Tensor] = {"D0": torch.zeros_like(h_base)}
        gradient_norms: dict[str, Tensor] = {"D0": torch.zeros(h.shape[0], dtype=torch.float64, device=h.device)}
        displacement_norms: dict[str, Tensor] = {"D0": torch.zeros(h.shape[0], dtype=torch.float64, device=h.device)}
        statuses: dict[str, tuple[str, ...]] = {"D0": tuple("baseline" for _ in range(h.shape[0]))}
        for name in ("D1", "D2", "D4", "D5"):
            delta, gradient_norm, displacement_norm, arm_status = _make_correction(
                name, raw_gradients[name], h_base, support, float(rho),
                float(min_gradient_norm), float(min_feature_norm),
            )
            deltas[name] = delta
            gradient_norms[name] = gradient_norm
            displacement_norms[name] = displacement_norm
            statuses[name] = arm_status
        d3_sources = statuses["D2"]
        delta3, norm3, displacement3, status3 = _make_correction(
            "D3", raw_gradients["D3"], h_base, support, float(rho),
            float(min_gradient_norm), float(min_feature_norm), source_status=d3_sources,
        )
        status3 = tuple(state.removeprefix("source_") for state in status3)
        deltas["D3"] = delta3
        gradient_norms["D3"] = norm3
        displacement_norms["D3"] = displacement3
        statuses["D3"] = status3

        logits: dict[str, Tensor] = {"D0": z_student.detach()}
        per_arm_kd_losses: dict[str, Mapping[str, Tensor]] = {
            "D0": {"student": loss_student.detach(), "bridge": loss_bridge.detach()}
        }
        with torch.no_grad():
            for name in ("D1", "D2", "D3", "D4", "D5"):
                # Both heads are evaluated at each common h+delta; the saved
                # prediction remains the native student output for compatibility.
                corrected_h = h_base + deltas[name]
                student_result = model_forward("student_model", student_logits_fn, corrected_h)
                bridge_result = model_forward("bridge_model", bridge_logits_fn, corrected_h)
                if student_result.shape != kd.target_probs.shape or bridge_result.shape != kd.target_probs.shape:
                    raise ValueError(f"{name} corrected heads changed the locked target geometry")
                _finite(f"{name} corrected native-student logits", student_result)
                _finite(f"{name} corrected bridge logits", bridge_result)
                logits[name] = student_result.detach()
                per_arm_kd_losses[name] = {
                    "student": kd_forward(student_result).detach(),
                    "bridge": kd_forward(bridge_result).detach(),
                }

    accepted_pixels = int(kd.confidence_mask.sum().item())
    return GradientArmResult(
        logits=logits,
        deltas=deltas,
        gradients={name: value.detach() for name, value in raw_gradients.items()},
        gradient_norms=gradient_norms,
        displacement_norms=displacement_norms,
        statuses=statuses,
        kd_losses={"student": loss_student.detach(), "bridge": loss_bridge.detach()},
        cotangents={"student": cotangent_student.detach(), "bridge": cotangent_bridge.detach()},
        denominator=kd.denominator,
        accepted_pixels=accepted_pixels,
        gradient_sources={"D1": "J_S^T u_S", "D2": "J_B^T u_B", "D3": "valid_coordinate_shuffle(D2)",
                          "D4": "J_B^T u_S", "D5": "J_S^T u_B"},
        per_arm_kd_losses=per_arm_kd_losses,
        forward_counts=forward_counts,
        backward_counts=backward_counts,
        shuffle_modes={"D3": tuple(d3_shuffle_status)},
    )


__all__ = [
    "GradientArmResult",
    "InvalidGradientDiagnostic",
    "KDLossContext",
    "compute_gradient_arms",
]
