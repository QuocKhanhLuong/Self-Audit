"""Functional one-step meta-Dice updates for the adaptive-window pilot.

The helpers in this module deliberately do not mutate a model or an optimizer.
They implement the exact optimizer family used by ``ema_full`` (plain AdamW
for the body and momentum-free SGD for the head) on a detached parameter
snapshot, while retaining the gradient from the pseudo update to the window
geometry.  The caller owns the real student update and the subsequent EMA.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections import OrderedDict
import inspect
import math
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.func import functional_call


_GEOMETRY_PREFIX = "window.geometry."


class _SafeSqrt(torch.autograd.Function):
    """Square root with a zero derivative at exactly zero.

    AdamW's first step can have a zero second moment for a parameter.  The
    forward value must remain ``sqrt(v)`` (so the functional step matches
    PyTorch), but the ordinary derivative of sqrt at zero is undefined and
    poisons a mixed hypergradient.  A zero subgradient at that point is the
    stable extension used here.  The backward expression is itself
    differentiable for positive inputs, so double backward remains available.
    """

    @staticmethod
    def forward(ctx, value: Tensor) -> Tensor:
        root = torch.sqrt(value)
        ctx.save_for_backward(value, root)
        return root

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:
        value, root = ctx.saved_tensors
        positive = value > 0
        # Build a finite denominator before masking.  Evaluating
        # ``grad_output / (2 * root)`` at root==0 can still materialize inf or
        # NaN in a higher-order graph even though torch.where selects zero.
        denominator = torch.where(positive, 2 * root, torch.ones_like(root))
        derivative = torch.where(positive, grad_output / denominator, torch.zeros_like(grad_output))
        return (derivative,)


def _safe_sqrt(value: Tensor) -> Tensor:
    return _SafeSqrt.apply(value)


def geometry_named_parameters(model: nn.Module) -> OrderedDict[str, nn.Parameter]:
    """Return only adaptive-window geometry parameters in module order.

    Geometry is intentionally identified by the stable pilot contract
    ``window.geometry.*``.  Query/key/value projections and the residual gate
    therefore remain ordinary student parameters.
    """

    return OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(_GEOMETRY_PREFIX)
    )


@contextmanager
def frozen_geometry(model: nn.Module):
    """Temporarily freeze geometry while preserving gradients to input features.

    This context only toggles ``requires_grad`` on geometry parameters; it does
    not enter ``no_grad`` and consequently does not cut gradients through an
    input feature tensor.  Existing and newly produced geometry ``.grad``
    values are cleared so ordinary student steps cannot accidentally clip or
    retain a meta-gradient.
    """

    parameters = tuple(geometry_named_parameters(model).values())
    flags = tuple(parameter.requires_grad for parameter in parameters)
    for parameter in parameters:
        parameter.grad = None
        parameter.requires_grad_(False)
    try:
        yield model
    finally:
        for parameter, flag in zip(parameters, flags, strict=True):
            parameter.grad = None
            parameter.requires_grad_(flag)


def _validate_optimizer(optimizer: torch.optim.Optimizer) -> None:
    if isinstance(optimizer, torch.optim.AdamW):
        defaults = optimizer.defaults
        unsupported = {
            "amsgrad": defaults.get("amsgrad", False),
            "maximize": defaults.get("maximize", False),
            "capturable": defaults.get("capturable", False),
            "differentiable": defaults.get("differentiable", False),
            "fused": defaults.get("fused", None),
        }
        if any(bool(value) for value in unsupported.values()):
            raise ValueError("functional meta step supports only plain non-capturable AdamW")
        foreach = defaults.get("foreach", None)
        if foreach:
            raise ValueError("functional meta step rejects foreach AdamW")
        betas = defaults.get("betas")
        if betas is None or len(betas) != 2 or not all(math.isfinite(float(v)) for v in betas):
            raise ValueError("invalid AdamW betas")
        if not 0 <= float(betas[0]) < 1 or not 0 <= float(betas[1]) < 1:
            raise ValueError("AdamW betas must lie in [0, 1)")
        if float(defaults.get("eps", 0.0)) < 0:
            raise ValueError("AdamW eps must be non-negative")
        if not bool(defaults.get("decoupled_weight_decay", True)):
            raise ValueError("functional meta step requires decoupled AdamW weight decay")
        groups = optimizer.param_groups
        for group in groups:
            if bool(group.get("amsgrad", defaults.get("amsgrad", False))) or bool(group.get("maximize", defaults.get("maximize", False))):
                raise ValueError("functional meta step supports only plain non-capturable AdamW")
            if bool(group.get("capturable", defaults.get("capturable", False))) or bool(group.get("differentiable", defaults.get("differentiable", False))):
                raise ValueError("functional meta step supports only plain non-capturable AdamW")
            if bool(group.get("fused", defaults.get("fused", None))) or bool(group.get("foreach", defaults.get("foreach", None))):
                raise ValueError("functional meta step rejects foreach/fused AdamW")
            if not bool(group.get("decoupled_weight_decay", defaults.get("decoupled_weight_decay", True))):
                raise ValueError("functional meta step requires decoupled AdamW weight decay")
            group_betas = group.get("betas", defaults["betas"])
            if len(group_betas) != 2 or not all(math.isfinite(float(v)) for v in group_betas):
                raise ValueError("invalid AdamW group betas")
            if not 0 <= float(group_betas[0]) < 1 or not 0 <= float(group_betas[1]) < 1:
                raise ValueError("AdamW group betas must lie in [0, 1)")
            if float(group.get("eps", defaults.get("eps", 0.0))) < 0:
                raise ValueError("AdamW group eps must be non-negative")
            if not math.isfinite(float(group.get("lr", defaults["lr"]))) or float(group.get("lr", defaults["lr"])) < 0:
                raise ValueError("AdamW group learning rate must be finite and non-negative")
            if not math.isfinite(float(group.get("weight_decay", defaults.get("weight_decay", 0.0)))):
                raise ValueError("AdamW group weight decay must be finite")
        return
    if isinstance(optimizer, torch.optim.SGD):
        defaults = optimizer.defaults
        unsupported = {
            "momentum": defaults.get("momentum", 0.0),
            "dampening": defaults.get("dampening", 0.0),
            "weight_decay": defaults.get("weight_decay", 0.0),
            "nesterov": defaults.get("nesterov", False),
            "maximize": defaults.get("maximize", False),
            "fused": defaults.get("fused", None),
            "differentiable": defaults.get("differentiable", False),
        }
        if any(bool(value) for value in unsupported.values()):
            raise ValueError("functional meta step supports only momentum-free plain SGD")
        if defaults.get("foreach", None):
            raise ValueError("functional meta step rejects foreach SGD")
        for group in optimizer.param_groups:
            if (float(group.get("momentum", defaults.get("momentum", 0.0))) != 0 or
                    float(group.get("dampening", defaults.get("dampening", 0.0))) != 0 or
                    float(group.get("weight_decay", defaults.get("weight_decay", 0.0))) != 0 or
                    bool(group.get("nesterov", defaults.get("nesterov", False))) or
                    bool(group.get("maximize", defaults.get("maximize", False))) or
                    bool(group.get("fused", defaults.get("fused", None))) or
                    bool(group.get("differentiable", defaults.get("differentiable", False))) or
                    bool(group.get("foreach", defaults.get("foreach", None)))):
                raise ValueError("functional meta step supports only momentum-free plain SGD")
            if not math.isfinite(float(group.get("lr", defaults["lr"]))) or float(group.get("lr", defaults["lr"])) < 0:
                raise ValueError("SGD group learning rate must be finite and non-negative")
        return
    raise TypeError(f"unsupported optimizer for functional meta step: {type(optimizer).__name__}")


def _model_parameter_index(model: nn.Module, optimizers: Iterable[torch.optim.Optimizer]):
    named = OrderedDict(model.named_parameters())
    by_id = {id(parameter): name for name, parameter in named.items()}
    geometry = geometry_named_parameters(model)
    owners: dict[str, tuple[torch.optim.Optimizer, Mapping[str, object]]] = {}
    for optimizer in optimizers:
        _validate_optimizer(optimizer)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                name = by_id.get(id(parameter))
                if name is None:
                    raise ValueError("optimizer contains a parameter outside the model")
                if name in geometry:
                    raise ValueError("window geometry must be excluded from ordinary optimizers")
                if name in owners:
                    raise ValueError(f"parameter appears in multiple optimizer groups: {name}")
                owners[name] = (optimizer, group)
    missing = [
        name for name, parameter in named.items()
        if name not in geometry and parameter.requires_grad and name not in owners
    ]
    if missing:
        raise ValueError(
            "ordinary optimizers omit trainable theta parameters: " + ", ".join(missing)
        )
    return named, geometry, owners


def _clone_buffers(model: nn.Module) -> dict[str, Tensor]:
    return {name: buffer.detach().clone() for name, buffer in model.named_buffers()}


def _snapshot_parameters(model: nn.Module, names: Iterable[str], *, requires_grad: bool | None) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name in names:
        parameter = dict(model.named_parameters())[name]
        value = parameter.detach().clone()
        result[name] = value.requires_grad_(parameter.requires_grad if requires_grad is None else requires_grad)
    return result


def _supports_valid_mask(model: nn.Module) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return "valid_mask" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )


def _functional_forward(
    model: nn.Module,
    state: Mapping[str, Tensor],
    inputs: Tensor,
    valid_mask: Tensor | None,
) -> Tensor:
    kwargs = {"valid_mask": valid_mask} if valid_mask is not None and _supports_valid_mask(model) else None
    return functional_call(model, state, (inputs,), kwargs=kwargs, tie_weights=True, strict=False)


def _masked_pseudo_ce(logits: Tensor, target: Tensor, valid: Tensor) -> tuple[Tensor, int]:
    if target.dtype != torch.long:
        target = target.long()
    if logits.ndim != 4 or target.shape != logits.shape[:1] + logits.shape[2:] or valid.shape != target.shape:
        raise ValueError("pseudo logits, target, and valid mask have incompatible shapes")
    if valid.dtype != torch.bool:
        raise ValueError("pseudo valid mask must be boolean")
    count = int(valid.sum().detach())
    if count == 0:
        return logits.sum() * 0, 0
    pixel_loss = F.cross_entropy(logits, target, reduction="none")
    return (pixel_loss * valid).sum() / valid.sum(), count


def _present_foreground_dice(
    logits: Tensor,
    labels: Tensor,
    valid_mask: Tensor | None,
) -> tuple[Tensor | None, list[int], int]:
    if logits.ndim != 4 or labels.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("audit logits and labels have incompatible shapes")
    if labels.dtype != torch.long:
        labels = labels.long()
    if ((labels < 0) | ((labels > 3) & (labels != 255))).any():
        raise ValueError("audit labels must be in 0..3 or 255")
    support = labels != 255
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool or valid_mask.shape != support.shape:
            raise ValueError("audit valid mask has incompatible shape")
        support = support & valid_mask
    y = F.one_hot(labels.clamp_max(3), 4).permute(0, 3, 1, 2).to(logits.dtype) * support[:, None]
    p = logits.softmax(1) * support[:, None]
    dims = (0, 2, 3)
    denominator = p.sum(dims) + y.sum(dims)
    dice = (2 * (p * y).sum(dims) + 1e-5) / (denominator + 1e-5)
    present_tensor = y[:, 1:].sum((0, 2, 3)) > 0
    present = [index + 1 for index, flag in enumerate(present_tensor.detach().tolist()) if flag]
    foreground = int(y[:, 1:].sum().detach())
    if not present:
        return None, [], foreground
    return (1 - dice[1:][present_tensor]).mean(), present, foreground


def _clip_gradients(grads: Mapping[str, Tensor | None], clip_norm: float) -> tuple[dict[str, Tensor | None], Tensor, Tensor]:
    values = [gradient for gradient in grads.values() if gradient is not None]
    if not values:
        zero = torch.tensor(0.0)
        return dict(grads), zero, torch.tensor(1.0)
    norms = torch.stack([torch.linalg.vector_norm(gradient) for gradient in values])
    total_norm = torch.linalg.vector_norm(norms)
    if not torch.isfinite(total_norm):
        raise FloatingPointError("nonfinite virtual gradient norm")
    coefficient = torch.clamp(torch.as_tensor(float(clip_norm), device=total_norm.device, dtype=total_norm.dtype) /
                              (total_norm + 1e-6), max=1.0)
    return {
        name: None if gradient is None else gradient * coefficient
        for name, gradient in grads.items()
    }, total_norm, coefficient


def _state_tensor(state: Mapping[str, object], key: str, parameter: Tensor) -> Tensor:
    value = state.get(key)
    if value is None:
        return torch.zeros_like(parameter)
    if not isinstance(value, Tensor) or value.shape != parameter.shape:
        raise ValueError(f"invalid optimizer state {key}")
    return value.detach().clone().to(device=parameter.device, dtype=parameter.dtype)


def _step_value(state: Mapping[str, object], parameter: Tensor) -> Tensor:
    value = state.get("step")
    if value is None:
        return torch.zeros((), device=parameter.device, dtype=torch.float32)
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("AdamW step state must be scalar")
        return value.detach().clone().to(device=parameter.device)
    return torch.as_tensor(float(value), device=parameter.device, dtype=torch.float32)


def _functional_step_with_states(
    parameters: Mapping[str, Tensor],
    gradients: Mapping[str, Tensor | None],
    owners: Mapping[str, tuple[torch.optim.Optimizer, Mapping[str, object]]],
    model_parameters: Mapping[str, nn.Parameter],
    *,
    clip_norm: float,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    clipped, total_norm, coefficient = _clip_gradients(gradients, clip_norm)
    updated = dict(parameters)
    updated_count = 0
    for name, (optimizer, group) in owners.items():
        gradient = clipped.get(name)
        if gradient is None:
            continue
        parameter = parameters[name]
        real_parameter = model_parameters[name]
        if isinstance(optimizer, torch.optim.AdamW):
            defaults = optimizer.defaults
            beta1, beta2 = (float(v) for v in group.get("betas", defaults["betas"]))
            lr = float(group.get("lr", defaults["lr"]))
            eps = float(group.get("eps", defaults["eps"]))
            weight_decay = float(group.get("weight_decay", defaults["weight_decay"]))
            state = optimizer.state.get(real_parameter, {})
            exp_avg = _state_tensor(state, "exp_avg", parameter)
            exp_avg_sq = _state_tensor(state, "exp_avg_sq", parameter)
            step = _step_value(state, parameter) + 1
            exp_avg = exp_avg * beta1 + gradient * (1 - beta1)
            exp_avg_sq = exp_avg_sq * beta2 + gradient.square() * (1 - beta2)
            # The source optimizer is the ordinary non-capturable AdamW path.
            # PyTorch extracts ``state['step']`` to a Python scalar there, so
            # ``beta**step`` is evaluated in Python double precision.  Doing
            # the correction in a float32 tensor changes float64 parity and
            # creates an avoidable hypergradient discrepancy.
            step_value = float(step.detach().cpu().item())
            bias1 = 1.0 - beta1 ** step_value
            bias2 = 1.0 - beta2 ** step_value
            denominator = _safe_sqrt(exp_avg_sq) / math.sqrt(bias2) + eps
            step_size = lr / bias1
            updated[name] = parameter * (1 - lr * weight_decay) - step_size * exp_avg / denominator
        elif isinstance(optimizer, torch.optim.SGD):
            lr = float(group.get("lr", optimizer.defaults["lr"]))
            updated[name] = parameter - lr * gradient
        else:  # validated before this helper is entered
            raise TypeError(type(optimizer).__name__)
        updated_count += 1
    return updated, {"total_norm": total_norm, "clip_scale": coefficient, "updated_count": updated_count}


def meta_geometry_gradients(
    model: nn.Module,
    optimizers: Iterable[torch.optim.Optimizer],
    strong: Tensor,
    py: Tensor,
    valid: Tensor,
    xa: Tensor,
    ya: Tensor,
    *,
    input_valid: Tensor | None = None,
    audit_valid: Tensor | None = None,
    clip_norm: float = 5.0,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Compute an update-mediated geometry gradient without mutating state.

    ``valid`` is the detached confidence mask from the EMA teacher.  If
    ``input_valid`` is supplied, it is intersected with that mask.  ``xa, ya``
    are a distinct labeled audit batch.  The returned geometry gradients are
    detached tensors suitable for a separate pilot SGD step; this function
    never writes ``parameter.grad`` or advances either real optimizer.
    """

    optimizers = tuple(optimizers)
    if not optimizers:
        raise ValueError("at least one ordinary optimizer is required")
    if not math.isfinite(float(clip_norm)) or float(clip_norm) < 0:
        raise ValueError("clip_norm must be finite and non-negative")
    named, geometry, owners = _model_parameter_index(model, optimizers)
    if not geometry:
        raise ValueError("model has no window.geometry.* parameters")
    if any(not parameter.requires_grad for parameter in geometry.values()):
        raise ValueError("meta geometry parameters must require gradients")
    pseudo_valid = valid.detach()
    if pseudo_valid.dtype != torch.bool or pseudo_valid.shape != py.shape:
        raise ValueError("pseudo valid mask must match py and be boolean")
    if input_valid is not None:
        if input_valid.dtype != torch.bool or input_valid.shape != py.shape:
            raise ValueError("input_valid must match py and be boolean")
        pseudo_valid = pseudo_valid & input_valid.detach()
    accepted = int(pseudo_valid.sum().detach())
    if accepted == 0:
        return {}, {"status": "skip_no_pseudo_support", "accepted_pixels": 0, "outer_foreground": []}

    if ya.ndim != 3 or xa.ndim != 4 or xa.shape[0] != ya.shape[0]:
        raise ValueError("audit input/labels have incompatible shapes")
    audit_support = audit_valid.detach() if audit_valid is not None else None
    if audit_support is not None and (audit_support.dtype != torch.bool or audit_support.shape != ya.shape):
        raise ValueError("audit_valid must match ya and be boolean")
    outer_labels = ya.detach().long()
    outer_present = outer_labels != 255
    if audit_support is not None:
        outer_present = outer_present & audit_support
    outer_classes = [int(index) for index in range(1, 4) if bool(((outer_labels == index) & outer_present).any())]
    if not outer_classes:
        return {}, {"status": "skip_no_outer_foreground", "accepted_pixels": accepted, "outer_foreground": []}

    # Snapshot every parameter and buffer.  Geometry has an independent leaf
    # variable so no gradient is written to the live model parameter.
    theta_names = [name for name in named if name not in geometry]
    theta = _snapshot_parameters(model, theta_names, requires_grad=None)
    psi = {name: parameter.detach().clone().requires_grad_(True) for name, parameter in geometry.items()}
    state = _clone_buffers(model)
    inner_state = dict(state)
    inner_state.update(theta)
    inner_state.update(psi)

    strong_fixed, py_fixed = strong.detach(), py.detach().long()
    with _preserve_eval_mode(model):
        inner_logits = _functional_forward(model, inner_state, strong_fixed, input_valid.detach() if input_valid is not None else None)
        inner_loss, _ = _masked_pseudo_ce(inner_logits, py_fixed, pseudo_valid)
        if not torch.isfinite(inner_loss):
            raise FloatingPointError("nonfinite virtual pseudo loss")
        differentiable_theta_names = [name for name, value in theta.items() if value.requires_grad]
        differentiable_theta = [theta[name] for name in differentiable_theta_names]
        inner_grads_tuple = (torch.autograd.grad(inner_loss, differentiable_theta, create_graph=True, allow_unused=True)
                             if differentiable_theta else tuple())
        inner_grads = {name: gradient for name, gradient in zip(differentiable_theta_names, inner_grads_tuple, strict=True)}
        all_grads: dict[str, Tensor | None] = {name: inner_grads.get(name) for name in theta}
        virtual_theta, opt_diag = _functional_step_with_states(theta, all_grads, owners, named, clip_norm=clip_norm)

        outer_state = dict(state)
        outer_state.update(virtual_theta)
        # This detached copy blocks the direct outer Dice -> geometry path.
        outer_state.update({name: value.detach() for name, value in psi.items()})
        # Record the same audit Dice loss at the supervised snapshot before
        # constructing the virtual outer graph.  This is a no-grad reference;
        # it is a predicted local signal, not the committed post-real-update
        # Dice that the engine may log later.
        reference_state = dict(state)
        reference_state.update({name: value.detach() for name, value in theta.items()})
        reference_state.update({name: value.detach() for name, value in psi.items()})
        with torch.no_grad():
            reference_logits = _functional_forward(model, reference_state, xa.detach(), audit_support)
            reference_loss, _, _ = _present_foreground_dice(reference_logits, outer_labels, audit_support)
        if reference_loss is None:
            return {}, {"status": "skip_no_outer_foreground", "accepted_pixels": accepted, "outer_foreground": []}
        if not torch.isfinite(reference_loss):
            raise FloatingPointError("nonfinite reference outer Dice loss")
        outer_logits = _functional_forward(model, outer_state, xa.detach(), audit_support)
        outer_loss, present, outer_foreground = _present_foreground_dice(outer_logits, outer_labels, audit_support)
        if outer_loss is None:
            return {}, {"status": "skip_no_outer_foreground", "accepted_pixels": accepted, "outer_foreground": []}
        if not torch.isfinite(outer_loss):
            raise FloatingPointError("nonfinite outer Dice loss")
        gradients_tuple = (torch.autograd.grad(outer_loss, tuple(psi.values()), allow_unused=True)
                           if outer_loss.requires_grad else tuple(None for _ in psi))

    gradients: dict[str, Tensor] = {}
    for name, gradient in zip(psi, gradients_tuple, strict=True):
        value = torch.zeros_like(psi[name]) if gradient is None else gradient.detach().clone()
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"nonfinite geometry meta-gradient: {name}")
        gradients[name] = value
    geometry_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(value) for value in gradients.values()]))
    delta_values = [virtual_theta[name].detach() - theta[name].detach() for name in owners if name in virtual_theta]
    delta_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(value) for value in delta_values])) if delta_values else torch.tensor(0.0)
    diagnostics = {
        "status": "ok",
        "accepted_pixels": accepted,
        "outer_foreground": present,
        "outer_foreground_pixels": outer_foreground,
        "inner_pseudo_loss": float(inner_loss.detach()),
        "outer_dice_loss": float(outer_loss.detach()),
        "reference_outer_dice_loss": float(reference_loss.detach()),
        "predicted_dice_loss_reduction": float((reference_loss.detach() - outer_loss.detach())),
        "inner_grad_norm": float(opt_diag["total_norm"].detach()),
        "clip_scale": float(opt_diag["clip_scale"].detach()),
        "virtual_parameter_delta_norm": float(delta_norm.detach()),
        "geometry_grad_norm": float(geometry_norm.detach()),
        "ordinary_parameter_updates": int(opt_diag["updated_count"]),
        "ordinary_parameter_count": int(len(owners)),
        "geometry_parameter_count": int(len(geometry)),
        "geometry_nonzero_gradient_count": int(sum(bool(torch.count_nonzero(value)) for value in gradients.values())),
    }
    return gradients, diagnostics


@contextmanager
def _preserve_eval_mode(model: nn.Module):
    modes = {module: module.training for module in model.modules()}
    model.eval()
    try:
        yield
    finally:
        for module, training in modes.items():
            module.train(training)


__all__ = ["geometry_named_parameters", "frozen_geometry", "meta_geometry_gradients"]
