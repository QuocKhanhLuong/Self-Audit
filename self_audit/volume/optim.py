"""Explicit optimizer recipes and outer-iteration scheduler for volume arms.

The module keeps optimizer construction separate from the training timeline.
Callers may perform one supervised step and one pseudo step in an outer
iteration, but should invoke the scheduler once after the final transaction.
``safe_optimizer_step`` refuses to call ``step`` when no supported loss
produced gradients, preventing AdamW weight decay from changing a model on an
empty pseudo-support transaction.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn


DEFAULT_OPTIMIZATION = {
    "recipe": "grouped_adamw",
    "body_lr": 3e-5,
    "head_lr": 3e-3,
    "encoder_lr": None,
    "decoder_lr": None,
    "weight_decay": 1e-4,
    "clip_norm": 5.0,
    "scheduler": "cosine",
    "lr_warmup_steps": 0,
    "min_lr_ratio": 0.0,
    "total_steps": 0,
    "pseudo_delay": 0,
}


def _config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_OPTIMIZATION)
    if config is not None:
        if not isinstance(config, Mapping):
            raise ValueError("optimization_config must be a mapping")
        unknown = set(config) - set(DEFAULT_OPTIMIZATION)
        if unknown:
            raise ValueError(f"unknown optimization fields: {sorted(unknown)}")
        merged.update(dict(config))
    unknown = set(merged) - set(DEFAULT_OPTIMIZATION)
    if unknown:
        raise ValueError(f"unknown optimization fields: {sorted(unknown)}")
    recipe = merged["recipe"]
    if recipe not in {"grouped_adamw", "legacy_adamw_sgd", "adamw", "legacy"}:
        raise ValueError("recipe must be grouped_adamw or legacy_adamw_sgd")
    for name in ("body_lr", "head_lr", "weight_decay", "min_lr_ratio"):
        value = merged[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid optimization field {name}")
    for name in ("encoder_lr", "decoder_lr"):
        value = merged[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError(f"invalid optimization field {name}")
    for name in ("lr_warmup_steps", "total_steps", "pseudo_delay"):
        value = merged[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if merged["clip_norm"] is not None and (isinstance(merged["clip_norm"], bool) or not isinstance(merged["clip_norm"], (int, float)) or merged["clip_norm"] <= 0):
        raise ValueError("clip_norm must be positive or None")
    if not 0 <= merged["min_lr_ratio"] <= 1:
        raise ValueError("min_lr_ratio must be in [0,1]")
    return merged


def _named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def _extract_groups(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Extract encoder/decoder/head groups while rejecting duplicates."""
    named = _named_parameters(model)
    by_id = {id(parameter): name for name, parameter in named}
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {"encoder": [], "decoder": [], "head": []}

    declared = getattr(model, "parameter_groups", None)
    if callable(declared):
        declared = declared()
    if declared is not None:
        if not isinstance(declared, Mapping):
            raise ValueError("model.parameter_groups() must return a mapping")
        owners: dict[int, str] = {}
        for group_name, values in declared.items():
            canonical = str(group_name).lower()
            if canonical in {"backbone", "encoder_body"}:
                canonical = "encoder"
            if canonical in {"body", "detail", "fusion", "decoder_body"}:
                canonical = "decoder"
            if canonical not in groups:
                raise ValueError(f"unsupported model parameter group {group_name}")
            for parameter in values:
                if not isinstance(parameter, nn.Parameter) or id(parameter) not in by_id:
                    raise ValueError("model parameter_groups contains unknown parameter")
                if id(parameter) in owners:
                    raise ValueError(f"parameter appears in multiple groups: {by_id[id(parameter)]}")
                owners[id(parameter)] = canonical
                groups[canonical].append((by_id[id(parameter)], parameter))
        # Any omitted trainable parameters are assigned by stable name below.
        named = [(name, parameter) for name, parameter in named if id(parameter) not in owners]

    for name, parameter in named:
        low = name.lower()
        if low.startswith(("head.", "classifier.", "out.", "logits.")) or ".head." in low:
            group = "head"
        elif low.startswith(("encoder.", "backbone.", "input_adapter.")) or ".encoder." in low:
            group = "encoder"
        else:
            group = "decoder"
        groups[group].append((name, parameter))
    if not any(groups.values()):
        raise ValueError("model has no trainable parameters")
    return groups


def _group_metadata(
    groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]],
    lrs: Mapping[str, float],
    weight_decays: Mapping[str, float],
) -> list[dict[str, Any]]:
    records = []
    for name in ("encoder", "decoder", "head"):
        values = list(groups[name])
        if not values:
            continue
        records.append({
            "name": name,
            "parameter_names": [item[0] for item in values],
            "parameter_count": len(values),
            "numel": int(sum(parameter.numel() for _, parameter in values)),
            "lr": float(lrs[name]),
            "weight_decay": float(weight_decays[name]),
        })
    return records


def build_optimizers(model: nn.Module, optimization_config: Mapping[str, Any] | None = None):
    """Build explicit volume optimizer groups.

    Returns ``(optimizers, metadata)``.  Grouped AdamW uses one optimizer with
    independently declared encoder/decoder/head rates.  The legacy recipe
    returns two optimizers (AdamW body, momentum-free SGD head) so update order
    remains inspectable and the head LR is never silently copied to AdamW.
    """
    config = _config(optimization_config)
    groups = _extract_groups(model)
    rates = {
        "encoder": float(config["encoder_lr"] if config["encoder_lr"] is not None else config["body_lr"]),
        "decoder": float(config["decoder_lr"] if config["decoder_lr"] is not None else config["body_lr"]),
        "head": float(config["head_lr"]),
    }
    wd = float(config["weight_decay"])
    recipe = config["recipe"]
    if recipe in {"legacy", "legacy_adamw_sgd"}:
        body = [item for key in ("encoder", "decoder") for item in groups[key]]
        head = list(groups["head"])
        optimizers: list[torch.optim.Optimizer] = []
        if body:
            if groups["encoder"] and groups["decoder"] and rates["encoder"] == rates["decoder"]:
                body_groups = [{"params": [p for _, p in body], "lr": rates["encoder"], "weight_decay": wd}]
            else:
                body_groups = []
                if groups["encoder"]:
                    body_groups.append({"params": [p for _, p in groups["encoder"]], "lr": rates["encoder"], "weight_decay": wd})
                if groups["decoder"]:
                    body_groups.append({"params": [p for _, p in groups["decoder"]], "lr": rates["decoder"], "weight_decay": wd})
            optimizers.append(torch.optim.AdamW(body_groups, betas=(0.9, 0.999)))
            # Keep encoder and decoder rates explicit even in the legacy body
            # optimizer; the head remains a separate SGD optimizer.
        if head:
            optimizers.append(torch.optim.SGD([{"params": [p for _, p in head], "lr": rates["head"], "weight_decay": 0.0}], momentum=0.0))
        recipe_kind = "legacy_adamw_sgd"
    else:
        parameter_groups = []
        for name in ("encoder", "decoder", "head"):
            if groups[name]:
                parameter_groups.append({"params": [p for _, p in groups[name]], "lr": rates[name], "weight_decay": wd})
        optimizers = [torch.optim.AdamW(parameter_groups, betas=(0.9, 0.999))]
        recipe_kind = "grouped_adamw"
    metadata = {
        "schema": "self_audit.volume.optim.v1",
        "recipe": recipe_kind,
        "groups": _group_metadata(
            groups,
            rates,
            {name: (0.0 if recipe_kind == "legacy_adamw_sgd" and name == "head" else wd) for name in ("encoder", "decoder", "head")},
        ),
        "optimizer_count": len(optimizers),
        "scheduler": str(config["scheduler"]),
        "scheduler_index": "outer_iteration",
        "lr_warmup_steps": int(config["lr_warmup_steps"]),
        "pseudo_delay": int(config["pseudo_delay"]),
        "min_lr_ratio": float(config["min_lr_ratio"]),
        "clip_norm": config["clip_norm"],
    }
    return optimizers, metadata


class OuterIterationScheduler:
    """A serializable scheduler stepped once per outer transaction.

    Warmup and decay are applied identically to supervised and pseudo updates
    in one outer iteration.  ``pseudo_active`` only reports the declared delay;
    it does not mutate optimizer rates or optimizer state.
    """

    def __init__(self, optimizers: Sequence[torch.optim.Optimizer], config: Mapping[str, Any] | None = None):
        self.optimizers = list(optimizers)
        cfg = _config(config)
        self.mode = str(cfg["scheduler"]).lower()
        if self.mode not in {"none", "constant", "cosine", "linear"}:
            raise ValueError("scheduler must be none, constant, cosine or linear")
        self.warmup_steps = int(cfg["lr_warmup_steps"])
        self.total_steps = int(cfg["total_steps"])
        self.min_lr_ratio = float(cfg["min_lr_ratio"])
        self.pseudo_delay = int(cfg["pseudo_delay"])
        self.base_lrs = [[float(group["lr"]) for group in optimizer.param_groups] for optimizer in self.optimizers]
        self.last_outer_iteration = -1
        self.last_lrs = [list(row) for row in self.base_lrs]

    def _factor(self, outer_iteration: int) -> float:
        index = outer_iteration + 1
        if self.warmup_steps > 0 and index <= self.warmup_steps:
            return index / self.warmup_steps
        if self.mode in {"none", "constant"} or self.total_steps <= 0:
            return 1.0
        progress = min(1.0, max(0.0, (index - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)))
        if self.mode == "cosine":
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0 - (1.0 - self.min_lr_ratio) * progress

    def step(self, outer_iteration: int | None = None) -> list[float]:
        if outer_iteration is None:
            outer_iteration = self.last_outer_iteration + 1
        if type(outer_iteration) is not int or outer_iteration < 0:
            raise ValueError("outer_iteration must be a non-negative integer")
        if outer_iteration < self.last_outer_iteration:
            raise ValueError("scheduler cannot move backwards")
        factor = self._factor(outer_iteration)
        for optimizer, base, last in zip(self.optimizers, self.base_lrs, self.last_lrs, strict=True):
            current = []
            for group, start in zip(optimizer.param_groups, base, strict=True):
                group["lr"] = start * factor
                current.append(float(group["lr"]))
            last[:] = current
        self.last_outer_iteration = outer_iteration
        return self.get_last_lr()

    def pseudo_active(self, outer_iteration: int | None = None) -> bool:
        index = self.last_outer_iteration if outer_iteration is None else outer_iteration
        if type(index) is not int or index < 0:
            return False
        return index >= self.pseudo_delay

    def get_last_lr(self) -> list[float]:
        return [value for row in self.last_lrs for value in row]

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "pseudo_delay": self.pseudo_delay,
            "base_lrs": copy.deepcopy(self.base_lrs),
            "last_outer_iteration": self.last_outer_iteration,
            "last_lrs": copy.deepcopy(self.last_lrs),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {"mode", "warmup_steps", "total_steps", "min_lr_ratio", "pseudo_delay", "base_lrs", "last_outer_iteration", "last_lrs"}
        if set(state) != required:
            raise ValueError("invalid scheduler state")
        if len(state["base_lrs"]) != len(self.optimizers) or any(
            len(values) != len(optimizer.param_groups)
            for values, optimizer in zip(state["base_lrs"], self.optimizers, strict=False)
        ):
            raise ValueError("scheduler state does not match optimizer group structure")
        # Optimizer checkpoints commonly restore the *current* (decayed) LR;
        # scheduler base rates therefore come from this state, not from the
        # optimizer's transient param-group values.
        self.mode = str(state["mode"])
        self.warmup_steps = int(state["warmup_steps"])
        self.total_steps = int(state["total_steps"])
        self.min_lr_ratio = float(state["min_lr_ratio"])
        self.pseudo_delay = int(state["pseudo_delay"])
        self.base_lrs = copy.deepcopy(state["base_lrs"])
        self.last_outer_iteration = int(state["last_outer_iteration"])
        self.last_lrs = copy.deepcopy(state["last_lrs"])
        for optimizer, values in zip(self.optimizers, self.last_lrs, strict=True):
            for group, lr in zip(optimizer.param_groups, values, strict=True):
                group["lr"] = float(lr)


def build_schedulers(optimizers: Sequence[torch.optim.Optimizer], optimization_config: Mapping[str, Any] | None = None):
    """Return one outer-iteration scheduler per optimizer and serializable metadata."""
    scheduler = OuterIterationScheduler(optimizers, optimization_config)
    metadata = {
        "schema": "self_audit.volume.scheduler.v1",
        "index": "outer_iteration",
        "warmup_steps": scheduler.warmup_steps,
        "pseudo_delay": scheduler.pseudo_delay,
        "mode": scheduler.mode,
    }
    return [scheduler], metadata


def _has_finite_gradient(optimizers: Sequence[torch.optim.Optimizer]) -> bool:
    found = False
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                found = True
                if not bool(torch.isfinite(parameter.grad).all().detach()):
                    raise FloatingPointError("non-finite optimizer gradient")
    return found


def safe_optimizer_step(
    optimizers: Sequence[torch.optim.Optimizer],
    *,
    clip_norm: float | None = None,
    has_support: bool = True,
) -> dict[str, Any]:
    """Step optimizers iff a supported, finite gradient exists.

    The returned receipt records actual optimizer/update counts and parameter
    and gradient norms.  Empty support means no ``step`` call and therefore no
    AdamW weight-decay mutation.
    """
    if not has_support:
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        return {"updated": False, "optimizer_updates": 0, "gradient_norm": 0.0, "parameter_update_norm": 0.0}
    if not _has_finite_gradient(optimizers):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        return {"updated": False, "optimizer_updates": 0, "gradient_norm": 0.0, "parameter_update_norm": 0.0}
    parameters = []
    before = []
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameters.append(parameter)
                    before.append(parameter.detach().clone())
    gradient_norm_tensor = torch.sqrt(sum((parameter.grad.detach().float().pow(2).sum() for parameter in parameters), torch.zeros((), device=parameters[0].device)))
    gradient_norm = float(gradient_norm_tensor.detach())
    if clip_norm is not None:
        if isinstance(clip_norm, bool) or not isinstance(clip_norm, (int, float)) or clip_norm <= 0:
            raise ValueError("clip_norm must be positive or None")
        torch.nn.utils.clip_grad_norm_(parameters, float(clip_norm), error_if_nonfinite=True)
    for optimizer in optimizers:
        optimizer.step()
    delta_sq = torch.zeros((), device=parameters[0].device)
    for parameter, initial in zip(parameters, before, strict=True):
        delta_sq = delta_sq + (parameter.detach() - initial).float().pow(2).sum()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    return {
        "updated": True,
        "optimizer_updates": len(optimizers),
        "gradient_norm": gradient_norm,
        "parameter_update_norm": float(torch.sqrt(delta_sq).detach()),
    }


# More descriptive aliases used by engine integrations.
step_optimizers = safe_optimizer_step
build_outer_schedulers = build_schedulers
