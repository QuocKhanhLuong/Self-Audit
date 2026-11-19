"""Exact Candidate PG student update math.

This module intentionally owns one optimizer-step worth of student work.  The
engine supplies frames and already materialised views; this function does not
sample data, update an optimizer, or mutate model mode.  All teacher/target
paths are detached, while the labelled fusion path, the student-only path and
the optional CBKD branch remain connected to ``student`` parameters.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from self_audit.candidate_q3d.alignment import pool_native_cells

from . import codebook as cb
from . import losses, reliability
from .views import CutMixBox, center_crop_pad, cutmix_box, mix_packet, mix_tensor

_MISSING = object()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    value = mapping.get(key, _MISSING)
    if value is _MISSING:
        raise KeyError(f"missing required config key {path}.{key}")
    return value


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _required(config, key, "config")
    return _mapping(value, key)


def _tensor(value: Any, *, name: str, device: torch.device | None = None) -> Tensor:
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, device=device)
    elif device is not None:
        value = value.to(device=device)
    return value


def _valid4(value: Tensor, shape: Sequence[int], *, name: str = "valid") -> Tensor:
    value = _tensor(value, name=name)
    if value.ndim == len(shape) + 1 and value.shape[1] == 1:
        value = value[:, 0]
    if tuple(value.shape) != tuple(int(v) for v in shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if value.dtype.is_complex:
        raise TypeError(f"{name} must be real")
    if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN/Inf")
    return value.bool()


def _frame_image(frame: Any, *, device: torch.device | None = None) -> Tensor:
    image = _tensor(getattr(frame, "image", _MISSING), name="frame.image", device=device).float()
    if image.ndim == 4:
        image = image.unsqueeze(0)
    if image.ndim != 5 or image.shape[1] != 1:
        raise ValueError("frame image must have shape [B,1,D,H,W]")
    if not bool(torch.isfinite(image).all()) or bool(((image < 0) | (image > 1)).any()):
        raise ValueError("frame image must be finite and normalized to [0,1]")
    return image


def _frame_target(frame: Any, *, device: torch.device) -> Tensor:
    target = getattr(frame, "target", _MISSING)
    if target is _MISSING or target is None:
        raise ValueError("labelled frame is missing target")
    target = _tensor(target, name="frame.target", device=device).long()
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if target.ndim != 4 or bool(((target < 0) | (target > 3)).any()):
        raise ValueError("frame target must be integer [B,D,H,W] with classes 0..3")
    return target


def _frame_geometry(frame: Any, image: Tensor) -> Mapping[str, Any]:
    geometry = getattr(frame, "geometry", None)
    if not isinstance(geometry, Mapping):
        raise TypeError("frame geometry must be a mapping")
    shape = geometry.get("shape")
    if shape is None or len(shape) != 3:
        raise ValueError("frame geometry must provide shape=(D,H,W)")
    return geometry


def _native_valid(image: Tensor) -> Tensor:
    return torch.ones((image.shape[0], *image.shape[-3:]), dtype=torch.bool, device=image.device)


def _view_get(view: Any, *names: str, default: Any = _MISSING) -> Any:
    if isinstance(view, Mapping):
        for name in names:
            if name in view:
                return view[name]
    else:
        for name in names:
            if hasattr(view, name):
                return getattr(view, name)
    if default is not _MISSING:
        return default
    raise KeyError(f"view is missing one of {names}")


def _normalize_view_container(view: Any, *, labelled: bool) -> Any:
    if isinstance(view, Sequence) and not isinstance(view, (str, bytes, Mapping)):
        values = list(view)
        if labelled and len(values) == 5:
            return {"weak": values[0], "strong": values[1], "target": values[2], "valid": values[3], "geometry": values[4]}
        if not labelled and len(values) in (4, 5):
            result = {"weak": values[0], "strong": values[1], "valid": values[2], "geometry": values[3]}
            if len(values) == 5:
                result["strong2"] = values[4]
            return result
    return view


def _coerce_view(view: Any, frame: Any, *, target: Tensor | None = None, device: torch.device | None = None) -> dict[str, Any]:
    """Return a canonical view mapping without drawing any random numbers."""

    view = _normalize_view_container(view, labelled=target is not None)
    if view is None:
        raise ValueError("precomputed view mapping is required")
    image = _frame_image(frame, device=device)
    weak = _view_get(view, "weak", "weak_image", "image")
    weak = _tensor(weak, name="weak", device=image.device).float()
    if weak.ndim == 4:
        weak = weak.unsqueeze(0)
    if weak.ndim != 5 or weak.shape[1] != 1 or not bool(torch.isfinite(weak).all()):
        raise ValueError("weak view must be finite [B,1,D,H,W]")
    if bool(((weak < 0) | (weak > 1)).any()):
        raise ValueError("weak view must be normalized to [0,1]")
    strong = _view_get(view, "strong", "strong1", "strong_image")
    strong = _tensor(strong, name="strong", device=image.device).float()
    if strong.ndim == 4:
        strong = strong.unsqueeze(0)
    if strong.ndim != 5 or strong.shape[1] != 1 or tuple(strong.shape[-3:]) != tuple(weak.shape[-3:]):
        raise ValueError("strong view must share weak [B,1,D,H,W] shape")
    if not bool(torch.isfinite(strong).all()) or bool(((strong < 0) | (strong > 1)).any()):
        raise ValueError("strong view must be finite and normalized to [0,1]")
    valid = _view_get(view, "valid", "weak_valid")
    valid = _valid4(_tensor(valid, name="valid", device=image.device), (weak.shape[0], *weak.shape[-3:]))
    geometry = _view_get(view, "geometry", "geom")
    if target is not None:
        view_target = _view_get(view, "target", "weak_target")
        view_target = _tensor(view_target, name="target", device=image.device).long()
        if view_target.ndim == 5 and view_target.shape[1] == 1:
            view_target = view_target[:, 0]
        if tuple(view_target.shape) != tuple(valid.shape):
            raise ValueError("labelled target must align with weak view")
    else:
        view_target = None
    return {"weak": weak, "strong": strong, "valid": valid, "geometry": geometry, "target": view_target}


def _split_unlabelled_views(u_views: Any, a: Any, b: Any) -> tuple[Any, Any]:
    if isinstance(u_views, Mapping) and ("a" in u_views or "A" in u_views):
        va = u_views.get("a", u_views.get("A"))
        vb = u_views.get("b", u_views.get("B"))
        if vb is None:
            raise ValueError("u_views mapping must contain both A and B views")
        return _normalize_view_container(va, labelled=False), _normalize_view_container(vb, labelled=False)
    if isinstance(u_views, Sequence) and not isinstance(u_views, (str, bytes)) and len(u_views) == 2:
        return _normalize_view_container(u_views[0], labelled=False), _normalize_view_container(u_views[1], labelled=False)
    if u_views is None:
        raise ValueError("u_views must provide explicit A and B view mappings")
    raise TypeError("u_views must provide A/B mappings or a two-item sequence")


def _strong2(view: Any, canonical: Mapping[str, Any]) -> Tensor:
    if isinstance(view, Mapping):
        value = _view_get(view, "strong2", "strong_view2", default=None)
        if value is None:
            raise ValueError("n_strong_views=2 requires an independently generated strong2 view")
        value = _tensor(value, name="strong2", device=canonical["strong"].device).float()
        if value.ndim == 4:
            value = value.unsqueeze(0)
        if tuple(value.shape) != tuple(canonical["strong"].shape):
            raise ValueError("strong2 must share strong1 shape")
        if value.data_ptr() == canonical["strong"].data_ptr():
            raise ValueError("strong2 must be an independently generated tensor")
        if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
            raise ValueError("strong2 must be finite and normalized to [0,1]")
        return value
    raise TypeError("n_strong_views=2 requires mapping views with strong2")


def _student_logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if not isinstance(output, Mapping):
        raise TypeError("student output must be a tensor or mapping")
    for key in ("logits", "region_logits", "fusion_logits"):
        value = output.get(key)
        if isinstance(value, Tensor):
            return value
    raise KeyError("student output has no logits")


def _teacher_output(teacher: nn.Module, image: Tensor, geometry: Mapping[str, Any]) -> Mapping[str, Any]:
    output = teacher(image, geometry, return_features=True)
    if not isinstance(output, Mapping):
        raise TypeError("teacher must return a mapping")
    if not isinstance(output.get("region_logits"), Tensor):
        raise KeyError("teacher output must contain region_logits")
    features = output.get("features")
    if features is not None and not isinstance(features, (tuple, list)):
        raise TypeError("teacher features must be a tuple/list")
    return output


def _teacher_features(output: Mapping[str, Any]) -> tuple[Tensor, ...] | list[Tensor]:
    features = output.get("features")
    if not isinstance(features, (tuple, list)) or len(features) != 5 or not all(isinstance(value, Tensor) for value in features):
        raise RuntimeError("teacher fusion path requires five tensor feature levels")
    return features


def _generator_required(generator: torch.Generator | None, name: str) -> torch.Generator:
    if not isinstance(generator, torch.Generator):
        raise TypeError(f"{name} must be an explicit torch.Generator")
    return generator


def _keep_mask(config: Mapping[str, Any], batch: int, device: torch.device,
               update: int, generator: torch.Generator) -> Tensor:
    student_cfg = _section(config, "student")
    schedule = _mapping(_required(student_cfg, "teacher_dropout_on_labeled", "student"), "teacher_dropout_on_labeled")
    start = float(_required(schedule, "start", "student.teacher_dropout_on_labeled"))
    end = float(_required(schedule, "end", "student.teacher_dropout_on_labeled"))
    training_cfg = _section(config, "training")
    total = int(_required(training_cfg, "updates", "training"))
    if not all(math.isfinite(v) for v in (start, end)) or not (0 <= start <= 1 and 0 <= end <= 1):
        raise ValueError("student teacher-dropout schedule must lie in [0,1]")
    if update < 0 or total <= 0:
        raise ValueError("update/training.updates are invalid")
    probability = start + (end - start) * update / max(total - 1, 1)
    draw_device = torch.device(getattr(generator, "device", "cpu"))
    draw = torch.rand((batch, 1, 1, 1, 1), generator=generator, device=draw_device)
    return (draw >= probability).to(device=device, dtype=torch.float32)


def _extract_thresholds(threshold: Any, device: torch.device) -> Tensor:
    if threshold is None:
        raise ValueError("PG requires a class-adaptive threshold state")
    values = getattr(threshold, "thresholds", _MISSING)
    if values is _MISSING:
        if isinstance(threshold, Tensor):
            values = threshold
        else:
            raise TypeError("threshold must expose thresholds")
    values = _tensor(values, name="thresholds", device=device).detach().reshape(-1)
    if values.numel() != 4 or not bool(torch.isfinite(values).all()) or bool((values < 0).any()) or bool((values > 1).any()):
        raise ValueError("thresholds must contain four finite probabilities")
    return values


def _flatten_threshold_observation(probabilities: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    valid = _valid4(valid, (probabilities.shape[0], *probabilities.shape[2:]))
    flat = probabilities.permute(1, 0, *range(2, probabilities.ndim)).reshape(probabilities.shape[1], -1).T
    mask = valid.reshape(-1)
    flat = flat[mask]
    if flat.numel() == 0:
        flat = probabilities.new_zeros((1, probabilities.shape[1]))
        support = torch.zeros((1,), dtype=torch.bool, device=probabilities.device)
    else:
        support = torch.ones((flat.shape[0],), dtype=torch.bool, device=probabilities.device)
    return flat.T.reshape(1, probabilities.shape[1], -1, 1, 1), support.reshape(1, -1, 1, 1)


def _update_threshold(threshold: Any, p_a: Tensor, valid_a: Tensor, p_b: Tensor, valid_b: Tensor) -> None:
    update = getattr(threshold, "update", None)
    if not callable(update):
        raise TypeError("threshold must expose update(probabilities, valid)")
    pa, va = _flatten_threshold_observation(p_a.detach(), valid_a)
    pb, vb = _flatten_threshold_observation(p_b.detach(), valid_b)
    probs = torch.cat((pa, pb), dim=2)
    valid = torch.cat((va, vb), dim=1)
    update(probs, valid)


def _sdf_output(output: Mapping[str, Any]) -> Tensor | None:
    value = output.get("sdf")
    if isinstance(value, Tensor):
        return value.detach()
    value = output.get("sdf_all")
    if isinstance(value, Tensor) and value.shape[1] >= 4:
        return value[:, 1:].detach()
    return None


def _crop_to_anchor(value: Tensor, anchor_shape: Sequence[int], *, pad_value: float = 0) -> Tensor:
    if tuple(value.shape[-3:]) == tuple(int(v) for v in anchor_shape):
        return value
    return center_crop_pad(value, tuple(int(v) for v in anchor_shape), pad_value=pad_value)


def _pad_high_grid(value: Tensor, target_shape: Sequence[int], *, pad_value: float = 0) -> Tensor:
    """High-side pad pooled native cells to an internal student grid.

    ``pool_native_cells`` intentionally returns only cells whose centers lie
    in the native volume; PGStudent can have one or more high-side internal pad
    cells after its native H/W canvas is rounded to a multiple of 16.  Those
    cells must be explicit zero-support cells, never nearest labels.
    """

    target = tuple(int(v) for v in target_shape)
    current = tuple(int(v) for v in value.shape[-3:])
    if any(current[axis] > target[axis] for axis in range(3)):
        raise RuntimeError(f"pooled grid {current} exceeds student grid {target}")
    if current == target:
        return value
    pads = (0, target[2] - current[2], 0, target[1] - current[1], 0, target[0] - current[0])
    return F.pad(value, pads, mode="constant", value=float(pad_value))


def _sample_box(view_a: Mapping[str, Any], view_b: Mapping[str, Any], valid_a: Tensor, valid_b: Tensor,
                *, generator: torch.Generator | None, cutmix_cfg: Mapping[str, Any], update: int, index: int) -> CutMixBox:
    boxes = view_a.get("boxes")
    if isinstance(boxes, Sequence) and not isinstance(boxes, (str, bytes)) and boxes:
        value = boxes[index] if index < len(boxes) else boxes[-1]
        return tuple(int(v) for v in value)  # type: ignore[return-value]
    value = view_a.get("box")
    if value is not None and index == 0:
        return tuple(int(v) for v in value)  # type: ignore[return-value]
    sampler = view_a.get("sample_box") or view_b.get("sample_box")
    if callable(sampler):
        return sampler(valid_a, partner_valid=valid_b, **dict(cutmix_cfg))
    if generator is None:
        raise RuntimeError("real PG steps require the checkpointable CutMix generator or explicit boxes")
    return cutmix_box(valid_a, generator, partner_valid=valid_b, **dict(cutmix_cfg))


def _pg_loss(logits: Tensor, packet: Mapping[str, Tensor], p_teacher: Tensor,
             p_student: Tensor, thresholds: Tensor, sdf: Tensor | None) -> Tensor:
    valid = packet["valid"]
    if valid.ndim == 5 and valid.shape[1] == 1:
        valid = valid[:, 0]
    weights = reliability.voxel_weights(
        packet["p_W"].detach(), p_teacher.detach(), p_student.detach(), valid,
        thresholds.detach(), d_T=None if sdf is None else sdf.detach(),
    )
    probs = logits.softmax(dim=1)
    return losses.weighted_ce(logits, packet["y"], weights, valid) + losses.weighted_dice(probs, packet["y"], weights, valid)


def _aligned_teacher_level3(teacher: nn.Module, student: nn.Module, output: Mapping[str, Any],
                            geometry: Mapping[str, Any], target_shape: Sequence[int]) -> Tensor:
    features = output.get("features")
    if not isinstance(features, (tuple, list)) or len(features) < 4 or not isinstance(features[3], Tensor):
        raise RuntimeError("CBKD requires teacher level-3 features")
    teacher_phases = getattr(teacher, "feature_phases", getattr(teacher, "teacher_phases", None))
    student_phases = getattr(student, "phases", None)
    if not isinstance(teacher_phases, (tuple, list)) or len(teacher_phases) < 4:
        raise RuntimeError("teacher does not expose feature phases")
    if not isinstance(student_phases, (tuple, list)) or len(student_phases) < 4:
        raise RuntimeError("student does not expose feature phases")
    # This helper detaches teacher inputs/outputs by contract, which is exactly
    # what CBKD needs.  Student tensors never pass through it.
    from self_audit.candidate_fuse.fusion import align_feature
    return align_feature(features[3], tuple(int(v) for v in geometry["shape"]),
                         teacher_phases[3], student_phases[3], tuple(int(v) for v in target_shape)).detach()


def _pooled_gt(target: Tensor, geometry: Mapping[str, Any], student: nn.Module,
               target_shape: Sequence[int], valid: Tensor | None = None) -> tuple[Tensor, Tensor]:
    phases = getattr(student, "phases", None)
    if not isinstance(phases, (tuple, list)) or len(phases) < 4:
        raise RuntimeError("student does not expose stage-3 phase")
    native_shape = tuple(int(v) for v in geometry["shape"])
    labels = target.long()
    onehot = F.one_hot(labels.clamp(0, 3), num_classes=4).permute(0, 4, 1, 2, 3).to(dtype=torch.float32)
    native_mask = None if valid is None else valid[:, None].to(dtype=torch.float32)
    pooled = pool_native_cells(onehot, native_shape, phases[3], native_mask=native_mask)
    pooled_support = pool_native_cells(torch.ones_like(onehot[:, :1]), native_shape, phases[3], native_mask=native_mask)[:, 0]
    pooled, pooled_support = _pad_high_grid(pooled, target_shape, pad_value=0.0), _pad_high_grid(pooled_support[:, None], target_shape, pad_value=0.0)[:, 0]
    support = pooled_support > 0
    return pooled, support


def _pooled_valid(valid: Tensor, geometry: Mapping[str, Any], student: nn.Module,
                  target_shape: Sequence[int]) -> Tensor:
    phases = getattr(student, "phases", None)
    if not isinstance(phases, (tuple, list)) or len(phases) < 4:
        raise RuntimeError("student does not expose stage-3 phase")
    native_shape = tuple(int(v) for v in geometry["shape"])
    native = valid[:, None].to(dtype=torch.float32)
    pooled = pool_native_cells(native, native_shape, phases[3], native_mask=native)
    return _pad_high_grid(pooled, target_shape, pad_value=0.0)[:, 0] > 0


def _cbkd_loss(student: nn.Module, teacher: nn.Module, codebook: nn.Module, teacher_l: Mapping[str, Any], teacher_a: Mapping[str, Any],
               student_l: Mapping[str, Any], student_a: Mapping[str, Any], labeled_target: Tensor,
               labeled_geometry: Mapping[str, Any], u_geometry: Mapping[str, Any], valid_l: Tensor, valid_a: Tensor,
               weights: Mapping[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
    branch = getattr(student, "code_branch", None)
    if not isinstance(branch, nn.Module):
        raise TypeError("CBKD requires PGStudent.code_branch")
    if not isinstance(codebook, nn.Module):
        raise TypeError("CBKD requires a frozen codebook module")
    if any(parameter.requires_grad for parameter in codebook.parameters()):
        raise RuntimeError("CBKD codebook must be frozen")
    sl_code = student_l.get("code_branch")
    sa_code = student_a.get("code_branch")
    if not isinstance(sl_code, Mapping) or not isinstance(sa_code, Mapping):
        raise TypeError("CBKD requires labelled and clean-U-A code packets")
    stage_l = student_l.get("stage3")
    stage_a = student_a.get("stage3")
    if not isinstance(stage_l, Tensor) or not isinstance(stage_a, Tensor):
        raise TypeError("CBKD requires stage3 outputs")
    support_l = _pooled_valid(valid_l, labeled_geometry, student, stage_l.shape[-3:])
    support_a = _pooled_valid(valid_a, u_geometry, student, stage_a.shape[-3:])
    valid_l_code = sl_code.get("valid_mask")
    valid_a_code = sa_code.get("valid_mask")
    if isinstance(valid_l_code, Tensor):
        valid_l_code = valid_l_code[:, 0].bool() & support_l
    if isinstance(valid_a_code, Tensor):
        valid_a_code = valid_a_code[:, 0].bool() & support_a
    with torch.no_grad():
        t_l_feature = _aligned_teacher_level3(teacher, student, teacher_l,
                                               labeled_geometry, stage_l.shape[-3:])
        t_a_feature = _aligned_teacher_level3(teacher, student, teacher_a,
                                               u_geometry, stage_a.shape[-3:])
        t_l = codebook(t_l_feature, valid_mask=valid_l_code)
        t_a = codebook(t_a_feature, valid_mask=valid_a_code)
    rho_l = t_l["rho"].detach(); rho_a = t_a["rho"].detach()
    # The branch computes P_S/H_S_Q with gradient.  Its codebook is a buffer,
    # so rho/STE/commit cannot update C.
    rho_sl = sl_code["rho"]; rho_sa = sa_code["rho"]
    valid_sl = valid_l_code
    valid_sa = valid_a_code
    assign_l = branch.loss_assign_kl(rho_l, rho_sl, valid_sl)
    assign_a = branch.loss_assign_kl(rho_a, rho_sa, valid_sa)
    commit_l = branch.loss_commit(sl_code, valid_sl)
    commit_a = branch.loss_commit(sa_code, valid_sa)
    pooled, pooled_valid = _pooled_gt(labeled_target, labeled_geometry, student, stage_l.shape[-3:], valid_l)
    # qtask is the exact soft-CE task on pooled native GT; no nearest-label
    # interpolation and no Dice term are used for the student CBKD branch.
    qtask = cb.soft_cross_entropy(sl_code["q_logits"], pooled, pooled_valid & support_l)
    terms = {
        "assign": assign_l + assign_a,
        "commit": commit_l + commit_a,
        "qtask": qtask,
    }
    assign_weight = float(_required(weights, "assign", "cbkd.weights"))
    qtask_weight = float(_required(weights, "qtask", "cbkd.weights"))
    commit_weight = float(_required(weights, "commit", "cbkd.weights"))
    qpred_weight = float(_required(weights, "qpred", "cbkd.weights"))
    if min(assign_weight, qtask_weight, commit_weight, qpred_weight) < 0:
        raise ValueError("CBKD weights must be nonnegative")
    qpred = sl_code["z"].sum() * 0.0
    total = assign_weight * terms["assign"] + qtask_weight * terms["qtask"] + commit_weight * terms["commit"] + qpred_weight * qpred
    return total, terms


def _cbkd_sa_loss(student: nn.Module, teacher: nn.Module, codebook: nn.Module, teacher_l: Mapping[str, Any], teacher_a: Mapping[str, Any],
                  student_l: Mapping[str, Any], student_a: Mapping[str, Any], labeled_target: Tensor,
                  labeled_geometry: Mapping[str, Any], u_geometry: Mapping[str, Any], valid_l: Tensor, valid_a: Tensor,
                  weights: Mapping[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
    """CBKD-SA (rev 4 section 5): ``L_pm`` to the atom-reconstructed teacher plus ``L_atom``.

    * ``L_pm`` : student-only full-res softmax vs ``q_T^full`` (teacher re-assembled
      from its codes through the frozen ``D_A``); weight ``m_T`` on clean weak U, 1 on L_K.
    * ``L_atom``: student stage3 -> ``P_S`` -> hard VQ (frozen ``C``, STE) -> frozen ``D_A``
      vs GT patch (CE + Dice + SDF L1) on L_K, plus commitment on L_K and clean weak U.
    Teacher and codebook tensors are detached; only ``P_S`` and the student receive gradient.
    """

    from . import shape_atoms as sa

    branch = getattr(student, "code_branch", None)
    if not isinstance(branch, sa.StudentAtomBranch):
        raise TypeError("cbkd_sa requires PGStudent.code_branch = StudentAtomBranch")
    if not isinstance(codebook, sa.ShapeAtomCodebook):
        raise TypeError("cbkd_sa requires a frozen ShapeAtomCodebook")
    if any(parameter.requires_grad for parameter in codebook.parameters()):
        raise RuntimeError("CBKD-SA codebook must be frozen")
    code_l, code_a = student_l.get("code_branch"), student_a.get("code_branch")
    if not isinstance(code_l, Mapping) or not isinstance(code_a, Mapping):
        raise TypeError("cbkd_sa requires labelled and clean-U code packets")
    stage_l, stage_a = student_l["stage3"], student_a["stage3"]
    native_l = tuple(int(v) for v in labeled_target.shape[-3:])
    native_a = tuple(int(v) for v in valid_a.shape[-3:])
    with torch.no_grad():
        atoms_l = sa.teacher_atoms(codebook, _aligned_teacher_level3(teacher, student, teacher_l, labeled_geometry, stage_l.shape[-3:]), native_l)
        atoms_a = sa.teacher_atoms(codebook, _aligned_teacher_level3(teacher, student, teacher_a, u_geometry, stage_a.shape[-3:]), native_a)
    p_l = _student_logits(student_l).softmax(1)
    p_a = _student_logits(student_a).softmax(1)
    pm_l = sa.pm_loss(p_l, atoms_l["q"], torch.ones_like(atoms_l["margin"]), torch.ones_like(atoms_l["margin"]), valid_l)
    pm_a = sa.pm_loss(p_a, atoms_a["q"], atoms_a["margin"], torch.ones_like(atoms_a["margin"]), valid_a)
    loss_pm = pm_l + pm_a
    sdf_target = None
    if "spacing_zyx" in labeled_geometry:
        from .teacher_training import _sdf_targets4

        sdf_target = _sdf_targets4(labeled_target, labeled_geometry)[:, 1:].detach()
    atom, atom_terms = sa.atom_loss(code_l, labeled_target, sdf_target, valid_l, native_l, codebook.patch)
    commit = sa.commitment(code_l["z"], code_l["e"], code_l.get("valid_mask")) + sa.commitment(code_a["z"], code_a["e"], code_a.get("valid_mask"))
    commit_weight = float(_required(weights, "commit", "cbkd.weights"))
    loss_atom = atom + commit_weight * commit
    usage = sa.code_usage(code_a["idx"].detach(), code_a.get("valid_mask"), codebook.size)
    m = atoms_a["margin"]
    terms = {"pm": loss_pm, "atom": loss_atom, "commit": commit, "pm_l": pm_l, "pm_a": pm_a,
             "m_T_mean": m.mean(), "student_code_perplexity": torch.tensor(usage["perplexity"]),
             "student_code_dead": torch.tensor(float(usage["dead"])), **atom_terms}
    return loss_pm + loss_atom, terms


def _config_lambda_u(config: Mapping[str, Any], update: int) -> float:
    pg = _section(config, "pg")
    lam = _mapping(_required(pg, "lambda_u", "pg"), "pg.lambda_u")
    maximum = float(_required(lam, "max", "pg.lambda_u"))
    warmup = int(_required(lam, "warmup_updates_labeled_only", "pg.lambda_u"))
    fraction = float(_required(lam, "ramp_fraction", "pg.lambda_u"))
    total = int(_required(_section(config, "training"), "updates", "training"))
    if maximum < 0 or warmup < 0 or fraction <= 0 or total <= 0:
        raise ValueError("invalid PG lambda_u configuration")
    if update < warmup:
        return 0.0
    return min(maximum, (update - warmup + 1) / max(1, math.ceil(fraction * total)))


def _config_lambda_cb(config: Mapping[str, Any], update: int) -> float:
    cbkd_cfg = _section(config, "cbkd")
    warmup = int(_required(cbkd_cfg, "warmup_updates_labeled_only", "cbkd"))
    value = float(_required(cbkd_cfg, "lambda_cb", "cbkd"))
    if value < 0 or warmup < 0:
        raise ValueError("invalid CBKD warmup/lambda")
    if cbkd_cfg.get("variant") == "cbkd_sa":
        # Staggered onset with its own linear ramp (rev 4 section 8.8).
        from . import shape_atoms as sa

        return sa.lambda_cb(update, lam=value, onset=warmup, ramp=int(cbkd_cfg.get("ramp_updates", 400)))
    return value if update >= warmup else 0.0


def student_training_step(
    config: Mapping[str, Any],
    student: nn.Module,
    teacher: nn.Module,
    codebook: nn.Module | None,
    labeled: Any,
    u_a: Any,
    u_b: Any,
    update: int,
    threshold: Any,
    *,
    labeled_views: Any,
    u_views: Any,
    keep_generator: torch.Generator,
    fp_generator: torch.Generator,
) -> dict[str, Any]:
    """Compute one exact PG student step.

    The returned ``loss``, ``loss_sup``, ``loss_pg`` and ``loss_cbkd`` retain
    autograd graphs.  Every value under ``diagnostics`` (and every raw scalar
    component there) is detached, so logging cannot accidentally influence the
    update.
    """

    config = _mapping(config, "config")
    target_mode = _mapping(config.get("pg", {}), "pg").get("target_mode", "fusion")
    if target_mode != "fusion":
        # W2S-Match arms (student-only labelled path; no teacher-feature fusion).
        from .w2s_step import TARGET_MODES, w2s_student_training_step

        if target_mode not in TARGET_MODES:
            raise ValueError(f"unknown pg.target_mode {target_mode!r}")
        return w2s_student_training_step(
            config, student, teacher, codebook, labeled, u_a, u_b, update,
            labeled_views=labeled_views, u_views=u_views, keep_generator=keep_generator, fp_generator=fp_generator,
        )
    if not isinstance(student, nn.Module) or not isinstance(teacher, nn.Module):
        raise TypeError("student and teacher must be torch modules")
    _generator_required(keep_generator, "keep_generator")
    _generator_required(fp_generator, "fp_generator")
    if isinstance(update, bool) or int(update) != update or update < 0:
        raise ValueError("update must be a nonnegative integer")
    update = int(update)
    device = next(student.parameters(), torch.empty((), device="cpu")).device
    labeled_image = _frame_image(labeled, device=device)
    labeled_target = _frame_target(labeled, device=device)
    lv = _coerce_view(labeled_views, labeled, target=labeled_target, device=device)
    if tuple(lv["target"].shape) != tuple(labeled_target.shape):
        labeled_target = lv["target"]
    ua_view, ub_view = _split_unlabelled_views(u_views, u_a, u_b)
    ua = _coerce_view(ua_view, u_a, device=device)
    ub = _coerce_view(ub_view, u_b, device=device)
    if tuple(ua["weak"].shape[1:2]) != (1,) or tuple(ub["weak"].shape[1:2]) != (1,):
        raise ValueError("unlabelled views must be single-channel")

    keep_l = _keep_mask(config, labeled_image.shape[0], labeled_image.device, update, keep_generator)
    # Teacher on weak L is always frozen/no-grad.  Fusion L and student-only L
    # consume the exact same strong tensor and target.
    with torch.no_grad():
        teacher_l = _teacher_output(teacher, lv["weak"], lv["geometry"])
    features_l = _teacher_features(teacher_l)
    pg_cfg = _section(config, "pg")
    cbkd_cfg = _section(config, "cbkd")
    pg_enabled = bool(_required(pg_cfg, "enabled", "pg"))
    cbkd_enabled = bool(_required(cbkd_cfg, "enabled", "cbkd"))
    warmup_pg = int(_required(_mapping(_required(pg_cfg, "lambda_u", "pg"), "pg.lambda_u"), "warmup_updates_labeled_only", "pg.lambda_u"))
    warmup_cb = int(_required(cbkd_cfg, "warmup_updates_labeled_only", "cbkd"))
    cb_active_now = cbkd_enabled and update >= warmup_cb
    out_l_fusion = student(lv["strong"], lv["geometry"], teacher_features=features_l, keep_mask=keep_l)
    out_l_student = student(lv["strong"], lv["geometry"], return_stage3=cb_active_now, return_code=cb_active_now)
    loss_sup_fusion = _supervised_loss(_student_logits(out_l_fusion), lv["target"])
    loss_sup_student = _supervised_loss(_student_logits(out_l_student), lv["target"])
    loss_sup = 0.5 * loss_sup_fusion + 0.5 * loss_sup_student
    zero = loss_sup * 0.0
    loss_pg = zero
    loss_cbkd = zero
    diagnostics: dict[str, Any] = {
        "lambda_u": 0.0,
        "lambda_cb": 0.0,
        "keep_mean": keep_l.detach().mean(),
        "loss_sup_fusion": loss_sup_fusion.detach(),
        "loss_sup_student": loss_sup_student.detach(),
        "loss_pg_s1": zero.detach(),
        "loss_pg_s2": zero.detach(),
        "loss_pg_fp": zero.detach(),
        "loss_cbkd_assign": zero.detach(),
        "loss_cbkd_commit": zero.detach(),
        "loss_cbkd_qtask": zero.detach(),
        "code_perplexity": None,
        "code_dead": None,
        "gate_means": tuple(float(v) if v is not None else float("nan") for v in out_l_fusion.get("gate_means", ())) if isinstance(out_l_fusion, Mapping) else (),
    }

    any_u_active = (pg_enabled and update >= warmup_pg) or (cbkd_enabled and update >= warmup_cb)
    if any_u_active:
        # U teacher targets and fusion-weak pW are no-grad.  There is no U
        # forward at all during the locked first-800 labelled-only updates.
        with torch.no_grad():
            teacher_a = _teacher_output(teacher, ua["weak"], ua["geometry"])
            p_teacher_a = teacher_a["region_logits"].softmax(1).detach()
            teacher_b: Mapping[str, Any] | None = None
            p_teacher_b: Tensor | None = None
            if pg_enabled and update >= warmup_pg:
                teacher_b = _teacher_output(teacher, ub["weak"], ub["geometry"])
                p_teacher_b = teacher_b["region_logits"].softmax(1).detach()
            p_fusion_a: Tensor | None = None
            fusion_b: Mapping[str, Any] | None = None
            p_fusion_b: Tensor | None = None
            if pg_enabled:
                keep_a = torch.ones((ua["weak"].shape[0], 1, 1, 1, 1), dtype=torch.float32, device=device)
                fusion_a = student(ua["weak"], ua["geometry"], teacher_features=_teacher_features(teacher_a), keep_mask=keep_a)
                p_fusion_a = _student_logits(fusion_a).softmax(1).detach()
            if pg_enabled and teacher_b is not None:
                fusion_b = student(ub["weak"], ub["geometry"], teacher_features=_teacher_features(teacher_b), keep_mask=torch.ones_like(keep_a))
                p_fusion_b = _student_logits(fusion_b).softmax(1).detach()
        # Clean weak A has gradient only when CBKD is active; clean B never
        # participates in CBKD and is evaluated under no_grad.
        if cb_active_now:
            clean_a = student(ua["weak"], ua["geometry"], return_stage3=True, return_code=True)
        else:
            with torch.no_grad():
                clean_a = student(ua["weak"], ua["geometry"])
        with torch.no_grad():
            clean_b = student(ub["weak"], ub["geometry"]) if pg_enabled and update >= warmup_pg else None
        p_student_a = _student_logits(clean_a).softmax(1).detach()
        p_student_b = None if clean_b is None else _student_logits(clean_b).softmax(1).detach()

        if pg_enabled and update >= warmup_pg:
            assert teacher_b is not None and p_teacher_b is not None and p_fusion_a is not None and p_fusion_b is not None and p_student_b is not None
            pg_lambda = _config_lambda_u(config, update)
            thresholds = _extract_thresholds(threshold, device)
            y_a = p_fusion_a.argmax(1); y_b = p_fusion_b.argmax(1)
            confidence_a = p_fusion_a.max(1).values; confidence_b = p_fusion_b.max(1).values
            packet_a = {"p_W": p_fusion_a, "y": y_a, "w": confidence_a, "valid": ua["valid"]}
            packet_b = {"p_W": p_fusion_b, "y": y_b, "w": confidence_b, "valid": ub["valid"]}
            cutmix_cfg = _mapping(_required(_section(config, "augmentation"), "cutmix", "augmentation"), "augmentation.cutmix")
            if _required(cutmix_cfg, "full_depth", "augmentation.cutmix") is not True:
                raise ValueError("PG CutMix must be full-depth")
            cutmix_args = {"area_fraction": _required(cutmix_cfg, "area_fraction", "augmentation.cutmix"),
                           "align_px": int(_required(cutmix_cfg, "align_px", "augmentation.cutmix")), "full_depth": True}
            cutmix_generator = None
            if isinstance(u_views, Mapping):
                cutmix_generator = u_views.get("cutmix_generator") or u_views.get("generator")
            explicit_boxes = u_views.get("boxes") if isinstance(u_views, Mapping) else None
            boxes: list[CutMixBox] = []
            for view_index in range(2 if int(_required(_section(config, "augmentation"), "n_strong_views", "augmentation")) == 2 else 1):
                if isinstance(explicit_boxes, Sequence) and not isinstance(explicit_boxes, (str, bytes)) and len(explicit_boxes) > view_index:
                    boxes.append(tuple(int(v) for v in explicit_boxes[view_index]))  # type: ignore[assignment]
                else:
                    boxes.append(_sample_box(ua_view if isinstance(ua_view, Mapping) else {}, ub_view if isinstance(ub_view, Mapping) else {}, ua["valid"], _crop_to_anchor(ub["valid"], ua["valid"].shape[-3:], pad_value=0), generator=cutmix_generator, cutmix_cfg=cutmix_args, update=update, index=view_index))
            partner_shape = tuple(int(v) for v in ua["weak"].shape[-3:])
            # Normalize every B packet field to A's native canvas before the
            # packet mixer.  This makes the crop/pad contract explicit (and
            # keeps padded B cells valid=0/weight=0) instead of relying on a
            # hidden shape adjustment inside mix_packet.
            packet_b_anchor = {
                "p_W": _crop_to_anchor(packet_b["p_W"], partner_shape, pad_value=0.0),
                "y": _crop_to_anchor(packet_b["y"], partner_shape, pad_value=0),
                "w": _crop_to_anchor(packet_b["w"], partner_shape, pad_value=0.0),
                "valid": _crop_to_anchor(packet_b["valid"], partner_shape, pad_value=0).bool(),
            }
            def _mixed_packet(box: CutMixBox) -> Mapping[str, Tensor]:
                return mix_packet(packet_a["p_W"], packet_a["y"], packet_a["w"], packet_a["valid"], box,
                                  partner_p_W=packet_b_anchor["p_W"], partner_y=packet_b_anchor["y"],
                                  partner_w=packet_b_anchor["w"], partner_valid=packet_b_anchor["valid"])
            # pT/pS/sdf are mixed with exactly the same box and the same B
            # center-crop/pad as the packet.
            pta = p_teacher_a; ptb = _crop_to_anchor(p_teacher_b, partner_shape)
            psa = p_student_a; psb = _crop_to_anchor(p_student_b, partner_shape)
            sdf_a = _sdf_output(teacher_a); sdf_b = None if teacher_b is None else _sdf_output(teacher_b)
            sdf_b_anchor = None if sdf_b is None else _crop_to_anchor(sdf_b, partner_shape)
            strong_a1 = ua["strong"]
            strong_b1 = _crop_to_anchor(ub["strong"], partner_shape)
            mix_losses: list[Tensor] = []
            for view_index, box in enumerate(boxes):
                packet = _mixed_packet(box)
                p_t_mix = mix_tensor(pta, ptb, box)
                p_s_mix = mix_tensor(psa, psb, box)
                sdf_mix = None if sdf_a is None or sdf_b_anchor is None else mix_tensor(sdf_a, sdf_b_anchor, box)
                if view_index == 0:
                    image_a = mix_tensor(strong_a1, strong_b1, box)
                else:
                    strong_a2 = _strong2(ua_view, ua)
                    strong_b2 = _crop_to_anchor(_strong2(ub_view, ub), partner_shape)
                    image_a = mix_tensor(strong_a2, strong_b2, box)
                out_stream = student(image_a, ua["geometry"])
                mix_losses.append(_pg_loss(_student_logits(out_stream), packet, p_t_mix, p_s_mix, thresholds, sdf_mix))
            s1_loss = mix_losses[0]
            if len(mix_losses) == 2:
                s2_loss = mix_losses[1]
            else:
                s2_loss = s1_loss * 0.0
            # FP is deliberately weak-A (not strong-A), with explicit dropout
            # generator and the unmixed A packet.
            fp_dropout = float(_required(_section(config, "student"), "fp_skip_dropout", "student"))
            out_fp = student(ua["weak"], ua["geometry"], skip_dropout=fp_dropout, dropout_generator=fp_generator)
            fp_packet = packet_a
            fp_loss = _pg_loss(_student_logits(out_fp), fp_packet, p_teacher_a, p_student_a, thresholds, sdf_a)
            stream_cfg = _mapping(_required(pg_cfg, "stream_weights", "pg"), "pg.stream_weights")
            n_views = int(_required(_section(config, "augmentation"), "n_strong_views", "augmentation"))
            if n_views == 1:
                w_s1, w_s2, w_fp = 0.5, 0.0, 0.5
            elif n_views == 2:
                w_s1 = float(_required(stream_cfg, "s1", "pg.stream_weights")); w_s2 = float(_required(stream_cfg, "s2", "pg.stream_weights")); w_fp = float(_required(stream_cfg, "fp", "pg.stream_weights"))
            else:
                raise ValueError("augmentation.n_strong_views must be 1 or 2")
            loss_pg = w_s1 * s1_loss + w_s2 * s2_loss + w_fp * fp_loss
            def _class_coverage(labels: Tensor, valid: Tensor) -> tuple[float, ...]:
                support = valid.bool()
                denominator = support.sum().clamp_min(1).to(dtype=torch.float32)
                return tuple(float((((labels == class_id) & support).sum() / denominator).detach()) for class_id in range(4))

            disagreement_a = ((p_teacher_a.argmax(dim=1) != p_fusion_a.argmax(dim=1)) & ua["valid"]).float().sum() / ua["valid"].sum().clamp_min(1)
            disagreement_b = ((p_teacher_b.argmax(dim=1) != p_fusion_b.argmax(dim=1)) & ub["valid"]).float().sum() / ub["valid"].sum().clamp_min(1)
            band_coverage_a = None
            band_coverage_b = None
            if sdf_a is not None:
                band_coverage_a = float((reliability.boundary_band(sdf_a) & ua["valid"]).float().sum().div(ua["valid"].sum().clamp_min(1)).detach())
            if sdf_b is not None:
                band_coverage_b = float((reliability.boundary_band(sdf_b) & ub["valid"]).float().sum().div(ub["valid"].sum().clamp_min(1)).detach())
            diagnostics.update({
                "lambda_u": float(pg_lambda), "loss_pg_s1": s1_loss.detach(), "loss_pg_s2": s2_loss.detach(),
                "loss_pg_fp": fp_loss.detach(), "agreement_a": reliability.voxel_weights(p_fusion_a, p_teacher_a, p_student_a, ua["valid"], thresholds, d_T=sdf_a).detach().mean(),
                "cutmix_boxes": tuple(tuple(int(v) for v in box) for box in boxes),
                "coverage_per_class_a": _class_coverage(y_a, ua["valid"]),
                "coverage_per_class_b": _class_coverage(y_b, ub["valid"]),
                "boundary_band_coverage_a": band_coverage_a,
                "boundary_band_coverage_b": band_coverage_b,
                "teacher_fusion_disagreement_a": disagreement_a.detach(),
                "teacher_fusion_disagreement_b": disagreement_b.detach(),
            })
            diagnostics.update({
                "pW_target_a": p_fusion_a.argmax(dim=1).detach(),
                "teacher_target_a": p_teacher_a.argmax(dim=1).detach(),
                "pW_confidence_a": p_fusion_a.max(dim=1).values.detach(),
            })
            # Reliability weights use the pre-update threshold; only now update
            # the combined A/B observation, including distinct native shapes.
            _update_threshold(threshold, p_fusion_a, ua["valid"], p_fusion_b, ub["valid"])
        if cb_active_now:
            if codebook is None:
                raise RuntimeError("CBKD is enabled but no codebook was supplied")
            cbkd_weights = _mapping(_required(cbkd_cfg, "weights", "cbkd"), "cbkd.weights")
            if cbkd_cfg.get("variant") == "cbkd_sa":
                cbkd_value, sa_terms = _cbkd_sa_loss(student, teacher, codebook, teacher_l, teacher_a, out_l_student, clean_a, lv["target"], lv["geometry"], ua["geometry"], lv["valid"], ua["valid"], cbkd_weights)
                loss_cbkd = cbkd_value
                diagnostics.update({
                    "lambda_cb": _config_lambda_cb(config, update), "loss_pm": sa_terms["pm"].detach(), "loss_atom": sa_terms["atom"].detach(),
                    "loss_commit": sa_terms["commit"].detach(), "m_T_mean": sa_terms["m_T_mean"].detach(),
                    "student_code_perplexity": float(sa_terms["student_code_perplexity"]), "student_code_dead": int(sa_terms["student_code_dead"]),
                    "code_perplexity": float(sa_terms["student_code_perplexity"]), "code_dead": int(sa_terms["student_code_dead"]),
                    "loss_pm_l": sa_terms["pm_l"].detach(), "loss_pm_u": sa_terms["pm_a"].detach(),
                })
                cbkd_terms = {"assign": sa_terms["pm"], "commit": sa_terms["commit"], "qtask": sa_terms["atom"]}
            else:
                cbkd_value, cbkd_terms = _cbkd_loss(student, teacher, codebook, teacher_l, teacher_a, out_l_student, clean_a, lv["target"], lv["geometry"], ua["geometry"], lv["valid"], ua["valid"], cbkd_weights)
                loss_cbkd = cbkd_value
            code_packet = clean_a.get("code_branch")
            if isinstance(code_packet, Mapping) and isinstance(code_packet.get("assignments"), Tensor):
                code_metrics = cb.usage_metrics(
                    code_packet["assignments"].detach(),
                    int(code_packet["rho"].shape[1]),
                    code_packet.get("valid_mask"),
                )
                diagnostics.update({
                    "code_perplexity": float(code_metrics["perplexity"]),
                    "code_dead": int(code_metrics["dead_code_count"]),
                    "code_valid_count": int(code_metrics["valid_count"]),
                })
            diagnostics.update({"lambda_cb": _config_lambda_cb(config, update), "loss_cbkd_assign": cbkd_terms["assign"].detach(), "loss_cbkd_commit": cbkd_terms["commit"].detach(), "loss_cbkd_qtask": cbkd_terms["qtask"].detach()})

    lambda_u = float(diagnostics["lambda_u"])
    lambda_cb = float(diagnostics["lambda_cb"])
    total = loss_sup + lambda_u * loss_pg + lambda_cb * loss_cbkd
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("PG student loss is nonfinite")
    # Ensure required graph-bearing tensors are returned, while diagnostics are
    # explicitly detached (including nested mappings/tuples).
    diagnostics = _detach_tree(diagnostics)
    return {"loss": total, "loss_sup": loss_sup, "loss_pg": loss_pg, "loss_cbkd": loss_cbkd, "diagnostics": diagnostics}


def _supervised_loss(logits: Tensor, target: Tensor) -> Tensor:
    from self_audit.losses import supervised_loss
    return supervised_loss(logits, target)


def _detach_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach()
    if isinstance(value, Mapping):
        return {key: _detach_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    return value


__all__ = ["student_training_step"]
