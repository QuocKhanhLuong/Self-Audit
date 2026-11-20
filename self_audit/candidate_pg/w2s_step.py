"""W2S-Match student update (``pg.target_mode`` in ``{"self", "w2s_arbiter"}``).

One optimizer step worth of student math.  The labelled loss is student-only
(no teacher-feature fusion: inference and training use the same path).  The
unlabelled loss is weighted CE + weighted Dice on a packet ``(y, w, valid)``
whose construction is the only difference between arms:

* ``self``         : y = argmax of the student's own clean-weak prediction,
                     w = 1 where its max-prob >= tau (UniMatch self-training).
* ``w2s_arbiter``  : (y, w) from :mod:`w2s_arbiter` (teacher W vs student).

With ``cbkd.variant == "cbkd_sa_gated"`` the gated shape-atom terms
``L_pm`` (U) and ``L_atom`` (L_K + commitment) are added.  Everything coming
from the teacher or the codebook is detached; the codebook never gets
gradient and ``P_S`` is the only trainable codebook-side module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from . import losses, shape_atoms as cbkd_sa, w2s_arbiter as arb
from .views import CutMixBox, mix_tensor

TARGET_MODES = ("fusion", "self", "w2s_arbiter", "teacher")
CBKD_VARIANT = "cbkd_sa_gated"


def w2s_params(config: Mapping[str, Any]) -> dict[str, Any]:
    pg = config["pg"]
    p = dict(pg.get("w2s", {}))
    return {
        "tau": float(p.get("tau", 0.95)), "tau_v": float(p.get("tau_v", 0.99)),
        "alpha_max": float(p.get("alpha_max", 0.5)),
        "alpha_start": float(p.get("alpha_start_fraction", 0.2)),
        "alpha_end": float(p.get("alpha_end_fraction", 0.5)),
    }


def cbkd_sa_active(config: Mapping[str, Any]) -> bool:
    c = config.get("cbkd", {})
    return bool(c.get("enabled")) and c.get("variant") == CBKD_VARIANT


def _packet_loss(logits: Tensor, y: Tensor, w: Tensor, valid: Tensor) -> Tensor:
    probs = logits.softmax(dim=1)
    return losses.weighted_ce(logits, y, w, valid) + losses.weighted_dice(probs, y, w, valid)


def _lambda_u(config: Mapping[str, Any], update: int) -> float:
    from .training_step import _config_lambda_u

    return _config_lambda_u(config, update)


def _boxes(config: Mapping[str, Any], ua_view: Any, ub_view: Any, ua: Mapping[str, Any], ub: Mapping[str, Any],
           u_views: Any, update: int) -> list[CutMixBox]:
    from .training_step import _crop_to_anchor, _mapping, _required, _sample_box, _section

    cut = _mapping(_required(_section(config, "augmentation"), "cutmix", "augmentation"), "augmentation.cutmix")
    if _required(cut, "full_depth", "augmentation.cutmix") is not True:
        raise ValueError("PG CutMix must be full-depth")
    args = {"area_fraction": cut["area_fraction"], "align_px": int(cut["align_px"]), "full_depth": True}
    n_views = 2 if int(config["augmentation"]["n_strong_views"]) == 2 else 1
    explicit = u_views.get("boxes") if isinstance(u_views, Mapping) else None
    generator = None
    if isinstance(u_views, Mapping):
        generator = u_views.get("cutmix_generator") or u_views.get("generator")
    boxes: list[CutMixBox] = []
    for i in range(n_views):
        if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)) and len(explicit) > i:
            boxes.append(tuple(int(v) for v in explicit[i]))  # type: ignore[arg-type]
        else:
            boxes.append(_sample_box(ua_view if isinstance(ua_view, Mapping) else {}, ub_view if isinstance(ub_view, Mapping) else {},
                                     ua["valid"], _crop_to_anchor(ub["valid"], ua["valid"].shape[-3:], pad_value=0),
                                     generator=generator, cutmix_cfg=args, update=update, index=i))
    return boxes


def w2s_student_training_step(
    config: Mapping[str, Any], student: nn.Module, teacher: nn.Module | None, codebook: nn.Module | None,
    labeled: Any, u_a: Any, u_b: Any, update: int, *, labeled_views: Any, u_views: Any,
    keep_generator: torch.Generator | None = None, fp_generator: torch.Generator,
) -> dict[str, Any]:
    from .training_step import (
        _aligned_teacher_level3, _coerce_view, _crop_to_anchor, _detach_tree, _frame_image, _frame_target,
        _split_unlabelled_views, _strong2, _student_logits, _supervised_loss, _teacher_output,
    )

    mode = config["pg"]["target_mode"]
    if mode not in ("self", "w2s_arbiter", "teacher"):
        raise ValueError(f"unsupported W2S target_mode {mode!r}")
    if isinstance(update, bool) or int(update) != update or update < 0:
        raise ValueError("update must be a nonnegative integer")
    update = int(update)
    use_teacher = mode in ("w2s_arbiter", "teacher")
    if use_teacher and not isinstance(teacher, nn.Module):
        raise TypeError("w2s_arbiter requires the frozen teacher W")
    cb_on_cfg = cbkd_sa_active(config)
    if cb_on_cfg and use_teacher is False:
        raise ValueError("cbkd_sa_gated requires target_mode='w2s_arbiter'")
    params = w2s_params(config)
    total = int(config["training"]["updates"])
    device = next(student.parameters()).device
    labeled_image = _frame_image(labeled, device=device)
    labeled_target = _frame_target(labeled, device=device)
    lv = _coerce_view(labeled_views, labeled, target=labeled_target, device=device)
    labeled_target = lv["target"]

    warmup_u = int(config["pg"]["lambda_u"]["warmup_updates_labeled_only"])
    u_active = update >= warmup_u
    cb_onset = int(config["cbkd"].get("warmup_updates_labeled_only", 0)) if cb_on_cfg else 0
    cb_now = cb_on_cfg and update >= cb_onset
    lam_cb = cbkd_sa.lambda_cb(update, lam=float(config["cbkd"]["lambda_cb"]), onset=cb_onset,
                               ramp=int(config["cbkd"].get("ramp_updates", 400))) if cb_on_cfg else 0.0

    out_l = student(lv["strong"], lv["geometry"], return_stage3=cb_now, return_code=cb_now)
    loss_sup = _supervised_loss(_student_logits(out_l), labeled_target)
    zero = loss_sup * 0.0
    loss_u, loss_pm, loss_atom = zero, zero, zero
    alpha = arb.alpha_schedule(update, total, start_fraction=params["alpha_start"], end_fraction=params["alpha_end"],
                               alpha_max=params["alpha_max"]) if mode == "w2s_arbiter" else 0.0
    lam_u = _lambda_u(config, update) if u_active else 0.0
    diagnostics: dict[str, Any] = {
        "lambda_u": lam_u, "lambda_cb": lam_cb if cb_now else 0.0, "alpha": alpha, "tau": params["tau"],
        "tau_v": params["tau_v"], "target_mode": mode, "loss_sup": loss_sup.detach(),
    }
    terms: dict[str, Tensor] = {}

    if u_active:
        ua_view, ub_view = _split_unlabelled_views(u_views, u_a, u_b)
        ua = _coerce_view(ua_view, u_a, device=device)
        ub = _coerce_view(ub_view, u_b, device=device)
        anchor = tuple(int(v) for v in ua["weak"].shape[-3:])
        valid_a = ua["valid"]
        valid_b = _crop_to_anchor(ub["valid"], anchor, pad_value=0).bool()

        # Clean weak predictions.  A keeps its graph only when L_pm needs it.
        if cb_now:
            clean_a = student(ua["weak"], ua["geometry"], return_stage3=True, return_code=True)
        else:
            with torch.no_grad():
                clean_a = student(ua["weak"], ua["geometry"])
        with torch.no_grad():
            clean_b = student(ub["weak"], ub["geometry"])
        logits_clean_a = _student_logits(clean_a)
        p_s_a = logits_clean_a.detach().softmax(1)
        p_s_b = _crop_to_anchor(_student_logits(clean_b).softmax(1), anchor)

        p_t_a = p_t_b = None
        teacher_a = None
        if use_teacher:
            with torch.no_grad():
                teacher_a = _teacher_output(teacher, ua["weak"], ua["geometry"])
                teacher_b = _teacher_output(teacher, ub["weak"], ub["geometry"])
                p_t_a = teacher_a["region_logits"].softmax(1).detach()
                p_t_b = _crop_to_anchor(teacher_b["region_logits"].softmax(1).detach(), anchor)

        boxes = _boxes(config, ua_view, ub_view, ua, ub, u_views, update)
        images = []
        for i, box in enumerate(boxes):
            if i == 0:
                images.append(mix_tensor(ua["strong"], _crop_to_anchor(ub["strong"], anchor), box))
            else:
                images.append(mix_tensor(_strong2(ua_view, ua), _crop_to_anchor(_strong2(ub_view, ub), anchor), box))
        fp_dropout = float(config["student"]["fp_skip_dropout"])
        logits_views = [_student_logits(student(img, ua["geometry"])) for img in images]
        logits_fp = _student_logits(student(ua["weak"], ua["geometry"], skip_dropout=fp_dropout, dropout_generator=fp_generator))
        all_logits = logits_views + [logits_fp]

        with torch.no_grad():
            valid_views = [mix_tensor(valid_a, valid_b, box) for box in boxes] + [valid_a]
            y_s_a, y_s_b = p_s_a.argmax(1), p_s_b.argmax(1)
            if mode == "self":
                packets = []
                for i, box in enumerate(boxes):
                    p_mix = mix_tensor(p_s_a, p_s_b, box)
                    c, y = p_mix.max(1)
                    packets.append((y, (c >= params["tau"]).float(), torch.zeros_like(y), torch.zeros_like(y, dtype=torch.bool)))
                c, y = p_s_a.max(1)
                packets.append((y, (c >= params["tau"]).float(), torch.zeros_like(y), torch.zeros_like(y, dtype=torch.bool)))
            elif mode == "teacher":  # naive KD: teacher pseudo-labels at the fixed 0.95, no arbiter, no stability
                packets = []
                pairs_t = [mix_tensor(p_t_a, p_t_b, box) for box in boxes] + [p_t_a]
                for p_mix in pairs_t:
                    c, y = p_mix.max(1)
                    packets.append((y, (c >= params["tau"]).float(), torch.zeros_like(y), torch.zeros_like(y, dtype=torch.bool)))
            else:
                _, _, stable_views = arb.source_aligned_stability(
                    y_s_a, y_s_b, [l.detach().argmax(1) for l in logits_views], logits_fp.detach().argmax(1), boxes)
                packets = []
                pairs = [(mix_tensor(p_t_a, p_t_b, box), mix_tensor(p_s_a, p_s_b, box)) for box in boxes] + [(p_t_a, p_s_a)]
                for i, (pt, ps) in enumerate(pairs):
                    y_v, w_v, r_v = arb.arbitrate(pt, ps, stable_views[i], valid_views[i], alpha, tau=params["tau"], tau_v=params["tau_v"])
                    packets.append((y_v, w_v, r_v, arb.fg_disagreement(pt, ps, valid_views[i])))

        stream = config["pg"]["stream_weights"]
        n_views = len(boxes)
        weights = (0.5, 0.0, 0.5) if n_views == 1 else (float(stream["s1"]), float(stream["s2"]), float(stream["fp"]))
        stream_losses = [_packet_loss(all_logits[i], packets[i][0], packets[i][1], valid_views[i]) for i in range(len(all_logits))]
        loss_u = weights[0] * stream_losses[0]
        if n_views == 2:
            loss_u = loss_u + weights[1] * stream_losses[1]
        loss_u = loss_u + weights[2] * stream_losses[-1]
        diagnostics.update({
            "loss_u_s1": stream_losses[0].detach(), "loss_u_fp": stream_losses[-1].detach(),
            "loss_u": loss_u.detach(),
        })
        if mode == "w2s_arbiter":
            diagnostics.update(_rule_diagnostics(packets, valid_views, p_t_a, p_s_a))

        if cb_now:
            if not isinstance(codebook, cbkd_sa.ShapeAtomCodebook):
                raise RuntimeError("cbkd_sa_gated requires a frozen ShapeAtomCodebook")
            if any(p.requires_grad for p in codebook.parameters()):
                raise RuntimeError("CBKD-SA codebook must be frozen")
            branch = getattr(student, "code_branch", None)
            if not isinstance(branch, cbkd_sa.StudentAtomBranch):
                raise TypeError("cbkd_sa_gated requires PGStudent.code_branch = StudentAtomBranch")
            stage3_shape = clean_a["stage3"].shape[-3:]
            with torch.no_grad():
                feat = _aligned_teacher_level3(teacher, student, teacher_a, ua["geometry"], stage3_shape)
                atoms = cbkd_sa.teacher_atoms(codebook, feat, anchor)
            rule_fp = packets[-1][2]
            gate = ((rule_fp == 1) | (rule_fp == 2) | (rule_fp == 4)).float()
            loss_pm = cbkd_sa.pm_loss(logits_clean_a.softmax(1), atoms["q"], atoms["margin"], gate, valid_a)
            code_l, code_a = out_l["code_branch"], clean_a["code_branch"]
            sdf_target = _gt_sdf(labeled_target, lv["geometry"])
            atom, atom_terms = cbkd_sa.atom_loss(code_l, labeled_target, sdf_target, lv["valid"],
                                                 tuple(int(v) for v in labeled_target.shape[-3:]), codebook.patch)
            commit = cbkd_sa.commitment(code_l["z"], code_l["e"], code_l.get("valid_mask")) \
                + cbkd_sa.commitment(code_a["z"], code_a["e"], code_a.get("valid_mask"))
            loss_atom = atom + float(config["cbkd"]["weights"].get("commit", 0.25)) * commit
            usage = cbkd_sa.code_usage(torch.cat((code_l["idx"].detach(), code_a["idx"].detach())),
                                       None if code_l.get("valid_mask") is None else torch.cat((code_l["valid_mask"], code_a["valid_mask"])),
                                       codebook.size)
            gated = (gate * valid_a.float()).sum()
            diagnostics.update({
                "loss_pm": loss_pm.detach(), "loss_atom": loss_atom.detach(), "loss_commit": commit.detach(),
                "code_perplexity": usage["perplexity"], "code_dead": usage["dead"],
                "gate_fraction": float(gated / valid_a.float().sum().clamp_min(1)),
                "mean_m_T_gated": float((atoms["margin"] * gate * valid_a.float()).sum() / gated.clamp_min(1)),
                **{k: v for k, v in atom_terms.items()},
            })

    cb_term = loss_pm + loss_atom
    total_loss = loss_sup + lam_u * loss_u + (lam_cb if cb_now else 0.0) * cb_term
    if not bool(torch.isfinite(total_loss)):
        raise FloatingPointError("W2S student loss is nonfinite")
    terms.update({"sup": loss_sup, "u": loss_u, "cbkd": cb_term})
    return {"loss": total_loss, "loss_sup": loss_sup, "loss_pg": loss_u, "loss_cbkd": cb_term,
            "terms": terms, "diagnostics": _detach_tree(diagnostics)}


def _gt_sdf(target: Tensor, geometry: Mapping[str, Any]) -> Tensor | None:
    """Foreground-3 SDF of the GT (same target the teacher stage uses); ``None`` without spacing."""

    if "spacing_zyx" not in geometry:
        return None
    from .teacher_training import _sdf_targets4

    return _sdf_targets4(target, geometry)[:, 1:].detach()


def _rule_diagnostics(packets: Sequence[tuple[Tensor, Tensor, Tensor, Tensor]], valid_views: Sequence[Tensor],
                      p_t_a: Tensor, p_s_a: Tensor) -> dict[str, Any]:
    """Pooled per-rule voxel fractions, override share, FG coverage (all detached)."""

    y = torch.cat([p[0].reshape(-1) for p in packets])
    rule = torch.cat([p[2].reshape(-1) for p in packets])
    w = torch.cat([p[1].reshape(-1) for p in packets])
    valid = torch.cat([v.reshape(-1) for v in valid_views]).bool()
    stats = arb.rule_statistics(rule, y, valid)
    # Foreground disagreement voxels: teacher/student argmax differ and either is FG.
    dis = torch.cat([p[3].reshape(-1) for p in packets]).bool() & valid
    fg_dis = dis.sum().clamp_min(1)
    override = ((rule == 3) & dis).sum()
    supervised = valid & (w > 0)
    coverage = ((y > 0) & supervised).sum() / supervised.sum().clamp_min(1)
    return {
        "rule_fraction": stats["overall"], "rule_fraction_per_class": stats["per_class"],
        "override_share_fg_disagree": float(override / fg_dis),
        "fg_coverage": float(coverage), "weight_mean": float(w[valid].mean()) if bool(valid.any()) else 0.0,
    }


__all__ = ["CBKD_VARIANT", "TARGET_MODES", "cbkd_sa_active", "w2s_params", "w2s_student_training_step"]
