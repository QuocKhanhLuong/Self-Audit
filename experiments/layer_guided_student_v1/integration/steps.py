"""Isolated opt-in primitives; no full trainer, AdamW guard, or label admission."""

import math

from layer_upgrade import FrozenImageEnergy, guarded_sgd_step, refine_distribution

from .artifacts import canonical_hash
from .geometry import require_digest


def witness_plan(train_patients, witnesses):
    """Lock supplied class/source witnesses covering every selected TRAIN patient."""
    if not train_patients or len(train_patients) != len(set(train_patients)):
        raise ValueError("unique selected TRAIN patients required")
    if not witnesses or {w["patient_id"] for w in witnesses} != set(train_patients):
        raise ValueError("witnesses must cover exactly all selected TRAIN patients")
    for row in witnesses:
        if row.get("split") != "train" or row.get("class_name") not in {
            "BG",
            "RV",
            "MYO",
            "LV",
        }:
            raise ValueError("fixed TRAIN class/source strata required")
        require_digest(row.get("source_manifest_sha256"))
        if not row.get("sample_id"):
            raise ValueError("witness sample identity required")
    plan = {"train_patients": sorted(train_patients), "witnesses": list(witnesses)}
    return {**plan, "sha256": canonical_hash(plan)}


def guarded_decoder_step(
    model, objective, guard_losses, *, plan, step_size, blocks=("refine.0",)
):
    """One verified additive substep. Callers must supply frozen pure closures.

    The plan identifies witness batches; closures are responsible for loading
    exactly those samples. This API is not a claim about future batches or any
    subsequent unrestricted optimizer step.
    """
    expected = witness_plan(plan["train_patients"], plan["witnesses"])
    if expected != plan:
        raise ValueError("witness plan changed")
    if any(module.training for module in model.modules()):
        raise ValueError("guarded closures require eval mode, no dropout updates")
    allowed = {f"refine.{i}" for i in range(4)}
    if not blocks or not set(blocks) <= allowed or len(set(blocks)) != len(blocks):
        raise ValueError("select explicit decoder refinement blocks only")
    selected = [
        p
        for name, p in model.named_parameters()
        if any(name.startswith(b + ".") for b in blocks)
    ]
    if not selected or any(not p.requires_grad for p in selected):
        raise ValueError("selected decoder block must be trainable")
    old = [p.detach().clone() for p in selected]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    selected_ids = {id(p) for p in selected}
    try:
        for p, _ in flags:
            if id(p) not in selected_ids:
                p.requires_grad_(False)
        receipt = guarded_sgd_step(
            selected, objective, guard_losses, step_size=step_size
        )
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
    displacement = math.sqrt(
        sum(
            float((p.detach() - v).double().square().sum())
            for p, v in zip(selected, old)
        )
    )
    return {
        **receipt,
        "actual_displacement_norm": displacement,
        "blocks": list(blocks),
        "witness_plan_sha256": plan["sha256"],
        "optimizer": "additive_sgd_no_state_no_decay",
        "projection_alone_certifies_post_cast_feasibility": False,
        "scope": "this_substep_fixed_train_witness_losses_only",
    }


def image_proposal(logits, energy: FrozenImageEnergy, *, steps=3):
    """Own the frozen energy buffers; return a detached, unadmitted proposal."""
    if type(steps) is not int or steps < 0:
        raise ValueError("nonnegative integer iteration count required")
    frozen = FrozenImageEnergy(
        energy.unary.clone(),
        energy.edges.clone(),
        energy.edge_weight.clone(),
        energy.anchor.clone(),
        energy.anchor_weight.clone(),
        energy.lam,
        energy.beta,
    )
    proposal, receipt = refine_distribution(logits, frozen, steps=steps)
    return proposal, {
        **receipt,
        "semantic_admission": False,
        "source_correctness": "UNVERIFIED",
        "energy_buffers": "owned_detached_clones",
    }
