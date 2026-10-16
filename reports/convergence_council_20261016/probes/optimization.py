"""CPU manufactured-tensor/source-count audit; never opens patient arrays or weights.

Invoke with the council's already-installed CPU torch interpreter. Results concern
gradient wiring, reduction algebra and published text, not learned anatomy.
"""
from pathlib import Path
import hashlib
import json
import math
import sys

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "reports/stable_guidance_main513ddda_b12_20261016"
SOURCE = BASE / "completed_run/source/src"
DEST = Path(__file__).resolve().parents[1] / "checks/optimization.json"
sys.path.insert(0, str(SOURCE))
from self_audit_guidance.student import GuidanceStudent, guidance_loss


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def l2(grads):
    return math.sqrt(sum(float(x.detach().double().square().sum()) for x in grads if x is not None))


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    recipe = json.loads((BASE / "completed_run/run/RECIPE.json").read_text())
    source_file = SOURCE / "self_audit_guidance/student.py"
    assert sha(source_file) == recipe["source_sha256"]["student.py"]
    model = GuidanceStudent(64, 32).cpu().train()
    x = torch.randn(2, 3, 20, 24)
    teacher = torch.randn(2, 64, 5, 6, requires_grad=True)
    q = torch.tensor([.1, .7, .1, .1])[None, :, None, None].expand(2, 4, 5, 6).clone().requires_grad_()
    w = torch.ones(2, 5, 6, requires_grad=True)
    params = list(model.named_parameters())
    out = model(x)
    fl, sl, active = guidance_loss(out, teacher, q, w)
    assert active
    gf = torch.autograd.grad(fl, [p for _, p in params], retain_graph=True, allow_unused=True)
    gs = torch.autograd.grad(sl, [p for _, p in params], retain_graph=True, allow_unused=True)
    groups = {}
    for prefix in ("enc0", "enc1", "enc2", "project", "decode", "head"):
        ids = [i for i, (n, _) in enumerate(params) if n.startswith(prefix + ".")]
        a, b = [gf[i] for i in ids], [gs[i] for i in ids]
        nf, ns = l2(a), l2(b)
        dot = sum(float(u.detach().double().mul(v.detach().double()).sum()) for u, v in zip(a, b) if u is not None and v is not None)
        groups[prefix] = {"feature_grad_norm": nf, "semantic_grad_norm": ns,
                          "cosine": dot / (nf * ns) if nf and ns else None}
    assert all(groups[g]["feature_grad_norm"] > 0 and groups[g]["semantic_grad_norm"] > 0 for g in ("enc0", "enc1", "enc2"))
    assert groups["project"]["feature_grad_norm"] > 0 and groups["project"]["semantic_grad_norm"] == 0
    assert all(groups[g]["feature_grad_norm"] == 0 and groups[g]["semantic_grad_norm"] > 0 for g in ("decode", "head"))
    (fl + sl).backward()
    assert teacher.grad is None and q.grad is None and w.grad is None

    # Follow the executed loop's feature-only update, including its grad=None rule.
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    before = {n: p.detach().clone() for n, p in params}
    qbg = torch.tensor([.7, .1, .1, .1])[None, :, None, None].expand_as(q)
    pred_before = model(x)["logits"].detach().clone()
    fl, sl, active = guidance_loss(model(x), teacher, qbg, w)
    assert not active
    opt.zero_grad(set_to_none=True)
    (fl + sl).backward()
    for p in model.semantic_parameters():
        p.grad = None
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    opt.step()
    changes = {g: max(float((p.detach() - before[n]).abs().max()) for n, p in params if n.startswith(g + ".")) for g in groups}
    output_change = float((model(x)["logits"].detach() - pred_before).abs().max())
    assert changes["decode"] == changes["head"] == 0
    assert all(changes[g] > 0 for g in ("enc0", "enc1", "enc2", "project"))
    assert output_change > 0

    # Same three examples; regrouping changes the exact semantic objective.
    bias = torch.zeros(4, requires_grad=True)
    toy_q = torch.tensor([[.1, .7, .1, .1], [.7, .1, .1, .1], [.7, .1, .1, .1]])[:, :, None, None]
    toy_w = torch.ones(3, 1, 1)
    toy_teacher = torch.ones(3, 64, 1, 1)
    def semantic(indices):
        logits = bias[None, :, None, None].expand(len(indices), 4, 1, 1)
        _, sem, on = guidance_loss({"features": toy_teacher[indices], "logits": logits}, toy_teacher[indices], toy_q[indices], toy_w[indices])
        return sem, on
    separate = [semantic([i]) for i in range(3)]
    separate_loss = sum(s for s, _ in separate) / 3
    separate_grad = torch.autograd.grad(separate_loss, bias)[0]
    together_loss, together_active = semantic([0, 1, 2])
    together_grad = torch.autograd.grad(together_loss, bias)[0]
    assert not torch.allclose(separate_grad, together_grad)

    # Output-space counterexample: bilinear supervision can miss native detail.
    raw0 = torch.zeros(1, 4, 4, 4)
    raw1 = raw0.clone()
    raw1[:, 1] = torch.tensor([[5., -5., 5., -5.], [-5., 5., -5., 5.]] * 2)
    coarse0 = F.interpolate(raw0, (2, 2), mode="bilinear", align_corners=False)
    coarse1 = F.interpolate(raw1, (2, 2), mode="bilinear", align_corners=False)
    assert torch.equal(coarse0, coarse1)
    def coverage(logits):
        top = logits.softmax(1).topk(2, 1).values
        return float(((top[:, 0] >= .7) & (top[:, 0] - top[:, 1] >= .2)).float().mean())

    bm = json.loads((BASE / "completed_run/run/bank/FROZEN.json").read_text())
    records = [bm["image_records"][p] for p in bm["patients"]]
    sizes = [r["shape_xyzt"][2] * r["shape_xyzt"][3] for r in records]
    steps = {str(b): sum(math.ceil(n / b) for n in sizes) for b in (8, 12)}
    tails = {str(b): {"partial_batches": sum(n % b != 0 for n in sizes), "partial_samples": sum(n % b for n in sizes)} for b in (8, 12)}
    history = json.loads((BASE / "completed_run/run/student/training.json").read_text())
    summary = json.loads((BASE / "summary.json").read_text())
    assert all(r["steps"] == steps["12"] for r in history)
    result = {
        "status": "PASS_CPU_MANUFACTURED_AND_PUBLISHED_TEXT_ONLY",
        "evidence_commit": "92a60744efef85558d7c0226c84b7c49d47217ec",
        "executed_source_commit": "513ddda8cec3a166f13cfe68fe6379cab014a1a7",
        "student_source_sha256": sha(source_file),
        "patient_arrays_read": False, "checkpoints_loaded": False,
        "teacher_instantiated": False, "cuda_used": False, "patient_training": False,
        "manufactured_optimizer_steps": 1, "torch_version": torch.__version__, "seed": 42,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameters_by_top_level": {g: sum(p.numel() for n, p in params if n.startswith(g + ".")) for g in groups},
        "manufactured_gradient_groups": groups,
        "frozen_targets_no_gradient": True,
        "feature_only_step": {"parameter_max_change": changes, "logit_max_change": output_change, "preclip_norm": float(norm)},
        "batch_regrouping_counterexample": {
            "separate_active": [on for _, on in separate], "together_active": together_active,
            "separate_all_step_mean_ce": float(separate_loss.detach()), "together_ce": float(together_loss.detach()),
            "separate_bias_gradient": separate_grad.tolist(), "together_bias_gradient": together_grad.tolist(),
            "interpretation": "Same example set, different foreground gate and batch normalization. Not an observed B8/B12 causal effect."
        },
        "native_coarse_nullspace_counterexample": {
            "coarse_logit_max_difference": float((coarse0-coarse1).abs().max()),
            "uniform_native_coverage": coverage(raw0), "perturbed_native_coverage": coverage(raw1),
            "scope": "Manufactured output tensors, not proof that trained model realizes this perturbation."
        },
        "published_train_inventory": {
            "patients": len(records), "samples_per_epoch": sum(sizes), "steps_per_epoch": steps,
            "hypothetical_50_epoch_b8_updates": 50 * steps["8"], "observed_b12_updates": sum(r["steps"] for r in history),
            "partial_batches": tails,
            "native_height_range": [min(r["shape_xyzt"][1] for r in records), max(r["shape_xyzt"][1] for r in records)],
            "native_width_range": [min(r["shape_xyzt"][0] for r in records), max(r["shape_xyzt"][0] for r in records)],
            "reported_epoch_elapsed_seconds_sum": sum(r["elapsed_seconds"] for r in history),
            "late_online_kl_relative_decrease": 1 - summary["semantic_convergence"]["means"]["epochs_41_50"]["online_kl_estimate"] / summary["semantic_convergence"]["means"]["epochs_31_40"]["online_kl_estimate"],
        },
        "limitations": ["Synthetic gradients do not establish real-data gradient conflict or capacity insufficiency.",
                        "B8 update count is arithmetic for the published inventory, not a completed B8 run.",
                        "No patient/model convergence or segmentation-quality experiment was executed."],
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
