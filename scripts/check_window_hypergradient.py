#!/usr/bin/env python3
"""Numerical gate on manufactured NativeStudent inputs; never patient accuracy.

Compare update-mediated derivatives to central differences with the OUTER
geometry value held constant. Also compare the virtual optimizer to the actual
ordinary update, from both cold and initialized optimizer states.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.func import functional_call

from self_audit.engine import source_digest, supervised_step
from self_audit.meta_window import (
    _functional_step_with_states, _masked_pseudo_ce, _model_parameter_index,
    _present_foreground_dice, _snapshot_parameters, geometry_named_parameters,
    meta_geometry_gradients,
)
from self_audit.model import build_model


def ordinary_optimizers(model):
    geometry = geometry_named_parameters(model)
    body = [p for n, p in model.named_parameters() if not n.startswith("head.") and n not in geometry]
    return [torch.optim.AdamW(body, lr=.003, weight_decay=1e-4, foreach=False),
            torch.optim.SGD(model.head.parameters(), lr=.01, foreach=False)]


def virtual_objective(model, opts, strong, py, valid, xa, ya, inner_psi, outer_psi):
    named, geometry, owners = _model_parameter_index(model, opts)
    theta = _snapshot_parameters(model, [n for n in named if n not in geometry], requires_grad=None)
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    inner = functional_call(model, {**buffers, **theta, **inner_psi}, (strong,), {"valid_mask": valid})
    loss, _ = _masked_pseudo_ce(inner, py, valid)
    names = [n for n, p in theta.items() if p.requires_grad]
    gradients = torch.autograd.grad(loss, [theta[n] for n in names], allow_unused=True)
    virtual, _ = _functional_step_with_states(theta, dict(zip(names, gradients)), owners, named, clip_norm=5.)
    outer = functional_call(model, {**buffers, **virtual, **outer_psi}, (xa,), {"valid_mask": valid})
    objective, _, _ = _present_foreground_dice(outer, ya, valid)
    return objective.detach(), virtual, dict(zip(names, gradients))


def numerical_case(device, *, warm):
    torch.manual_seed(41)
    model, _ = build_model({"backbone": "tiny_test"}, synthetic=True,
                           window={"mode": "adaptive", "dim": 8})
    model = model.to(device=device, dtype=torch.float64).eval()
    opts = ordinary_optimizers(model)
    strong = torch.rand(1, 3, 32, 32, dtype=torch.float64, device=device)
    xa = torch.rand_like(strong)
    pattern = torch.tensor([[0,1,2,3], [1,2,3,0], [2,3,0,1], [3,0,1,2]], device=device)
    py = pattern.repeat(8, 8)[None]; ya = py.clone()
    valid = torch.ones_like(py, dtype=torch.bool)
    if warm:
        supervised_step(model, opts, xa, ya, valid_mask=valid, freeze_geometry=True)
    model.eval(); model.zero_grad(set_to_none=True)
    gradients, diag = meta_geometry_gradients(model, opts, strong, py, valid, xa, ya,
                                              input_valid=valid, audit_valid=valid)
    geometry = {n: p.detach().clone() for n, p in geometry_named_parameters(model).items()}
    name = "window.geometry.offset.bias"
    checks = []
    for index, label in ((0, "center_x"), (2, "radius_x")):
        epsilon = 2e-4
        direction = torch.zeros_like(geometry[name]); direction[index] = epsilon
        plus = {n: p + (direction if n == name else 0) for n,p in geometry.items()}
        minus = {n: p - (direction if n == name else 0) for n,p in geometry.items()}
        upper, _, _ = virtual_objective(model, opts, strong, py, valid, xa, ya, plus, geometry)
        lower, _, _ = virtual_objective(model, opts, strong, py, valid, xa, ya, minus, geometry)
        finite_difference = (upper-lower) / (2*epsilon)
        analytic = gradients[name][index]
        if analytic.abs() <= 1e-10:
            raise AssertionError(f"uninformative zero {label} hypergradient")
        torch.testing.assert_close(analytic, finite_difference, rtol=2e-3, atol=1e-9)
        checks.append({"direction": label, "analytic": float(analytic),
                       "finite_difference": float(finite_difference),
                       "absolute_error": float((analytic-finite_difference).abs()),
                       "relative_error": float((analytic-finite_difference).abs()/analytic.abs())})
    _, virtual, inner_grads = virtual_objective(model, opts, strong, py, valid, xa, ya, geometry, geometry)
    real = copy.deepcopy(model); real_opts = ordinary_optimizers(real)
    for dst, src in zip(real_opts, opts, strict=True): dst.load_state_dict(copy.deepcopy(src.state_dict()))
    for n, p in real.named_parameters():
        p.grad = inner_grads[n].detach().clone() if inner_grads.get(n) is not None else None
    torch.nn.utils.clip_grad_norm_([p for n,p in real.named_parameters() if n not in geometry], 5.)
    for opt in real_opts: opt.step()
    error = 0.
    for n,p in real.named_parameters():
        if n in virtual:
            torch.testing.assert_close(virtual[n], p, rtol=1e-11, atol=1e-12)
            error = max(error, float((virtual[n]-p).abs().max().detach()))
    return {"optimizer_state": "warm" if warm else "cold", "checks": checks,
            "optimizer_max_absolute_error": error, "meta": diag}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", required=True, help="New JSON receipt path")
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists(): parser.error("receipt already exists")
    if args.device == "cuda" and not torch.cuda.is_available(): parser.error("CUDA unavailable")
    torch.set_num_threads(2)
    result = {"schema": "self_audit.window_hypergradient.v1", "status": "RUNNING",
              "source_sha256": source_digest(), "torch": torch.__version__, "device": args.device,
              "dtype": "float64", "model": "NativeStudent(TinyEncoder); not pretrained DINOv3",
              "synthetic": True, "patient_training": False, "patient_accuracy": "NOT_EVALUATED"}
    try:
        result["cases"] = [numerical_case(args.device, warm=warm) for warm in (False, True)]
        result["status"] = "PASS"
    except Exception as exc:
        result.update(status="FAIL", error=repr(exc))
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
