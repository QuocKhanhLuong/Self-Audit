#!/usr/bin/env python3
"""One discarded-weight TRAIN step through the canonical B8 focal trainer."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify(config, data_path, data_sha, out):
    import torch
    from self_audit import engine
    from self_audit.model import digest

    c = json.loads(Path(config).read_text())
    if c.get("supervised_loss", {}).get("name") != "focal_dice" or c["method"] != "ema_full":
        raise ValueError("this diagnostic requires the declared full-model focal recipe")
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=False)
    original = engine.supervised_step
    observed = []

    def observed_step(model, optimizers, x, y, **kwargs):
        if not any(isinstance(v, dict) and v == c["supervised_loss"] for v in kwargs.values()):
            raise RuntimeError("canonical trainer did not pass the configured focal loss")
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        result = original(model, optimizers, x, y, **kwargs)
        scopes = {}
        for scope in ("encoder", "input_adapter", "lateral", "up", "refine", "detail", "fuse", "head"):
            params = [(n, p) for n, p in model.named_parameters() if n.startswith(scope + ".")]
            if not params:
                raise RuntimeError(f"required trainable scope missing: {scope}")
            gradients = [p.grad.detach().square().sum() for _, p in params if p.grad is not None]
            row = {
                "parameters": sum(p.numel() for _, p in params),
                "grad_norm": float(torch.stack(gradients).sum().sqrt()) if gradients else 0.0,
                "weight_change_norm": float(torch.stack([(p.detach()-before[n]).square().sum() for n, p in params]).sum().sqrt()),
                "optimizer_owns_all": all(any(p is q for o in optimizers for g in o.param_groups for q in g["params"]) for _, p in params),
            }
            if not (row["grad_norm"] > 0 and row["weight_change_norm"] > 0 and row["optimizer_owns_all"]):
                raise RuntimeError(f"optimizer scope failed: {scope}: {row}")
            scopes[scope] = row
        observed.append({"scopes": scopes, "loss": result[0], "batch_shape": list(x.shape),
                         "optimizers": [{"type": type(o).__name__, "groups": [
                             {"lr": g["lr"], "weight_decay": g["weight_decay"],
                              "parameters": sum(p.numel() for p in g["params"])}
                             for g in o.param_groups]} for o in optimizers]})
        return result

    start = time.monotonic()
    engine.supervised_step = observed_step
    try:
        report = engine.train(c, data_path, data_sha, destination / "diagnostic_training",
                              device="cuda", stop_after_step=1)
    finally:
        engine.supervised_step = original
    torch.cuda.synchronize()
    if len(observed) != 1 or report["supervised_updates"] != 1 or report["ema_updates"] != 1 or report["label_reads_outside_budget"]:
        raise RuntimeError("unexpected diagnostic update count or label access")
    receipt = {"status": "PASS", "scope": "one discarded-weight canonical supervised diagnostic; no quality claim",
               "source_sha256": engine.source_digest(), "config_sha256": digest(config),
               "data_sha256": data_sha, "supervised_loss": c["supervised_loss"], "lr_scheduler": c["lr_scheduler"],
               "observed": observed, "report": report, "seconds": time.monotonic()-start,
               "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
    (destination / "GPU_SCOPE_SMOKE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "data-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    verify(args.config, args.data, args.data_sha256, args.out)
