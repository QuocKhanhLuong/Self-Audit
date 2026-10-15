"""Local approved-backbone preflight only; never launches patient training."""

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import torch

from .artifacts import construct_student, source_hashes
from .geometry import require_digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--approval", required=True, type=Path)
    parser.add_argument("--expected-approval-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("preflight output must be a new directory")
    from .artifacts import sha256

    require_digest(args.expected_approval_sha256)
    if (
        not args.approval.is_file()
        or sha256(args.approval) != args.expected_approval_sha256
    ):
        parser.error("BLOCKED_PRETRAINED: approval receipt missing or digest differs")
    recipe, approval = (
        json.loads(path.read_text()) for path in (args.recipe, args.approval)
    )
    started = time.perf_counter()
    model, provenance = construct_student(recipe, approval, args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.random.fork_rng(
        devices=[torch.cuda.current_device()] if args.device == "cuda" else []
    ):
        torch.manual_seed(recipe["seed"])
        images = torch.rand(1, 3, 65, 97, device=args.device)
        outputs = model(images)
        loss = outputs["logits"].square().mean() + outputs["features"].square().mean()
        loss.backward()
    if not all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    ):
        raise ValueError("nonfinite preflight gradient")
    if args.device == "cuda":
        torch.cuda.synchronize()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = {
        "status": "PASS_MANUFACTURED_INPUT_EXECUTION_ONLY",
        "recipe": recipe,
        "backbone": provenance,
        "sources": source_hashes(),
        "approval_sha256": args.expected_approval_sha256,
        "outputs": {k: list(v.shape) for k, v in outputs.items()},
        "parameters": sum(p.numel() for p in model.parameters()),
        "optimizer_steps": 0,
        "patient_arrays_read": False,
        "anatomy_quality": "NOT_MEASURED",
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else None,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": args.device,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "PREFLIGHT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
