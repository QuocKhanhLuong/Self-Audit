"""Versioned student inference contract. No implicit legacy/profile/source override."""
from __future__ import annotations
import platform
import subprocess
from pathlib import Path
import numpy as np
import torch
from .freeze import sha256_file
from .adaptive import RuntimeBudget, PROFILE_TURNS

PREPROCESSING = {"axes": "native_XYZT_to_TZYX", "normalization": "cine_p1_p99_clip_zscore",
                 "input": "replicated_z_triplet", "classes": ["BG", "RV", "MYO", "LV"]}


def source_identity():
    package = Path(__file__).resolve().parent
    files = list(package.glob("*.py"))
    files += [package.parent / "self_audit/models" / f for f in
              ("annotation_expert.py", "dynamic_window.py")]
    return {str(p.relative_to(package.parent)): sha256_file(p) for p in sorted(files)}


def environment_identity():
    import nibabel
    import scipy
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "nibabel": nibabel.__version__, "scipy": scipy.__version__}


def git_identity():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def save_student_checkpoint(path, model, *, args, model_config, manifest_id, coverage):
    profile = args["profile"]
    if sum(coverage["class_pixels"][1:]) == 0:
        raise ValueError("NO_OBSERVED_FOREGROUND: no checkpoint written")
    model.mark_profile_trained(profile)
    budget = RuntimeBudget.from_config(model_config, max_profile=profile)
    model.runtime_thresholds.copy_(torch.tensor([budget.entropy_compact, budget.entropy_accurate],dtype=torch.float64,device=model.runtime_thresholds.device))
    payload = {"checkpoint_schema": 1, "model": model.state_dict(), "args": args,
               "model_config": model_config, "manifest_id": manifest_id, "profile_trained": profile,
               "coverage": coverage, "source_sha256": source_identity(), "commit": git_identity(),
               "environment": environment_identity(), "preprocessing": PREPROCESSING}
    with Path(path).open("xb") as handle:
        torch.save(payload, handle)
    return sha256_file(path)


def load_student_checkpoint(path, *, device="cpu", max_profile=None, manifest_id=None):
    from .system_v3 import AdaptiveAnnotationStudent
    digest = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_schema") != 1:
        raise ValueError("legacy checkpoint lacks inference contract; regenerate")
    if payload.get("source_sha256") != source_identity():
        raise ValueError("student source/dependency identity mismatch")
    if payload.get("environment") != environment_identity():
        raise ValueError("student environment identity mismatch")
    if payload.get("preprocessing") != PREPROCESSING:
        raise ValueError("student preprocessing mismatch")
    if manifest_id is not None and payload.get("manifest_id") != manifest_id:
        raise ValueError("student freeze manifest mismatch")
    profile = payload.get("profile_trained")
    if profile not in PROFILE_TURNS:
        raise ValueError("missing trained profile")
    coverage = payload.get("coverage", {})
    counts = coverage.get("class_pixels", [])
    if (len(counts) != 4 or any(type(n) is not int or n < 0 for n in counts)
            or sum(counts[1:]) == 0 or not coverage.get("patient_ids")):
        raise ValueError("missing observed foreground supervision")
    requested = max_profile or profile
    if requested not in PROFILE_TURNS or PROFILE_TURNS[requested] > PROFILE_TURNS[profile]:
        raise ValueError("requested profile exceeds trained checkpoint coverage")
    cfg = payload["model_config"]
    budget = RuntimeBudget.from_config(cfg, max_profile=requested)
    model = AdaptiveAnnotationStudent(width=cfg["student_width"], window_k=cfg["window_k"])
    model.load_state_dict(payload["model"], strict=True)
    if int(model.trained_profile_cap) != PROFILE_TURNS[profile]:
        raise ValueError("state_dict trained profile mismatch")
    if not np.allclose(model.runtime_thresholds.numpy(), [budget.entropy_compact, budget.entropy_accurate],
                       atol=1e-7, rtol=0):
        raise ValueError("runtime threshold/config mismatch")
    if sha256_file(path) != digest:
        raise ValueError("checkpoint changed while loading")
    model.to(device).eval()
    return model, budget, {**payload, "checkpoint_sha256": digest}
