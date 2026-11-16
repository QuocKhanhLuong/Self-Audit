"""Candidate FUSE's determinism/TF32 policy, shared by PG stages."""
from __future__ import annotations

import os
import torch


def configure_runtime(device: str | torch.device, cublas_workspace: str) -> dict:
    cuda = torch.device(device).type == "cuda"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace
    torch.use_deterministic_algorithms(True, warn_only=cuda)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "deterministic": True, "deterministic_warn_only": cuda,
        "determinism_status": "BEST_EFFORT_GPU_WARN_ONLY" if cuda else "STRICT_CPU",
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "tf32": False, "cublas_workspace_config": cublas_workspace,
    }
