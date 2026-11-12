from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from self_audit.candidate_q3d.export import (
    EXPORT_SCHEMA,
    _assert_teacher_free_imports,
    verify_cold_load,
)
from self_audit.candidate_q3d.student import CandidateQ3DStudent


def _config() -> dict:
    return {
        "schema": "q3d.config.v1",
        "arm": "V1",
        "model": {"student": {
            "channels": [4, 4, 4, 4, 4],
            "in_channels": 1,
            "num_classes": 4,
            "attention": {"support": "uniform"},
            "vq": {"size": 4, "dim": 4, "tau": 0.1},
            "residual_vq": {"enabled": False},
        }},
        "tokenizer": {"codebook_size": 4, "codebook_dim": 4, "tau": 0.1},
    }


def _frozen_codebook(codes: torch.Tensor) -> dict:
    codes = codes.detach().to(dtype=torch.float32).contiguous().cpu()
    return {
        "codes": codes,
        "version": "q3d_teacher_codebook.v1",
        "sha256": hashlib.sha256(codes.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "size": int(codes.shape[0]),
        "dimension": int(codes.shape[1]),
        "initialized": True,
        "frozen": True,
    }


def _write_artifact(path: Path) -> CandidateQ3DStudent:
    config = _config()
    model = CandidateQ3DStudent(
        config, codebook=_frozen_codebook(torch.eye(4, dtype=torch.float32))
    ).to(dtype=torch.float32).eval()
    model.sdf_head = None
    state = {
        key: (dict(value) if key == "_extra_state" else value.detach().cpu().contiguous())
        for key, value in model.state_dict().items()
    }
    torch.save({"schema": EXPORT_SCHEMA, "config": config, "state_dict": state, "metadata": {}}, path)
    return model


def _write_volume(path: Path, shape: tuple[int, int, int], seed: int) -> Path:
    image = np.random.default_rng(seed).random(shape, dtype=np.float32)
    spacing = np.asarray((8.0, 1.25, 1.5), dtype=np.float64)
    np.savez_compressed(
        path,
        image=image,
        shape_zyx=np.asarray(shape, dtype=np.int32),
        spacing_zyx=spacing,
        affine_zyx=np.diag([*spacing, 1.0]),
        context=np.asarray("full_volume"),
    )
    return path


@pytest.mark.parametrize(
    "module_name",
    [
        "self_audit.candidate_q3d.teacher_resenc",
        "dynamic_network_architectures.architectures.unet",
        "nnssl.networks",
    ],
)
def test_student_only_export_rejects_teacher_and_encoder_dependency_imports(module_name: str) -> None:
    with pytest.raises(RuntimeError, match="teacher modules"):
        _assert_teacher_free_imports({module_name})


@pytest.mark.parametrize(
    ("thread_mode", "default_dtype"),
    [("parent_default", torch.float32), (1, torch.float64)],
)
def test_cold_load_inherits_runtime_and_keeps_bitwise_logits(
    tmp_path: Path, thread_mode: int | str, default_dtype: torch.dtype
) -> None:
    old_threads = torch.get_num_threads()
    old_default_dtype = torch.get_default_dtype()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_cudnn_deterministic = torch.backends.cudnn.deterministic
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    old_cuda_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_matmul_precision = torch.get_float32_matmul_precision()
    try:
        if thread_mode != "parent_default":
            torch.set_num_threads(thread_mode)
        torch.set_default_dtype(default_dtype)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("high")

        artifact = tmp_path / "student.pt"
        model = _write_artifact(artifact)
        volumes = [
            _write_volume(tmp_path / f"volume_{index}.npz", shape, seed=100 + index)
            for index, shape in enumerate(((6, 17, 19), (9, 19, 20), (10, 16, 17)))
        ]

        result = verify_cold_load(model, artifact, volumes)

        assert result["status"] == "PASSED_BITWISE_THREE_NATIVE_VOLUMES"
        assert result["volumes"] == 3
        assert result["teacher_modules_imported"] == []
        assert len(result["logit_sha256"]) == 3
    finally:
        torch.use_deterministic_algorithms(old_determinism, warn_only=old_warn_only)
        torch.set_float32_matmul_precision(old_matmul_precision)
        torch.backends.cudnn.deterministic = old_cudnn_deterministic
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = old_cudnn_allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = old_cuda_matmul_allow_tf32
        torch.set_default_dtype(old_default_dtype)
        if torch.get_num_threads() != old_threads:
            torch.set_num_threads(old_threads)
