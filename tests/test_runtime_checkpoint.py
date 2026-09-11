"""Comprehensive tests for Wave 1 runtime checkpoint hardening.

Covers:
- Root torch 2.4.1 KeyError uint32 reproduction and lossless int64 MT keys
- Python / NumPy cached gaussian / Torch / CUDA RNG exact streams and validation
- Recursive CPU-normalization and weights_only safety
- Rejection of unsupported custom objects and dtypes before replacement with exact path errors
- Explicit nonfinite metadata sentinel policy without zeroing
- Atomic temp save, unswallowed fsync, and replace cleanup preserving old checkpoint
- Cleanup failure not masking primary error
- Full save/load/restore/resave roundtrip with model, optimizer, scheduler, scaler
- Deterministic continuation after checkpoint reload and resave
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from src.self_audit.serialization import (
    CheckpointSerializationError,
    SAFE_TENSOR_DTYPES,
    atomic_save_torch,
    normalize_checkpoint_payload,
)
from src.self_audit.training._utils import (
    _cpu_state_dict,
    _finite_tree,
    _restore_rng_state,
    _rng_state,
    load_checkpoint,
    save_checkpoint,
)


class _SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# 1. Root PyTorch 2.4.1 uint32 KeyError regression & lossless MT keys
# ---------------------------------------------------------------------------

def test_rng_numpy_mt19937_uint32_regression_and_lossless_roundtrip(tmp_path: Path) -> None:
    """_rng_state stores MT19937 keys as int64 tensor, preventing torch 2.4.1 uint32 KeyError."""
    np.random.seed(42)
    state = _rng_state()
    algo, keys_tensor, pos, has_gauss, cached_gaussian = state["numpy"]

    assert algo == "MT19937"
    assert torch.is_tensor(keys_tensor)
    assert keys_tensor.dtype == torch.int64
    assert keys_tensor.shape == (624,)

    model = _SimpleModel()
    ckpt_path = tmp_path / "test_uint32_regression.pt"
    save_checkpoint(ckpt_path, model, epoch=1)

    payload = torch.load(ckpt_path, weights_only=True)
    saved_keys = payload["rng_state"]["numpy"][1]
    assert saved_keys.dtype == torch.int64
    assert torch.equal(saved_keys, keys_tensor)


def test_restore_rng_validates_uint32_range_and_shape() -> None:
    """_restore_rng_state validates MT19937 key shape and uint32 range before setting state."""
    with pytest.raises(ValueError, match="must have length 624"):
        _restore_rng_state({"numpy": ("MT19937", torch.zeros(10, dtype=torch.int64), 0, 0, 0.0)})

    with pytest.raises(ValueError, match="outside valid uint32 range"):
        bad_keys = torch.zeros(624, dtype=torch.int64)
        bad_keys[0] = -1
        _restore_rng_state({"numpy": ("MT19937", bad_keys, 0, 0, 0.0)})

    with pytest.raises(ValueError, match="outside valid uint32 range"):
        overflow_keys = torch.zeros(624, dtype=torch.int64)
        overflow_keys[5] = 4294967296  # 2^32, exceeds uint32
        _restore_rng_state({"numpy": ("MT19937", overflow_keys, 0, 0, 0.0)})

    with pytest.raises(ValueError, match="pos must be in"):
        _restore_rng_state({"numpy": ("MT19937", torch.zeros(624, dtype=torch.int64), 999, 0, 0.0)})

    with pytest.raises(ValueError, match="has_gauss must be 0 or 1"):
        _restore_rng_state({"numpy": ("MT19937", torch.zeros(624, dtype=torch.int64), 0, 2, 0.0)})

    with pytest.raises(ValueError, match="cached_gaussian must be finite"):
        _restore_rng_state({"numpy": ("MT19937", torch.zeros(624, dtype=torch.int64), 0, 1, float("nan"))})


# ---------------------------------------------------------------------------
# 2. Exact stream preservation (Python, NumPy cached gaussian, Torch, CUDA)
# ---------------------------------------------------------------------------

def test_rng_exact_stream_continuation_including_cached_gaussian(tmp_path: Path) -> None:
    """Python and NumPy cached gaussian states are preserved bit-for-bit across save/load."""
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)

    # Prime Box-Muller Gaussian caches
    _ = random.gauss(0.0, 1.0)
    _ = np.random.normal(0.0, 1.0)

    model = _SimpleModel()
    ckpt_path = tmp_path / "rng_gaussian_stream.pt"
    save_checkpoint(ckpt_path, model, epoch=1)

    # Generate expected continuation stream
    expected_py = [random.gauss(0.0, 1.0) for _ in range(10)]
    expected_np = [float(np.random.normal(0.0, 1.0)) for _ in range(10)]
    expected_th = torch.randn(10).tolist()

    # Perturb RNGs
    random.seed(9999)
    np.random.seed(9999)
    torch.manual_seed(9999)
    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(5)

    # Restore via load_checkpoint
    restored_model = _SimpleModel()
    load_checkpoint(ckpt_path, model=restored_model, restore_rng=True)

    actual_py = [random.gauss(0.0, 1.0) for _ in range(10)]
    actual_np = [float(np.random.normal(0.0, 1.0)) for _ in range(10)]
    actual_th = torch.randn(10).tolist()

    assert actual_py == expected_py
    assert actual_np == expected_np
    assert actual_th == expected_th


def test_cuda_rng_state_handling() -> None:
    """CUDA RNG validation enforces list/tuple of tensors and device count match."""
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA RNG restore requested but CUDA is not available"):
            _restore_rng_state({"cuda": []}, exact_cuda=True)
    else:
        dev_count = torch.cuda.device_count()
        # The implementation names both counts; the old regex
        # ("does not match available devices") never matched this message and,
        # living inside the CUDA-only arm, could not fail on a CPU-only host.
        # The count branch is additionally covered under mocked CUDA in
        # tests/test_runtime_amp_resume_review.py, so it runs everywhere.
        with pytest.raises(
            ValueError,
            match=rf"Checkpoint contains {dev_count + 1} CUDA RNG state\(s\), but {dev_count} device\(s\) are available",
        ):
            _restore_rng_state({"cuda": [torch.zeros(10, dtype=torch.uint8)] * (dev_count + 1)})


# ---------------------------------------------------------------------------
# 3. Path metadata & weights_only safety
# ---------------------------------------------------------------------------

def test_path_metadata_normalizes_to_str_surviving_weights_only(tmp_path: Path) -> None:
    """Path objects in metadata are recursively normalized to strings for weights_only safety."""
    model = _SimpleModel()
    ckpt_path = tmp_path / "path_metadata.pt"

    config = {
        "model_dir": Path("/tmp/models"),
        "nested": {"manifest": Path("/data/manifest.json")},
        "path_list": [Path("/a"), Path("/b")],
    }
    extra = {"best_checkpoint_path": Path("/out/best.pt")}

    save_checkpoint(ckpt_path, model, config=config, extra=extra)

    payload = torch.load(ckpt_path, weights_only=True)
    assert payload["config"]["model_dir"] == "/tmp/models"
    assert payload["config"]["nested"]["manifest"] == "/data/manifest.json"
    assert payload["config"]["path_list"] == ["/a", "/b"]
    assert payload["best_checkpoint_path"] == "/out/best.pt"


# ---------------------------------------------------------------------------
# 4. Rejection of unsupported custom objects/dtypes with exact path errors
# ---------------------------------------------------------------------------

def test_rejection_of_unsupported_custom_objects_with_exact_path_error(tmp_path: Path) -> None:
    """Unsupported custom objects in metadata/config/extra fail before replace with exact path."""
    model = _SimpleModel()
    ckpt_path = tmp_path / "custom_obj_test.pt"

    class CustomMetadata:
        def __init__(self, value: int):
            self.value = value

    # Deeply nested object
    config = {
        "pipeline": {
            "stages": [
                {"name": "preprocess"},
                {"name": "custom", "handler": CustomMetadata(42)},
            ]
        }
    }

    with pytest.raises((TypeError, ValueError)) as exc_info:
        save_checkpoint(ckpt_path, model, config=config)

    assert "config.pipeline.stages[1].handler" in str(exc_info.value)
    assert not ckpt_path.exists(), "Destination file must not be created on validation failure"


def test_rejection_of_unsupported_tensor_dtype_with_exact_path(tmp_path: Path) -> None:
    """Tensors with unsupported dtypes (like uint32) are rejected before writing."""
    model = _SimpleModel()
    ckpt_path = tmp_path / "bad_dtype_test.pt"

    if hasattr(torch, "uint32"):
        bad_tensor = torch.zeros(4, dtype=torch.uint32)
        extra = {"debug_buffer": bad_tensor}

        with pytest.raises((TypeError, ValueError)) as exc_info:
            save_checkpoint(ckpt_path, model, extra=extra)

        assert "extra.debug_buffer" in str(exc_info.value)
        assert not ckpt_path.exists()


# ---------------------------------------------------------------------------
# 5. Explicit nonfinite metadata sentinel policy without zeroing
# ---------------------------------------------------------------------------

def test_nonfinite_metadata_sentinel_policy_without_zeroing(tmp_path: Path) -> None:
    """Metadata sentinels (None, NaN, -Inf, Inf) are preserved without zeroing or corruption."""
    model = _SimpleModel()
    ckpt_path = tmp_path / "nonfinite_metadata.pt"

    extra = {
        "val_dice": None,  # Undefined Dice sentinel
        "worst_dice": float("nan"),  # Nonfinite metric sentinel
        "min_metric": float("-inf"),  # Best-tracker initialization sentinel
        "max_metric": float("inf"),
        "numpy_nan": np.float32(np.nan),
        "numpy_inf": np.float64(np.inf),
    }

    save_checkpoint(ckpt_path, model, extra=extra)

    loaded = load_checkpoint(ckpt_path)
    assert loaded["val_dice"] is None, "None must not be converted to 0.0"
    assert math.isnan(loaded["worst_dice"]), "NaN must not be converted to 0.0"
    assert loaded["min_metric"] == float("-inf"), "-Inf must not be converted to 0.0"
    assert loaded["max_metric"] == float("inf"), "+Inf must not be converted to 0.0"
    assert math.isnan(loaded["numpy_nan"]), "NumPy NaN must be preserved as float NaN"
    assert loaded["numpy_inf"] == float("inf"), "NumPy Inf must be preserved as float Inf"


def test_nonfinite_in_training_states_strictly_raises(tmp_path: Path) -> None:
    """Non-finite values in model weights or optimizer states strictly raise FloatingPointError."""
    model = _SimpleModel()
    model.fc1.weight.data[0, 0] = float("nan")

    with pytest.raises(FloatingPointError, match="Non-finite"):
        save_checkpoint(tmp_path / "nan_model.pt", model)

    clean_model = _SimpleModel()
    optimizer = torch.optim.SGD(clean_model.parameters(), lr=0.01)
    optimizer.param_groups[0]["lr"] = float("nan")

    with pytest.raises(FloatingPointError, match="Non-finite"):
        save_checkpoint(tmp_path / "nan_opt.pt", clean_model, optimizer=optimizer)


# ---------------------------------------------------------------------------
# 6. Unswallowed fsync failure & atomic replacement preservation
# ---------------------------------------------------------------------------

def test_fsync_failure_is_not_swallowed_and_preserves_old_checkpoint(tmp_path: Path) -> None:
    """Injected OSError in fsync propagates and leaves the existing checkpoint untouched."""
    model = _SimpleModel()
    ckpt_path = tmp_path / "phase_c_best.pt"

    # Step 1: Establish valid checkpoint
    save_checkpoint(ckpt_path, model, epoch=1)
    initial_bytes = ckpt_path.read_bytes()

    # Step 2: Attempt save with injected fsync failure (e.g. ENOSPC 28)
    with patch("os.fsync", side_effect=OSError(28, "No space left on device")):
        with pytest.raises(OSError) as exc_info:
            save_checkpoint(ckpt_path, model, epoch=2)
        assert exc_info.value.errno == 28

    # Verify old checkpoint is byte-identical and untouched
    assert ckpt_path.read_bytes() == initial_bytes

    # Verify no leaked temporary files in the directory
    temp_files = [f for f in tmp_path.iterdir() if f.name.endswith(".tmp")]
    assert len(temp_files) == 0


def test_replace_failure_preserves_old_checkpoint(tmp_path: Path) -> None:
    """Injected failure in os.replace preserves old checkpoint and cleans up tempfile."""
    model = _SimpleModel()
    ckpt_path = tmp_path / "phase_c_best.pt"
    save_checkpoint(ckpt_path, model, epoch=1)
    initial_bytes = ckpt_path.read_bytes()

    with patch("os.replace", side_effect=OSError(13, "Permission denied")):
        with pytest.raises(OSError) as exc_info:
            save_checkpoint(ckpt_path, model, epoch=2)
        assert exc_info.value.errno == 13

    assert ckpt_path.read_bytes() == initial_bytes
    temp_files = [f for f in tmp_path.iterdir() if f.name.endswith(".tmp")]
    assert len(temp_files) == 0


def test_cleanup_failure_does_not_mask_primary_error(tmp_path: Path) -> None:
    """If tempfile cleanup raises during an exception, the primary error is not masked."""
    model = _SimpleModel()
    ckpt_path = tmp_path / "phase_c_best.pt"

    with patch("torch.save", side_effect=RuntimeError("Primary serialization failure")):
        with patch.object(Path, "unlink", side_effect=OSError(1, "Operation not permitted")):
            with pytest.raises(RuntimeError, match="Primary serialization failure"):
                save_checkpoint(ckpt_path, model)


# ---------------------------------------------------------------------------
# 7. Optimizer, scheduler, scaler roundtrip & deterministic continuation
# ---------------------------------------------------------------------------

def test_full_roundtrip_and_deterministic_continuation(tmp_path: Path) -> None:
    """Full roundtrip save/load/resave with optimizer, scheduler, scaler and exact replay."""
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)

    model1 = _SimpleModel()
    opt1 = torch.optim.AdamW(model1.parameters(), lr=0.01, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=2, gamma=0.5)
    scaler1 = torch.amp.GradScaler("cpu", enabled=False)

    # Initial 2 optimization steps
    for step in range(2):
        x = torch.randn(4, 4)
        target = torch.tensor([1, 0, 1, 0], dtype=torch.int64)
        out = model1(x)
        loss = nn.functional.cross_entropy(out, target)
        loss.backward()
        opt1.step()
        opt1.zero_grad()
        sched1.step()

    ckpt1_path = tmp_path / "ckpt_epoch2.pt"
    save_checkpoint(
        ckpt1_path,
        model1,
        optimizer=opt1,
        scheduler=sched1,
        scaler=scaler1,
        epoch=2,
        global_step=2,
        optimizer_step=2,
        config={"lr": 0.01},
    )

    # Continue training model1 for 3 more steps (reference trajectory)
    ref_weights: list[torch.Tensor] = []
    ref_lr: list[float] = []
    ref_randoms: list[float] = []
    for step in range(3):
        x = torch.randn(4, 4)
        ref_randoms.append(random.random())
        target = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        out = model1(x)
        loss = nn.functional.cross_entropy(out, target)
        loss.backward()
        opt1.step()
        opt1.zero_grad()
        sched1.step()
        ref_lr.append(float(sched1.get_last_lr()[0]))
    for p in model1.parameters():
        ref_weights.append(p.detach().clone())

    # Branch 2: Fresh instance loaded from checkpoint, resaved, reloaded, then trained
    model2 = _SimpleModel()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=0.01, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=2, gamma=0.5)
    scaler2 = torch.amp.GradScaler("cpu", enabled=False)

    loaded = load_checkpoint(
        ckpt1_path,
        model=model2,
        optimizer=opt2,
        scheduler=sched2,
        scaler=scaler2,
        restore_rng=True,
    )
    assert loaded["epoch"] == 2
    assert loaded["global_step"] == 2
    assert loaded["optimizer_step"] == 2

    # Resave and reload to verify idempotence
    ckpt_resaved = tmp_path / "ckpt_resaved.pt"
    save_checkpoint(
        ckpt_resaved,
        model2,
        optimizer=opt2,
        scheduler=sched2,
        scaler=scaler2,
        epoch=2,
        global_step=2,
        optimizer_step=2,
    )

    model3 = _SimpleModel()
    opt3 = torch.optim.AdamW(model3.parameters(), lr=0.01, weight_decay=1e-4)
    sched3 = torch.optim.lr_scheduler.StepLR(opt3, step_size=2, gamma=0.5)
    scaler3 = torch.amp.GradScaler("cpu", enabled=False)

    load_checkpoint(
        ckpt_resaved,
        model=model3,
        optimizer=opt3,
        scheduler=sched3,
        scaler=scaler3,
        restore_rng=True,
    )

    test_lr: list[float] = []
    test_randoms: list[float] = []
    for step in range(3):
        x = torch.randn(4, 4)
        test_randoms.append(random.random())
        target = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        out = model3(x)
        loss = nn.functional.cross_entropy(out, target)
        loss.backward()
        opt3.step()
        opt3.zero_grad()
        sched3.step()
        test_lr.append(float(sched3.get_last_lr()[0]))

    # Assert exact deterministic continuation
    assert test_randoms == ref_randoms, "Random stream continuation did not match exactly"
    assert test_lr == ref_lr, "LR schedule progression did not match exactly"
    for p_ref, p_test in zip(ref_weights, model3.parameters()):
        assert torch.equal(p_ref, p_test), "Model weights after continuation did not match exactly"


# ---------------------------------------------------------------------------
# 8. Unsafe container rejection (set, frozenset)
# ---------------------------------------------------------------------------

def test_rejection_of_sets_and_frozensets_with_exact_path(tmp_path: Path) -> None:
    """Sets and frozensets are strictly rejected before writing with exact path errors."""
    model = _SimpleModel()

    # In extra
    with pytest.raises((TypeError, ValueError)) as exc_info1:
        save_checkpoint(tmp_path / "bad_set.pt", model, extra={"tags": {1, 2}})
    assert "extra.tags" in str(exc_info1.value)
    assert "sets and frozensets are not supported" in str(exc_info1.value)

    # In config
    with pytest.raises((TypeError, ValueError)) as exc_info2:
        save_checkpoint(tmp_path / "bad_frozenset.pt", model, config={"frozen": frozenset([3, 4])})
    assert "config.frozen" in str(exc_info2.value)


# ---------------------------------------------------------------------------
# 9. TorchVersion & primitive subclass normalization
# ---------------------------------------------------------------------------

def test_torch_version_and_string_byte_subclasses_normalized_surviving_weights_only(tmp_path: Path) -> None:
    """TorchVersion and string/byte subclasses are normalized to builtin primitives."""
    model = _SimpleModel()
    ckpt_path = tmp_path / "torch_version.pt"

    # torch.__version__ is a torch.torch_version.TorchVersion instance (a str subclass)
    assert type(torch.__version__) is not str or isinstance(torch.__version__, str)

    save_checkpoint(ckpt_path, model, config={"torch_version": torch.__version__})

    payload = torch.load(ckpt_path, weights_only=True)
    loaded_version = payload["config"]["torch_version"]
    assert type(loaded_version) is str, "TorchVersion must be normalized to builtin str"
    assert loaded_version == str(torch.__version__)


# ---------------------------------------------------------------------------
# 10. Mapping key validation, collision detection, and unsafe key rejection
# ---------------------------------------------------------------------------

def test_mapping_keys_normalization_collision_detection_and_unsafe_key_rejection(tmp_path: Path) -> None:
    """Mapping keys are normalized to builtin int/str, collisions detected, unsafe keys rejected."""
    model = _SimpleModel()
    ckpt_path = tmp_path / "mapping_keys.pt"

    # np.integer key normalizes to builtin int
    config = {"params": {np.int64(42): "layer_state"}}
    save_checkpoint(ckpt_path, model, config=config)

    payload = torch.load(ckpt_path, weights_only=True)
    assert 42 in payload["config"]["params"]
    assert type(list(payload["config"]["params"].keys())[0]) is int

    # Key collision detection (Path and str both normalize to str)
    collision_config = {"params": {Path("/test/key"): "val1", "/test/key": "val2"}}
    with pytest.raises((TypeError, ValueError)) as exc_info:
        save_checkpoint(tmp_path / "collision.pt", model, config=collision_config)
    assert "collision" in str(exc_info.value).lower()

    # Boolean key rejection
    with pytest.raises((TypeError, ValueError)) as exc_info_bool:
        save_checkpoint(tmp_path / "bool_key.pt", model, config={"bad": {True: "yes"}})
    assert "bool" in str(exc_info_bool.value).lower()

    # Float key rejection
    with pytest.raises((TypeError, ValueError)) as exc_info_float:
        save_checkpoint(tmp_path / "float_key.pt", model, config={"bad": {1.5: "fraction"}})
    assert "keys must be string or integer" in str(exc_info_float.value)


# ---------------------------------------------------------------------------
# 11. rng_state recursive validation
# ---------------------------------------------------------------------------

def test_rng_state_recursively_normalized_not_bypassed(tmp_path: Path) -> None:
    """rng_state does not bypass recursive validation in normalize_checkpoint_payload."""
    model = _SimpleModel()
    bad_payload = {
        "model": model.state_dict(),
        "epoch": 1,
        "rng_state": {
            "python": random.getstate(),
            "numpy": _rng_state()["numpy"],
            "torch": torch.get_rng_state(),
            "unsafe_field": {1, 2, 3},  # Unsafe set inside rng_state
        },
    }
    with pytest.raises((TypeError, ValueError)) as exc_info:
        atomic_save_torch(bad_payload, tmp_path / "bad_rng.pt")
    assert "rng_state.unsafe_field" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 12. atomic_save_torch public API auto-normalization
# ---------------------------------------------------------------------------

def test_atomic_save_torch_public_api_normalizes_itself(tmp_path: Path) -> None:
    """atomic_save_torch auto-normalizes unnormalized payloads before saving."""
    ckpt_path = tmp_path / "auto_norm.pt"
    raw_payload = {
        "format_version": 1,
        "epoch": 1,
        "config": {"path": Path("/test/path"), "np_val": np.int64(99)},
    }
    atomic_save_torch(raw_payload, ckpt_path)

    loaded = torch.load(ckpt_path, weights_only=True)
    assert loaded["config"]["path"] == "/test/path"
    assert type(loaded["config"]["path"]) is str
    assert loaded["config"]["np_val"] == 99
    assert type(loaded["config"]["np_val"]) is int


# ---------------------------------------------------------------------------
# 13. RNG restore: non-truncating validation for keys, pos, and has_gauss
# ---------------------------------------------------------------------------

def test_rng_restore_rejects_fractional_nan_bool_and_out_of_range_keys() -> None:
    """_restore_rng_state rejects fractional, NaN, bool, and out-of-range keys without lossy cast."""
    # Fractional tensor keys
    with pytest.raises(ValueError, match="fractional"):
        _restore_rng_state({"numpy": ("MT19937", torch.full((624,), 1.5), 0, 0, 0.0)})

    # Fractional ndarray keys
    with pytest.raises(ValueError, match="fractional"):
        _restore_rng_state({"numpy": ("MT19937", np.full((624,), 2.5), 0, 0, 0.0)})

    # Fractional list keys
    with pytest.raises(ValueError, match="fractional"):
        _restore_rng_state({"numpy": ("MT19937", [3.5] * 624, 0, 0, 0.0)})

    # NaN tensor keys
    with pytest.raises(ValueError, match="non-finite"):
        _restore_rng_state({"numpy": ("MT19937", torch.full((624,), float("nan")), 0, 0, 0.0)})

    # Boolean keys
    with pytest.raises(ValueError, match="(boolean|bool)"):
        _restore_rng_state({"numpy": ("MT19937", np.zeros(624, dtype=bool), 0, 0, 0.0)})


def test_rng_restore_pos_has_gauss_rejects_bool_and_fractional() -> None:
    """pos and has_gauss reject bool and fractional values instead of truncating via int()."""
    valid_keys = torch.zeros(624, dtype=torch.int64)

    # pos rejection
    with pytest.raises(ValueError, match="pos cannot be boolean"):
        _restore_rng_state({"numpy": ("MT19937", valid_keys, True, 0, 0.0)})
    with pytest.raises(ValueError, match="pos cannot be fractional"):
        _restore_rng_state({"numpy": ("MT19937", valid_keys, 1.9, 0, 0.0)})
    with pytest.raises(ValueError, match="pos must be finite"):
        _restore_rng_state({"numpy": ("MT19937", valid_keys, float("nan"), 0, 0.0)})
    with pytest.raises(ValueError, match="pos must be in"):
        _restore_rng_state({"numpy": ("MT19937", valid_keys, -1, 0, 0.0)})
    with pytest.raises(ValueError, match="pos must be in"):
        _restore_rng_state({"numpy": ("MT19937", valid_keys, 625, 0, 0.0)})

    # has_gauss rejection
    with pytest.raises(ValueError, match="has_gauss cannot be boolean"):
        _restore_rng_state({"numpy": ("MT19937", valid_keys, 0, True, 0.0)})
    with pytest.raises(ValueError, match="has_gauss cannot be fractional"):
        _restore_rng_state({"numpy": ("MT19937", valid_keys, 0, 0.5, 0.0)})
    with pytest.raises(ValueError, match="has_gauss must be 0 or 1"):
        _restore_rng_state({"numpy": ("MT19937", valid_keys, 0, 2, 0.0)})


def test_historical_uint32_tensor_restores_without_unsupported_comparison() -> None:
    """Historical uint32 tensors restore cleanly on PyTorch 2.4 without operator crashes."""
    if hasattr(torch, "uint32"):
        legacy_uint32_keys = torch.zeros(624, dtype=torch.uint32)
        # In PyTorch 2.4, (legacy_uint32_keys < 0) raises RuntimeError lt_cpu not implemented
        # _restore_rng_state must validate via CPU NumPy and succeed without error
        _restore_rng_state({"numpy": ("MT19937", legacy_uint32_keys, 0, 0, 0.0)})


def test_scalar_counter_strict_validation(tmp_path: Path) -> None:
    """Checkpoint epoch/step counters reject booleans, negative integers, and non-integrals."""
    model = _SimpleModel()

    with pytest.raises(ValueError, match="epoch must be an integer >= 0"):
        save_checkpoint(tmp_path / "c1.pt", model, epoch=True)

    with pytest.raises(ValueError, match="epoch must be an integer >= 0"):
        save_checkpoint(tmp_path / "c2.pt", model, epoch=-1)

    with pytest.raises(ValueError, match="global_step must be an integer >= 0"):
        save_checkpoint(tmp_path / "c3.pt", model, global_step=False)

    ckpt_path = tmp_path / "valid.pt"
    save_checkpoint(ckpt_path, model, epoch=2, global_step=10, optimizer_step=5)

    loaded = load_checkpoint(ckpt_path)
    assert loaded["epoch"] == 2
    assert loaded["global_step"] == 10
    assert loaded["optimizer_step"] == 5


# ---------------------------------------------------------------------------
# 14. Additional edge probes: bytes/bytearray rejection
# ---------------------------------------------------------------------------

def test_rejection_of_bytes_and_bytearray_with_exact_path(tmp_path: Path) -> None:
    """bytes and bytearray are rejected with exact path errors because PyTorch weights_only cannot load them."""
    with pytest.raises(CheckpointSerializationError) as exc_info1:
        atomic_save_torch({"extra": b"raw_bytes"}, tmp_path / "bytes.pt")
    assert "extra" in str(exc_info1.value)
    assert "bytes and bytearray are not supported" in str(exc_info1.value)

    with pytest.raises(CheckpointSerializationError) as exc_info2:
        atomic_save_torch({"config": {"payload": bytearray(b"raw_bytes")}}, tmp_path / "bytearray.pt")
    assert "config.payload" in str(exc_info2.value)
    assert "bytes and bytearray are not supported" in str(exc_info2.value)


# ---------------------------------------------------------------------------
# 15. Root-level mapping key normalization and collision detection
# ---------------------------------------------------------------------------

def test_root_level_mapping_keys_normalized_and_collision_detected(tmp_path: Path) -> None:
    """Root-level mapping keys are normalized (e.g. np.int64 -> int) and collision-checked."""
    # np.int64 root key normalizes to builtin int and survives weights_only=True
    ckpt_path = tmp_path / "root_np_key.pt"
    atomic_save_torch({np.int64(1): "val"}, ckpt_path)
    loaded = torch.load(ckpt_path, weights_only=True)
    assert 1 in loaded
    assert type(list(loaded.keys())[0]) is int

    # Root key collision detection
    with pytest.raises(CheckpointSerializationError, match="collision"):
        atomic_save_torch({Path("epoch"): 1, "epoch": 2}, tmp_path / "root_collision.pt")


# ---------------------------------------------------------------------------
# 16. Normalized reserved-key override prevention in save_checkpoint
# ---------------------------------------------------------------------------

def test_save_checkpoint_normalizes_extra_before_reserved_key_check(tmp_path: Path) -> None:
    """extra keys are normalized before checking reserved keys, preventing override via Path/np.int."""
    model = _SimpleModel()

    # Path("epoch") normalizes to "epoch", which is reserved
    with pytest.raises(ValueError, match="reserved key"):
        save_checkpoint(tmp_path / "reserved_override.pt", model, epoch=1, extra={Path("epoch"): -999})

    # Path("model") normalizes to "model", which is reserved
    with pytest.raises(ValueError, match="reserved key"):
        save_checkpoint(tmp_path / "reserved_model.pt", model, extra={Path("model"): "fake"})


# ---------------------------------------------------------------------------
# 17. RNG restore: object, complex, and non-real dtypes rejected without lossy cast
# ---------------------------------------------------------------------------

def test_rng_restore_rejects_object_and_complex_dtypes_without_lossy_cast() -> None:
    """_restore_rng_state rejects object, complex, and string dtypes before comparison or cast."""
    # ndarray dtype=object filled with 1.5
    obj_keys = np.array([1.5] * 624, dtype=object)
    with pytest.raises(ValueError, match="must have integer or real float dtype"):
        _restore_rng_state({"numpy": ("MT19937", obj_keys, 0, 0, 0.0)})

    # ndarray dtype=complex filled with 1+2j
    complex_keys = np.array([1 + 2j] * 624, dtype=complex)
    with pytest.raises(ValueError, match="must have integer or real float dtype"):
        _restore_rng_state({"numpy": ("MT19937", complex_keys, 0, 0, 0.0)})

    # ndarray dtype=str
    str_keys = np.array(["1"] * 624)
    with pytest.raises(ValueError, match="must have integer or real float dtype"):
        _restore_rng_state({"numpy": ("MT19937", str_keys, 0, 0, 0.0)})

    # list containing complex number
    complex_list = [1 + 2j] * 624
    with pytest.raises(ValueError, match="cannot be complex"):
        _restore_rng_state({"numpy": ("MT19937", complex_list, 0, 0, 0.0)})


# ---------------------------------------------------------------------------
# 18. Custom Tensor subclasses fail closed with exact path error
# ---------------------------------------------------------------------------

def test_custom_tensor_subclasses_fail_closed_with_exact_path(tmp_path: Path) -> None:
    """Custom Tensor subclasses are rejected before save with exact path errors."""
    class CustomTensor(torch.Tensor):
        pass

    custom_t = torch.ones(2).as_subclass(CustomTensor)
    ckpt_path = tmp_path / "custom_tensor.pt"

    # Top-level / extra
    with pytest.raises(CheckpointSerializationError) as exc_info1:
        atomic_save_torch({"extra": custom_t}, ckpt_path)
    assert "extra" in str(exc_info1.value)
    assert "CustomTensor" in str(exc_info1.value)
    assert "custom tensor subclasses cannot be serialized safely" in str(exc_info1.value)

    # Nested in metadata
    with pytest.raises(CheckpointSerializationError) as exc_info2:
        atomic_save_torch({"config": {"nested": {"tensor": custom_t}}}, ckpt_path)
    assert "config.nested.tensor" in str(exc_info2.value)
    assert "CustomTensor" in str(exc_info2.value)

    # In training tree (e.g. model or rng_state)
    with pytest.raises(CheckpointSerializationError) as exc_info3:
        atomic_save_torch({"model": {"weight": custom_t}}, ckpt_path)
    assert "model.weight" in str(exc_info3.value)
    assert "CustomTensor" in str(exc_info3.value)
