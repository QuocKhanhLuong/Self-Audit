"""Student-only Candidate PG exports and locked TEST transaction helpers."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

EXPORT_SCHEMA = "self_audit.candidate_pg.student_export.v2"
SIDEcar_SCHEMA = "self_audit.candidate_pg.student_export_sidecar.v2"
SIDECAR_SCHEMA = SIDEcar_SCHEMA
FORBIDDEN_PARTS = (
    "fusion", "fusions", "mimic", "mimic_heads", "code_branch", "p_s", "h_s_q",
    "teacher", "teacher_adapter", "codebook_teacher",
)
ARM_NAMES = ("A1", "A2", "A3", "A4")


def _validate_factory(factory: Any) -> None:
    if factory is None:
        return
    from self_audit.candidate_fuse.student import FuseStudent
    if not isinstance(factory, type) or not issubclass(factory, FuseStudent):
        raise TypeError("PG export model_factory must be a FuseStudent subclass")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _key_forbidden(key: str) -> bool:
    parts = {part.lower().replace("-", "_") for part in key.split(".")}
    lowered = key.lower().replace("-", "_")
    return bool(parts & set(FORBIDDEN_PARTS)) or any(
        f".{part}." in f".{lowered}." for part in FORBIDDEN_PARTS
    )


def strip_export_keys(state: Mapping[str, Any], *, assert_no_teacher_keys: bool = True,
                      strip_prefixes: Sequence[str] | None = None) -> dict[str, Any]:
    """Remove PG train/fusion-only state while preserving student weights.

    The source checkpoint may be a full training state.  Keys belonging to
    fusion, mimic, code-branch and teacher modules are excluded; no state is
    silently renamed.  ``assert_no_teacher_keys`` applies to the resulting
    export (and is intentionally strict for deployment).
    """
    if not isinstance(state, Mapping):
        raise TypeError("PG export state must be a mapping")
    configured = tuple(str(v).lower().rstrip(".") for v in (strip_prefixes or ()))
    result: dict[str, Any] = {}
    for original, value in state.items():
        key = str(original)
        lowered = key.lower()
        if configured and any(lowered == prefix or lowered.startswith(prefix + ".")
                              for prefix in configured):
            continue
        if _key_forbidden(key):
            continue
        if key in result:
            raise ValueError(f"duplicate PG export key: {key}")
        if not torch.is_tensor(value):
            raise ValueError(f"PG export state entry {key!r} is not a tensor")
        if not torch.isfinite(value).all():
            raise ValueError(f"PG export state entry {key!r} contains NaN/Inf")
        result[key] = value.detach().cpu().contiguous()
    if not result:
        raise ValueError("PG export contains no student weights")
    if assert_no_teacher_keys and any(_key_forbidden(key) for key in result):
        raise ValueError("PG student export retained forbidden teacher/fusion state")
    return result


def _deployment_model(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> nn.Module | None:
    """Construct the plain FUSE student used by a PG student-only artifact."""
    if metadata.get("deployment_factory") != "self_audit.candidate_fuse.student:FuseStudent":
        raise ValueError("PG export lacks the explicit plain FuseStudent deployment factory")
    from self_audit.candidate_fuse.student import FuseStudent
    channels = metadata.get("teacher_channels")
    phases = metadata.get("teacher_phases")
    teacher_channels = tuple(int(v) for v in channels) if channels is not None else None
    teacher_phases = tuple(tuple((tuple(int(v) for v in level[0]), tuple(float(v) for v in level[1])))
                           for level in phases) if phases is not None else None
    return FuseStudent(config, teacher_channels=teacher_channels, teacher_phases=teacher_phases,
                       include_fusion=False, include_mimic=False)


def _payload(path: Path, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ValueError("PG export artifact is missing")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("PG export SHA-256 mismatch")
    sidecar = _sidecar_path(path)
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("PG export sidecar is missing or invalid") from exc
    if (metadata.get("schema") != SIDEcar_SCHEMA or metadata.get("artifact") != path.name
            or metadata.get("artifact_sha256") != digest
            or metadata.get("artifact_bytes") != path.stat().st_size):
        raise ValueError("PG export sidecar does not identify artifact bytes")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != EXPORT_SCHEMA:
        raise ValueError("invalid PG student export schema")
    state = payload.get("student_state")
    if not isinstance(state, Mapping):
        raise ValueError("PG export lacks student_state")
    return dict(payload), digest


def export_student(model: nn.Module | Mapping[str, Any], config: Mapping[str, Any], path: str | Path,
                   *, epoch: int | None = None, metadata: Mapping[str, Any] | None = None,
                   model_factory: Any = None, strip_prefixes: Sequence[str] | None = None,
                   verify: bool = False) -> dict[str, Any]:
    """Write a strict student-only artifact and byte-identifying sidecar."""
    if not isinstance(config, Mapping):
        raise TypeError("PG export config must be a mapping")
    from self_audit.candidate_fuse.student import FuseStudent
    if not isinstance(model, FuseStudent):
        raise TypeError("PG export requires a FuseStudent/PGStudent model")
    _validate_factory(model_factory)
    source = model.state_dict() if isinstance(model, nn.Module) else model
    state = strip_export_keys(source, strip_prefixes=strip_prefixes)
    export_metadata = dict(metadata or {})
    export_metadata.update({"selected_epoch": epoch,
                            "assert_no_teacher_keys": True,
                            "model_factory": "self_audit.candidate_fuse.student:FuseStudent",
                            "student_only": True})
    export_metadata["training_parameters_total"] = sum(p.numel() for p in model.parameters())
    export_metadata["parameters_total"] = sum(p.numel() for name, p in model.named_parameters() if name in state)
    export_metadata["parameters_trainable"] = sum(p.numel() for name, p in model.named_parameters() if name in state and p.requires_grad)
    export_metadata["deployment_factory"] = "self_audit.candidate_fuse.student:FuseStudent"
    export_metadata["teacher_channels"] = list(getattr(model, "teacher_channels", ()))
    export_metadata["teacher_phases"] = [
        [list(step), list(offset)] for step, offset in getattr(model, "teacher_phases", ())
    ]
    payload = {"schema": EXPORT_SCHEMA, "version": 2, "config": dict(config),
               "student_state": state, "metadata": export_metadata}
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    digest = sha256_file(destination)
    sidecar = {"schema": SIDEcar_SCHEMA, "artifact": destination.name,
               "artifact_sha256": digest, "artifact_bytes": destination.stat().st_size,
               "metadata": export_metadata}
    _sidecar_path(destination).write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    result = {"schema": EXPORT_SCHEMA, "path": str(destination), "sha256": digest,
              "mode": "student_only", "epoch": epoch, "metadata": export_metadata,
              "manifest_path": str(_sidecar_path(destination))}
    if verify:
        # Verify state can be instantiated before handing the receipt to the
        # transaction code.  Full fresh-process equality is explicit below.
        loaded = load_student_export(destination)
        loaded_keys = set(loaded.state_dict())
        if loaded_keys != set(state):
            raise ValueError("PG loaded student state keys differ from exported state")
    return result


def load_student_export(path: str | Path, *, model_factory: Any = None,
                        device: str | torch.device = "cpu",
                        expected_sha256: str | None = None) -> nn.Module:
    _validate_factory(model_factory)
    payload, _ = _payload(Path(path).resolve(), expected_sha256)
    metadata = payload.get("metadata", {})
    model = _deployment_model(payload.get("config", {}), metadata)
    state = payload["student_state"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"PG student deployment state is not exact: {exc}") from exc
    return model.to(device).eval()


def _call_model(model: nn.Module, image: torch.Tensor, geometry: Mapping[str, Any]) -> torch.Tensor:
    if callable(getattr(model, "infer_logits", None)):
        output = model.infer_logits(image, geometry)
    else:
        output = model(image, geometry=geometry)
    if isinstance(output, Mapping):
        for key in ("logits", "student_logits", "student_only_logits"):
            if torch.is_tensor(output.get(key)):
                output = output[key]; break
    if not torch.is_tensor(output):
        raise TypeError("PG student model output lacks logits Tensor")
    return output


def _cold_load_runtime() -> dict[str, Any]:
    """Capture the inference settings that can change deployed logits."""

    return {
        "num_threads": int(torch.get_num_threads()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _configure_cold_load_runtime(runtime: Mapping[str, Any]) -> None:
    """Restore parent inference settings before constructing the deploy model."""

    torch.set_num_threads(int(runtime["num_threads"]))
    torch.use_deterministic_algorithms(
        bool(runtime["deterministic_algorithms"]),
        warn_only=bool(runtime["deterministic_warn_only"]),
    )
    torch.set_float32_matmul_precision(str(runtime["float32_matmul_precision"]))
    torch.backends.cudnn.deterministic = bool(runtime["cudnn_deterministic"])
    torch.backends.cudnn.benchmark = bool(runtime["cudnn_benchmark"])
    torch.backends.cudnn.allow_tf32 = bool(runtime["cudnn_allow_tf32"])
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime["cuda_matmul_allow_tf32"])


def _cold_worker(artifact: str, result_file: str, volumes_file: str, device: str,
                 runtime: Mapping[str, Any] | None = None) -> None:
    if runtime is not None:
        if isinstance(runtime, str):
            runtime = json.loads(runtime)
        if not isinstance(runtime, Mapping):
            raise TypeError("PG cold-load runtime must be a mapping")
        _configure_cold_load_runtime(runtime)
    model = load_student_export(artifact, device=device)
    volumes = torch.load(volumes_file, map_location="cpu", weights_only=True)
    outputs = []
    with torch.inference_mode():
        for image, geometry in volumes:
            outputs.append(_call_model(model, image.to(device), geometry).detach().cpu())
    torch.save({"outputs": outputs, "runtime": _cold_load_runtime()}, result_file)


def verify_cold_load(model: nn.Module, artifact_path: str | Path, volumes: Sequence[tuple[Any, Mapping[str, Any]]],
                     *, model_factory: Any = None, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Compare in-memory, loaded, and fresh-interpreter logits.

    Numerical drift is accepted when the largest absolute logit delta is at
    most ``1e-4`` and the class argmax is unchanged.  Such drift is reported as
    ``WARN`` only when it exceeds that tolerance (or changes an argmax); load,
    subprocess, and provenance failures remain hard errors.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("PG cold-load reference model must be nn.Module")
    _validate_factory(model_factory)
    if not volumes:
        raise ValueError("PG cold-load requires at least one native volume")
    path = Path(artifact_path).resolve()
    loaded = load_student_export(path, device=device)
    expected = []
    loaded_outputs = []
    reference_flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.inference_mode():
            for image, geometry in volumes:
                tensor = torch.as_tensor(image, dtype=torch.float32, device=device)
                reference = _call_model(model, tensor, geometry).detach().cpu()
                actual = _call_model(loaded, tensor, geometry).detach().cpu()
                if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
                    raise ValueError("PG cold-load logits contain NaN/Inf")
                expected.append(reference)
                loaded_outputs.append(actual)
    finally:
        for module, flag in reference_flags:
            module.training = flag

    runtime = _cold_load_runtime()
    with tempfile.TemporaryDirectory(prefix="pg-cold-load-") as directory:
        volume_file = Path(directory) / "volumes.pt"; result_file = Path(directory) / "outputs.pt"
        torch.save([(torch.as_tensor(image).cpu(), dict(geometry)) for image, geometry in volumes], volume_file)
        code = ("from self_audit.candidate_pg.export import _cold_worker;"
                f"_cold_worker({str(path)!r},{str(result_file)!r},{str(volume_file)!r},{str(device)!r},"
                f"{json.dumps(runtime, sort_keys=True)!r})")
        env = os.environ.copy(); env["PYTHONHASHSEED"] = "0"
        if runtime["cublas_workspace_config"] is None:
            env.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            env["CUBLAS_WORKSPACE_CONFIG"] = str(runtime["cublas_workspace_config"])
        completed = subprocess.run([sys.executable, "-c", code], cwd=str(Path(__file__).parents[2]),
                                   env=env, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"PG cold-load subprocess failed: {completed.stderr.strip()}")
        child_result = torch.load(result_file, map_location="cpu", weights_only=True)
    if not isinstance(child_result, Mapping):
        raise RuntimeError("PG cold-load subprocess returned malformed output")
    actual = child_result.get("outputs")
    child_runtime = child_result.get("runtime")
    if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
        raise RuntimeError("PG cold-load subprocess returned malformed logits")
    if not isinstance(child_runtime, Mapping):
        raise RuntimeError("PG cold-load subprocess returned malformed runtime")
    child_runtime = dict(child_runtime)
    if child_runtime != runtime:
        raise RuntimeError("PG cold-load subprocess runtime differs from parent")
    if len(actual) != len(expected):
        raise RuntimeError("PG cold-load subprocess returned the wrong number of volumes")

    comparisons: list[dict[str, Any]] = []
    bitwise_equal = True
    argmax_equal = True
    max_delta = 0.0
    for index, (reference, loaded_output, fresh) in enumerate(zip(expected, loaded_outputs, actual, strict=True)):
        if not torch.is_tensor(fresh) or not torch.isfinite(fresh).all():
            raise RuntimeError(f"PG cold-load subprocess returned invalid logits for volume {index}")
        if tuple(reference.shape) != tuple(loaded_output.shape) or tuple(reference.shape) != tuple(fresh.shape):
            raise RuntimeError(f"PG cold-load logits shape mismatch for volume {index}")
        volume_bitwise = bool(torch.equal(reference, loaded_output) and torch.equal(reference, fresh))
        volume_delta = max(
            float((reference - loaded_output).abs().max().item()),
            float((reference - fresh).abs().max().item()),
        )
        class_dim = 1 if reference.ndim > 1 else 0
        volume_argmax = bool(
            torch.equal(torch.argmax(reference, dim=class_dim), torch.argmax(loaded_output, dim=class_dim))
            and torch.equal(torch.argmax(reference, dim=class_dim), torch.argmax(fresh, dim=class_dim))
        )
        bitwise_equal = bitwise_equal and volume_bitwise
        argmax_equal = argmax_equal and volume_argmax
        max_delta = max(max_delta, volume_delta)
        comparisons.append({"volume": index, "bitwise_equal": volume_bitwise,
                            "max_abs_logit_delta": volume_delta,
                            "argmax_equal": volume_argmax})
    within_tolerance = max_delta <= 1e-4 and argmax_equal
    return {
        "status": "PASS" if within_tolerance else "WARN",
        "bitwise_equal": bitwise_equal,
        "max_abs_logit_delta": max_delta,
        "argmax_equal": argmax_equal,
        "tolerance": 1e-4,
        "within_tolerance": within_tolerance,
        "volumes": len(volumes),
        "sha256": sha256_file(path),
        "mode": "student_only",
        "runtime": {"parent": runtime, "child": child_runtime},
        "per_volume": comparisons,
    }


def recompute_selected_epoch(export_path: str | Path, *, model_factory: Any,
                             records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
                             epoch: int, out_dir: str | Path | None = None) -> dict[str, Any]:
    """Recompute selected VAL/TEST rows from the exact exported bytes."""
    from .evaluation import evaluate_epoch
    model = load_student_export(export_path, model_factory=model_factory)
    result = {split: evaluate_epoch(model, records=records, epoch=epoch, split=split,
                                    path="student_only", out_dir=None)
              for split, records in records_by_split.items() if split in {"VAL18", "TEST50", "VAL", "TEST"}}
    required = {"VAL18", "TEST50"}
    missing = [name for name in required if name not in result and ("VAL" if name == "VAL18" else "TEST") not in result]
    if missing:
        raise ValueError(f"selected export recompute requires VAL18 and TEST50 records; missing {missing}")
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "SELECTED_RECOMPUTE.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def evaluate_all4_transaction(bundle: Mapping[str, Mapping[str, Any]], out: str | Path,
                              evaluator: Any, *, role: str = "test", device: str = "cpu",
                              expected_manifest_sha256: str | None = None,
                              manifest_path: str | Path | None = None,
                              monitoring_receipt: Mapping[str, Any] | str | Path | None = None) -> dict[str, Any]:
    """Create one locked all-four export TEST transaction.

    Monitoring evaluation is intentionally separate: this function only owns
    ``TEST_TRANSACTION.json`` and refuses a second transaction for the same
    output directory.
    """
    if role not in {"test", "TEST"}:
        raise ValueError("PG all4 transaction is reserved for TEST")
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_manifest_sha256.lower())):
        raise ValueError("PG TEST transaction requires the pinned manifest SHA-256")
    if manifest_path is None:
        raise ValueError("PG TEST transaction requires the pinned manifest path")
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("PG TEST transaction manifest SHA-256 mismatch")
    if set(bundle) != set(ARM_NAMES):
        raise ValueError("PG final transaction requires exactly A1/A2/A3/A4 exports")
    root = Path(out).resolve(); root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "TEST_TRANSACTION.json"
    lock_path = root / "TEST_TRANSACTION.json.lock"
    import fcntl
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if ledger_path.exists():
            raise ValueError("PG TEST transaction already exists; refusing rerun")
        artifacts = {}
        models = {}
        for arm in ARM_NAMES:
            receipt = bundle[arm]
            if not isinstance(receipt, Mapping) or not receipt.get("path"):
                raise ValueError(f"PG {arm} receipt requires path")
            path = Path(receipt["path"]).resolve(); digest = sha256_file(path)
            expected = receipt.get("sha256")
            if (not isinstance(expected, str) or len(expected) != 64
                    or any(char not in "0123456789abcdef" for char in expected.lower())):
                raise ValueError(f"PG {arm} receipt requires artifact SHA-256")
            if expected != digest:
                raise ValueError(f"PG {arm} export SHA-256 mismatch")
            _payload(path, digest)
            models[arm] = load_student_export(path, expected_sha256=digest, device=device)
            artifacts[arm] = {"path": str(path), "sha256": digest}
        if monitoring_receipt is None:
            monitoring = {"status": "NOT_RUN_MONITORING_RECEIPT"}
        elif isinstance(monitoring_receipt, Mapping):
            monitoring = dict(monitoring_receipt)
        else:
            monitoring = {"status": "DECLARED", "path": str(monitoring_receipt)}
        ledger = {"schema": "self_audit.candidate_pg.test_transaction.v1", "status": "RUNNING",
                  "role": "test", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "artifacts": artifacts, "manifest": str(manifest_path),
                  "manifest_sha256": expected_manifest_sha256,
                  "monitoring": monitoring}
        _atomic_json(ledger_path, ledger)
        try:
            if not callable(evaluator):
                raise TypeError("PG TEST evaluator must consume loaded student models")
            metrics = evaluator(models, manifest_path=manifest_path, device=device)
            ledger.update({"status": "COMMITTED", "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                           "metrics": metrics})
            _atomic_json(ledger_path, ledger)
            return ledger
        except Exception as exc:
            ledger.update({"status": "FAILED_LOCKED", "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                           "error": f"{type(exc).__name__}: {exc}"})
            _atomic_json(ledger_path, ledger)
            raise


evaluate_exports = evaluate_all4_transaction
verify_student_cold_load = verify_cold_load
export_pg_student = export_student

__all__ = [
    "ARM_NAMES", "EXPORT_SCHEMA", "SIDECAR_SCHEMA", "export_student", "evaluate_all4_transaction",
    "evaluate_exports", "load_student_export", "recompute_selected_epoch", "sha256_file",
    "strip_export_keys", "verify_cold_load",
    "verify_student_cold_load", "export_pg_student",
]
