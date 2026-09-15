"""Runtime services for the mask-free pipeline: atomic IO, identity hashing,
RNG capture, device resolution, metric logging and an optional W&B wrapper.

Nothing here imports the supervised packages. Checkpoint and JSON writing are
re-implemented locally so the mask-free run cannot inherit supervised targets,
stale metric schemas or reference-derived state.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from .progress import current_progress

RUNTIME_SCHEMA_VERSION = "maskfree150.runtime.v1"

_PACKAGE_DIR = Path(__file__).resolve().parent


class RuntimeContractError(RuntimeError):
    """Raised when the runtime cannot honour an identity or IO guarantee."""


# ---------------------------------------------------------------------------
# atomic IO
# ---------------------------------------------------------------------------
def _atomic_replace(tmp_path: Path, path: Path) -> None:
    fd = os.open(tmp_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp_path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _atomic_replace(tmp_path, path)
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    text = json.dumps(payload, indent=indent, sort_keys=True, default=json_default)
    return atomic_write_text(path, text + "\n")


def atomic_save_torch(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".tmp-{os.getpid()}")
    with current_progress().stage("checkpoint.write", path=str(path)):
        torch.save(payload, tmp_path)
        _atomic_replace(tmp_path, path)
    return path


def append_jsonl(path: str | Path, payload: Any) -> Path:
    """Append one record. Not atomic across the whole file by design: the
    per-epoch stream is an append log, and the authoritative state is the
    checkpoint plus the atomically written report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, default=json_default)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


# ---------------------------------------------------------------------------
# hashing and identity
# ---------------------------------------------------------------------------
def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with current_progress().stage("file.sha256", path=str(path)):
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=json_default)
    return sha256_bytes(text.encode("utf-8"))


_SOURCE_SNAPSHOT: dict[str, Any] | None = None


def package_source_hash(*, refresh: bool = False) -> dict[str, Any]:
    """Hash of the mask-free package sources actually executing this run.

    Snapshotted once per process: the code that a process executes is fixed at
    import time, so two trainers in the same process must agree even if the
    working tree is edited underneath them. A new process re-reads the tree, so
    a genuine source change still refuses to resume another build's checkpoint.
    """
    global _SOURCE_SNAPSHOT
    if _SOURCE_SNAPSHOT is not None and not refresh:
        return _SOURCE_SNAPSHOT
    files = sorted(p for p in _PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    per_file = {str(p.relative_to(_PACKAGE_DIR)): sha256_file(p) for p in files}
    _SOURCE_SNAPSHOT = {"files": per_file, "combined": sha256_json(per_file)}
    return _SOURCE_SNAPSHOT


def git_identity(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else _PACKAGE_DIR
    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        return out.stdout.strip()

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "available": commit is not None,
        "dirty": None if status is None else bool(status.strip()),
    }


def environment_identity() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
    }


def source_identity(repo_root: str | Path | None = None) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "git": git_identity(repo_root),
        "package": package_source_hash(),
        "environment": environment_identity(),
    }


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:  # pragma: no cover
        state["numpy"] = None
    state["cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if state.get("python") is not None:
        python_state = state["python"]
        if isinstance(python_state, list):  # survives a JSON round trip
            python_state = (python_state[0], tuple(python_state[1]), python_state[2])
        random.setstate(python_state)
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if state.get("numpy") is not None:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:  # pragma: no cover
            pass
    if state.get("cuda") is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("exact resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def epoch_generator(seed: int, epoch: int) -> torch.Generator:
    """Deterministic per-epoch generator for the sampling-unit permutation."""
    generator = torch.Generator()
    generator.manual_seed((seed * 1_000_003 + epoch * 7919) % (2**63 - 1))
    return generator


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------
def resolve_device(requested: str, *, allow_cpu: bool) -> torch.device:
    requested = str(requested)
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        if not allow_cpu:
            raise RuntimeContractError(
                "requested CUDA device is unavailable; pass allow_cpu=True only for "
                "bounded CPU checks, never for a full 150-epoch run"
            )
        return torch.device("cpu")
    if requested != "cpu":
        return torch.device(requested)
    if not allow_cpu:
        raise RuntimeContractError("device='cpu' requires allow_cpu=True")
    return torch.device("cpu")


def device_identity(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"type": device.type, "name": None, "uuid": None, "index": None,
                "total_memory_bytes": None}
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    uuid = getattr(props, "uuid", None)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if uuid is None and visible and visible.startswith("GPU-") and "," not in visible:
        uuid = visible
    return {
        "type": "cuda",
        "index": int(index),
        "name": props.name,
        "uuid": str(uuid) if uuid is not None else None,
        "total_memory_bytes": int(props.total_memory),
        "capability": f"{props.major}.{props.minor}",
        "visible_devices": visible,
    }


def gpu_stats(device: torch.device) -> dict[str, Any] | None:
    """Peak/current CUDA memory, or None on CPU. Never fabricated."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "reserved_bytes": int(torch.cuda.memory_reserved(index)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
    }


def make_grad_scaler(device: torch.device, *, enabled: bool) -> Any:
    """AMP scaler built through whichever public API this torch exposes.

    The target environment is torch 2.4.1; newer torch prefers
    ``torch.amp.GradScaler(device_type, ...)``. Both paths are public API.
    """
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - torch version dependent
        return torch.cuda.amp.GradScaler(enabled=enabled)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
@dataclass
class TimingAccumulator:
    """Wall-clock seconds per named stage, plus call counts."""

    seconds: dict[str, float]
    calls: dict[str, int]
    device: torch.device | None = None

    @classmethod
    def empty(cls) -> "TimingAccumulator":
        return cls(seconds={}, calls={})

    @contextmanager
    def stage(self, name: str, **details: Any) -> Iterator[None]:
        with current_progress().stage(name, **details):
            if self.device is not None and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            try:
                yield
            finally:
                if self.device is not None and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                elapsed = time.perf_counter() - start
                self.seconds[name] = self.seconds.get(name, 0.0) + elapsed
                self.calls[name] = self.calls.get(name, 0) + 1

    def add(self, name: str, seconds: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + float(seconds)
        self.calls[name] = self.calls.get(name, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {"seconds": dict(self.seconds), "calls": dict(self.calls),
                "cuda_synchronized": self.device is not None and self.device.type == "cuda",
                "aggregation_note": "inclusive stages may overlap; do not sum into end-to-end latency"}


# ---------------------------------------------------------------------------
# optimisation helpers
# ---------------------------------------------------------------------------
def warmup_cosine_multiplier(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """Single global schedule shared by every optimizer in the run."""
    import math

    total_steps = max(int(total_steps), 1)
    step = max(int(step), 0)
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    denominator = max(total_steps - warmup_steps, 1)
    progress = min(max(step - warmup_steps, 0) / denominator, 1.0)
    return float(0.5 * (1.0 + math.cos(math.pi * progress)))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def global_grad_norm(parameters: Any) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        value = param.grad.detach()
        total += float(value.float().norm(2).item() ** 2)
    return float(total**0.5)


def label_ramp(epoch: int, *, ramp_epochs: int = 20) -> float:
    """0.05 + 0.95 * min(1, (epoch+1)/ramp_epochs); recorded every epoch."""
    ramp_epochs = max(int(ramp_epochs), 1)
    return float(0.05 + 0.95 * min(1.0, (int(epoch) + 1) / ramp_epochs))


# ---------------------------------------------------------------------------
# run paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def banks(self) -> Path:
        return self.root / "banks"

    @property
    def last_checkpoint(self) -> Path:
        return self.checkpoints / "last.pt"

    @property
    def epoch_metrics(self) -> Path:
        return self.reports / "epoch_metrics.jsonl"

    @property
    def pipeline_report(self) -> Path:
        return self.reports / "pipeline_report.json"

    @property
    def failure_report(self) -> Path:
        return self.reports / "failure_report.json"

    @property
    def gate_receipt(self) -> Path:
        return self.reports / "gate_receipt.json"

    @property
    def freeze_manifest(self) -> Path:
        return self.exports / "freeze_manifest.json"

    def ensure(self) -> "RunPaths":
        for path in (self.root, self.checkpoints, self.reports, self.exports, self.banks):
            path.mkdir(parents=True, exist_ok=True)
        return self


def write_gate_receipt(path: str | Path, payload: dict[str, Any]) -> Path:
    """Batch/GPU gate receipt consumed by the full launcher before a real run."""
    required = {"status", "device", "physical_batch", "accumulation_steps", "effective_batch"}
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeContractError(f"gate receipt missing fields: {missing}")
    if payload["status"] not in ("pass", "fail"):
        raise RuntimeContractError("gate receipt status must be 'pass' or 'fail'")
    record = dict(payload)
    record["schema_version"] = RUNTIME_SCHEMA_VERSION
    record["written_at"] = time.time()
    return atomic_write_json(path, record)


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------
class MetricSink:
    """One run per dataset with producer/audit/student namespaces.

    W&B is optional. When the package or credentials are missing the sink stays
    local and records the reason instead of failing the run or pretending an
    online run exists.
    """

    def __init__(self, *, project: str, mode: str, run_id: str, config: dict[str, Any],
                 directory: str | Path) -> None:
        self.project = project
        self.mode = mode
        self.run_id = run_id
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.local_path = self.directory / "wandb_stream.jsonl"
        self._run = None
        self.status = "disabled"
        self.reason: str | None = None
        if mode == "disabled":
            self.reason = "wandb_mode=disabled"
            return
        for variable, leaf in (("WANDB_CACHE_DIR", "cache"), ("WANDB_CONFIG_DIR", "config"),
                               ("WANDB_DATA_DIR", "data")):
            destination = self.directory / leaf
            destination.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault(variable, str(destination))
        try:
            import wandb  # type: ignore
        except ImportError:
            self.status = "unavailable"
            self.reason = "wandb package not installed; metrics streamed locally only"
            return
        try:
            self._run = wandb.init(project=project, id=run_id, name=run_id, mode=mode,
                                   dir=str(self.directory), config=config, resume="allow")
            self.status = mode
        except Exception as exc:  # pragma: no cover - depends on credentials
            self._run = None
            self.status = "unavailable"
            self.reason = f"wandb init failed: {exc}"

    def log(self, namespace: str, payload: dict[str, Any], *, step: int | None = None,
            commit: bool = True) -> None:
        flat = {f"{namespace}/{key}": value for key, value in payload.items()}
        append_jsonl(self.local_path, {"step": step, **flat})
        if self._run is not None:
            try:
                self._run.log(flat, step=step, commit=commit)
            except Exception as exc:  # pragma: no cover
                self.reason = f"wandb log failed: {exc}"
                self._run = None
                self.status = "unavailable"

    def summary(self) -> dict[str, Any]:
        return {"project": self.project, "run_id": self.run_id, "mode": self.mode,
                "status": self.status, "reason": self.reason,
                "local_stream": str(self.local_path)}

    def finish(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:  # pragma: no cover
                pass
            self._run = None
