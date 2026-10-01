"""Fail-fast check that an official run uses a declared, pinned environment.

Environments are declared under ``environments/<environment_id>/environment.json``.
Official validation and experiment entrypoints call
:func:`require_official_environment`; a mismatch in the Python series, a
critical package version or the PyTorch build variant raises
:class:`EnvironmentMismatch`. Setting ``SELF_AUDIT_ALLOW_UNOFFICIAL_ENVIRONMENT=1``
downgrades the failure to a loud warning and marks the run ``official: false``
in its provenance; it never passes silently.

Only installed distribution metadata is inspected, so this module imports no
numerical package and can run before any of them.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS_DIR = REPO_ROOT / "environments"
CANONICAL_ENVIRONMENT = "self-audit-canonical"
ALLOW_UNOFFICIAL_ENV_VAR = "SELF_AUDIT_ALLOW_UNOFFICIAL_ENVIRONMENT"


class EnvironmentMismatch(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(
            f"environment does not match {report['environment_id']} v{report['environment_version']}: "
            + "; ".join(report["problems"])
            + f" (set {ALLOW_UNOFFICIAL_ENV_VAR}=1 only for non-official portability runs)"
        )


def load_environment(environment_id: str = CANONICAL_ENVIRONMENT) -> dict[str, Any]:
    path = ENVIRONMENTS_DIR / environment_id / "environment.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != "self_audit.environment.v1" or spec.get("environment_id") != environment_id:
        raise ValueError(f"invalid environment declaration: {path}")
    return spec


def _installed(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _public(version: str | None) -> str | None:
    return None if version is None else version.split("+", 1)[0]


def _torch_variant(torch_version: str | None) -> str | None:
    if torch_version is None:
        return None
    local = torch_version.split("+", 1)[1] if "+" in torch_version else ""
    if local:
        return local
    requires = importlib.metadata.requires("torch") or []
    runtime = [r for r in requires if r.startswith("nvidia-cuda-runtime-cu12")]
    if any(re.search(r"==\s*12\.1\.", r) for r in runtime):
        return "cu121"
    return "unknown"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_environment(environment_id: str = CANONICAL_ENVIRONMENT, *, variant: str | None = None) -> dict[str, Any]:
    """Compare the running interpreter with a declared environment; never raises."""
    spec = load_environment(environment_id)
    problems: list[str] = []
    series = spec["python"]["series"]
    running = ".".join(str(part) for part in sys.version_info[:3])
    if ".".join(running.split(".")[:2]) != series:
        problems.append(f"Python {running} is not the declared {series} series")
    packages = {}
    for name, expected in spec["critical_packages"].items():
        installed = _installed(name)
        ok = _public(installed) == expected
        packages[name] = {"expected": expected, "installed": installed, "ok": ok}
        if not ok:
            problems.append(f"{name} {installed or 'missing'} != {expected}")
    detected = _torch_variant(_installed("torch"))
    if variant is None:
        variant = detected if detected in spec["variants"] else None
    if variant not in spec["variants"]:
        problems.append(f"PyTorch build variant {detected!r} is not one of {sorted(spec['variants'])}")
    elif detected != variant:
        problems.append(f"PyTorch build variant {detected!r} != requested {variant!r}")
    lock_name = spec["variants"].get(variant, {}).get("lock") if variant else None
    lock_path = ENVIRONMENTS_DIR / environment_id / lock_name if lock_name else None
    return {
        "environment_id": environment_id,
        "environment_version": spec["environment_version"],
        "kind": spec["kind"],
        "variant": variant,
        "lock": None if lock_path is None else str(lock_path.relative_to(REPO_ROOT)),
        "lock_sha256": None if lock_path is None or not lock_path.is_file() else _file_sha256(lock_path),
        "environment_json_sha256": _file_sha256(ENVIRONMENTS_DIR / environment_id / "environment.json"),
        "python": {"version": running, "implementation": platform.python_implementation(),
                   "executable": sys.executable},
        "platform": platform.platform(),
        "critical_packages": packages,
        "torch_variant": detected,
        "problems": problems,
        "matches": not problems,
    }


def require_official_environment(
    environment_id: str = CANONICAL_ENVIRONMENT, *, variant: str | None = None,
) -> dict[str, Any]:
    """Return the environment identity for provenance, or fail fast on a mismatch."""
    report = check_environment(environment_id, variant=variant)
    report["official"] = report["matches"]
    if report["matches"]:
        return report
    if os.environ.get(ALLOW_UNOFFICIAL_ENV_VAR) == "1":
        report["unofficial_override"] = ALLOW_UNOFFICIAL_ENV_VAR
        print(f"WARNING: UNOFFICIAL ENVIRONMENT ({'; '.join(report['problems'])}); "
              "results are not official evidence", file=sys.stderr)
        return report
    raise EnvironmentMismatch(report)
