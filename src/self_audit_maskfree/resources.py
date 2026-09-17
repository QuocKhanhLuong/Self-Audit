"""Resource and profiling diagnostics for the mask-free pipeline.

Provides lightweight, read-only resource inspection:
- Cgroup v2 and v1 resolver from /proc/{pid}/cgroup and /proc/{pid}/mountinfo:
  * Strict fail-closed resolution: never reports host/mountpoint limits when target
    subgroup is missing.
  * Mountinfo octal-escape decoding and /proc/{pid}/root namespace resolution.
  * Strict differentiation between unlimited ("max" / -1) and unknown/missing/invalid.
  * Hierarchical quota resolution: distinguishes leaf quota from a visible upper bound
    by evaluating visible ancestor cgroup limits, cpuset, and process affinity.
- Safe process metrics (/proc/{pid}/status, /proc/{pid}/stat, /proc/{pid}/smaps_rollup,
  /proc/{pid}/io) with starttime identity to guard against PID reuse.
- Filesystem inspection resolving mount source, fstype, and options via mountinfo,
  with strict isolation for /dev/shm (never reports parent device capacity if unmounted).
- Bounded GPU sampling with driver version, graceful per-field N/A handling,
  and tracking of all competing compute applications.
- Delta computations across intervals with starttime PID-reuse verification.
- Explicit null + reason returns on macOS / non-Linux / unavailable subsystems.
"""
from __future__ import annotations

import datetime
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

RESOURCE_SCHEMA_VERSION = "maskfree150.resources.v2"


# ---------------------------------------------------------------------------
# String, octal-escape, and parsing helpers
# ---------------------------------------------------------------------------
def _decode_mountinfo_escapes(s: str) -> str:
    """Decode octal escape sequences used in Linux /proc/mountinfo (e.g. \\040 for space)."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), s)


def _safe_read_text(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError, UnicodeDecodeError):
        return None


def _safe_read_lines(path: Path) -> list[str]:
    text = _safe_read_text(path)
    if text is None:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_kv_lines(text: str, delimiter: str = ":") -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or delimiter not in line:
            continue
        key, val = line.split(delimiter, 1)
        result[key.strip()] = val.strip()
    return result


def _parse_space_kv_lines(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            result[parts[0]] = parts[1]
        elif len(parts) == 1:
            result[parts[0]] = ""
    return result


def _parse_kb_to_bytes(val_str: str | None) -> int | None:
    if not val_str:
        return None
    val_str = val_str.strip()
    match = re.match(r"^(\d+)\s*(?:kB|KB|kb)?$", val_str)
    if match:
        return int(match.group(1)) * 1024
    try:
        return int(val_str)
    except ValueError:
        return None


def _parse_float_or_none(val_str: str | None) -> float | None:
    if not val_str:
        return None
    val_str = val_str.strip()
    if val_str.upper() in ("N/A", "[NOT SUPPORTED]", "NONE", "NULL", ""):
        return None
    try:
        return float(val_str)
    except ValueError:
        return None


def _parse_int_or_none(val_str: str | None) -> int | None:
    if not val_str:
        return None
    val_str = val_str.strip()
    if val_str.upper() in ("N/A", "[NOT SUPPORTED]", "NONE", "NULL", ""):
        return None
    try:
        return int(val_str)
    except ValueError:
        return None


def _parse_cpuset_range(spec: str) -> dict[str, Any]:
    spec = spec.strip()
    if not spec:
        return {"raw": "", "count": 0, "cores": []}
    cores: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            sub = part.split("-", 1)
            try:
                start, end = int(sub[0]), int(sub[1])
                cores.update(range(start, end + 1))
            except ValueError:
                pass
        else:
            try:
                cores.add(int(part))
            except ValueError:
                pass
    sorted_cores = sorted(cores)
    return {
        "raw": spec,
        "count": len(sorted_cores),
        "cores": sorted_cores,
    }


def _parse_psi_block(text: str | None) -> dict[str, Any]:
    if not text:
        return {"available": False, "reason": "PSI file empty or unreadable"}
    result: dict[str, Any] = {"available": True}
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        kind = parts[0]
        metrics: dict[str, float | int] = {}
        for item in parts[1:]:
            if "=" in item:
                k, v = item.split("=", 1)
                try:
                    if k == "total":
                        metrics[k] = int(v)
                    else:
                        metrics[k] = float(v)
                except ValueError:
                    pass
        result[kind] = metrics
    return result


# ---------------------------------------------------------------------------
# Mountinfo parser (strictly bound to target pid)
# ---------------------------------------------------------------------------
def _parse_target_mountinfo(proc_root: Path, pid: int) -> list[dict[str, Any]]:
    """Parse /proc/{pid}/mountinfo strictly without fallback to self."""
    mountinfo_path = proc_root / str(pid) / "mountinfo"
    lines = _safe_read_lines(mountinfo_path)
    if not lines:
        return []

    mounts: list[dict[str, Any]] = []
    for line in lines:
        parts = line.split(" - ")
        if len(parts) != 2:
            continue
        left = parts[0].split()
        right = parts[1].split()
        if len(left) < 6 or len(right) < 2:
            continue

        fstype = right[0]
        mount_source = _decode_mountinfo_escapes(right[1])
        super_options = right[2] if len(right) > 2 else ""

        mountroot = _decode_mountinfo_escapes(left[3])
        mountpoint = _decode_mountinfo_escapes(left[4])
        options = left[5]

        mounts.append({
            "mount_id": left[0],
            "parent_id": left[1],
            "major_minor": left[2],
            "mountroot": mountroot,
            "mountpoint": mountpoint,
            "options": options,
            "fstype": fstype,
            "mount_source": mount_source,
            "super_options": super_options,
        })
    return mounts


# ---------------------------------------------------------------------------
# Strict fail-closed cgroup directory resolution
# ---------------------------------------------------------------------------
def _resolve_cgroup_dir_strict(
    cgroup_path_str: str,
    mounts: list[dict[str, Any]],
    expected_fstype: str,
    subsystem: str | None = None,
    proc_root: Path = Path("/proc"),
    pid: int = 1,
    default_sys_root: Path = Path("/sys/fs/cgroup"),
) -> tuple[Path | None, Path | None, str | None]:
    """Strictly resolve target cgroup directory without leaking mountpoint.

    Returns: (target_cgroup_dir, mountpoint_dir, error_reason)
    """
    cgroup_path_str = cgroup_path_str.strip()
    norm_cgroup = "/" + cgroup_path_str.strip("/")

    # Check if /proc/{pid}/root is accessible for cross-namespace translation
    target_root = proc_root / str(pid) / "root"
    base_prefix = target_root if target_root.is_dir() else None
    if proc_root == Path("/proc") and base_prefix is None:
        return None, None, "Target mount namespace root is inaccessible; refusing host-path fallback"

    # Filter matching mounts
    candidate_mounts: list[dict[str, Any]] = []
    for mount in mounts:
        fstype = mount["fstype"]
        if expected_fstype == "cgroup2" and fstype != "cgroup2":
            continue
        if expected_fstype == "cgroup":
            if fstype != "cgroup":
                continue
            opts = mount["super_options"].split(",")
            mp_name = Path(mount["mountpoint"]).name
            if subsystem and (subsystem not in opts and subsystem not in mp_name):
                continue
        candidate_mounts.append(mount)

    for mount in candidate_mounts:
        mountroot = "/" + mount["mountroot"].strip("/")
        raw_mp = Path(mount["mountpoint"])
        mountpoint = (base_prefix / raw_mp.relative_to("/")) if (base_prefix and raw_mp.is_absolute()) else raw_mp

        # Handle mountroot translation
        if norm_cgroup == mountroot:
            if mountpoint.is_dir():
                return mountpoint, mountpoint, None
        elif mountroot == "/" or norm_cgroup.startswith(mountroot + "/"):
            rel = norm_cgroup.lstrip("/") if mountroot == "/" else norm_cgroup[len(mountroot):].lstrip("/")
            candidate = mountpoint / rel
            if candidate.is_dir():
                return candidate, mountpoint, None
            return None, mountpoint, f"target cgroup subgroup {rel} not found under mountpoint {mountpoint}"

    # FAIL CLOSED: Do not return parent mountpoint or sysroot
    return None, None, f"cgroup path {cgroup_path_str} could not be strictly resolved to an existing directory"


# ---------------------------------------------------------------------------
# Cgroup v2 Resolver with Ancestor Hierarchy and Unknown!=Unlimited
# ---------------------------------------------------------------------------
def _parse_cgroup_v2_cpu_max(cpu_max_text: str | None) -> dict[str, Any]:
    if not cpu_max_text:
        return {
            "quota_us": None,
            "period_us": None,
            "quota_cores": None,
            "unlimited": False,
            "valid": False,
            "reason": "cpu.max missing or empty",
        }
    parts = cpu_max_text.strip().split()
    if len(parts) < 2:
        return {
            "quota_us": None,
            "period_us": None,
            "quota_cores": None,
            "unlimited": False,
            "valid": False,
            "reason": f"malformed cpu.max line: {cpu_max_text!r}",
        }

    quota_str, period_str = parts[0], parts[1]
    try:
        period_us = int(period_str)
        if period_us <= 0:
            return {
                "quota_us": None,
                "period_us": period_us,
                "quota_cores": None,
                "unlimited": False,
                "valid": False,
                "reason": f"invalid non-positive period: {period_us}",
            }
    except ValueError:
        return {
            "quota_us": None,
            "period_us": None,
            "quota_cores": None,
            "unlimited": False,
            "valid": False,
            "reason": f"invalid period format: {period_str}",
        }

    if quota_str == "max":
        return {
            "quota_us": None,
            "period_us": period_us,
            "quota_cores": None,
            "unlimited": True,
            "valid": True,
            "reason": None,
        }

    try:
        quota_us = int(quota_str)
        if quota_us <= 0:
            return {
                "quota_us": quota_us,
                "period_us": period_us,
                "quota_cores": None,
                "unlimited": False,
                "valid": False,
                "reason": f"invalid non-positive quota: {quota_us}",
            }
        quota_cores = round(quota_us / period_us, 4)
        return {
            "quota_us": quota_us,
            "period_us": period_us,
            "quota_cores": quota_cores,
            "unlimited": False,
            "valid": True,
            "reason": None,
        }
    except ValueError:
        return {
            "quota_us": None,
            "period_us": period_us,
            "quota_cores": None,
            "unlimited": False,
            "valid": False,
            "reason": f"invalid quota format: {quota_str}",
        }


def _resolve_cgroup_v2(
    target_dir: Path,
    mountpoint: Path,
    cgroup_rel_path: str,
) -> dict[str, Any]:
    leaf_cpu = _parse_cgroup_v2_cpu_max(_safe_read_text(target_dir / "cpu.max"))

    # Inspect ancestor cgroups up to mountpoint to find stricter parent quotas
    ancestor_quotas: list[dict[str, Any]] = []
    curr = target_dir.parent
    while curr != curr.parent and (curr == mountpoint or curr.is_relative_to(mountpoint)):
        parent_cpu_file = curr / "cpu.max"
        if parent_cpu_file.is_file():
            parent_parsed = _parse_cgroup_v2_cpu_max(_safe_read_text(parent_cpu_file))
            if parent_parsed["valid"]:
                ancestor_quotas.append({
                    "dir": str(curr),
                    "quota_cores": parent_parsed["quota_cores"],
                    "unlimited": parent_parsed["unlimited"],
                })
        if curr == mountpoint:
            break
        curr = curr.parent

    # Cpuset
    cpuset_text = _safe_read_text(target_dir / "cpuset.cpus.effective") or _safe_read_text(target_dir / "cpuset.cpus")
    cpuset_info = _parse_cpuset_range(cpuset_text) if cpuset_text else {"raw": None, "count": None, "cores": []}

    # Effective cores computation
    finite_constraints: list[float] = []
    if leaf_cpu["valid"] and leaf_cpu["quota_cores"] is not None:
        finite_constraints.append(leaf_cpu["quota_cores"])
    for anc in ancestor_quotas:
        if anc["quota_cores"] is not None:
            finite_constraints.append(anc["quota_cores"])
    if cpuset_info["count"] is not None and cpuset_info["count"] > 0:
        finite_constraints.append(float(cpuset_info["count"]))

    effective_cores = min(finite_constraints) if finite_constraints else None

    # CPU stat
    cpu_stat_raw = _safe_read_text(target_dir / "cpu.stat")
    if cpu_stat_raw:
        parsed_stat = _parse_space_kv_lines(cpu_stat_raw)
        cpu_stat: dict[str, Any] | None = {
            "usage_usec": _parse_int_or_none(parsed_stat.get("usage_usec")),
            "user_usec": _parse_int_or_none(parsed_stat.get("user_usec")),
            "system_usec": _parse_int_or_none(parsed_stat.get("system_usec")),
            "nr_periods": _parse_int_or_none(parsed_stat.get("nr_periods")),
            "nr_throttled": _parse_int_or_none(parsed_stat.get("nr_throttled")),
            "throttled_usec": _parse_int_or_none(parsed_stat.get("throttled_usec")),
        }
    else:
        cpu_stat = None

    # Memory limits (strict: unknown != unlimited)
    mem_cur = _parse_int_or_none(_safe_read_text(target_dir / "memory.current"))
    mem_max_raw = _safe_read_text(target_dir / "memory.max")
    if mem_max_raw:
        val = mem_max_raw.strip()
        if val == "max":
            mem_max = None
            mem_unlimited: bool | None = True
            mem_reason = None
        elif val.isdigit():
            mem_max = int(val)
            mem_unlimited = False
            mem_reason = None
        else:
            mem_max = None
            mem_unlimited = None
            mem_reason = f"malformed memory.max: {val}"
    else:
        mem_max = None
        mem_unlimited = None
        mem_reason = "memory.max missing or unreadable"

    mem_high_raw = _safe_read_text(target_dir / "memory.high")
    mem_high = None
    if mem_high_raw and mem_high_raw.strip() != "max":
        mem_high = _parse_int_or_none(mem_high_raw)

    # Memory events
    mem_events_raw = _safe_read_text(target_dir / "memory.events")
    mem_events = {k: _parse_int_or_none(v) for k, v in _parse_space_kv_lines(mem_events_raw).items()} if mem_events_raw else None

    # Memory stat
    mem_stat_raw = _safe_read_text(target_dir / "memory.stat")
    mem_stat = {k: _parse_int_or_none(v) for k, v in _parse_space_kv_lines(mem_stat_raw).items()} if mem_stat_raw else None

    return {
        "version": 2,
        "available": True,
        "cgroup_path": cgroup_rel_path,
        "resolved_dir": str(target_dir),
        "leaf_cpu": leaf_cpu,
        "ancestor_quotas": ancestor_quotas,
        "visible_cpu_upper_bound_cores": effective_cores,
        "effective_quota_cores": None,
        "effective_quota_reason": "Hidden ancestors and sibling contention cannot be established from this namespace; visible limits are upper bounds.",
        "cpu_stat": cpu_stat,
        "cpuset": cpuset_info,
        "memory_current_bytes": mem_cur,
        "memory_max_bytes": mem_max,
        "memory_unlimited": mem_unlimited,
        "memory_high_bytes": mem_high,
        "memory_reason": mem_reason,
        "memory_events": mem_events,
        "memory_stat": mem_stat,
        "anon_bytes": mem_stat.get("anon") if mem_stat else None,
        "file_bytes": mem_stat.get("file") if mem_stat else None,
        "cpu_pressure": _parse_psi_block(_safe_read_text(target_dir / "cpu.pressure")),
        "memory_pressure": _parse_psi_block(_safe_read_text(target_dir / "memory.pressure")),
        "io_pressure": _parse_psi_block(_safe_read_text(target_dir / "io.pressure")),
    }


# ---------------------------------------------------------------------------
# Cgroup v1 Resolver
# ---------------------------------------------------------------------------
def _resolve_cgroup_v1(
    cgroup_paths: dict[str, str],
    mounts: list[dict[str, Any]],
    proc_root: Path,
    pid: int,
    default_sys_root: Path,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "version": 1,
        "available": True,
    }

    # Resolve CPU
    cpu_path = cgroup_paths.get("cpu") or cgroup_paths.get("cpu,cpuacct") or "/"
    cpu_dir, _, err = _resolve_cgroup_dir_strict(
        cpu_path, mounts, "cgroup", subsystem="cpu",
        proc_root=proc_root, pid=pid, default_sys_root=default_sys_root
    )
    if cpu_dir and cpu_dir.is_dir():
        quota_raw = _safe_read_text(cpu_dir / "cpu.cfs_quota_us")
        period_raw = _safe_read_text(cpu_dir / "cpu.cfs_period_us")
        if quota_raw and period_raw:
            try:
                quota_us = int(quota_raw.strip())
                period_us = int(period_raw.strip())
                if period_us <= 0:
                    info["leaf_cpu"] = {"quota_us": None, "period_us": period_us, "quota_cores": None, "unlimited": False, "valid": False, "reason": "period <= 0"}
                elif quota_us == -1:
                    info["leaf_cpu"] = {"quota_us": None, "period_us": period_us, "quota_cores": None, "unlimited": True, "valid": True, "reason": None}
                elif quota_us > 0:
                    info["leaf_cpu"] = {"quota_us": quota_us, "period_us": period_us, "quota_cores": round(quota_us / period_us, 4), "unlimited": False, "valid": True, "reason": None}
                else:
                    info["leaf_cpu"] = {"quota_us": quota_us, "period_us": period_us, "quota_cores": None, "unlimited": False, "valid": False, "reason": "invalid quota"}
            except ValueError:
                info["leaf_cpu"] = {"quota_us": None, "period_us": None, "quota_cores": None, "unlimited": False, "valid": False, "reason": "unparseable cpu.cfs values"}
        else:
            info["leaf_cpu"] = {"quota_us": None, "period_us": None, "quota_cores": None, "unlimited": False, "valid": False, "reason": "cfs files missing"}

        stat_raw = _safe_read_text(cpu_dir / "cpu.stat")
        if stat_raw:
            stat_parsed = _parse_space_kv_lines(stat_raw)
            throttled_time_ns = _parse_int_or_none(stat_parsed.get("throttled_time"))
            info["cpu_stat"] = {
                "nr_periods": _parse_int_or_none(stat_parsed.get("nr_periods")),
                "nr_throttled": _parse_int_or_none(stat_parsed.get("nr_throttled")),
                "throttled_usec": (throttled_time_ns // 1000) if throttled_time_ns is not None else None,
            }
        else:
            info["cpu_stat"] = None
    else:
        info["leaf_cpu"] = {"quota_us": None, "period_us": None, "quota_cores": None, "unlimited": False, "valid": False, "reason": err or "cpu cgroup dir not found"}
        info["cpu_stat"] = None

    # Cpuset
    cpuset_path = cgroup_paths.get("cpuset") or "/"
    cpuset_dir, _, _ = _resolve_cgroup_dir_strict(
        cpuset_path, mounts, "cgroup", subsystem="cpuset",
        proc_root=proc_root, pid=pid, default_sys_root=default_sys_root
    )
    if cpuset_dir and cpuset_dir.is_dir():
        cpuset_text = _safe_read_text(cpuset_dir / "cpuset.cpus")
        info["cpuset"] = _parse_cpuset_range(cpuset_text) if cpuset_text else {"raw": None, "count": None, "cores": []}
    else:
        info["cpuset"] = {"raw": None, "count": None, "cores": []}

    # Effective cores (leaf vs cpuset)
    finite_c: list[float] = []
    if info["leaf_cpu"].get("valid") and info["leaf_cpu"].get("quota_cores") is not None:
        finite_c.append(info["leaf_cpu"]["quota_cores"])
    if info["cpuset"]["count"] is not None and info["cpuset"]["count"] > 0:
        finite_c.append(float(info["cpuset"]["count"]))
    info["visible_cpu_upper_bound_cores"] = min(finite_c) if finite_c else None
    info["effective_quota_cores"] = None
    info["effective_quota_reason"] = "v1 leaf/cpuset upper bound only; parent quota hierarchy not evaluated"

    # Memory
    mem_path = cgroup_paths.get("memory") or "/"
    mem_dir, _, mem_err = _resolve_cgroup_dir_strict(
        mem_path, mounts, "cgroup", subsystem="memory",
        proc_root=proc_root, pid=pid, default_sys_root=default_sys_root
    )
    if mem_dir and mem_dir.is_dir():
        info["memory_current_bytes"] = _parse_int_or_none(_safe_read_text(mem_dir / "memory.usage_in_bytes"))
        limit_val = _parse_int_or_none(_safe_read_text(mem_dir / "memory.limit_in_bytes"))
        if limit_val is not None:
            if limit_val >= (1 << 60) or limit_val >= 9223372036854771712:
                info["memory_max_bytes"] = None
                info["memory_unlimited"] = True
                info["memory_reason"] = None
            else:
                info["memory_max_bytes"] = limit_val
                info["memory_unlimited"] = False
                info["memory_reason"] = None
        else:
            info["memory_max_bytes"] = None
            info["memory_unlimited"] = None
            info["memory_reason"] = "memory.limit_in_bytes missing or invalid"

        stat_raw = _safe_read_text(mem_dir / "memory.stat")
        if stat_raw:
            stat_parsed = _parse_space_kv_lines(stat_raw)
            info["memory_stat"] = {k: _parse_int_or_none(v) for k, v in stat_parsed.items()}
            info["anon_bytes"] = info["memory_stat"].get("total_rss") or info["memory_stat"].get("rss")
            info["file_bytes"] = info["memory_stat"].get("total_cache") or info["memory_stat"].get("cache")
        else:
            info["memory_stat"] = None
            info["anon_bytes"] = None
            info["file_bytes"] = None
    else:
        info["memory_current_bytes"] = None
        info["memory_max_bytes"] = None
        info["memory_unlimited"] = None
        info["memory_reason"] = mem_err or "memory cgroup dir not found"
        info["memory_stat"] = None
        info["anon_bytes"] = None
        info["file_bytes"] = None

    return info


# ---------------------------------------------------------------------------
# Master Cgroup Resolver
# ---------------------------------------------------------------------------
def resolve_cgroup(
    pid: int,
    proc_root: Path = Path("/proc"),
    sys_cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, Any]:
    cgroup_file = proc_root / str(pid) / "cgroup"
    lines = _safe_read_lines(cgroup_file)
    if not lines:
        return {
            "available": False,
            "reason": f"cgroup file unreadable or not present at {cgroup_file}",
            "version": None,
        }

    mounts = _parse_target_mountinfo(proc_root, pid)

    # Distinguish v2 vs v1
    v2_line = None
    v1_subs: dict[str, str] = {}
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3:
            hierarchy_id, subs, path = parts[0], parts[1], parts[2]
            if hierarchy_id == "0" and subs == "":
                v2_line = path
            else:
                for sub in subs.split(","):
                    v1_subs[sub] = path

    # Process affinity
    affinity_info: dict[str, Any] = {}
    if hasattr(os, "sched_getaffinity"):
        try:
            aff = sorted(os.sched_getaffinity(pid))  # type: ignore[attr-defined]
            affinity_info = {"count": len(aff), "cores": aff}
        except (OSError, PermissionError) as exc:
            affinity_info = {"count": None, "cores": [], "reason": str(exc)}
    else:
        affinity_info = {"count": None, "cores": [], "reason": "os.sched_getaffinity unavailable"}

    if v2_line is not None:
        target_dir, mountpoint, err = _resolve_cgroup_dir_strict(
            v2_line, mounts, "cgroup2",
            proc_root=proc_root, pid=pid, default_sys_root=sys_cgroup_root
        )
        if target_dir and mountpoint and target_dir.is_dir():
            res = _resolve_cgroup_v2(target_dir, mountpoint, v2_line)
            res["affinity"] = affinity_info
            # Further refine effective quota with process affinity if available
            if res.get("visible_cpu_upper_bound_cores") is not None and affinity_info.get("count"):
                res["visible_cpu_upper_bound_cores"] = min(res["visible_cpu_upper_bound_cores"], float(affinity_info["count"]))
            return res
        return {
            "available": False,
            "reason": err or f"target cgroup v2 dir for {v2_line} does not exist",
            "version": 2,
            "cgroup_path": v2_line,
            "affinity": affinity_info,
        }

    if v1_subs:
        res = _resolve_cgroup_v1(v1_subs, mounts, proc_root=proc_root, pid=pid, default_sys_root=sys_cgroup_root)
        res["affinity"] = affinity_info
        if res.get("visible_cpu_upper_bound_cores") is not None and affinity_info.get("count"):
            res["visible_cpu_upper_bound_cores"] = min(res["visible_cpu_upper_bound_cores"], float(affinity_info["count"]))
        return res

    return {
        "available": False,
        "reason": "cgroup entries found but could not match v1 or v2 hierarchy",
        "version": None,
        "affinity": affinity_info,
    }


# ---------------------------------------------------------------------------
# Process-Level Metrics (/proc/{pid}) with StartTime Identity Guard
# ---------------------------------------------------------------------------
def resolve_process_metrics(
    pid: int,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    proc_dir = proc_root / str(pid)
    if not proc_dir.is_dir():
        return {
            "available": False,
            "reason": f"procfs directory not found at {proc_dir}",
        }

    # /proc/{pid}/stat for starttime, ticks, and faults
    stat_line = _safe_read_text(proc_dir / "stat")
    stat_info: dict[str, Any] = {"available": False}
    start_time_ticks = None
    if stat_line:
        rparen = stat_line.rfind(")")
        if rparen != -1 and rparen + 2 < len(stat_line):
            fields = stat_line[rparen + 2:].split()
            # fields: [0]=state(3), [7]=minflt(10), [9]=majflt(12), [11]=utime(14),
            # [12]=stime(15), [17]=num_threads(20), [19]=starttime(22)
            try:
                start_time_ticks = int(fields[19])
                stat_info = {
                    "available": True,
                    "minflt": int(fields[7]),
                    "majflt": int(fields[9]),
                    "utime_ticks": int(fields[11]),
                    "stime_ticks": int(fields[12]),
                    "total_cpu_ticks": int(fields[11]) + int(fields[12]),
                    "num_threads": int(fields[17]),
                    "starttime_ticks": start_time_ticks,
                }
            except (IndexError, ValueError) as exc:
                stat_info = {"available": False, "reason": f"stat parse error: {exc}"}

    status_text = _safe_read_text(proc_dir / "status")
    parsed_status = _parse_kv_lines(status_text) if status_text else {}

    # smaps_rollup
    smaps_text = _safe_read_text(proc_dir / "smaps_rollup")
    if smaps_text:
        parsed_smaps = _parse_kv_lines(smaps_text)
        smaps_info: dict[str, Any] = {
            "available": True,
            "rss_bytes": _parse_kb_to_bytes(parsed_smaps.get("Rss")),
            "pss_bytes": _parse_kb_to_bytes(parsed_smaps.get("Pss")),
            "pss_anon_bytes": _parse_kb_to_bytes(parsed_smaps.get("Pss_Anon")),
            "pss_file_bytes": _parse_kb_to_bytes(parsed_smaps.get("Pss_File")),
            "pss_shmem_bytes": _parse_kb_to_bytes(parsed_smaps.get("Pss_Shmem")),
            "private_clean_bytes": _parse_kb_to_bytes(parsed_smaps.get("Private_Clean")),
            "private_dirty_bytes": _parse_kb_to_bytes(parsed_smaps.get("Private_Dirty")),
            "referenced_bytes": _parse_kb_to_bytes(parsed_smaps.get("Referenced")),
            "anonymous_bytes": _parse_kb_to_bytes(parsed_smaps.get("Anonymous")),
            "swap_bytes": _parse_kb_to_bytes(parsed_smaps.get("Swap")),
            "swap_pss_bytes": _parse_kb_to_bytes(parsed_smaps.get("SwapPss")),
        }
    else:
        smaps_info = {"available": False, "reason": "smaps_rollup unreadable"}

    # io counters
    io_text = _safe_read_text(proc_dir / "io")
    if io_text:
        parsed_io = _parse_kv_lines(io_text)
        io_info: dict[str, Any] = {
            "available": True,
            "rchar": _parse_int_or_none(parsed_io.get("rchar")),
            "wchar": _parse_int_or_none(parsed_io.get("wchar")),
            "syscr": _parse_int_or_none(parsed_io.get("syscr")),
            "syscw": _parse_int_or_none(parsed_io.get("syscw")),
            "read_bytes": _parse_int_or_none(parsed_io.get("read_bytes")),
            "write_bytes": _parse_int_or_none(parsed_io.get("write_bytes")),
            "cancelled_write_bytes": _parse_int_or_none(parsed_io.get("cancelled_write_bytes")),
        }
    else:
        io_info = {"available": False, "reason": "io unreadable"}

    return {
        "available": True,
        "pid": pid,
        "name": parsed_status.get("Name"),
        "state": parsed_status.get("State"),
        "starttime_ticks": start_time_ticks,
        "threads": _parse_int_or_none(parsed_status.get("Threads")),
        "vm_peak_bytes": _parse_kb_to_bytes(parsed_status.get("VmPeak")),
        "vm_size_bytes": _parse_kb_to_bytes(parsed_status.get("VmSize")),
        "vm_hwm_bytes": _parse_kb_to_bytes(parsed_status.get("VmHWM")),
        "vm_rss_bytes": _parse_kb_to_bytes(parsed_status.get("VmRSS")),
        "rss_anon_bytes": _parse_kb_to_bytes(parsed_status.get("RssAnon")),
        "rss_file_bytes": _parse_kb_to_bytes(parsed_status.get("RssFile")),
        "rss_shmem_bytes": _parse_kb_to_bytes(parsed_status.get("RssShmem")),
        "vm_swap_bytes": _parse_kb_to_bytes(parsed_status.get("VmSwap")),
        "stat": stat_info,
        "smaps_rollup": smaps_info,
        "io": io_info,
    }


# ---------------------------------------------------------------------------
# Filesystem Resolution with Mount Source and /dev/shm Isolation
# ---------------------------------------------------------------------------
def _find_mount_for_path(
    resolved_path: Path,
    mounts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the most specific (longest matching) mountpoint for a given path."""
    path_str = str(resolved_path)
    best_match = None
    best_len = -1
    for m in mounts:
        mp = m["mountpoint"]
        if path_str == mp or path_str.startswith(mp.rstrip("/") + "/"):
            if len(mp) > best_len:
                best_len = len(mp)
                best_match = m
    return best_match


def _inspect_path_filesystem(
    target_path: Path,
    mounts: list[dict[str, Any]],
    is_shm: bool = False,
) -> dict[str, Any]:
    try:
        resolved = target_path.resolve()
        exists = resolved.exists()
        matched_mount = _find_mount_for_path(resolved, mounts)

        # STRICT /dev/shm isolation: must be an independent tmpfs/shm mount!
        if is_shm:
            if not exists:
                return {
                    "path": str(target_path),
                    "exists": False,
                    "available": False,
                    "reason": "/dev/shm is not an independent tmpfs mount (does not exist)",
                }
            if matched_mount is None or matched_mount["fstype"] != "tmpfs" or matched_mount["mountpoint"] != str(resolved):
                return {
                    "path": str(target_path),
                    "exists": True,
                    "available": False,
                    "reason": "/dev/shm is not an independent tmpfs mount; refusing to report parent capacity",
                }

        info: dict[str, Any] = {
            "path": str(target_path),
            "resolved_path": str(resolved),
            "exists": exists,
            "mount_source": matched_mount["mount_source"] if matched_mount else None,
            "mount_point": matched_mount["mountpoint"] if matched_mount else None,
            "fstype": matched_mount["fstype"] if matched_mount else None,
            "mount_options": matched_mount["options"] if matched_mount else None,
        }

        check_path = resolved if exists else resolved.parent
        while not check_path.exists() and check_path != check_path.parent:
            check_path = check_path.parent

        if check_path.exists():
            st = os.statvfs(check_path)
            total_bytes = st.f_blocks * st.f_frsize
            free_bytes = st.f_bfree * st.f_frsize
            avail_bytes = st.f_bavail * st.f_frsize
            used_bytes = total_bytes - free_bytes
            info.update({
                "statvfs_available": True,
                "total_bytes": total_bytes,
                "free_bytes": free_bytes,
                "available_bytes": avail_bytes,
                "used_bytes": used_bytes,
                "used_pct": round((used_bytes / total_bytes * 100), 2) if total_bytes > 0 else 0.0,
                "total_inodes": st.f_files,
                "free_inodes": st.f_ffree,
                "available_inodes": st.f_favail,
            })
        else:
            info["statvfs_available"] = False
            info["reason"] = "path and parent directories do not exist"
        return info
    except (OSError, PermissionError) as exc:
        return {
            "path": str(target_path),
            "exists": False,
            "statvfs_available": False,
            "reason": f"statvfs error: {type(exc).__name__}: {exc}",
        }


def resolve_filesystems(
    data_roots: Sequence[str | Path] = (),
    report_dir: str | Path | None = None,
    proc_root: Path = Path("/proc"),
    pid: int = 1,
    include_shm: bool = True,
    include_tmp: bool = True,
    shm_path: Path = Path("/dev/shm"),
) -> dict[str, Any]:
    if proc_root == Path("/proc") and pid != os.getpid() and platform.system() == "Linux":
        try:
            same_namespace = (proc_root / str(pid) / "ns/mnt").stat().st_ino == (proc_root / "self/ns/mnt").stat().st_ino
        except OSError:
            same_namespace = False
        if not same_namespace:
            return {"available": False, "reason": "Target mount namespace differs or is inaccessible; run diagnostics inside the target container", "data_roots": [], "report_dir": None, "shm": None}
    mounts = _parse_target_mountinfo(proc_root, pid)

    fs_report: dict[str, Any] = {
        "data_roots": [_inspect_path_filesystem(Path(p), mounts) for p in data_roots],
        "report_dir": _inspect_path_filesystem(Path(report_dir), mounts) if report_dir is not None else None,
    }

    if include_shm:
        fs_report["shm"] = _inspect_path_filesystem(shm_path, mounts, is_shm=True)
    else:
        fs_report["shm"] = None

    if include_tmp:
        fs_report["temp_dir"] = _inspect_path_filesystem(Path(tempfile.gettempdir()), mounts)
    else:
        fs_report["temp_dir"] = None

    return fs_report


# ---------------------------------------------------------------------------
# Bounded GPU Sampler with driver_version and N/A handling
# ---------------------------------------------------------------------------
def sample_gpu_metrics(
    target_pid: int | None = None,
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return {
            "available": False,
            "reason": "nvidia-smi executable not found in PATH",
            "driver_version": None,
            "devices": [],
            "competing_apps": [],
        }

    # Query driver version and GPUs
    query_gpu = (
        "driver_version,index,uuid,name,memory.used,memory.total,memory.free,"
        "utilization.gpu,utilization.memory,power.draw,temperature.gpu"
    )
    cmd_gpu = [
        nvidia_smi,
        f"--query-gpu={query_gpu}",
        "--format=csv,noheader,nounits",
    ]

    devices: list[dict[str, Any]] = []
    driver_version: str | None = None
    try:
        proc = subprocess.run(
            cmd_gpu,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 11:
                    if driver_version is None and parts[0] not in ("N/A", ""):
                        driver_version = parts[0]
                    # Robust per-field parsing: handle N/A gracefully without dropping GPU
                    devices.append({
                        "index": _parse_int_or_none(parts[1]),
                        "uuid": parts[2],
                        "name": parts[3],
                        "memory_used_mb": _parse_float_or_none(parts[4]),
                        "memory_total_mb": _parse_float_or_none(parts[5]),
                        "memory_free_mb": _parse_float_or_none(parts[6]),
                        "utilization_gpu_pct": _parse_float_or_none(parts[7]),
                        "utilization_memory_pct": _parse_float_or_none(parts[8]),
                        "power_draw_w": _parse_float_or_none(parts[9]),
                        "temperature_c": _parse_float_or_none(parts[10]),
                    })
        else:
            return {
                "available": False,
                "reason": f"nvidia-smi failed with code {proc.returncode}: {proc.stderr.strip()}",
                "driver_version": None,
                "devices": [],
                "competing_apps": [],
            }
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "reason": f"nvidia-smi timed out after {timeout_seconds}s",
            "driver_version": None,
            "devices": [],
            "competing_apps": [],
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "reason": f"nvidia-smi error: {type(exc).__name__}: {exc}",
            "driver_version": None,
            "devices": [],
            "competing_apps": [],
        }

    # Query compute applications: RETAIN all competing apps, mark target
    competing_apps: list[dict[str, Any]] = []
    apps_available = False
    apps_reason = None
    cmd_apps = [
        nvidia_smi,
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc_apps = subprocess.run(
            cmd_apps,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if proc_apps.returncode == 0:
            apps_available = True
            for line in proc_apps.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    app_pid = _parse_int_or_none(parts[1])
                    competing_apps.append({
                        "gpu_uuid": parts[0],
                        "pid": app_pid,
                        "process_name": parts[2],
                        "used_memory_mb": _parse_float_or_none(parts[3]),
                        "is_target_process": (app_pid == target_pid) if (app_pid and target_pid) else False,
                    })
        else:
            apps_reason = f"compute-app query exited {proc_apps.returncode}"
    except (subprocess.TimeoutExpired, OSError, subprocess.SubprocessError) as exc:
        apps_reason = str(exc)

    return {
        "available": True,
        "driver_version": driver_version,
        "device_count": len(devices),
        "devices": devices,
        "competing_apps": competing_apps,
        "compute_apps_available": apps_available,
        "compute_apps_reason": apps_reason,
    }


# ---------------------------------------------------------------------------
# Deltas Computation with PID-Reuse StartTime Verification
# ---------------------------------------------------------------------------
def compute_resource_deltas(
    snapshot_start: dict[str, Any],
    snapshot_end: dict[str, Any],
) -> dict[str, Any]:
    t_start = snapshot_start.get("timestamp_unix", 0.0)
    t_end = snapshot_end.get("timestamp_unix", 0.0)
    elapsed = max(0.0001, t_end - t_start)

    deltas: dict[str, Any] = {
        "elapsed_seconds": round(elapsed, 4),
    }

    # CPU throttling deltas
    stat_start = (snapshot_start.get("cgroup") or {}).get("cpu_stat")
    stat_end = (snapshot_end.get("cgroup") or {}).get("cpu_stat")
    if isinstance(stat_start, dict) and isinstance(stat_end, dict):
        nr_throttled_0 = stat_start.get("nr_throttled")
        nr_throttled_1 = stat_end.get("nr_throttled")
        throttled_usec_0 = stat_start.get("throttled_usec")
        throttled_usec_1 = stat_end.get("throttled_usec")
        nr_periods_0 = stat_start.get("nr_periods")
        nr_periods_1 = stat_end.get("nr_periods")
        usage_usec_0 = stat_start.get("usage_usec")
        usage_usec_1 = stat_end.get("usage_usec")

        throttled_periods_delta = (nr_throttled_1 - nr_throttled_0) if nr_throttled_0 is not None and nr_throttled_1 is not None else None
        throttled_usec_delta = (throttled_usec_1 - throttled_usec_0) if throttled_usec_0 is not None and throttled_usec_1 is not None else None
        periods_delta = (nr_periods_1 - nr_periods_0) if nr_periods_0 is not None and nr_periods_1 is not None else None
        usage_usec_delta = (usage_usec_1 - usage_usec_0) if usage_usec_0 is not None and usage_usec_1 is not None else None

        throttling_rate = None
        if throttled_periods_delta is not None and periods_delta is not None and periods_delta > 0:
            throttling_rate = round(throttled_periods_delta / periods_delta * 100, 2)

        effective_cores_used = None
        if usage_usec_delta is not None:
            effective_cores_used = round(usage_usec_delta / (elapsed * 1e6), 3)

        deltas["cpu_throttling"] = {
            "periods_delta": periods_delta,
            "throttled_periods_delta": throttled_periods_delta,
            "throttled_usec_delta": throttled_usec_delta,
            "throttling_rate_pct": throttling_rate,
            "usage_usec_delta": usage_usec_delta,
            "effective_cores_used": effective_cores_used,
        }
    else:
        deltas["cpu_throttling"] = {"available": False, "reason": "cpu_stat not available in both snapshots"}

    # Memory delta
    cg_start = snapshot_start.get("cgroup") or {}
    cg_end = snapshot_end.get("cgroup") or {}
    mem_0 = cg_start.get("memory_current_bytes")
    mem_1 = cg_end.get("memory_current_bytes")
    deltas["memory_delta_bytes"] = (mem_1 - mem_0) if mem_0 is not None and mem_1 is not None else None
    events_0, events_1 = cg_start.get("memory_events") or {}, cg_end.get("memory_events") or {}
    deltas["memory_events"] = {
        name: events_1[name] - events_0[name]
        if isinstance(events_0.get(name), int) and isinstance(events_1.get(name), int) else None
        for name in ("high", "max", "oom", "oom_kill")
    }

    # Process metrics deltas with PID reuse guard
    p_start = snapshot_start.get("process") or {}
    p_end = snapshot_end.get("process") or {}
    if p_start.get("available") and p_end.get("available"):
        # Guard against PID reuse: check starttime_ticks
        st_0 = p_start.get("starttime_ticks")
        st_1 = p_end.get("starttime_ticks")
        if st_0 is None or st_1 is None or p_start.get("pid") != p_end.get("pid") or st_0 != st_1:
            deltas["process"] = {
                "available": False,
                "reason": f"PID reuse detected or process identity unavailable: starttime {st_0} -> {st_1}",
            }
        else:
            stat_0 = p_start.get("stat") or {}
            stat_1 = p_end.get("stat") or {}
            io_0 = p_start.get("io") or {}
            io_1 = p_end.get("io") or {}

            minflt_delta = (stat_1["minflt"] - stat_0["minflt"]) if "minflt" in stat_0 and "minflt" in stat_1 else None
            majflt_delta = (stat_1["majflt"] - stat_0["majflt"]) if "majflt" in stat_0 and "majflt" in stat_1 else None
            cpu_ticks_delta = (stat_1["total_cpu_ticks"] - stat_0["total_cpu_ticks"]) if "total_cpu_ticks" in stat_0 and "total_cpu_ticks" in stat_1 else None

            read_bytes_delta = None
            write_bytes_delta = None
            if io_0.get("available") and io_1.get("available"):
                r0, r1 = io_0.get("read_bytes"), io_1.get("read_bytes")
                w0, w1 = io_0.get("write_bytes"), io_1.get("write_bytes")
                if r0 is not None and r1 is not None:
                    read_bytes_delta = r1 - r0
                if w0 is not None and w1 is not None:
                    write_bytes_delta = w1 - w0

            deltas["process"] = {
                "minflt_delta": minflt_delta,
                "majflt_delta": majflt_delta,
                "cpu_ticks_delta": cpu_ticks_delta,
                "cpu_percent_one_core": (100 * cpu_ticks_delta / os.sysconf("SC_CLK_TCK") / elapsed)
                if cpu_ticks_delta is not None else None,
                "read_bytes_delta": read_bytes_delta,
                "write_bytes_delta": write_bytes_delta,
                "read_rate_bytes_per_sec": round(read_bytes_delta / elapsed, 1) if read_bytes_delta is not None else None,
                "write_rate_bytes_per_sec": round(write_bytes_delta / elapsed, 1) if write_bytes_delta is not None else None,
            }
    else:
        deltas["process"] = {"available": False, "reason": "process metrics not available in both snapshots"}

    return deltas


# ---------------------------------------------------------------------------
# Main Lightweight Snapshot API
# ---------------------------------------------------------------------------
def resource_snapshot(
    pid: int | None = None,
    data_roots: Sequence[str | Path] = (),
    report_dir: str | Path | None = None,
    *,
    include_gpu: bool = True,
    include_children: bool = False,
    gpu_timeout_seconds: float = 5.0,
    proc_root: Path = Path("/proc"),
    sys_cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, Any]:
    """Lightweight single-pass resource snapshot without pauses or sleeps."""
    target_pid = os.getpid() if pid is None else int(pid)
    now_ts = time.time()
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    system_name = platform.system()
    platform_info = {
        "system": system_name,
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "is_linux": (system_name == "Linux"),
    }

    if system_name == "Linux" or (proc_root / str(target_pid) / "cgroup").exists():
        cgroup_info = resolve_cgroup(target_pid, proc_root=proc_root, sys_cgroup_root=sys_cgroup_root)
    else:
        cgroup_info = {
            "available": False,
            "reason": f"cgroups not supported on {system_name}",
            "version": None,
        }

    if system_name == "Linux" or (proc_root / str(target_pid) / "status").exists():
        process_info = resolve_process_metrics(target_pid, proc_root=proc_root)
    else:
        process_info = {
            "available": False,
            "reason": f"procfs not supported on {system_name}",
        }

    fs_info = resolve_filesystems(
        data_roots=data_roots,
        report_dir=report_dir,
        proc_root=proc_root,
        pid=target_pid,
        include_shm=True,
        include_tmp=True,
    )

    if include_gpu:
        gpu_info = sample_gpu_metrics(target_pid=target_pid, timeout_seconds=gpu_timeout_seconds)
    else:
        gpu_info = {
            "available": False,
            "reason": "GPU sampling disabled by caller",
            "driver_version": None,
            "devices": [],
            "competing_apps": [],
        }

    children = []
    if include_children and system_name == "Linux":
        pending = [target_pid]
        seen = {target_pid}
        while pending and len(seen) <= 32:
            parent_pid = pending.pop(0)
            raw = _safe_read_text(proc_root / str(parent_pid) / "task" / str(parent_pid) / "children") or ""
            for token in raw.split():
                if token.isdigit() and int(token) not in seen and len(seen) < 32:
                    child_pid = int(token)
                    seen.add(child_pid)
                    pending.append(child_pid)
                    children.append(resolve_process_metrics(child_pid, proc_root))
    return {
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "timestamp_iso": now_iso,
        "timestamp_unix": now_ts,
        "pid": target_pid,
        "platform": platform_info,
        "cgroup": cgroup_info,
        "process": process_info,
        "children": children,
        "children_note": "bounded to 31 descendants of the main-thread process tree; PSS availability is per process",
        "filesystem": fs_info,
        "gpu": gpu_info,
    }
