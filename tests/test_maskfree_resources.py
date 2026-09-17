"""Focused tests for maskfree resource diagnostics and cgroup resolvers.

Verifies:
- Strict fail-closed cgroup resolution (never leaks parent/host limits on missing subgroups).
- Ancestor hierarchy inspection (parent stricter quota overrides leaf quota in effective cores).
- Unknown vs unlimited distinction (missing files or period=0 are unknown, never unlimited).
- Mountinfo octal escape decoding (e.g. \\040 for spaces) and namespace resolution.
- Filesystem mount source/fstype tracking and strict /dev/shm tmpfs isolation.
- Bounded GPU sampler with driver version, graceful N/A handling, and competing apps retention.
- Process metrics with PID-reuse starttime guard.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from self_audit_maskfree.resources import (
    RESOURCE_SCHEMA_VERSION,
    _decode_mountinfo_escapes,
    _parse_cgroup_v2_cpu_max,
    _parse_cpuset_range,
    compute_resource_deltas,
    resolve_cgroup,
    resolve_filesystems,
    resolve_process_metrics,
    resource_snapshot,
    sample_gpu_metrics,
)


def test_decode_mountinfo_escapes() -> None:
    assert _decode_mountinfo_escapes(r"/mnt/data\040with\040spaces") == "/mnt/data with spaces"
    assert _decode_mountinfo_escapes(r"/mnt/tab\011path") == "/mnt/tab\tpath"
    assert _decode_mountinfo_escapes("/var/log") == "/var/log"


def test_parse_cgroup_v2_cpu_max() -> None:
    # Valid finite quota
    p1 = _parse_cgroup_v2_cpu_max("400000 100000\n")
    assert p1["valid"] is True
    assert p1["quota_cores"] == 4.0
    assert p1["unlimited"] is False

    # Valid unlimited quota
    p2 = _parse_cgroup_v2_cpu_max("max 100000\n")
    assert p2["valid"] is True
    assert p2["quota_cores"] is None
    assert p2["unlimited"] is True

    # Missing file / empty
    p3 = _parse_cgroup_v2_cpu_max(None)
    assert p3["valid"] is False
    assert p3["unlimited"] is False
    assert p3["quota_cores"] is None

    # Invalid period zero
    p4 = _parse_cgroup_v2_cpu_max("100000 0\n")
    assert p4["valid"] is False
    assert p4["unlimited"] is False
    assert "period" in p4["reason"]


def test_cgroup_v2_fail_closed_on_missing_target_subgroup(tmp_path: Path) -> None:
    proc_dir = tmp_path / "proc"
    sys_dir = tmp_path / "sys" / "fs" / "cgroup"
    sys_dir.mkdir(parents=True)

    pid = 1111
    (proc_dir / str(pid)).mkdir(parents=True)
    (proc_dir / str(pid) / "cgroup").write_text("0::/user.slice/nonexistent.scope\n", encoding="utf-8")
    (proc_dir / str(pid) / "mountinfo").write_text(
        f"28 0 0:25 / {sys_dir} rw - cgroup2 cgroup2 rw\n",
        encoding="utf-8",
    )

    # Note: nonexistent.scope was NOT created under sys_dir!
    info = resolve_cgroup(pid, proc_root=proc_dir, sys_cgroup_root=sys_dir)
    assert info["available"] is False
    # Crucial: Must fail closed! Must not fall back to sys_dir and report host limits!
    assert "could not be strictly resolved" in info["reason"] or "not found" in info["reason"]


def test_cgroup_v2_hierarchy_parent_stricter(tmp_path: Path) -> None:
    proc_dir = tmp_path / "proc"
    sys_dir = tmp_path / "sys" / "fs" / "cgroup"
    parent_cg = sys_dir / "parent.slice"
    child_cg = parent_cg / "child.scope"
    child_cg.mkdir(parents=True)

    pid = 2222
    (proc_dir / str(pid)).mkdir(parents=True)
    (proc_dir / str(pid) / "cgroup").write_text("0::/parent.slice/child.scope\n", encoding="utf-8")
    (proc_dir / str(pid) / "mountinfo").write_text(
        f"28 0 0:25 / {sys_dir} rw - cgroup2 cgroup2 rw\n",
        encoding="utf-8",
    )

    # Parent has strict 2.0 cores quota; Child leaf has 4.0 cores quota
    (parent_cg / "cpu.max").write_text("200000 100000\n", encoding="utf-8")
    (child_cg / "cpu.max").write_text("400000 100000\n", encoding="utf-8")
    (child_cg / "cpu.stat").write_text("usage_usec 100000\nnr_periods 10\n", encoding="utf-8")
    (child_cg / "cpuset.cpus.effective").write_text("0-7\n", encoding="utf-8")

    info = resolve_cgroup(pid, proc_root=proc_dir, sys_cgroup_root=sys_dir)
    assert info["available"] is True
    assert info["version"] == 2
    assert info["leaf_cpu"]["quota_cores"] == 4.0
    # Effective quota must be bounded by the stricter parent (2.0 cores)!
    assert info["visible_cpu_upper_bound_cores"] == 2.0
    assert info["effective_quota_cores"] is None
    assert len(info["ancestor_quotas"]) >= 1


def test_cgroup_v1_fail_closed_and_effective(tmp_path: Path) -> None:
    proc_dir = tmp_path / "proc"
    sys_dir = tmp_path / "sys" / "fs" / "cgroup"
    cpu_cg = sys_dir / "cpu" / "container1"
    cpu_cg.mkdir(parents=True)

    pid = 3333
    (proc_dir / str(pid)).mkdir(parents=True)
    (proc_dir / str(pid) / "cgroup").write_text("1:cpu:/container1\n", encoding="utf-8")
    (proc_dir / str(pid) / "mountinfo").write_text(
        f"50 0 0:30 / {sys_dir / 'cpu'} rw - cgroup cgroup rw,cpu\n",
        encoding="utf-8",
    )

    (cpu_cg / "cpu.cfs_quota_us").write_text("600000\n", encoding="utf-8")
    (cpu_cg / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8")

    info = resolve_cgroup(pid, proc_root=proc_dir, sys_cgroup_root=sys_dir)
    assert info["available"] is True
    assert info["version"] == 1
    assert info["leaf_cpu"]["quota_cores"] == 6.0
    assert info["visible_cpu_upper_bound_cores"] == 6.0


def test_shm_isolation_refuses_parent_capacity(tmp_path: Path) -> None:
    proc_dir = tmp_path / "proc"
    pid = 4444
    (proc_dir / str(pid)).mkdir(parents=True)
    mock_shm = tmp_path / "mock_shm"
    mock_shm.mkdir()

    # Mountinfo where mock_shm is NOT an independent tmpfs mount
    (proc_dir / str(pid) / "mountinfo").write_text(
        "1 0 0:1 / / rw - ext4 /dev/sda1 rw\n",
        encoding="utf-8",
    )

    fs = resolve_filesystems(
        data_roots=[],
        proc_root=proc_dir,
        pid=pid,
        include_shm=True,
        include_tmp=False,
        shm_path=mock_shm,
    )
    shm = fs["shm"]
    assert shm["available"] is False
    assert "not an independent tmpfs mount" in shm["reason"]


def test_gpu_sampler_driver_version_na_handling_and_apps() -> None:
    sample_gpu_stdout = (
        "550.54.14,0,GPU-1234,NVIDIA GeForce RTX 4080 SUPER,1024,16384,15360,25.0,15.0,N/A,[Not Supported]\n"
    )
    sample_apps_stdout = (
        "GPU-1234,1001,python,512\n"
        "GPU-1234,1002,other_worker,400\n"
    )

    def mock_run(cmd, *args, **kwargs):
        res = MagicMock()
        res.returncode = 0
        if "--query-gpu=" in cmd[1]:
            res.stdout = sample_gpu_stdout
        elif "--query-compute-apps=" in cmd[1]:
            res.stdout = sample_apps_stdout
        return res

    with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), patch("subprocess.run", side_effect=mock_run):
        res = sample_gpu_metrics(target_pid=1001)
        assert res["available"] is True
        assert res["driver_version"] == "550.54.14"
        assert len(res["devices"]) == 1
        dev = res["devices"][0]
        assert dev["name"] == "NVIDIA GeForce RTX 4080 SUPER"
        assert dev["power_draw_w"] is None  # Handled N/A gracefully
        assert dev["temperature_c"] is None  # Handled [Not Supported] gracefully
        assert dev["memory_used_mb"] == 1024.0

        # Competing apps: retained ALL apps and marked target
        apps = res["competing_apps"]
        assert len(apps) == 2
        target_app = next(a for a in apps if a["pid"] == 1001)
        assert target_app["is_target_process"] is True
        other_app = next(a for a in apps if a["pid"] == 1002)
        assert other_app["is_target_process"] is False


def test_compute_deltas_pid_reuse_guard() -> None:
    s1 = {
        "timestamp_unix": 100.0,
        "process": {
            "available": True,
            "pid": 5000,
            "starttime_ticks": 12345,
            "stat": {"minflt": 10, "majflt": 0, "total_cpu_ticks": 100},
        },
    }
    # s2 has the same PID 5000 but different starttime_ticks -> PID reuse!
    s2 = {
        "timestamp_unix": 105.0,
        "process": {
            "available": True,
            "pid": 5000,
            "starttime_ticks": 99999,
            "stat": {"minflt": 50, "majflt": 1, "total_cpu_ticks": 200},
        },
    }

    deltas = compute_resource_deltas(s1, s2)
    assert deltas["process"]["available"] is False
    assert "PID reuse detected" in deltas["process"]["reason"]


def test_compute_deltas_valid_ticks_and_faults() -> None:
    s1 = {
        "timestamp_unix": 100.0,
        "process": {
            "available": True,
            "pid": 5000,
            "starttime_ticks": 12345,
            "stat": {"minflt": 100, "majflt": 2, "total_cpu_ticks": 500},
            "io": {"available": True, "read_bytes": 1000, "write_bytes": 2000},
        },
    }
    s2 = {
        "timestamp_unix": 105.0,
        "process": {
            "available": True,
            "pid": 5000,
            "starttime_ticks": 12345,
            "stat": {"minflt": 150, "majflt": 5, "total_cpu_ticks": 750},
            "io": {"available": True, "read_bytes": 6000, "write_bytes": 4000},
        },
    }

    deltas = compute_resource_deltas(s1, s2)
    assert deltas["elapsed_seconds"] == 5.0
    p = deltas["process"]
    assert p["minflt_delta"] == 50
    assert p["majflt_delta"] == 3
    assert p["cpu_ticks_delta"] == 250
    assert p["read_bytes_delta"] == 5000
    assert p["write_bytes_delta"] == 2000


def test_host_resource_snapshot_smoke() -> None:
    snap = resource_snapshot(
        pid=os.getpid(),
        data_roots=[Path(".")],
        report_dir=Path("reports"),
        include_gpu=False,
    )
    assert snap["schema_version"] == RESOURCE_SCHEMA_VERSION
    assert snap["pid"] == os.getpid()
    assert "platform" in snap
    assert "filesystem" in snap
    if platform.system() != "Linux":
        assert snap["cgroup"]["available"] is False
        assert snap["process"]["available"] is False


def test_cgroup_v1_unlimited(tmp_path: Path) -> None:
    proc_dir = tmp_path / "proc"
    sys_dir = tmp_path / "sys" / "fs" / "cgroup"
    cpu_cg = sys_dir / "cpu" / "unlimited"
    mem_cg = sys_dir / "memory" / "unlimited"
    cpu_cg.mkdir(parents=True)
    mem_cg.mkdir(parents=True)

    pid = 7777
    (proc_dir / str(pid)).mkdir(parents=True)
    (proc_dir / str(pid) / "cgroup").write_text("1:cpu:/unlimited\n2:memory:/unlimited\n", encoding="utf-8")
    (proc_dir / str(pid) / "mountinfo").write_text(
        f"50 0 0:30 / {sys_dir / 'cpu'} rw - cgroup cgroup rw,cpu\n"
        f"51 0 0:31 / {sys_dir / 'memory'} rw - cgroup cgroup rw,memory\n",
        encoding="utf-8",
    )

    (cpu_cg / "cpu.cfs_quota_us").write_text("-1\n", encoding="utf-8")
    (cpu_cg / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8")
    (mem_cg / "memory.limit_in_bytes").write_text("9223372036854771712\n", encoding="utf-8")

    info = resolve_cgroup(pid, proc_root=proc_dir, sys_cgroup_root=sys_dir)
    assert info["available"] is True
    assert info["leaf_cpu"]["unlimited"] is True
    assert info["leaf_cpu"]["quota_cores"] is None
    assert info["memory_unlimited"] is True
    assert info["memory_max_bytes"] is None


def test_diagnose_cli_smoke(tmp_path: Path) -> None:
    from scripts.diagnose_maskfree_resources import parse_args, run_diagnostics

    out_file = tmp_path / "diag_out.json"
    args = parse_args(["--data-root", str(tmp_path), "--output", str(out_file), "--no-gpu", "--count", "1"])
    res = run_diagnostics(args)
    assert res["schema_version"] == RESOURCE_SCHEMA_VERSION
    assert "snapshot" in res
    assert res["snapshot"]["pid"] == os.getpid()


def test_missing_target_mountinfo_never_uses_existing_host_root(tmp_path):
    proc = tmp_path / "proc" / "42"
    proc.mkdir(parents=True)
    (proc / "cgroup").write_text("0::/\n")
    host = tmp_path / "host"
    host.mkdir()
    (host / "cpu.max").write_text("9600000 100000")
    result = resolve_cgroup(42, proc_root=proc.parent, sys_cgroup_root=host)
    assert result["available"] is False


def test_memory_event_deltas_are_not_missing_as_zero():
    before = {"timestamp_unix": 2, "cgroup": {"memory_events": {"high": 3, "oom": 0}}}
    after = {"timestamp_unix": 5, "cgroup": {"memory_events": {"high": 7, "oom": 1}}}
    result = compute_resource_deltas(before, after)["memory_events"]
    assert result == {"high": 4, "max": None, "oom": 1, "oom_kill": None}
