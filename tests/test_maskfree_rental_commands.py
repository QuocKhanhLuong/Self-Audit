"""Verify the handoff actually requests the locked experiment, without CUDA."""
from pathlib import Path
import pytest
from scripts.benchmark_maskfree_rental import configs_and_commands, parser
from scripts.profile_maskfree import build_parser


def test_rental_plan_paths_and_work_budget(tmp_path):
    args = parser().parse_args(["--output", str(tmp_path), "--prefetch-batches", "1"])
    configs, commands = configs_and_commands(args)
    assert configs["acdc"].data_root == "/root/Self-Audit/data/ACDC"
    assert configs["mnms"].data_root == "/root/Self-Audit/data/MnM/extracted/M&M"
    for config in configs.values():
        assert (config.image_size, config.batch_size, config.accumulation_steps) == (224, 8, 1)
        assert config.total_epochs == 150 and config.amp is False
        assert config.audit_device == "cpu" and config.epoch_validation
        assert config.prefetch_batches == 1
    assert all(command[-1] == "--preflight" for command in commands[:2])
    profile = build_parser().parse_args(commands[-1][2:])
    assert (profile.warmup_batches, profile.measured_batches, profile.timing_mode) == (5, 30, "ordinary")
    assert profile.cprofile is False


def test_diagnostic_is_separate_and_short(tmp_path):
    args = parser().parse_args(["--output", str(tmp_path), "--diagnostic"])
    configs, commands = configs_and_commands(args)
    profile = build_parser().parse_args(commands[-1][2:])
    assert (profile.warmup_batches, profile.measured_batches, profile.timing_mode) == (1, 3, "instrumented")
    assert profile.cprofile is True
    assert configs["acdc"].timing_mode == "diagnostic"


def test_profiler_runtime_options_are_explicit():
    args = build_parser().parse_args(["--synthetic", "--runtime-timing-mode", "production",
                                     "--logging-mode", "buffered", "--data-cache-bytes", "0"])
    assert args.runtime_timing_mode == "production" and args.data_cache_bytes == 0
