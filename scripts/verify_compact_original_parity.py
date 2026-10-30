#!/usr/bin/env python3
"""Compare current original selector with executable, pinned pre-extension source.

This manufactured CPU check establishes software parity, not patient quality.
It deliberately does not use the new selector's selection helper as its oracle.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import types

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from self_audit import compact_control as current

BASE_REVISION = "d93c4dea4f42895297ec5941bb938b47140aa9e4"
BASE_SOURCE_SHA256 = "17878be6becada6f6fefa2ed7d56f52357c7d049f6ff07f853ea319e91d082b8"


def old_keys_equal(expected, actual, path="receipt"):
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(f"{path}.{key} missing")
            old_keys_equal(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(expected) == len(actual), path
        for i, (left, right) in enumerate(zip(expected, actual)):
            old_keys_equal(left, right, f"{path}[{i}]")
    else:
        assert expected == actual, (path, expected, actual)


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify():
    source = subprocess.check_output(
        ["git", "show", f"{BASE_REVISION}:self_audit/compact_control.py"], cwd=ROOT
    )
    assert hashlib.sha256(source).hexdigest() == BASE_SOURCE_SHA256
    legacy = types.ModuleType("pinned_compact_control")
    exec(compile(source, f"git:{BASE_REVISION}:compact_control.py", "exec"), legacy.__dict__)
    torch.set_num_threads(1)
    reports = []
    for i in range(80):
        seed = 20261030 + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = nn.Sequential(nn.Conv2d(3, 6, 1), nn.BatchNorm2d(6), nn.SiLU(), nn.Conv2d(6, 4, 1))
        model.train()
        x = torch.randn(2, 3, 5, 7)
        y = torch.randint(0, 4, (2, 5, 7))
        if i % 5 == 0:
            y[y == 3] = 0  # An absent-class guard fixture.
        y[:, 0, 0] = 255
        valid = y != 255
        targets = torch.softmax(torch.randn(2, 4, 5, 7), 1)
        kwargs = dict(lr=0.01, temperature=(1.0, 0.7, 2.0)[i % 3],
                      coefficient=0.8, confidence=(0.0, 0.4, 0.99)[i % 3],
                      max_norm=0.1 if i % 7 == 0 else None)
        legacy_candidates = legacy.build_class_candidates(model, x, targets, valid, **kwargs)
        new_candidates = current.build_class_candidates(model, x, targets, valid, **kwargs)
        assert legacy_candidates["denominator"] == new_candidates["denominator"]
        assert legacy_candidates["source_ids"] == new_candidates["source_ids"]
        assert len(legacy_candidates["directions"]) == len(new_candidates["directions"])
        for ld, nd in zip(legacy_candidates["directions"], new_candidates["directions"]):
            assert ld.keys() == nd.keys()
            for name in ld:
                assert torch.equal(ld[name], nd[name]), (i, name, "candidate")
        if i % 4 == 0:
            m = 1 if i % 8 == 0 else 4
            amp = (0.0, 0.01, 0.3, 3.0)[(i // 4) % 4]
            directions = [{name: torch.randn_like(p) * amp for name, p in model.named_parameters()}
                          for _ in range(m)]
            legacy_candidates = {"directions": directions, "source_ids": [f"direction_{j}" for j in range(m)], "denominator": 1}
            new_candidates = copy.deepcopy(legacy_candidates)
        if i >= 64:
            # Legacy public config supports custom zero-first and empty grids;
            # their historical fold cannot be replaced by a naive objective min.
            directions = [{name: torch.ones_like(p) * 1e-4 for name, p in model.named_parameters()}]
            legacy_candidates = {"directions": directions, "source_ids": ["direction_0"], "denominator": 1}
            new_candidates = copy.deepcopy(legacy_candidates)
        original = copy.deepcopy(model)
        updated = copy.deepcopy(model)
        fixture_hash = state_digest(original.state_dict())
        method = ("scalar_guard", "scalar_guard", "joint_guard", "plain_kd")[i % 4]
        cfg = (None, {}, {"ridge": 0.01, "tolerance": 1e-6}, {"max_trials": 3})[i % 4]
        if i % 11 == 0:
            cfg = {"max_norm": 1e-6}
        if i >= 64:
            cfg = {"trial_alphas": [[0.0], [1.0], [0.5]] if i < 72 else [],
                   "eps": 1.0, "absent_fp_budget": 1.0}
        def audit():
            yield {"x": x, "y": y, "valid": valid, "patient": ["a", "b"], "frame": [0, 0]}
        before_rng = torch.get_rng_state().clone()
        expected = legacy.apply_compact_update(original, legacy_candidates, method=method, audit_batches=audit, config=cfg)
        actual = current.apply_compact_update(updated, new_candidates, method=method, audit_batches=audit, config=cfg)
        old_keys_equal(expected, actual, path=f"case[{i}].receipt")
        assert torch.equal(before_rng, torch.get_rng_state())
        assert original.training == updated.training
        for name, value in original.state_dict().items():
            assert torch.equal(value, updated.state_dict()[name]), (i, name, "commit")
        reports.append({"case": i, "seed": seed, "method": method, "status": actual["status"],
                        "alpha": actual["alpha"], "initial_weights_sha256": fixture_hash,
                        "final_weights_sha256": state_digest(updated.state_dict())})
    return {
        "status": "PASS", "scope": "manufactured CPU executable original-source parity only",
        "base_revision": BASE_REVISION, "base_source_sha256": BASE_SOURCE_SHA256,
        "current_source_sha256": hashlib.sha256((ROOT / "self_audit/compact_control.py").read_bytes()).hexdigest(),
        "cases": reports, "patient_quality": "NOT_RUN", "gpu": "NOT_RUN",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("new output file required")
    report = verify()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"PASS: {len(report['cases'])} pinned-source parity cases; {args.out}")


if __name__ == "__main__":
    main()
