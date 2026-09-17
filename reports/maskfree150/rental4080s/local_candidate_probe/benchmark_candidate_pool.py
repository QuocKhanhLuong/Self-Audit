"""Standalone candidate execution benchmark for serial vs spawn pool (W3).

Worker-supplied reproduction runner. The root's independently captured output
is stored separately; the earlier worker report is not authoritative evidence.
Scope: synthetic-only, CPU-only, resolution 224, 8 units, candidate_worker_threads=1.
"""
from __future__ import annotations

import gc
import json
import pickle
import platform
import socket
import time
import torch

from scripts.check_maskfree_audit_device import build_synthetic_units
from self_audit_maskfree.candidate_execution import (
    CandidatePoolExecutor,
    _prepare_task_payloads,
    generate_candidate_banks,
)
from self_audit_maskfree.hypotheses import BANK_SIZE


def run_candidate_benchmark(
    *,
    count: int = 8,
    image_size: int = 224,
    repeats: int = 5,
    seed: int = 42,
) -> dict:
    units = build_synthetic_units(size=image_size, seed=seed, count=count)
    fitting_views = [u.fitting for u in units]
    generator = torch.Generator().manual_seed(1234)
    features = [torch.randn(16, image_size, image_size, generator=generator) for _ in range(count)]
    seeds = [1000 + i for i in range(count)]

    # 1. Serialization footprint
    payloads = _prepare_task_payloads(fitting_views, features, seeds)
    t0 = time.perf_counter()
    raw_input_bytes = pickle.dumps(payloads)
    t1 = time.perf_counter()
    in_bytes = len(raw_input_bytes)
    in_time_ms = (t1 - t0) * 1000.0

    # 2. Canonical Serial baseline (workers=0)
    _ = generate_candidate_banks(fitting_views, features, seeds, workers=0)
    serial_times: list[float] = []
    res_serial = None
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        res_serial = generate_candidate_banks(fitting_views, features, seeds, workers=0)
        t1 = time.perf_counter()
        serial_times.append(t1 - t0)

    assert res_serial is not None
    res_bytes = len(pickle.dumps(res_serial))

    # 3. Persistent Spawn Pool (workers=2, worker_threads=1)
    pool2_times: list[float] = []
    res_pool2 = None
    with CandidatePoolExecutor(workers=2, worker_threads=1) as pool2:
        _ = pool2.generate_banks(fitting_views, features, seeds)
        for _ in range(repeats):
            gc.collect()
            t0 = time.perf_counter()
            res_pool2 = pool2.generate_banks(fitting_views, features, seeds)
            t1 = time.perf_counter()
            pool2_times.append(t1 - t0)

    assert res_pool2 is not None

    # 4. Persistent Spawn Pool (workers=4, worker_threads=1)
    pool4_times: list[float] = []
    res_pool4 = None
    with CandidatePoolExecutor(workers=4, worker_threads=1) as pool4:
        _ = pool4.generate_banks(fitting_views, features, seeds)
        for _ in range(repeats):
            gc.collect()
            t0 = time.perf_counter()
            res_pool4 = pool4.generate_banks(fitting_views, features, seeds)
            t1 = time.perf_counter()
            pool4_times.append(t1 - t0)

    assert res_pool4 is not None

    # 5. Exact bitwise numerical equality check
    match_p2 = all(
        torch.equal(res_pool2[u][c].labels, res_serial[u][c].labels)
        and res_pool2[u][c].candidate_id == res_serial[u][c].candidate_id
        and res_pool2[u][c].metadata["bank_id"] == res_serial[u][c].metadata["bank_id"]
        and res_pool2[u][c].metadata["candidate_content_hash"] == res_serial[u][c].metadata["candidate_content_hash"]
        for u in range(count)
        for c in range(BANK_SIZE)
    )
    match_p4 = all(
        torch.equal(res_pool4[u][c].labels, res_serial[u][c].labels)
        and res_pool4[u][c].candidate_id == res_serial[u][c].candidate_id
        and res_pool4[u][c].metadata["bank_id"] == res_serial[u][c].metadata["bank_id"]
        and res_pool4[u][c].metadata["candidate_content_hash"] == res_serial[u][c].metadata["candidate_content_hash"]
        for u in range(count)
        for c in range(BANK_SIZE)
    )

    serial_mean = sum(serial_times) / len(serial_times)
    p2_mean = sum(pool2_times) / len(pool2_times)
    p4_mean = sum(pool4_times) / len(pool4_times)

    return {
        "metadata": {
            "hostname": socket.gethostname(),
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "python_version": platform.python_version(),
            "scope": "synthetic-only",
            "units": count,
            "image_size": image_size,
            "candidate_worker_threads": 1,
            "parent_torch_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
        },
        "serialization": {
            "input_payload_bytes": in_bytes,
            "input_serialize_ms": round(in_time_ms, 2),
            "result_payload_bytes": res_bytes,
        },
        "serial": {
            "workers": 0,
            "worker_threads": 1,
            "repeats": repeats,
            "times": [round(t, 4) for t in serial_times],
            "mean_s": round(serial_mean, 4),
            "p50_s": round(sorted(serial_times)[len(serial_times) // 2], 4),
            "per_unit_ms": round((serial_mean / count) * 1000, 2),
        },
        "pool_workers_2": {
            "workers": 2,
            "worker_threads": 1,
            "repeats": repeats,
            "times": [round(t, 4) for t in pool2_times],
            "mean_s": round(p2_mean, 4),
            "p50_s": round(sorted(pool2_times)[len(pool2_times) // 2], 4),
            "speedup_vs_serial": round(serial_mean / p2_mean, 2),
            "exact_match": match_p2,
        },
        "pool_workers_4": {
            "workers": 4,
            "worker_threads": 1,
            "repeats": repeats,
            "times": [round(t, 4) for t in pool4_times],
            "mean_s": round(p4_mean, 4),
            "p50_s": round(sorted(pool4_times)[len(pool4_times) // 2], 4),
            "speedup_vs_serial": round(serial_mean / p4_mean, 2),
            "exact_match": match_p4,
        },
    }


if __name__ == "__main__":
    results = run_candidate_benchmark()
    print(json.dumps(results, indent=2))
