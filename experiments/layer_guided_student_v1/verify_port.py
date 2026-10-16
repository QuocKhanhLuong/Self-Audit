"""Reproduce parity and manufactured smokes into a fresh receipt directory."""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
REFERENCE_COMMIT = "f4b0fe7d9e49ee398354e20ff0981c818840f5d4"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        parser.error("receipt directory must be new")
    spec = importlib.util.spec_from_file_location(
        "reference_verifier", REPO / "scripts/materialize_layer_upgrade_reference.py"
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    payloads = verifier.verified_payloads()
    exact = [name for name in payloads if name.startswith("layer_upgrade/")]
    exact += [
        "tests/test_operators.py",
        "verify_full_model.py",
        "probe_report.py",
        "pytest.ini",
    ]
    for name in exact:
        if (ROOT / name).read_bytes() != payloads[name]:
            raise ValueError("exact-reference bytes differ: " + name)
    protected = (
        "baseline",
        "src",
        "configs",
        "run_configs",
        "splits",
        "benchmark_freezes",
        "checkpoints",
    )
    protected_diff = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO),
            "diff",
            "--name-only",
            REFERENCE_COMMIT,
            "--",
            *protected,
        ],
        text=True,
    ).splitlines()
    if protected_diff:
        raise ValueError("protected historical paths differ: " + repr(protected_diff))
    out.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=os.pathsep.join((str(ROOT), str(REPO / "src"))),
    )
    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-c",
            str(ROOT / "pytest.ini"),
            "--rootdir=" + str(ROOT),
            "--confcutdir=" + str(ROOT),
            str(ROOT / "tests"),
            "--junitxml=" + str(out / "TEST_RESULTS.xml"),
        ],
        [sys.executable, str(ROOT / "verify_full_model.py")],
        [sys.executable, str(ROOT / "probe_report.py")],
    ]
    results = []
    for label, command in zip(("tests", "full_model", "probes"), commands):
        with (out / f"{label}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        results.append(
            {"label": label, "argv": command, "exit_code": result.returncode}
        )
        if result.returncode:
            (out / "FAILURE.json").write_text(json.dumps(results, indent=2))
            print(f"FAILED: {out / (label + '.log')}")
            return result.returncode
    suites = ET.parse(out / "TEST_RESULTS.xml").getroot().findall("testsuite")
    totals = {
        key: sum(int(s.attrib.get(key, 0)) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    smoke = json.loads((out / "FULL_MODEL_CHECK.json").read_text())
    if (
        smoke["parameters"],
        smoke["encoder_parameters"],
        smoke["other_parameters"],
    ) != (32890558, 27818592, 5071966):
        raise ValueError("random full-model parameter count differs from reference")
    report = {
        "schema": "layer_guided.parity_report.v1",
        "status": "PASS_CPU_SOFTWARE_ONLY",
        "reference_commit": REFERENCE_COMMIT,
        "checkout_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "reference_manifest_sha256": verifier.MANIFEST_SHA256,
        "byte_identical_reference_paths": {name: sha(ROOT / name) for name in exact},
        "implementation_sha256": {
            str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob("*.py"))
        },
        "tests": totals,
        "commands": results,
        "python": platform.python_version(),
        "device": "cpu",
        "random_full_model": smoke,
        "official_dinov3_runtime": "NOT_RUN_MISSING_APPROVED_LOCAL_SOURCE_AND_WEIGHTS",
        "patient_training": "NOT_RUN",
        "gpu_memory": "NOT_MEASURED",
        "semantic_quality": "NOT_MEASURED",
        "baseline_modified": False,
        "protected_paths": list(protected),
        "protected_diff_from_reference": protected_diff,
        "interpretation": "Independent module identities; same states/inputs; outputs, gradients, rollback and failure parity. Known reference limits preserved.",
    }
    (out / "PARITY_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "tests": totals,
                "receipt": str(out / "PARITY_REPORT.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
