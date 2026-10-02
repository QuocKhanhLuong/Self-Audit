"""ACDC readiness for the native baselines (DSS-US, SGSCN), one status per question.

Reads no GT and runs no method. Producer status reflects the protocol/adaptation gates only;
data, environment and full-run status are reported separately.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "baseline/DSS_US/scripts"), str(ROOT / "baseline/DSS_US/src"),
                str(ROOT / "baseline/SGSCN/src")]
MANIFEST = ROOT / "benchmark_freezes/cardiac_benchmark_v12_historical_224/data/acdc_shared_manifest.json"


def _load_runner(method):
    import importlib.util
    path = ROOT / "baseline" / method / "scripts/run_acdc.py"
    spec = importlib.util.spec_from_file_location(f"run_acdc_{method.lower()}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def method_status(method, data_reasons):
    from shared_benchmark.acdc_tracks import TrackError, tracks_contract
    from shared_benchmark.native_protocol import ProtocolBlocked
    runner = _load_runner(method)
    adaptations, blocked = [], []
    for path in sorted((ROOT / "baseline" / method / "config/acdc").glob("*.json")):
        try:
            adaptation = runner.load_profile(path)[0]
            adaptations.append({"adaptation": adaptation["adaptation_id"],
                                "profile_class": adaptation["source_profile"]["profile_class"]})
        except ProtocolBlocked as error:
            blocked.append({"adaptation": path.stem, "reasons": error.reasons})
    try:
        tracks_contract()
        track_a, track_b = "ADAPTER_READY (frozen cardiac_adapter_v2)", "EVALUATOR_READY (raw_id_majority_vote_v1)"
    except TrackError as error:
        track_a = track_b = f"BLOCKED_PROTOCOL: {error}"
    return {"ACDC_PRODUCER_STATUS": "PROTOCOL_READY" if adaptations and not blocked else "BLOCKED_PROTOCOL",
            "ACDC_PRODUCER_RUNNABLE_NOW": not data_reasons and bool(adaptations),
            "ACDC_TRACK_A_STATUS": track_a, "ACDC_TRACK_B_STATUS": track_b,
            "ACDC_DATA_STATUS": "AVAILABLE" if not data_reasons else "BLOCKED_DATA",
            "data_reasons": data_reasons, "adaptations": adaptations, "blocked_adaptations": blocked}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--dino-repo", type=Path, default=ROOT / ".scratch/dino")
    parser.add_argument("--dino-checkpoint", type=Path)
    parser.add_argument("--dino-checkpoint-sha256")
    args = parser.parse_args(argv)
    from environment_contract import require_official_environment
    try:
        environment = require_official_environment()
        env_status = "CANONICAL" if environment["official"] else "UNOFFICIAL_OVERRIDE"
    except SystemExit as error:
        env_status = f"NOT_CANONICAL: {error}"
    except Exception as error:  # noqa: BLE001 - report, never crash a readiness probe
        env_status = f"NOT_CANONICAL: {error}"
    from shared_benchmark.acdc_native import check_acdc_data
    import run_native as dss_native
    images = check_acdc_data(args.manifest, args.image_root)
    dino = dss_native.check_dino(args.dino_repo, args.dino_checkpoint, args.dino_checkpoint_sha256)
    report = {"ACDC_ENV_STATUS": env_status,
              "ACDC_DATA_STATUS": "AVAILABLE" if not images and not dino else "BLOCKED_DATA",
              "data_reasons": {"acdc_images": images, "dino_vits8": dino},
              "DSS-US": method_status("DSS_US", images + dino), "SGSCN": method_status("SGSCN", images)}
    ready = env_status == "CANONICAL" and not images and not dino
    report["ACDC_FULL_RUN_STATUS"] = "NOT_STARTED (awaiting explicit approval)" if ready else "NOT_STARTED (BLOCKED_DATA)"
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
