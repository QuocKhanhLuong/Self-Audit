"""Export a frozen Q-3D training checkpoint as a student-only deploy artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.candidate_q3d.export import (  # noqa: E402
    _sha256_file,
    _write_json_atomic,
    export_checkpoint,
    load_export,
    verify_cold_load,
)


def export(checkpoint: str | Path, output: str | Path, *, checkpoint_sha256: str | None = None,
           config_path: str | Path | None = None,
           code_commit: str | None = None, cost_report_path: str | Path | None = None,
           verify_volumes: list[str] | None = None) -> dict:
    cost_report = None
    if cost_report_path is not None:
        cost_report = json.loads(Path(cost_report_path).read_text())
        if not isinstance(cost_report, dict):
            raise ValueError("Q-3D cost report must be a JSON object")
        if not isinstance(cost_report.get("artifact_sha256"), str):
            raise ValueError("Q-3D cost report must include the exact artifact SHA-256")
    artifact_path = Path(output).resolve()
    manifest_path = artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")
    manifest: dict
    model = None
    if (cost_report is not None and artifact_path.is_file()
            and _sha256_file(artifact_path) == cost_report["artifact_sha256"]
            and manifest_path.is_file()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_checkpoint_sha256 = _sha256_file(checkpoint)
        source_config_sha256 = _sha256_file(config_path) if config_path is not None else None
        if (manifest.get("artifact_sha256") != cost_report["artifact_sha256"]
                or manifest.get("source_checkpoint_sha256") != source_checkpoint_sha256
                or manifest.get("source_config_sha256") != source_config_sha256):
            raise ValueError("profiled Q-3D artifact provenance differs from the requested checkpoint/config")
        model, artifact_metadata = load_export(
            artifact_path, expected_sha256=cost_report["artifact_sha256"], device="cpu",
        )
        if (artifact_metadata.get("source_checkpoint_sha256") != source_checkpoint_sha256
                or artifact_metadata.get("source_config_sha256") != source_config_sha256):
            raise ValueError("profiled Q-3D artifact metadata provenance mismatch")
        if code_commit is not None:
            manifest["code_commit"] = code_commit
        parameters = int(sum(parameter.numel() for parameter in model.parameters()))
        manifest["cost"] = {
            "parameters": {"status": "COMPUTED", "value": parameters, "unit": "parameters"},
            "flops": cost_report.get("flops", {"status": "NOT_RUN"}),
            "peak_vram_mb": cost_report.get("peak_vram_mb", {"status": "NOT_RUN"}),
            "gpu_latency_ms_per_volume": cost_report.get(
                "gpu_latency_ms_per_volume", {"status": "NOT_RUN"}),
            "cpu_latency_ms_per_volume": cost_report.get(
                "cpu_latency_ms_per_volume", {"status": "NOT_RUN"}),
        }
    else:
        artifact_path, manifest, model = export_checkpoint(
            checkpoint, output, expected_checkpoint_sha256=checkpoint_sha256,
            config_path=config_path, code_commit=code_commit,
        )
        if cost_report is not None and cost_report["artifact_sha256"] != manifest["artifact_sha256"]:
            raise ValueError(
                "cost report is for other artifact bytes; profile the newly exported artifact and retry"
            )
    cold_load = verify_cold_load(model, artifact_path, verify_volumes or [])
    manifest["cold_load"] = cold_load
    _write_json_atomic(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="frozen Q-3D training checkpoint file")
    parser.add_argument("--output", required=True, help="student-only artifact output path")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--config", help="exact JSON recipe for a Q-3D engine resume checkpoint")
    parser.add_argument("--code-commit")
    parser.add_argument("--cost-report", help="q3d_profile.py JSON for this artifact")
    parser.add_argument("--verify-volume", action="append", default=[],
                        help="native NPZ input; supply exactly three to verify cold-load parity")
    args = parser.parse_args(argv)
    result = export(args.checkpoint, args.output, checkpoint_sha256=args.checkpoint_sha256,
                    config_path=args.config,
                    code_commit=args.code_commit, cost_report_path=args.cost_report,
                    verify_volumes=args.verify_volume)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
