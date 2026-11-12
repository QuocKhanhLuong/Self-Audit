#!/usr/bin/env python3
"""Prepare a native Q3D stride proposal; run ERF only with explicit opt-in.

The default path reads only the two TRAIN manifest JSON files and their
successful geometry-repair report. It never reads patient arrays or NIfTI
headers. ``--run-probe`` explicitly opts into the historical synthetic CPU
ERF probe and requires all pinned source and checkpoint assets locally.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.candidate_q3d.geometry_planning import (  # noqa: E402
    CANONICAL_PROBE_CONTRACT,
    NNUNET_COMMIT,
    NNUNET_RESENC_PLANNER_GIT_BLOB_SHA1,
    NNUNET_RESENC_PLANNER_PATH,
    NNUNET_RESENC_PLANNER_SHA256,
    NNUNET_TOPOLOGY_GIT_BLOB_SHA1,
    NNUNET_TOPOLOGY_PATH,
    NNUNET_TOPOLOGY_SHA256,
    prepare_geometry_proposal,
    sha256_file,
)


NNSSL_REGISTRY_REVISION_DECLARED = "3d774e942256bd70e5e264c6b94fa38e25f7ed74"
RESENC_HF_REVISION = "f443c09319988e833b3d9589bf95909e4443d7b2"
RESENC_CHECKPOINT_SHA256 = "ac262d3e8c226c79f9567fddc34d38e011284730ddf2294b5c6c319ce2f22bbd"
RESENC_REGISTRY_SHA256 = "3fbe0fc4da69a403ca99c5c82af680d57ea375b35458913bc6b6eaab70b19610"
RESENC_ARCHITECTURE = "ResEncL"
DYNAMIC_NETWORK_ARCHITECTURES_VERSION = "0.4.4"
CANONICAL_PROBE_PATH = ROOT / CANONICAL_PROBE_CONTRACT["historical_probe"]
CHECKPOINT_INSPECTION_PATH = ROOT / "reports/q3d-resenc-phase0-20261112/checkpoint_inspection.json"
DEFAULT_EXTERNAL_ROOT = Path("/Users/alvinluong/Self-Audit-runs/q3d-resenc-20261112")


def _git_output(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"cannot verify pinned nnU-Net source checkout: {detail}")
    return result.stdout.strip()


def validate_nnunet_source(source_root: str | Path) -> dict[str, Any]:
    """Require the exact pinned nnU-Net commit and topology source bytes."""

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"pinned nnU-Net source checkout does not exist: {root}")
    head = _git_output(root, "rev-parse", "HEAD")
    if head != NNUNET_COMMIT:
        raise ValueError(f"nnU-Net checkout must be {NNUNET_COMMIT}, got {head}")
    files = {}
    for relative_path, expected_blob, expected_sha256 in (
        (NNUNET_TOPOLOGY_PATH, NNUNET_TOPOLOGY_GIT_BLOB_SHA1, NNUNET_TOPOLOGY_SHA256),
        (
            NNUNET_RESENC_PLANNER_PATH,
            NNUNET_RESENC_PLANNER_GIT_BLOB_SHA1,
            NNUNET_RESENC_PLANNER_SHA256,
        ),
    ):
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"pinned nnU-Net source file is missing: {path}")
        actual_blob = _git_output(root, "rev-parse", f"HEAD:{relative_path}")
        actual_sha256 = sha256_file(path)
        if actual_blob != expected_blob or actual_sha256 != expected_sha256:
            raise ValueError(f"pinned nnU-Net source fingerprint mismatch: {relative_path}")
        files[relative_path] = {
            "git_blob_sha1": actual_blob,
            "sha256": actual_sha256,
        }
    return {"repository": "MIC-DKFZ/nnUNet", "commit": head, "files": files}


def validate_probe_assets(
    *,
    checkpoint_path: str | Path,
    registry_path: str | Path,
    checkpoint_inspection_path: str | Path,
    canonical_probe_path: str | Path,
    nnunet_source_path: str | Path,
) -> dict[str, Any]:
    """Fail closed unless all historical source, code, and weight pins match."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve()
    inspection_path = Path(checkpoint_inspection_path).expanduser().resolve()
    probe_path = Path(canonical_probe_path).expanduser().resolve()
    for label, path in (
        ("checkpoint", checkpoint),
        ("ResEnc registry", registry),
        ("checkpoint inspection", inspection_path),
        ("historical ERF probe", probe_path),
    ):
        if not path.is_file():
            raise ValueError(f"pinned {label} asset is missing: {path}")

    checkpoint_sha = sha256_file(checkpoint)
    registry_sha = sha256_file(registry)
    probe_sha = sha256_file(probe_path)
    inspection_sha = sha256_file(inspection_path)
    if checkpoint_sha != RESENC_CHECKPOINT_SHA256:
        raise ValueError("ResEnc checkpoint SHA-256 does not match the pinned weights")
    if registry_sha != RESENC_REGISTRY_SHA256:
        raise ValueError("ResEnc architecture registry SHA-256 differs from the historical source pin")
    if probe_sha != CANONICAL_PROBE_CONTRACT["historical_probe_sha256"]:
        raise ValueError("historical canonical ERF probe source was modified")

    try:
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint inspection is invalid: {exc}") from exc
    if not isinstance(inspection, dict):
        raise ValueError("checkpoint inspection must be a JSON object")
    plan = inspection.get("nnssl_adaptation_plan")
    architecture_plans = plan.get("architecture_plans") if isinstance(plan, dict) else None
    required = {
        "status": "PASS_STRICT_ENCODER",
        "hf_revision": RESENC_HF_REVISION,
        "weights_sha256": RESENC_CHECKPOINT_SHA256,
        "source_sha256": RESENC_REGISTRY_SHA256,
        "encoder_key_count": 448,
        "missing_keys": [],
        "unexpected_keys": [],
    }
    for field, expected in required.items():
        if inspection.get(field) != expected:
            raise ValueError(f"checkpoint inspection {field} differs from its pinned value")
    if not isinstance(architecture_plans, dict) or architecture_plans.get("arch_class_name") != RESENC_ARCHITECTURE:
        raise ValueError("checkpoint inspection architecture is not the pinned ResEncL")
    if not isinstance(plan, dict) or plan.get("key_to_stem") != "encoder.stem" \
            or plan.get("key_to_encoder") != "encoder.stages":
        raise ValueError("checkpoint inspection encoder stem/stage mapping differs from the pin")

    try:
        installed_dna = importlib.metadata.version("dynamic-network-architectures")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("dynamic-network-architectures==0.4.4 is required for the canonical probe") from exc
    if installed_dna != DYNAMIC_NETWORK_ARCHITECTURES_VERSION:
        raise ValueError(
            "dynamic-network-architectures version mismatch: "
            f"expected {DYNAMIC_NETWORK_ARCHITECTURES_VERSION}, got {installed_dna}"
        )

    nnunet_source = validate_nnunet_source(nnunet_source_path)
    return {
        "architecture": RESENC_ARCHITECTURE,
        "nnssl_registry_revision_declared": NNSSL_REGISTRY_REVISION_DECLARED,
        "registry_validation_scope": "pinned file SHA-256; standalone file has no independently checked Git history",
        "hf_revision": RESENC_HF_REVISION,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_inspection_path": str(inspection_path),
        "checkpoint_inspection_sha256": inspection_sha,
        "checkpoint_inspection_validation_scope": "required fields validated; actual file SHA-256 recorded",
        "registry_path": str(registry),
        "registry_sha256": registry_sha,
        "historical_probe_path": str(probe_path),
        "historical_probe_sha256": probe_sha,
        "dynamic_network_architectures_version": installed_dna,
        "nnunet_source": nnunet_source,
    }


def canonical_probe_parameters(stride_schedule: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Return the immutable Phase-0 ERF settings with an explicit stride list."""

    try:
        raw_schedule = tuple(tuple(stage) for stage in stride_schedule)
    except TypeError as exc:
        raise ValueError("canonical ResEnc-L probe requires a sized stride schedule") from exc
    if len(raw_schedule) != 6 or any(len(stage) != 3 for stage in raw_schedule):
        raise ValueError("canonical ResEnc-L probe requires six positive ZYX strides")
    parsed_schedule = []
    for stage in raw_schedule:
        parsed = []
        for axis in stage:
            try:
                number = int(axis)
            except (TypeError, ValueError) as exc:
                raise ValueError("canonical stride axes must be positive integers") from exc
            if isinstance(axis, bool) or number != axis or number < 1:
                raise ValueError("canonical stride axes must be positive integers")
            parsed.append(number)
        parsed_schedule.append(tuple(parsed))
    schedule = tuple(parsed_schedule)
    parameters = dict(CANONICAL_PROBE_CONTRACT)
    parameters["stride_schedule_zyx"] = [list(stage) for stage in schedule]
    return parameters


def _load_canonical_probe(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("q3d_historical_erf_probe", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load historical ERF probe source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def phase_proposal_from_probe_result(result: dict[str, Any]) -> dict[str, Any]:
    """Apply the registered signed-mean offset rule; keep sample maxima diagnostic."""

    summary = result.get("summary")
    decision = result.get("decision_rule")
    if not isinstance(summary, dict) or not isinstance(decision, dict):
        raise ValueError("canonical ERF result lacks summary or decision_rule")
    raw_mean = summary.get("mean_centroid_offset_voxels")
    reported_absolute_mean = summary.get("absolute_mean_centroid_offset_voxels")
    if not isinstance(raw_mean, list) or len(raw_mean) != 3 \
            or not isinstance(reported_absolute_mean, list) or len(reported_absolute_mean) != 3:
        raise ValueError("canonical ERF result must report signed and absolute mean offsets for ZYX")
    mean_offset = [float(value) for value in raw_mean]
    abs_mean = [float(value) for value in reported_absolute_mean]
    threshold = float(decision.get("threshold_voxels", float("nan")))
    if not all(math.isfinite(value) for value in mean_offset + abs_mean + [threshold]) or threshold <= 0:
        raise ValueError("canonical ERF mean offsets and threshold must be finite")
    if any(not math.isclose(abs(value), reported, rel_tol=1e-7, abs_tol=1e-7)
           for value, reported in zip(mean_offset, abs_mean)):
        raise ValueError("canonical ERF absolute mean offsets do not match signed means")
    per_axis_pass = [value <= threshold for value in abs_mean]
    passes = all(per_axis_pass)
    expected_status = "PASS" if passes else "REVIEW_MEAN_OFFSET_OVER_THRESHOLD"
    if result.get("status") != expected_status:
        raise ValueError("canonical ERF status differs from the registered mean-offset gate")
    max_individual = float(summary.get("max_abs_centroid_offset_voxels", float("nan")))
    if not math.isfinite(max_individual):
        raise ValueError("canonical ERF maximum individual offset must be finite for reporting")
    return {
        "status": "MAIN_PHASE_ZERO_CORRECTION_PROPOSED" if passes else "MEASURED_OFFSET_REVIEW_REQUIRED",
        "gate_metric": "all(abs(signed_mean_centroid_offset_axis) <= threshold_voxels)",
        "threshold_voxels": threshold,
        "signed_mean_offset_zyx_voxels": mean_offset,
        "absolute_mean_offset_zyx_voxels": abs_mean,
        "per_axis_pass_zyx": per_axis_pass,
        "proposed_phase": "main_convolution_phase" if passes else "review_measured_mean_offset",
        "proposed_correction_zyx_voxels": [0.0, 0.0, 0.0] if passes else None,
        "maximum_individual_absolute_offset_voxels": max_individual,
        "maximum_individual_is_diagnostic_only": True,
        "maximum_individual_used_for_gate": False,
        "config_lock": "NOT_CREATED_ROOT_OWNS_LOCK",
    }


def _run_canonical_probe(
    proposal: dict[str, Any],
    *,
    checkpoint: Path,
    registry: Path,
    inspection: Path,
    probe_path: Path,
    assets: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    stride_schedule = proposal["geometry"]["planner_output"]["stage_strides_zyx"]
    parameters = canonical_probe_parameters(stride_schedule)
    probe = _load_canonical_probe(probe_path)
    probe.NATIVE_PLANNER_STRIDES = tuple(tuple(stage) for stage in parameters["stride_schedule_zyx"])
    args = SimpleNamespace(
        checkpoint=checkpoint,
        checkpoint_inspection=inspection,
        registry=registry,
        out=out_path,
        md_out=out_path.with_suffix(".canonical.md"),
        samples=parameters["samples"],
        seed=parameters["seed_start"],
        threads=parameters["threads"],
        offset_threshold=parameters["decision_threshold_voxels"],
        shape=list(parameters["shape_zyx"]),
    )
    result = probe.run(args)
    _validate_probe_identity(result, assets)
    if result.get("model", {}).get("stride_schedule_zyx") != parameters["stride_schedule_zyx"]:
        raise ValueError("canonical probe did not use the explicit corrected native stride schedule")
    compute = result.get("compute", {})
    decision = result.get("decision_rule", {})
    if compute.get("samples") != parameters["samples"] or compute.get("seed_start") != parameters["seed_start"]:
        raise ValueError("canonical probe did not preserve the fixed sample/seed contract")
    if compute.get("shape_zyx") != parameters["shape_zyx"] or compute.get("padding_mode") != parameters["padding_mode"]:
        raise ValueError("canonical probe did not preserve the fixed shape/padding contract")
    if decision.get("threshold_voxels") != parameters["decision_threshold_voxels"]:
        raise ValueError("canonical probe did not preserve the fixed threshold")
    if result.get("model", {}).get("stage_index") != parameters["stage_index"]:
        raise ValueError("canonical probe did not target the locked stage-3 feature")
    sample_rows = result.get("samples", [])
    expected_seeds = list(range(parameters["seed_start"], parameters["seed_end_inclusive"] + 1))
    if len(sample_rows) != parameters["samples"] or [row.get("seed") for row in sample_rows] != expected_seeds:
        raise ValueError("canonical probe seed sequence differs from the fixed 42-57 contract")
    if any(row.get("channel_aggregation") != parameters["channel_aggregation"] for row in sample_rows):
        raise ValueError("canonical probe channel aggregation differs from the signed-mean contract")
    if compute.get("random_zscore_input") is not True or compute.get("patient_data") != "NOT_USED":
        raise ValueError("canonical probe must use patient-free random Z-score inputs")
    phase_proposal = phase_proposal_from_probe_result(result)
    result["evidence_class"] = "SOFTWARE_ONLY_SYNTHETIC_ARCHITECTURE_PROBE"
    result["geometry_reprobe"] = {
        "status": "PROBE_COMPLETED_NOT_A_GEOMETRY_LOCK",
        "manifest_lineage": proposal["manifest_lineage"],
        "proposal_geometry": proposal["geometry"],
        "topology_source": proposal["topology_source"],
        "resenc_planner_source": proposal["resenc_planner_source"],
        "probe_contract": parameters,
        "phase_proposal": phase_proposal,
        "invocation": "import pinned historical erf_probe.py and call its run(args); inject explicit stride_schedule_zyx without modifying historical source",
        "pinned_probe_assets": assets,
        "script_source_hashes": {
            **proposal["script_source_hashes"],
            "scripts/q3d_reprobe_geometry.py": sha256_file(Path(__file__)),
            "reports/q3d-resenc-phase0-20261112/erf_probe.py": assets["historical_probe_sha256"],
        },
        "limitations": [
            "This random-input CPU probe is architecture evidence only and is not patient evidence.",
            "The stride schedule remains a proposal; root owns any later config/spec lock.",
            "No voxel equivalence is claimed between repaired header geometry and cached training arrays.",
        ],
    }
    return result


def _validate_probe_identity(result: dict[str, Any], assets: dict[str, Any]) -> None:
    """Check returned compute/model identity against the verified invocation."""
    compute = result.get("compute", {})
    model = result.get("model", {})
    if compute.get("device") != "cpu" or compute.get("threads") != 4:
        raise ValueError("canonical probe requires CPU with exactly four threads")
    if model.get("architecture") != RESENC_ARCHITECTURE:
        raise ValueError("canonical probe architecture differs from ResEncL")
    for name in ("checkpoint_sha256", "registry_sha256"):
        if not assets.get(name) or model.get(name) != assets[name]:
            raise ValueError(f"canonical probe {name} differs from verified assets")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--repaired-manifest", type=Path, required=True)
    parser.add_argument("--repair-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("/tmp/q3d_geometry_reprobe.json"))
    parser.add_argument(
        "--run-probe",
        action="store_true",
        help="explicitly run the canonical 16-sample synthetic CPU ERF probe after all pin checks",
    )
    parser.add_argument("--nnunet-source", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_EXTERNAL_ROOT / "checkpoint_final.pth")
    parser.add_argument("--registry", type=Path, default=DEFAULT_EXTERNAL_ROOT / "architecture_registry.py")
    parser.add_argument("--checkpoint-inspection", type=Path, default=CHECKPOINT_INSPECTION_PATH)
    parser.add_argument("--canonical-probe", type=Path, default=CANONICAL_PROBE_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        proposal = prepare_geometry_proposal(
            source_manifest_path=args.source_manifest,
            repaired_manifest_path=args.repaired_manifest,
            repair_report_path=args.repair_report,
        )
        proposal["execution_mode"] = "RUN_PROBE" if args.run_probe else "PREPARE_ONLY"
        proposal["script_source_hashes"]["scripts/q3d_reprobe_geometry.py"] = sha256_file(Path(__file__))
        output_path = args.out.expanduser().resolve()
        if output_path.exists():
            raise ValueError(f"refusing to overwrite existing output: {output_path}")

        if args.run_probe:
            if args.nnunet_source is None:
                raise ValueError("--nnunet-source is required with --run-probe")
            assets = validate_probe_assets(
                checkpoint_path=args.checkpoint,
                registry_path=args.registry,
                checkpoint_inspection_path=args.checkpoint_inspection,
                canonical_probe_path=args.canonical_probe,
                nnunet_source_path=args.nnunet_source,
            )
            result = _run_canonical_probe(
                proposal,
                checkpoint=args.checkpoint.expanduser().resolve(),
                registry=args.registry.expanduser().resolve(),
                inspection=args.checkpoint_inspection.expanduser().resolve(),
                probe_path=args.canonical_probe.expanduser().resolve(),
                assets=assets,
                out_path=output_path,
            )
        else:
            result = proposal

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"q3d_reprobe_geometry: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "out": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
