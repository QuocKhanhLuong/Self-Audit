"""ACDC plumbing for the native baselines (DSS-US, SGSCN): GT-free producer side only.

Reuses the frozen historical-224 ACDC contract (v12 freeze: manifest, split, grid,
image preprocessing) and the shared ``cardiac_raw_partition.v1`` seal that the frozen
``cardiac_adapter_v2`` consumes. Adds only what the native methods need to consume it:

* one explicit, deterministic input conversion (z-scored central plane -> 8-bit gray,
  replicated to three identical channels) because both methods' locked input
  encodings decode 8-bit images;
* an image-only source object exposing ``records`` and ``read_bgr(record)``;
* a run-level seal over the per-sample shared seals plus the GT-firewall access log,
  with an external write-once receipt (same semantics as the native raw seal receipt).

No GT loader, adapter or metric is imported here.
"""
from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .artifacts import (
    ArtifactError,
    RAW_COMPLETE,
    artifact_directory,
    seal_raw_partition,
    select_manifest_records,
    validate_scientific_execution,
    verify_raw_partition,
)
from .firewall import validate_image_only_value
from .native_artifacts import NativeContractError, ProducerAccessGuard
from .native_protocol import canonical_json, file_hash, value_hash
from .semantic_contract import (
    FROZEN_ADAPTER_SPEC_SHA256,
    FROZEN_SELF_AUDIT_HISTORICAL_224_SHARED_GRID_SHA256,
    array_hash,
)
from .spatial import (
    SELF_AUDIT_HISTORICAL_224_SPATIAL_CONTRACT_VERSION,
    grid_hash,
    read_self_audit_historical_224_context_stack,
    resolve_source_path,
)

ROOT = Path(__file__).resolve().parents[2]
FREEZE_DIR = ROOT / "benchmark_freezes" / "cardiac_benchmark_v12_historical_224"
FROZEN_MANIFEST = FREEZE_DIR / "data" / "acdc_shared_manifest.json"
FROZEN_ADAPTER_SPEC = FREEZE_DIR / "configs" / "adapter_v2_spec.json"
TRACKS_CONTRACT = ROOT / "configs" / "acdc_native_tracks_v1.json"

ADAPTATION_SCHEMA = "native_acdc_adaptation.v1"
RUN_SCHEMA = "native_acdc.raw-run.v1"
SEAL_SCHEMA = "native_acdc.raw-seal.v1"
RECEIPT_SCHEMA = "native_acdc.raw-seal-receipt.v1"
CENTRAL_PLANE = 1  # context [z-1, z, z+1]: the central plane, as CUTS-2.5D and the adapter use it
INPUT_CONVERSION = "acdc_historical224_central_plane_minmax_uint8_gray3_v1"
INPUT_CONVERSION_SPEC = {
    "id": INPUT_CONVERSION,
    "source": "read_self_audit_historical_224_context_stack(record)[1] (frozen v12 preprocessing, 224x224 float32)",
    "scaling": "per-plane min-max in float64: u = (x - min) / (max - min); constant plane -> 0",
    "quantization": "uint8 = floor(255 * u + 0.5)",
    "channels": "gray replicated to 3 identical channels (BGR == RGB)",
    "gt_derived": False,
    "reason": ("both methods' locked input encodings decode 8-bit images (SGSCN bgr_unit_interval = uint8/255; "
               "DSS-US RGB ToTensor of an 8-bit image), while the frozen ACDC contract delivers z-scored floats"),
}
_RUN_ENTRIES = {"run.json", "access_log.json", "acdc_raw_seal.json", "raw"}
_SAMPLE_ENTRIES = {"raw_partition.npy", "metadata.json", "state.json"}


class AcdcBlockedData(Exception):
    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("BLOCKED_DATA: " + "; ".join(self.reasons))


@dataclass(frozen=True)
class AcdcContract:
    manifest: dict
    manifest_hash: str
    shared_grid_hash: str
    adapter_spec_sha256: str
    freeze_id: str
    synthetic: bool = False

    def identity(self):
        return {"freeze_id": self.freeze_id, "manifest_hash": self.manifest_hash,
                "shared_grid_hash": self.shared_grid_hash, "adapter_spec_sha256": self.adapter_spec_sha256,
                "synthetic_fixture": self.synthetic}


def _freeze_payload():
    return json.loads((FREEZE_DIR / "FREEZE_MANIFEST.json").read_text(encoding="utf-8"))


def check_acdc_data(manifest_path, image_root):
    """Precise BLOCKED_DATA reasons for the frozen ACDC image inventory; nothing is downloaded."""
    reasons = []
    if manifest_path is None or not Path(manifest_path).is_file():
        reasons.append(f"frozen ACDC shared manifest not found: {manifest_path}")
    if image_root is None or not Path(image_root).is_dir():
        reasons.append(f"ACDC image root not found: {image_root} (expects training/patientXXX/patientXXX_frameYY.nii)")
    if reasons:
        return reasons
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    root = Path(image_root)
    missing = [record["source"]["locator"] for record in manifest.get("records", [])
               if not (root / record["source"]["locator"]).is_file()]
    if missing:
        unique = sorted(set(missing))
        reasons.append(f"{len(unique)} ACDC source image files absent under {image_root} (e.g. {unique[0]})")
    return reasons


def load_frozen_acdc_contract(manifest_path, image_root):
    """The v12 historical-224 contract: manifest file/hash, grid and adapter spec must match the freeze."""
    payload = _freeze_payload()["scientific_payload"]
    bound = {item["path"]: item["sha256"] for item in payload["bound_freeze_files"]}
    if file_hash(Path(manifest_path)) != bound["data/acdc_shared_manifest.json"]:
        raise ArtifactError("ACDC manifest file differs from the v12 freeze")
    if file_hash(FROZEN_ADAPTER_SPEC) != bound["configs/adapter_v2_spec.json"] or bound["configs/adapter_v2_spec.json"] != FROZEN_ADAPTER_SPEC_SHA256:
        raise ArtifactError("adapter spec differs from the v12 freeze")
    manifest = validate_scientific_execution(manifest_path, image_root)  # verifies every source image hash
    grid = manifest.get("shared_grid", {})
    if (manifest["manifest_hash"] != payload["shared_manifest"]["manifest_sha256"]
            or grid.get("version") != SELF_AUDIT_HISTORICAL_224_SPATIAL_CONTRACT_VERSION
            or manifest.get("shared_grid_hash") != FROZEN_SELF_AUDIT_HISTORICAL_224_SHARED_GRID_SHA256
            or grid_hash(grid) != FROZEN_SELF_AUDIT_HISTORICAL_224_SHARED_GRID_SHA256):
        raise ArtifactError("manifest is not the frozen v12 historical-224 ACDC contract")
    return AcdcContract(manifest=manifest, manifest_hash=manifest["manifest_hash"],
                        shared_grid_hash=manifest["shared_grid_hash"], adapter_spec_sha256=FROZEN_ADAPTER_SPEC_SHA256,
                        freeze_id=_freeze_payload()["freeze_id"])


def select_records(contract, *, split, limit=None, sample_list=None):
    return select_manifest_records(contract.manifest, split=split, limit=limit, sample_list=sample_list)


def central_plane_to_uint8(plane):
    value = np.asarray(plane, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("central plane must be a finite 2-D array")
    low, high = float(value.min()), float(value.max())
    unit = (value - low) / (high - low) if high > low else np.zeros_like(value)
    return np.floor(255.0 * unit + 0.5).astype(np.uint8)


def gray_to_three_channels(gray):
    gray = np.asarray(gray)
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("expected a 2-D uint8 image")
    return np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))


class AcdcImageSource:
    """Image-only ACDC cohort decoded up front: ``records``, ``central_plane`` and ``read_bgr`` from memory.

    Decoding uses the frozen v12 reader on exactly the records' source files, each verified
    against its manifest SHA-256, before any method code runs. Methods then execute inside a
    GT firewall that permits no reads under the image root at all.
    """

    def __init__(self, records, image_root):
        self.records = [dict(record) for record in records]
        self.image_root = Path(image_root).resolve()
        self._planes, self.decoded = {}, {}
        for record in self.records:
            path = resolve_source_path(record, self.image_root)
            if str(path) not in self.decoded:
                if file_hash(path) != record["source"]["sha256"]:
                    raise ArtifactError(f"source image hash mismatch: {record['source']['locator']}")
                self.decoded[str(path)] = record["source"]["sha256"]
            stack = read_self_audit_historical_224_context_stack(record, source_root=self.image_root)
            self._planes[record["sample_id"]] = np.ascontiguousarray(stack[CENTRAL_PLANE], dtype=np.float32)

    def central_plane(self, record):
        return self._planes[record["sample_id"]]

    def read_bgr(self, record):
        return gray_to_three_channels(central_plane_to_uint8(self.central_plane(record)))

    def input_receipt(self, record):
        plane = self.central_plane(record)
        model_input = self.read_bgr(record)
        return {"input_conversion": INPUT_CONVERSION, "central_image_sha256": array_hash(plane),
                "model_input_uint8_sha256": array_hash(model_input),
                "central_plane_min": float(plane.min()), "central_plane_max": float(plane.max())}


def load_adaptation(path):
    """An ACDC adaptation config: binds one runnable native profile, unchanged, to the ACDC contract."""
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != ADAPTATION_SCHEMA:
        raise ValueError(f"not an ACDC adaptation config: {path}")
    profile = config["source_profile"]
    source = ROOT / profile["path"]
    if file_hash(source) != profile["sha256"]:
        raise ValueError("source native profile changed since the ACDC adaptation was declared")
    contract = config["acdc_contract"]
    if (contract["shared_grid_sha256"] != FROZEN_SELF_AUDIT_HISTORICAL_224_SHARED_GRID_SHA256
            or contract["adapter_spec_sha256"] != FROZEN_ADAPTER_SPEC_SHA256
            or contract["input_conversion"] != INPUT_CONVERSION_SPEC
            or file_hash(TRACKS_CONTRACT) != contract["tracks_contract_sha256"]):
        raise ValueError("ACDC adaptation is not bound to the frozen ACDC contract")
    return config, source, value_hash(config)


@contextlib.contextmanager
def working_directory(path):
    """The shared locator firewall probes relative locators against the working directory.
    Inside the GT firewall that directory is the run's own output root, never the image root."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def access_guard(source, *, output_root, trusted_roots=(), readable_files=()):
    """GT firewall around method code: nothing under the image root (images are already decoded)."""
    return ProducerAccessGuard(images=[], output_root=output_root, image_root=source.image_root,
                               trusted_roots=trusted_roots, readable_files=readable_files)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    temporary.replace(path)


class AcdcRawRun:
    """Writes shared ``cardiac_raw_partition.v1`` artifacts for one cohort and seals the run."""

    def __init__(self, output_root, *, contract, records, method, baseline_mode, adaptation, adaptation_sha256,
                 provenance):
        self.root = Path(output_root).resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise NativeContractError("refuse nonempty output root")
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        self.method, self.mode = method, baseline_mode
        self.adaptation_sha256 = adaptation_sha256
        self.samples = {}
        self.run = {"schema": RUN_SCHEMA, "method": method, "baseline_mode": baseline_mode,
                    "adaptation": adaptation, "adaptation_sha256": adaptation_sha256,
                    "contract": contract.identity(), "records": [dict(record) for record in records],
                    "expected_samples": [record["sample_id"] for record in records], "provenance": provenance}
        validate_image_only_value(self.run, where="acdc_run")
        _write_json(self.root / "run.json", self.run)

    def write(self, record, partition, *, seed, metadata):
        if (self.root / "acdc_raw_seal.json").exists():
            raise NativeContractError("raw run is already sealed")
        sample_id = record["sample_id"]
        if sample_id not in self.run["expected_samples"] or sample_id in self.samples:
            raise NativeContractError("unknown or duplicate output sample")
        partition = np.asarray(partition)
        if not np.issubdtype(partition.dtype, np.integer) or partition.size == 0 or np.any(partition < 0):
            raise NativeContractError("raw partition must be a nonnegative integer map")
        artifact = seal_raw_partition(
            self.root / "raw", np.ascontiguousarray(partition, dtype=np.int32), record=record,
            baseline_name=self.method, baseline_mode=self.mode, manifest_hash=self.contract.manifest_hash,
            shared_grid_hash=self.contract.shared_grid_hash,
            repository=self.run["provenance"]["repository"], code_identity_value=self.run["provenance"]["code_identity"],
            baseline_config_hash=self.adaptation_sha256, seed=int(seed), baseline_metadata=metadata)
        self.samples[sample_id] = artifact
        return artifact

    def seal(self, access_log):
        if set(self.samples) != set(self.run["expected_samples"]):
            raise NativeContractError("cannot seal incomplete cohort")
        _write_json(self.root / "access_log.json", access_log)
        if any(not entry["allowed"] for entry in access_log):
            raise NativeContractError("cannot seal a GT-firewall violation")
        rows = []
        for sample_id in sorted(self.samples):
            directory = self.samples[sample_id].directory
            rows.append({"sample_id": sample_id, "directory": str(directory.relative_to(self.root)),
                         "scientific_payload_hash": self.samples[sample_id].metadata["scientific_payload_hash"],
                         "raw_partition_sha256": self.samples[sample_id].metadata["raw_partition_sha256"],
                         **{f"{name.split('.')[0]}_file_sha256": file_hash(directory / name) for name in sorted(_SAMPLE_ENTRIES)}})
        seal = {"schema": SEAL_SCHEMA, "status": RAW_COMPLETE, "run_sha256": file_hash(self.root / "run.json"),
                "access_log_sha256": file_hash(self.root / "access_log.json"), "samples": rows}
        seal["seal_sha256"] = value_hash(seal)
        with (self.root / "acdc_raw_seal.json").open("xb") as stream:  # exclusive: never replace a seal
            stream.write(canonical_json(seal) + b"\n")
        return verify_acdc_raw_run(self.root)


def verify_acdc_raw_run(root):
    """Verify the run seal, the access log, and every per-sample shared raw seal."""
    root = Path(root).resolve()
    seal = json.loads((root / "acdc_raw_seal.json").read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in seal.items() if key != "seal_sha256"}
    if seal.get("schema") != SEAL_SCHEMA or seal.get("status") != RAW_COMPLETE or value_hash(unsigned) != seal.get("seal_sha256"):
        raise NativeContractError("invalid ACDC raw seal")
    for name, key in (("run.json", "run_sha256"), ("access_log.json", "access_log_sha256")):
        if file_hash(root / name) != seal[key]:
            raise NativeContractError(f"sealed file changed: {name}")
    if {path.name for path in root.iterdir()} != _RUN_ENTRIES:
        raise NativeContractError("unsealed entries in ACDC raw root")
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if any(not entry["allowed"] for entry in json.loads((root / "access_log.json").read_text(encoding="utf-8"))):
        raise NativeContractError("sealed access violation")
    if value_hash(run["adaptation"]) != run["adaptation_sha256"]:
        raise NativeContractError("adaptation receipt changed")
    if [row["sample_id"] for row in seal["samples"]] != sorted(run["expected_samples"]):
        raise NativeContractError("sealed cohort mismatch")
    records = {record["sample_id"]: record for record in run["records"]}
    artifacts = {}
    for row in seal["samples"]:
        directory = root / row["directory"]
        expected_dir = artifact_directory(root / "raw", baseline_name=run["method"], baseline_mode=run["baseline_mode"],
                                          sample_id=row["sample_id"])
        if directory.resolve() != expected_dir.resolve() or {path.name for path in directory.iterdir()} != _SAMPLE_ENTRIES:
            raise NativeContractError("unsealed sample entries")
        for name in _SAMPLE_ENTRIES:
            if file_hash(directory / name) != row[f"{name.split('.')[0]}_file_sha256"]:
                raise NativeContractError(f"sealed sample file changed: {name}")
        artifact = verify_raw_partition(directory, expected={
            "sample_id": row["sample_id"], "baseline_name": run["method"], "baseline_mode": run["baseline_mode"],
            "baseline_config_hash": run["adaptation_sha256"], "shared_manifest_hash": run["contract"]["manifest_hash"],
            "shared_grid_hash": run["contract"]["shared_grid_hash"],
            "source_image_sha256": records[row["sample_id"]]["source"]["sha256"]})
        if (artifact.metadata["scientific_payload_hash"] != row["scientific_payload_hash"]
                or artifact.metadata["raw_partition_sha256"] != row["raw_partition_sha256"]):
            raise NativeContractError("raw artifact does not match the run seal")
        artifacts[row["sample_id"]] = artifact
    return {"run": run, "seal": seal, "artifacts": artifacts}


def _receipt_body(verified, root):
    run, seal = verified["run"], verified["seal"]
    return {"schema": RECEIPT_SCHEMA, "status": "RAW_SEAL_FINALIZED", "method": run["method"],
            "baseline_mode": run["baseline_mode"], "adaptation_sha256": run["adaptation_sha256"],
            "contract": run["contract"], "seal_sha256": seal["seal_sha256"],
            "seal_file_sha256": file_hash(root / "acdc_raw_seal.json"),
            "samples": [{"sample_id": row["sample_id"], "raw_partition_sha256": row["raw_partition_sha256"],
                         "scientific_payload_hash": row["scientific_payload_hash"]} for row in seal["samples"]]}


def write_acdc_seal_receipt(raw_root, receipt_path):
    """Write-once receipt outside the raw root; Track A/B verify against it before reading anything else."""
    root = Path(raw_root).resolve()
    verified = verify_acdc_raw_run(root)
    receipt_path = Path(receipt_path).absolute()
    if receipt_path.parent.resolve() == root or root in receipt_path.parent.resolve().parents:
        raise NativeContractError("seal receipt must be stored outside the raw root")
    receipt = _receipt_body(verified, root)
    receipt["receipt_sha256"] = value_hash(receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("xb") as stream:  # FileExistsError: never overwrite a receipt
        stream.write(canonical_json(receipt) + b"\n")
    os.chmod(receipt_path, 0o444)
    return receipt


def verify_acdc_seal_receipt(raw_root, receipt_path):
    root = Path(raw_root).resolve()
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("schema") != RECEIPT_SCHEMA or value_hash(unsigned) != receipt.get("receipt_sha256"):
        raise NativeContractError("invalid ACDC seal receipt")
    verified = verify_acdc_raw_run(root)
    if _receipt_body(verified, root) != unsigned:
        raise NativeContractError("raw run does not match its finalized seal receipt")
    return {**verified, "receipt": receipt}


def run_acdc_producer(*, contract, records, image_root, output_root, receipt_path, method, baseline_mode,
                      adaptation, adaptation_sha256, provenance,
                      produce: Callable[[AcdcImageSource], Mapping[str, Any]],
                      seed_for_record: Callable[[Mapping[str, Any]], int],
                      trusted_roots=(), readable_files=()):
    """Image-only cohort -> method -> anonymous [224,224] raw maps -> per-sample seals -> run seal -> receipt.

    ``produce(source)`` returns ``{sample_id: (partition, method_receipt)}``; it runs inside the
    GT firewall and sees only the image source. Raw IDs are stored as produced (no remapping).
    """
    if not records:
        raise ValueError("empty ACDC cohort")
    source = AcdcImageSource(records, image_root)
    output = Path(output_root).resolve()
    receipt_parent = Path(receipt_path).absolute().parent.resolve()
    if output == source.image_root or source.image_root in output.parents or output in source.image_root.parents:
        raise ValueError("output root must not overlap the image root")
    if receipt_parent == output or output in receipt_parent.parents:
        raise ValueError("seal receipt must be stored outside the output root")
    decoded = sorted({(r["source"]["locator"], r["source"]["sha256"]) for r in source.records})
    provenance = {**provenance, "decoded_sources": [{"locator": locator, "sha256": digest} for locator, digest in decoded],
                  "decode_policy": "frozen v12 reader on the cohort's exact source files, hash-verified, before any method code"}
    run = AcdcRawRun(output, contract=contract, records=source.records, method=method, baseline_mode=baseline_mode,
                     adaptation=adaptation, adaptation_sha256=adaptation_sha256, provenance=provenance)
    guard = access_guard(source, output_root=run.root, trusted_roots=trusted_roots, readable_files=readable_files)
    with working_directory(run.root), guard:
        outputs = produce(source)
        for record in source.records:
            partition, method_receipt = outputs[record["sample_id"]]
            target_hw = list(record["shared_grid"]["target_hw"])
            if list(np.shape(partition)) != target_hw:
                raise NativeContractError(f"raw map {list(np.shape(partition))} is not the shared grid {target_hw}")
            metadata = {**method_receipt, **source.input_receipt(record)}
            run.write(record, partition, seed=seed_for_record(record), metadata=metadata)
    verified = run.seal(guard.log)
    receipt = write_acdc_seal_receipt(run.root, receipt_path)
    return {"verified": verified, "receipt": receipt, "source": source}
