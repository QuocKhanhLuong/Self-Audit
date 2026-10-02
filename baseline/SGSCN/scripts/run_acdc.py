# SPDX-License-Identifier: GPL-3.0
"""GT-free SGSCN producer on the frozen ACDC contract (v12 historical-224).

Runs one runnable native profile unchanged (paper-faithful with declared conventions or
official reference) on the frozen ACDC central planes and seals anonymous [224,224] raw maps.
Exit codes: 0 RAW_COMPLETE / ready, 2 BLOCKED_PROTOCOL, 3 BLOCKED_DATA.
"""
import argparse
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE / "src"), str(ROOT / "src")]
from shared_benchmark.native_protocol import ProtocolBlocked, file_hash, load_lock  # noqa: E402

BASE_SEED = 1  # the native runner's --seed default; unchanged on ACDC


def load_profile(adaptation_path):
    """(adaptation, sha, profile config, settings, predict, sample_seed, kind); raises ProtocolBlocked."""
    from shared_benchmark.acdc_native import load_adaptation
    try:
        adaptation, source, adaptation_sha = load_adaptation(adaptation_path)
    except (ValueError, KeyError, OSError) as error:
        raise ProtocolBlocked([f"ACDC adaptation invalid: {error}"])
    if adaptation["method"] != "SGSCN":
        raise ProtocolBlocked(["not an SGSCN ACDC adaptation"])
    config = load_lock(source)
    if config["implementation"] == "paper_faithful_reimplementation":
        from sgscn.paper_faithful import (ProtocolSettingError, predict_paper_faithful, sample_seed,
                                          settings_from_config, validate_settings)
        try:
            settings = validate_settings(settings_from_config(config))
        except ProtocolSettingError as error:
            raise ProtocolBlocked([str(error)])
        return adaptation, adaptation_sha, config, settings, predict_paper_faithful, sample_seed, config["implementation"]
    from sgscn.producer import predict_native, sample_seed
    return adaptation, adaptation_sha, config, config["scientific"], predict_native, sample_seed, "official_code_reference"


def make_produce(settings, predict, sample_seed, *, device):
    def produce(source):
        from shared_benchmark.native_artifacts import array_hash
        outputs = {}
        for record in source.records:
            image = source.read_bgr(record)
            partition, receipt = predict(image, settings, seed=sample_seed(BASE_SEED, record["sample_id"]), device=device)
            receipt.pop("seconds", None)  # operational timing is not part of the sealed scientific payload
            receipt["input_native_hw"] = list(image.shape[:2])
            receipt["model_input_sha256"] = array_hash(image.transpose(2, 0, 1).astype("float32") / 255.)
            outputs[record["sample_id"]] = (partition, receipt)
        return outputs
    return produce


def run(*, contract, records, image_root, output, receipt, adaptation_path, device="cpu", threads=1, environment=None, extra_trusted_roots=()):
    """Library entry (used by the CLI after the frozen contract and data checks).

    ``extra_trusted_roots`` lets a test harness's own code directory appear on the call stack
    (stack formatting stats caller files); it never covers the image root or any GT location.
    """
    import torch
    import cv2
    import torch._dynamo  # noqa: F401  framework discovery before the GT firewall
    from shared_benchmark.acdc_native import run_acdc_producer
    from shared_benchmark.artifacts import code_identity, repository_identity
    from shared_benchmark.native_artifacts import environment_receipt
    adaptation, adaptation_sha, config, settings, predict, sample_seed, kind = load_profile(adaptation_path)
    torch.set_num_threads(threads)
    sources = [*sorted((BASE / "src").rglob("*.py")), Path(__file__), ROOT / "src/shared_benchmark/acdc_native.py",
               ROOT / "src/shared_benchmark/artifacts.py", ROOT / "src/shared_benchmark/native_artifacts.py",
               ROOT / "src/shared_benchmark/spatial.py", ROOT / "src/environment_contract.py"]
    provenance = {"repository": repository_identity(ROOT), "code_identity": code_identity(sources, repo_root=ROOT),
                  "environment": environment_receipt(), "environment_contract": environment, "device": device,
                  "threads": threads, "implementation_kind": kind,
                  "profile_class": config.get("profile_class", "OFFICIAL_REFERENCE"),
                  "paper_equivalence": config["paper_equivalence"], "source_profile": adaptation["source_profile"],
                  "base_seed": BASE_SEED, "seed_policy": "sha256_base_seed_and_inventory_sample_id_v1",
                  "upstream_commit": "592efb6e72ceeef15c8be0630a4673eda5dce6f5"}
    return run_acdc_producer(
        contract=contract, records=records, image_root=image_root, output_root=output, receipt_path=receipt,
        method="SGSCN", baseline_mode=adaptation["adaptation_id"], adaptation=adaptation, adaptation_sha256=adaptation_sha,
        provenance=provenance, produce=make_produce(settings, predict, sample_seed, device=device),
        seed_for_record=lambda record: sample_seed(BASE_SEED, record["sample_id"]),
        trusted_roots=[sys.prefix, sys.base_prefix, BASE / "src", ROOT / "src", Path(torch.__file__).parent,
                       Path(cv2.__file__).parent, *extra_trusted_roots],
        readable_files=[Path(adaptation_path), *sources])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adaptation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "benchmark_freezes/cardiac_benchmark_v12_historical_224/data/acdc_shared_manifest.json")
    parser.add_argument("--image-root", type=Path, help="ACDC root holding training/patientXXX/*.nii")
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample", action="append", dest="samples")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path, help="external write-once seal receipt (outside --output)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--check-protocol", action="store_true")
    args = parser.parse_args()
    try:
        adaptation, _, config, *_ = load_profile(args.adaptation)
    except ProtocolBlocked as error:
        print(json.dumps({"status": error.status, "reasons": error.reasons}))
        return 2
    if args.check_protocol:
        print(json.dumps({"status": "ACDC_PRODUCER_READY", "adaptation": adaptation["adaptation_id"],
                          "profile_class": adaptation["source_profile"]["profile_class"],
                          "paper_equivalence": config["paper_equivalence"]}))
        return 0
    from environment_contract import require_official_environment
    environment = require_official_environment()
    from shared_benchmark.acdc_native import check_acdc_data, load_frozen_acdc_contract, select_records
    reasons = check_acdc_data(args.manifest, args.image_root)
    if reasons:
        print(json.dumps({"status": "BLOCKED_DATA", "reasons": reasons}))
        return 3
    if args.output is None or args.receipt is None or args.threads < 1:
        parser.error("--output, --receipt and positive --threads required")
    contract = load_frozen_acdc_contract(args.manifest, args.image_root)
    records = select_records(contract, split=args.split, limit=args.limit, sample_list=args.samples)
    result = run(contract=contract, records=records, image_root=args.image_root, output=args.output,
                 receipt=args.receipt, adaptation_path=args.adaptation, device=args.device, threads=args.threads,
                 environment=environment)
    print(json.dumps({"status": "RAW_COMPLETE", "seal_sha256": result["verified"]["seal"]["seal_sha256"],
                      "receipt_sha256": result["receipt"]["receipt_sha256"], "samples": len(records),
                      "paper_equivalence": config["paper_equivalence"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
