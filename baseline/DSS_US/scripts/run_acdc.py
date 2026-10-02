"""GT-free DSS-US producer on the frozen ACDC contract (v12 historical-224).

Runs the runnable paper-faithful-with-declared-conventions Step II (DSS step2) profile
unchanged: frozen ACDC central plane -> DINO ViT-S/8 (pinned source, hashed checkpoint) ->
Step I eigensegments -> official-fallback Step II over the run's cohort -> anonymous
[224,224] raw maps -> shared seals + run seal + external receipt.
Step I/CRF, preprocessing, affinity and Ours-step2 rows stay BLOCKED_PROTOCOL.
Exit codes: 0 RAW_COMPLETE / ready, 2 BLOCKED_PROTOCOL, 3 BLOCKED_DATA.
"""
import argparse
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE / "src"), str(ROOT / "src"), str(BASE / "scripts")]
from shared_benchmark.native_protocol import ProtocolBlocked, load_lock  # noqa: E402
import run_native  # noqa: E402  (shared DSS-US producer flow: produce, warm_up_frameworks, check_dino)


def load_profile(adaptation_path):
    from dss_us import paper_faithful as pf
    from shared_benchmark.acdc_native import load_adaptation
    try:
        adaptation, source, adaptation_sha = load_adaptation(adaptation_path)
    except (ValueError, KeyError, OSError) as error:
        raise ProtocolBlocked([f"ACDC adaptation invalid: {error}"])
    if adaptation["method"] != "DSS-US":
        raise ProtocolBlocked(["not a DSS-US ACDC adaptation"])
    config = load_lock(source)
    if config.get("implementation") != "paper_faithful_reimplementation" or config.get("stage") != "II":
        raise ProtocolBlocked(["only the runnable paper-faithful Step II profile is adapted to ACDC"])
    settings = pf.dss_settings(config)
    try:
        pf.validate_dss_step2_settings(settings)
    except pf.ProtocolSettingError as error:
        raise ProtocolBlocked([str(error)])
    return adaptation, adaptation_sha, config, settings


def run(*, contract, records, image_root, output, receipt, adaptation_path, dino_repo, dino_checkpoint,
        dino_checkpoint_sha256, threads=1, environment=None, extra_trusted_roots=()):
    """Library entry (used by the CLI after the frozen contract and data checks).

    ``extra_trusted_roots`` lets a test harness's own code directory appear on the call stack
    (stack formatting stats caller files); it never covers the image root or any GT location.
    """
    import torch
    import cv2
    import sklearn
    import skimage
    from dss_us.features import DinoKeys
    from shared_benchmark.acdc_native import run_acdc_producer
    from shared_benchmark.artifacts import code_identity, repository_identity
    from shared_benchmark.native_artifacts import environment_receipt
    adaptation, adaptation_sha, config, settings = load_profile(adaptation_path)
    torch.set_num_threads(threads)
    dino = DinoKeys(dino_repo, dino_checkpoint, checkpoint_sha256=dino_checkpoint_sha256, device="cpu")
    run_native.warm_up_frameworks(dino)
    sources = [*sorted((BASE / "src").rglob("*.py")), Path(__file__), BASE / "scripts/run_native.py",
               ROOT / "src/shared_benchmark/acdc_native.py", ROOT / "src/shared_benchmark/artifacts.py",
               ROOT / "src/shared_benchmark/native_artifacts.py", ROOT / "src/shared_benchmark/spatial.py",
               ROOT / "src/environment_contract.py"]
    cohort = sorted(record["sample_id"] for record in records)
    provenance = {"repository": repository_identity(ROOT), "code_identity": code_identity(sources, repo_root=ROOT),
                  "environment": environment_receipt(), "environment_contract": environment, "threads": threads,
                  "implementation_kind": config["implementation"], "profile_class": config["profile_class"],
                  "paper_equivalence": config["paper_equivalence"], "source_profile": adaptation["source_profile"],
                  "dino": dino.receipt, "step2_fit_cohort": cohort,
                  "upstream_evidence_commit": "d4ac44c60df18b921c590796f6994a4c8ac0726c"}

    def produce(source):
        outputs = run_native.produce(source, settings, keys_provider=dino)
        for sample_id, (_partition, receipt) in outputs.items():
            receipt["step2_fit_cohort_size"] = len(cohort)
        return outputs

    return run_acdc_producer(
        contract=contract, records=records, image_root=image_root, output_root=output, receipt_path=receipt,
        method="DSS-US", baseline_mode=adaptation["adaptation_id"], adaptation=adaptation, adaptation_sha256=adaptation_sha,
        provenance=provenance, produce=produce, seed_for_record=lambda _record: int(settings["kmeans_seed"]),
        trusted_roots=[sys.prefix, sys.base_prefix, BASE / "src", BASE / "scripts", ROOT / "src", Path(dino_repo),
                       Path(torch.__file__).parent, Path(cv2.__file__).parent, Path(sklearn.__file__).parent,
                       Path(skimage.__file__).parent, *extra_trusted_roots],
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
    parser.add_argument("--dino-repo", type=Path, default=ROOT / ".scratch/dino")
    parser.add_argument("--dino-checkpoint", type=Path)
    parser.add_argument("--dino-checkpoint-sha256")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--check-protocol", action="store_true")
    args = parser.parse_args()
    try:
        adaptation, _, config, _ = load_profile(args.adaptation)
    except ProtocolBlocked as error:
        print(json.dumps({"status": error.status, "reasons": error.reasons}))
        return 2
    if args.check_protocol:
        print(json.dumps({"status": "ACDC_PRODUCER_READY", "adaptation": adaptation["adaptation_id"],
                          "profile_class": config["profile_class"], "paper_equivalence": config["paper_equivalence"]}))
        return 0
    from environment_contract import require_official_environment
    environment = require_official_environment()
    from shared_benchmark.acdc_native import check_acdc_data, load_frozen_acdc_contract, select_records
    reasons = check_acdc_data(args.manifest, args.image_root)
    reasons += run_native.check_dino(args.dino_repo, args.dino_checkpoint, args.dino_checkpoint_sha256)
    if reasons:
        print(json.dumps({"status": "BLOCKED_DATA", "reasons": reasons}))
        return 3
    if args.output is None or args.receipt is None or args.threads < 1:
        parser.error("--output, --receipt and positive --threads required")
    contract = load_frozen_acdc_contract(args.manifest, args.image_root)
    records = select_records(contract, split=args.split, limit=args.limit, sample_list=args.samples)
    result = run(contract=contract, records=records, image_root=args.image_root, output=args.output,
                 receipt=args.receipt, adaptation_path=args.adaptation, dino_repo=args.dino_repo,
                 dino_checkpoint=args.dino_checkpoint, dino_checkpoint_sha256=args.dino_checkpoint_sha256,
                 threads=args.threads, environment=environment)
    print(json.dumps({"status": "RAW_COMPLETE", "seal_sha256": result["verified"]["seal"]["seal_sha256"],
                      "receipt_sha256": result["receipt"]["receipt_sha256"], "samples": len(records),
                      "paper_equivalence": config["paper_equivalence"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
