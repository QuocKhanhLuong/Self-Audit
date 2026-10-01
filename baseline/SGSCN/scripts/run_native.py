# SPDX-License-Identifier: GPL-3.0
"""GT-free SGSCN native runner. Paper profiles are gated; reference runs are labeled."""
import argparse
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE / "src"), str(ROOT / "src")]
from shared_benchmark.native_protocol import ProtocolBlocked, load_lock, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images-manifest", type=Path)
    parser.add_argument("--image-root", type=Path,
                        help="image-only staging directory holding exactly the inventory images (no GT)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--check-protocol", action="store_true")
    args = parser.parse_args()
    try:
        config = load_lock(args.config)
    except ProtocolBlocked as error:
        print(json.dumps({"status": error.status, "reasons": error.reasons}))
        return 2
    if args.check_protocol:
        print(json.dumps({"status": "REFERENCE_READY", "paper_equivalence": config["paper_equivalence"]}))
        return 0
    if args.images_manifest is None or args.image_root is None or args.output is None or args.threads < 1:
        parser.error("--images-manifest, --image-root, --output and positive --threads required")
    # Warm framework/library imports before the allowlist guard, never a GT evaluator.
    import torch
    import cv2
    import torch._dynamo  # initialize framework/cache discovery before dataset access guard
    from sgscn.producer import predict_native, sample_seed
    from shared_benchmark.native_artifacts import ImageInventory, RawRunWriter, ProducerAccessGuard, environment_receipt, array_hash
    torch.set_num_threads(args.threads)
    inventory = ImageInventory(args.images_manifest, image_root=args.image_root)
    if inventory.document["dataset"] != config["dataset"]:
        raise ValueError("native dataset/profile mismatch")
    sources = [*sorted((BASE / "src").rglob("*.py")), Path(__file__),
               ROOT / "src/shared_benchmark/native_artifacts.py", ROOT / "src/shared_benchmark/native_protocol.py"]
    provenance = {"base_seed": args.seed, "seed_policy": "sha256_base_seed_and_inventory_sample_id_v1",
                  "code_files": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                  "environment": environment_receipt(), "device": args.device, "threads": args.threads,
                  "implementation_kind": "official_code_reference",
                  "upstream_commit": "592efb6e72ceeef15c8be0630a4673eda5dce6f5"}
    output = args.output.resolve()
    if output == inventory.image_root or inventory.image_root in output.parents or output in inventory.image_root.parents:
        raise ValueError("output root must not overlap the image-only staging root")
    writer = RawRunWriter(args.output, inventory=inventory, config=config, provenance=provenance)
    guard = ProducerAccessGuard(images=[r["resolved_image_path"] for r in inventory.records],
                                output_root=writer.root, image_root=inventory.image_root,
                                readable_files=[inventory.path, args.config, *sources],
                                trusted_roots=[sys.prefix, sys.base_prefix, BASE / "src", ROOT / "src",
                                               Path(torch.__file__).parent, Path(cv2.__file__).parent])
    try:
        with guard:
            for record in inventory.records:
                image = inventory.read_bgr(record)
                partition, receipt = predict_native(image, config["scientific"],
                                                    seed=sample_seed(args.seed, record["sample_id"]), device=args.device)
                receipt["input_native_hw"] = list(image.shape[:2])
                receipt["decoded_image_sha256"] = array_hash(image)
                receipt["model_input_sha256"] = array_hash(image.transpose(2, 0, 1).astype("float32") / 255.)
                writer.write(record, partition, receipt)
        verified = writer.seal(guard.log)
    except Exception:
        (writer.root / "FAILED.json").write_text(json.dumps({"status": "RAW_INCOMPLETE", "access_log": guard.log}), encoding="utf-8")
        raise
    print(json.dumps({"status": "RAW_COMPLETE", "seal_sha256": verified["seal"]["seal_sha256"],
                      "paper_equivalence": "UNRESOLVED", "kind": "official_code_reference"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
