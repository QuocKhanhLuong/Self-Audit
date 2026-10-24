#!/usr/bin/env python3
"""Create a strict image-only native-geometry manifest for TRAIN patients."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.memory_bank import create_native_image_manifest, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True,
                        help="self_audit.data.v2 role=training JSON; masks are never followed")
    parser.add_argument("--train-manifest-sha256", required=True)
    parser.add_argument("--native-root", type=Path, required=True,
                        help="root containing <patientNNN>/<patientNNN>_4d.nii[.gz]")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-patients", type=int,
                        help="deterministic sorted-ID smoke bound; output is marked incomplete")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output must be new; preserve existing manifests")
    report = create_native_image_manifest(
        args.train_manifest, args.train_manifest_sha256, args.native_root, args.out,
        max_patients=args.max_patients,
    )
    print(json.dumps({"manifest": str(args.out.resolve()), "manifest_sha256": sha256_file(args.out),
                      "coverage": report["coverage"], "mask_paths_read": 0,
                      "geometry_source": report["geometry_source"]}, indent=2))


if __name__ == "__main__":
    main()
