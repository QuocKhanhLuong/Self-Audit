"""Track A / Track B for a sealed DSS-US or SGSCN ACDC raw run (same raw run for both tracks).

  track-a      GT-free: frozen cardiac_adapter_v2 -> sealed BG/RV/MYO/LV/VOID semantic maps
  evaluate-a   common ACDC metrics on the Track A semantic maps (reads GT)
  track-b      raw_id_majority_vote_v1 GT-assisted diagnostic + the same common metrics

Every command verifies the external seal receipt and the run seal before reading anything else.
Track B results are diagnostic only and must never feed producers, hyperparameters, seeds,
Track A or model selection.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("track-a", "evaluate-a", "track-b"):
        command = commands.add_parser(name)
        command.add_argument("--raw-root", type=Path, required=True)
        command.add_argument("--receipt", type=Path, required=True)
        if name in ("track-a", "evaluate-a"):
            command.add_argument("--image-root", type=Path, required=True)
            command.add_argument("--semantic-root", type=Path, required=True)
        if name in ("evaluate-a", "track-b"):
            command.add_argument("--gt-root", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from shared_benchmark import acdc_tracks
    if args.command == "track-a":
        result = acdc_tracks.run_track_a(args.raw_root, args.receipt, args.image_root, args.semantic_root,
                                         trusted_roots=[sys.prefix, sys.base_prefix, ROOT / "src", ROOT / "scripts"])
        print(json.dumps({"status": "TRACK_A_SEMANTIC_COMPLETE", "index_sha256": result["index_sha256"],
                          "samples": len(result["samples"])}))
    elif args.command == "evaluate-a":
        summary = acdc_tracks.evaluate_track_a(args.raw_root, args.receipt, args.image_root, args.semantic_root,
                                               args.gt_root, args.output_dir)
        print(json.dumps({"status": "TRACK_A_EVALUATED", "output": str(args.output_dir / "summary.json"),
                          "volumes": len(summary["per_volume"])}))
    else:
        summary = acdc_tracks.evaluate_track_b(args.raw_root, args.receipt, args.gt_root, args.output_dir)
        print(json.dumps({"status": "TRACK_B_DIAGNOSTIC_COMPLETE", "kind": summary["kind"],
                          "output": str(args.output_dir / "summary.json"), "volumes": len(summary["per_volume"])}))
    return 0


if __name__ == "__main__":
    from environment_contract import enforce_official_entrypoint

    enforce_official_entrypoint(__file__)  # official runs require self-audit-canonical v1
    raise SystemExit(main())
