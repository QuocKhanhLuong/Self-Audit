"""Write the ACDC adaptation configs that bind each runnable native profile, unchanged, to ACDC.

An adaptation declares dataset/input/output plumbing only. The source profile file is bound
by SHA-256 and its scientific values are never copied or overridden. Refuses to change an
existing adaptation file unless ``--force-rewrite`` is given (rewrites are visible in git).
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_benchmark.acdc_native import (  # noqa: E402
    ADAPTATION_SCHEMA, FREEZE_DIR, INPUT_CONVERSION_SPEC, TRACKS_CONTRACT, _freeze_payload,
)
from shared_benchmark.native_protocol import file_hash, load_lock  # noqa: E402
from shared_benchmark.semantic_contract import (  # noqa: E402
    FROZEN_ADAPTER_SPEC_SHA256, FROZEN_SELF_AUDIT_HISTORICAL_224_SHARED_GRID_SHA256,
)

SGSCN_UNCHANGED = ["architecture", "layer_order", "output_normalization", "optimizer", "momentum", "learning_rate",
                   "losses_and_reduction", "context_density", "input_encoding", "stopping_rule", "max_iterations",
                   "min_labels", "final_forward_mode", "initialization", "seed_derivation"]
DSS_UNCHANGED = ["dino_vits8_feature_path", "dino_input_policy", "feature_l2_normalization", "step1_affinity",
                 "eigenvectors", "step1_segments_15", "kmeans_settings_and_seed", "background_relabel",
                 "step2_dss_official_bbox_pipeline", "semantic_clusters", "step2_crf_applied_false"]
ADAPTATIONS = {
    "SGSCN": ["ph2_paper_faithful_declared_conventions", "sysu_us_paper_faithful_declared_conventions",
              "ph2_official_reference", "sysu_us_official_reference"],
    "DSS_US": ["step2_dss_baseline_dss_paper_faithful_declared_conventions"],
}


def adaptation(method_dir, profile):
    path = ROOT / "baseline" / method_dir / "config/native" / f"{profile}.yaml"
    config = load_lock(path)  # only runnable (unblocked) profiles can be adapted
    payload = _freeze_payload()
    bound = {item["path"]: item["sha256"] for item in payload["scientific_payload"]["bound_freeze_files"]}
    sgscn = method_dir == "SGSCN"
    document = {
        "schema": ADAPTATION_SCHEMA,
        "method": "SGSCN" if sgscn else "DSS-US",
        "adaptation_id": f"acdc_v12_{profile}",
        "status": "ACDC_PLUMBING_ONLY: the source profile's scientific behaviour is unchanged",
        "source_profile": {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path), "profile": config["profile"],
                           "native_dataset": config["dataset"], "implementation": config["implementation"],
                           "profile_class": config.get("profile_class", "OFFICIAL_REFERENCE"),
                           "paper_equivalence": config["paper_equivalence"]},
        "acdc_contract": {"freeze_id": payload["freeze_id"], "freeze_dir": str(FREEZE_DIR.relative_to(ROOT)),
                          "manifest_sha256": payload["scientific_payload"]["shared_manifest"]["manifest_sha256"],
                          "manifest_file_sha256": bound["data/acdc_shared_manifest.json"],
                          "shared_grid_sha256": FROZEN_SELF_AUDIT_HISTORICAL_224_SHARED_GRID_SHA256,
                          "adapter_spec_sha256": FROZEN_ADAPTER_SPEC_SHA256,
                          "tracks_contract_sha256": file_hash(TRACKS_CONTRACT),
                          "input_conversion": INPUT_CONVERSION_SPEC},
        "plumbing": {
            "input": "uint8 3-channel image from the frozen central plane, passed unchanged to the source profile's own input path",
            "spatial": ("method runs on the shared 224x224 grid and its output is the raw map (no resize)" if sgscn else
                        "224x224 = 28x28 ViT-S/8 patches (no crop needed); Step II output resized to the image with the profile's nearest rule"),
            "output": "anonymous int32 [224,224] raw IDs exactly as produced; no semantic remapping",
            "persistence": "seal_raw_partition (cardiac_raw_partition.v1) per sample + native_acdc run seal + external write-once receipt",
            "seed": ({"base_seed": 1, "derivation": "sgscn sample_seed: sha256(base_seed, sample_id)[:4] over the ACDC sample_id"} if sgscn else
                     {"kmeans_seed": "from the source profile (unchanged)", "per_sample_seed": "none (deterministic DINO features)"}),
        },
        "unchanged": SGSCN_UNCHANGED if sgscn else DSS_UNCHANGED,
    }
    if not sgscn:
        document["plumbing"]["step2_fit_cohort"] = ("exactly the run's selected records (one split, sorted by sample_id); "
                                                    "every raw map depends on this cohort, which run.json records")
        document["not_applicable_on_acdc"] = {
            "camus_input_conversion": "replaced by the declared ACDC input conversion above",
            "camus_cohort_inventory": "replaced by the frozen ACDC manifest hash",
            "dino_checkpoint_sha256": "still supplied at runtime (--dino-checkpoint-sha256)"}
    elif "cohort_inventory" in config.get("required_data", {}):
        document["not_applicable_on_acdc"] = {"cohort_inventory": "native SYSU-US cohort binding replaced by the frozen ACDC manifest hash"}
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-rewrite", action="store_true")
    args = parser.parse_args()
    for method_dir, profiles in ADAPTATIONS.items():
        for profile in profiles:
            target = ROOT / "baseline" / method_dir / "config/acdc" / f"acdc_{profile}.json"
            text = json.dumps(adaptation(method_dir, profile), indent=2, sort_keys=True) + "\n"
            if target.exists() and target.read_text(encoding="utf-8") != text and not args.force_rewrite:
                raise SystemExit(f"refusing to change existing adaptation {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            print(target.relative_to(ROOT), file_hash(target))


if __name__ == "__main__":
    main()
