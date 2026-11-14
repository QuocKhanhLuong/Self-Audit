#!/usr/bin/env python3
"""Run the synthetic CPU A-aug FUSE pipeline and write a SOFTWARE_ONLY receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.smoke_fuse import run as run_base


def run(out: Path, threads: int = 1) -> dict:
    """Reuse the complete v1 CPU smoke with the opt-in v2 A-aug config."""
    from self_audit.candidate_fuse.config import default_config, normalize_config
    from self_audit.candidate_fuse import engine
    from scripts.fuse_build_semi_manifest import build as build_semi_manifest
    from tests.test_fuse_semi_data import _semi_fixture as build_source_fixture
    import scripts.smoke_fuse as base_smoke

    original_default_config = default_config
    original_run_training = engine.run_training

    def wrapped_run_base(out, threads):
        """Generate base eval fixtures, then the full eight-patient L8 train fixture."""
        original_run_training_local = engine.run_training

        def prepare_then_train(config, output, **kwargs):
            source_path, _, locked_path, locked_sha, _ = build_source_fixture(
                Path(out).expanduser().resolve(), patients=9, labeled_count=8)
            semi = build_semi_manifest(source_path, locked_path, Path(out).resolve() / "v2-semi",
                                       expected_locked_sha256=locked_sha, synthetic=True)
            config["schema"] = "self_audit.candidate_fuse.v2"
            config["experiment"] = {"track": "A", "dataset": "ACDC"}
            config["augmentation"] = {"policy": "inplane_strong_v1"}
            config["data"]["train_manifest"] = semi["manifest"]
            config["data"]["train_manifest_sha256"] = semi["sha256"]
            train_manifest = json.loads(Path(semi["manifest"]).read_text())
            for prefix in ("val", "test"):
                path = Path(config["data"][f"{prefix}_manifest"])
                evaluation = json.loads(path.read_text())
                evaluation["split"] = train_manifest["split"]
                path.write_text(json.dumps(evaluation, sort_keys=True) + "\n")
                config["data"][f"{prefix}_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            prepared = normalize_config(config)
            return original_run_training_local(prepared, output, **kwargs)

        engine.run_training = prepare_then_train
        try:
            return run_base(out, threads)
        finally:
            engine.run_training = original_run_training_local

    def a_aug_default(*, synthetic: bool = False):
        config = original_default_config(synthetic=synthetic)
        config["schema"] = "self_audit.candidate_fuse.v2"
        config["experiment"] = {"track": "A", "dataset": "ACDC"}
        config["augmentation"] = {"policy": "inplane_strong_v1"}
        return config

    import self_audit.candidate_fuse.config as config_module
    config_module.default_config = a_aug_default
    try:
        receipt = wrapped_run_base(out, threads)
    finally:
        config_module.default_config = original_default_config
        engine.run_training = original_run_training

    export_path = Path(receipt["deployment"]["path"])
    payload = __import__("torch").load(export_path, map_location="cpu", weights_only=True)
    config = payload.get("config", {})
    if config.get("schema") != "self_audit.candidate_fuse.v2" or config.get("experiment") != {"track": "A", "dataset": "ACDC"}:
        raise RuntimeError("A-aug synthetic smoke did not run the v2 Track A experiment")
    if config.get("augmentation") != {"policy": "inplane_strong_v1"}:
        raise RuntimeError("A-aug synthetic smoke did not persist the locked augmentation policy")
    receipt["augmentation"] = config["augmentation"]

    resume_path = Path(receipt["output"]) / "run" / "resume.pt"
    import torch
    state = torch.load(resume_path, map_location="cpu", weights_only=False)
    strong = state.get("rng", {}).get("strong_augmentation")
    if not isinstance(strong, dict) or set(strong) != {"generator_state", "probabilities"}:
        raise RuntimeError("A-aug checkpoint is missing its independent augmentation RNG state")
    receipt["schema"] = "self_audit.candidate_fuse.aug_smoke.v1"
    receipt["augmentation_rng_checkpointed"] = True
    output = Path(receipt["output"])
    (output / "SOFTWARE_ONLY_A_AUG.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Explicit output directory (use /tmp)")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    print(json.dumps(run(args.out, args.threads), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
