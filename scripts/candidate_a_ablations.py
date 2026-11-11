#!/usr/bin/env python3
"""Materialize and run a hash-pinned Candidate A ablation plan.

Planning and checking never start training. Execution requires --execute.
Only the canonical train.py is invoked, with one shared teacher preparation
per seed and the same TRAIN/development manifests for every arm.
"""
import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from self_audit.candidate_a.config import validate_config as validate_candidate_a
from self_audit.candidate_a.io import source_inventory as candidate_a_source_inventory
from self_audit.data import TrainingData, write_json
from self_audit.model import digest


def _track_modules(track):
    """Resolve the versioned config/data/IO contracts without changing A."""
    if track == "candidate_a":
        return {
            "validate": validate_candidate_a,
            "source_inventory": candidate_a_source_inventory,
            "data": TrainingData,
            "config_dir": ROOT / "configs/candidate_a",
            "ablations": ROOT / "configs/ablations/candidate_a.json",
            "plan_schema": "self_audit.candidate_a.plan.v1",
        }
    if track == "candidate_a3d":
        from self_audit.candidate_a3d.config import validate_config
        from self_audit.candidate_a3d.data import CandidateA3DData
        from self_audit.candidate_a3d.io import source_inventory
        return {
            "validate": validate_config,
            "source_inventory": source_inventory,
            "data": CandidateA3DData,
            "config_dir": ROOT / "configs/candidate_a3d",
            "ablations": ROOT / "configs/ablations/candidate_a3d.json",
            "plan_schema": "self_audit.candidate_a3d.plan.v1",
        }
    raise ValueError("track must be candidate_a or candidate_a3d")


def arm_config(base, arm, *, validate=validate_candidate_a):
    config = deepcopy(base)
    for dotted, value in arm["overrides"].items():
        keys = dotted.split(".")
        target = config
        for key in keys[:-1]:
            target = target[key]
        if keys[-1] not in target:
            raise ValueError(f"unknown ablation field {dotted}")
        target[keys[-1]] = value
    for dotted, multiplier in arm.get("multipliers", {}).items():
        keys = dotted.split(".")
        target = config
        for key in keys[:-1]:
            target = target[key]
        original = target[keys[-1]]
        scaled = original * multiplier
        if type(original) is not int or not float(scaled).is_integer() or scaled <= 0:
            raise ValueError(f"capacity factor must produce a positive integer: {dotted}")
        target[keys[-1]] = int(scaled)
    return validate(config)


def _load_semi_config(path, *, track):
    """Load and normalize the opt-in A-3D semi-supervised block.

    The CLI takes a JSON file so the exact bytes can be pinned in a plan.  For
    convenience the file may contain either the block itself or an object with
    a single ``semi_supervised`` key.  Candidate A (2-D) deliberately rejects
    this option rather than silently accepting an A-3D recipe.
    """
    if path is None:
        return None, None, None
    if track != "candidate_a3d":
        raise ValueError("--semi-config is supported only for track=candidate_a3d")
    from self_audit.candidate_a3d.config import _validate_semi_supervised

    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"semi-supervised config does not exist: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid semi-supervised JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("semi-supervised JSON must contain one object")
    block = payload.get("semi_supervised", payload)
    normalized = _validate_semi_supervised(block)
    return path, digest(path), normalized


def create_plan(*, dataset, train_manifest, development_manifest, model_config, out,
                seeds=(42,), arms=None, epochs=100, pretrain_epochs=100,
                warmup_epochs=10, synthetic=False, track="candidate_a",
                checkpoint_every_epochs=None, decoder_chunk_size=None,
                semi_config=None):
    modules = _track_modules(track)
    semi_path, semi_sha256, semi_block = _load_semi_config(semi_config, track=track)
    folder = Path(out).resolve()
    if folder.exists():
        raise ValueError("plan output must be a new directory")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("training seeds must be unique nonnegative integers")
    profile = json.loads((ROOT / "configs/datasets" / (dataset + ".json")).read_text())
    train_path = Path(train_manifest).resolve()
    data = modules["data"](train_path, digest(train_path), synthetic=synthetic)
    if not synthetic and data.manifest.get("dataset") != dataset:
        raise ValueError("prepared TRAIN dataset identity does not match requested dataset")
    dev_path = Path(development_manifest).resolve() if development_manifest else None
    if dev_path is not None:
        dev = json.loads(dev_path.read_text())
        if (dev.get("role") != "development" or dev.get("schema") != "self_audit.data.v2"
                or bool(dev.get("synthetic", False)) != synthetic):
            raise ValueError("epoch validation requires a matching development manifest")
        ids = [r["id"] for r in dev["patients"]]
        if not ids or len(set(ids)) != len(ids) or set(ids) & set(data.lids + data.uids):
            raise ValueError("development cohort is empty, overlapping or contains duplicate IDs")
        if not synthetic and dev.get("dataset") != dataset:
            raise ValueError("prepared development dataset identity mismatch")
    if folder.is_relative_to(train_path.parent) or train_path.parent.is_relative_to(folder):
        raise ValueError("ablation outputs must be separate from source data")
    base = json.loads((modules["config_dir"] / ("smoke.json" if synthetic else "full.json")).read_text())
    if model_config is not None:
        base["model"] = json.loads(Path(model_config).read_text())
    base["dataset"] = {"name": dataset, "protocol": "native_3d" if track == "candidate_a3d" else profile["protocol"]}
    if semi_block is not None:
        # Put the same normalized block in the base recipe before arm overrides
        # are applied.  Therefore teacher stages and every arm, including the
        # teacher-free control, carry exactly the same U-consistency protocol.
        base["semi_supervised"] = deepcopy(semi_block)
    base["epochs"] = epochs
    if checkpoint_every_epochs is not None:
        if (isinstance(checkpoint_every_epochs, bool)
                or not isinstance(checkpoint_every_epochs, int)
                or checkpoint_every_epochs < 0):
            raise ValueError("checkpoint_every_epochs must be a non-negative integer")
        base["checkpoint_every_epochs"] = checkpoint_every_epochs
    if decoder_chunk_size is not None:
        if (isinstance(decoder_chunk_size, bool)
                or not isinstance(decoder_chunk_size, int)
                or decoder_chunk_size < 1):
            raise ValueError("decoder_chunk_size must be a positive integer")
        base["student"]["chunk_size"] = decoder_chunk_size
    if dev_path is None:
        base["validation_every"] = 0
    catalogue = json.loads(modules["ablations"].read_text())
    definitions = catalogue["arms"]
    if arms is None:
        arms = catalogue.get("default_arms")
    if arms is not None:
        if not arms or len(set(arms)) != len(arms) or set(arms) - {a["name"] for a in definitions}:
            raise ValueError("unknown, empty or duplicate ablation selection")
        definitions = [a for a in definitions if a["name"] in arms]
    if not synthetic and model_config is None:
        # A teacher-free control does not instantiate the teacher at all, so it
        # is valid to plan it without claiming a local backbone/checkpoint.
        # Any arm that uses a teacher still requires an explicit verified local
        # model configuration; the placeholder in full.json is never used as a
        # silent random/pretrained fallback.
        probe = deepcopy(base)
        probe["seed"] = seeds[0]
        probe["rollout"]["seed"] = seeds[0]
        resolved_probe = [arm_config(probe, definition, validate=modules["validate"])
                          for definition in definitions]
        if any(config["use_teacher"] for config in resolved_probe):
            raise ValueError("real-data plans with teacher arms require an explicit local backbone config")
    jobs = []
    for seed in seeds:
        recipe = deepcopy(base)
        recipe["seed"] = seed
        recipe["rollout"]["seed"] = seed
        pretrain_id, warmup_id = f"s{seed}_pretrain", f"s{seed}_warmup"
        resolved_arms = [(definition, arm_config(recipe, definition, validate=modules["validate"])) for definition in definitions]
        teacher_stages = (
            ("teacher_pretrain", pretrain_epochs, pretrain_id, None),
            ("teacher_warmup", warmup_epochs, warmup_id, pretrain_id),
        ) if any(config["use_teacher"] for _, config in resolved_arms) else ()
        for stage, count, identifier, dependency in teacher_stages:
            config = deepcopy(recipe)
            config.update(stage=stage, epochs=count)
            jobs.append({"id": identifier, "factor": "shared_teacher", "teacher_from": dependency,
                         "config": modules["validate"](config)})
        for definition, config in resolved_arms:
            jobs.append({"id": f"s{seed}_{definition['name']}", "factor": definition["factor"],
                         "teacher_from": warmup_id if config["use_teacher"] else None, "config": config})
    folder.mkdir(parents=True)
    (folder / "configs").mkdir()
    for job in jobs:
        path = folder / "configs" / (job["id"] + ".json")
        write_json(path, job.pop("config"))
        job.update(config=str(path), config_sha256=digest(path), out=str(folder / "runs" / job["id"]))
    plan = {"schema": modules["plan_schema"], "track": track, "dataset": dataset, "synthetic": synthetic,
            "train": str(train_path), "train_sha256": digest(train_path),
            "development": str(dev_path) if dev_path else None,
            "development_sha256": digest(dev_path) if dev_path else None,
            "label_budget": len(data.lids), "training_seeds": list(seeds), "source_files": modules["source_inventory"](),
            "ablation_arms": [definition["name"] for definition in definitions],
            "comparison_contract": "same data, label budget, epoch schedule and seed within each seed block; changed capacity/bit precision declared by factor",
            "selection": "final epoch only; no test access or validation-based checkpoint selection", "jobs": jobs}
    if semi_path is not None:
        plan["semi_config"] = {"path": str(semi_path), "sha256": semi_sha256}
        plan["semi_supervised"] = deepcopy(semi_block)
    if checkpoint_every_epochs is not None or decoder_chunk_size is not None:
        plan["runtime_overrides"] = {
            "checkpoint_every_epochs": checkpoint_every_epochs,
            "decoder_chunk_size": decoder_chunk_size,
        }
    write_json(folder / "PLAN.json", plan)
    return folder / "PLAN.json"


def run_plan(path, *, execute=False, device="cuda"):
    plan = json.loads(Path(path).read_text())
    track = plan.get("track", "candidate_a")
    modules = _track_modules(track)
    if plan.get("schema") != modules["plan_schema"] or plan["source_files"] != modules["source_inventory"]():
        raise ValueError("plan source/schema mismatch; regenerate for this pinned checkout")
    for key in ("train", "development"):
        if plan[key] is not None and digest(plan[key]) != plan[key + "_sha256"]:
            raise ValueError(f"{key} manifest changed after planning")
    known = set()
    for job in plan["jobs"]:
        if job["id"] in known or (job["teacher_from"] is not None and job["teacher_from"] not in known):
            raise ValueError("invalid teacher stage dependency/order")
        known.add(job["id"])
        if digest(job["config"]) != job["config_sha256"]:
            raise ValueError("ablation recipe changed after planning")
        recipe = json.loads(Path(job["config"]).read_text())
        modules["validate"](recipe)
        if plan.get("semi_supervised") is not None and recipe.get("semi_supervised") != plan["semi_supervised"]:
            raise ValueError("semi-supervised block differs between A-3D jobs")
    semi_descriptor = plan.get("semi_config")
    if semi_descriptor is not None:
        if track != "candidate_a3d" or not isinstance(semi_descriptor, dict):
            raise ValueError("invalid A-3D semi-supervised config provenance")
        semi_path, semi_sha256, semi_block = _load_semi_config(semi_descriptor.get("path"), track=track)
        if semi_sha256 != semi_descriptor.get("sha256") or semi_block != plan.get("semi_supervised"):
            raise ValueError("semi-supervised config changed after planning")
    if not execute:
        return {"status": "PLAN_VALIDATED_NOT_RUN", "dataset": plan["dataset"], "jobs": len(plan["jobs"])}
    # Never overwrite partial or previous experiments, or silently resume them.
    if any(Path(job["out"]).exists() for job in plan["jobs"]):
        raise ValueError("run directories already exist; create a new plan or inspect the previous run")
    completed = {}
    for job in plan["jobs"]:
        command = [sys.executable, str(ROOT / "train.py"), "--config", job["config"],
                   "--data", plan["train"], "--data-sha256", plan["train_sha256"],
                   "--out", job["out"], "--device", device, "--allow-full-run"]
        if plan["synthetic"]:
            command.append("--synthetic")
        if plan["development"] is not None:
            command += ["--validation-data", plan["development"], "--validation-data-sha256", plan["development_sha256"]]
        if job["teacher_from"] is not None:
            checkpoint = Path(completed[job["teacher_from"]]) / "final"
            command += ["--teacher-checkpoint", str(checkpoint), "--teacher-checkpoint-sha256", digest(checkpoint / "MANIFEST.json")]
        subprocess.run(command, cwd=ROOT, check=True)
        completed[job["id"]] = job["out"]
    result = {"status": "COMPLETED", "synthetic": plan["synthetic"], "runs": completed}
    write_json(Path(path).parent / "RUN_RECEIPT.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--dataset", choices=("acdc", "mnms"), required=True)
    plan.add_argument("--train", required=True)
    plan.add_argument("--development")
    plan.add_argument("--model-config")
    plan.add_argument("--out", required=True)
    plan.add_argument("--seeds", type=int, nargs="+", default=[42], help="training seeds (default: 42)")
    plan.add_argument("--arms", nargs="+", help="explicit catalogue arms; A-3D defaults to its four main comparisons")
    plan.add_argument("--epochs", type=int, default=100)
    plan.add_argument("--pretrain-epochs", type=int, default=100)
    plan.add_argument("--warmup-epochs", type=int, default=10)
    plan.add_argument("--checkpoint-every-epochs", type=int,
                      help="override periodic checkpoint cadence; 0 keeps final only")
    plan.add_argument("--decoder-chunk-size", type=int,
                      help="override student decoder chunk size")
    plan.add_argument("--semi-config", help="pinned JSON file containing the A-3D semi_supervised block")
    plan.add_argument("--synthetic", action="store_true")
    plan.add_argument("--track", choices=("candidate_a", "candidate_a3d"), default="candidate_a")
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "plan":
        path = create_plan(dataset=args.dataset, train_manifest=args.train,
            development_manifest=args.development, model_config=args.model_config, out=args.out,
            seeds=args.seeds, arms=args.arms, epochs=args.epochs, pretrain_epochs=args.pretrain_epochs,
            warmup_epochs=args.warmup_epochs, synthetic=args.synthetic, track=args.track,
            checkpoint_every_epochs=args.checkpoint_every_epochs,
            decoder_chunk_size=args.decoder_chunk_size, semi_config=args.semi_config)
        print(json.dumps({"plan": str(path), "sha256": digest(path), "status": "NOT_RUN"}))
    else:
        print(json.dumps(run_plan(args.plan, execute=args.execute, device=args.device), indent=2))


if __name__ == "__main__":
    main()
