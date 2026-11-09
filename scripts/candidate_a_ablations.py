#!/usr/bin/env python3
"""Materialize and run a hash-pinned Candidate A ablation plan.

Planning and checking never start training. Execution requires --execute.
Only the canonical train.py is invoked, with one shared teacher preparation
per seed and the same TRAIN/development manifests for every arm.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from self_audit.candidate_a.config import validate_config
from self_audit.candidate_a.io import source_inventory
from self_audit.data import TrainingData, write_json
from self_audit.model import digest


def arm_config(base, arm):
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
    return validate_config(config)


def create_plan(*, dataset, train_manifest, development_manifest, model_config, out,
                seeds=(42, 43, 44), arms=None, epochs=100, pretrain_epochs=100,
                warmup_epochs=10, synthetic=False):
    folder = Path(out).resolve()
    if folder.exists():
        raise ValueError("plan output must be a new directory")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("training seeds must be unique nonnegative integers")
    profile = json.loads((ROOT / "configs/datasets" / (dataset + ".json")).read_text())
    train_path = Path(train_manifest).resolve()
    data = TrainingData(train_path, digest(train_path), synthetic=synthetic)
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
    base = json.loads((ROOT / "configs/candidate_a" / ("smoke.json" if synthetic else "full.json")).read_text())
    if model_config is not None:
        base["model"] = json.loads(Path(model_config).read_text())
    elif not synthetic:
        raise ValueError("real-data plans require an explicit local backbone config")
    base["dataset"] = {"name": dataset, "protocol": profile["protocol"]}
    base["epochs"] = epochs
    if dev_path is None:
        base["validation_every"] = 0
    definitions = json.loads((ROOT / "configs/ablations/candidate_a.json").read_text())["arms"]
    if arms is not None:
        if not arms or len(set(arms)) != len(arms) or set(arms) - {a["name"] for a in definitions}:
            raise ValueError("unknown, empty or duplicate ablation selection")
        definitions = [a for a in definitions if a["name"] in arms]
    jobs = []
    for seed in seeds:
        recipe = deepcopy(base)
        recipe["seed"] = seed
        recipe["rollout"]["seed"] = seed
        pretrain_id, warmup_id = f"s{seed}_pretrain", f"s{seed}_warmup"
        resolved_arms = [(definition, arm_config(recipe, definition)) for definition in definitions]
        teacher_stages = (
            ("teacher_pretrain", pretrain_epochs, pretrain_id, None),
            ("teacher_warmup", warmup_epochs, warmup_id, pretrain_id),
        ) if any(config["use_teacher"] for _, config in resolved_arms) else ()
        for stage, count, identifier, dependency in teacher_stages:
            config = deepcopy(recipe)
            config.update(stage=stage, epochs=count)
            jobs.append({"id": identifier, "factor": "shared_teacher", "teacher_from": dependency,
                         "config": validate_config(config)})
        for definition, config in resolved_arms:
            jobs.append({"id": f"s{seed}_{definition['name']}", "factor": definition["factor"],
                         "teacher_from": warmup_id if config["use_teacher"] else None, "config": config})
    folder.mkdir(parents=True)
    (folder / "configs").mkdir()
    for job in jobs:
        path = folder / "configs" / (job["id"] + ".json")
        write_json(path, job.pop("config"))
        job.update(config=str(path), config_sha256=digest(path), out=str(folder / "runs" / job["id"]))
    plan = {"schema": "self_audit.candidate_a.plan.v1", "dataset": dataset, "synthetic": synthetic,
            "train": str(train_path), "train_sha256": digest(train_path),
            "development": str(dev_path) if dev_path else None,
            "development_sha256": digest(dev_path) if dev_path else None,
            "label_budget": len(data.lids), "training_seeds": list(seeds), "source_files": source_inventory(),
            "comparison_contract": "same data, label budget, epoch schedule and seed within each seed block; changed capacity/bit precision declared by factor",
            "selection": "final epoch only; no test access or validation-based checkpoint selection", "jobs": jobs}
    write_json(folder / "PLAN.json", plan)
    return folder / "PLAN.json"


def run_plan(path, *, execute=False, device="cuda"):
    plan = json.loads(Path(path).read_text())
    if plan.get("schema") != "self_audit.candidate_a.plan.v1" or plan["source_files"] != source_inventory():
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
        validate_config(json.loads(Path(job["config"]).read_text()))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--dataset", choices=("acdc", "mnms"), required=True)
    plan.add_argument("--train", required=True)
    plan.add_argument("--development")
    plan.add_argument("--model-config")
    plan.add_argument("--out", required=True)
    plan.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    plan.add_argument("--arms", nargs="+")
    plan.add_argument("--epochs", type=int, default=100)
    plan.add_argument("--pretrain-epochs", type=int, default=100)
    plan.add_argument("--warmup-epochs", type=int, default=10)
    plan.add_argument("--synthetic", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "plan":
        path = create_plan(dataset=args.dataset, train_manifest=args.train,
            development_manifest=args.development, model_config=args.model_config, out=args.out,
            seeds=args.seeds, arms=args.arms, epochs=args.epochs, pretrain_epochs=args.pretrain_epochs,
            warmup_epochs=args.warmup_epochs, synthetic=args.synthetic)
        print(json.dumps({"plan": str(path), "sha256": digest(path), "status": "NOT_RUN"}))
    else:
        print(json.dumps(run_plan(args.plan, execute=args.execute, device=args.device), indent=2))


if __name__ == "__main__":
    main()
