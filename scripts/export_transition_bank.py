#!/usr/bin/env python3
"""Export a frozen transition bank from a bound checkpoint and a real cohort.

The export is a two-stage pipeline that mirrors the module contract:

1. Proposals are generated from ``batch['image']`` only.  The batch mask is
   never handed to the generator; it is held back here and passed to the
   evaluator afterwards.
2. The evaluator attaches the actual ``Q`` under an explicit metric contract.

Running this script needs a real trained checkpoint and a real split.  It does
not synthesize either, and an absent checkpoint is an error rather than an
evaluation of whatever weights happen to be live.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.evaluation.transition_bank import (
    ROLLOUT_ALWAYS_ACCEPT,
    ROLLOUT_SELF_AUDIT,
    SOURCE_ALWAYS_ACCEPT_PREFIX,
    SOURCE_ON_POLICY,
    SOURCE_SYNTHETIC,
    TransitionEvaluator,
    bank_summary,
    build_bank,
    dump_bank,
    evaluation_provenance_record,
    generate_on_policy_proposals,
    generate_synthetic_proposals,
    generation_provenance_record,
)
from self_audit.training.unified_config import (
    ResolvedExecutionConfig,
    UnifiedConfig,
    resolve_downstream_config,
)
from self_audit.training._utils import (
    bind_evaluation_checkpoint,
    build_data_loader,
    build_model_from_config,
    build_patient_dataset,
    load_config,
    move_batch,
    resolve_device,
    seed_everything,
    validate_dataset_splits,
    verify_bound_state,
)


def _identity(batch: dict, key: str) -> list:
    """Read one identity column straight off the batch.

    A real export never invents an identifier.  A batch that cannot say which
    patient, case and slice a row came from fails here rather than producing a
    bank whose rows are labelled with positional placeholders.
    """

    value = batch.get(key)
    if value is None:
        raise KeyError(
            f"Batch does not carry {key!r}; a transition bank row must name the real "
            "patient/case/slice it came from. Use a dataset that supplies it."
        )
    if torch.is_tensor(value):
        column = [value[index] for index in range(value.shape[0])]
    else:
        column = list(value)
    if len(column) != int(batch["image"].shape[0]):
        raise ValueError(
            f"Batch column {key!r} has {len(column)} entries for {int(batch['image'].shape[0])} samples"
        )
    return column


def export_transition_bank(
    config_source: str | Path | Mapping[str, Any] | UnifiedConfig,
    checkpoint: str | Path,
    output: str | Path,
    *,
    data_root: str | None = None,
    split_manifest: str | None = None,
    split: str | None = None,
    image_size: int | None = None,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    tau_accept: float | None = None,
    t_max: int | None = None,
    rollout_policy: str = ROLLOUT_SELF_AUDIT,
    include_synthetic: bool = False,
    synthetic_kind: str = "negative",
    synthetic_operation: str | None = None,
    max_batches: int | None = None,
    metric_contract: str | None = None,
) -> dict[str, Any]:
    """Export frozen Proposal-1 transition bank using resolved configuration."""
    overrides: dict[str, Any] = {}
    if data_root is not None:
        overrides["data_root"] = data_root
    if split_manifest is not None:
        overrides["split_manifest"] = split_manifest
    if image_size is not None:
        overrides["image_size"] = image_size
    if device is not None:
        overrides["device"] = str(device)
    if batch_size is not None:
        overrides["batch_size"] = batch_size

    resolved = resolve_downstream_config(config_source, overrides=overrides)
    split_name = str(split or resolved.val_split)
    target_device = resolve_device(str(device) if device is not None else str(resolved.device))
    seed_everything(resolved.seed, deterministic=resolved.deterministic)
    stats = resolved.validate_splits()
    print(f"split_stats={stats}")

    model = resolved.build_model(target_device)
    binding = bind_evaluation_checkpoint(
        model,
        [("checkpoint", checkpoint)],
        map_location=target_device,
        config=resolved.to_legacy_dict(),
    )
    print(
        f"bound checkpoint={binding.path} state_digest={binding.state_digest} "
        f"producer_git_sha={binding.producer.get('producer_git_sha')}"
    )

    dataset = resolved.build_dataset(split=split_name, train=False)
    loader = resolved.build_dataloader(dataset, train=False, batch_size=batch_size)
    effective_t_max = int(t_max) if t_max is not None else resolved.rollout_max_turns
    effective_tau = float(tau_accept) if tau_accept is not None else resolved.rollout_tau
    effective_contract = str(metric_contract or resolved.metric_contract)

    verify_bound_state(model, binding, boundary="transition_bank_export")
    evaluator = TransitionEvaluator(contract=effective_contract)

    rows: list[dict] = []
    sources = [SOURCE_ON_POLICY if rollout_policy == ROLLOUT_SELF_AUDIT else SOURCE_ALWAYS_ACCEPT_PREFIX]
    if include_synthetic:
        sources.append(SOURCE_SYNTHETIC)
    observed = 0
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = move_batch(raw_batch, target_device)
        images = batch["image"]
        # The mask is deliberately held back from generation and used only by
        # the evaluator below.
        masks = batch["mask"].detach().cpu().numpy()
        patients = _identity(batch, "patient_id")
        cases = _identity(batch, "case_id")
        slices = _identity(batch, "slice_idx")
        proposals = generate_on_policy_proposals(
            model,
            images,
            patient_ids=patients,
            case_ids=cases,
            slice_indices=slices,
            tau_accept=effective_tau,
            t_max=effective_t_max,
            rollout_policy=rollout_policy,
        )
        if include_synthetic:
            proposals = proposals + generate_synthetic_proposals(
                model,
                images,
                batch["mask"],
                patient_ids=patients,
                case_ids=cases,
                slice_indices=slices,
                kind=synthetic_kind,
                operation=synthetic_operation,
                tau_accept=effective_tau,
                draw_index=batch_index,
            )
        targets = {}
        for index in range(int(images.shape[0])):
            patient = patients[index]
            case = cases[index]
            slice_index = slices[index]
            key = (
                str(patient.item()) if torch.is_tensor(patient) else str(patient),
                str(case.item()) if torch.is_tensor(case) else str(case),
                int(slice_index.item()) if torch.is_tensor(slice_index) else int(slice_index),
            )
            targets[key] = masks[index]
        rows.extend(evaluator.evaluate(proposals, targets=targets))
        observed += int(images.shape[0])

    if not rows:
        raise SystemExit("No transitions were generated; refusing to write an empty bank.")

    # The state must still be the bound state after the whole export, not only
    # before it: a mutation midway would otherwise be recorded under the
    # identity of the checkpoint that was bound at the start.
    verify_bound_state(model, binding, boundary="transition_bank_export_complete")

    bank = build_bank(
        rows,
        generation=generation_provenance_record(
            model=model,
            binding=binding.as_dict(),
            loader=loader,
            split_name=split_name,
            rollout_policy=rollout_policy,
            gt_used_in_generation=bool(include_synthetic),
            max_batches=max_batches,
            batch_size=getattr(loader, "batch_size", None),
            observed_samples=observed,
        ),
        evaluation=evaluation_provenance_record(
            contract=effective_contract,
            reference_source=f"{split_name}_split_reference_masks",
        ),
        protocol={
            "tau_accept": effective_tau,
            "t_max": int(effective_t_max),
            "sources": sources,
            "rollout_policy": rollout_policy,
            "include_synthetic": bool(include_synthetic),
            "max_batches": max_batches,
            "note": (
                "always_accept_refinement rows are an analysis prefix rollout, not the "
                "deployable self-audit trajectory"
            ),
        },
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_bank(bank, output_path)
    print(f"bank_summary={bank_summary(bank)} saved={output_path}")
    return bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/self_audit_full.yaml",
        help="Path to unified or historical config YAML (default: configs/self_audit_full.yaml)",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--split", default=None, help="Split name (default: config val_split)")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--tau_accept", type=float, default=None)
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument(
        "--rollout_policy",
        default=ROLLOUT_SELF_AUDIT,
        choices=[ROLLOUT_SELF_AUDIT, ROLLOUT_ALWAYS_ACCEPT],
        help=(
            "self_audit is the deployable trajectory; always_accept_refinement is an "
            "analysis-only prefix rollout and is tagged as such in every row"
        ),
    )
    parser.add_argument(
        "--include_synthetic",
        action="store_true",
        help="Also emit counterfactual rows. These consume training GT and are tagged "
        "gt_used_in_generation=True; they are never a GT-free deployment claim.",
    )
    parser.add_argument("--synthetic_kind", default="negative", choices=["positive", "negative", "hard_neutral"])
    parser.add_argument("--synthetic_operation", default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--metric_contract", default="foreground_dice_exclude_v1")
    args = parser.parse_args()

    export_transition_bank(
        config_source=args.config,
        checkpoint=args.checkpoint,
        output=args.output,
        data_root=args.data_root,
        split_manifest=args.split_manifest,
        split=args.split,
        image_size=args.image_size,
        device=args.device,
        batch_size=args.batch_size,
        tau_accept=args.tau_accept,
        t_max=args.t_max,
        rollout_policy=args.rollout_policy,
        include_synthetic=args.include_synthetic,
        synthetic_kind=args.synthetic_kind,
        synthetic_operation=args.synthetic_operation,
        max_batches=args.max_batches,
        metric_contract=args.metric_contract,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
