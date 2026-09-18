"""Generate subject-level manifests for audited cardiac datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_audit.data.cmr_multi import CMRMultiAdapter
from self_audit.data.cmrxmotion import CMRxMotionAdapter
from self_audit.data.splits import subject_level_split


def generate(
    dataset: str,
    data_root: str | Path,
    output: str | Path,
    *,
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    allow_affine_mismatch: bool = False,
) -> Path:
    name = str(dataset).strip().lower()
    if name == "cmr_multi":
        adapter = CMRMultiAdapter(data_root, require_confident=True)
        mapping = {
            pair.case_id: pair.subject_id
            for pair in adapter.pairs
            if adapter.infer_case(pair).accepted
        }
    elif name == "cmr_motion":
        adapter = CMRxMotionAdapter(
            data_root,
            allow_affine_mismatch=allow_affine_mismatch,
        )
        records = adapter.audit()
        valid = {
            record.case_id
            for record in records
            if record.has_mask
            and not record.error
            and (record.affine_equal is True or allow_affine_mismatch)
        }
        mapping = {
            case.case_id: case.subject_id
            for case in adapter.cases
            if case.case_id in valid
        }
    else:
        raise ValueError("dataset must be cmr_multi or cmr_motion")
    splits = subject_level_split(
        mapping,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
    )
    subject_sets = {
        split: sorted({mapping[case_id] for case_id in case_ids})
        for split, case_ids in splits.items()
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": name,
                "split_level": "subject",
                "seed": int(seed),
                "train_fraction": float(train_fraction),
                "val_fraction": float(val_fraction),
                "test_fraction": float(1.0 - train_fraction - val_fraction),
                "train_cases": splits["train"],
                "val_cases": splits["val"],
                "test_cases": splits["test"],
                "train_subjects": subject_sets["train"],
                "val_subjects": subject_sets["val"],
                "test_subjects": subject_sets["test"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cmr_multi", "cmr_motion"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--allow-affine-mismatch", action="store_true")
    args = parser.parse_args()
    print(generate(
        args.dataset,
        args.data_root,
        args.output,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        allow_affine_mismatch=args.allow_affine_mismatch,
    ))


if __name__ == "__main__":
    main()
