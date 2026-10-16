"""Plot published online loss summaries; never load a patient or checkpoint."""

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = "92a60744efef85558d7c0226c84b7c49d47217ec"
CSV = (
    "reports/stable_guidance_main513ddda_b12_20261016/"
    "completed_run/analysis/semantic_convergence/epochs.csv"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = (args.repo / CSV).read_bytes()
    pinned = subprocess.check_output(
        ["git", "-C", str(args.repo), "show", BASE + ":" + CSV]
    )
    if raw != pinned:
        raise ValueError("published epoch table differs from pinned evidence")
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    x = [int(row["epoch"]) for row in rows]
    if x != list(range(1, 51)):
        raise ValueError("expected all fifty published epochs")
    values = {k: [float(row[k]) for row in rows] for k in rows[0] if k != "epoch"}
    for ce, entropy, kl in zip(
        values["semantic_ce"],
        values["target_entropy_floor"],
        values["online_kl_estimate"],
    ):
        if abs(ce - entropy - kl) > 1e-12:
            raise ValueError("CE minus entropy differs from the published KL estimate")
    args.out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": BASE}
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
    axes[0].plot(x, values["semantic_ce"], color="#0072B2", label="Semantic CE")
    axes[0].plot(
        x, values["target_entropy_floor"], color="#E69F00", label="Target entropy"
    )
    axes[0].fill_between(
        x,
        values["target_entropy_floor"],
        values["semantic_ce"],
        color="#0072B2",
        alpha=0.15,
    )
    axes[0].set(
        title="Nonzero CE has a target-entropy floor",
        ylabel="All-step epoch mean",
        ylim=(0, 1.4),
    )
    axes[0].legend(loc="lower right", frameon=False)
    axes[1].plot(x, values["online_kl_estimate"], color="#0072B2")
    axes[1].set(title="Estimated online KL keeps decreasing", ylim=(0, 0.09))
    axes[2].plot(x, values["feature_loss"], color="#009E73")
    axes[2].set(title="Feature objective also improves", ylim=(0, 0.4))
    for ax in axes:
        ax.set(xlabel="Epoch", xlim=(1, 50))
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("GuidanceStudent / cardinal_v1 — published 50-epoch run", fontsize=14)
    fig.supxlabel(
        "One run, seed 42, batch 12. Online changing-model losses; not fixed-checkpoint KL or anatomical accuracy.\n"
        "Source: main 513ddda; published evidence 92a6074. No patient arrays or new training used.",
        fontsize=9,
    )
    fig.savefig(args.out / "convergence.png", dpi=180)
    fig.savefig(args.out / "convergence.svg", metadata={"Date": None})
    plt.close(fig)
    svg = args.out / "convergence.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    receipt = {
        "status": "PLOT_OF_PUBLISHED_LOGS_ONLY",
        "evidence_commit": BASE,
        "source_path": CSV,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "epochs": len(rows),
        "patient_arrays_read": False,
        "training_launched": False,
        "outputs": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.out.iterdir())
        },
    }
    (args.out / "PLOT_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
