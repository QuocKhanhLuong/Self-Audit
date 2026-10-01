"""Render CUTS run figures (GT-free) into the project.

Reads the latent/diffusion NPZ files written by ``main.py --mode test`` and
``generate_diffusion.py`` and, optionally, the persistent partitions written by
``export_diffusion_persistent.py``.  It uses only ``image``, ``recon``,
``latent`` and ``labels_diffusion``; it never reads ``label``.

Outputs (under ``--output-dir``):
  training_curve.png        train/val losses parsed from the training log
  samples_<k>_<name>.png    image | recon | latent PCA | persistent partition
  overview.png              grid of all selected partitions
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def kmeans_partition(latent: np.ndarray, cache: Path) -> np.ndarray:
    """CUTS's label-free PHATE + K=10 clustering."""
    if cache.exists():
        return np.load(cache)
    import phate  # same calls/parameters as cardiac_benchmark.cluster_kmeans.phate_kmeans

    operator = phate.PHATE(n_components=3, knn=100, n_landmark=500, t=2,
                           verbose=False, random_state=1, n_jobs=1)
    operator.fit_transform(latent)
    partition = np.asarray(phate.cluster.kmeans(operator, n_clusters=10, random_state=1)).astype(np.int64)
    np.save(cache, partition)
    return partition


def _hw(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    return array[..., 0] if array.ndim == 3 else array


def latent_pca_rgb(latent: np.ndarray, height: int, width: int) -> np.ndarray:
    """PCA of the [H*W, C] latent map to 3 components, shown as RGB."""
    flat = latent.reshape(height * width, -1).astype(np.float64)
    flat -= flat.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    proj = flat @ vt[:3].T
    lo, hi = np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0)
    proj = np.clip((proj - lo) / np.maximum(hi - lo, 1e-8), 0, 1)
    return proj.reshape(height, width, 3)


def plot_training_curve(log_path: Path, out_path: Path) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    pattern = r"^(Train|Validation) \[(\d+)/\d+\] recon loss: ([\d.]+), contrastive loss: ([\d.]+), total loss: ([\d.]+)"
    rows = {"Train": [], "Validation": []}
    for match in re.finditer(pattern, text, flags=re.M):
        rows[match.group(1)].append((int(match.group(2)), *map(float, match.groups()[2:])))
    if not rows["Train"]:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for name, style in (("Train", "-"), ("Validation", "--")):
        data = np.asarray(rows[name])
        axes[0].plot(data[:, 0], np.maximum(data[:, 1], 5e-4), style, label=name)
        axes[1].plot(data[:, 0], data[:, 2], style, label=name)
    axes[0].set(title="Reconstruction loss (log has 3 decimals; floor 5e-4)", xlabel="epoch", yscale="log")
    axes[1].set(title="Contrastive loss", xlabel="epoch")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path,
                        help="run results dir containing numpy_files/ (and optionally numpy_files_seg_diffusion/)")
    parser.add_argument("--persistent-root", type=Path,
                        help="dir of *.npy from export_diffusion_persistent.py")
    parser.add_argument("--kmeans", action="store_true",
                        help="cluster each latent with PHATE + K-means (k=10) when no diffusion partition exists")
    parser.add_argument("--log", type=Path, help="training log for the loss curve")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.log is not None and args.log.exists():
        plot_training_curve(args.log, args.output_dir / "training_curve.png")

    paths = sorted((args.results_dir / "numpy_files").glob("*.npz"))
    if not paths:
        raise SystemExit("no latent NPZ files found; run main.py --mode test first")
    picks = [paths[i] for i in np.unique(np.linspace(0, len(paths) - 1, args.num_samples).astype(int))]
    diffusion_dir = args.results_dir / "numpy_files_seg_diffusion"

    partitions, images = [], []
    for k, path in enumerate(picks):
        with np.load(path, allow_pickle=False) as data:
            image, recon, latent = _hw(data["image"]), _hw(data["recon"]), data["latent"]
        height, width = image.shape
        partition, title = None, ""
        persistent_path = args.persistent_root / f"{path.stem}.npy" if args.persistent_root else None
        if persistent_path is not None and persistent_path.exists():
            partition, title = np.load(persistent_path), "persistent partition"
        elif (diffusion_dir / path.name).exists():
            with np.load(diffusion_dir / path.name, allow_pickle=False) as data:
                levels = data["labels_diffusion"]
            partition, title = levels[len(levels) // 2].reshape(height, width), "mid diffusion level"
        if partition is None and args.kmeans:
            cache_dir = args.output_dir / "kmeans_partitions"
            cache_dir.mkdir(exist_ok=True)
            partition = kmeans_partition(latent, cache_dir / f"{path.stem}.npy").reshape(height, width)
            title = "PHATE+KMeans k=10"
        ncols = 4 if partition is not None else 3
        fig, axes = plt.subplots(1, ncols, figsize=(3.3 * ncols, 3.4))
        axes[0].imshow(image, cmap="gray"); axes[0].set_title("image")
        axes[1].imshow(recon, cmap="gray"); axes[1].set_title("reconstruction")
        axes[2].imshow(latent_pca_rgb(latent, height, width)); axes[2].set_title("latent PCA (3 comp.)")
        if partition is not None:
            colors = plt.get_cmap("tab10")(np.arange(int(partition.max()) + 1) % 10)[:, :3]
            axes[3].imshow(image, cmap="gray")
            axes[3].imshow(colors[partition], alpha=0.55)
            axes[3].set_title(f"{title} ({len(np.unique(partition))} regions)")
            partitions.append(colors[partition]); images.append(image)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(path.stem, fontsize=9)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"samples_{k:02d}_{path.stem}.png", dpi=130)
        plt.close(fig)

    if not images:
        print(f"wrote figures to {args.output_dir} (no diffusion partitions available)")
        return 0
    cols = min(6, len(picks))
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, image, colored in zip(axes.ravel(), images, partitions):
        ax.imshow(image, cmap="gray"); ax.imshow(colored, alpha=0.55)
    fig.tight_layout()
    fig.savefig(args.output_dir / "overview.png", dpi=130)
    plt.close(fig)
    print(f"wrote figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
