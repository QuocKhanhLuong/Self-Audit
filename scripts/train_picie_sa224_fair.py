#!/usr/bin/env python
"""Train a fair PiCIE-SA224 checkpoint on ACDC train images only."""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
PICIE_ROOT = ROOT / "baseline" / "PICIE"
for extra in (ROOT, ROOT / "src", PICIE_ROOT, PICIE_ROOT / "modules"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from self_audit.data.acdc import ACDCDataset
from shared_benchmark.checkpoint_contract import SCHEMA_VERSION
from shared_benchmark.provenance import sha256_file
from shared_benchmark.spatial import grid_hash, load_self_audit_compat_224_grid_spec
from baseline.PICIE.modules.fpn import PanopticFPN
from baseline.PICIE.src.cardiac_benchmark.dataset import SA224_NORMALIZATION_VERSION


@dataclass(frozen=True)
class FrozenPreset:
    augment: bool = True
    blur: bool = True
    grey: bool = True
    jitter: bool = True
    equiv: bool = True
    h_flip: bool = True
    v_flip: bool = False
    random_crop: bool = True
    min_scale: float = 0.8


@dataclass(frozen=True)
class EquivarianceSpec:
    scale: float
    top_ratio: float
    left_ratio: float
    h_flip: bool
    v_flip: bool


PRESET = FrozenPreset()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _self_audit_affine_unit_interval(image: torch.Tensor) -> torch.Tensor:
    finite = torch.nan_to_num(image.float(), nan=0.0, posinf=0.0, neginf=0.0)
    clipped = finite.clamp(-3.0, 3.0)
    return (clipped + 3.0) / 6.0


def _imagenet_normalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(3, 1, 1)
    return (image - mean) / std


def _feature_flatten(feats: torch.Tensor) -> torch.Tensor:
    if feats.ndim == 2:
        return feats
    return feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1]).contiguous()


class ACDCPicieTrainDataset(Dataset[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        data_root: Path,
        split_manifest: Path,
        *,
        resolution: int,
        seed: int,
        preset: FrozenPreset,
    ) -> None:
        self.base = ACDCDataset(
            data_root=data_root,
            split="train",
            split_manifest=split_manifest,
            image_size=resolution,
        )
        self.seed = int(seed)
        self.preset = preset
        self.epoch = 0
        self.view1_labels: list[torch.Tensor] | None = None
        self.view2_labels: list[torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_pseudolabels(self, view1: Sequence[torch.Tensor], view2: Sequence[torch.Tensor]) -> None:
        if len(view1) != len(self.base) or len(view2) != len(self.base):
            raise ValueError("pseudo-label inventory does not match the train split")
        self.view1_labels = [value.clone().long() for value in view1]
        self.view2_labels = [value.clone().long() for value in view2]

    def render_invariant_view(self, index: int, view: int) -> torch.Tensor:
        sample = self.base[index]
        image = _self_audit_affine_unit_interval(sample["image"])
        rng = random.Random(self.seed + self.epoch * 1_000_003 + int(index) * 104729 + int(view) * 15485863)
        image = self._apply_photometric(image, rng)
        return _imagenet_normalize(image.contiguous())

    def transform_eqv(self, indices: Sequence[int], value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(f"expected BCHW tensor for equivariant transform, got shape {tuple(value.shape)}")
        if not self.preset.equiv:
            return value
        transformed = [self._apply_equivariance(value[offset], self._equivariance_spec(int(index))) for offset, index in enumerate(indices)]
        return torch.stack(transformed, dim=0)

    def _equivariance_spec(self, index: int) -> EquivarianceSpec:
        rng = random.Random(self.seed + self.epoch * 1_000_003 + int(index) * 104729 + 999_983)
        scale = 1.0
        top_ratio = 0.0
        left_ratio = 0.0
        if self.preset.random_crop:
            scale = self.preset.min_scale + (1.0 - self.preset.min_scale) * rng.random()
            top_ratio = rng.random()
            left_ratio = rng.random()
        return EquivarianceSpec(
            scale=scale,
            top_ratio=top_ratio,
            left_ratio=left_ratio,
            h_flip=self.preset.h_flip and rng.random() < 0.5,
            v_flip=self.preset.v_flip and rng.random() < 0.5,
        )

    def _apply_equivariance(self, tensor: torch.Tensor, spec: EquivarianceSpec) -> torch.Tensor:
        _, height, width = tensor.shape
        out = tensor
        if self.preset.random_crop:
            crop_h = max(1, min(height, int(round(height * spec.scale))))
            crop_w = max(1, min(width, int(round(width * spec.scale))))
            max_top = max(height - crop_h, 0)
            max_left = max(width - crop_w, 0)
            top = 0 if max_top == 0 else min(max_top, int(round(spec.top_ratio * max_top)))
            left = 0 if max_left == 0 else min(max_left, int(round(spec.left_ratio * max_left)))
            out = TF.resized_crop(
                out,
                top=top,
                left=left,
                height=crop_h,
                width=crop_w,
                size=[height, width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        if spec.h_flip:
            out = TF.hflip(out)
        if spec.v_flip:
            out = TF.vflip(out)
        return out

    def _apply_photometric(self, image: torch.Tensor, rng: random.Random) -> torch.Tensor:
        out = image
        if self.preset.jitter:
            out = TF.adjust_brightness(out, 1.0 + rng.uniform(-0.2, 0.2))
            out = TF.adjust_contrast(out, 1.0 + rng.uniform(-0.2, 0.2))
            out = TF.adjust_saturation(out, 1.0 + rng.uniform(-0.1, 0.1))
            out = TF.adjust_hue(out, rng.uniform(-0.03, 0.03))
        if self.preset.grey and rng.random() < 0.2:
            out = TF.rgb_to_grayscale(out, num_output_channels=3)
        if self.preset.blur and rng.random() < 0.5:
            sigma = 0.1 + 1.9 * rng.random()
            out = TF.gaussian_blur(out, kernel_size=[5, 5], sigma=[sigma, sigma])
        return out.clamp(0.0, 1.0)

    def __getitem__(self, index: int):
        image1 = self.render_invariant_view(index, 1)
        image2 = self.render_invariant_view(index, 2)
        if self.view1_labels is None or self.view2_labels is None:
            raise RuntimeError("pseudo-labels must be set before supervised training batches are drawn")
        return int(index), image1, image2, self.view1_labels[index], self.view2_labels[index]


class FeatureViewDataset(Dataset[tuple[int, torch.Tensor]]):
    def __init__(self, base: ACDCPicieTrainDataset, view: int) -> None:
        self.base = base
        self.view = int(view)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        return int(index), self.base.render_invariant_view(index, self.view)


def _collate_features(batch: list[tuple[int, torch.Tensor]]) -> tuple[list[int], torch.Tensor]:
    return [item[0] for item in batch], torch.stack([item[1] for item in batch], dim=0)


def _collate_train(batch: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    return (
        [item[0] for item in batch],
        torch.stack([item[1] for item in batch], dim=0),
        torch.stack([item[2] for item in batch], dim=0),
        torch.stack([item[3] for item in batch], dim=0),
        torch.stack([item[4] for item in batch], dim=0),
    )


def _mini_batch_kmeans(
    model: nn.Module,
    dataset: ACDCPicieTrainDataset,
    *,
    view: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    metric_test: str,
    n_clusters: int,
    kmeans_iters: int,
    num_init_batches: int,
    num_batches: int,
    seed: int,
) -> tuple[torch.Tensor, float]:
    feature_dataset = FeatureViewDataset(dataset, view)
    loader = DataLoader(
        feature_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_collate_features,
    )
    index = faiss.IndexFlatL2(128)
    data_count = np.zeros(n_clusters, dtype=np.float64)
    buffered: list[torch.Tensor] = []
    buffered_batches = 0
    first_pass = True
    kmeans_losses: list[float] = []
    centroids: np.ndarray | None = None

    model.eval()
    with torch.no_grad():
        for step, (indices, images) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            if view == 1 and dataset.preset.equiv:
                images = dataset.transform_eqv(indices, images)
            feats = model(images)
            if view == 2 and dataset.preset.equiv:
                feats = dataset.transform_eqv(indices, feats)
            if metric_test == "cosine":
                feats = F.normalize(feats, dim=1, p=2)
            buffered.append(_feature_flatten(feats).detach().cpu())
            buffered_batches += 1

            target_batches = num_init_batches if first_pass else num_batches
            last_step = step + 1 == len(loader)
            if buffered_batches < target_batches and not last_step:
                continue

            batch_feats = torch.cat(buffered, dim=0).numpy().astype("float32", copy=False)
            if batch_feats.shape[0] < n_clusters:
                raise ValueError(
                    f"Need at least {n_clusters} feature vectors for k-means, got {batch_feats.shape[0]}"
                )

            if first_pass:
                clus = faiss.Clustering(batch_feats.shape[1], n_clusters)
                clus.seed = seed
                clus.niter = kmeans_iters
                clus.max_points_per_centroid = 10_000_000
                clus.train(batch_feats, index)
                centroids = faiss.vector_float_to_array(clus.centroids).reshape(n_clusters, batch_feats.shape[1]).astype("float32")
                index.reset()
                index.add(centroids)
                distances, assignments = index.search(batch_feats, 1)
                kmeans_losses.append(float(distances.mean()))
                for cluster_id in np.unique(assignments):
                    data_count[int(cluster_id)] += len(np.where(assignments == cluster_id)[0])
                first_pass = False
            else:
                assert centroids is not None
                index.reset()
                index.add(centroids)
                distances, assignments = index.search(batch_feats, 1)
                kmeans_losses.append(float(distances.mean()))
                for cluster_id in np.unique(assignments):
                    member_idx = np.where(assignments == cluster_id)[0]
                    cluster_id_int = int(cluster_id)
                    data_count[cluster_id_int] += len(member_idx)
                    centroid_lr = len(member_idx) / (data_count[cluster_id_int] + 1e-6)
                    centroids[cluster_id_int] = (
                        (1.0 - centroid_lr) * centroids[cluster_id_int]
                        + centroid_lr * batch_feats[member_idx].mean(0)
                    )

            buffered = []
            buffered_batches = 0

    if centroids is None:
        raise RuntimeError("k-means did not produce centroids")
    return torch.from_numpy(centroids.copy()).float(), float(sum(kmeans_losses) / max(len(kmeans_losses), 1))


def _scores_from_centroids(feats: torch.Tensor, centroids: torch.Tensor, *, metric_train: str) -> torch.Tensor:
    device_centroids = centroids.to(feats.device)
    flat = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])
    if metric_train == "cosine":
        flat = F.normalize(flat, dim=1, p=2)
    centroid_norm = (device_centroids * device_centroids).sum(dim=1)
    distances = 1.0 - 2.0 * (flat @ device_centroids.t()) + centroid_norm.unsqueeze(0)
    scores = (-distances).reshape(feats.shape[0], feats.shape[2], feats.shape[3], centroids.shape[0]).permute(0, 3, 1, 2)
    return scores.contiguous()


def _compute_pseudolabels(
    model: nn.Module,
    dataset: ACDCPicieTrainDataset,
    centroids: torch.Tensor,
    *,
    view: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    metric_train: str,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    feature_dataset = FeatureViewDataset(dataset, view)
    loader = DataLoader(
        feature_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_collate_features,
    )
    labels: list[torch.Tensor] = [torch.empty(0, dtype=torch.long) for _ in range(len(feature_dataset))]
    counts = torch.zeros(centroids.shape[0], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for indices, images in loader:
            images = images.to(device, non_blocking=True)
            if view == 1 and dataset.preset.equiv:
                images = dataset.transform_eqv(indices, images)
            feats = model(images)
            if view == 2 and dataset.preset.equiv:
                feats = dataset.transform_eqv(indices, feats)
            if metric_train == "cosine":
                feats = F.normalize(feats, dim=1, p=2)
            batch_labels = _scores_from_centroids(feats, centroids, metric_train=metric_train).argmax(dim=1).cpu()
            for offset, index in enumerate(indices):
                sample_labels = batch_labels[offset].long().contiguous()
                labels[index] = sample_labels
                counts += torch.bincount(sample_labels.flatten(), minlength=centroids.shape[0]).float()
    weight = counts / counts.sum().clamp_min(1.0)
    return labels, weight


def _build_nonparametric_classifier(centroids: torch.Tensor, *, in_dim: int, device: torch.device) -> nn.Conv2d:
    classifier = nn.Conv2d(in_dim, centroids.shape[0], kernel_size=1, stride=1, padding=0, bias=True).to(device)
    with torch.no_grad():
        classifier.weight.copy_(centroids[:, :, None, None].to(device))
        classifier.bias.zero_()
    for param in classifier.parameters():
        param.requires_grad = False
    classifier.eval()
    return classifier


def _mean_loss(history: list[float]) -> float:
    return float(sum(history) / max(len(history), 1))


def build_recipe_audit() -> dict[str, Any]:
    items = [
        {
            "item": "two_view_generation",
            "status": "matched",
            "current_branch": "two stochastic views are rendered for every train sample each epoch",
            "upstream_anchor": "official train_picie.py trains on two coupled views",
        },
        {
            "item": "geometric_equivariance_transforms",
            "status": "intentional_adaptation",
            "current_branch": "the fair trainer mirrors PiCIE's split image-space/feature-space equivariance path with a local MRI dataset wrapper",
            "upstream_anchor": "vendored utils expose transform_eqv plus eqv augmentation flags",
        },
        {
            "item": "photometric_invariance_transforms",
            "status": "intentional_adaptation",
            "current_branch": "blur, greyscale, and jitter remain enabled under a frozen MRI preset",
            "upstream_anchor": "vendored utils gate blur/grey/jitter under args.augment",
        },
        {
            "item": "feature_extraction_cadence",
            "status": "matched",
            "current_branch": "both train views are featurized each epoch before the supervised train loop",
            "upstream_anchor": "official train_picie.py recomputes features before each epoch's cluster refresh",
        },
        {
            "item": "clustering_refresh_cadence",
            "status": "matched",
            "current_branch": "mini-batch k-means is refreshed once per epoch for both views",
            "upstream_anchor": "official train_picie.py calls run_mini_batch_kmeans for view 1 and view 2 each epoch",
        },
        {
            "item": "pseudo_label_assignment_path",
            "status": "matched",
            "current_branch": "pseudo-labels are assigned per view from centroid-distance scores after the official-style equivariance path",
            "upstream_anchor": "official train_picie.py calls compute_labels for view 1 and view 2 before training",
        },
        {
            "item": "classifier_update_path",
            "status": "matched",
            "current_branch": "frozen nonparametric classifiers are rebuilt from centroids each epoch and used for within/across-view CE losses",
            "upstream_anchor": "official train_picie.py initializes classifier1/classifier2 from centroids and freezes them",
        },
        {
            "item": "epoch_iteration_semantics",
            "status": "intentional_adaptation",
            "current_branch": "each epoch follows cluster-refresh then train-loop semantics, but the ACDC adapter uses a local dataset wrapper instead of the upstream vendored train dataset classes",
            "upstream_anchor": "README points to train_picie.py plus dataset-specific loaders that are not fully vendored in this branch snapshot",
        },
    ]
    return {
        "reference": {
            "readme": "baseline/PICIE/README.md",
            "transforms": "baseline/PICIE/utils.py",
            "official_train_entrypoint": "train_picie.py from the official PiCIE repository",
        },
        "items": items,
        "known_fidelity_gaps": [
            item["item"] for item in items if item["status"] == "fidelity_gap"
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "preprocessed_data" / "ACDC")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "splits" / "acdc_patient_split_seed42.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints" / "picie" / "fair_sa224")
    parser.add_argument("--run-name", default="acdc_train_only")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8, help="Legacy alias used when the dedicated cluster/train batch sizes are not set.")
    parser.add_argument("--batch-size-cluster", type=int, default=None)
    parser.add_argument("--batch-size-train", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--metric-train", choices=("cosine", "l2"), default="cosine")
    parser.add_argument("--metric-test", choices=("cosine", "l2"), default="cosine")
    parser.add_argument("--kmeans-iters", type=int, default=20)
    parser.add_argument("--num-init-batches", type=int, default=30)
    parser.add_argument("--num-batches", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--mse", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    output_dir = args.output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    batch_size_cluster = int(args.batch_size_cluster or args.batch_size)
    batch_size_train = int(args.batch_size_train or args.batch_size)
    recipe_audit = build_recipe_audit()

    train_dataset = ACDCPicieTrainDataset(
        args.data_root,
        args.split_manifest,
        resolution=args.resolution,
        seed=args.seed,
        preset=PRESET,
    )

    model = PanopticFPN(type("Args", (), {"arch": "resnet18", "pretrain": False, "in_dim": 128})()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_logs: list[dict[str, Any]] = []
    last_classifier1: nn.Module | None = None
    last_classifier2: nn.Module | None = None
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        centroids1, kmloss1 = _mini_batch_kmeans(
            model,
            train_dataset,
            view=1,
            batch_size=batch_size_cluster,
            num_workers=args.num_workers,
            device=device,
            metric_test=args.metric_test,
            n_clusters=4,
            kmeans_iters=args.kmeans_iters,
            num_init_batches=args.num_init_batches,
            num_batches=args.num_batches,
            seed=args.seed + epoch * 2,
        )
        centroids2, kmloss2 = _mini_batch_kmeans(
            model,
            train_dataset,
            view=2,
            batch_size=batch_size_cluster,
            num_workers=args.num_workers,
            device=device,
            metric_test=args.metric_test,
            n_clusters=4,
            kmeans_iters=args.kmeans_iters,
            num_init_batches=args.num_init_batches,
            num_batches=args.num_batches,
            seed=args.seed + epoch * 2 + 1,
        )
        labels1, weight1 = _compute_pseudolabels(
            model,
            train_dataset,
            centroids1,
            view=1,
            batch_size=batch_size_cluster,
            num_workers=args.num_workers,
            device=device,
            metric_train=args.metric_train,
        )
        labels2, weight2 = _compute_pseudolabels(
            model,
            train_dataset,
            centroids2,
            view=2,
            batch_size=batch_size_cluster,
            num_workers=args.num_workers,
            device=device,
            metric_train=args.metric_train,
        )
        train_dataset.set_pseudolabels(labels1, labels2)

        criterion1 = nn.CrossEntropyLoss(weight=None if args.no_balance else weight1.to(device))
        criterion2 = nn.CrossEntropyLoss(weight=None if args.no_balance else weight2.to(device))
        classifier1 = _build_nonparametric_classifier(centroids1, in_dim=128, device=device)
        classifier2 = _build_nonparametric_classifier(centroids2, in_dim=128, device=device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=_collate_train,
        )
        model.train()
        batch_losses: list[float] = []
        batch_losses_cet: list[float] = []
        batch_losses_within: list[float] = []
        batch_losses_across: list[float] = []
        batch_losses_mse: list[float] = []
        for indices, image1, image2, target1, target2 in train_loader:
            image1 = train_dataset.transform_eqv(indices, image1.to(device, non_blocking=True))
            image2 = image2.to(device, non_blocking=True)
            target1 = target1.to(device, non_blocking=True)
            target2 = target2.to(device, non_blocking=True)
            featmap1 = model(image1)
            featmap2 = model(image2)
            featmap2 = train_dataset.transform_eqv(indices, featmap2)
            if args.metric_train == "cosine":
                featmap1 = F.normalize(featmap1, dim=1, p=2)
                featmap2 = F.normalize(featmap2, dim=1, p=2)

            loss11 = criterion1(classifier1(featmap1), target1)
            loss22 = criterion2(classifier2(featmap2), target2)
            loss_within = (loss11 + loss22) / 2.0

            loss12 = criterion2(classifier2(featmap1), target2)
            loss21 = criterion1(classifier1(featmap2), target1)
            loss_across = (loss12 + loss21) / 2.0

            loss_cet = (loss_within + loss_across) / 2.0
            total_loss = loss_cet
            loss_mse_value = torch.tensor(0.0, device=device)
            if args.mse:
                loss_mse_value = F.mse_loss(featmap1, featmap2)
                total_loss = (loss_cet + loss_mse_value) / 2.0

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            batch_losses.append(float(total_loss.detach().cpu()))
            batch_losses_cet.append(float(loss_cet.detach().cpu()))
            batch_losses_within.append(float(loss_within.detach().cpu()))
            batch_losses_across.append(float(loss_across.detach().cpu()))
            batch_losses_mse.append(float(loss_mse_value.detach().cpu()))

        hist1 = torch.bincount(torch.cat([label.flatten() for label in labels1], dim=0), minlength=4)
        hist2 = torch.bincount(torch.cat([label.flatten() for label in labels2], dim=0), minlength=4)
        epoch_logs.append({
            "epoch": epoch,
            "kmeans_loss_view1": float(kmloss1),
            "kmeans_loss_view2": float(kmloss2),
            "loss_total": _mean_loss(batch_losses),
            "loss_ce_total": _mean_loss(batch_losses_cet),
            "loss_ce_within": _mean_loss(batch_losses_within),
            "loss_ce_across": _mean_loss(batch_losses_across),
            "loss_mse": _mean_loss(batch_losses_mse),
            "cluster_histogram_view1": hist1.tolist(),
            "cluster_histogram_view2": hist2.tolist(),
            "class_weight_view1": weight1.tolist(),
            "class_weight_view2": weight2.tolist(),
        })
        last_classifier1 = classifier1.cpu()
        last_classifier2 = classifier2.cpu()

    if last_classifier1 is None or last_classifier2 is None:
        raise RuntimeError("training did not produce classifiers")

    model = model.cpu()
    checkpoint_path = output_dir / f"{args.run_name}.pth"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "state_dict": model.state_dict(),
        "classifier_state_dict": last_classifier1.state_dict(),
        "classifier1_state_dict": last_classifier1.state_dict(),
        "classifier2_state_dict": last_classifier2.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            "arch": "resnet18",
            "pretrain": False,
            "in_dim": 128,
            "K_train": 4,
            "K_test": 4,
            "resolution": args.resolution,
            "metric_train": args.metric_train,
            "metric_test": args.metric_test,
            "seed": args.seed,
            "batch_size_cluster": batch_size_cluster,
            "batch_size_train": batch_size_train,
        },
        "augmentation_preset": asdict(PRESET),
        "recipe_audit": recipe_audit,
        "training_log": epoch_logs,
    }, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    grid = load_self_audit_compat_224_grid_spec(ROOT)

    provenance = {
        "schema_version": "self_audit.picie_fair_training.v1",
        "baseline_name": "PICIE",
        "baseline_mode": "PICIE-SA224-FAIR",
        "dataset": "acdc",
        "run_name": args.run_name,
        "source_data_root": str(args.data_root.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_selection_policy": "final_epoch_without_dev_selection",
        "cluster_count": 4,
        "metric_train": args.metric_train,
        "metric_test": args.metric_test,
        "batch_size_cluster": batch_size_cluster,
        "batch_size_train": batch_size_train,
        "num_init_batches": args.num_init_batches,
        "num_batches": args.num_batches,
        "no_balance": bool(args.no_balance),
        "mse": bool(args.mse),
        "augmentation_preset": asdict(PRESET),
        "recipe_audit": recipe_audit,
        "training_tags": [
            "in_domain_acdc",
            "self_audit_normalized",
            "frozen_picie_preset",
            "train_split_only",
        ],
        "epoch_log": epoch_logs,
    }
    contract = {
        "schema_version": SCHEMA_VERSION,
        "baseline_name": "PICIE",
        "baseline_mode": "PICIE-SA224-FAIR",
        "benchmark_tier": "fair",
        "checkpoint_sha256": checkpoint_sha,
        "dataset": "acdc",
        "split_policy_version": "self_audit.acdc.patient_split.v1",
        "shared_grid_hash": grid_hash(grid),
        "normalization_version": SA224_NORMALIZATION_VERSION,
        "training_data_schema": "self_audit.acdc.image_only.v1",
        "training_tags": provenance["training_tags"],
    }
    _write_json(output_dir / f"{args.run_name}.training_provenance.json", provenance)
    _write_json(output_dir / f"{args.run_name}.checkpoint_contract.json", contract)
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "checkpoint_contract": str(output_dir / f"{args.run_name}.checkpoint_contract.json"),
        "training_provenance": str(output_dir / f"{args.run_name}.training_provenance.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
