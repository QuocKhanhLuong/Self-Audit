"""Teacher-anchored EMA vector quantizer used by Candidate Q-3D."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CODEBOOK_VERSION = "q3d_teacher_codebook.v1"
_CLASS_NAMES = ("BG", "RV", "MYO", "LV")
_TOKENIZER_OPTION_KEYS = {
    "codebook_size", "embedding_dim", "dimension", "hidden_channels", "adapter_hidden",
    "temperature", "commitment_beta", "reconstruction_weight", "anatomy_weight",
    "anatomy_loss_weight", "ema_decay", "initialization_volumes",
    "initialization_points_per_volume", "history_window", "usage_window_updates",
    "dead_code_threshold", "dead_code_fraction", "min_perplexity", "minimum_perplexity",
    "max_dead_code_fraction", "maximum_dead_code_fraction", "min_class_codes",
    "minimum_class_codes", "seed", "enabled", "codebook_frozen_after_stage", "source_feature",
}
_TOKENIZER_ALIASES = (
    ("embedding_dim", "dimension"),
    ("hidden_channels", "adapter_hidden"),
    ("anatomy_weight", "anatomy_loss_weight"),
    ("history_window", "usage_window_updates"),
    ("dead_code_threshold", "dead_code_fraction"),
    ("min_perplexity", "minimum_perplexity"),
    ("max_dead_code_fraction", "maximum_dead_code_fraction"),
    ("min_class_codes", "minimum_class_codes"),
)


def _tokenizer_options(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Accept a tokenizer mapping directly or nested in the Q-3D config."""

    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise TypeError("tokenizer config must be a nested mapping")
    current: Mapping[str, Any] = config
    for key in ("tokenizer", "teacher_tokenizer"):
        candidate = current.get(key)
        if isinstance(candidate, Mapping):
            current = candidate
            break
        # The published config nests model/training sections at the root; walk
        # those known containers without interpreting unrelated sections.
        for parent in ("q3d", "model", "training"):
            section = current.get(parent)
            if isinstance(section, Mapping) and isinstance(section.get(key), Mapping):
                current = section[key]
                break
        else:
            continue
        break
    options = dict(current)
    unknown = set(options) - _TOKENIZER_OPTION_KEYS
    if unknown:
        raise ValueError(f"unknown tokenizer config keys: {sorted(unknown)}")
    for alias, canonical in _TOKENIZER_ALIASES:
        if alias in options and canonical in options and options[alias] != options[canonical]:
            raise ValueError(f"conflicting tokenizer config keys: {alias} and {canonical}")
    return options


class TeacherTokenizer(nn.Module):
    """Learn an EMA codebook from the first 200 teacher-feature volumes.

    ``forward`` never silently trains on a random codebook.  It returns
    ``initialized=False`` while it gathers the configured number of
    initialization volumes; the engine must skip tokenizer losses for those
    calls.  Once initialized, call ``update_codebook`` exactly once per Q-stage
    optimizer update.  The frozen stage-J interface remains teacher-free at
    deployment because only the C0 buffer and student adapter are needed.
    """

    def __init__(self, teacher_channels: int, config: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if isinstance(teacher_channels, bool) or not isinstance(teacher_channels, int) \
                or teacher_channels <= 0:
            raise ValueError("teacher_channels must be a positive integer")
        options = _tokenizer_options(config)
        self.teacher_channels = int(teacher_channels)
        self.codebook_size = int(options.get("codebook_size", 256))
        self.embedding_dim = int(options.get("embedding_dim", options.get("dimension", 64)))
        self.hidden_channels = int(options.get("hidden_channels", options.get("adapter_hidden", 128)))
        self.temperature = float(options.get("temperature", 0.1))
        self.commitment_beta = float(options.get("commitment_beta", 0.25))
        self.reconstruction_weight = float(options.get("reconstruction_weight", 1.0))
        self.anatomy_weight = float(
            options.get("anatomy_weight", options.get("anatomy_loss_weight", 0.5))
        )
        self.ema_decay = float(options.get("ema_decay", 0.99))
        self.initialization_volumes = int(options.get("initialization_volumes", 200))
        self.initialization_points_per_volume = int(
            options.get("initialization_points_per_volume", 1024)
        )
        self.history_window = int(options.get("history_window", options.get("usage_window_updates", 200)))
        self.dead_code_threshold = float(
            options.get("dead_code_threshold", options.get("dead_code_fraction", 0.001))
        )
        self.min_perplexity = float(options.get("min_perplexity", options.get("minimum_perplexity", 32.0)))
        self.max_dead_code_fraction = float(
            options.get("max_dead_code_fraction", options.get("maximum_dead_code_fraction", 0.25))
        )
        self.min_class_codes = int(options.get("min_class_codes", options.get("minimum_class_codes", 4)))
        self.seed = int(options.get("seed", 42))
        self._validate_config()

        self.adapter = nn.Sequential(
            nn.Conv3d(self.teacher_channels, self.hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(self.hidden_channels, self.embedding_dim, kernel_size=1),
        )
        self.reconstructor = nn.Conv3d(self.embedding_dim, self.teacher_channels, kernel_size=1)
        self.anatomy_head = nn.Conv3d(self.embedding_dim, 4, kernel_size=1)

        # C0 is a buffer by contract: the tokenizer may update it by EMA, but
        # autograd and the optimizer can never change it.
        self.register_buffer("C0", torch.zeros(self.codebook_size, self.embedding_dim))
        self.register_buffer("ema_counts", torch.zeros(self.codebook_size))
        self.register_buffer("ema_sums", torch.zeros(self.codebook_size, self.embedding_dim))
        self.register_buffer("restart_cooldown", torch.zeros(self.codebook_size, dtype=torch.long))
        self.register_buffer("usage_history", torch.zeros(self.history_window, self.codebook_size))
        self.register_buffer("history_index", torch.zeros((), dtype=torch.long))
        self.register_buffer("history_count", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "class_mass_history", torch.zeros(self.history_window, 4, self.codebook_size),
        )
        self.register_buffer("class_history_index", torch.zeros((), dtype=torch.long))
        self.register_buffer("class_history_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("class_cohort_counter", torch.zeros((), dtype=torch.long))
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("initialization_volume_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("is_initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("is_frozen", torch.zeros((), dtype=torch.bool))
        self._initialization_points: list[Tensor] = []
        self._snapshot_codes: dict[str, Tensor] = {}
        self._class_mass_cohort_ids: list[str | None] = [None] * self.history_window
        self._last_class_code_coverage: dict[str, int] | None = None
        self._restart_generator = torch.Generator(device="cpu").manual_seed(self.seed + 1)

    def _validate_config(self) -> None:
        positive_ints = {
            "codebook_size": self.codebook_size,
            "embedding_dim": self.embedding_dim,
            "hidden_channels": self.hidden_channels,
            "initialization_volumes": self.initialization_volumes,
            "initialization_points_per_volume": self.initialization_points_per_volume,
            "history_window": self.history_window,
            "min_class_codes": self.min_class_codes,
        }
        if any(value <= 0 for value in positive_ints.values()):
            raise ValueError(f"tokenizer integer settings must be positive: {positive_ints}")
        if self.codebook_size < 2 or self.embedding_dim < 2:
            raise ValueError("codebook_size and embedding_dim must be at least two")
        if not (0.0 < self.ema_decay < 1.0):
            raise ValueError("ema_decay must be in (0,1)")
        if not (0.0 < self.temperature) or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and positive")
        for name, value in (
            ("commitment_beta", self.commitment_beta),
            ("reconstruction_weight", self.reconstruction_weight),
            ("anatomy_weight", self.anatomy_weight),
            ("min_perplexity", self.min_perplexity),
        ):
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
        if not (0.0 <= self.dead_code_threshold < 1.0):
            raise ValueError("dead_code_threshold must be in [0,1)")
        if not (0.0 <= self.max_dead_code_fraction <= 1.0):
            raise ValueError("max_dead_code_fraction must be in [0,1]")

    @property
    def codebook_initialized(self) -> bool:
        return bool(self.is_initialized.item())

    @property
    def codebook_frozen(self) -> bool:
        return bool(self.is_frozen.item())

    @property
    def codebook_sha256(self) -> str:
        payload = self.C0.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "version": CODEBOOK_VERSION,
            "sha256": self.codebook_sha256,
            "size": self.codebook_size,
            "dimension": self.embedding_dim,
            "initialized": self.codebook_initialized,
            "frozen": self.codebook_frozen,
            "train_heads_present": self.reconstructor is not None and self.anatomy_head is not None,
        }

    def _validate_feature(self, feature: Tensor) -> None:
        if not isinstance(feature, Tensor) or feature.ndim != 5:
            raise ValueError("feature must have shape [B,C_T,D,H,W]")
        if feature.shape[0] <= 0 or feature.shape[1] != self.teacher_channels \
                or any(size <= 0 for size in feature.shape[2:]):
            raise ValueError("feature shape does not match the tokenizer teacher_channels")
        if not torch.is_floating_point(feature) or not torch.isfinite(feature).all():
            raise ValueError("feature must be a finite floating tensor")

    def _encode(self, feature: Tensor) -> Tensor:
        self._validate_feature(feature)
        return F.normalize(self.adapter(feature), p=2, dim=1, eps=1e-8)

    @torch.no_grad()
    def collect_initialization(self, feature: Tensor) -> bool:
        """Collect one or more image-only teacher volumes toward k-means++ init."""

        if self.codebook_initialized:
            return True
        self._validate_feature(feature)
        vectors = self._encode(feature.detach()).permute(0, 2, 3, 4, 1).reshape(
            feature.shape[0], -1, self.embedding_dim,
        )
        for volume in vectors:
            if int(self.initialization_volume_count.item()) >= self.initialization_volumes:
                break
            point_count = volume.shape[0]
            keep = min(point_count, self.initialization_points_per_volume)
            indices = torch.linspace(
                0, point_count - 1, steps=keep, device=volume.device,
            ).round().long()
            self._initialization_points.append(volume.index_select(0, indices).detach().cpu())
            self.initialization_volume_count.add_(1)
        if int(self.initialization_volume_count.item()) >= self.initialization_volumes:
            self._initialize_kmeans_plus_plus()
        return self.codebook_initialized

    @torch.no_grad()
    def _initialize_kmeans_plus_plus(self) -> None:
        if self.codebook_initialized:
            return
        if len(self._initialization_points) < self.initialization_volumes:
            raise RuntimeError("k-means++ requires the configured number of distinct volumes")
        points = F.normalize(torch.cat(self._initialization_points, dim=0).float(), dim=1, eps=1e-8)
        if points.shape[0] < self.codebook_size:
            raise RuntimeError(
                f"k-means++ needs at least {self.codebook_size} points, got {points.shape[0]}"
            )

        # Deterministic cosine-distance k-means++ seeding. For unit vectors,
        # squared Euclidean distance is proportional to 1 - cosine similarity.
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        first = int(torch.randint(points.shape[0], (1,), generator=generator).item())
        chosen = [first]
        minimum_distance = (1.0 - points @ points[first]).clamp_min_(0.0)
        for _ in range(1, self.codebook_size):
            weights = minimum_distance.clone()
            weights[torch.tensor(chosen, dtype=torch.long)] = 0.0
            total = weights.sum()
            if float(total) <= 0.0:
                candidate = next(index for index in range(points.shape[0]) if index not in set(chosen))
            else:
                candidate = int(torch.multinomial(weights, 1, generator=generator).item())
            chosen.append(candidate)
            distance = (1.0 - points @ points[candidate]).clamp_min_(0.0)
            minimum_distance = torch.minimum(minimum_distance, distance)

        centers = F.normalize(points[torch.tensor(chosen, dtype=torch.long)], dim=1, eps=1e-8)
        assignments = (points @ centers.T).argmax(dim=1)
        counts = torch.bincount(assignments, minlength=self.codebook_size).float()
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, points)
        active = counts > 0
        centers[active] = F.normalize(sums[active] / counts[active, None], dim=1, eps=1e-8)

        self.C0.copy_(centers.to(device=self.C0.device, dtype=self.C0.dtype))
        # Start EMA sufficient statistics at one unit per initialized center,
        # rather than the aggregate 200-volume initialization sample count.
        # This lets ordinary minibatches move centroids at the configured
        # 0.99 decay from the first Q-stage update.
        self.ema_counts.fill_(1.0)
        self.ema_sums.copy_(self.C0.to(self.ema_sums.device))
        self.is_initialized.fill_(True)
        self._initialization_points.clear()

    def forward(self, feature: Tensor, class_probs: Tensor | None = None) -> dict[str, Any]:
        """Return soft/hard codes and tokenizer losses for aligned teacher features."""

        self._validate_feature(feature)
        u = self._encode(feature)
        if not self.codebook_initialized:
            initialized = self.collect_initialization(feature)
            if not initialized:
                return {
                    "initialized": False,
                    "initialization_volumes": int(self.initialization_volume_count.item()),
                    "u": u,
                    "pi": None,
                    "codes": None,
                    "quantized": None,
                    "commitment_loss": None,
                    "reconstruction_loss": None,
                    "anatomy_loss": None,
                    "loss": None,
                }
            # Recompute after k-means initialization with the now-fixed C0.
            u = self._encode(feature)

        if class_probs is not None:
            expected = (feature.shape[0], 4, *feature.shape[2:])
            if not isinstance(class_probs, Tensor) or tuple(class_probs.shape) != expected:
                raise ValueError(f"class_probs must have shape {expected}")
            if not torch.is_floating_point(class_probs) or not torch.isfinite(class_probs).all():
                raise ValueError("class_probs must be a finite floating tensor")
            if bool((class_probs < 0).any()):
                raise ValueError("class_probs cannot be negative")
            class_probs = class_probs.detach()
            normalizer = class_probs.sum(dim=1, keepdim=True)
            if bool((normalizer <= 0).any()):
                raise ValueError("class_probs must have positive mass at every cell")
            class_probs = class_probs / normalizer

        flat_u = u.permute(0, 2, 3, 4, 1).reshape(-1, self.embedding_dim)
        similarities = flat_u @ self.C0.to(dtype=flat_u.dtype).T
        pi_flat = torch.softmax(similarities / self.temperature, dim=-1)
        codes_flat = similarities.argmax(dim=-1)
        quantized_flat = F.embedding(codes_flat, self.C0.to(dtype=flat_u.dtype))
        target_shape = (feature.shape[0], *feature.shape[2:])
        codes = codes_flat.reshape(target_shape)
        quantized = quantized_flat.reshape(*target_shape, self.embedding_dim).permute(0, 4, 1, 2, 3)
        pi = pi_flat.reshape(*target_shape, self.codebook_size).permute(0, 4, 1, 2, 3)
        straight_through = u + (quantized - u).detach()

        commitment_loss = self.commitment_beta * (u - quantized.detach()).square().mean()
        target = feature.detach()
        if self.reconstructor is None:
            reconstruction_loss = feature.new_zeros(())
        else:
            reconstruction = self.reconstructor(straight_through)
            channel_std = target.std(dim=(0, 2, 3, 4), unbiased=False).clamp_min(1e-6)
            reconstruction_loss = (
                (reconstruction - target).abs()
                / channel_std[None, :, None, None, None]
            ).mean()
        if class_probs is None or self.anatomy_head is None:
            anatomy_loss = feature.new_zeros(())
        else:
            anatomy_logits = self.anatomy_head(straight_through)
            anatomy_loss = -(
                class_probs * F.log_softmax(anatomy_logits, dim=1)
            ).sum(dim=1).mean()
            anatomy_loss = anatomy_loss * self.anatomy_weight
        total_loss = (
            commitment_loss
            + self.reconstruction_weight * reconstruction_loss
            + anatomy_loss
        )
        return {
            "initialized": True,
            "initialization_volumes": int(self.initialization_volume_count.item()),
            "u": u,
            "pi": pi,
            "codes": codes,
            "quantized": quantized,
            "straight_through": straight_through,
            "commitment_loss": commitment_loss,
            "reconstruction_loss": reconstruction_loss,
            "anatomy_loss": anatomy_loss,
            "loss": total_loss,
            "codebook_sha256": self.codebook_sha256,
        }

    @torch.no_grad()
    def update_codebook(self, u: Tensor, codes: Tensor | None = None) -> dict[str, Any]:
        """Apply one EMA update and restart codes dead for the full history window."""

        if not self.codebook_initialized:
            raise RuntimeError("cannot update EMA codebook before k-means++ initialization")
        if self.codebook_frozen:
            raise RuntimeError("cannot update EMA codebook after tokenizer freeze")
        if not isinstance(u, Tensor) or u.ndim != 5 \
                or tuple(u.shape[1:2]) != (self.embedding_dim,) \
                or not torch.isfinite(u).all():
            raise ValueError("u must be a finite [B,embedding_dim,D,H,W] tensor")
        flat_u = F.normalize(u.detach(), dim=1, eps=1e-8).permute(0, 2, 3, 4, 1).reshape(
            -1, self.embedding_dim,
        )
        if codes is None:
            codes = (flat_u @ self.C0.to(dtype=flat_u.dtype).T).argmax(dim=1)
        else:
            if tuple(codes.shape) != (u.shape[0], *u.shape[2:]):
                raise ValueError("codes must match u's [B,D,H,W] grid")
            codes = codes.detach().reshape(-1).long()
        if codes.numel() != flat_u.shape[0] or bool((codes < 0).any()) \
                or bool((codes >= self.codebook_size).any()):
            raise ValueError("codes contain invalid codebook indices")

        counts = torch.bincount(codes, minlength=self.codebook_size).to(
            device=self.ema_counts.device, dtype=self.ema_counts.dtype,
        )
        sums = torch.zeros_like(self.ema_sums)
        sums.index_add_(0, codes.to(sums.device), flat_u.to(sums.device, sums.dtype))
        self.ema_counts.mul_(self.ema_decay).add_(counts, alpha=1.0 - self.ema_decay)
        self.ema_sums.mul_(self.ema_decay).add_(sums, alpha=1.0 - self.ema_decay)
        centers = self.ema_sums / self.ema_counts.clamp_min(1e-5)[:, None]
        self.C0.copy_(F.normalize(centers, dim=1, eps=1e-8).to(self.C0.dtype))

        frequencies = counts / counts.sum().clamp_min(1.0)
        index = int(self.history_index.item())
        self.usage_history[index].copy_(frequencies)
        self.history_index.fill_((index + 1) % self.history_window)
        self.history_count.fill_(min(self.history_window, int(self.history_count.item()) + 1))
        self.ema_updates.add_(1)
        self.restart_cooldown.sub_(1).clamp_min_(0)

        restarted: list[int] = []
        if int(self.history_count.item()) >= self.history_window:
            mean_usage = self.usage_history.mean(dim=0)
            dead_mask = (mean_usage < self.dead_code_threshold) & (self.restart_cooldown == 0)
            dead = torch.nonzero(dead_mask, as_tuple=False).flatten()
            if dead.numel():
                chosen = torch.randint(
                    flat_u.shape[0], (dead.numel(),), generator=self._restart_generator,
                    device="cpu",
                ).to(flat_u.device)
                replacements = flat_u.index_select(0, chosen).to(self.C0.device, self.C0.dtype)
                self.C0.index_copy_(0, dead.to(self.C0.device), F.normalize(replacements, dim=1, eps=1e-8))
                # A restarted code gets one unit of mass so the next EMA step
                # cannot immediately divide its zero count by a tiny epsilon.
                self.ema_counts[dead] = 1.0
                self.ema_sums.index_copy_(0, dead, self.C0.index_select(0, dead))
                self.restart_cooldown[dead] = self.history_window
                restarted = [int(value) for value in dead.cpu().tolist()]
        return {
            "ema_updates": int(self.ema_updates.item()),
            "restarted_codes": restarted,
            "codebook_sha256": self.codebook_sha256,
        }

    @torch.no_grad()
    def metrics(
        self,
        *,
        codes: Tensor | None = None,
        class_probs: Tensor | None = None,
        snapshot_id: str | None = None,
        cohort_id: str | None = None,
    ) -> dict[str, Any]:
        """Report rolling code usage, soft class ownership, churn, and artifact identity.

        Each class-mass observation represents one cohort. Repeated cohort IDs
        replace their previous row; new IDs advance the configured rolling
        window. ``snapshot_id`` doubles as the cohort ID when no separate ID is
        supplied, while anonymous observations receive persisted unique IDs.
        """

        if not self.codebook_initialized or int(self.history_count.item()) == 0:
            usage = torch.zeros(self.codebook_size, device=self.C0.device)
            perplexity: float | None = None
            dead_count: int | None = None
            dead_fraction: float | None = None
        else:
            history_size = int(self.history_count.item())
            if history_size < self.history_window:
                usage_rows = self.usage_history[:history_size]
            else:
                # After wrap, the order is irrelevant for the mean usage.
                usage_rows = self.usage_history
            usage = usage_rows.mean(dim=0)
            usage = usage / usage.sum().clamp_min(1e-12)
            entropy = -(usage * usage.clamp_min(1e-12).log()).sum()
            perplexity = float(entropy.exp().item())
            if history_size >= self.history_window:
                dead_mask = usage < self.dead_code_threshold
                dead_count = int(dead_mask.sum().item())
                dead_fraction = float(dead_mask.float().mean().item())
            else:
                dead_count = None
                dead_fraction = None

        coverage: dict[str, int] | None = None
        if (codes is None) != (class_probs is None):
            raise ValueError("codes and class_probs must be supplied together")
        if codes is not None and class_probs is not None:
            if not isinstance(codes, Tensor) or codes.ndim != 4:
                raise ValueError("codes must be a [B,D,H,W] tensor")
            if not isinstance(class_probs, Tensor) or class_probs.ndim != 5 or class_probs.shape[1] != 4 \
                    or tuple(class_probs.shape[0:1] + class_probs.shape[2:]) != tuple(codes.shape):
                raise ValueError("class_probs must be [B,4,D,H,W] and match codes")
            if not torch.is_floating_point(class_probs) or not torch.isfinite(class_probs).all():
                raise ValueError("class_probs must be finite floating probabilities")
            if bool((class_probs < 0).any()):
                raise ValueError("class_probs cannot be negative")
            normalizer = class_probs.sum(dim=1, keepdim=True)
            if bool((normalizer <= 0).any()):
                raise ValueError("class_probs must have positive mass at every cell")
            if bool((codes < 0).any()) or bool((codes >= self.codebook_size).any()):
                raise ValueError("codes contain invalid codebook indices")
            if cohort_id is None:
                if snapshot_id is not None:
                    cohort_id = snapshot_id
                else:
                    counter = int(self.class_cohort_counter.item())
                    cohort_id = f"auto:{counter}"
                    self.class_cohort_counter.add_(1)
            if not isinstance(cohort_id, str) or not cohort_id:
                raise ValueError("cohort_id must be a non-empty string")

            probabilities = class_probs.detach() / normalizer.detach()
            code_ids = codes.detach().reshape(-1).long().to(self.C0.device)
            class_vectors = probabilities.permute(0, 2, 3, 4, 1).reshape(-1, 4)
            mass_by_code = torch.zeros(
                self.codebook_size, 4, device=self.C0.device, dtype=self.class_mass_history.dtype,
            )
            mass_by_code.index_add_(
                0, code_ids, class_vectors.to(device=self.C0.device, dtype=mass_by_code.dtype),
            )
            mass = mass_by_code.T.contiguous()

            try:
                row = self._class_mass_cohort_ids.index(cohort_id)
            except ValueError:
                row = int(self.class_history_index.item())
                self.class_history_index.fill_((row + 1) % self.history_window)
                self.class_history_count.fill_(
                    min(self.history_window, int(self.class_history_count.item()) + 1)
                )
                self._class_mass_cohort_ids[row] = cohort_id
            self.class_mass_history[row].copy_(mass)

        class_history_size = int(self.class_history_count.item())
        if class_history_size:
            if class_history_size < self.history_window:
                accumulated_mass = self.class_mass_history[:class_history_size].sum(dim=0)
            else:
                accumulated_mass = self.class_mass_history.sum(dim=0)
            total_mass = accumulated_mass.sum(dim=0)
            majority = accumulated_mass > (total_mass.unsqueeze(0) * 0.5)
            coverage = {
                name: int(majority[class_id].sum().item())
                for class_id, name in enumerate(_CLASS_NAMES)
            }
            self._last_class_code_coverage = coverage

        churn: float | None = None
        if snapshot_id is not None:
            if not isinstance(snapshot_id, str) or not snapshot_id:
                raise ValueError("snapshot_id must be a non-empty string")
            if codes is None:
                raise ValueError("snapshot_id requires hard codes")
            current = codes.detach().cpu().long()
            previous = self._snapshot_codes.get(snapshot_id)
            if previous is not None and previous.shape == current.shape:
                churn = float((previous != current).float().mean().item())
            self._snapshot_codes[snapshot_id] = current.clone()

        return {
            "initialized": self.codebook_initialized,
            "ema_updates": int(self.ema_updates.item()),
            "history_updates": int(self.history_count.item()),
            "perplexity": perplexity,
            "dead_code_count": dead_count,
            "dead_code_fraction": dead_fraction,
            "class_code_coverage": coverage,
            "class_mass_cohort_count": class_history_size,
            "churn": churn,
            "usage_frequency": usage.detach().cpu().tolist(),
            "codebook": self.artifact_metadata(),
        }

    def check_gate(self, metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Evaluate the frozen-before-joint gate without filling missing values."""

        observed = dict(metrics or self.metrics())
        coverage = observed.get("class_code_coverage") or self._last_class_code_coverage
        missing = []
        for name in ("perplexity", "dead_code_fraction"):
            if observed.get(name) is None:
                missing.append(name)
        if coverage is None:
            missing.append("class_code_coverage")
        if not self.codebook_initialized:
            missing.append("initialized_codebook")
        if int(self.history_count.item()) < self.history_window:
            missing.append(f"{self.history_window}-update_usage_history")
        if missing:
            return {
                "status": "NOT_READY",
                "passed": False,
                "missing": missing,
                "thresholds": {
                    "min_perplexity": self.min_perplexity,
                    "max_dead_code_fraction": self.max_dead_code_fraction,
                    "min_class_codes": self.min_class_codes,
                },
            }

        checks = {
            "perplexity": float(observed["perplexity"]) >= self.min_perplexity,
            "dead_code_fraction": float(observed["dead_code_fraction"]) <= self.max_dead_code_fraction,
            "RV_codes": int(coverage.get("RV", 0)) >= self.min_class_codes,
            "MYO_codes": int(coverage.get("MYO", 0)) >= self.min_class_codes,
            "LV_codes": int(coverage.get("LV", 0)) >= self.min_class_codes,
        }
        passed = all(checks.values())
        return {
            "status": "PASS" if passed else "NO_GO",
            "passed": passed,
            "checks": checks,
            "observed": {
                "perplexity": float(observed["perplexity"]),
                "dead_code_fraction": float(observed["dead_code_fraction"]),
                "class_code_coverage": dict(coverage),
            },
            "thresholds": {
                "min_perplexity": self.min_perplexity,
                "max_dead_code_fraction": self.max_dead_code_fraction,
                "min_class_codes": self.min_class_codes,
            },
        }

    @torch.no_grad()
    def freeze(self) -> dict[str, Any]:
        """Freeze tokenizer weights/codebook only after the explicit gate passes."""

        result = self.check_gate()
        if result["status"] != "PASS":
            raise RuntimeError(f"cannot freeze tokenizer before PASS gate: {result}")
        self.is_frozen.fill_(True)
        # R_T and H_A are tokenizer-training heads. Q-3D's frozen teacher
        # interface retains only A_T and C0 for stage P/J and deploy metadata.
        self.reconstructor = None
        self.anatomy_head = None
        self.requires_grad_(False)
        self.eval()
        return {**self.artifact_metadata(), "gate": result}

    def train(self, mode: bool = True) -> TeacherTokenizer:
        super().train(mode)
        if self.codebook_frozen:
            super().train(False)
        return self

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": CODEBOOK_VERSION,
            "train_heads_dropped": self.reconstructor is None and self.anatomy_head is None,
            "initialization_points": [point.clone() for point in self._initialization_points],
            "snapshot_codes": {key: value.clone() for key, value in self._snapshot_codes.items()},
            "class_mass_cohort_ids": list(self._class_mass_cohort_ids),
            "restart_generator_state": self._restart_generator.get_state().clone(),
            "last_class_code_coverage": self._last_class_code_coverage,
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("version") != CODEBOOK_VERSION:
            raise ValueError("tokenizer checkpoint version mismatch")
        points = state.get("initialization_points", [])
        if not isinstance(points, list) or any(not isinstance(value, Tensor) for value in points):
            raise ValueError("tokenizer checkpoint initialization points are malformed")
        self._initialization_points = [value.detach().cpu().clone() for value in points]
        snapshots = state.get("snapshot_codes", {})
        if not isinstance(snapshots, Mapping):
            raise ValueError("tokenizer checkpoint snapshots are malformed")
        self._snapshot_codes = {
            str(key): value.detach().cpu().long().clone()
            for key, value in snapshots.items() if isinstance(value, Tensor)
        }
        cohort_ids = state.get("class_mass_cohort_ids", [None] * self.history_window)
        if not isinstance(cohort_ids, list) or len(cohort_ids) != self.history_window \
                or any(value is not None and (not isinstance(value, str) or not value) for value in cohort_ids):
            raise ValueError("tokenizer checkpoint class-mass cohort IDs are malformed")
        self._class_mass_cohort_ids = list(cohort_ids)
        generator_state = state.get("restart_generator_state")
        if generator_state is not None:
            if not isinstance(generator_state, Tensor):
                raise ValueError("tokenizer checkpoint restart RNG state is malformed")
            self._restart_generator.set_state(generator_state.detach().cpu())
        coverage = state.get("last_class_code_coverage")
        self._last_class_code_coverage = dict(coverage) if isinstance(coverage, Mapping) else None

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: Mapping[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        extra = state_dict.get(prefix + "_extra_state")
        if isinstance(extra, Mapping) and extra.get("train_heads_dropped") is True:
            self.reconstructor = None
            self.anatomy_head = None
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
        )
        if self.codebook_frozen:
            self.requires_grad_(False)
            self.eval()


__all__ = ["CODEBOOK_VERSION", "TeacherTokenizer"]
