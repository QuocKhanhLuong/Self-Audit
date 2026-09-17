"""Bounded anatomical candidate banks (W2).

Scientific role
---------------
``generate_bank`` produces exactly :data:`BANK_SIZE` competing partitions of one
study unit, the incumbent included, using **fitting observations only**. The
same bank is later handed unchanged to every selector compared in the experiment
suite, so a selector difference can never be confused with a search-budget
difference.

Sources of the four candidates, in fixed order:

0. ``grouping`` - bounded k-means over fit-side intensity, optional producer
   features and lightly weighted spatial coordinates. An anonymous grouping.
1. ``anatomical_initializer`` - an image-only rule-based initializer: brightest
   compact blob near the field-of-view centre, its surrounding annulus, and the
   nearest bright neighbour blob. Still anonymous groups.
2. ``boundary_edit`` - the incumbent partition with its largest foreground group
   grown by :data:`BOUNDARY_EDIT_PIXELS` pixel(s) into its neighbours.
3. ``split_merge`` or ``semantic_alternative`` - the incumbent's largest
   foreground group split at its fit-intensity median, unless the incumbent came
   back ``semantic_unresolved`` with recorded alternatives, in which case the
   competing *naming* of the same geometry is banked instead so that a tie in
   appearance likelihood still leaves a competing hypothesis on the table.

Every candidate is named by :func:`~self_audit_maskfree.ontology.resolve_roles`.
No step here calls a grouping output "anatomy" on its own.

Observation firewall
--------------------
Only :class:`~self_audit_maskfree.contracts.FittingView` is accepted. Hidden
pixels are physically zero in that view, so clustering runs on the fit support
and hidden locations receive the label of the nearest fit pixel
(:func:`nearest_fit_extension`). Selection and verification intensities are not
reachable from this module.

An edit type never decides whether an edit is good. Nothing here ranks
candidates; ranking belongs to :mod:`self_audit_maskfree.auditor`.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Sequence

import torch

from .contracts import VERSION, FittingView, Hypothesis
from .ontology import (
    BG,
    OntologyError,
    apply_alternative,
    connected_components,
    resolve_roles,
)

#: Four total candidates, incumbent included. Provisional fixed budget.
BANK_SIZE = 4
#: Number of *anonymous* groups the incumbent grouping looks for. Deliberately
#: larger than the four anatomical roles: background routinely splits into
#: several appearance groups, and pretending that k groups are k anatomical
#: classes is exactly what the ontology layer exists to prevent. R1 folds the
#: border-touching groups into BG and R2 bounds what is left.
GROUPING_K = 6
#: Bounded Lloyd iterations. Deterministic init, so no restart loop.
KMEANS_ITERATIONS = 10
#: Producer feature channels consumed, if features are supplied at all.
MAX_FEATURE_CHANNELS = 8
#: Weight of the normalized (y,x) coordinates in the grouping feature vector.
SPATIAL_WEIGHT = 0.25
#: Pixels the boundary edit grows the largest foreground group by.
BOUNDARY_EDIT_PIXELS = 1
#: Intensity quantile above which the anatomical initializer looks for cavity.
INITIALIZER_CAVITY_QUANTILE = 0.90
#: Half-width of the central box the initializer searches, as a share of the FOV.
INITIALIZER_CENTER_SHARE = 0.25
#: Annulus thickness in pixels the initializer uses for the wall group.
INITIALIZER_WALL_PIXELS = 3
#: Radius, as a share of the FOV diagonal, within which a second bright blob is
#: taken as the neighbouring cavity group.
INITIALIZER_NEIGHBOUR_SHARE = 0.35


class BankError(ValueError):
    """Raised when bank generation violates the frozen maskfree150 contract."""


# ----------------------------------------------------------------------
# grouping primitives
# ----------------------------------------------------------------------
def _standardize(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(dim=0, keepdim=True)
    std = values.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (values - mean) / std


def _feature_matrix(
    fitting_view: FittingView, features: torch.Tensor | None
) -> torch.Tensor:
    """Per-fit-pixel feature rows: intensity, optional producer channels, coordinates."""
    support = fitting_view.support
    height, width = support.shape
    image = fitting_view.image.detach()[0]
    columns = [image[support].reshape(-1, 1)]
    if features is not None:
        if features.ndim == 4 and features.shape[0] == 1:
            features = features[0]
        if features.ndim != 3 or tuple(features.shape[1:]) != (height, width):
            raise BankError("features must be [C,H,W] matching the fitting view")
        # Candidate generation is an image-only CPU audit boundary. Detach and
        # materialise the consumed producer channels on CPU before any hashing
        # or clustering; move only the bounded matrix to the fitting-view device
        # for the final tensor operations.
        channels = features.detach().to(device="cpu", dtype=torch.float32)[:MAX_FEATURE_CHANNELS]
        channels = channels.to(device=support.device)
        columns.append(channels.permute(1, 2, 0)[support])
    rows = _standardize(torch.cat(columns, dim=1).to(torch.float32))
    device = support.device
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    coordinates = torch.stack([ys[support], xs[support]], dim=1) * SPATIAL_WEIGHT
    return torch.cat([rows, coordinates], dim=1)


def _kmeans(rows: torch.Tensor, k: int, iterations: int, seed: int) -> torch.Tensor:
    """Deterministic bounded k-means. Returns anonymous group ids per row.

    Initialization is by evenly spaced quantiles of the first feature column, so
    the result does not depend on a random draw; ``seed`` only perturbs exact
    ties and is recorded for reproducibility.
    """
    device = rows.device
    if rows.shape[0] < k:
        return torch.zeros(rows.shape[0], dtype=torch.long, device=device)
    quantiles = torch.linspace(0.5 / k, 1.0 - 0.5 / k, steps=k, device=device)
    anchors = torch.quantile(rows[:, 0], quantiles)
    centroids = torch.stack([rows[(rows[:, 0] - a).abs().argmin()] for a in anchors])
    # The generator is a CPU generator by contract, so the jitter is drawn on the
    # CPU and moved: the same seed gives the same bank on any device.
    generator = torch.Generator().manual_seed(int(seed))
    jitter = torch.randn(centroids.shape, generator=generator) * 1e-6
    centroids = centroids + jitter.to(device=device, dtype=centroids.dtype)
    assignments = torch.zeros(rows.shape[0], dtype=torch.long, device=device)
    for _ in range(iterations):
        distances = torch.cdist(rows, centroids)
        updated = distances.argmin(dim=1)
        if torch.equal(updated, assignments):
            break
        assignments = updated
        for index in range(k):
            member = assignments == index
            if bool(member.any()):
                centroids[index] = rows[member].mean(dim=0)
    return assignments


def nearest_fit_extension(partition: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Fill non-fit locations with the group of the nearest fit pixel.

    Bounded 4-connected dilation. Hidden intensities are never read; only the
    *fit-side* labels travel outwards, which is what the contract means by
    "nearest-fit extension for hidden locations".
    """
    height, width = support.shape
    filled = partition.clone()
    filled[~support] = -1
    known = support.clone()
    for _ in range(height + width):
        if bool(known.all()):
            break
        padded = torch.nn.functional.pad(filled.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), value=-1)
        neighbours = torch.stack(
            [
                padded[0, 0, :-2, 1:-1],
                padded[0, 0, 2:, 1:-1],
                padded[0, 0, 1:-1, :-2],
                padded[0, 0, 1:-1, 2:],
            ]
        )
        best = neighbours.max(dim=0).values
        newly = (~known) & (best >= 0)
        if not bool(newly.any()):
            break
        filled[newly] = best[newly]
        known |= newly
    if bool((filled < 0).any()):  # pragma: no cover - support is never empty
        filled[filled < 0] = 0
    return filled


def _dense_partition(
    assignments: torch.Tensor, fitting_view: FittingView
) -> torch.Tensor:
    support = fitting_view.support
    partition = torch.zeros(support.shape, dtype=torch.long, device=support.device)
    partition[support] = assignments.to(partition.device)
    return nearest_fit_extension(partition, support)


# ----------------------------------------------------------------------
# the four candidate constructions
# ----------------------------------------------------------------------
def _grouping_partition(
    fitting_view: FittingView, features: torch.Tensor | None, seed: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    rows = _feature_matrix(fitting_view, features)
    assignments = _kmeans(rows, GROUPING_K, KMEANS_ITERATIONS, seed)
    cost = {
        "algorithm": "bounded_kmeans",
        "k": GROUPING_K,
        "max_iterations": KMEANS_ITERATIONS,
        "feature_columns": int(rows.shape[1]),
        "fit_pixels": int(rows.shape[0]),
        "uses_producer_features": features is not None,
    }
    return _dense_partition(assignments, fitting_view), cost


def _initializer_partition(fitting_view: FittingView) -> tuple[torch.Tensor, dict[str, Any]]:
    """Rule-based image-only anatomical initializer over fit observations."""
    support = fitting_view.support
    height, width = support.shape
    image = fitting_view.image.detach()[0]
    observed = image[support]
    threshold = float(torch.quantile(observed, INITIALIZER_CAVITY_QUANTILE).item())
    bright = (image >= threshold) & support

    share = INITIALIZER_CENTER_SHARE
    center_box = torch.zeros_like(support)
    y0, y1 = int(height * (0.5 - share)), int(height * (0.5 + share))
    x0, x1 = int(width * (0.5 - share)), int(width * (0.5 + share))
    center_box[y0:y1, x0:x1] = True

    labels, converged = connected_components(bright)
    partition = torch.zeros(support.shape, dtype=torch.long, device=support.device)
    cost: dict[str, Any] = {
        "algorithm": "rule_based_anatomical_initializer",
        "cavity_quantile": INITIALIZER_CAVITY_QUANTILE,
        "components_converged": converged,
        "wall_pixels": INITIALIZER_WALL_PIXELS,
    }
    central = labels[bright & center_box]
    if not converged or central.numel() == 0:
        cost["resolved"] = False
        return partition, cost

    identifiers, counts = torch.unique(central, return_counts=True)
    cavity_id = int(identifiers[counts.argmax()].item())
    cavity = labels == cavity_id
    partition[cavity] = 1

    wall = cavity.clone()
    for _ in range(INITIALIZER_WALL_PIXELS):
        wall = _dilate(wall)
    partition[wall & ~cavity] = 2

    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=support.device),
        torch.arange(width, dtype=torch.float32, device=support.device),
        indexing="ij",
    )
    cavity_y = float(ys[cavity].mean().item())
    cavity_x = float(xs[cavity].mean().item())
    radius = INITIALIZER_NEIGHBOUR_SHARE * float((height**2 + width**2) ** 0.5)
    near = ((ys - cavity_y).pow(2) + (xs - cavity_x).pow(2)).sqrt() <= radius
    neighbour_candidates = bright & near & (partition == 0)
    if bool(neighbour_candidates.any()):
        neighbour_ids, neighbour_counts = torch.unique(
            labels[neighbour_candidates], return_counts=True
        )
        neighbour_id = int(neighbour_ids[neighbour_counts.argmax()].item())
        partition[(labels == neighbour_id) & (partition == 0)] = 3
    cost["resolved"] = True
    return partition, cost


def _dilate(mask: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(
        mask.unsqueeze(0).unsqueeze(0).float(), (1, 1, 1, 1)
    )
    stacked = torch.stack(
        [
            padded[0, 0, :-2, 1:-1],
            padded[0, 0, 2:, 1:-1],
            padded[0, 0, 1:-1, :-2],
            padded[0, 0, 1:-1, 2:],
            padded[0, 0, 1:-1, 1:-1],
        ]
    )
    return stacked.max(dim=0).values > 0


def _largest_foreground_group(
    partition: torch.Tensor, labels: torch.Tensor
) -> int | None:
    """Anonymous group holding the most non-background *named* pixels."""
    best, best_count = None, 0
    for group in torch.unique(partition).tolist():
        mask = partition == int(group)
        count = int((mask & (labels != BG)).sum().item())
        if count > best_count:
            best, best_count = int(group), count
    return best


def _boundary_edit_partition(
    partition: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    group = _largest_foreground_group(partition, labels)
    edited = partition.clone()
    cost = {
        "algorithm": "boundary_growth",
        "pixels": BOUNDARY_EDIT_PIXELS,
        "target_group": group,
    }
    if group is None:
        cost["resolved"] = False
        return edited, cost
    mask = partition == group
    for _ in range(BOUNDARY_EDIT_PIXELS):
        mask = _dilate(mask)
    edited[mask] = group
    cost["resolved"] = True
    cost["changed_pixels"] = int((edited != partition).sum().item())
    return edited, cost


def _split_partition(
    partition: torch.Tensor, labels: torch.Tensor, fitting_view: FittingView
) -> tuple[torch.Tensor, dict[str, Any]]:
    group = _largest_foreground_group(partition, labels)
    edited = partition.clone()
    cost: dict[str, Any] = {"algorithm": "intensity_median_split", "target_group": group}
    if group is None:
        cost["resolved"] = False
        return edited, cost
    image = fitting_view.image.detach()[0]
    mask = (partition == group) & fitting_view.support
    if int(mask.sum().item()) < 2:
        cost["resolved"] = False
        return edited, cost
    median = float(image[mask].median().item())
    new_group = int(partition.max().item()) + 1
    edited[(partition == group) & (image > median)] = new_group
    cost["resolved"] = True
    cost["median_fit_intensity"] = median
    cost["changed_pixels"] = int((edited != partition).sum().item())
    return edited, cost


# ----------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------
def _hamming(a: Hypothesis, b: Hypothesis) -> float:
    return float((a.labels != b.labels).float().mean().item())


def _digest(*parts: Any) -> str:
    """SHA256 over exact tensor bytes and scalar identity, never over latency.

    Tensors are moved to the CPU and made contiguous first so that the same
    content hashes identically on any device. Timing, wall clock and any other
    non-reproducible quantity is deliberately excluded: a bank identity must
    change when and only when the bank's *content* changes.
    """
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, torch.Tensor):
            tensor = part.detach().cpu().contiguous()
            hasher.update(str(tuple(tensor.shape)).encode())
            hasher.update(str(tensor.dtype).encode())
            hasher.update(tensor.numpy().tobytes())
        else:
            hasher.update(repr(part).encode())
        hasher.update(b"|")
    return hasher.hexdigest()


def feature_identity(features: torch.Tensor | None) -> str:
    """Identity of the producer features a bank was generated from.

    ``'none'`` when the bank used intensities alone. Otherwise the digest of the
    exact channels consumed, so a bank built from one producer epoch can never be
    mistaken for a bank built from another.
    """
    if features is None:
        return "none"
    channels = features.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if channels.ndim == 4 and channels.shape[0] == 1:
        channels = channels[0]
    return _digest("features", channels[:MAX_FEATURE_CHANNELS])


def bank_identity(
    fitting_view: FittingView,
    bank: Sequence[Hypothesis],
    *,
    features: torch.Tensor | None,
    seed: int,
) -> dict[str, str]:
    """Content hash binding a bank to the exact inputs that produced it.

    A stale-cache join is the failure this prevents: a bank keyed by
    ``unit_id`` alone would silently match a later epoch's different bank. The
    hash covers the observation partition, the fitting inputs, the producer
    feature identity, the generation seed and every candidate's label tensor.
    Latency and wall clock are excluded.
    """
    # The bank identity is a content hash of exactly four identities: the
    # observation partition, fit-side inputs, consumed feature channels and the
    # candidate contents. Generation latency and wall clock are operational
    # measurements and must never affect a cache join.
    partition_hash = _digest(
        "partition", fitting_view.study_id, fitting_view.unit_id,
        fitting_view.partition_id, fitting_view.protocol,
    )
    fit_input_hash = _digest(
        "fit_input", fitting_view.image, fitting_view.support, fitting_view.context,
    )
    feature_hash = feature_identity(features)
    # Include every candidate field that can change a draft or its supervision
    # meaning, while deliberately excluding metadata such as generation latency,
    # bank ids and device strings.
    candidate_parts: list[Any] = []
    for candidate in bank:
        candidate_parts.extend(
            (
                candidate.candidate_id,
                candidate.source,
                bool(candidate.semantic_unresolved),
                candidate.labels,
                candidate.probabilities,
                candidate.validity,
                json.dumps(candidate.alternatives, sort_keys=True, separators=(",", ":"), default=str),
                float(candidate.metadata.get("prior_penalty", 0.0)),
            )
        )
    candidate_content_hash = _digest("candidate_content", *candidate_parts)
    bank_hash = _digest(
        "maskfree_bank_v3", partition_hash, fit_input_hash, feature_hash,
        candidate_content_hash,
    )
    return {
        "bank_id": bank_hash[:32],
        "partition_hash": partition_hash[:32],
        "fit_input_hash": fit_input_hash[:32],
        "feature_hash": feature_hash if feature_hash == "none" else feature_hash[:32],
        "candidate_content_hash": candidate_content_hash[:32],
        # Backward-compatible aliases used by existing consumers.
        "fit_inputs_digest": fit_input_hash[:32],
        "feature_identity": feature_hash if feature_hash == "none" else feature_hash[:32],
        "candidates_digest": candidate_content_hash[:32],
    }


def generate_bank(
    fitting_view: FittingView,
    features: torch.Tensor | None = None,
    *,
    seed: int = 42,
) -> list[Hypothesis]:
    """Build the frozen four-candidate bank for one unit, fitting observations only.

    Parameters
    ----------
    fitting_view:
        The only observation source. ``ScoringView`` is deliberately not a
        parameter: nothing here may see a selection or verification intensity.
    features:
        Optional detached producer features ``[C,H,W]`` (or ``[1,C,H,W]``).
        Producer gradients never flow through this function.
    seed:
        Recorded and used for tie perturbation only; the bank is deterministic.

    Returns
    -------
    list[Hypothesis]
        Exactly :data:`BANK_SIZE` hypotheses, index 0 the incumbent. Each carries
        ``metadata['generation']`` with its source, cost and distinctness against
        the incumbent, and ``metadata['bank_id']``.
    """
    if not isinstance(fitting_view, FittingView):
        raise BankError("generate_bank requires a contracts.FittingView")
    fitting_view.validate()
    if features is not None:
        # Make the CPU audit boundary explicit. The producer's graph never
        # reaches candidate construction, and the exact consumed channels are
        # stable across devices for both the bank hash and k-means input.
        features = features.detach().to(device="cpu", dtype=torch.float32).contiguous()

    started = time.perf_counter()
    partitions: list[torch.Tensor] = []
    bank: list[Hypothesis] = []
    costs: list[dict[str, Any]] = []

    incumbent_partition, grouping_cost = _grouping_partition(fitting_view, features, seed)
    incumbent = resolve_roles(
        incumbent_partition,
        fitting_view,
        candidate_id=f"{fitting_view.unit_id}:c0_grouping",
        source="grouping",
    )
    partitions.append(incumbent_partition)
    bank.append(incumbent)
    costs.append(grouping_cost)

    initializer_partition, initializer_cost = _initializer_partition(fitting_view)
    partitions.append(initializer_partition)
    bank.append(
        resolve_roles(
            initializer_partition,
            fitting_view,
            candidate_id=f"{fitting_view.unit_id}:c1_initializer",
            source="anatomical_initializer",
        )
    )
    costs.append(initializer_cost)

    boundary_partition, boundary_cost = _boundary_edit_partition(
        incumbent_partition, incumbent.labels
    )
    partitions.append(boundary_partition)
    bank.append(
        resolve_roles(
            boundary_partition,
            fitting_view,
            candidate_id=f"{fitting_view.unit_id}:c2_boundary",
            source="boundary_edit",
        )
    )
    costs.append(boundary_cost)

    semantic_alternatives = [
        alternative
        for alternative in incumbent.alternatives
        if isinstance(alternative.get("assignment"), dict)
    ]
    if incumbent.semantic_unresolved and semantic_alternatives:
        # A tie in appearance likelihood must still leave a competing hypothesis
        # in the bank; the geometry is identical, only the naming differs.
        chosen = _distinct_alternative(incumbent, incumbent_partition, semantic_alternatives)
        partitions.append(incumbent_partition)
        bank.append(chosen)
        costs.append({"algorithm": "semantic_alternative", "resolved": True})
    else:
        split_partition, split_cost = _split_partition(
            incumbent_partition, incumbent.labels, fitting_view
        )
        partitions.append(split_partition)
        bank.append(
            resolve_roles(
                split_partition,
                fitting_view,
                candidate_id=f"{fitting_view.unit_id}:c3_split",
                source="split_merge",
            )
        )
        costs.append(split_cost)

    if len(bank) != BANK_SIZE:  # pragma: no cover - construction is fixed
        raise BankError(f"bank must hold exactly {BANK_SIZE} candidates")

    # Identity is computed from content *before* any timing is attached, so the
    # hash cannot depend on how fast this machine happened to be.
    identity = bank_identity(fitting_view, bank, features=features, seed=seed)
    elapsed = time.perf_counter() - started
    for index, (hypothesis, cost) in enumerate(zip(bank, costs)):
        hypothesis.metadata.update(identity)
        hypothesis.metadata["bank_index"] = index
        hypothesis.metadata["generation"] = {
            **cost,
            "source": hypothesis.source,
            "seed": seed,
            "device": str(fitting_view.support.device),
            "features_detached_cpu": features is not None,
            "hamming_vs_incumbent": _hamming(hypothesis, bank[0]),
            "identical_to_incumbent": index > 0 and _hamming(hypothesis, bank[0]) == 0.0,
            "contract_version": VERSION,
        }
        # Latency lives outside `generation` and outside the hash: it is an
        # operational measurement, not part of the bank's identity.
        hypothesis.metadata["generation_latency"] = {"bank_wall_seconds": elapsed}
    return bank


def _distinct_alternative(
    incumbent: Hypothesis, partition: torch.Tensor, alternatives: list[dict[str, Any]]
) -> Hypothesis:
    """Materialise the first recorded naming that actually differs from the incumbent."""
    last_error: OntologyError | None = None
    for index, alternative in enumerate(alternatives):
        try:
            candidate = apply_alternative(
                incumbent, partition, alternative,
                candidate_id=f"{incumbent.metadata['unit_id']}:c3_semantic{index}",
            )
        except OntologyError as error:  # pragma: no cover - guarded by the caller
            last_error = error
            continue
        if _hamming(candidate, incumbent) > 0.0:
            return candidate
    if last_error is not None:  # pragma: no cover
        raise last_error
    # Every recorded naming coincided with the draft. Keep it and let the
    # auditor report the bank as degenerate rather than inventing a difference.
    return apply_alternative(
        incumbent, partition, alternatives[0],
        candidate_id=f"{incumbent.metadata['unit_id']}:c3_semantic0",
    )
