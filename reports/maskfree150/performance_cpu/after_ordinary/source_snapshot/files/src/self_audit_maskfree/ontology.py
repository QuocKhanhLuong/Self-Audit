"""Image-only anatomical role resolution (W2).

Scientific role
---------------
A grouping algorithm returns *anonymous* groups. This module is the only place
where an anonymous group may acquire an anatomical name, and it does so through
explicitly enumerated, machine-readable rules over image geometry and fit-side
appearance. No ground-truth mask, no per-case Hungarian matching against a
reference, and no oracle cluster permutation is reachable from here.

The frozen semantic order is ``0=BG, 1=RV, 2=MYO, 3=LV``.

Enumerated rules
----------------
Every rule below is applied in this fixed order and recorded in
``Hypothesis.metadata['ontology_trace']['rules_fired']``.

``R1_background``
    Every group covering at least :data:`BG_RING_SHARE` of the image border ring
    is background. The union of those groups is BG. If no group reaches the
    share, the largest-area group is BG and ``bg_rule='largest_area'`` is
    recorded as a weaker resolution. This rule is purely geometric; the
    intensity check that background is darker than the fit-global median is
    recorded as a diagnostic (``background_intensity_atypical``) and never
    overrides the geometry.

``R2_merge_excess``
    If more than :data:`MAX_FOREGROUND_GROUPS` non-background groups remain, the
    smallest is merged into its most-adjacent neighbour, repeatedly, until the
    limit holds. Merges are recorded. This bounds the resolution problem; it is
    not a claim that the merged groups were the same tissue.

``R3_enclosure``
    Group ``A`` encloses group ``B`` when at least :data:`ENCLOSURE_FRACTION` of
    ``B``'s pixels lie inside a hole of ``A`` (a connected component of the
    complement of ``A`` that does not touch the image border). The unique
    enclosing pair becomes ``MYO=A`` (the ring) and ``LV=B`` (the cavity it
    surrounds). Several competing pairs are resolved by the largest enclosure
    fraction; an exact tie is left unresolved.

``R4_remainder``
    Non-background groups that are neither MYO nor LV and that are 4-adjacent to
    MYO become RV. A remaining group that is not adjacent to MYO is left
    unresolved rather than being renamed or deleted.

``R5_intensity_prior``
    Used only when R3 cannot resolve. Enumerated hand-specified prior for the
    cardiac short-axis appearance used here: blood pool (LV, RV) is brighter
    than myocardium on the fit-normalized scale, so the darkest remaining group
    is the MYO draft. This is a prior, not a measurement, and whenever it is the
    deciding rule the result is marked ``semantic_unresolved`` with the
    alternatives recorded.

``R6_continuity``
    An optional ``continuity`` mapping of role -> mean fit intensity, carried
    from already-resolved units of the same study, breaks a residual tie by
    nearest mean intensity. Recorded as ``R6_continuity`` when it fires.

``R7_absent``
    Absent classes are legal. A missing RV, MYO or LV is recorded in
    ``absent_roles`` and carries **no** prior penalty: a pathological or simply
    out-of-plane structure must never be manufactured to satisfy a template.

``R8_contour_convention``
    LV denotes the cavity/blood pool including papillary muscle pixels; MYO
    denotes the surrounding wall and excludes the cavity; RV denotes the right
    ventricular cavity. Recorded on every hypothesis for downstream reporting.

``R9_unsupported``
    Zero foreground groups, a non-converged connected-component pass, or a
    required-but-absent geometry key leaves the unit ``semantic_unresolved``
    with ``validity=0`` on the affected pixels. Unresolved pixels are never
    relabeled to background.

``R5b_orientation``
    Left/right disambiguation, implemented, not merely reported. It fires only
    when every key in :data:`ORIENTATION_KEYS` is present *and usable*
    (``geometry_valid`` true, ``view == 'short_axis'``, ``inplane_left_axis`` one
    of :data:`INPLANE_LEFT_AXES`). It then compares the drafted RV and LV
    centroids along the patient-left direction and swaps them if the short-axis
    convention (LV lies towards the patient's left of RV) is contradicted. The
    pair it settles is removed from the recorded alternatives. It answers RV
    versus LV only: which group is the myocardial *ring* is an enclosure
    question, and enclosure is exactly what failed before R5 ran, so the
    hypothesis stays ``semantic_unresolved``. Without usable metadata the rule
    does not fire, the missing and unusable keys are both listed, and the
    left/right question is marked unresolved rather than guessed.

Data dependency (W3)
--------------------
``inplane_left_axis`` must be derived from the native affine by the data layer,
never inferred from pixel content. This checkout contains no real data, so in
every synthetic check the key is absent by construction and the resolver reports
that honestly.

Trainer dependency (W5)
-----------------------
R6 continuity is inert unless the trainer threads it through.
:func:`study_continuity` turns already-resolved units of one study into the
mapping ``resolve_roles(..., continuity=...)`` expects. Until W5 calls it, R6 is
implemented code with no effect, which the trace records as such.

Prior penalty
-------------
``metadata['prior_penalty']`` is read by the W1 observation model. Root review
decision for v1: it is **always** :data:`PRIMARY_PRIOR_PENALTY` = ``0.0``, so a
candidate's score advantage can only come from predictive evidence. Geometric
violations are still detected and recorded under
``ontology_trace['prior_violations']``, together with the diagnostic-only
weights :data:`DIAGNOSTIC_PRIOR_PENALTIES` and the penalty they *would* imply, so
a later contract revision can price them from measurement rather than from a
guess. Because the scored prior is a constant, a semantic permutation can never
acquire an invented likelihood distinction here.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .contracts import SEMANTIC_ORDER, VERSION, FittingView, Hypothesis

BG, RV, MYO, LV = 0, 1, 2, 3
ROLE_NAMES = dict(SEMANTIC_ORDER)

#: Share of the image border ring a group must cover to be called background.
BG_RING_SHARE = 0.10
#: Width in pixels of the border ring used by R1.
BORDER_RING_WIDTH = 1
#: Maximum number of non-background groups kept before R3 runs.
MAX_FOREGROUND_GROUPS = 3
#: Fraction of a group's pixels that must lie inside another group's holes
#: before the second group is called its encloser.
ENCLOSURE_FRACTION = 0.90
#: Two enclosure fractions closer than this count as an exact tie.
ENCLOSURE_TIE = 1e-6
#: Validity assigned to pixels whose anatomical role could not be resolved.
#: Ambiguity is excluded from supervision; it is never converted to background.
UNRESOLVED_VALIDITY = 0.0
#: Whitelisted metadata keys required before any left/right rule may fire.
#: ``inplane_left_axis`` names the image-array direction that points towards the
#: patient's LEFT, as one of ``'+x' | '-x' | '+y' | '-y'`` where ``x`` is the
#: last array axis (columns) and ``y`` the second-to-last (rows). W3 derives it
#: from the native affine; it is never guessed from the pixels.
ORIENTATION_KEYS = ("view", "inplane_left_axis", "geometry_valid")
INPLANE_LEFT_AXES = ("+x", "-x", "+y", "-y")

#: **v1 primary prior penalty.** Root review decision: the anatomical prior
#: contributes exactly zero nats to the score so that a candidate's advantage can
#: only come from predictive evidence. Geometric violations are still detected
#: and reported; they simply do not move the likelihood.
PRIMARY_PRIOR_PENALTY = 0.0
#: Diagnostic-only weights, recorded under
#: ``ontology_trace['diagnostic_prior_penalty']`` and never added to any score in
#: v1. They exist so that a later contract revision can price these violations
#: from measured evidence instead of from a guess.
DIAGNOSTIC_PRIOR_PENALTIES = {
    # The LV cavity should sit inside the myocardial ring.
    "lv_not_enclosed_by_myo": 0.02,
    # A myocardial ring fragmented into several components, per extra component.
    "myo_fragment": 0.01,
    # The RV cavity should not be enclosed by the myocardial ring.
    "rv_enclosed_by_myo": 0.02,
}
#: Cap on the *diagnostic* total, so that no single geometric oddity would
#: dominate the predictive term if these weights were ever switched on.
DIAGNOSTIC_PRIOR_PENALTY_CAP = 0.10

CONTOUR_CONVENTION = (
    "LV = cavity/blood pool including papillary muscle pixels; "
    "MYO = surrounding wall, cavity excluded; RV = right ventricular cavity"
)


class OntologyError(ValueError):
    """Raised when an input violates the frozen maskfree150 ontology contract."""


# ----------------------------------------------------------------------
# geometry helpers (deterministic torch/NumPy; bounded CPU certificate uses scipy)
# ----------------------------------------------------------------------
def _neighbor_max(values: torch.Tensor) -> torch.Tensor:
    """4-connected neighbourhood maximum of a [H,W] tensor, zero-padded."""
    padded = torch.nn.functional.pad(values.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1))
    up = padded[0, 0, :-2, 1:-1]
    down = padded[0, 0, 2:, 1:-1]
    left = padded[0, 0, 1:-1, :-2]
    right = padded[0, 0, 1:-1, 2:]
    return torch.maximum(torch.maximum(up, down), torch.maximum(left, right))


def _connected_components_numpy(mask: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Label a CPU mask with a bounded certificate and exact fallback.

    The SciPy cross-dilation certificate returns only when the frozen recurrence
    must be complete by its cap. Otherwise this function runs the same bounded
    synchronous recurrence as the torch path below: each iteration takes the
    maximum of the current value and its four neighbours, applies the original
    mask, checks for stability, and only then advances the state. The two
    alternating NumPy buffers keep that fallback synchronous without allocating a
    padded tensor on every iteration.
    """
    # ``mask`` is already on CPU at this call site, so this view performs no
    # transfer.  We never write through it; non-contiguous masks remain valid.
    mask_np = mask.detach().numpy()
    height, width = mask_np.shape
    if not bool(mask_np.any()):
        return torch.zeros_like(mask, dtype=torch.long), True

    # The source recurrence starts from float64 raster IDs.  For all material
    # tensor shapes these IDs are exactly representable; keeping the same dtype
    # also preserves the original conversion/overflow behaviour at the API
    # boundary while NumPy handles the stencil in native C loops.
    ids = np.arange(1, height * width + 1, dtype=np.float64).reshape(height, width)
    ids = np.where(mask_np, ids, 0.0)
    budget = 2 * (height + width)

    # A component's maximum raster ID is the sole source that can eventually
    # label every pixel in that component.  ``ndimage.label`` and the bounded
    # dilation below are only a certificate for the fast return; the exact
    # synchronous recurrence remains the fallback whenever the certificate is
    # not full.  Keep all SciPy inputs as ordinary NumPy arrays (rather than
    # relying on an experimental array API) so the CPU contract is explicit.
    from scipy import ndimage

    cross = np.array(
        [[False, True, False], [True, True, True], [False, True, False]],
        dtype=bool,
    )
    component_labels, component_count = ndimage.label(mask_np, structure=cross)
    if component_count:
        # ``ids`` is zero off-mask, so the indexed maximum for each component
        # is exactly the raster ID that the original max-propagation would
        # return after convergence.  Raster IDs are unique, hence each maximum
        # identifies one seed pixel without a tie-breaking choice.
        component_max_ids = np.zeros(component_count + 1, dtype=np.float64)
        np.maximum.at(component_max_ids, component_labels, ids)
        seeds = (component_labels != 0) & (
            ids == component_max_ids[component_labels]
        )
        covered = ndimage.binary_dilation(
            seeds,
            structure=cross,
            mask=mask_np,
            iterations=budget - 1,
            brute_force=False,
        )
        if np.array_equal(covered, mask_np):
            remapped = component_max_ids[component_labels]
            return torch.from_numpy(remapped.astype(np.int64, copy=False)), True

    propagated = np.empty_like(ids)

    for _ in range(budget):
        # Start with the current value (the explicit self maximum in the frozen
        # torch recurrence).  Each in-place maximum then consumes one of the four
        # valid shifted views; the zero-filled image boundary provides padding.
        np.copyto(propagated, ids)
        if height > 1:
            np.maximum(propagated[:-1, :], ids[1:, :], out=propagated[:-1, :])
            np.maximum(propagated[1:, :], ids[:-1, :], out=propagated[1:, :])
        if width > 1:
            np.maximum(propagated[:, :-1], ids[:, 1:], out=propagated[:, :-1])
            np.maximum(propagated[:, 1:], ids[:, :-1], out=propagated[:, 1:])
        propagated[~mask_np] = 0.0

        # The frozen implementation checks convergence before assigning the
        # propagated state, so retain that placement and return the prior IDs.
        if np.array_equal(propagated, ids):
            return torch.from_numpy(ids.astype(np.int64, copy=False)), True
        ids, propagated = propagated, ids

    return torch.from_numpy(ids.astype(np.int64, copy=False)), False


def connected_components(mask: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Label 4-connected components of a bool mask by iterative propagation.

    Returns ``(labels, converged)`` where ``labels`` is zero outside ``mask`` and
    holds an arbitrary positive integer per component. ``converged`` is False if
    the bounded propagation budget ran out; callers must then treat the geometry
    as unresolved rather than trusting a partial labelling.
    """
    if mask.dtype != torch.bool or mask.ndim != 2:
        raise OntologyError("connected_components expects a bool [H,W] mask")
    if mask.device.type == "cpu":
        return _connected_components_numpy(mask)
    height, width = mask.shape
    if not bool(mask.any()):
        return torch.zeros_like(mask, dtype=torch.long), True
    ids = torch.arange(
        1, height * width + 1, dtype=torch.float64, device=mask.device
    ).reshape(height, width)
    ids = torch.where(mask, ids, torch.zeros_like(ids))
    budget = 2 * (height + width)
    for _ in range(budget):
        propagated = torch.where(mask, torch.maximum(ids, _neighbor_max(ids)), torch.zeros_like(ids))
        if bool(torch.equal(propagated, ids)):
            return ids.long(), True
        ids = propagated
    return ids.long(), False


def _holes(mask: torch.Tensor) -> torch.Tensor:
    """Pixels enclosed by ``mask``: complement components not touching the border."""
    complement = ~mask
    labels, converged = connected_components(complement)
    if not converged:
        return torch.zeros_like(mask)
    border = border_ring(*mask.shape, device=mask.device)
    outside = torch.unique(labels[complement & border])
    holes = complement.clone()
    for component in outside.tolist():
        if component != 0:
            holes &= labels != component
    return holes


def border_ring(height: int, width: int, *, device: torch.device | None = None) -> torch.Tensor:
    ring = torch.zeros(height, width, dtype=torch.bool, device=device)
    thickness = BORDER_RING_WIDTH
    ring[:thickness, :] = True
    ring[-thickness:, :] = True
    ring[:, :thickness] = True
    ring[:, -thickness:] = True
    return ring


def adjacency_counts(labels: torch.Tensor, group_a: int, group_b: int) -> int:
    """Number of 4-neighbour pixel pairs crossing from ``group_a`` to ``group_b``."""
    mask_a = labels == group_a
    mask_b = labels == group_b
    horizontal = int((mask_a[:, :-1] & mask_b[:, 1:]).sum() + (mask_b[:, :-1] & mask_a[:, 1:]).sum())
    vertical = int((mask_a[:-1, :] & mask_b[1:, :]).sum() + (mask_b[:-1, :] & mask_a[1:, :]).sum())
    return horizontal + vertical


# ----------------------------------------------------------------------
# resolver
# ----------------------------------------------------------------------
def _group_stats(
    partition: torch.Tensor, image: torch.Tensor, support: torch.Tensor
) -> dict[int, dict[str, Any]]:
    height, width = partition.shape
    ring = border_ring(height, width, device=partition.device)
    ring_total = float(ring.sum().item())
    rows = torch.arange(height, dtype=torch.float32, device=partition.device)
    columns = torch.arange(width, dtype=torch.float32, device=partition.device)
    grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
    stats: dict[int, dict[str, Any]] = {}
    for group in torch.unique(partition).tolist():
        mask = partition == group
        observed = mask & support
        count = int(mask.sum().item())
        observed_count = int(observed.sum().item())
        mean = float(image[observed].mean().item()) if observed_count > 0 else None
        stats[int(group)] = {
            "pixels": count,
            "observed_pixels": observed_count,
            "mean_fit_intensity": mean,
            "ring_share": float((mask & ring).sum().item()) / ring_total if ring_total else 0.0,
            "centroid_y": float(grid_y[mask].mean().item()) if count else None,
            "centroid_x": float(grid_x[mask].mean().item()) if count else None,
            "mask": mask,
        }
    return stats


def _resolve_background(
    stats: Mapping[int, dict[str, Any]], trace: dict[str, Any]
) -> list[int]:
    bg_groups = [g for g, s in stats.items() if s["ring_share"] >= BG_RING_SHARE]
    if bg_groups:
        trace["bg_rule"] = "border_ring_share"
    else:
        largest = max(stats.items(), key=lambda item: item[1]["pixels"])[0]
        bg_groups = [largest]
        trace["bg_rule"] = "largest_area"
    trace["rules_fired"].append("R1_background")
    trace["bg_groups"] = sorted(bg_groups)
    return sorted(bg_groups)


def _merge_excess(
    partition: torch.Tensor,
    foreground: list[int],
    stats: dict[int, dict[str, Any]],
    trace: dict[str, Any],
) -> tuple[torch.Tensor, list[int]]:
    merges: list[dict[str, int]] = []
    partition = partition.clone()
    while len(foreground) > MAX_FOREGROUND_GROUPS:
        smallest = min(foreground, key=lambda g: stats[g]["pixels"])
        others = [g for g in foreground if g != smallest]
        target = max(others, key=lambda g: (adjacency_counts(partition, smallest, g), stats[g]["pixels"]))
        partition[partition == smallest] = target
        merges.append({"merged": smallest, "into": target})
        foreground = others
        stats[target]["pixels"] += stats[smallest]["pixels"]
        stats[target]["mask"] = partition == target
    if merges:
        trace["rules_fired"].append("R2_merge_excess")
        trace["merges"] = merges
    return partition, foreground


def _enclosure_table(
    partition: torch.Tensor, foreground: Sequence[int]
) -> dict[tuple[int, int], float]:
    table: dict[tuple[int, int], float] = {}
    hole_cache = {g: _holes(partition == g) for g in foreground}
    for encloser in foreground:
        holes = hole_cache[encloser]
        if not bool(holes.any()):
            continue
        for enclosed in foreground:
            if enclosed == encloser:
                continue
            mask = partition == enclosed
            total = float(mask.sum().item())
            if total == 0:
                continue
            table[(encloser, enclosed)] = float((mask & holes).sum().item()) / total
    return table


def _intensity_order(foreground: Sequence[int], stats: Mapping[int, dict[str, Any]]) -> list[int]:
    """Foreground groups sorted by mean fit intensity ascending; unobserved last."""
    observed = [g for g in foreground if stats[g]["mean_fit_intensity"] is not None]
    unobserved = [g for g in foreground if stats[g]["mean_fit_intensity"] is None]
    observed.sort(key=lambda g: (stats[g]["mean_fit_intensity"], g))
    return observed + sorted(unobserved)


def _orientation_available(metadata: Mapping[str, Any]) -> bool:
    """True only when every whitelisted key is present, valid and usable.

    A present-but-unusable key (an unknown axis code, ``geometry_valid=False``, a
    long-axis view) counts as unavailable: the rule may not fire on metadata it
    cannot interpret.
    """
    if any(key not in metadata for key in ORIENTATION_KEYS):
        return False
    if not bool(metadata.get("geometry_valid")) or metadata.get("view") != "short_axis":
        return False
    return metadata.get("inplane_left_axis") in INPLANE_LEFT_AXES


def _patient_left_projection(
    group: int, stats: Mapping[int, dict[str, Any]], axis: str
) -> float | None:
    """Signed coordinate of a group's centroid along the patient-left direction.

    ``axis`` is the image-array direction pointing towards the patient's left.
    A larger projection therefore means "further towards the patient's left".
    """
    centroid_y = stats[group]["centroid_y"]
    centroid_x = stats[group]["centroid_x"]
    if centroid_y is None or centroid_x is None:
        return None
    return {
        "+x": centroid_x,
        "-x": -centroid_x,
        "+y": centroid_y,
        "-y": -centroid_y,
    }[axis]


def _apply_orientation_rule(
    order: Sequence[int],
    draft_roles: Sequence[int],
    stats: Mapping[int, dict[str, Any]],
    metadata: Mapping[str, Any],
    trace: dict[str, Any],
) -> list[int]:
    """R5b: order the two blood-pool roles by real in-plane orientation.

    In a short-axis acquisition the left ventricle lies towards the patient's
    left of the right ventricle. Given the affine-derived ``inplane_left_axis``
    from W3, the drafted RV and LV are swapped if their centroids contradict
    that. This rule needs both roles to be drafted and both centroids to exist;
    otherwise it does not fire and the ambiguity stays recorded.

    It resolves RV versus LV only. Which group is the myocardial *ring* is an
    enclosure question, and enclosure is exactly what failed before R5 ran, so
    the hypothesis stays ``semantic_unresolved``.
    """
    roles = list(draft_roles)
    groups = list(order)
    if RV not in roles or LV not in roles:
        trace["orientation_resolution"] = "not_applicable_missing_blood_pool_role"
        return groups
    axis = str(metadata["inplane_left_axis"])
    rv_group = groups[roles.index(RV)]
    lv_group = groups[roles.index(LV)]
    rv_projection = _patient_left_projection(rv_group, stats, axis)
    lv_projection = _patient_left_projection(lv_group, stats, axis)
    if rv_projection is None or lv_projection is None:
        trace["orientation_resolution"] = "unresolved_missing_centroid"
        trace["notes"].append("orientation_rule_skipped_missing_centroid")
        return groups
    trace["rules_fired"].append("R5b_orientation")
    trace["orientation_rule_executed"] = True
    trace["orientation_resolution"] = "resolved_blood_pool_order"
    trace["orientation_rule"] = {
        "inplane_left_axis": axis,
        "rv_patient_left_projection": rv_projection,
        "lv_patient_left_projection": lv_projection,
        "swapped": lv_projection < rv_projection,
    }
    if lv_projection < rv_projection:
        groups[roles.index(RV)], groups[roles.index(LV)] = lv_group, rv_group
    return groups


def _prior_penalty(
    partition: torch.Tensor, assignment: Mapping[int, int], trace: dict[str, Any]
) -> float:
    """Geometric prior violations under the assigned roles, in nats."""
    violations: dict[str, int] = {}
    role_groups = {role: [g for g, r in assignment.items() if r == role] for role in (RV, MYO, LV)}
    myo_mask = torch.zeros_like(partition, dtype=torch.bool)
    for group in role_groups[MYO]:
        myo_mask |= partition == group
    if bool(myo_mask.any()):
        labels, converged = connected_components(myo_mask)
        fragments = int(len(torch.unique(labels[myo_mask]))) if converged else 1
        if fragments > 1:
            violations["myo_fragment"] = fragments - 1
        holes = _holes(myo_mask)
        for role, name in ((LV, "lv_not_enclosed_by_myo"), (RV, "rv_enclosed_by_myo")):
            mask = torch.zeros_like(partition, dtype=torch.bool)
            for group in role_groups[role]:
                mask |= partition == group
            total = float(mask.sum().item())
            if total == 0:
                continue  # R7: an absent class is legal and never penalised.
            inside = float((mask & holes).sum().item()) / total
            if role == LV and inside < ENCLOSURE_FRACTION:
                violations[name] = 1
            if role == RV and inside >= ENCLOSURE_FRACTION:
                violations[name] = 1
    diagnostic = sum(
        DIAGNOSTIC_PRIOR_PENALTIES[name] * count for name, count in violations.items()
    )
    diagnostic = min(diagnostic, DIAGNOSTIC_PRIOR_PENALTY_CAP)
    trace["prior_violations"] = violations
    trace["diagnostic_prior_penalty"] = float(diagnostic)
    trace["diagnostic_prior_penalty_capped"] = diagnostic >= DIAGNOSTIC_PRIOR_PENALTY_CAP
    trace["diagnostic_prior_penalty_weights"] = dict(DIAGNOSTIC_PRIOR_PENALTIES)
    trace["primary_prior_penalty"] = float(PRIMARY_PRIOR_PENALTY)
    trace["prior_penalty_scored"] = False
    # v1 scores zero prior: the predictive term is isolated on purpose.
    return float(PRIMARY_PRIOR_PENALTY)


def resolve_roles(
    partition: torch.Tensor,
    fitting_view: FittingView,
    *,
    candidate_id: str = "candidate",
    source: str = "unspecified",
    continuity: Mapping[str, float] | None = None,
) -> Hypothesis:
    """Name anonymous groups with anatomical roles using image-only rules.

    Parameters
    ----------
    partition:
        ``[H,W]`` long tensor of *anonymous* group identifiers. Values carry no
        anatomical meaning and are never used as class indices.
    fitting_view:
        The only observation source. Selection and verification intensities are
        unreachable from here by construction.
    continuity:
        Optional ``{role_name: mean_fit_intensity}`` carried from already
        resolved units of the same study (R6). Tie-break only.

    Returns
    -------
    Hypothesis
        Labels in the frozen order, one-hot probabilities, per-pixel validity
        (``0`` where the role is unresolved), ``semantic_unresolved``, the
        recorded ``alternatives`` and a full ``ontology_trace``.
    """
    if not isinstance(partition, torch.Tensor) or partition.ndim != 2:
        raise OntologyError("partition must be a [H,W] tensor")
    if not isinstance(fitting_view, FittingView):
        raise OntologyError("resolve_roles requires a contracts.FittingView")
    fitting_view.validate()
    if tuple(partition.shape) != tuple(fitting_view.support.shape):
        raise OntologyError("partition shape must match the fitting view")
    partition = partition.detach().long()
    image = fitting_view.image.detach()[0]
    support = fitting_view.support.detach()
    metadata = fitting_view.metadata or {}

    orientation_available = _orientation_available(metadata)
    trace: dict[str, Any] = {
        "rules_fired": [],
        "contour_convention": CONTOUR_CONVENTION,
        # Availability means the whitelisted metadata can be interpreted. It
        # does not mean that the spatial rule ran; execution is recorded
        # separately only after the rule compares real group centroids.
        "orientation_available": orientation_available,
        "orientation_rule_executed": False,
        "orientation_resolution": (
            "metadata_available_pending" if orientation_available
            else "unresolved_metadata_unavailable"
        ),
        "notes": [],
    }
    if not orientation_available:
        trace["missing_orientation_keys"] = [
            key for key in ORIENTATION_KEYS if key not in metadata
        ]
        trace["unusable_orientation_keys"] = [
            key for key in ORIENTATION_KEYS
            if key in metadata
            and (
                (key == "view" and metadata[key] != "short_axis")
                or (key == "geometry_valid" and not bool(metadata[key]))
                or (
                    key == "inplane_left_axis"
                    and metadata[key] not in INPLANE_LEFT_AXES
                )
            )
        ]
    stats = _group_stats(partition, image, support)
    global_mean = float(image[support].mean().item())

    bg_groups = _resolve_background(stats, trace)
    if all(stats[g]["mean_fit_intensity"] is None or stats[g]["mean_fit_intensity"] > global_mean
           for g in bg_groups):
        trace["notes"].append("background_intensity_atypical")

    foreground = [g for g in stats if g not in bg_groups]
    partition, foreground = _merge_excess(partition, foreground, stats, trace)
    if trace.get("merges"):
        stats = _group_stats(partition, image, support)

    assignment: dict[int, int] = {g: BG for g in bg_groups}
    unresolved_groups: list[int] = []
    alternatives: list[dict[str, Any]] = []
    semantic_unresolved = False

    if not foreground:
        # All-background partition. Legal, scored honestly, but it names no
        # anatomy, so nothing in it may supervise a student.
        trace["rules_fired"].append("R9_unsupported")
        trace["notes"].append("no_foreground_groups")
        trace["orientation_resolution"] = "not_applicable_no_foreground"
        semantic_unresolved = True
    else:
        table = _enclosure_table(partition, foreground)
        pairs = sorted(
            ((value, pair) for pair, value in table.items() if value >= ENCLOSURE_FRACTION),
            key=lambda item: (-item[0], item[1]),
        )
        resolved_pair: tuple[int, int] | None = None
        if pairs:
            best_value, best_pair = pairs[0]
            tied = [p for v, p in pairs if abs(v - best_value) <= ENCLOSURE_TIE]
            if len(tied) == 1:
                resolved_pair = best_pair
                trace["rules_fired"].append("R3_enclosure")
                trace["enclosure_fraction"] = best_value
            else:
                trace["notes"].append("enclosure_tie")
                alternatives.extend(
                    {"rule": "R3_enclosure", "myo_group": int(a), "lv_group": int(b)}
                    for a, b in tied
                )
        trace["enclosure_table"] = {f"{a}->{b}": v for (a, b), v in table.items()}

        if resolved_pair is not None:
            if orientation_available:
                trace["orientation_resolution"] = "not_needed_enclosure_rule_resolved"
            encloser, enclosed = resolved_pair
            assignment[encloser] = MYO
            assignment[enclosed] = LV
            remainder = [g for g in foreground if g not in resolved_pair]
            adjacent = [g for g in remainder if adjacency_counts(partition, g, encloser) > 0]
            for group in adjacent:
                assignment[group] = RV
            detached = [g for g in remainder if g not in adjacent]
            if adjacent:
                trace["rules_fired"].append("R4_remainder")
            if detached:
                unresolved_groups.extend(detached)
                trace["notes"].append("foreground_group_not_adjacent_to_myo")
                semantic_unresolved = True
        else:
            # R5: no usable enclosure. Draft from the enumerated intensity prior
            # and report the ambiguity instead of inventing a distinction.
            semantic_unresolved = True
            trace["rules_fired"].append("R5_intensity_prior")
            order = _intensity_order(foreground, stats)
            draft_roles = {1: [LV], 2: [MYO, LV], 3: [MYO, LV, RV]}.get(len(order))
            if draft_roles is None:  # pragma: no cover - merge rule bounds this
                unresolved_groups.extend(foreground)
            else:
                if continuity:
                    trace["rules_fired"].append("R6_continuity")
                    order = _continuity_order(order, stats, continuity, draft_roles)
                orientation_resolved_pair: tuple[int, int] | None = None
                if orientation_available:
                    order = _apply_orientation_rule(
                        order, draft_roles, stats, metadata, trace
                    )
                    if "R5b_orientation" in trace["rules_fired"]:
                        roles = list(draft_roles)
                        orientation_resolved_pair = (
                            order[roles.index(RV)],
                            order[roles.index(LV)],
                        )
                for group, role in zip(order, draft_roles):
                    assignment[group] = role
                alternatives.extend(
                    _role_permutations(
                        order, draft_roles, resolved_pair=orientation_resolved_pair
                    )
                )
            if not orientation_available:
                trace["notes"].append("orientation_metadata_absent_left_right_unresolved")

    # R9: an unresolved foreground group keeps a *draft* anatomical role and a
    # zero validity. It is never rewritten to background, which would silently
    # delete a structure the rules merely failed to name.
    drafted: dict[int, str] = {}
    for group in unresolved_groups:
        if group in assignment:
            continue
        free = [role for role in (RV, MYO, LV) if role not in assignment.values()]
        role = free[0] if free else RV
        assignment[group] = role
        drafted[int(group)] = ROLE_NAMES[role]
    if drafted:
        trace["drafted_unresolved"] = drafted

    labels = torch.zeros_like(partition)
    for group, role in assignment.items():
        labels[partition == group] = role

    validity = torch.ones_like(labels, dtype=torch.float32)
    unresolved_mask = torch.zeros_like(labels, dtype=torch.bool)
    for group in unresolved_groups:
        unresolved_mask |= partition == group
    if semantic_unresolved and not unresolved_groups:
        if foreground:
            for group in foreground:
                unresolved_mask |= partition == group
        else:
            # All-background partition names no anatomy at all: nothing here may
            # supervise a student, so no pixel is valid.
            unresolved_mask |= torch.ones_like(unresolved_mask)
    validity[unresolved_mask] = UNRESOLVED_VALIDITY

    present = {int(role) for role in torch.unique(labels).tolist()}
    absent = [ROLE_NAMES[role] for role in (RV, MYO, LV) if role not in present]
    if absent:
        trace["rules_fired"].append("R7_absent")
    trace["absent_roles"] = absent
    trace["rules_fired"].append("R8_contour_convention")
    trace["assignment"] = {str(int(g)): ROLE_NAMES[r] for g, r in sorted(assignment.items())}
    trace["unresolved_groups"] = [int(g) for g in unresolved_groups]
    trace["unresolved_pixels"] = int(unresolved_mask.sum().item())

    prior_penalty = _prior_penalty(partition, assignment, trace)
    if semantic_unresolved and alternatives:
        # The enumerated rules declared these namings indistinguishable. Giving
        # them different prior penalties would invent exactly the likelihood
        # distinction the resolver just said it cannot make, so every recorded
        # alternative shares the smallest penalty of the tied set.
        shared = [prior_penalty]
        for alternative in alternatives:
            spec = alternative.get("assignment")
            if not isinstance(spec, Mapping):
                continue
            inverse = {name: role for role, name in ROLE_NAMES.items()}
            candidate_assignment = dict(assignment)
            candidate_assignment.update(
                {int(group): inverse[str(name)] for group, name in spec.items()}
            )
            shared.append(_prior_penalty(partition, candidate_assignment, dict(trace)))
        prior_penalty = min(shared)
        trace["prior_penalty_shared_across_alternatives"] = True
        trace["prior_penalty_tied_values"] = [float(value) for value in shared]

    probabilities = torch.nn.functional.one_hot(labels, 4).permute(2, 0, 1).float()
    return Hypothesis(
        candidate_id=candidate_id,
        labels=labels,
        probabilities=probabilities,
        validity=validity,
        source=source,
        semantic_unresolved=semantic_unresolved,
        alternatives=alternatives,
        metadata={
            "contract_version": VERSION,
            "prior_penalty": prior_penalty,
            "ontology_trace": trace,
            "anonymous_partition_groups": sorted(int(g) for g in stats),
            "study_id": fitting_view.study_id,
            "unit_id": fitting_view.unit_id,
            "partition_id": fitting_view.partition_id,
        },
    )


def _continuity_order(
    order: list[int],
    stats: Mapping[int, dict[str, Any]],
    continuity: Mapping[str, float],
    draft_roles: Sequence[int],
) -> list[int]:
    """Reorder the intensity draft so each role lands on its nearest study mean."""
    remaining = list(order)
    reordered: list[int] = []
    for role in draft_roles:
        target = continuity.get(ROLE_NAMES[role])
        if target is None or not remaining:
            if remaining:
                reordered.append(remaining.pop(0))
            continue
        best = min(
            remaining,
            key=lambda g: (
                abs((stats[g]["mean_fit_intensity"] or 0.0) - float(target)),
                g,
            ),
        )
        remaining.remove(best)
        reordered.append(best)
    reordered.extend(remaining)
    return reordered


def _role_permutations(
    order: Sequence[int],
    draft_roles: Sequence[int],
    *,
    resolved_pair: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Record the competing namings the enumerated rules could not separate.

    R5 only fires when R3 found no enclosure, and enclosure is the single rule
    that distinguishes a myocardial ring from the cavity it surrounds. What is
    left is the intensity prior, which orders the groups but does not identify
    them, so every pairwise swap of the drafted roles stays admissible and is
    recorded. Each alternative describes the same geometry, so all of them carry
    the same prior penalty and the same appearance likelihood.
    """
    roles = list(draft_roles)
    groups = list(order)
    recorded: list[dict[str, Any]] = []
    for first in range(len(roles)):
        for second in range(first + 1, len(roles)):
            if resolved_pair is not None and {groups[first], groups[second]} == set(resolved_pair):
                # R5b answered this one with real geometry, so it is no longer
                # an admissible alternative.
                continue
            swapped = list(roles)
            swapped[first], swapped[second] = swapped[second], swapped[first]
            recorded.append(
                {
                    "rule": "R5_intensity_prior_permutation",
                    "assignment": {
                        str(int(g)): ROLE_NAMES[r] for g, r in zip(groups, swapped)
                    },
                    "note": (
                        f"{ROLE_NAMES[roles[first]]}/{ROLE_NAMES[roles[second]]} are not "
                        "separable here: no enclosure rule fired, only the intensity prior"
                    ),
                }
            )
    return recorded


def apply_alternative(hypothesis: Hypothesis, partition: torch.Tensor,
                      alternative: Mapping[str, Any], candidate_id: str) -> Hypothesis:
    """Materialise a recorded semantic alternative as its own draft hypothesis.

    The partition geometry is unchanged; only the names move. The prior penalty
    is recomputed from the same enumerated rules so that a permutation the rules
    cannot separate keeps an identical score contribution.
    """
    assignment_spec = alternative.get("assignment")
    if not isinstance(assignment_spec, Mapping):
        raise OntologyError("alternative has no 'assignment' mapping")
    inverse = {name: role for role, name in ROLE_NAMES.items()}
    assignment = {int(group): inverse[str(name)] for group, name in assignment_spec.items()}
    labels = hypothesis.labels.clone()
    partition = partition.detach().long()
    for group, role in assignment.items():
        labels[partition == group] = role
    trace = dict(hypothesis.metadata.get("ontology_trace", {}))
    trace = {**trace, "rules_fired": list(trace.get("rules_fired", [])) + ["semantic_alternative"]}
    # Same geometry, same tied prior: a permutation the rules cannot separate
    # must not gain or lose likelihood by being written down differently.
    prior_penalty = float(hypothesis.metadata.get("prior_penalty", 0.0))
    probabilities = torch.nn.functional.one_hot(labels, 4).permute(2, 0, 1).float()
    return Hypothesis(
        candidate_id=candidate_id,
        labels=labels,
        probabilities=probabilities,
        validity=hypothesis.validity.clone(),
        source="semantic_alternative",
        semantic_unresolved=True,
        alternatives=[dict(alternative)],
        metadata={
            **hypothesis.metadata,
            "prior_penalty": prior_penalty,
            "ontology_trace": trace,
            "derived_from": hypothesis.candidate_id,
        },
    )


def study_continuity(hypotheses: Sequence[Hypothesis], fitting_views: Sequence[FittingView]
                     ) -> dict[str, float]:
    """Build the R6 continuity mapping from already-resolved units of one study.

    This is the API the trainer (W5) must call to make R6 do anything: resolve
    the units of a study in order, pass the accumulated mapping into
    ``resolve_roles(..., continuity=...)`` for the next unit, and reset it at a
    study boundary. Only fitting observations are read, and only hypotheses whose
    roles were actually resolved contribute.

    Returns ``{role_name: mean fit intensity}`` over resolved, non-background
    roles. An empty mapping means "no usable continuity yet", which is the
    correct input for the first unit of a study.
    """
    if len(hypotheses) != len(fitting_views):
        raise OntologyError("study_continuity needs one fitting view per hypothesis")
    totals: dict[str, list[float]] = {}
    for hypothesis, view in zip(hypotheses, fitting_views):
        if hypothesis.semantic_unresolved:
            continue
        if view.study_id != fitting_views[0].study_id:
            raise OntologyError("study_continuity must not mix studies")
        image = view.image.detach()[0]
        support = view.support.detach()
        for role in (RV, MYO, LV):
            mask = (hypothesis.labels == role) & support
            if not bool(mask.any()):
                continue
            totals.setdefault(ROLE_NAMES[role], []).append(float(image[mask].mean().item()))
    return {name: sum(values) / len(values) for name, values in totals.items()}
