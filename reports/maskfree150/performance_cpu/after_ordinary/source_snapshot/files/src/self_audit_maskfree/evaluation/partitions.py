"""Degenerate and control partitions, plus frozen-parameter substitution.

These constructors exist so that the pipeline can be attacked with partitions it
must *not* prefer: an all-background draft, a spatially random draft, a class
permutation of the selected draft, and a deliberately over-split draft. They are
diagnostics. None of them is ever fed back into training as an augmentation, and
none is labelled good or bad by its edit type.

A degenerate challenger must be *fitted* like any other candidate, at the same
capacity and the same budget, before it can be scored. Swapping a partition into
someone else's frozen appearance parameters is not available: the observation
model binds each fit to the exact partition it was fitted on and refuses a
mutated one. That is the correct restriction, so the degenerate challenges that
appear in post-freeze verification are the ones that were fitted *before* the
freeze and frozen alongside every other compared method.
"""
from __future__ import annotations

from typing import Sequence

import torch

from ..contracts import Hypothesis, hypothesis_from_labels

NUM_CLASSES = 4


def all_background_hypothesis(shape: tuple[int, int], *, candidate_id: str = "control_all_background") -> Hypothesis:
    """Everything is background. The classic degenerate winner of a weak likelihood."""
    labels = torch.zeros(shape, dtype=torch.long)
    hypothesis = hypothesis_from_labels(labels, candidate_id, "negative_control")
    hypothesis.metadata["degenerate"] = "all_background"
    return hypothesis


def random_mask_hypothesis(
    shape: tuple[int, int],
    *,
    seed: int = 42,
    block: int = 8,
    candidate_id: str = "control_random_spatial_masks",
) -> Hypothesis:
    """Spatially random blocky partition with the same number of classes.

    Blocks rather than per-pixel noise, so the control has a comparable boundary
    length to a real draft instead of losing purely on the complexity term.
    """
    height, width = shape
    generator = torch.Generator().manual_seed(seed)
    blocks = torch.randint(
        0, NUM_CLASSES,
        ((height + block - 1) // block, (width + block - 1) // block),
        generator=generator,
        dtype=torch.long,
    )
    labels = blocks.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:height, :width]
    hypothesis = hypothesis_from_labels(labels.contiguous(), candidate_id, "negative_control")
    hypothesis.metadata["degenerate"] = "random_mask"
    hypothesis.metadata["block"] = block
    return hypothesis


def class_permutation_hypothesis(
    source: Hypothesis,
    permutation: Sequence[int] = (0, 2, 3, 1),
    *,
    candidate_id: str = "control_class_permutation",
) -> Hypothesis:
    """Relabel an existing draft under a fixed class permutation.

    Appearance likelihood is invariant to this relabelling once the regions are
    refitted, and nearly invariant under frozen parameters only because the
    components move with the labels. The control exists to expose that the
    ontology - not the likelihood - is what assigns anatomical names.
    """
    mapping = torch.tensor(list(permutation), dtype=torch.long)
    if mapping.numel() != NUM_CLASSES or sorted(mapping.tolist()) != list(range(NUM_CLASSES)):
        raise ValueError("permutation must be a permutation of 0..3")
    labels = mapping[source.labels.detach().long()]
    hypothesis = hypothesis_from_labels(labels, candidate_id, "negative_control")
    hypothesis.validity = source.validity.detach().clone()
    hypothesis.semantic_unresolved = True
    hypothesis.metadata["degenerate"] = "class_permutation"
    hypothesis.metadata["permutation"] = list(permutation)
    hypothesis.metadata["permuted_from"] = source.candidate_id
    hypothesis.metadata["note"] = (
        "semantic names permuted; appearance evidence cannot distinguish this from the source"
    )
    return hypothesis


def excessive_partition_hypothesis(
    shape: tuple[int, int],
    *,
    cell: int = 2,
    candidate_id: str = "control_excessive_partition",
) -> Hypothesis:
    """Deterministic fine tiling: maximum boundary length inside the 4-class space."""
    height, width = shape
    rows = torch.arange(height).unsqueeze(1) // cell
    columns = torch.arange(width).unsqueeze(0) // cell
    labels = ((rows + columns) % NUM_CLASSES).long()
    hypothesis = hypothesis_from_labels(labels, candidate_id, "negative_control")
    hypothesis.metadata["degenerate"] = "excessive_partition"
    hypothesis.metadata["cell"] = cell
    return hypothesis


def intensity_grouping_hypothesis(
    image: torch.Tensor,
    support: torch.Tensor,
    *,
    candidate_id: str = "e0_intensity_grouping",
    source: str = "image_intensity_quantiles",
) -> Hypothesis:
    """Simple image-only intensity grouping: quantile bins over the fit support.

    This is the E0 baseline's partition generator. It uses nothing but fitting
    intensities: no features, no audit, no reference. Pixels outside the fitting
    support take the nearest available bin boundary through the same quantile
    rule applied to the (zeroed) value, which is a fitting-side extension, not a
    read of the withheld intensity.
    """
    if image.ndim == 3:
        image = image[0]
    values = image.detach().to(torch.float32)
    observed = values[support]
    if observed.numel() == 0:
        raise ValueError("intensity grouping needs a non-empty fitting support")
    quantiles = torch.quantile(observed, torch.tensor([0.25, 0.5, 0.75], dtype=observed.dtype))
    labels = torch.bucketize(values, quantiles.to(values.dtype)).long().clamp_(0, NUM_CLASSES - 1)
    hypothesis = hypothesis_from_labels(labels, candidate_id, source)
    hypothesis.semantic_unresolved = True
    hypothesis.metadata["thresholds"] = [float(value) for value in quantiles.tolist()]
    hypothesis.metadata["note"] = (
        "intensity quantile bins are not anatomical roles until an ontology resolver runs"
    )
    return hypothesis
