"""Fixed observation-role partitions with guard bands.

One partition per **study grid**, decided *before* any learning, from the study
identity and the global seed only -- never from intensities, features, masks,
unit identity or model state. It is hashed into ``partition_id`` so a trainer,
an auditor and a post-freeze verifier can prove they are talking about the same
partition.

The partition is deliberately *not* seeded per unit. If every slice or cine
time of one study drew its own role map, a voxel sealed as ``verify`` in one
unit would be a fitting context voxel in the neighbouring unit, and global
self-supervised training would consume the sealed observation through the back
door. Every unit of a study, at a given native grid, therefore shares one XY
role mask -- across Z and across time.

Roles (architecture contract, spatial profile):

* ``select`` -- 20% of 8-pixel blocks; adaptive evidence, the auditor may see it;
* ``verify`` -- 20% of blocks; sealed until every prediction is frozen;
* ``guard``  -- a 2-pixel band around every withheld block, removed from fit so
  a smooth interpolator cannot read a withheld value off its neighbours;
* ``fit``    -- everything else; the only pixels proposals, normalization
  statistics, context and nuisance fitting may touch.

The same XY role mask is applied to all three context slices. A per-slice role
mask would let the fit view read the same anatomy one slice over, which is the
leak this mask geometry exists to prevent.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

BLOCK_SIZE = 8
SELECT_FRACTION = 0.2
VERIFY_FRACTION = 0.2
GUARD_BAND = 2
PARTITION_SPEC_VERSION = "spatial_blocks.v1"


class PartitionError(ValueError):
    """Raised when a role partition cannot be built or is inconsistent."""


@dataclass(frozen=True)
class RolePartition:
    """Disjoint boolean role masks on the native grid, plus their identity."""

    fit: torch.Tensor      # [H,W] bool
    select: torch.Tensor   # [H,W] bool
    verify: torch.Tensor   # [H,W] bool
    guard: torch.Tensor    # [H,W] bool
    partition_id: str
    spec: dict[str, Any]

    def counts(self) -> dict[str, int]:
        return {
            "fit": int(self.fit.sum()),
            "select": int(self.select.sum()),
            "verify": int(self.verify.sum()),
            "guard_dropped": int(self.guard.sum()),
        }


def _stable_seed(*parts: str) -> int:
    digest = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def partition_identity(*, study_id: str, height: int, width: int, seed: int) -> str:
    """Content hash of everything that determines the partition.

    ``unit_id`` is intentionally absent: the role map belongs to the study grid,
    not to the individual slice or cine time.
    """
    payload = "|".join(
        [
            PARTITION_SPEC_VERSION,
            study_id,
            f"{height}x{width}",
            f"seed={seed}",
            f"block={BLOCK_SIZE}",
            f"select={SELECT_FRACTION}",
            f"verify={VERIFY_FRACTION}",
            f"guard={GUARD_BAND}",
        ]
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.clone()
    grown = mask.clone()
    for _ in range(radius):
        padded = torch.nn.functional.pad(grown.unsqueeze(0).unsqueeze(0).float(), (1, 1, 1, 1))
        pooled = torch.nn.functional.max_pool2d(padded, kernel_size=3, stride=1)
        grown = pooled[0, 0] > 0.5
    return grown


def build_partition(
    height: int,
    width: int,
    *,
    study_id: str,
    seed: int = 42,
) -> RolePartition:
    """Build the frozen role partition shared by every unit of one study grid."""
    if height < BLOCK_SIZE or width < BLOCK_SIZE:
        raise PartitionError(
            f"unit grid {height}x{width} is smaller than one {BLOCK_SIZE}px block"
        )

    blocks_h = (height + BLOCK_SIZE - 1) // BLOCK_SIZE
    blocks_w = (width + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_blocks = blocks_h * blocks_w

    generator = torch.Generator().manual_seed(
        _stable_seed(PARTITION_SPEC_VERSION, study_id, str(seed)) % (2**63 - 1)
    )
    order = torch.randperm(n_blocks, generator=generator)

    n_select = max(1, int(round(SELECT_FRACTION * n_blocks)))
    n_verify = max(1, int(round(VERIFY_FRACTION * n_blocks)))
    if n_select + n_verify >= n_blocks:
        raise PartitionError(
            f"grid {height}x{width} yields {n_blocks} blocks, too few to withhold "
            f"{n_select} selection and {n_verify} verification blocks and keep a fit set"
        )

    role_blocks = torch.zeros(n_blocks, dtype=torch.long)  # 0=fit 1=select 2=verify
    role_blocks[order[:n_select]] = 1
    role_blocks[order[n_select : n_select + n_verify]] = 2
    block_grid = role_blocks.view(blocks_h, blocks_w)

    per_pixel = block_grid.repeat_interleave(BLOCK_SIZE, dim=0).repeat_interleave(
        BLOCK_SIZE, dim=1
    )[:height, :width]

    select = per_pixel == 1
    verify = per_pixel == 2
    withheld = select | verify
    guard = _dilate(withheld, GUARD_BAND) & ~withheld
    fit = ~(withheld | guard)

    if not fit.any():
        raise PartitionError("guard band consumed the entire fitting support")
    if bool((select & verify).any()) or bool((fit & withheld).any()) or bool((fit & guard).any()):
        raise PartitionError("observation roles are not disjoint")

    spec = {
        "spec_version": PARTITION_SPEC_VERSION,
        "block_size": BLOCK_SIZE,
        "select_fraction": SELECT_FRACTION,
        "verify_fraction": VERIFY_FRACTION,
        "guard_band": GUARD_BAND,
        "blocks": [blocks_h, blocks_w],
        "n_blocks": n_blocks,
        "n_select_blocks": n_select,
        "n_verify_blocks": n_verify,
        "seed": int(seed),
        "grid": [int(height), int(width)],
        "role_mask_shared_across_context_slices": True,
        "role_mask_scope": "study_grid",
        "role_mask_shared_across_units_of_study": True,
        "study_id": study_id,
    }
    return RolePartition(
        fit=fit,
        select=select,
        verify=verify,
        guard=guard,
        partition_id=partition_identity(
            study_id=study_id, height=height, width=width, seed=seed
        ),
        spec=spec,
    )
