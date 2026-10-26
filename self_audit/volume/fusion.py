"""Small, explicit fusion blocks used by the native-volume models.

The volume models deliberately keep fusion in a separate module.  This makes
the G6 skip interventions auditable and prevents a control arm from silently
gaining a depth-mixing path.  All blocks operate on ``B,C,Z,H,W`` tensors.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

import torch
from torch import Tensor, nn


FusionMode = Literal["legacy", "delayed", "matched_additive"]


def _check_pair(skip: Tensor, context: Tensor) -> None:
    if skip.ndim != 5 or context.ndim != 5:
        raise ValueError("skip and context must be BxCxZxHxW tensors")
    if skip.shape[0] != context.shape[0] or skip.shape[2:] != context.shape[2:]:
        raise ValueError("skip/context batch and spatial grids must match")


class SkipFusion(nn.Module):
    """Fuse a native detail skip with decoder context.

    ``legacy`` and ``matched_additive`` are additive controls (two projected
    branches).  ``delayed`` retains both branches through a spatial nonlinear
    block before compression.  ``stop_gradient`` is an explicit frozen-
    intervention used by the G6 diagnostic; it never changes the model's
    parameter ownership or optimizer groups.
    """

    def __init__(
        self,
        skip_channels: int,
        context_channels: int,
        out_channels: int,
        *,
        mode: FusionMode = "legacy",
        stop_gradient: bool = False,
        frozen_intervention: bool = False,
        residual_init: str = "default",
        zero_init: bool | None = None,
    ) -> None:
        super().__init__()
        if zero_init is not None:
            residual_init = "zero" if bool(zero_init) else "default"
        if mode not in {"legacy", "delayed", "matched_additive"}:
            raise ValueError(f"unsupported skip fusion mode: {mode}")
        if residual_init not in {"default", "zero", "zero_init", "old_function"}:
            raise ValueError(f"unsupported residual initialization: {residual_init}")
        for name, value in {
            "skip_channels": skip_channels,
            "context_channels": context_channels,
            "out_channels": out_channels,
        }.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        self.mode = mode
        self.stop_gradient = bool(stop_gradient)
        self.frozen_intervention = bool(frozen_intervention)
        self._inference_intervention: str | None = None
        self._persistent_intervention: str | None = "skip" if frozen_intervention else None
        self.residual_init = residual_init
        self.skip_proj = nn.Conv3d(skip_channels, out_channels, kernel_size=1)
        self.context_proj = nn.Conv3d(context_channels, out_channels, kernel_size=1)
        if mode == "delayed":
            # A (1,3,3) block mixes spatial neighbours but does not introduce
            # a hidden cross-Z dependency before the explicit hybrid mixer.
            hidden = max(out_channels, min(2 * out_channels, 64))
            self.nonlinear = nn.Sequential(
                nn.Conv3d(2 * out_channels, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.GELU(),
                nn.Conv3d(hidden, out_channels, kernel_size=1),
            )
        elif mode == "matched_additive":
            # Two independent nonlinear branches retain additive semantics
            # while matching the delayed block's capacity.  Hidden=1.9C is a
            # near-equal analytical match once the 3x3 first convolution and
            # 1x1 compression biases are counted; exact counts are exposed by
            # ``capacity_report`` rather than called identical.
            hidden = max(out_channels, int(round(1.9 * out_channels)))
            self.nonlinear = nn.ModuleList(
                nn.Sequential(
                    nn.Conv3d(out_channels, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                    nn.GELU(),
                    nn.Conv3d(hidden, out_channels, kernel_size=1),
                )
                for _ in range(2)
            )
        else:
            self.nonlinear = nn.Identity()

        # The skip path is itself a residual around the untouched native
        # decoder.  To retain the exact old function for a zero-residual
        # control, only this *final* projection is zeroed.  The upstream
        # projections/nonlinear block remain normally initialized, so their
        # outputs provide a nonzero learning signal after the first update.
        self.residual_projection = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        if residual_init in {"zero", "zero_init", "old_function"}:
            nn.init.zeros_(self.residual_projection.weight)
            nn.init.zeros_(self.residual_projection.bias)
        else:
            with torch.no_grad():
                self.residual_projection.weight.zero_()
                diagonal = min(out_channels, self.residual_projection.weight.shape[0])
                for index in range(diagonal):
                    self.residual_projection.weight[index, index, 0, 0, 0] = 1.0
                self.residual_projection.bias.zero_()
        if self.frozen_intervention:
            for parameter in self.skip_proj.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _validate_intervention(branch: str) -> str:
        if branch not in {"none", "skip", "context", "both"}:
            raise ValueError("intervention branch must be none, skip, context, or both")
        return branch

    @contextmanager
    def branch_intervention(self, branch: str = "skip"):
        """Temporarily zero a residual input branch during frozen inference.

        This is an operational diagnostic, not a retraining intervention. It
        changes no parameters, ``requires_grad`` flags, optimizer state, or
        buffers; the previous mode is restored even if inference raises.
        ``skip``/``context`` refer to the *added residual* inputs. The native
        decoder/detail path outside this block remains live by design.
        """

        branch = self._validate_intervention(branch)
        previous = self._inference_intervention
        self._inference_intervention = branch
        try:
            yield self
        finally:
            self._inference_intervention = previous

    # Descriptive alias for callers that want to make the frozen-evaluation
    # boundary explicit in reports and tests.
    inference_intervention = branch_intervention

    def set_frozen_intervention(self, enabled: bool = True, *, branch: str = "skip") -> None:
        """Freeze skip parameters and enable a persistent branch diagnostic.

        This is deliberately separate from ``stop_gradient``: the latter is a
        retraining ablation that blocks feature gradients while still learning
        the projection parameters; this checkpoint intervention freezes the
        skip projection parameters and zeros the selected residual input at
        inference. Use :meth:`branch_intervention` for a reversible diagnostic
        without changing trainability.
        """

        branch = self._validate_intervention(branch)
        self.frozen_intervention = bool(enabled)
        self._persistent_intervention = branch if enabled else None
        for parameter in self.skip_proj.parameters():
            parameter.requires_grad_(not self.frozen_intervention)

    def clear_frozen_intervention(self) -> None:
        """Restore trainability and remove the persistent checkpoint hook."""

        self.set_frozen_intervention(False, branch="none")

    @staticmethod
    def _conv_cost(module: nn.Module) -> tuple[int, int]:
        params = 0
        macs = 0
        for child in module.modules():
            if isinstance(child, nn.Conv3d):
                params += child.weight.numel() + (child.bias.numel() if child.bias is not None else 0)
                macs += (
                    child.weight.shape[0]
                    * (child.weight.shape[1])
                    * int(torch.tensor(child.kernel_size).prod())
                )
        return int(params), int(macs)

    def capacity_report(self) -> dict[str, int | str]:
        """Analytical per-output-voxel cost of this block's convolutions."""

        params, macs = self._conv_cost(self)
        return {"mode": self.mode, "parameters": params, "macs_per_voxel": macs}

    def forward(self, skip: Tensor, context: Tensor) -> Tensor:
        _check_pair(skip, context)
        if self.stop_gradient or self.frozen_intervention:
            skip = skip.detach()
        intervention = self._inference_intervention
        if intervention is None and self.frozen_intervention:
            intervention = getattr(self, "_persistent_intervention", "skip")
        if intervention in {"skip", "both"}:
            skip = torch.zeros_like(skip)
        if intervention in {"context", "both"}:
            context = torch.zeros_like(context)
        skip_p = self.skip_proj(skip)
        context_p = self.context_proj(context)
        if self.mode == "delayed":
            fused = self.nonlinear(torch.cat((skip_p, context_p), dim=1))
        elif self.mode == "matched_additive":
            fused = self.nonlinear[0](skip_p) + self.nonlinear[1](context_p)
        else:
            fused = skip_p + context_p
        return self.residual_projection(fused)


class CrossZResidual(nn.Module):
    """Residual depth mixer with an explicit no-mixing control.

    The ``off`` path contains no depth convolution and is therefore exactly
    slice independent.  ``mix`` uses a learned 3x1x1 convolution.  ``zero_init``
    is useful for a matched old-function control; enabled ``mix`` defaults to
    a nonzero Kaiming initialization so neighbour influence is observable.
    """

    def __init__(
        self,
        channels: int,
        *,
        mode: str = "mix",
        zero_init: bool = False,
        hidden_channels: int | None = None,
        depth_kernel: int = 3,
    ) -> None:
        super().__init__()
        if mode not in {"off", "mix", "planar_control"}:
            raise ValueError(f"unsupported depth mode: {mode}")
        if int(channels) <= 0:
            raise ValueError("channels must be positive")
        if int(depth_kernel) <= 0 or int(depth_kernel) % 2 == 0:
            raise ValueError("depth_kernel must be a positive odd integer")
        if mode == "planar_control" and int(depth_kernel) != 1:
            raise ValueError("planar_control cannot mix Z; depth_kernel must be 1")
        self.mode = mode
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels) if hidden_channels is not None else int(channels)
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        self.depth_kernel = int(depth_kernel)
        if mode == "off":
            # The off control has no dormant parameters that could be mistaken
            # for a trainable depth path.
            self.mixer = nn.Identity()
        elif mode == "planar_control":
            hidden = self.hidden_channels
            self.mixer = nn.Sequential(
                nn.Conv3d(channels, hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(hidden, channels, kernel_size=1),
            )
        else:
            if self.hidden_channels == int(channels):
                # Keep the simple Conv3d surface for the common case: tests
                # and diagnostics can set its kernel directly.
                self.mixer = nn.Conv3d(
                    channels,
                    channels,
                    kernel_size=(self.depth_kernel, 1, 1),
                    padding=(self.depth_kernel // 2, 0, 0),
                    bias=True,
                )
            else:
                self.mixer = nn.Sequential(
                    nn.Conv3d(channels, self.hidden_channels, kernel_size=1),
                    nn.GELU(),
                    nn.Conv3d(
                        self.hidden_channels,
                        self.hidden_channels,
                        kernel_size=(self.depth_kernel, 1, 1),
                        padding=(self.depth_kernel // 2, 0, 0),
                    ),
                    nn.GELU(),
                    nn.Conv3d(self.hidden_channels, channels, kernel_size=1),
                )
        if zero_init and mode != "off":
            # Only the terminal projection is zeroed.  Earlier layers are
            # deliberately left at their ordinary random initialization so a
            # first update can learn the residual and later updates can train
            # the upstream planar/depth features.
            terminal = self.mixer if isinstance(self.mixer, nn.Conv3d) else next(
                module for module in reversed(list(self.mixer.modules())) if isinstance(module, nn.Conv3d)
            )
            nn.init.zeros_(terminal.weight)
            if terminal.bias is not None:
                nn.init.zeros_(terminal.bias)
        elif mode == "mix":
            for module in self.mixer.modules():
                if isinstance(module, nn.Conv3d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="linear")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        else:
            for module in self.mixer.modules():
                if isinstance(module, nn.Conv3d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="linear")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError("CrossZResidual expects BxCxZxHxW")
        if self.mode in {"mix", "planar_control"}:
            return x + self.mixer(x)
        return x


# Descriptive aliases keep downstream ablation/config code readable while the
# single implementation retains one audited set of semantics.
DelayedNonlinearFusion = SkipFusion
AdditiveFusion = SkipFusion

__all__ = ["AdditiveFusion", "CrossZResidual", "DelayedNonlinearFusion", "SkipFusion"]
