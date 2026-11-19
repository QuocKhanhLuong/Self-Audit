"""W2S-Match run-time checks: teacher saturation rule and training watchdogs (no training here)."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor

TAU_V_DEFAULT = 0.99
TAU_V_SATURATED = 0.999
SATURATION_LIMIT = 0.90


class SaturationAccumulator:
    """Accumulates confidence statistics of frozen W on predicted-foreground voxels of U."""

    def __init__(self, threshold: float = TAU_V_DEFAULT, bins: int = 20) -> None:
        self.threshold, self.bins = float(threshold), int(bins)
        self.count = torch.zeros(4, dtype=torch.float64)
        self.above = torch.zeros(4, dtype=torch.float64)
        self.hist = torch.zeros(self.bins, dtype=torch.float64)
        self.volumes = 0

    @torch.no_grad()
    def add(self, probabilities: Tensor, valid: Tensor | None = None) -> None:
        conf, label = probabilities.detach().float().max(dim=1)
        keep = label > 0
        if valid is not None:
            keep = keep & valid.bool()
        for cls in range(1, 4):
            sel = keep & (label == cls)
            self.count[cls] += float(sel.sum())
            self.above[cls] += float((sel & (conf >= self.threshold)).sum())
        self.hist += torch.histc(conf[keep].cpu().double(), bins=self.bins, min=0.0, max=1.0)
        self.volumes += int(probabilities.shape[0])

    def report(self) -> dict[str, Any]:
        total = float(self.count.sum())
        overall = float(self.above.sum()) / total if total else 0.0
        return {
            "volumes": self.volumes, "threshold": self.threshold, "overall_fraction": overall,
            "per_class_fraction": {c: (float(self.above[c]) / float(self.count[c]) if self.count[c] else 0.0) for c in (1, 2, 3)},
            "histogram": self.hist.tolist(), "tau_v": choose_tau_v(overall),
        }


def choose_tau_v(overall_fraction: float) -> float:
    """The only permitted tau_v adjustment: 0.999 when > 90 % of predicted-FG voxels are >= 0.99."""

    return TAU_V_SATURATED if overall_fraction > SATURATION_LIMIT else TAU_V_DEFAULT


class Watchdogs:
    """Console WARN once per condition; never stops training."""

    def __init__(self, fg_prior: float, alpha_onset_update: int, *, emit: Any = None) -> None:
        self.prior, self.onset = float(fg_prior), int(alpha_onset_update)
        self.emit = emit or (lambda message: warnings.warn(message, RuntimeWarning, stacklevel=2))
        self.fired: set[str] = set()
        self._low_since: int | None = None

    def _once(self, key: str, message: str) -> None:
        if key not in self.fired:
            self.fired.add(key)
            self.emit(f"[W2S watchdog] {message}")

    def update(self, update: int, diag: Mapping[str, Any]) -> list[str]:
        before = set(self.fired)
        share = diag.get("override_share_fg_disagree")
        if share is not None:
            if share > 0.30:
                self._once("override_high", f"override share {share:.3f} > 0.30 of FG disagreement voxels at update {update}")
            if update >= self.onset and diag.get("alpha", 0.0) > 0:
                if share < 0.005:
                    self._low_since = update if self._low_since is None else self._low_since
                    if update - self._low_since >= 500:
                        self._once("veto_saturation", f"override share < 0.5% for 500 updates after alpha onset (veto saturation) at update {update}")
                else:
                    self._low_since = None
        cov = diag.get("fg_coverage")
        if cov is not None and self.prior > 0 and abs(cov - self.prior) / self.prior > 0.10:
            self._once("fg_drift", f"FG coverage {cov:.3f} drifted >10% from K8 prior {self.prior:.3f} at update {update}")
        return sorted(self.fired - before)


__all__ = ["SaturationAccumulator", "Watchdogs", "choose_tau_v"]
