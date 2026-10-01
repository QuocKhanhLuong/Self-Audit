# SPDX-License-Identifier: GPL-3.0
"""SGSCN PAPER_FAITHFUL_REIMPLEMENTATION (Ahn et al., MICCAI 2021, arXiv 2107.04934).

Implemented from the paper text and equations, not from the official demo:

- Network F (section 3.2): 3 convolutional layers, kernel 3x3, stride 1, padding 1,
  100 filters each, each layer with ReLU activation and batch normalisation.
- S_hat (section 2.2): S = F(x) normalised to zero mean and unit variance; cluster
  labels C = argmax over channels of S_hat.
- L_ce (Eq. 1, section 2.3 "as in a standard CNN"): standard (softmax) cross-entropy
  between S_hat and C, summed over pixels. Eq. 1 as printed (``ln S_hat_{n-1}``) is not
  computable on a zero-mean map; the text's standard cross-entropy is used.
- L_ss (Eq. 2): sum over k = 1..W-1 and l = 1..H-1 of the channel L1 norms of the
  horizontal and vertical differences of S_hat (summed, not averaged).
- L_cc (Eqs. 3-4): per-cluster centre (C^k, C^l) of a spatial density over pixel
  coordinates; sum over clusters and pixels of the squared distance to that centre
  weighted by the normalised density.
- L = L_ce + L_ss + L_cc (section 2.6, unweighted sum).
- SGD with momentum 0.9 (learning rate per dataset, section 3.2), trained per image.

Choices the paper leaves open are explicit required options (no defaults):
``layer_order``, ``output_normalization``, ``context_density``, ``input_encoding`` and
the stopping convention ("until the clustering and the loss become stable").
"""
from __future__ import annotations

import hashlib
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER_ORDERS = {"conv_relu_bn", "conv_bn_relu"}
OUTPUT_NORMALIZATIONS = {"final_batchnorm", "per_channel_standardization", "global_standardization"}
CONTEXT_DENSITIES = {
    "literal_normalized_map": "PAPER_LITERAL (signed map; refused when a cluster's sum is zero)",
    "channel_softmax": "IMPLEMENTATION_CONVENTION (per-pixel softmax over clusters)",
    "epsilon_shift": "OFFICIAL_CODE_FALLBACK (S_hat + epsilon, as in demo_final.py)",
}
INPUT_ENCODINGS = {"rgb_unit_interval", "bgr_unit_interval", "grayscale_unit_interval"}


class ProtocolSettingError(ValueError):
    pass


def _require(settings, name, allowed=None):
    value = settings.get(name)
    if value is None:
        raise ProtocolSettingError(f"paper-faithful setting required: {name}")
    if allowed is not None and value not in allowed:
        raise ProtocolSettingError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return value


class PaperFaithfulSGSCN(nn.Module):
    def __init__(self, in_channels, settings):
        super().__init__()
        self.layer_order = _require(settings, "layer_order", LAYER_ORDERS)
        self.output_normalization = _require(settings, "output_normalization", OUTPUT_NORMALIZATIONS)
        if self.output_normalization == "final_batchnorm" and self.layer_order != "conv_relu_bn":
            raise ProtocolSettingError("final_batchnorm requires batch normalisation as the last operation (conv_relu_bn)")
        if self.output_normalization != "final_batchnorm":
            self.standardization_epsilon = float(_require(settings, "standardization_epsilon"))
        filters = int(settings["filters"])
        layers = int(settings["conv_layers"])
        kernel, stride, padding = int(settings["kernel_size"]), int(settings["stride"]), int(settings["padding"])
        channels = [in_channels] + [filters] * layers
        self.convs = nn.ModuleList(nn.Conv2d(channels[i], filters, kernel, stride=stride, padding=padding)
                                   for i in range(layers))
        self.norms = nn.ModuleList(nn.BatchNorm2d(filters) for _ in range(layers))

    def forward(self, x):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = norm(F.relu(x)) if self.layer_order == "conv_relu_bn" else F.relu(norm(x))
        if self.output_normalization == "final_batchnorm":
            return x
        dims = (2, 3) if self.output_normalization == "per_channel_standardization" else (1, 2, 3)
        mean = x.mean(dim=dims, keepdim=True)
        variance = x.var(dim=dims, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(variance + self.standardization_epsilon)


def cross_entropy_sum(s_hat, labels):
    """Eq. 1 as standard cross-entropy, summed over pixels. s_hat: [C,H,W]; labels: [H*W]."""
    channels = s_hat.shape[0]
    return F.cross_entropy(s_hat.permute(1, 2, 0).reshape(-1, channels), labels, reduction="sum")


def sparse_spatial_sum(s_hat):
    """Eq. 2: k = 1..W-1 (width), l = 1..H-1 (height); channel L1 norms, summed."""
    _, height, width = s_hat.shape
    base = s_hat[:, : height - 1, : width - 1]
    along_k = s_hat[:, : height - 1, 1:width] - base
    along_l = s_hat[:, 1:height, : width - 1] - base
    return along_k.abs().sum() + along_l.abs().sum()


def _density(s_hat, settings):
    mode = _require(settings, "context_density", CONTEXT_DENSITIES)
    if mode == "literal_normalized_map":
        return s_hat
    if mode == "channel_softmax":
        return torch.softmax(s_hat, dim=0)
    return s_hat + float(_require(settings, "context_epsilon"))


def context_consistency_sum(s_hat, settings):
    """Eqs. 3-4 over pixel coordinates (k = column, l = row); squared Euclidean distance."""
    density = _density(s_hat, settings)
    _, height, width = s_hat.shape
    k = torch.arange(width, dtype=s_hat.dtype, device=s_hat.device).view(1, 1, width)
    l = torch.arange(height, dtype=s_hat.dtype, device=s_hat.device).view(1, height, 1)
    mass = density.sum(dim=(1, 2), keepdim=True)
    if torch.any(mass == 0):
        raise FloatingPointError("context density has zero mass for a cluster; Eq. 3 is undefined (no repair applied)")
    weights = density / mass
    centre_k = (weights * k).sum(dim=(1, 2), keepdim=True)
    centre_l = (weights * l).sum(dim=(1, 2), keepdim=True)
    return (((k - centre_k) ** 2 + (l - centre_l) ** 2) * weights).sum()


def paper_loss(s_hat, settings):
    labels = s_hat.detach().argmax(dim=0).reshape(-1)
    terms = {"ce": cross_entropy_sum(s_hat, labels), "ss": sparse_spatial_sum(s_hat),
             "cc": context_consistency_sum(s_hat, settings)}
    return terms["ce"] + terms["ss"] + terms["cc"], labels, terms


def encode_input(image_bgr, settings):
    encoding = _require(settings, "input_encoding", INPUT_ENCODINGS)
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("decoded input must be OpenCV uint8 BGR")
    if encoding == "rgb_unit_interval":
        array = image_bgr[:, :, ::-1]
    elif encoding == "bgr_unit_interval":
        array = image_bgr
    else:
        import cv2
        array = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)[:, :, None]
    return np.ascontiguousarray(array.transpose(2, 0, 1)).astype("float32") / 255.0


def sample_seed(base_seed, sample_id):
    return int.from_bytes(hashlib.sha256(f"{base_seed}\0{sample_id}".encode()).digest()[:4], "big")


def predict_paper_faithful(image_bgr, settings, *, seed, device="cpu"):
    """Per-image training with the declared stability convention; returns (labels, receipt)."""
    max_iterations = int(_require(settings, "max_iterations"))
    patience = int(_require(settings, "stability_patience"))
    tolerance = float(_require(settings, "relative_loss_tolerance"))
    final_mode = _require(settings, "final_forward_mode", {"train", "eval"})
    if max_iterations < 1 or patience < 1 or tolerance < 0:
        raise ProtocolSettingError("stopping convention values must be positive")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data = torch.from_numpy(encode_input(image_bgr, settings))[None].to(device)
    model = PaperFaithfulSGSCN(data.shape[1], settings).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=float(settings["learning_rate"]),
                                momentum=float(settings["momentum"]), weight_decay=float(settings["weight_decay"]))
    trajectory, stable, previous, stop_reason = [], 0, None, "max_iterations"
    started = time.perf_counter()
    for index in range(max_iterations):
        optimizer.zero_grad()
        loss, labels, terms = paper_loss(model(data)[0], settings)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite paper-faithful loss at iteration {index}; no repair applied")
        loss.backward()
        optimizer.step()
        count, value = int(torch.unique(labels).numel()), float(loss.detach())
        if previous is not None and count == previous[0] and abs(value - previous[1]) <= tolerance * abs(previous[1]):
            stable += 1
        else:
            stable = 0
        trajectory.append({"iteration": index, "active_labels": count, "loss": value,
                           **{name: float(term.detach()) for name, term in terms.items()}})
        previous = (count, value)
        if stable >= patience:
            stop_reason = "stable"
            break
    model.train(final_mode == "train")
    with torch.no_grad():
        final = model(data)[0]
    if not torch.isfinite(final).all():
        raise FloatingPointError("nonfinite final output; no repair applied")
    partition = final.argmax(dim=0).cpu().numpy().astype(np.int32)
    return partition, {"seed": seed, "iterations": len(trajectory), "stop_reason": stop_reason,
                       "final_active_labels": int(len(np.unique(partition))), "trajectory": trajectory,
                       "final_forward_mode": final_mode, "seconds": time.perf_counter() - started,
                       "torch_version": torch.__version__, "device": str(device)}


def settings_from_config(config):
    """Flatten scientific values and declared field values into runtime settings."""
    settings = dict(config["scientific"])
    for group in ("paper_unspecified", "implementation_conventions", "conditional_values"):
        settings.update({name: entry.get("value") for name, entry in config.get(group, {}).items()})
    return settings


def validate_settings(settings):
    """Fail before any data access when a required or conditionally required value is missing."""
    _require(settings, "layer_order", LAYER_ORDERS)
    normalization = _require(settings, "output_normalization", OUTPUT_NORMALIZATIONS)
    if normalization == "final_batchnorm" and settings["layer_order"] != "conv_relu_bn":
        raise ProtocolSettingError("final_batchnorm requires batch normalisation as the last operation (conv_relu_bn)")
    if normalization != "final_batchnorm":
        _require(settings, "standardization_epsilon")
    if _require(settings, "context_density", CONTEXT_DENSITIES) == "epsilon_shift":
        _require(settings, "context_epsilon")
    _require(settings, "input_encoding", INPUT_ENCODINGS)
    for name in ("max_iterations", "stability_patience", "relative_loss_tolerance"):
        _require(settings, name)
    _require(settings, "final_forward_mode", {"train", "eval"})
    if settings.get("stopping_rule") != "stable_label_count_and_relative_loss":
        raise ProtocolSettingError("unsupported stopping convention")
    return settings
