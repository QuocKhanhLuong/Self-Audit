"""Explicit preprocessing primitives. No row recipe is inferred from defaults."""
import numpy as np


def imagenet_tensor(rgb):
    import torch
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected RGB uint8")
    value = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255
    mean = value.new_tensor([.485, .456, .406])[:, None, None]
    std = value.new_tensor([.229, .224, .225])[:, None, None]
    return (value - mean) / std


def ultrasound_preprocess(rgb, *, histogram_equalization, gaussian_kernel, gaussian_sigma):
    import cv2
    result = rgb.copy()
    if histogram_equalization:
        if not np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) or not np.array_equal(rgb[:, :, 0], rgb[:, :, 2]):
            raise ValueError("US histogram equalization requires explicit grayscale input")
        result = np.repeat(cv2.equalizeHist(rgb[:, :, 0])[:, :, None], 3, axis=2)
    if gaussian_kernel is not None:
        if gaussian_sigma is None or gaussian_kernel <= 0 or gaussian_kernel % 2 == 0:
            raise ValueError("explicit Gaussian kernel/sigma required")
        result = cv2.GaussianBlur(result, (gaussian_kernel, gaussian_kernel), gaussian_sigma)
    return result
