from __future__ import annotations

import math

import numpy as np


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred.astype(np.float32) - target.astype(np.float32))))


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    diff = pred.astype(np.float32) - target.astype(np.float32)
    return float(np.mean(diff * diff))


def pearson(pred: np.ndarray, target: np.ndarray) -> float:
    x = pred.astype(np.float64).reshape(-1)
    y = target.astype(np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 255.0) -> float:
    value = mse(pred, target)
    if value == 0.0:
        return float("inf")
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(value))


def global_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 255.0) -> float:
    x = pred.astype(np.float64)
    y = target.astype(np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ux = float(x.mean())
    uy = float(y.mean())
    vx = float(((x - ux) ** 2).mean())
    vy = float(((y - uy) ** 2).mean())
    cov = float(((x - ux) * (y - uy)).mean())
    denom = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    if denom == 0.0:
        return float("nan")
    return float(((2 * ux * uy + c1) * (2 * cov + c2)) / denom)

