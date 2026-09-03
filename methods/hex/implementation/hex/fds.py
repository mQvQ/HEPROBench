import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn


TensorLike = Union[torch.Tensor, "numpy.ndarray"]  # type: ignore[name-defined]


def _as_tensor(x: TensorLike, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    # numpy array path (avoid importing numpy at module import time)
    return torch.as_tensor(x, device=device, dtype=dtype)


def gaussian_kernel1d(kernel_size: int, sigma: float, *, device: torch.device) -> torch.Tensor:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


@dataclass(frozen=True)
class FDSConfig:
    num_bins: int = 100
    label_min: float = 0.0
    label_max: float = 1.0
    feature_dim: int = 128
    momentum: float = 0.9
    eps: float = 1e-6
    kernel_size: int = 9
    kernel_sigma: float = 2.0
    # Paper: "start update"=0, "start smooth"=10 (we map "smooth" to "apply calibration")
    start_update: int = 0
    start_apply: int = 10
    covariance_type: Literal["diag"] = "diag"
    scalarize: Literal["mean", "sum", "l2"] = "mean"
    use_pred_bins_in_eval: bool = True


class FeatureDistributionSmoothing(nn.Module):
    """
    Minimal Feature Distribution Smoothing (FDS) for imbalanced regression.

    This implementation follows the common FDS recipe:
      1) bin samples by (scalarized) target value
      2) maintain per-bin feature mean/variance with EMA updates
      3) smooth the per-bin stats with a Gaussian kernel along the bin axis
      4) calibrate features via (whiten with raw stats) -> (re-color with smoothed stats)

    Notes:
    - Uses diagonal covariance (per-dim variance) for stability and simplicity.
    - For multi-output targets, a scalar target is derived via `scalarize`.
    """

    def __init__(self, config: FDSConfig):
        super().__init__()
        if config.num_bins <= 1:
            raise ValueError("num_bins must be > 1")
        if config.label_max <= config.label_min:
            raise ValueError("label_max must be > label_min")
        if not (0.0 <= config.momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if config.covariance_type != "diag":
            raise ValueError("only covariance_type='diag' is implemented")
        if config.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.config = config
        self.start_update = int(config.start_update)
        self.start_apply = int(config.start_apply)

        # Fixed bin edges for reproducibility / benchmark fairness.
        edges = torch.linspace(
            float(config.label_min),
            float(config.label_max),
            steps=config.num_bins + 1,
            dtype=torch.float32,
        )
        self.register_buffer("bin_edges", edges, persistent=True)

        self.register_buffer(
            "running_mean",
            torch.zeros(config.num_bins, config.feature_dim, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "running_var",
            torch.ones(config.num_bins, config.feature_dim, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "running_count",
            torch.zeros(config.num_bins, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "last_epoch_count",
            torch.zeros(config.num_bins, dtype=torch.float32),
            persistent=True,
        )

        self.register_buffer(
            "smoothed_mean",
            torch.zeros(config.num_bins, config.feature_dim, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "smoothed_var",
            torch.ones(config.num_bins, config.feature_dim, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer("stats_ready", torch.tensor(False), persistent=True)

        kernel = gaussian_kernel1d(config.kernel_size, config.kernel_sigma, device=torch.device("cpu"))
        self.register_buffer("kernel", kernel, persistent=True)

    def _scalarize_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim == 1:
            return labels
        if labels.ndim == 2:
            if self.config.scalarize == "mean":
                return labels.nanmean(dim=1)
            if self.config.scalarize == "sum":
                return labels.nansum(dim=1)
            if self.config.scalarize == "l2":
                return torch.sqrt(torch.nan_to_num(labels, nan=0.0).pow(2).sum(dim=1))
        raise ValueError(f"unsupported labels shape: {tuple(labels.shape)}")

    def _bin_indices(self, scalar_targets: torch.Tensor) -> torch.Tensor:
        # Clamp into [min, max] to avoid out-of-range bucketize behavior.
        scalar_targets = torch.clamp(scalar_targets, float(self.config.label_min), float(self.config.label_max))
        # bucketize returns in [1..num_bins], shift to [0..num_bins-1]
        idx = torch.bucketize(scalar_targets, self.bin_edges[1:-1], right=False)
        return idx.to(dtype=torch.long)

    @torch.no_grad()
    def update_running_stats(self, features: TensorLike, labels: TensorLike, epoch: int) -> None:
        if epoch < self.start_update:
            return

        device = self.running_mean.device
        feats = _as_tensor(features, device=device, dtype=torch.float32)
        labs = _as_tensor(labels, device=device, dtype=torch.float32)

        if feats.ndim != 2 or feats.shape[1] != self.config.feature_dim:
            raise ValueError(f"features must be (N, {self.config.feature_dim}), got {tuple(feats.shape)}")

        scalar = self._scalarize_labels(labs)
        bin_idx = self._bin_indices(scalar)

        for b in range(self.config.num_bins):
            mask = bin_idx == b
            count = int(mask.sum().item())
            if count <= 1:
                continue

            x = feats[mask]
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False).clamp_min(self.config.eps)

            if self.running_count[b].item() == 0:
                self.running_mean[b] = mean
                self.running_var[b] = var
                self.running_count[b] = float(count)
                continue

            m = float(self.config.momentum)
            self.running_mean[b] = m * self.running_mean[b] + (1.0 - m) * mean
            self.running_var[b] = m * self.running_var[b] + (1.0 - m) * var
            self.running_count[b] = self.running_count[b] + float(count)

        self.last_epoch_count.zero_()
        self.last_epoch_count.index_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=torch.float32))
        self.stats_ready = torch.tensor(True, device=device)

    def init_moments(self, *, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = device or self.running_mean.device
        count = torch.zeros(self.config.num_bins, device=device, dtype=torch.float32)
        sum1 = torch.zeros(self.config.num_bins, self.config.feature_dim, device=device, dtype=torch.float32)
        sum2 = torch.zeros(self.config.num_bins, self.config.feature_dim, device=device, dtype=torch.float32)
        return count, sum1, sum2

    @torch.no_grad()
    def accumulate_moments(
        self,
        moments: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.config.feature_dim:
            raise ValueError(f"features must be (N, {self.config.feature_dim}), got {tuple(features.shape)}")

        count, sum1, sum2 = moments
        scalar = self._scalarize_labels(labels)
        bin_idx = self._bin_indices(scalar)

        ones = torch.ones_like(bin_idx, dtype=torch.float32, device=bin_idx.device)
        count.scatter_add_(0, bin_idx, ones)
        sum1.index_add_(0, bin_idx, features.to(dtype=torch.float32))
        sum2.index_add_(0, bin_idx, features.to(dtype=torch.float32).pow(2))
        return count, sum1, sum2

    @torch.no_grad()
    def update_running_stats_from_moments(
        self,
        count: torch.Tensor,
        sum1: torch.Tensor,
        sum2: torch.Tensor,
        *,
        epoch: int,
    ) -> None:
        if epoch < self.start_update:
            return
        if count.shape != (self.config.num_bins,):
            raise ValueError(f"count shape must be {(self.config.num_bins,)}, got {tuple(count.shape)}")
        if sum1.shape != (self.config.num_bins, self.config.feature_dim):
            raise ValueError(
                f"sum1 shape must be {(self.config.num_bins, self.config.feature_dim)}, got {tuple(sum1.shape)}"
            )
        if sum2.shape != (self.config.num_bins, self.config.feature_dim):
            raise ValueError(
                f"sum2 shape must be {(self.config.num_bins, self.config.feature_dim)}, got {tuple(sum2.shape)}"
            )

        device = self.running_mean.device
        count = count.to(device=device, dtype=torch.float32)
        sum1 = sum1.to(device=device, dtype=torch.float32)
        sum2 = sum2.to(device=device, dtype=torch.float32)

        self.last_epoch_count.copy_(count)

        valid = count > 1.0
        if not bool(valid.any().item()):
            return

        mean = torch.zeros_like(self.running_mean)
        var = torch.ones_like(self.running_var)

        mean[valid] = sum1[valid] / count[valid].unsqueeze(1)
        ex2 = sum2[valid] / count[valid].unsqueeze(1)
        var[valid] = (ex2 - mean[valid].pow(2)).clamp_min(self.config.eps)

        m = float(self.config.momentum)
        cold = self.running_count == 0
        self.running_mean[cold & valid] = mean[cold & valid]
        self.running_var[cold & valid] = var[cold & valid]

        warm = (~cold) & valid
        self.running_mean[warm] = m * self.running_mean[warm] + (1.0 - m) * mean[warm]
        self.running_var[warm] = m * self.running_var[warm] + (1.0 - m) * var[warm]

        self.running_count = self.running_count + count
        self.stats_ready = torch.tensor(True, device=device)

    @torch.no_grad()
    def update_last_epoch_stats(self, epoch: int) -> None:
        if epoch < self.start_update:
            return
        if not bool(self.stats_ready.item()):
            return

        device = self.running_mean.device
        kernel = self.kernel.to(device=device, dtype=torch.float32)
        radius = kernel.numel() // 2

        sm_mean = torch.zeros_like(self.running_mean)
        sm_var = torch.zeros_like(self.running_var)

        # Weighted smoothing using per-bin counts to avoid tail bins dominating due to noise.
        for b in range(self.config.num_bins):
            w_sum = 0.0
            mean_acc = torch.zeros(self.config.feature_dim, device=device, dtype=torch.float32)
            var_acc = torch.zeros(self.config.feature_dim, device=device, dtype=torch.float32)

            for k, d in enumerate(range(-radius, radius + 1)):
                j = b + d
                if j < 0 or j >= self.config.num_bins:
                    continue
                cnt = float(self.last_epoch_count[j].item())
                if cnt <= 0:
                    continue
                w = float(kernel[k].item()) * cnt
                w_sum += w
                mean_acc += w * self.running_mean[j]
                var_acc += w * self.running_var[j]

            if w_sum > 0:
                sm_mean[b] = mean_acc / w_sum
                sm_var[b] = (var_acc / w_sum).clamp_min(self.config.eps)
            else:
                sm_mean[b] = self.running_mean[b]
                sm_var[b] = self.running_var[b].clamp_min(self.config.eps)

        self.smoothed_mean.copy_(sm_mean)
        self.smoothed_var.copy_(sm_var)

    @torch.no_grad()
    def calibrate(
        self,
        features: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        preds: Optional[torch.Tensor] = None,
        use_pred_bins: bool,
    ) -> torch.Tensor:
        """
        Calibrate features using current running/smoothed stats.

        At inference, labels are unavailable, so bins should be derived from `preds`.
        """
        if not bool(self.stats_ready.item()):
            return features

        if labels is None and preds is None:
            raise ValueError("either labels or preds must be provided")

        if use_pred_bins:
            scalar = self._scalarize_labels(preds if preds is not None else labels)  # type: ignore[arg-type]
        else:
            scalar = self._scalarize_labels(labels if labels is not None else preds)  # type: ignore[arg-type]

        bin_idx = self._bin_indices(scalar)
        mu = self.running_mean[bin_idx]
        var = self.running_var[bin_idx].clamp_min(self.config.eps)
        mu_s = self.smoothed_mean[bin_idx]
        var_s = self.smoothed_var[bin_idx].clamp_min(self.config.eps)

        x = (features - mu) / torch.sqrt(var)
        x = x * torch.sqrt(var_s) + mu_s
        return x
