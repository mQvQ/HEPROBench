from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hex.fds import FDSConfig, FeatureDistributionSmoothing
from timm import create_model
from musk import utils, modeling as _musk_modeling  # noqa: F401


class CustomModelFDS(nn.Module):
    """
    Wraps the original HEX `CustomModel` and adds a `.FDS` attribute plus
    feature calibration logic, without modifying the upstream file.

    Design choices:
    - Return `raw_features` (pre-calibration) as the 2nd output so existing
      training scripts can collect stable statistics.
    - Compute predictions from calibrated features once FDS is ready.
    """

    def __init__(
        self,
        visual_output_dim: int,
        num_outputs: int,
        *,
        fds_config: Optional[FDSConfig] = None,
        musk_img_size: int = 384,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        model_config = f"musk_large_patch16_{int(musk_img_size)}"
        model_musk = create_model(model_config, vocab_size=64010)
        if pretrained:
            utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model_musk, "model|module", "")
        self.visual = model_musk

        self.regression_head = nn.Sequential(
            nn.Linear(visual_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.regression_head1 = nn.Sequential(nn.Linear(128, num_outputs))
        self.FDS = FeatureDistributionSmoothing(fds_config or FDSConfig(feature_dim=128))
        self.training_status: bool = True

    def forward(self, x: torch.Tensor, labels=None, epoch: int = 0):
        x = self.visual(image=x, with_head=False, out_norm=False)[0]
        raw_features = self.regression_head(x)
        preds_raw = self.regression_head1(raw_features)

        if epoch < self.FDS.start_apply or not bool(self.FDS.stats_ready.item()):
            return preds_raw, raw_features

        # For calibration binning:
        # - training: use GT labels (available)
        # - eval/inference: default to predicted bins (no GT in real deployment)
        use_pred_bins = (not self.training_status) and bool(self.FDS.config.use_pred_bins_in_eval)

        calibrated = self.FDS.calibrate(
            raw_features,
            labels=labels if isinstance(labels, torch.Tensor) else None,
            preds=preds_raw.detach(),
            use_pred_bins=use_pred_bins,
        )
        preds = self.regression_head1(calibrated)
        return preds, raw_features
