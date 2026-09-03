"""GRPO-style objective for regression-based HE->mIF post-training."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class GRPOObjective(nn.Module):
    def __init__(
        self,
        beta: float = 0.5,
        adv_clip: float = 2.0,
        weight_min: float = 0.5,
        weight_max: float = 2.0,
        lambda_rl: float = 0.2,
        lambda_sup: float = 1.0,
        lambda_anchor: float = 0.5,
        eps: float = 1.0e-6,
    ):
        super().__init__()
        self.beta = float(beta)
        self.adv_clip = float(adv_clip)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.lambda_rl = float(lambda_rl)
        self.lambda_sup = float(lambda_sup)
        self.lambda_anchor = float(lambda_anchor)
        self.eps = float(eps)

    def forward(
        self,
        mse_group: torch.Tensor,
        rewards: torch.Tensor,
        sup_loss: torch.Tensor,
        anchor_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if mse_group.shape != rewards.shape:
            raise ValueError(
                f"mse_group and rewards must have same shape, got {mse_group.shape} vs {rewards.shape}"
            )

        reward_mean = rewards.mean(dim=1, keepdim=True)
        reward_std = rewards.std(dim=1, keepdim=True, unbiased=False)

        advantages = (rewards - reward_mean) / (reward_std + self.eps)
        advantages = advantages.clamp(min=-self.adv_clip, max=self.adv_clip)

        weights = torch.exp(self.beta * advantages)
        weights = weights.clamp(min=self.weight_min, max=self.weight_max)

        rl_loss = (weights * mse_group).mean()

        total_loss = (
            self.lambda_rl * rl_loss
            + self.lambda_sup * sup_loss
            + self.lambda_anchor * anchor_loss
        )

        return {
            "total_loss": total_loss,
            "rl_loss": rl_loss,
            "sup_loss": sup_loss,
            "anchor_loss": anchor_loss,
            "reward_mean": rewards.mean(),
            "reward_std": reward_std.mean(),
            "adv_mean": advantages.mean(),
            "adv_std": advantages.std(unbiased=False),
            "weight_mean": weights.mean(),
            "weight_std": weights.std(unbiased=False),
        }
